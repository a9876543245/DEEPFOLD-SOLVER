/**
 * @file gpu_backend.cu
 * @brief CUDA implementation of ISolverBackend.
 *
 * Phase 4.1 status:
 *   - prepare() uploads tree, matchup matrix, reach probabilities to device,
 *     allocates per-iteration solver state (regrets, strategy_sum,
 *     current_strategy).
 *   - iterate() still throws — Phase 4.5 implements the backward pass.
 *   - finalize() still throws — Phase 4.9 implements strategy download.
 *
 * The Impl struct owns all device memory and guarantees cleanup via RAII.
 */

#include "gpu_backend.h"
#include "types.h"
#include "isomorphism.h"

#include "util.cuh"
#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Forward declarations for kernel launch helpers defined in other .cu files
// (cfr_kernel.cu and eval_kernel.cu). These are in namespace deepsolver::gpu.
// ============================================================================
namespace deepsolver {
namespace gpu {

void launch_compute_strategy(
    const float* d_regrets, float* d_current_strategy,
    const uint8_t* d_num_children, const uint8_t* d_node_types,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions);

void launch_propagate_reach(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_current_strategy, uint8_t max_actions,
    float* d_reach_oop, float* d_reach_ip, uint16_t nc);

void launch_aggregate_node_values(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_current_strategy, uint8_t max_actions,
    const float* d_action_values, float* d_node_values,
    uint16_t nc, int traverser);

void launch_lift_child_values(
    const uint8_t* d_node_types, const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_node_values, float* d_action_values,
    uint8_t max_actions, uint16_t nc);

void launch_update_regrets(
    float* d_regrets,
    const float* d_action_values, const float* d_node_values,
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions,
    int traverser, int iteration, float alpha, float beta);

void launch_update_strategy_sum(
    float* d_strategy_sum, const float* d_current_strategy,
    const float* d_reach_own,
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions,
    int traverser, int iteration, float gamma);

void launch_showdown_terminal(
    const float* d_matchup_ev, const float* d_matchup_valid,
    const float* d_canonical_weights, const float* d_reach_opp,
    uint16_t nc, float half_pot, int perspective,
    float* d_out);

void launch_fold_terminal(
    const float* d_matchup_valid, const float* d_canonical_weights,
    const float* d_reach_opp, uint16_t nc, float half_pot,
    float sign_for_perspective, int perspective, float* d_out);

} // namespace gpu
} // namespace deepsolver

namespace deepsolver {

// ============================================================================
// Device data structures (internal to this TU)
// ============================================================================

namespace {

/// SoA tree on device. Mirrors FlatGameTree but with device pointers.
struct DeviceTree {
    uint8_t*  node_types          = nullptr;
    float*    pots                = nullptr;
    uint32_t* children_offset     = nullptr;
    uint8_t*  num_children        = nullptr;
    uint8_t*  terminal_types      = nullptr;
    uint8_t*  active_player       = nullptr;
    uint32_t* children            = nullptr;
    uint8_t*  child_action_types  = nullptr;
    // Phase 2 additions: needed by per-runout terminal eval and chance
    // enumeration. parent_indices + bet_into power the fold-value fix
    // ((pot - bet_into[parent]) / 2). runout_weight is the orbit size of a
    // chance child. matchup_idx selects which per-runout matchup table to
    // use at a terminal.
    uint32_t* parent_indices      = nullptr;
    float*    bet_into            = nullptr;
    uint8_t*  runout_weight       = nullptr;
    int32_t*  matchup_idx         = nullptr;
    uint32_t  num_nodes           = 0;
    uint32_t  num_edges           = 0;
};

/// Precomputed showdown matchup matrix + canonical weights, on device.
/// Phase 2 stores ALL per-runout tables in one big concat buffer so kernels
/// can index by `matchup_idx[node]` without per-table pointer chasing.
///   matchup_ev[i, c, cj] = matchup_ev_concat[i * nc * nc + c * nc + cj]
struct DeviceMatchup {
    float*   matchup_ev          = nullptr;  // [num_runouts * nc * nc]
    float*   matchup_valid       = nullptr;  // [num_runouts * nc * nc]
    float*   canonical_weights   = nullptr;  // [nc] (float)
    uint16_t num_canonical       = 0;
    uint32_t num_runouts         = 1;        // 1 for legacy single-board
};

/// Per-player root-level reach probabilities, on device.
struct DeviceReach {
    float* ip_reach   = nullptr;  // [nc]
    float* oop_reach  = nullptr;  // [nc]
};

/// Node locks on device — sparse list of forced strategies at (node, combo) pairs.
struct DeviceNodeLocks {
    uint32_t* node_indices     = nullptr;  // [num_locks]
    uint16_t* combo_indices    = nullptr;  // [num_locks]
    float*    strategies_flat  = nullptr;  // [sum of per-lock strategy sizes]
    uint32_t* strategy_offsets = nullptr;  // [num_locks + 1]
    uint32_t  num_locks        = 0;
};

/// Per-iteration solver state, on device. Allocated once in prepare.
struct DeviceSolverState {
    float* regrets           = nullptr;  // [N * MAX_ACTIONS * nc]
    float* strategy_sum      = nullptr;  // [N * MAX_ACTIONS * nc]
    float* current_strategy  = nullptr;  // [N * MAX_ACTIONS * nc]
    // Scratch buffers for backward pass (Phase 4.5 uses these)
    float* reach_scratch_oop = nullptr;  // [N * nc]   per-node reach for OOP
    float* reach_scratch_ip  = nullptr;  // [N * nc]   per-node reach for IP
    float* node_values       = nullptr;  // [N * nc]
    float* action_values     = nullptr;  // [N * MAX_ACTIONS * nc]

    size_t state_stride      = 0;        // N * MAX_ACTIONS * nc
};

/// Topologically sorted level schedule: node indices grouped by depth
/// (distance to nearest leaf). Level 0 = terminals; level max_depth = root.
/// The backward pass processes levels 0 → max_depth (leaves-to-root).
///
/// Layout:
///   node_order[level_offsets[L] .. level_offsets[L+1]) = node indices at level L
struct DeviceLevels {
    uint32_t* node_order    = nullptr;  // [num_nodes]
    uint32_t* level_offsets = nullptr;  // [max_depth + 2] (prefix-sum, +1 sentinel)
    uint32_t  max_depth     = 0;
    uint32_t  num_levels    = 0;        // max_depth + 1
};

// ---- Generic upload helpers ----

template <typename T>
static T* upload_vector(const std::vector<T>& host) {
    if (host.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static T* alloc_device_zero(size_t count) {
    if (count == 0) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, count * sizeof(T)));
    CUDA_CHECK(cudaMemset(d, 0, count * sizeof(T)));
    return d;
}

template <typename T>
static void free_device(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// ---- Tree upload ----

static DeviceTree upload_tree(const FlatGameTree& tree) {
    DeviceTree dt;
    dt.num_nodes = tree.total_nodes;
    dt.num_edges = tree.total_edges;

    dt.node_types         = upload_vector(tree.node_types);
    dt.pots               = upload_vector(tree.pots);
    dt.children_offset    = upload_vector(tree.children_offset);
    dt.num_children       = upload_vector(tree.num_children);
    dt.terminal_types     = upload_vector(tree.terminal_types);
    dt.active_player      = upload_vector(tree.active_player);
    dt.children           = upload_vector(tree.children);
    dt.child_action_types = upload_vector(tree.child_action_types);

    // Phase 2 additions. Assert sizes — kernels index by node id, so a
    // short array would silently OOB into adjacent device memory.
    if (tree.parent_indices.size() != tree.total_nodes ||
        tree.bet_into.size()       != tree.total_nodes ||
        tree.runout_weight.size()  != tree.total_nodes ||
        tree.matchup_idx.size()    != tree.total_nodes) {
        throw std::runtime_error(
            "GpuBackend::upload_tree: per-node array size mismatch — "
            "FlatGameTree.flatten() must populate parent_indices/bet_into/"
            "runout_weight/matchup_idx for every node.");
    }
    dt.parent_indices     = upload_vector(tree.parent_indices);
    dt.bet_into           = upload_vector(tree.bet_into);
    dt.runout_weight      = upload_vector(tree.runout_weight);
    dt.matchup_idx        = upload_vector(tree.matchup_idx);
    return dt;
}

static void free_tree(DeviceTree& dt) {
    free_device(dt.node_types);
    free_device(dt.pots);
    free_device(dt.children_offset);
    free_device(dt.num_children);
    free_device(dt.terminal_types);
    free_device(dt.active_player);
    free_device(dt.children);
    free_device(dt.child_action_types);
    free_device(dt.parent_indices);
    free_device(dt.bet_into);
    free_device(dt.runout_weight);
    free_device(dt.matchup_idx);
    dt.num_nodes = dt.num_edges = 0;
}

// ---- Matchup upload ----

/// Upload one or many per-runout matchup tables into a single concat buffer.
/// `ev_per_runout[r]` is a vector of nc*nc floats (FlatGameTree convention).
static DeviceMatchup upload_matchup(
    const std::vector<std::vector<float>>& ev_per_runout,
    const std::vector<std::vector<float>>& valid_per_runout,
    const std::vector<uint16_t>& weights)
{
    DeviceMatchup dm;
    dm.num_canonical = static_cast<uint16_t>(weights.size());
    dm.num_runouts   = static_cast<uint32_t>(std::max<size_t>(1, ev_per_runout.size()));

    size_t per_table = static_cast<size_t>(dm.num_canonical) * dm.num_canonical;
    size_t total     = per_table * dm.num_runouts;

    if (total > 0) {
        // OOM pre-flight: each buffer is `total * 4B`, and we allocate two
        // (ev + valid) on host AND device. For a rainbow flop with 2000
        // canonical runouts and nc=400 this is ~1.3GB per buffer × 4 copies
        // (2 host + 2 device) = ~5GB. Crashing with bad_alloc inside the
        // host vector, or with cudaErrorMemoryAllocation inside cudaMalloc,
        // gives the user a confusing engine-died-with-no-message error and
        // the Tauri layer can't pattern-match it for a clean CPU fallback.
        // Throw a structured message instead so engine.rs's GPU-fallback
        // detector triggers ("out of memory" substring is in its allowlist).
        size_t bytes_per_buffer = total * sizeof(float);
        size_t total_required   = bytes_per_buffer * 2;  // ev + valid

        size_t free_dev = 0, total_dev = 0;
        if (cudaMemGetInfo(&free_dev, &total_dev) == cudaSuccess) {
            // Need device room for both buffers plus headroom for tree,
            // state, scratch (~few hundred MB). Cap at 60% of free memory.
            if (total_required > static_cast<size_t>(free_dev * 0.6)) {
                std::ostringstream oss;
                oss << "GpuBackend: matchup tables would require "
                    << (total_required >> 20) << " MB but only "
                    << (free_dev >> 20) << " MB free on device — out of memory. "
                    << "Re-run with --backend cpu.";
                throw std::runtime_error(oss.str());
            }
        }

        // Host alloc may also fail (e.g. on 8GB laptops with rainbow flop).
        // Wrap in try/catch and re-throw with the same OOM signature.
        std::vector<float> flat_ev, flat_valid;
        try {
            flat_ev.assign(total, 0.0f);
            flat_valid.assign(total, 0.0f);
        } catch (const std::bad_alloc&) {
            std::ostringstream oss;
            oss << "GpuBackend: host out of memory while staging "
                << (bytes_per_buffer >> 20) << " MB matchup tables. "
                << "Re-run with --backend cpu.";
            throw std::runtime_error(oss.str());
        }

        for (uint32_t r = 0; r < dm.num_runouts; ++r) {
            if (r < ev_per_runout.size() && ev_per_runout[r].size() == per_table) {
                std::copy(ev_per_runout[r].begin(), ev_per_runout[r].end(),
                          flat_ev.begin() + r * per_table);
            }
            if (r < valid_per_runout.size() && valid_per_runout[r].size() == per_table) {
                std::copy(valid_per_runout[r].begin(), valid_per_runout[r].end(),
                          flat_valid.begin() + r * per_table);
            }
        }
        dm.matchup_ev    = upload_vector(flat_ev);
        dm.matchup_valid = upload_vector(flat_valid);
    }

    std::vector<float> weights_f(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weights_f[i] = static_cast<float>(weights[i]);
    }
    dm.canonical_weights = upload_vector(weights_f);
    return dm;
}

static void free_matchup(DeviceMatchup& dm) {
    free_device(dm.matchup_ev);
    free_device(dm.matchup_valid);
    free_device(dm.canonical_weights);
    dm.num_canonical = 0;
    dm.num_runouts   = 1;
}

// ---- Reach upload ----

static DeviceReach upload_reach(const std::vector<float>& ip_reach,
                                 const std::vector<float>& oop_reach)
{
    DeviceReach dr;
    dr.ip_reach  = upload_vector(ip_reach);
    dr.oop_reach = upload_vector(oop_reach);
    return dr;
}

static void free_reach(DeviceReach& dr) {
    free_device(dr.ip_reach);
    free_device(dr.oop_reach);
}

// ---- Node lock upload ----

static DeviceNodeLocks upload_locks(
    const std::map<std::pair<uint32_t, uint16_t>, std::vector<float>>& resolved_locks)
{
    DeviceNodeLocks dl;
    dl.num_locks = static_cast<uint32_t>(resolved_locks.size());
    if (dl.num_locks == 0) return dl;

    std::vector<uint32_t> node_idx;   node_idx.reserve(dl.num_locks);
    std::vector<uint16_t> combo_idx;  combo_idx.reserve(dl.num_locks);
    std::vector<float>    flat;
    std::vector<uint32_t> offsets(dl.num_locks + 1, 0);

    uint32_t i = 0;
    for (const auto& [key, strat] : resolved_locks) {
        node_idx.push_back(key.first);
        combo_idx.push_back(key.second);
        offsets[i + 1] = offsets[i] + static_cast<uint32_t>(strat.size());
        flat.insert(flat.end(), strat.begin(), strat.end());
        ++i;
    }

    dl.node_indices     = upload_vector(node_idx);
    dl.combo_indices    = upload_vector(combo_idx);
    dl.strategies_flat  = upload_vector(flat);
    dl.strategy_offsets = upload_vector(offsets);
    return dl;
}

static void free_locks(DeviceNodeLocks& dl) {
    free_device(dl.node_indices);
    free_device(dl.combo_indices);
    free_device(dl.strategies_flat);
    free_device(dl.strategy_offsets);
    dl.num_locks = 0;
}

// Kernel: override current_strategy at locked (node, combo) cells.
// Launches num_locks threads — one per lock entry.
__global__ void apply_locks_kernel(
    float* __restrict__ current_strategy,         // [N * A * nc]
    const uint32_t* __restrict__ node_indices,
    const uint16_t* __restrict__ combo_indices,
    const float*    __restrict__ strategies_flat,
    const uint32_t* __restrict__ strategy_offsets,
    uint32_t num_locks,
    uint16_t nc,
    uint8_t  max_actions)
{
    uint32_t lock_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (lock_id >= num_locks) return;

    uint32_t node   = node_indices[lock_id];
    uint16_t combo  = combo_indices[lock_id];
    uint32_t off    = strategy_offsets[lock_id];
    uint32_t count  = strategy_offsets[lock_id + 1] - off;

    size_t base = (static_cast<size_t>(node) * max_actions) * nc + combo;
    size_t stride = nc;
    for (uint32_t a = 0; a < count && a < max_actions; ++a) {
        current_strategy[base + a * stride] = strategies_flat[off + a];
    }
}

// ---- Topological level sort ----

/// Compute each node's distance to nearest leaf (terminals = 0).
/// Requires: BFS-built tree where children have larger indices than parents,
/// OR falls back to iterative post-order if the assumption is violated.
static std::vector<uint32_t> compute_node_depths(const FlatGameTree& tree) {
    const uint32_t N = tree.total_nodes;
    std::vector<uint32_t> depth(N, 0);
    if (N == 0) return depth;

    // First pass: reverse iteration assuming child idx > parent idx (BFS order).
    // Safe because the tree is BFS-flattened from GameTreeBuilder.
    for (int64_t i = static_cast<int64_t>(N) - 1; i >= 0; --i) {
        uint8_t na = tree.num_children[i];
        if (na == 0) {
            depth[i] = 0;
            continue;
        }
        uint32_t max_child = 0;
        for (uint8_t a = 0; a < na; ++a) {
            uint32_t child = tree.children[tree.children_offset[i] + a];
            if (child < N && depth[child] + 1 > max_child) {
                max_child = depth[child] + 1;
            }
        }
        depth[i] = max_child;
    }
    return depth;
}

static DeviceLevels upload_levels(const FlatGameTree& tree) {
    DeviceLevels dl;
    const uint32_t N = tree.total_nodes;
    if (N == 0) return dl;

    std::vector<uint32_t> depth = compute_node_depths(tree);
    uint32_t max_d = 0;
    for (uint32_t d : depth) if (d > max_d) max_d = d;

    const uint32_t num_levels = max_d + 1;
    std::vector<uint32_t> count(num_levels, 0);
    for (uint32_t d : depth) count[d]++;

    // Prefix-sum offsets; offsets[num_levels] = N (sentinel)
    std::vector<uint32_t> offsets(num_levels + 1, 0);
    for (uint32_t L = 0; L < num_levels; ++L) {
        offsets[L + 1] = offsets[L] + count[L];
    }

    // Bucket fill: for each node, place at its level's cursor
    std::vector<uint32_t> node_order(N, 0);
    std::vector<uint32_t> cursor(offsets);
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t L = depth[n];
        node_order[cursor[L]++] = n;
    }

    dl.node_order    = upload_vector(node_order);
    dl.level_offsets = upload_vector(offsets);
    dl.max_depth     = max_d;
    dl.num_levels    = num_levels;
    return dl;
}

static void free_levels(DeviceLevels& dl) {
    free_device(dl.node_order);
    free_device(dl.level_offsets);
    dl.max_depth = 0;
    dl.num_levels = 0;
}

// ---- Solver state allocation ----

static DeviceSolverState alloc_solver_state(uint32_t num_nodes,
                                              uint8_t max_actions,
                                              uint16_t num_canonical,
                                              const FlatGameTree& tree)
{
    DeviceSolverState ds;
    size_t N  = num_nodes;
    size_t nc = num_canonical;
    size_t A  = max_actions;
    size_t strat_stride = N * A * nc;

    ds.state_stride       = strat_stride;
    ds.regrets            = alloc_device_zero<float>(strat_stride);
    ds.strategy_sum       = alloc_device_zero<float>(strat_stride);
    ds.current_strategy   = alloc_device_zero<float>(strat_stride);
    ds.reach_scratch_oop  = alloc_device_zero<float>(N * nc);
    ds.reach_scratch_ip   = alloc_device_zero<float>(N * nc);
    ds.node_values        = alloc_device_zero<float>(N * nc);
    ds.action_values      = alloc_device_zero<float>(strat_stride);

    // Initialize current_strategy to uniform 1/num_children per player node.
    // This mirrors CpuBackend::prepare(). We build on host then upload.
    std::vector<float> host_strat(strat_stride, 0.0f);
    for (uint32_t n = 0; n < num_nodes; ++n) {
        uint8_t na = tree.num_children[n];
        auto nt = static_cast<NodeType>(tree.node_types[n]);
        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            float u = 1.0f / static_cast<float>(na);
            for (uint8_t a = 0; a < na; ++a) {
                for (uint16_t c = 0; c < nc; ++c) {
                    size_t idx = (static_cast<size_t>(n) * A + a) * nc + c;
                    host_strat[idx] = u;
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(ds.current_strategy, host_strat.data(),
                           strat_stride * sizeof(float),
                           cudaMemcpyHostToDevice));

    return ds;
}

static void free_solver_state(DeviceSolverState& ds) {
    free_device(ds.regrets);
    free_device(ds.strategy_sum);
    free_device(ds.current_strategy);
    free_device(ds.reach_scratch_oop);
    free_device(ds.reach_scratch_ip);
    free_device(ds.node_values);
    free_device(ds.action_values);
    ds.state_stride = 0;
}

/// Build a GPU device-name string for the UI indicator
static std::string detect_device_name() {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return "CUDA";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "CUDA";
    std::ostringstream oss;
    oss << "CUDA (" << prop.name
        << ", " << (prop.totalGlobalMem / (1024ull * 1024ull)) << " MB, "
        << "CC " << prop.major << "." << prop.minor << ")";
    return oss.str();
}

} // anonymous namespace

// ============================================================================
// Impl struct: owns all device memory
// ============================================================================

struct GpuBackend::Impl {
    DeviceTree        tree{};
    DeviceMatchup     matchup{};
    DeviceReach       reach{};
    DeviceSolverState state{};
    DeviceLevels      levels{};
    DeviceNodeLocks   locks{};

    // Host-side copies of level schedule (for iterating on host to launch per-terminal kernels)
    std::vector<uint32_t> host_node_order;
    std::vector<uint32_t> host_level_offsets;

    // Cached host-side info for finalize() and bookkeeping
    const SolverConfig*       config = nullptr;
    const IsomorphismMapping* iso    = nullptr;
    const FlatGameTree*       host_tree = nullptr;
    bool prepared = false;

    ~Impl() {
        free_solver_state(state);
        free_locks(locks);
        free_levels(levels);
        free_reach(reach);
        free_matchup(matchup);
        free_tree(tree);
    }
};

// ============================================================================
// GpuBackend interface
// ============================================================================

GpuBackend::GpuBackend()
    : impl_(std::make_unique<Impl>()),
      name_(detect_device_name())
{}

GpuBackend::~GpuBackend() = default;

void GpuBackend::prepare(const SolverContext& ctx) {
    if (!ctx.tree || !ctx.iso || !ctx.config ||
        !ctx.matchup_ev || !ctx.matchup_valid ||
        !ctx.ip_reach || !ctx.oop_reach) {
        throw std::runtime_error("GpuBackend::prepare: null pointers in SolverContext");
    }

    // Phase 2: GPU kernels handle iso runout enumeration. The tree may have
    // multi-child CHANCE nodes (one per canonical runout) and per-runout
    // matchup tables. Validated by parity testing vs CpuBackend.

    // Free any previous allocations (re-prepare supported)
    free_solver_state(impl_->state);
    free_locks(impl_->locks);
    free_levels(impl_->levels);
    free_reach(impl_->reach);
    free_matchup(impl_->matchup);
    free_tree(impl_->tree);

    impl_->host_tree = ctx.tree;
    impl_->iso       = ctx.iso;
    impl_->config    = ctx.config;

    // Upload tree + matchup + reach + node locks
    impl_->tree    = upload_tree(*ctx.tree);
    // Phase 2: prefer per-runout tables. Fall back to a single-table view
    // wrapping the legacy matchup_ev/_valid if the per-runout vectors are
    // empty (Phase 0/1 callers).
    if (ctx.matchup_ev_per_runout && !ctx.matchup_ev_per_runout->empty() &&
        ctx.matchup_valid_per_runout && !ctx.matchup_valid_per_runout->empty()) {
        impl_->matchup = upload_matchup(
            *ctx.matchup_ev_per_runout, *ctx.matchup_valid_per_runout,
            ctx.iso->canonical_weights);
    } else {
        std::vector<std::vector<float>> ev_one  = { *ctx.matchup_ev };
        std::vector<std::vector<float>> val_one = { *ctx.matchup_valid };
        impl_->matchup = upload_matchup(ev_one, val_one,
                                        ctx.iso->canonical_weights);
    }
    impl_->reach   = upload_reach(*ctx.ip_reach, *ctx.oop_reach);
    impl_->locks   = upload_locks(*ctx.resolved_locks);

    // Compute level schedule, keep host copies + upload device copies
    {
        const uint32_t N = ctx.tree->total_nodes;
        std::vector<uint32_t> depth = compute_node_depths(*ctx.tree);
        uint32_t max_d = 0;
        for (uint32_t d : depth) if (d > max_d) max_d = d;
        const uint32_t num_levels = max_d + 1;

        std::vector<uint32_t> count(num_levels, 0);
        for (uint32_t d : depth) count[d]++;
        std::vector<uint32_t> offsets(num_levels + 1, 0);
        for (uint32_t L = 0; L < num_levels; ++L) offsets[L + 1] = offsets[L] + count[L];

        std::vector<uint32_t> order(N, 0);
        std::vector<uint32_t> cursor(offsets);
        for (uint32_t n = 0; n < N; ++n) order[cursor[depth[n]]++] = n;

        impl_->host_node_order    = order;
        impl_->host_level_offsets = offsets;
        impl_->levels.node_order    = upload_vector(order);
        impl_->levels.level_offsets = upload_vector(offsets);
        impl_->levels.max_depth     = max_d;
        impl_->levels.num_levels    = num_levels;
    }

    // Allocate per-iteration state
    impl_->state = alloc_solver_state(ctx.tree->total_nodes,
                                       MAX_ACTIONS,
                                       ctx.iso->num_canonical,
                                       *ctx.tree);

    impl_->prepared = true;
}

// ============================================================================
// iterate: one DCFR iteration
//   1. Regret matching → current_strategy
//   2. Forward reach propagation (root → leaves)
//   3. Backward pass for OOP traverser: level 0 → max_depth
//      - At terminals: launch showdown/fold kernel
//      - At player/chance: lift children values + aggregate
//   4. Update OOP regrets + strategy_sum
//   5. Repeat 3-4 for IP traverser
// ============================================================================

void GpuBackend::iterate(int iteration) {
    if (!impl_->prepared) {
        throw std::runtime_error("GpuBackend::iterate called before prepare()");
    }
    using namespace deepsolver::gpu;  // resolve launch_* symbols
    auto& I = *impl_;
    uint16_t nc = I.iso->num_canonical;
    uint32_t N  = I.tree.num_nodes;
    uint8_t  A  = MAX_ACTIONS;

    // 1) Regret matching → current_strategy
    launch_compute_strategy(
        I.state.regrets, I.state.current_strategy,
        I.tree.num_children, I.tree.node_types,
        N, nc, A);

    // 1b) Apply node locks: override current_strategy at locked (node, combo) pairs.
    //     Matches CpuBackend::compute_strategy which inlines the lock check.
    if (I.locks.num_locks > 0) {
        int block = 128;
        int grid = (I.locks.num_locks + block - 1) / block;
        apply_locks_kernel<<<grid, block>>>(
            I.state.current_strategy,
            I.locks.node_indices, I.locks.combo_indices,
            I.locks.strategies_flat, I.locks.strategy_offsets,
            I.locks.num_locks, nc, A);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) Forward reach propagation.
    //    Initialize root reach (node 0) from the uploaded range weights.
    CUDA_CHECK(cudaMemcpy(I.state.reach_scratch_oop,
                           I.reach.oop_reach,
                           static_cast<size_t>(nc) * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(I.state.reach_scratch_ip,
                           I.reach.ip_reach,
                           static_cast<size_t>(nc) * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    // Level schedule: level 0 = leaves, max_depth = root. Propagate parents→children,
    // i.e. iterate levels from ROOT (max_depth) DOWN to 1.
    const auto& offsets = I.host_level_offsets;
    for (int L = static_cast<int>(I.levels.max_depth); L >= 1; --L) {
        uint32_t start = offsets[L];
        uint32_t end   = offsets[L + 1];
        uint32_t count = end - start;
        if (count == 0) continue;
        const uint32_t* d_level = I.levels.node_order + start;
        launch_propagate_reach(
            I.tree.node_types, I.tree.active_player, I.tree.num_children,
            I.tree.children_offset, I.tree.children,
            d_level, count,
            I.state.current_strategy, A,
            I.state.reach_scratch_oop, I.state.reach_scratch_ip, nc);
    }

    // 3-5) Backward pass + regret/strategy_sum updates for each traverser.
    // Lambda since Impl is private and can't be accessed from a free function.
    auto run_traverser = [&](int traverser) {
        const auto& tree    = *I.host_tree;
        const auto& offsets = I.host_level_offsets;
        const auto& order   = I.host_node_order;
        const uint32_t num_levels = I.levels.num_levels;

        float* reach_opp_base = (traverser == 0) ? I.state.reach_scratch_ip
                                                  : I.state.reach_scratch_oop;

        // Backward: level 0 (leaves) → max_depth (root)
        for (uint32_t L = 0; L < num_levels; ++L) {
            uint32_t start = offsets[L];
            uint32_t end   = offsets[L + 1];
            uint32_t count = end - start;
            if (count == 0) continue;

            // Terminal nodes at this level — one kernel launch per terminal
            const size_t per_table = static_cast<size_t>(nc) * nc;
            for (uint32_t i = start; i < end; ++i) {
                uint32_t n = order[i];
                auto nt = static_cast<NodeType>(tree.node_types[n]);
                if (nt != NodeType::TERMINAL) continue;

                auto tt = static_cast<TerminalType>(tree.terminal_types[n]);
                const float* reach_opp_n = reach_opp_base + static_cast<size_t>(n) * nc;
                float* out_n = I.state.node_values + static_cast<size_t>(n) * nc;

                // Phase 2: pick the correct per-runout matchup table.
                int32_t mi = (n < tree.matchup_idx.size()) ? tree.matchup_idx[n] : 0;
                if (mi < 0 || static_cast<uint32_t>(mi) >= I.matchup.num_runouts) mi = 0;
                size_t table_off = static_cast<size_t>(mi) * per_table;
                const float* mev    = I.matchup.matchup_ev    + table_off;
                const float* mvalid = I.matchup.matchup_valid + table_off;

                if (tt == TerminalType::SHOWDOWN) {
                    float half_pot = tree.pots[n] * 0.5f;
                    launch_showdown_terminal(
                        mev, mvalid,
                        I.matchup.canonical_weights, reach_opp_n,
                        nc, half_pot, traverser, out_n);
                } else {
                    // Phase 0 fold-fix: gain = (pot - parent.bet_into) / 2,
                    // not pot/2. The unmatched portion is returned to the
                    // bettor; only the matched portion is up for grabs.
                    uint32_t parent = (n < tree.parent_indices.size())
                        ? tree.parent_indices[n] : 0;
                    float unmatched = (parent < tree.total_nodes)
                        ? tree.bet_into[parent] : 0.0f;
                    float gain = (tree.pots[n] - unmatched) * 0.5f;
                    float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
                    float sign = (traverser == 0) ? sign_oop : -sign_oop;
                    launch_fold_terminal(
                        mvalid, I.matchup.canonical_weights,
                        reach_opp_n, nc, gain, sign, traverser, out_n);
                }
            }

            // Non-terminal nodes: lift children values + aggregate (kernel filters by type)
            const uint32_t* d_level = I.levels.node_order + start;
            launch_lift_child_values(
                I.tree.node_types, I.tree.num_children,
                I.tree.children_offset, I.tree.children,
                d_level, count,
                I.state.node_values, I.state.action_values,
                A, nc);
            launch_aggregate_node_values(
                I.tree.node_types, I.tree.active_player, I.tree.num_children,
                I.tree.children_offset, I.tree.children,
                I.tree.runout_weight,
                d_level, count,
                I.state.current_strategy, A,
                I.state.action_values, I.state.node_values,
                nc, traverser);
        }

        // Regret update (at traverser's own nodes)
        launch_update_regrets(
            I.state.regrets, I.state.action_values, I.state.node_values,
            I.tree.node_types, I.tree.active_player, I.tree.num_children,
            N, nc, A, traverser, iteration,
            I.config->dcfr_alpha, I.config->dcfr_beta);

        // Strategy_sum update (per-node reach weighted)
        const float* reach_own = (traverser == 0) ? I.state.reach_scratch_oop
                                                  : I.state.reach_scratch_ip;
        launch_update_strategy_sum(
            I.state.strategy_sum, I.state.current_strategy, reach_own,
            I.tree.node_types, I.tree.active_player, I.tree.num_children,
            N, nc, A, traverser, iteration, I.config->dcfr_gamma);
    };

    run_traverser(0);  // OOP
    run_traverser(1);  // IP

    // Wait for all kernels to complete before next iteration
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// finalize: strategy_sum → normalized strategy (on device, then download)
// ============================================================================

namespace {

__global__ void normalize_strategy_kernel(
    const float* __restrict__ strategy_sum,
    float*       __restrict__ strategy_out,
    const uint8_t* __restrict__ num_children,
    const uint8_t* __restrict__ node_types,
    uint32_t num_nodes, uint16_t num_canonical, uint8_t max_actions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_nodes) * num_canonical;
    if (tid >= total) return;

    int node  = tid / num_canonical;
    int combo = tid % num_canonical;

    uint8_t nt = node_types[node];
    uint8_t na = num_children[node];

    size_t base = (static_cast<size_t>(node) * max_actions) * num_canonical
                + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    if ((nt != 0 /*OOP*/ && nt != 1 /*IP*/) || na == 0) {
        for (int a = 0; a < na; ++a) strategy_out[base + a * stride] = 0.0f;
        return;
    }

    float total_sum = 0.0f;
    for (int a = 0; a < na; ++a) total_sum += strategy_sum[base + a * stride];

    if (total_sum > 1e-7f) {
        float inv = 1.0f / total_sum;
        for (int a = 0; a < na; ++a) {
            strategy_out[base + a * stride] = strategy_sum[base + a * stride] * inv;
        }
    } else {
        float u = 1.0f / static_cast<float>(na);
        for (int a = 0; a < na; ++a) strategy_out[base + a * stride] = u;
    }
}

} // anonymous namespace

void GpuBackend::finalize() {
    if (!impl_->prepared) {
        throw std::runtime_error("GpuBackend::finalize called before prepare()");
    }
    auto& I = *impl_;
    uint16_t nc = I.iso->num_canonical;
    uint32_t N  = I.tree.num_nodes;
    uint8_t  A  = MAX_ACTIONS;

    // Reuse current_strategy buffer as target for normalized averaged strategy
    {
        int total = static_cast<int>(N) * nc;
        int block = 256;
        int grid  = (total + block - 1) / block;
        normalize_strategy_kernel<<<grid, block>>>(
            I.state.strategy_sum, I.state.current_strategy,
            I.tree.num_children, I.tree.node_types, N, nc, A);
        CUDA_CHECK(cudaGetLastError());
    }

    // Download to host: pack into [node][a*nc+c] layout matching CpuBackend
    std::vector<float> host_strat(static_cast<size_t>(N) * A * nc, 0.0f);
    CUDA_CHECK(cudaMemcpy(host_strat.data(),
                           I.state.current_strategy,
                           host_strat.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Build strategy_ in same per-node format as CpuBackend
    strategy_.assign(N, {});
    for (uint32_t n = 0; n < N; ++n) {
        auto nt = static_cast<NodeType>(I.host_tree->node_types[n]);
        uint8_t na = I.host_tree->num_children[n];
        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            strategy_[n].assign(static_cast<size_t>(na) * nc, 0.0f);
            for (uint8_t a = 0; a < na; ++a) {
                for (uint16_t c = 0; c < nc; ++c) {
                    size_t src = (static_cast<size_t>(n) * A + a) * nc + c;
                    size_t dst = static_cast<size_t>(a) * nc + c;
                    strategy_[n][dst] = host_strat[src];
                }
            }
        } else {
            strategy_[n].assign(static_cast<size_t>(na) * nc, 0.0f);
        }
    }
}

} // namespace deepsolver
