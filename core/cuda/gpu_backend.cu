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
#include "card.h"
#include "showdown_rank_blocker.h"
#include "terminal_plan.h"
#include "gpu_value_layout.h"

#include "util.cuh"
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
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

// B1a inc 3: strategy-source modes, mirroring cfr_kernel.cu's enum. Kept as
// named constants here so the call sites read as intent, not magic ints.
constexpr int kStratSrcMaterialized = 0;
constexpr int kStratSrcRegrets      = 1;
constexpr int kStratSrcSum          = 2;

// B1a: strat-shaped state is COMPACT — indexed by d_node_offset[node] slot
// table, player nodes only, stride nc. Increment 2: no action_values buffer —
// aggregate and update_regrets gather child node_values directly. Increment 3:
// no current_strategy buffer either — the readers take a source pointer plus a
// mode (0 = materialized buffer, 1 = regret-match on the fly, 2 = normalize
// strategy_sum on the fly). See cfr_kernel.cu's StrategyRow.
void launch_compute_strategy(
    const float* d_regrets, float* d_current_strategy,
    const uint8_t* d_num_children, const uint8_t* d_node_types,
    const uint32_t* d_node_offset,
    uint32_t num_nodes, uint16_t nc);

void launch_propagate_reach(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_node_offset,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_strat_src, int strat_src_mode,
    float* d_reach_oop, float* d_reach_ip, uint16_t nc);

void launch_aggregate_node_values(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_node_offset, const uint32_t* d_value_row,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_strat_src, int strat_src_mode,
    float* d_node_values,
    uint16_t nc, int traverser,
    float* d_regrets, float pos_disc, float neg_disc,
    // gridDim.z = num_traversers; blockIdx.z picks the traverser and offsets
    // node_values by blockIdx.z * value_span. 1 / 0 reproduces the old launch.
    int num_traversers = 1, size_t value_span = 0);

// Postsolve variant: max-over-actions at traverser-acting nodes.
// Same signature as launch_aggregate_node_values.
void launch_aggregate_node_values_br(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_node_offset, const uint32_t* d_value_row,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_strat_src, int strat_src_mode,
    float* d_node_values,
    uint16_t nc, int traverser);

void launch_update_strategy_sum(
    float* d_strategy_sum, const float* d_strat_src, int strat_src_mode,
    const float* d_reach_own,
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_node_offset,
    uint32_t num_nodes, uint16_t nc,
    int traverser, float strat_weight, int decay_and_add);

void launch_showdown_terminal(
    const float* d_matchup_ev, const float* d_matchup_valid,
    const float* d_canonical_weights, const float* d_reach_opp,
    uint16_t nc, float half_pot, int perspective,
    float* d_out);

void launch_fold_terminal(
    const float* d_matchup_valid, const float* d_canonical_weights,
    const float* d_reach_opp, uint16_t nc, float half_pot,
    float sign_for_perspective, int perspective, float* d_out);

void launch_terminal_level(
    const uint8_t* d_node_types,
    const uint8_t* d_terminal_types,
    const float* d_pots,
    const uint32_t* d_parent_indices,
    const float* d_bet_into,
    const int32_t* d_matchup_idx,
    const uint32_t* d_value_row,
    const uint32_t* d_level_indices,
    uint32_t num_level_nodes,
    const float* d_matchup_ev_concat,
    const float* d_matchup_valid_concat,
    const float* d_canonical_weights,
    uint32_t num_runouts,
    const float* d_reach_opp_base,
    const uint16_t* d_opp_live,
    uint16_t num_opp_live,
    uint16_t nc,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* d_node_values);

void launch_rank_blocker_terminal_level(
    const uint8_t* d_node_types,
    const uint8_t* d_terminal_types,
    const float* d_pots,
    const uint32_t* d_parent_indices,
    const float* d_bet_into,
    const int32_t* d_matchup_idx,
    const uint32_t* d_value_row,
    const uint32_t* d_level_indices,
    uint32_t num_level_nodes,
    const uint16_t* d_combo_bucket,
    const uint16_t* d_bucket_count,
    const uint8_t* d_combo_card0,
    const uint8_t* d_combo_card1,
    const uint32_t* d_card_off,
    const uint16_t* d_card_list,
    uint32_t num_runouts,
    const float* d_reach_opp_base,
    uint16_t nc,
    uint16_t max_bucket_count,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* d_node_values);

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

    // Rank-blocker metadata: O(nc·~92) showdown/fold replacement for the dense
    // O(nc²) terminal kernel. Only built for singleton-isomorphism boards
    // (every canonical = 1 original, weight 1). When rb_valid is false the
    // pointers are null and the backend falls back to terminal_level_kernel.
    uint16_t* rb_combo_bucket    = nullptr;  // [num_runouts * nc] per-runout rank bucket (0xFFFF = invalid)
    uint16_t* rb_bucket_count    = nullptr;  // [num_runouts] distinct-rank count per runout
    uint8_t*  rb_combo_card0     = nullptr;  // [nc] runout-independent
    uint8_t*  rb_combo_card1     = nullptr;  // [nc] runout-independent
    uint32_t* rb_card_off        = nullptr;  // [NUM_CARDS+1] CSR offsets
    uint16_t* rb_card_list       = nullptr;  // [2*nc] canonical combos using each card
    uint16_t  rb_max_bucket_count = 0;       // max over runouts (shared-mem sizing)
    bool      rb_valid           = false;
};

/// Per-player root-level reach probabilities, on device.
struct DeviceReach {
    float* ip_reach   = nullptr;  // [nc]
    float* oop_reach  = nullptr;  // [nc]

    // B1b inc 1: the canonical combos each player actually holds — root reach
    // strictly positive — in ASCENDING order. A hand with zero reach at the
    // root has zero reach everywhere (reach is a product of strategy
    // probabilities), so this is a STATIC property of the ranges, not a
    // per-iteration one.
    //
    // The terminal kernels sum `reach_opp[cj] * ...` over every opponent combo.
    // A zero-reach term contributes exactly ±0.0 and `x + 0.0 == x` bit-for-bit,
    // so iterating this list instead of [0, nc) is bit-identical while cutting
    // the inner loop to the opponent's real range. Measured across the app's 26
    // shipped matchups on a monotone board: 15.7%-61.0% of nc per player
    // (median ~23%), so this is a 1.6x-6.4x cut on the kernel that dominates
    // GPU time on iso boards.
    //
    // ASCENDING matters: the retained terms are then added in the same relative
    // order as the dense loop, which is what makes it bit-exact rather than
    // merely equivalent.
    uint16_t* live_oop     = nullptr;  // [num_live_oop]
    uint16_t* live_ip      = nullptr;  // [num_live_ip]
    uint16_t  num_live_oop = 0;
    uint16_t  num_live_ip  = 0;
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
///
/// B1a compact layout: the strat-shaped buffers hold state ONLY for
/// player-node actions that exist — [total_slots * nc] where total_slots =
/// Σ num_children over player nodes and node_offset[n] is each player node's
/// first slot (prefix sum; 0 sentinel for chance/terminal, never
/// dereferenced). Down from [N * MAX_ACTIONS * nc], which paid 6 action
/// rows for every node including chance/terminal — the measured 76 GB on
/// multi-sizing monotone flops.
///
/// Increment 2: action_values is GONE — a parent's per-action value is its
/// child's node_values row, gathered in-kernel by aggregate/update_regrets.
struct DeviceSolverState {
    float* regrets           = nullptr;  // [total_slots * nc] compact
    float* strategy_sum      = nullptr;  // [total_slots * nc] compact
    float* current_strategy  = nullptr;  // [total_slots * nc] compact
    // Scratch buffers for backward pass (Phase 4.5 uses these)
    float* reach_scratch_oop = nullptr;  // [N * nc]   per-node reach for OOP
    float* reach_scratch_ip  = nullptr;  // [N * nc]   per-node reach for IP
    // B3 inc 1: [value_rows * nc], NOT [N * nc] — compacted terminal rows plus
    // a 2-level non-terminal window. Index through value_row[n], never by n.
    //
    // Traverser fusion: TWO such regions back to back, one per traverser, so
    // both backward passes can be in flight in a single grid. The passes write
    // different values to the same rows, so sharing one region is what forced
    // them to be serialized into 2× the launches. Region t starts at
    // `node_values + t * value_span`. The postsolve path is mutex-serialized
    // and only ever uses region 0.
    float* node_values       = nullptr;
    uint32_t* value_row      = nullptr;  // [N] node index → value buffer row
    size_t    value_rows     = 0;
    size_t    value_span     = 0;        // value_rows * nc — one region

    uint32_t* node_offset    = nullptr;  // [N] device slot table
    size_t    total_slots    = 0;        // Σ na over player nodes
    size_t    state_stride   = 0;        // total_slots * nc

    // Host copy of the slot table for finalize()'s download repack.
    std::vector<uint32_t> host_node_offset;
};

/// Topologically sorted level schedule: node indices grouped by DEPTH FROM
/// ROOT. Level 0 = root; level max_depth = the deepest leaves. Forward
/// (reach) runs 0 → max_depth, backward (values) runs max_depth → 0.
/// Every child sits exactly one level below its parent — see
/// compute_depth_from_root for why that property is the whole point.
/// Terminals are spread across levels rather than all sitting at level 0.
///
/// Layout:
///   node_order[level_offsets[L] .. level_offsets[L+1]) = node indices at level L
struct DeviceLevels {
    uint32_t* node_order    = nullptr;  // [num_nodes]
    uint32_t* level_offsets = nullptr;  // [max_depth + 2] (prefix-sum, +1 sentinel)
    uint32_t  max_depth     = 0;
    uint32_t  num_levels    = 0;        // max_depth + 1

    // B3: every terminal node index, ascending. Under the old HEIGHT-keyed
    // schedule all terminals sat at level 0, so one launch over that level
    // covered them; depth keying scatters them across levels, and launching
    // the terminal kernel once per level cost 65% of throughput on small trees
    // (549-node tree, 9 levels, 2 traversers — pure launch overhead). A
    // terminal's value depends only on REACH, never on another node's value,
    // so the whole set is still evaluable in ONE launch once the forward pass
    // has finished. That also moves the last reach reader out of the backward
    // pass, which is what the reach buffers need before they can be windowed.
    uint32_t* terminal_order = nullptr;  // [num_terminals]
    uint32_t  num_terminals  = 0;
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
    const std::vector<uint16_t>& weights,
    bool materialize_dense)
{
    DeviceMatchup dm;
    dm.num_canonical = static_cast<uint16_t>(weights.size());
    dm.num_runouts   = static_cast<uint32_t>(std::max<size_t>(1, ev_per_runout.size()));

    size_t per_table = static_cast<size_t>(dm.num_canonical) * dm.num_canonical;
    size_t total     = per_table * dm.num_runouts;

    // On rank-blocker boards the dense EV/valid tables are never read (CFR and
    // postsolve both branch to the rank-blocker kernel when rb_valid), so the
    // caller passes materialize_dense=false to skip both the nc^2 VRAM and the
    // H2D upload -- the single biggest per-subgame prepare cost. The rb pointers
    // stay null and free_matchup tolerates that.
    if (total > 0 && materialize_dense) {
        // ----------------------------------------------------------------
        // Sprint 4 (resource policy guide): chunked device upload.
        //
        // OLD: build `flat_ev` / `flat_valid` on host (total*4 B each),
        // memcpy device-once. Peak host memory at this point was the
        // ORIGINAL per-runout vectors PLUS the flattened copies, i.e.
        // double the matchup table size. On a rainbow flop with nc=1300,
        // 2000 leaves, that's 1.3 GB × 2 buffers × 2 copies = ~10 GB peak.
        //
        // NEW: allocate the device buffers up-front, cudaMemset them to
        // zero (handles missing/short tables), then loop runout-by-runout
        // copying directly from `ev_per_runout[r].data()` into the device
        // offset. Peak host = original tables only. Peak device unchanged.
        // ----------------------------------------------------------------
        size_t bytes_per_buffer = total * sizeof(float);
        size_t total_required   = bytes_per_buffer * 2;  // ev + valid

        // OOM pre-flight unchanged — still want a structured error before
        // we commit to a multi-GB allocation Rust can't recover from.
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

        // Allocate device buffers once. cudaMemset clears the missing-table
        // slices so partial inputs land on zero EV / zero valid (CFR treats
        // those as "no equity contribution", same as the old explicit fill).
        CUDA_CHECK(cudaMalloc(&dm.matchup_ev,    bytes_per_buffer));
        CUDA_CHECK(cudaMalloc(&dm.matchup_valid, bytes_per_buffer));
        CUDA_CHECK(cudaMemset(dm.matchup_ev,    0, bytes_per_buffer));
        CUDA_CHECK(cudaMemset(dm.matchup_valid, 0, bytes_per_buffer));

        // Per-runout direct copy. No host flattened buffer.
        const size_t per_table_bytes = per_table * sizeof(float);
        for (uint32_t r = 0; r < dm.num_runouts; ++r) {
            if (r < ev_per_runout.size() && ev_per_runout[r].size() == per_table) {
                CUDA_CHECK(cudaMemcpy(
                    dm.matchup_ev + static_cast<size_t>(r) * per_table,
                    ev_per_runout[r].data(),
                    per_table_bytes,
                    cudaMemcpyHostToDevice));
            }
            // else: leave zeros from the cudaMemset above. Same effect as
            // the old "skip the std::copy" branch in the flatten loop.
            if (r < valid_per_runout.size() && valid_per_runout[r].size() == per_table) {
                CUDA_CHECK(cudaMemcpy(
                    dm.matchup_valid + static_cast<size_t>(r) * per_table,
                    valid_per_runout[r].data(),
                    per_table_bytes,
                    cudaMemcpyHostToDevice));
            }
        }
    }

    std::vector<float> weights_f(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weights_f[i] = static_cast<float>(weights[i]);
    }
    dm.canonical_weights = upload_vector(weights_f);
    return dm;
}

/// Build + upload rank-blocker metadata into an already-populated DeviceMatchup.
/// No-op (leaves rb_valid=false → dense-kernel fallback) unless the board has
/// singleton isomorphism and at least one runout produced a valid rank table.
static void upload_rank_blocker(
    DeviceMatchup& dm,
    const IsomorphismMapping& iso,
    const std::vector<std::vector<uint16_t>>& ranks_per_runout)
{
    const uint16_t nc = iso.num_canonical;
    if (nc == 0) return;
    if (!showdown_rank_blocker::supports_singleton_iso(iso)) return;
    // The kernel indexes combo_bucket by matchup_idx ∈ [0, num_runouts); the
    // per-runout rank tables must cover exactly that range or we fall back.
    if (ranks_per_runout.size() != dm.num_runouts) return;

    const uint32_t R = dm.num_runouts;
    constexpr uint16_t kNoBucket = showdown_rank_blocker::Scratch::kNoBucket;

    std::vector<uint16_t> combo_bucket(static_cast<size_t>(R) * nc, kNoBucket);
    std::vector<uint16_t> bucket_count(R, 0);
    uint16_t max_bucket = 0;
    bool any_valid = false;
    for (uint32_t r = 0; r < R; ++r) {
        auto meta = showdown_rank_blocker::build_metadata(iso, ranks_per_runout[r]);
        if (!meta.valid) continue;
        any_valid = true;
        bucket_count[r] = static_cast<uint16_t>(meta.bucket_count);
        if (bucket_count[r] > max_bucket) max_bucket = bucket_count[r];
        std::copy(meta.combo_bucket.begin(), meta.combo_bucket.end(),
                  combo_bucket.begin() + static_cast<size_t>(r) * nc);
    }
    if (!any_valid || max_bucket == 0) return;

    // Runout-independent: each canonical's two cards + per-card combo CSR.
    // (The canonical set and its card assignment are fixed by the config board's
    // isomorphism; only ranks/buckets vary per runout.)
    const auto& combo_table = get_combo_table();
    std::vector<uint8_t> card0(nc), card1(nc);
    std::vector<std::vector<uint16_t>> per_card(NUM_CARDS);
    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const Card a = combo_table[oi].cards[0];
        const Card b = combo_table[oi].cards[1];
        card0[c] = a;
        card1[c] = b;
        per_card[a].push_back(c);
        per_card[b].push_back(c);
    }
    std::vector<uint32_t> card_off(NUM_CARDS + 1, 0);
    std::vector<uint16_t> card_list;
    card_list.reserve(static_cast<size_t>(2) * nc);
    for (int card = 0; card < NUM_CARDS; ++card) {
        card_off[card] = static_cast<uint32_t>(card_list.size());
        for (uint16_t c : per_card[card]) card_list.push_back(c);
    }
    card_off[NUM_CARDS] = static_cast<uint32_t>(card_list.size());

    dm.rb_combo_bucket = upload_vector(combo_bucket);
    dm.rb_bucket_count = upload_vector(bucket_count);
    dm.rb_combo_card0  = upload_vector(card0);
    dm.rb_combo_card1  = upload_vector(card1);
    dm.rb_card_off     = upload_vector(card_off);
    dm.rb_card_list    = upload_vector(card_list);
    dm.rb_max_bucket_count = max_bucket;
    dm.rb_valid = true;
}

/// Predict, before the dense upload, whether upload_rank_blocker() will end up
/// setting rb_valid=true. Must mirror its early-returns EXACTLY (same iso check,
/// same size match, same "at least one valid runout") -- if this says yes but
/// upload_rank_blocker bails, we'd skip the dense tables and then the dense
/// fallback kernel would deref null device pointers. build_metadata is re-run
/// here, but only once per prepare(), so the cost is negligible next to the
/// nc^2 upload it lets us skip.
static bool rank_blocker_activatable(
    const IsomorphismMapping& iso,
    const std::vector<std::vector<uint16_t>>& ranks_per_runout,
    uint32_t num_runouts)
{
    if (iso.num_canonical == 0) return false;
    if (!showdown_rank_blocker::supports_singleton_iso(iso)) return false;
    if (ranks_per_runout.size() != num_runouts) return false;
    for (const auto& ranks : ranks_per_runout) {
        auto meta = showdown_rank_blocker::build_metadata(iso, ranks);
        if (meta.valid && meta.bucket_count > 0) return true;
    }
    return false;
}

static void free_matchup(DeviceMatchup& dm) {
    free_device(dm.matchup_ev);
    free_device(dm.matchup_valid);
    free_device(dm.canonical_weights);
    free_device(dm.rb_combo_bucket);
    free_device(dm.rb_bucket_count);
    free_device(dm.rb_combo_card0);
    free_device(dm.rb_combo_card1);
    free_device(dm.rb_card_off);
    free_device(dm.rb_card_list);
    dm.rb_max_bucket_count = 0;
    dm.rb_valid = false;
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

    auto live_list = [](const std::vector<float>& reach) {
        std::vector<uint16_t> live;
        live.reserve(reach.size());
        for (std::size_t c = 0; c < reach.size(); ++c) {
            if (reach[c] > 0.0f) live.push_back(static_cast<uint16_t>(c));
        }
        return live;
    };
    const std::vector<uint16_t> lo = live_list(oop_reach);
    const std::vector<uint16_t> li = live_list(ip_reach);
    dr.live_oop     = upload_vector(lo);
    dr.live_ip      = upload_vector(li);
    dr.num_live_oop = static_cast<uint16_t>(lo.size());
    dr.num_live_ip  = static_cast<uint16_t>(li.size());
    return dr;
}

static void free_reach(DeviceReach& dr) {
    free_device(dr.ip_reach);
    free_device(dr.oop_reach);
    free_device(dr.live_oop);
    free_device(dr.live_ip);
    dr.num_live_oop = 0;
    dr.num_live_ip  = 0;
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
// Launches num_locks threads — one per lock entry. Lock targets are player
// decision nodes by construction (Solver::resolve_node_locks navigates the
// betting tree), so node_offset[node] is always a real slot here.
__global__ void apply_locks_kernel(
    float* __restrict__ current_strategy,         // [total_slots * nc] compact
    const uint32_t* __restrict__ node_indices,
    const uint16_t* __restrict__ combo_indices,
    const float*    __restrict__ strategies_flat,
    const uint32_t* __restrict__ strategy_offsets,
    const uint32_t* __restrict__ node_offset,     // [N] per-node slot index
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

    size_t base = static_cast<size_t>(node_offset[node]) * nc + combo;
    size_t stride = nc;
    for (uint32_t a = 0; a < count && a < max_actions; ++a) {
        current_strategy[base + a * stride] = strategies_flat[off + a];
    }
}

// ---- Topological level sort ----

/// Compute each node's DEPTH FROM ROOT (root = 0, every child = parent + 1).
///
/// B3: this schedule used to key on HEIGHT (`1 + max child height`). Under
/// height keying a node's children can sit ANY number of levels below it, so a
/// node's reach — written by its PARENT, read when the node itself is
/// processed — has an unbounded live window, and all three full-tree buffers
/// (reach_scratch_oop/_ip, node_values) must be N×nc. Depth keying puts every
/// child exactly one level below its parent, which is what lets those buffers
/// become 2-level rolling windows.
///
/// Both passes stay correctly ordered under the new key: forward runs 0 →
/// max_depth (a level's reach is written before that level propagates), and
/// backward runs max_depth → 0 (a node's children are evaluated before it
/// aggregates them). The one behavioral difference is that terminals no longer
/// all live at one level — under height keying every terminal was at level 0,
/// under depth keying they scatter, so the terminal kernel runs per level. Same
/// total work, more launches (9-13 levels on real trees).
///
/// Requires the BFS-flattened tree (child index > parent index) — the same
/// assumption the height version made, and guaranteed by GameTreeBuilder.
static std::vector<uint32_t> compute_depth_from_root(const FlatGameTree& tree) {
    const uint32_t N = tree.total_nodes;
    std::vector<uint32_t> depth(N, 0);
    if (N == 0) return depth;

    for (uint32_t n = 0; n < N; ++n) {
        const uint8_t na = tree.num_children[n];
        for (uint8_t a = 0; a < na; ++a) {
            const uint32_t child = tree.children[tree.children_offset[n] + a];
            if (child < N) depth[child] = depth[n] + 1;
        }
    }
    return depth;
}

static void free_levels(DeviceLevels& dl) {
    free_device(dl.node_order);
    free_device(dl.level_offsets);
    free_device(dl.terminal_order);
    dl.max_depth = 0;
    dl.num_levels = 0;
    dl.num_terminals = 0;
}

// ---- Solver state allocation ----

/// `materialize_strategy` (B1a inc 3): allocate the current_strategy buffer.
/// Only node-locked solves need it — the lock override is a WRITE into a
/// materialized strategy. Everything else derives the strategy from regrets
/// (mid-CFR) or strategy_sum (postsolve) on the fly, so the third
/// strat-shaped buffer simply does not exist.
static DeviceSolverState alloc_solver_state(uint32_t num_nodes,
                                              uint8_t max_actions,
                                              uint16_t num_canonical,
                                              const FlatGameTree& tree,
                                              bool materialize_strategy,
                                              const std::vector<uint32_t>& host_value_row,
                                              size_t value_rows)
{
    DeviceSolverState ds;
    size_t N  = num_nodes;
    size_t nc = num_canonical;

    // B1a: build the compact slot table — prefix sum of num_children over
    // PLAYER nodes only. Chance/terminal nodes keep the 0 sentinel (they own
    // no strat-shaped state; every kernel returns on them before touching
    // it). Mirrors CPU levelized's node_state_offset_, but stores a SLOT
    // index (multiplied by nc in-kernel) instead of a float offset.
    ds.host_node_offset.assign(N, 0u);
    size_t total_slots = 0;
    for (uint32_t n = 0; n < num_nodes; ++n) {
        auto nt = static_cast<NodeType>(tree.node_types[n]);
        uint8_t na = tree.num_children[n];
        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            ds.host_node_offset[n] = static_cast<uint32_t>(total_slots);
            total_slots += na;
        }
    }
    ds.total_slots = total_slots;
    size_t strat_stride = total_slots * nc;

    // OOM pre-flight: state buffers dominate device memory on multi-bet-size
    // monotone-flop trees. Fail with a structured message before cudaMalloc
    // returns OOM, so engine.rs's CPU-fallback detector can trigger.
    //   strat-shaped buffers (regrets, strategy_sum, + current_strategy only
    //     on node-locked solves) = 2 or 3 buffers of total_slots*nc floats
    //     (compact; inc 2 dropped action_values, inc 3 dropped
    //     current_strategy — keep bytes_for_gpu_state_compact in sync)
    //   reach = 2 buffers of N*nc floats (full tree)
    //   node_values = 2 × value_rows*nc floats — B3 inc 1 split it into
    //     compacted terminal rows + a 2-level non-terminal window, so one
    //     region is smaller than N*nc (83.9% on the 4.65M-node target); the
    //     factor 2 is the per-traverser region that lets both backward passes
    //     share one grid (kGpuValueRegions)
    const size_t strat_buffers = materialize_strategy ? 3u : 2u;
    size_t bytes_strat = strat_stride * sizeof(float);
    size_t bytes_reach = N * nc * sizeof(float);
    size_t bytes_values =
        memory_budget::kGpuValueRegions * value_rows * nc * sizeof(float);
    size_t state_bytes_required = bytes_strat * strat_buffers
                                + bytes_reach * 2 + bytes_values;
    {
        size_t free_dev = 0, total_dev = 0;
        if (cudaMemGetInfo(&free_dev, &total_dev) == cudaSuccess) {
            if (state_bytes_required > static_cast<size_t>(free_dev * 0.80)) {
                std::ostringstream oss;
                oss << "GpuBackend: solver state would require "
                    << (state_bytes_required >> 20) << " MB but only "
                    << (free_dev >> 20) << " MB free on device — out of memory. "
                    << "Tree has " << num_nodes << " nodes ("
                    << total_slots << " action slots), nc=" << nc
                    << ". Reduce flop bet sizes or run with --backend cpu.";
                throw std::runtime_error(oss.str());
            }
        }
    }

    ds.state_stride       = strat_stride;
    ds.regrets            = alloc_device_zero<float>(strat_stride);
    ds.strategy_sum       = alloc_device_zero<float>(strat_stride);
    ds.current_strategy   = materialize_strategy
        ? alloc_device_zero<float>(strat_stride) : nullptr;
    ds.reach_scratch_oop  = alloc_device_zero<float>(N * nc);
    ds.reach_scratch_ip   = alloc_device_zero<float>(N * nc);
    ds.node_values        = alloc_device_zero<float>(
        memory_budget::kGpuValueRegions * value_rows * nc);
    ds.value_rows         = value_rows;
    ds.value_span         = value_rows * nc;
    ds.value_row          = upload_vector(host_value_row);
    ds.node_offset        = upload_vector(ds.host_node_offset);

    // Initialize current_strategy to uniform 1/num_children per player node.
    // This mirrors CpuBackend::prepare(). We build on host then upload.
    // Compact layout: a player node's na action rows are contiguous starting
    // at slot host_node_offset[n], so the fill is one contiguous run.
    if (materialize_strategy && strat_stride > 0) {
        std::vector<float> host_strat(strat_stride, 0.0f);
        for (uint32_t n = 0; n < num_nodes; ++n) {
            uint8_t na = tree.num_children[n];
            auto nt = static_cast<NodeType>(tree.node_types[n]);
            if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
                float u = 1.0f / static_cast<float>(na);
                size_t begin = static_cast<size_t>(ds.host_node_offset[n]) * nc;
                std::fill(host_strat.begin() + begin,
                          host_strat.begin() + begin + static_cast<size_t>(na) * nc,
                          u);
            }
        }
        CUDA_CHECK(cudaMemcpy(ds.current_strategy, host_strat.data(),
                               strat_stride * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    (void)max_actions;  // no longer part of state sizing (kept for signature stability)
    return ds;
}

static void free_solver_state(DeviceSolverState& ds) {
    free_device(ds.regrets);
    free_device(ds.strategy_sum);
    free_device(ds.current_strategy);
    free_device(ds.reach_scratch_oop);
    free_device(ds.reach_scratch_ip);
    free_device(ds.node_values);
    free_device(ds.value_row);
    free_device(ds.node_offset);
    ds.host_node_offset.clear();
    ds.total_slots = 0;
    ds.state_stride = 0;
    ds.value_rows = 0;
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

    // B3 inc 1: node index → row in the value buffer (see prepare()).
    std::vector<uint32_t> host_value_row;
    size_t value_rows = 0;


    // Cached host-side info for finalize() and bookkeeping
    const SolverConfig*       config = nullptr;
    const IsomorphismMapping* iso    = nullptr;
    const FlatGameTree*       host_tree = nullptr;
    bool prepared = false;
    bool finalized = false;  // true once finalize() has populated current_strategy with averaged

    // The postsolve scratch buffers (reach_scratch_*, node_values) are
    // shared across run_postsolve_pass invocations. The
    // Solver's parallel-postsolve mode runs compute_combo_evs and
    // compute_exploitability concurrently, which means up to 3 passes can
    // contend for the same device memory. This mutex serializes them — the
    // GPU is fast enough that doing them sequentially is still significantly
    // ahead of CPU parallel. (Future: per-pass scratch allocation if mutex
    // serialization becomes the bottleneck.)
    std::mutex postsolve_mutex;

    // ---- Measured VRAM telemetry (benchmark-truth PR-1) ----
    // Free-VRAM samples via cudaMemGetInfo. Baseline is taken at the start
    // of each full prepare() AFTER prior allocations are released, so
    // baseline − min_free = this solve's device high-water mark, including
    // allocator granularity. Concurrent device users would inflate it; on a
    // benchmark box the solver is the only allocator, so the delta is truth.
    size_t vram_free_baseline    = 0;         // 0 = no sample yet
    size_t vram_min_free         = SIZE_MAX;
    uint64_t device_prepare_delta = 0;        // baseline − free after prepare()
    // Exact device SOLVER-STATE bytes, computed from the compact layout at
    // prepare (3 strat-shaped + 3 full-tree + node_offset). Distinct from
    // device_prepare_delta, which also swallows matchup/tree/levels/locks
    // and allocator granularity.
    uint64_t device_state_bytes_exact = 0;
    /// B1a inc 3: does this solve keep a materialized current_strategy?
    /// True only for node-locked solves. Mirrors state.current_strategy !=
    /// nullptr; kept as a named flag so the intent is greppable.
    bool     materialize_strategy = false;

    void sample_vram(bool reset_baseline = false) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
        if (reset_baseline || vram_free_baseline == 0) {
            vram_free_baseline = free_b;
            vram_min_free      = free_b;
            return;
        }
        if (free_b < vram_min_free) vram_min_free = free_b;
    }
    uint64_t peak_vram_used() const {
        if (vram_free_baseline == 0 || vram_min_free == SIZE_MAX) return 0;
        return (vram_free_baseline > vram_min_free)
            ? static_cast<uint64_t>(vram_free_baseline - vram_min_free) : 0;
    }

    ~Impl() {
        free_solver_state(state);
        free_locks(locks);
        free_levels(levels);
        free_reach(reach);
        free_matchup(matchup);
        free_tree(tree);
    }

    /// Run one postsolve traversal:
    ///   - Reset reach scratch from root reach
    ///   - Forward propagate reach using averaged strategy (= current_strategy
    ///     after finalize)
    ///   - Bottom-up: terminal eval at level 0, then lift + aggregate at every
    ///     level. `best_response` selects which aggregator (sum vs per-combo max
    ///     at traverser-acting nodes).
    ///   - Returns root node_values[0..nc] downloaded to host.
    /// Empty on failure (not finalized, OOM-safe).
    std::vector<float> run_postsolve_pass(int traverser, bool best_response);
};

// ============================================================================
// GpuBackend interface
// ============================================================================

GpuBackend::GpuBackend()
    : impl_(std::make_unique<Impl>()),
      name_(detect_device_name())
{}

GpuBackend::~GpuBackend() = default;

uint64_t GpuBackend::allocated_state_bytes() const {
    return impl_->device_state_bytes_exact;
}

uint64_t GpuBackend::allocated_device_total_bytes() const {
    return impl_->device_prepare_delta;
}

uint64_t GpuBackend::measured_peak_vram_bytes() const {
    return impl_->peak_vram_used();
}

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
    impl_->finalized = false;
    impl_->prepared = false;

    impl_->host_tree = ctx.tree;
    impl_->iso       = ctx.iso;
    impl_->config    = ctx.config;

    // Baseline free-VRAM sample AFTER the frees above: peak/delta then
    // measure THIS solve's footprint, not leftovers from a prior prepare.
    impl_->sample_vram(/*reset_baseline=*/true);

    // Maturity Phase 4: previously CUDA_CHECK exit'd the process on cudaMalloc
    // failure, leaking everything we'd already allocated. Now CUDA_CHECK throws,
    // so we must release partial allocations on the way out — otherwise a later
    // CPU-fallback retry on the same Solver instance would still hold the leaked
    // device pages and could cascade to a second OOM. Catch, free, rethrow.
    try {
        // Upload tree + matchup + reach + node locks
        impl_->tree    = upload_tree(*ctx.tree);
        // Phase 2: prefer per-runout tables. Fall back to a single-table view
        // wrapping the legacy matchup_ev/_valid if the per-runout vectors are
        // empty (Phase 0/1 callers).
        // Skip dense EV/valid when the rank-blocker will serve every terminal.
        // The self-check needs both tables resident to diff them, so it forces
        // dense on. Must be decided BEFORE upload_matchup; rank_blocker_activatable
        // mirrors upload_rank_blocker's gate so the prediction can't disagree.
        if (ctx.matchup_ev_per_runout && !ctx.matchup_ev_per_runout->empty() &&
            ctx.matchup_valid_per_runout && !ctx.matchup_valid_per_runout->empty()) {
            const uint32_t num_runouts =
                static_cast<uint32_t>(ctx.matchup_ev_per_runout->size());
            const bool rb_activatable =
                ctx.matchup_original_ranks_per_runout &&
                rank_blocker_activatable(
                    *ctx.iso, *ctx.matchup_original_ranks_per_runout, num_runouts);
            const bool local_dense =
                terminal_selfcheck_forced() || !rb_activatable;
            // PR-4: the shared TerminalRepresentationPlan is the decision of
            // record — the memory gates already priced THIS choice. The local
            // derivation is kept as a hard consistency check: if plan and
            // backend ever disagree, the gate charged the wrong footprint, so
            // fail loudly instead of running past the user's budget.
            bool materialize_dense = local_dense;
            // A4-host inc 3: the host may have skipped building the dense
            // tables entirely. Uploading from them would read empty vectors —
            // catch it here, where the message can still name the cause.
            if (!ctx.matchup_dense_materialized && local_dense) {
                throw std::runtime_error(
                    "GpuBackend: dense matchup upload required but the host "
                    "skipped materializing the dense tables (A4-host inc 3 "
                    "predictor disagrees with the backend's rank-blocker "
                    "activation).");
            }
            if (ctx.terminal_plan && ctx.terminal_plan->refined_with_ranks) {
                materialize_dense = ctx.terminal_plan->device_dense_upload;
                if (materialize_dense != local_dense) {
                    throw std::runtime_error(
                        "TerminalRepresentationPlan disagrees with GPU dense-"
                        "upload derivation (plan says " +
                        std::string(materialize_dense ? "dense" : "skip") +
                        ", backend derived " +
                        std::string(local_dense ? "dense" : "skip") +
                        ") - the memory gates priced the wrong footprint.");
                }
            }
            impl_->matchup = upload_matchup(
                *ctx.matchup_ev_per_runout, *ctx.matchup_valid_per_runout,
                ctx.iso->canonical_weights, materialize_dense);
        } else {
            // Legacy single-table path (Phase 0/1 callers): no per-runout rank
            // tables, so the rank-blocker never activates -- keep dense.
            std::vector<std::vector<float>> ev_one  = { *ctx.matchup_ev };
            std::vector<std::vector<float>> val_one = { *ctx.matchup_valid };
            impl_->matchup = upload_matchup(ev_one, val_one,
                                            ctx.iso->canonical_weights, true);
        }
        // Rank-blocker fast path for singleton-isomorphism boards. Builds the
        // per-runout rank buckets + per-card combo CSR; leaves rb_valid=false
        // (dense-kernel fallback) for iso boards or missing rank tables.
        if (ctx.matchup_original_ranks_per_runout &&
            !ctx.matchup_original_ranks_per_runout->empty()) {
            upload_rank_blocker(impl_->matchup, *ctx.iso,
                                *ctx.matchup_original_ranks_per_runout);
        }
        impl_->reach   = upload_reach(*ctx.ip_reach, *ctx.oop_reach);
        impl_->locks   = upload_locks(*ctx.resolved_locks);

        // Compute level schedule, keep host copies + upload device copies
        {
            const uint32_t N = ctx.tree->total_nodes;
            std::vector<uint32_t> depth = compute_depth_from_root(*ctx.tree);
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

            // DEEPSOLVER_LEVEL_HISTOGRAM=1 — B3 sizing instrument. The three
            // full-tree buffers (reach_scratch_oop/_ip, node_values) can be
            // windowed to the widest ADJACENT PAIR of depth levels, because a
            // node's row is written one level above it and read on its own
            // level. That pair width, not N, is what those buffers must hold —
            // so this is the number that says whether a given tree fits.
            // Costs nothing when the env var is unset.
            if (std::getenv("DEEPSOLVER_LEVEL_HISTOGRAM") != nullptr) {
                uint32_t n_term = 0, n_player = 0, n_chance = 0;
                for (uint32_t n = 0; n < N; ++n) {
                    auto nt = static_cast<NodeType>(ctx.tree->node_types[n]);
                    if (nt == NodeType::TERMINAL) ++n_term;
                    else if (nt == NodeType::CHANCE) ++n_chance;
                    else ++n_player;
                }
                // Non-terminal counts are tracked separately because the value
                // buffer can drop terminal rows into a compacted side table
                // (terminals are evaluated in the forward descent, where their
                // reach is live), leaving only non-terminal rows to window.
                // Whether that split beats one full N×nc buffer depends
                // entirely on these widths, so print both.
                std::vector<uint32_t> nonterm(num_levels, 0);
                for (uint32_t n = 0; n < N; ++n) {
                    if (static_cast<NodeType>(ctx.tree->node_types[n]) !=
                        NodeType::TERMINAL) nonterm[depth[n]]++;
                }
                uint32_t widest = 0, widest_pair = 0, widest_nt_pair = 0;
                for (uint32_t L = 0; L < num_levels; ++L) {
                    widest = std::max(widest, count[L]);
                    const uint32_t pair =
                        count[L] + (L + 1 < num_levels ? count[L + 1] : 0u);
                    widest_pair = std::max(widest_pair, pair);
                    const uint32_t nt_pair =
                        nonterm[L] + (L + 1 < num_levels ? nonterm[L + 1] : 0u);
                    widest_nt_pair = std::max(widest_nt_pair, nt_pair);
                }
                std::fprintf(stderr,
                    "[levels] N=%u player=%u chance=%u terminal=%u | depth-levels=%u "
                    "widest=%u (%.1f%% of N) widest-2-level-window=%u (%.1f%% of N) "
                    "| nonterm-2-level-window=%u (%.1f%% of N) "
                    "value-split=%.1f%% of N vs 100%% whole\n",
                    N, n_player, n_chance, n_term, num_levels, widest,
                    100.0 * widest / static_cast<double>(N),
                    widest_pair, 100.0 * widest_pair / static_cast<double>(N),
                    widest_nt_pair, 100.0 * widest_nt_pair / static_cast<double>(N),
                    100.0 * (n_term + widest_nt_pair) / static_cast<double>(N));
                std::fprintf(stderr, "[levels] depth widths:");
                for (uint32_t L = 0; L < num_levels; ++L) std::fprintf(stderr, " %u", count[L]);
                std::fprintf(stderr, "\n[levels] nonterm widths:");
                for (uint32_t L = 0; L < num_levels; ++L) std::fprintf(stderr, " %u", nonterm[L]);
                std::fprintf(stderr, "\n");

                // B1b sizing: every nc-factored buffer is sized to the GLOBAL
                // canonical count, but a buffer only ever holds rows for one
                // player's hands — regrets/strategy_sum at an OOP-acting node
                // only index OOP hands, reach_oop only OOP hands. A hand with
                // zero reach at the ROOT is zero everywhere (reach is a product
                // of strategy probabilities), so per-player live counts are a
                // STATIC bound, not a per-iteration one. Print the ratio the
                // per-player dimension would buy.
                uint32_t live_oop = 0, live_ip = 0, live_union = 0;
                for (uint16_t c = 0; c < ctx.iso->num_canonical; ++c) {
                    const bool o = (*ctx.oop_reach)[c] > 0.0f;
                    const bool i = (*ctx.ip_reach)[c]  > 0.0f;
                    if (o) ++live_oop;
                    if (i) ++live_ip;
                    if (o || i) ++live_union;
                }
                const uint32_t ncu = ctx.iso->num_canonical;
                std::fprintf(stderr,
                    "[b1b] nc=%u live_oop=%u (%.1f%%) live_ip=%u (%.1f%%) "
                    "per-player mean=%.1f%% union=%u (%.1f%%) of nc\n",
                    ncu, live_oop, 100.0 * live_oop / ncu,
                    live_ip, 100.0 * live_ip / ncu,
                    100.0 * 0.5 * (live_oop + live_ip) / ncu,
                    live_union, 100.0 * live_union / ncu);
            }

            std::vector<uint32_t> terminals;
            terminals.reserve(N);
            for (uint32_t n = 0; n < N; ++n) {
                if (static_cast<NodeType>(ctx.tree->node_types[n]) ==
                    NodeType::TERMINAL) {
                    terminals.push_back(n);
                }
            }
            impl_->levels.terminal_order = upload_vector(terminals);
            impl_->levels.num_terminals  = static_cast<uint32_t>(terminals.size());

            // B3 inc 1: the value buffer's row map. node_values was [N][nc];
            // it is now a two-region buffer and every consumer indexes through
            // this table instead of by node index.
            //
            //   TERMINAL rows  — one per terminal, packed in node order. A
            //     terminal's value is written once per traverser pass (all of
            //     them in a single launch, since it depends only on reach) and
            //     read by its parent during the backward ascent. That span is
            //     the whole ascent, so these rows cannot be windowed.
            //   NON-TERMINAL rows — a 2-level rolling window, addressed by
            //     (depth parity, slot within level). A non-terminal's value is
            //     written when its level is aggregated and read one level up,
            //     so level L and L+1 must coexist and nothing else must. Level
            //     L-1 has the same parity as L+1 and overwrites exactly the
            //     rows that just went dead.
            //
            // Measured on the 4.65M-node target: 63.4% terminal + a 20.5%
            // non-terminal window = 83.9% of a full N×nc buffer.
            std::vector<uint32_t> nt_slot(N, 0);
            std::vector<uint32_t> nt_width(num_levels, 0);
            for (uint32_t n = 0; n < N; ++n) {
                if (static_cast<NodeType>(ctx.tree->node_types[n]) !=
                    NodeType::TERMINAL) {
                    nt_slot[n] = nt_width[depth[n]]++;
                }
            }
            uint32_t nt_parity_width[2] = {0, 0};
            for (uint32_t L = 0; L < num_levels; ++L) {
                nt_parity_width[L & 1u] =
                    std::max(nt_parity_width[L & 1u], nt_width[L]);
            }
            const uint32_t nt_base[2] = {
                static_cast<uint32_t>(terminals.size()),
                static_cast<uint32_t>(terminals.size()) + nt_parity_width[0]
            };
            std::vector<uint32_t> value_row(N, 0);
            uint32_t term_ordinal = 0;
            for (uint32_t n = 0; n < N; ++n) {
                if (static_cast<NodeType>(ctx.tree->node_types[n]) ==
                    NodeType::TERMINAL) {
                    value_row[n] = term_ordinal++;
                } else {
                    value_row[n] = nt_base[depth[n] & 1u] + nt_slot[n];
                }
            }
            impl_->host_value_row = std::move(value_row);
            impl_->value_rows = static_cast<size_t>(terminals.size())
                              + nt_parity_width[0] + nt_parity_width[1];

            // The estimator sizes the VRAM gate from gpu_value_rows(tree); if
            // that ever disagrees with the table actually built here, the gate
            // is deciding on a footprint the solve does not have. Same
            // hard-check discipline the TerminalRepresentationPlan uses.
            const uint64_t predicted = gpu_value_rows(*ctx.tree);
            if (predicted != impl_->value_rows) {
                std::ostringstream oss;
                oss << "GpuBackend: value-row layout disagrees with the "
                       "estimator — gpu_value_rows() says " << predicted
                    << " but prepare() built " << impl_->value_rows
                    << ". The VRAM gate would be pricing the wrong footprint.";
                throw std::runtime_error(oss.str());
            }

            impl_->host_node_order    = order;
            impl_->host_level_offsets = offsets;
            impl_->levels.node_order    = upload_vector(order);
            impl_->levels.level_offsets = upload_vector(offsets);
            impl_->levels.max_depth     = max_d;
            impl_->levels.num_levels    = num_levels;
        }

        // Allocate per-iteration state. B1a inc 3: the current_strategy
        // buffer exists only when node locks do (they override it); the locks
        // are already uploaded above, so this is the authoritative answer.
        impl_->materialize_strategy = (impl_->locks.num_locks > 0);
        impl_->state = alloc_solver_state(ctx.tree->total_nodes,
                                           MAX_ACTIONS,
                                           ctx.iso->num_canonical,
                                           *ctx.tree,
                                           impl_->materialize_strategy,
                                           impl_->host_value_row,
                                           impl_->value_rows);
        {
            const uint64_t nc64 = ctx.iso->num_canonical;
            const uint64_t n64  = ctx.tree->total_nodes;
            const uint64_t strat_buffers = impl_->materialize_strategy ? 3ULL : 2ULL;
            impl_->device_state_bytes_exact =
                strat_buffers * impl_->state.state_stride * sizeof(float)
              + 2ULL * n64 * nc64 * sizeof(float)                  // reach ×2
              + memory_budget::kGpuValueRegions                    // B3 inc 1 +
                    * static_cast<uint64_t>(impl_->value_rows)     // one region
                    * nc64 * sizeof(float)                         // per traverser
              + 2ULL * n64 * sizeof(uint32_t);   // node_offset + value_row
        }

        // Post-allocation sample: prepare() holds every buffer the CFR loop
        // uses, so baseline − now = the device bytes this solve allocated.
        impl_->sample_vram();
        if (impl_->vram_free_baseline > 0 &&
            impl_->vram_min_free != SIZE_MAX &&
            impl_->vram_free_baseline > impl_->vram_min_free) {
            impl_->device_prepare_delta = static_cast<uint64_t>(
                impl_->vram_free_baseline - impl_->vram_min_free);
        }

        impl_->prepared = true;
    } catch (...) {
        // Partial allocation rollback. Each free_* is safe on a half-populated
        // struct because they only free non-null members. Reset prepared so a
        // subsequent call to iterate() throws cleanly instead of UB-touching
        // freed pointers.
        free_solver_state(impl_->state);
        free_locks(impl_->locks);
        free_levels(impl_->levels);
        free_reach(impl_->reach);
        free_matchup(impl_->matchup);
        free_tree(impl_->tree);
        impl_->host_node_order.clear();
        impl_->host_level_offsets.clear();
        impl_->prepared = false;
        impl_->finalized = false;
        throw;
    }
}

void GpuBackend::reprepare_keep_board(const SolverContext& ctx) {
    if (!ctx.tree || !ctx.iso || !ctx.config ||
        !ctx.ip_reach || !ctx.oop_reach) {
        throw std::runtime_error("GpuBackend::reprepare_keep_board: null pointers in SolverContext");
    }
    // First prepare on this backend, or a board whose tree differs from the
    // resident one → full prepare (uploads tree + matchup + levels). Only when
    // the board matches do we keep them. tree node count is a cheap, sufficient
    // signature here: the decomposition pins ONE board per backend, so the tree
    // (hence matchup) never changes across re-solves of a pinned leaf.
    if (!impl_->host_tree ||
        ctx.tree->total_nodes != impl_->host_tree->total_nodes) {
        prepare(ctx);
        return;
    }

    // Keep board-fixed device data (tree, matchup + rank-blocker, levels). Re-do
    // only the range-dependent parts. alloc_solver_state re-zeros regrets/
    // strategy_sum and re-uniforms current_strategy EXACTLY as prepare() does,
    // and the kept tree/matchup bytes are identical to a re-upload — so the
    // ensuing CFR is bit-identical to a full prepare() for this board.
    free_solver_state(impl_->state);
    free_locks(impl_->locks);
    free_reach(impl_->reach);
    impl_->finalized = false;
    impl_->prepared  = false;
    impl_->config    = ctx.config;   // iso / host_tree are board-fixed; keep.

    try {
        impl_->reach = upload_reach(*ctx.ip_reach, *ctx.oop_reach);
        impl_->locks = upload_locks(*ctx.resolved_locks);
        impl_->materialize_strategy = (impl_->locks.num_locks > 0);
        impl_->state = alloc_solver_state(impl_->host_tree->total_nodes,
                                          MAX_ACTIONS,
                                          impl_->iso->num_canonical,
                                          *impl_->host_tree,
                                          impl_->materialize_strategy,
                                          impl_->host_value_row,
                                          impl_->value_rows);
        impl_->sample_vram();  // keep-board path: track min-free vs the
                               // original prepare()'s baseline
        impl_->prepared = true;
    } catch (...) {
        free_solver_state(impl_->state);
        free_locks(impl_->locks);
        free_reach(impl_->reach);
        impl_->prepared = false;
        impl_->finalized = false;
        throw;
    }
}

void GpuBackend::reprepare_keep_state(const SolverContext& ctx) {
    if (!ctx.tree || !ctx.iso || !ctx.config ||
        !ctx.ip_reach || !ctx.oop_reach) {
        throw std::runtime_error("GpuBackend::reprepare_keep_state: null pointers in SolverContext");
    }
    // Nothing resident to continue from (first solve on this backend, or a
    // different board) → cold path, which handles both cases correctly.
    if (!impl_->prepared || !impl_->host_tree || !impl_->state.regrets ||
        ctx.tree->total_nodes != impl_->host_tree->total_nodes) {
        reprepare_keep_board(ctx);
        return;
    }
    // B1a inc 3: whether a current_strategy buffer exists is decided at
    // ALLOCATION time from the lock set. If this re-solve's locks disagree
    // with the resident state's, keeping that state would either write locks
    // into a null buffer or read a stale one — reallocate instead.
    if ((!ctx.resolved_locks->empty()) != impl_->materialize_strategy) {
        reprepare_keep_board(ctx);
        return;
    }

    // Keep board-fixed device data (tree, matchup, levels) AND the solver
    // state (regrets, strategy_sum). The regret-matched strategy is derived
    // per iteration (inc 3) and finalize() normalizes on the host, so no
    // strategy state needs restoring here. Refresh only the range-dependent
    // uploads.
    free_locks(impl_->locks);
    free_reach(impl_->reach);
    impl_->finalized = false;
    impl_->config    = ctx.config;   // iso / host_tree are board-fixed; keep.

    try {
        impl_->reach = upload_reach(*ctx.ip_reach, *ctx.oop_reach);
        impl_->locks = upload_locks(*ctx.resolved_locks);
    } catch (...) {
        // Reach/locks are gone but state is intact and unusable without them:
        // drop the whole prepare so the next call rebuilds from scratch.
        free_solver_state(impl_->state);
        free_locks(impl_->locks);
        free_reach(impl_->reach);
        impl_->prepared  = false;
        impl_->finalized = false;
        throw;
    }
}

// ============================================================================
// iterate: one DCFR iteration
//   1. Regret matching → current_strategy
//   2. Forward reach propagation (root → leaves)
//   3. Backward pass for OOP traverser: max_depth → level 0 (root)
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

    // Compute DCFR discount factors per the configured schedule. Branches on
    // STANDARD vs POSTFLOP_STYLE — see solver_backend.h for formulas.
    float pos_disc, neg_disc, strat_weight;
    compute_dcfr_factors(iteration, *I.config, pos_disc, neg_disc, strat_weight);
    bool decay_and_add = dcfr_decay_and_add(*I.config);

    // [diag] DEEPSOLVER_GPU_ITERHASH: per-phase FNV-1a hashes of device state,
    // printed to stderr, for localizing run-to-run divergence between two runs
    // of the same solve. Diagnostic only.
    static const bool kIterHash =
        (std::getenv("DEEPSOLVER_GPU_ITERHASH") != nullptr);
    auto hash_dev = [](const float* dptr, size_t count) -> unsigned long long {
        if (!dptr || count == 0) return 0ull;
        std::vector<float> host(count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), dptr, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        unsigned long long h = 1469598103934665603ull;
        const unsigned char* b =
            reinterpret_cast<const unsigned char*>(host.data());
        for (size_t i = 0; i < count * sizeof(float); ++i) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
        return h;
    };

    // 1) Strategy source for this iteration (B1a inc 3).
    //    Locked solves materialize it (regret matching + the lock override
    //    written on top); everything else regret-matches inside each reader,
    //    which removes the third strat-shaped buffer entirely.
    const int   strat_src_mode = I.materialize_strategy
        ? kStratSrcMaterialized : kStratSrcRegrets;
    const float* strat_src = I.materialize_strategy
        ? I.state.current_strategy : I.state.regrets;

    if (I.materialize_strategy) {
        launch_compute_strategy(
            I.state.regrets, I.state.current_strategy,
            I.tree.num_children, I.tree.node_types,
            I.state.node_offset,
            N, nc);
    }

    // 1b) Apply node locks: override current_strategy at locked (node, combo) pairs.
    //     Matches CpuBackend::compute_strategy which inlines the lock check.
    if (I.locks.num_locks > 0) {
        int block = 128;
        int grid = (I.locks.num_locks + block - 1) / block;
        apply_locks_kernel<<<grid, block>>>(
            I.state.current_strategy,
            I.locks.node_indices, I.locks.combo_indices,
            I.locks.strategies_flat, I.locks.strategy_offsets,
            I.state.node_offset,
            I.locks.num_locks, nc, A);
        CUDA_CHECK(cudaGetLastError());
    }
    if (kIterHash) {
        // inc 3: hash the SOURCE the readers will use (mode 0 = materialized
        // buffer, 1 = regrets). Hashing a now-nullptr current_strategy would
        // have silently reported 0 on every default-path solve.
        std::fprintf(stderr, "[ih] it=%d strat_src(mode=%d)=%016llx\n",
                     iteration, strat_src_mode,
                     hash_dev(strat_src, I.state.state_stride));
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

    // Depth-keyed schedule: level 0 = root, max_depth = deepest leaves.
    // Reach propagates parents→children, so iterate ROOT-DOWN and launch on the
    // PARENT level. The deepest level has no children by construction, so it is
    // skipped rather than launched on.
    const auto& offsets = I.host_level_offsets;
    for (uint32_t L = 0; L < I.levels.max_depth; ++L) {
        uint32_t start = offsets[L];
        uint32_t end   = offsets[L + 1];
        uint32_t count = end - start;
        if (count == 0) continue;
        const uint32_t* d_level = I.levels.node_order + start;
        launch_propagate_reach(
            I.tree.node_types, I.tree.active_player, I.tree.num_children,
            I.tree.children_offset, I.tree.children,
            I.state.node_offset,
            d_level, count,
            strat_src, strat_src_mode,
            I.state.reach_scratch_oop, I.state.reach_scratch_ip, nc);
    }
    if (kIterHash) {
        const size_t nvspan = static_cast<size_t>(N) * nc;
        std::fprintf(stderr, "[ih] it=%d reachO=%016llx reachI=%016llx\n",
                     iteration,
                     hash_dev(I.state.reach_scratch_oop, nvspan),
                     hash_dev(I.state.reach_scratch_ip, nvspan));
    }

    // 3-5) Backward pass + regret/strategy_sum updates for each traverser.
    // Lambda since Impl is private and can't be accessed from a free function.
    auto run_traverser = [&](int traverser) {
        const auto& offsets = I.host_level_offsets;
        const uint32_t num_levels = I.levels.num_levels;

        // This traverser's own value region. Both passes write the same ROWS
        // with different values, so separate regions are what makes them
        // independent — and independence is what lets them share a grid.
        float* nv = I.state.node_values
                  + static_cast<size_t>(traverser) * I.state.value_span;

        float* reach_opp_base = (traverser == 0) ? I.state.reach_scratch_ip
                                                  : I.state.reach_scratch_oop;
        // B1b inc 1: whichever reach the terminals read, walk THAT player's
        // live list. Getting these two out of step would silently drop or
        // double-count opponent hands.
        const uint16_t* opp_live = (traverser == 0) ? I.reach.live_ip
                                                   : I.reach.live_oop;
        const uint16_t  num_opp_live = (traverser == 0) ? I.reach.num_live_ip
                                                       : I.reach.num_live_oop;

        // All terminals in ONE launch. Their value depends only on reach —
        // which the forward pass has already finished — so they do not need to
        // be interleaved with the level sweep, and keeping them out of it is
        // what keeps the depth-keyed schedule from paying 9-13× the launch
        // overhead on small trees.
        if (I.levels.num_terminals > 0) {
            if (I.matchup.rb_valid) {
                launch_rank_blocker_terminal_level(
                    I.tree.node_types, I.tree.terminal_types, I.tree.pots,
                    I.tree.parent_indices, I.tree.bet_into, I.tree.matchup_idx,
                    I.state.value_row,
                    I.levels.terminal_order, I.levels.num_terminals,
                    I.matchup.rb_combo_bucket, I.matchup.rb_bucket_count,
                    I.matchup.rb_combo_card0, I.matchup.rb_combo_card1,
                    I.matchup.rb_card_off, I.matchup.rb_card_list,
                    I.matchup.num_runouts, reach_opp_base, nc,
                    I.matchup.rb_max_bucket_count, traverser,
                    I.config->rake_rate, I.config->rake_cap,
                    nv);
            } else {
                launch_terminal_level(
                    I.tree.node_types,
                    I.tree.terminal_types,
                    I.tree.pots,
                    I.tree.parent_indices,
                    I.tree.bet_into,
                    I.tree.matchup_idx,
                    I.state.value_row,
                    I.levels.terminal_order, I.levels.num_terminals,
                    I.matchup.matchup_ev,
                    I.matchup.matchup_valid,
                    I.matchup.canonical_weights,
                    I.matchup.num_runouts,
                    reach_opp_base,
                    opp_live, num_opp_live,
                    nc, traverser,
                    I.config->rake_rate, I.config->rake_cap,
                    nv);
            }
        }

        // Strategy_sum update — branch on schedule (decay-and-add for
        // POSTFLOP, accumulative reach-weighted for STANDARD).
        //
        // ORDER MATTERS since B1a inc 3: this reads the ITERATION-START
        // strategy. With the materialized buffer that was a snapshot and the
        // order was free; deriving it from regrets makes "before
        // update_regrets" the only correct position.
        //
        // B3 inc 1 moved it AHEAD of the backward pass, because update_regrets
        // now runs inside that pass. Its inputs (reach + regrets) are untouched
        // by the backward pass, so the move is inert — but the constraint it
        // encodes is now structural rather than a convention.
        const float* reach_own = (traverser == 0) ? I.state.reach_scratch_oop
                                                  : I.state.reach_scratch_ip;
        launch_update_strategy_sum(
            I.state.strategy_sum, strat_src, strat_src_mode, reach_own,
            I.tree.node_types, I.tree.active_player, I.tree.num_children,
            I.state.node_offset,
            N, nc, traverser, strat_weight, decay_and_add ? 1 : 0);
    };

    // Backward: deepest level → root, BOTH traversers in one grid. Every child
    // sits exactly one level below its parent, so a level's children are all
    // evaluated before it.
    //
    // Traverser fusion (ROADMAP §2(4)): this used to run once per traverser,
    // which made 26 of the 42 launches per iteration 13 depth levels × 2. The
    // two passes are independent — the opponent-acting branch reads no
    // strategy at all ("absorbed into reach"), regret writes are partitioned by
    // active_player, and each traverser now owns its own value region — so the
    // only thing that ever coupled them was sharing one node_values buffer.
    // Measured launch cost: the kernel averages 2.6 µs to RUN and 3.26 µs to
    // ISSUE.
    auto run_backward_fused = [&]() {
        const auto& offsets = I.host_level_offsets;
        const uint32_t num_levels = I.levels.num_levels;
        for (int L = static_cast<int>(num_levels) - 1; L >= 0; --L) {
            uint32_t start = offsets[L];
            uint32_t end   = offsets[L + 1];
            uint32_t count = end - start;
            if (count == 0) continue;
            const uint32_t* d_level = I.levels.node_order + start;

            launch_aggregate_node_values(
                I.tree.node_types, I.tree.active_player, I.tree.num_children,
                I.tree.children_offset, I.tree.children,
                I.tree.runout_weight,
                I.state.node_offset, I.state.value_row,
                d_level, count,
                strat_src, strat_src_mode,
                I.state.node_values,
                nc, /*traverser=*/0,
                // Regret update for THIS level, fused. It used to be one sweep
                // over all N nodes after the pass, which only worked while
                // every value row stayed live; the window forces it per level,
                // and a separate launch per level cost 23% of small-tree
                // throughput. Fused it is launch-free and re-reads nothing.
                I.state.regrets, pos_disc, neg_disc,
                static_cast<int>(memory_budget::kGpuValueRegions),
                I.state.value_span);
        }
    };

    // Deterministic kernel-vs-kernel self-check (env DEEPSOLVER_RB_SELFCHECK).
    // Runs the dense terminal kernel and the rank-blocker kernel on the SAME
    // reach into separate buffers and reports the max divergence. Their only
    // legitimate difference is FP summation order (~ULP); a large diff means a
    // structural bug (indexing / category / per-runout slice). One-shot.
    static const bool kRbSelfCheck =
        (std::getenv("DEEPSOLVER_RB_SELFCHECK") != nullptr);
    if (kRbSelfCheck && iteration == 0 && I.matchup.rb_valid &&
        I.levels.num_levels > 0) {
        // B3 inc 1: these mirror node_values, so they are value-row shaped.
        const size_t span = I.state.value_rows * nc;
        float* d_dense = alloc_device_zero<float>(span);
        float* d_rb    = alloc_device_zero<float>(span);
        // Under height keying level 0 held EVERY terminal, so the check swept
        // that one level. Depth keying scatters terminals across levels, so it
        // sweeps the whole node list instead — both kernels filter by node
        // type, and this is a one-shot debug path.
        const uint32_t start = 0;
        const uint32_t count = N;
        const uint32_t* d_level = I.levels.node_order + start;
        for (int trav = 0; trav < 2; ++trav) {
            float* reach_opp = (trav == 0) ? I.state.reach_scratch_ip
                                           : I.state.reach_scratch_oop;
            launch_terminal_level(
                I.tree.node_types, I.tree.terminal_types, I.tree.pots,
                I.tree.parent_indices, I.tree.bet_into, I.tree.matchup_idx,
                I.state.value_row, d_level, count,
                I.matchup.matchup_ev, I.matchup.matchup_valid,
                I.matchup.canonical_weights, I.matchup.num_runouts,
                reach_opp,
                (trav == 0) ? I.reach.live_ip : I.reach.live_oop,
                (trav == 0) ? I.reach.num_live_ip : I.reach.num_live_oop,
                nc, trav,
                I.config->rake_rate, I.config->rake_cap, d_dense);
            launch_rank_blocker_terminal_level(
                I.tree.node_types, I.tree.terminal_types, I.tree.pots,
                I.tree.parent_indices, I.tree.bet_into, I.tree.matchup_idx,
                I.state.value_row, d_level, count,
                I.matchup.rb_combo_bucket, I.matchup.rb_bucket_count,
                I.matchup.rb_combo_card0, I.matchup.rb_combo_card1,
                I.matchup.rb_card_off, I.matchup.rb_card_list,
                I.matchup.num_runouts, reach_opp, nc,
                I.matchup.rb_max_bucket_count, trav,
                I.config->rake_rate, I.config->rake_cap, d_rb);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> hd(span), hr(span);
            CUDA_CHECK(cudaMemcpy(hd.data(), d_dense, span * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), d_rb, span * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double max_abs = 0.0, sum_ref = 0.0;
            uint32_t worst_node = 0; uint16_t worst_c = 0;
            for (uint32_t k = start; k < start + count; ++k) {
                uint32_t n2 = I.host_node_order[k];
                if (static_cast<NodeType>(I.host_tree->node_types[n2]) !=
                    NodeType::TERMINAL) continue;
                for (uint16_t c = 0; c < nc; ++c) {
                    size_t idx =
                        static_cast<size_t>(I.host_value_row[n2]) * nc + c;
                    double d = std::fabs(static_cast<double>(hd[idx]) - hr[idx]);
                    sum_ref += std::fabs(static_cast<double>(hd[idx]));
                    if (d > max_abs) { max_abs = d; worst_node = n2; worst_c = c; }
                }
            }
            std::fprintf(stderr,
                "[RB-SELFCHECK] trav=%d nc=%u runouts=%u maxB=%u terminals=%u "
                "max_abs_diff=%.6g (node=%u combo=%u dense=%.6g rb=%.6g) "
                "sum|dense|=%.6g\n",
                trav, static_cast<unsigned>(nc),
                static_cast<unsigned>(I.matchup.num_runouts),
                static_cast<unsigned>(I.matchup.rb_max_bucket_count),
                static_cast<unsigned>(count), max_abs,
                static_cast<unsigned>(worst_node),
                static_cast<unsigned>(worst_c),
                static_cast<double>(
                    hd[static_cast<size_t>(I.host_value_row[worst_node]) * nc + worst_c]),
                static_cast<double>(
                    hr[static_cast<size_t>(I.host_value_row[worst_node]) * nc + worst_c]),
                sum_ref);
        }
        cudaFree(d_dense);
        cudaFree(d_rb);
    }

    auto dump_traverser_hashes = [&](const char* tag) {
        // Both traverser regions — the buffer is 2× value_rows since fusion.
        const size_t nvspan =
            memory_budget::kGpuValueRegions * I.state.value_span;
        std::fprintf(stderr,
                     "[ih] it=%d %s nv=%016llx reg=%016llx "
                     "ss=%016llx\n",
                     iteration, tag,
                     hash_dev(I.state.node_values, nvspan),
                     hash_dev(I.state.regrets, I.state.state_stride),
                     hash_dev(I.state.strategy_sum, I.state.state_stride));
    };

    // Prologue per traverser (terminal values into its own region, then the
    // strategy_sum update that must read iteration-start regrets), then ONE
    // fused backward pass. Both strategy_sum launches stay ahead of every
    // regret write, which is the same ordering the serialized version had:
    // traverser 1's update only ever reads its own player's slots, so
    // traverser 0's regret writes never reached it.
    run_traverser(0);  // OOP prologue
    run_traverser(1);  // IP prologue
    run_backward_fused();
    if (kIterHash) dump_traverser_hashes("fused");

    // Wait for all kernels to complete before next iteration
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// finalize: strategy_sum → normalized strategy (on device, then download)
// ============================================================================

namespace {

// B1a inc 3: normalize_strategy_kernel is gone — finalize() normalizes the
// downloaded strategy_sum on the host (it needs the values host-side anyway)
// and the postsolve passes derive the same average on the fly via
// StrategyRow<STRAT_SRC_SUM>. Neither needs a device buffer to write into.

} // anonymous namespace

void GpuBackend::finalize() {
    if (!impl_->prepared) {
        throw std::runtime_error("GpuBackend::finalize called before prepare()");
    }
    auto& I = *impl_;
    uint16_t nc = I.iso->num_canonical;
    uint32_t N  = I.tree.num_nodes;

    // B1a inc 3: download strategy_sum and normalize on the HOST. The device
    // buffer this used to normalize into is gone, and allocating a scratch one
    // here would put the peak straight back (finalize can be called mid-loop
    // for the exploitability probe, so regrets and strategy_sum must both stay
    // live). The arithmetic below is normalize_strategy_kernel line for line —
    // same summation order over actions, same 1e-7 threshold, same
    // multiply-by-reciprocal — so the result is bit-identical.
    std::vector<float> host_strat(I.state.state_stride, 0.0f);
    if (!host_strat.empty()) {
        CUDA_CHECK(cudaMemcpy(host_strat.data(),
                               I.state.strategy_sum,
                               host_strat.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (uint32_t n = 0; n < N; ++n) {
        auto nt = static_cast<NodeType>(I.host_tree->node_types[n]);
        const uint8_t na = I.host_tree->num_children[n];
        if ((nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) || na == 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(I.state.host_node_offset[n]) * nc;
        const size_t stride = nc;
        for (uint16_t combo = 0; combo < nc; ++combo) {
            const size_t b = base + combo;
            float total_sum = 0.0f;
            for (int a = 0; a < na; ++a) total_sum += host_strat[b + a * stride];
            if (total_sum > 1e-7f) {
                const float inv = 1.0f / total_sum;
                for (int a = 0; a < na; ++a) {
                    host_strat[b + a * stride] = host_strat[b + a * stride] * inv;
                }
            } else {
                const float u = 1.0f / static_cast<float>(na);
                for (int a = 0; a < na; ++a) host_strat[b + a * stride] = u;
            }
        }
    }

    // Peak-VRAM honesty (P2 review): finalize allocates nothing device-side
    // today, but sample anyway so the high-water mark provably covers the
    // whole finalize path, not just prepare.
    impl_->sample_vram();

    // Build strategy_ in same per-node format as CpuBackend. A player node's
    // na action rows are contiguous at slot host_node_offset[n], already in
    // [a*nc+c] order — one straight copy per node.
    strategy_.assign(N, {});
    for (uint32_t n = 0; n < N; ++n) {
        auto nt = static_cast<NodeType>(I.host_tree->node_types[n]);
        uint8_t na = I.host_tree->num_children[n];
        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            size_t src = static_cast<size_t>(I.state.host_node_offset[n]) * nc;
            strategy_[n].assign(host_strat.begin() + src,
                                host_strat.begin() + src
                                    + static_cast<size_t>(na) * nc);
        } else {
            strategy_[n].assign(static_cast<size_t>(na) * nc, 0.0f);
        }
    }

    // current_strategy on device now holds the averaged strategy. Mark
    // postsolve-ready so compute_combo_evs_gpu / compute_best_response_gpu
    // can reuse it without re-uploading.
    impl_->finalized = true;
}

// ============================================================================
// GPU postsolve: per-combo EV and best response at root.
//
// Reuses the device buffers populated by iterate()+finalize(): tree, matchup
// tables, root reach, averaged strategy (in current_strategy). One forward
// reach pass + one bottom-up value pass per call. No host↔device copies
// except the final root-vector download.
// ============================================================================

std::vector<float> GpuBackend::Impl::run_postsolve_pass(int traverser, bool best_response) {
    if (!prepared || !finalized) return {};
    if (traverser != 0 && traverser != 1) return {};

    // Serialize against any other concurrent postsolve pass — the
    // reach/value/action scratch buffers are shared across passes.
    std::lock_guard<std::mutex> lock(postsolve_mutex);

    using namespace deepsolver::gpu;
    uint16_t nc = iso->num_canonical;

    // 1) Reset per-node reach scratch by seeding the root from the upload.
    //    All subsequent levels are written by propagate_reach_forward_kernel.
    CUDA_CHECK(cudaMemcpy(state.reach_scratch_oop,
                          reach.oop_reach,
                          static_cast<size_t>(nc) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.reach_scratch_ip,
                          reach.ip_reach,
                          static_cast<size_t>(nc) * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // B1a inc 3: the averaged strategy is normalized out of strategy_sum on
    // the fly — same arithmetic normalize_strategy_kernel used to write into
    // current_strategy, minus the buffer. Node locks make no difference here:
    // finalize never re-applied them either (they are already baked into
    // strategy_sum by the iterations that ran under them).
    const int    ps_src_mode = kStratSrcSum;
    const float* ps_src      = state.strategy_sum;

    // 2) Forward reach propagation, root → leaves, using averaged strategy.
    //    The kernel only multiplies the acting player's reach by their
    //    strategy, so opponent-of-traverser reach absorbs the population
    //    strategy correctly for both EV and BR variants.
    const auto& offsets = host_level_offsets;
    for (uint32_t L = 0; L < levels.max_depth; ++L) {
        uint32_t start = offsets[L];
        uint32_t end   = offsets[L + 1];
        uint32_t count = end - start;
        if (count == 0) continue;
        const uint32_t* d_level = levels.node_order + start;
        launch_propagate_reach(
            tree.node_types, tree.active_player, tree.num_children,
            tree.children_offset, tree.children,
            state.node_offset,
            d_level, count,
            ps_src, ps_src_mode,
            state.reach_scratch_oop, state.reach_scratch_ip, nc);
    }

    // 3) Terminals in ONE launch (their value depends only on reach, which the
    //    forward pass above has already finished), then the backward value
    //    pass deepest level → root, where aggregate gathers children's
    //    node_values directly (sum-mode for EV, max-at-traverser for BR).
    float* reach_opp_base = (traverser == 0) ? state.reach_scratch_ip
                                              : state.reach_scratch_oop;
    const uint16_t* opp_live = (traverser == 0) ? reach.live_ip : reach.live_oop;
    const uint16_t  num_opp_live =
        (traverser == 0) ? reach.num_live_ip : reach.num_live_oop;
    if (levels.num_terminals > 0) {
        if (matchup.rb_valid) {
            launch_rank_blocker_terminal_level(
                tree.node_types, tree.terminal_types, tree.pots,
                tree.parent_indices, tree.bet_into, tree.matchup_idx,
                state.value_row,
                levels.terminal_order, levels.num_terminals,
                matchup.rb_combo_bucket, matchup.rb_bucket_count,
                matchup.rb_combo_card0, matchup.rb_combo_card1,
                matchup.rb_card_off, matchup.rb_card_list,
                matchup.num_runouts, reach_opp_base, nc,
                matchup.rb_max_bucket_count, traverser,
                config->rake_rate, config->rake_cap,
                state.node_values);
        } else {
            launch_terminal_level(
                tree.node_types,
                tree.terminal_types,
                tree.pots,
                tree.parent_indices,
                tree.bet_into,
                tree.matchup_idx,
                state.value_row,
                levels.terminal_order, levels.num_terminals,
                matchup.matchup_ev,
                matchup.matchup_valid,
                matchup.canonical_weights,
                matchup.num_runouts,
                reach_opp_base,
                opp_live, num_opp_live,
                nc, traverser,
                config->rake_rate, config->rake_cap,
                state.node_values);
        }
    }

    const uint32_t num_levels = levels.num_levels;
    for (int L = static_cast<int>(num_levels) - 1; L >= 0; --L) {
        uint32_t start = offsets[L];
        uint32_t end   = offsets[L + 1];
        uint32_t count = end - start;
        if (count == 0) continue;
        const uint32_t* d_level = levels.node_order + start;

        if (best_response) {
            launch_aggregate_node_values_br(
                tree.node_types, tree.active_player, tree.num_children,
                tree.children_offset, tree.children,
                tree.runout_weight,
                state.node_offset, state.value_row,
                d_level, count,
                ps_src, ps_src_mode,
                state.node_values,
                nc, traverser);
        } else {
            launch_aggregate_node_values(
                tree.node_types, tree.active_player, tree.num_children,
                tree.children_offset, tree.children,
                tree.runout_weight,
                state.node_offset, state.value_row,
                d_level, count,
                ps_src, ps_src_mode,
                state.node_values,
                nc, traverser,
                /*regrets=*/nullptr, 0.0f, 0.0f);
        }
    }

    // 4) Download root node_values → host. B3 inc 1: the root is node 0 but
    //    its row is wherever value_row puts it (depth 0, non-terminal, so the
    //    first slot of the even-parity window), NOT the start of the buffer.
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> root_values(nc, 0.0f);
    const size_t root_row = host_value_row.empty()
        ? 0u : static_cast<size_t>(host_value_row[0]);
    CUDA_CHECK(cudaMemcpy(root_values.data(),
                          state.node_values + root_row * nc,
                          static_cast<size_t>(nc) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Peak-VRAM honesty (P2 review): cover the postsolve path in the
    // high-water mark, not just prepare.
    sample_vram();
    return root_values;
}

std::vector<float> GpuBackend::compute_combo_evs_gpu() {
    if (!impl_->prepared || !impl_->finalized) return {};
    return impl_->run_postsolve_pass(/*traverser=*/0, /*best_response=*/false);
}

std::vector<float> GpuBackend::compute_best_response_gpu(int player) {
    if (!impl_->prepared || !impl_->finalized) return {};
    if (player != 0 && player != 1) return {};
    return impl_->run_postsolve_pass(player, /*best_response=*/true);
}

} // namespace deepsolver
