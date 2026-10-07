/**
 * @file cfr_kernel.cu
 * @brief DCFR kernels in FP32.
 *
 * Contains (all pointwise per node, per combo):
 *   - compute_strategy_kernel          — regret matching
 *   - propagate_reach_forward_kernel   — per-level reach propagation (Phase 4.4)
 *   - aggregate_node_values_kernel     — backward values, fused with the
 *                                        regret and strategy_sum updates
 *
 * Complex backward-pass scheduling (Phase 4.5) lives in the host orchestrator
 * (gpu_backend.cu), which uses these plus the terminal kernels from eval_kernel.cu.
 *
 * Memory layout conventions (matches gpu_backend.cu):
 *   strategy / regrets → COMPACT [slot][combo], stride = nc.
 *     slot = node_offset[node] + action. node_offset is a per-node prefix sum
 *     of num_children over PLAYER nodes only (B1a); chance/terminal nodes own
 *     NO slots — their table entry is a 0 sentinel that must never be
 *     dereferenced, so every state-touching kernel returns on non-player
 *     nodes BEFORE computing its base.
 *   reach               → [N][combo]           stride = nc (full tree)
 *   node_values         → [N][combo]           stride = nc (full tree)
 *
 * B1a increment 2: there is NO action_values buffer. A parent's per-action
 * value IS its child's node_values row — node_values is written exactly once
 * per node per backward pass (at the node's own level) and never overwritten
 * within the pass, so both aggregate (same level, children already done) and
 * update_regrets (after the whole pass) gather node_values[child] directly.
 * The old lift kernel materialized exactly those reads into a 4th
 * strat-shaped buffer; fusing removed the buffer and the extra global-memory
 * round trip without changing any float op or its order.
 */

#include "util.cuh"
#include "types.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace deepsolver {
namespace gpu {

// Mirror of host NodeType values — keep in sync with types.h
constexpr uint8_t NT_PLAYER_OOP = 0;
constexpr uint8_t NT_PLAYER_IP  = 1;
constexpr uint8_t NT_CHANCE     = 2;
constexpr uint8_t NT_TERMINAL   = 3;

// Mirror of types.h::chance_runout_denominator — keep in sync. The chance
// children's weights sum to the undealt-card count given the BOARD; every
// legal matchup removes 4 more cards (both hole hands), so the conditional
// probability of a runout is weight / (total − 4). The collapsed single-child
// fallback (total 1) keeps 1/1; a real deal always totals ≥ 44.
constexpr uint32_t kChanceHoleCardsExcluded = 4;
__device__ __forceinline__ uint32_t chance_runout_denominator(uint32_t total_w) {
    if (total_w > kChanceHoleCardsExcluded) return total_w - kChanceHoleCardsExcluded;
    return total_w == 0 ? 1u : total_w;
}

// ============================================================================
// B1a increment 3: strategy sources
//
// The regret-matched strategy used to live in its own [total_slots * nc]
// buffer — a third strat-shaped array on top of regrets and strategy_sum,
// i.e. a third of the state footprint for a value that is a pure function of
// regrets. Consumers now derive it on the fly instead. All three sources have
// IDENTICAL layout, so a consumer takes one pointer plus a compile-time mode:
//
//   MATERIALIZED — read the buffer verbatim (node-locked solves, where the
//                  lock override is written into it before the readers run)
//   REGRETS      — regret matching, byte-for-byte compute_strategy_kernel
//   SUM          — normalized average, byte-for-byte normalize_strategy_kernel
//                  (the postsolve passes' "averaged strategy")
//
// Cost is neutral: a consumer reads na values from the source either way; it
// just also sums them first. What disappears is a full-buffer write + read per
// iteration and, more importantly, the allocation.
// ============================================================================

enum : int {
    STRAT_SRC_MATERIALIZED = 0,
    STRAT_SRC_REGRETS      = 1,
    STRAT_SRC_SUM          = 2,
};

template <int SRC>
struct StrategyRow {
    const float* __restrict__ p;
    size_t base;
    size_t stride;
    float  inv       = 0.0f;
    float  uniform   = 0.0f;
    bool   degenerate = false;

    __device__ StrategyRow(const float* __restrict__ p_, size_t base_,
                           size_t stride_, int na)
        : p(p_), base(base_), stride(stride_)
    {
        if (SRC == STRAT_SRC_MATERIALIZED) return;
        float sum = 0.0f;
        for (int a = 0; a < na; ++a) {
            const float v = p[base + a * stride];
            if (SRC == STRAT_SRC_REGRETS) {
                if (v > 0.0f) sum += v;
            } else {
                sum += v;
            }
        }
        // Below the threshold the row is uniform, mirroring the host finalize.
        // 2026-10-06 audit: one threshold, FLT_MIN, for both sources — the
        // old 1e-7 forced uniform strategies wherever a reach-weighted
        // strategy_sum was small (every schedule but the default).
        if (sum >= kGpuMinNormalSum) {
            inv = 1.0f / sum;
        } else {
            degenerate = true;
            uniform = 1.0f / static_cast<float>(na);
        }
    }

    __device__ float operator()(int a) const {
        if (SRC == STRAT_SRC_MATERIALIZED) return p[base + a * stride];
        if (degenerate) return uniform;
        const float v = p[base + a * stride];
        if (SRC == STRAT_SRC_REGRETS) return (v > 0.0f) ? v * inv : 0.0f;
        return v * inv;
    }
};

// ============================================================================
// Kernel 1: Regret Matching — current_strategy ← positive regrets normalized
//
// Only launched for node-locked solves now (B1a inc 3): the lock override is
// a write into the materialized buffer, so those solves keep one.
// ============================================================================

__global__ void compute_strategy_kernel(
    const float* __restrict__ regrets,         // [total_slots * nc] compact
    float*       __restrict__ current_strategy,// [total_slots * nc] compact
    const uint8_t*  __restrict__ num_children,
    const uint8_t*  __restrict__ node_types,
    const uint32_t* __restrict__ node_offset,  // [N] per-node slot index
    uint32_t num_nodes,
    uint16_t num_canonical)
{
    // 64-bit launch index: N × nc reaches 1.6e9 (75% of INT32_MAX) on the
    // 4.65M-node roadmap target — int math would go negative right when the
    // memory levers finally let those trees enumerate.
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_nodes) * num_canonical;
    if (tid >= total) return;

    const uint32_t node  = static_cast<uint32_t>(tid / num_canonical);
    const uint32_t combo = static_cast<uint32_t>(tid % num_canonical);

    uint8_t nt = node_types[node];
    if (nt != NT_PLAYER_OOP && nt != NT_PLAYER_IP) return;

    uint8_t na = num_children[node];
    if (na == 0) return;

    size_t base = static_cast<size_t>(node_offset[node]) * num_canonical
                + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    // Sum positive regrets
    float pos_sum = 0.0f;
    for (int a = 0; a < na; ++a) {
        float r = regrets[base + a * stride];
        if (r > 0.0f) pos_sum += r;
    }

    if (pos_sum >= kGpuMinNormalSum) {
        float inv = 1.0f / pos_sum;
        for (int a = 0; a < na; ++a) {
            float r = regrets[base + a * stride];
            current_strategy[base + a * stride] = (r > 0.0f) ? r * inv : 0.0f;
        }
    } else {
        float uniform = 1.0f / static_cast<float>(na);
        for (int a = 0; a < na; ++a) {
            current_strategy[base + a * stride] = uniform;
        }
    }
}

// ============================================================================
// Kernel 2: Reach Forward Propagation — one level of the forward pass
// ============================================================================

/**
 * For each (node, combo) at `level_node_indices`, propagate reach to children.
 *
 * At player-acting nodes: acting player's reach *= strategy, opponent unchanged.
 * At chance nodes (flattened to single child in this engine): pass through.
 * At terminals: no-op (no children).
 *
 * The kernel writes children's reach[child][combo]. One parent writes each
 * child exactly once (tree structure). No atomics needed.
 */
template <int SRC>
__global__ void propagate_reach_forward_kernel(
    const uint8_t*  __restrict__ node_types,
    const uint8_t*  __restrict__ active_player,
    const uint8_t*  __restrict__ num_children,
    const uint32_t* __restrict__ children_offset,
    const uint32_t* __restrict__ children,
    const uint32_t* __restrict__ node_offset,   // [N] per-node slot index
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ strat_src,        // compact [slot * nc]; see StrategyRow
    float* __restrict__ reach_oop,  // [N * nc]   read parent + write children
    float* __restrict__ reach_ip,   // [N * nc]
    uint16_t num_canonical,
    int players)                    // bit 0 OOP, bit 1 IP: whose reach to write
{
    // 64-bit launch index — see compute_strategy_kernel.
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    const uint32_t local_idx = static_cast<uint32_t>(tid / num_canonical);
    const uint32_t combo     = static_cast<uint32_t>(tid % num_canonical);
    uint32_t n    = level_node_indices[local_idx];

    uint8_t nt = node_types[n];
    if (nt == NT_TERMINAL) return;

    uint8_t na = num_children[n];
    if (na == 0) return;

    // 2026-10-07: a player's reach depends on that player's strategy alone,
    // so under alternating updates only the side whose regrets just changed
    // is propagated; the other side's rows are already current.
    const bool do_oop = (players & 1) != 0;
    const bool do_ip  = (players & 2) != 0;
    size_t node_reach_idx = static_cast<size_t>(n) * num_canonical + combo;
    const float r_oop = do_oop ? reach_oop[node_reach_idx] : 0.0f;
    const float r_ip  = do_ip  ? reach_ip [node_reach_idx] : 0.0f;
    uint32_t offset = children_offset[n];

    if (nt == NT_CHANCE) {
        // Phase 2: chance nodes can have multiple children (one per canonical
        // runout). Reach is unchanged by chance dealing — pass parent's reach
        // through to every child. The runout_weight is consumed in the
        // aggregate kernel, NOT here (we want each child to see the full
        // parent reach so its CFR computes counterfactual values correctly).
        for (int k = 0; k < na; ++k) {
            uint32_t child = children[offset + k];
            // Reach is still full-tree [N][nc] — B3 inc 1 only re-mapped the
            // VALUE buffer. Do not route this through value_row.
            size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
            if (do_oop) reach_oop[child_idx] = r_oop;
            if (do_ip)  reach_ip [child_idx] = r_ip;
        }
        return;
    }

    // Player decision: scale acting player's reach by their strategy per action
    uint8_t acting = active_player[n];
    const bool oop_acts = (acting == NT_PLAYER_OOP);
    if (oop_acts ? !do_oop : !do_ip) {
        // Only the waiting player's reach is written: it passes through, and
        // the actor's strategy is never read.
        for (int a = 0; a < na; ++a) {
            uint32_t child = children[offset + a];
            size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
            if (oop_acts) reach_ip[child_idx] = r_ip;
            else          reach_oop[child_idx] = r_oop;
        }
        return;
    }
    size_t strat_base = static_cast<size_t>(node_offset[n]) * num_canonical
                      + static_cast<size_t>(combo);
    size_t strat_stride = num_canonical;
    StrategyRow<SRC> strat(strat_src, strat_base, strat_stride, na);

    for (int a = 0; a < na; ++a) {
        uint32_t child = children[offset + a];
        size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
        float s = strat(a);
        if (oop_acts) {
            reach_oop[child_idx] = r_oop * s;
            if (do_ip) reach_ip[child_idx] = r_ip;
        } else {
            if (do_oop) reach_oop[child_idx] = r_oop;
            reach_ip [child_idx] = r_ip * s;
        }
    }
}

// ============================================================================
// Kernel 3: Node-Value Aggregation — node_value = Σ strat * action_value
// ============================================================================

/**
 * At each player decision node, compute (action value a = the a-th child's
 * node_values row, gathered directly — B1a inc 2, no action_values buffer):
 *   - If acting == traverser:
 *       node_val[c] = Σ_a strat[a][c] * node_val[child_a][c]
 *   - Else (opponent acting):
 *       node_val[c] = Σ_a node_val[child_a][c]   (opp strat already in reach)
 *
 * For chance nodes: copy first child's node_value.
 * For terminals: node_value is set separately by terminal kernels (eval_kernel.cu).
 */
/// `FuseRegrets` (B3 inc 1): also perform this node's DCFR regret update,
/// using the per-action child values this kernel has already gathered into
/// registers. update_regrets used to be one sweep over all N nodes after the
/// backward pass; the windowed value buffer forced it per level, and a
/// separate per-level launch cost 23% of small-tree throughput in pure launch
/// overhead. Fusing removes the launch AND the second read of every child row.
/// Safe because thread (node, combo) reads and writes only its own node's
/// regret slots at its own combo — no other thread in the launch touches them.
/// Off for the postsolve variants, which have no regret update.
template <bool BestResponse, bool FuseRegrets, int SRC>
__global__ void aggregate_node_values_kernel(
    const uint8_t*  __restrict__ node_types,
    const uint8_t*  __restrict__ active_player,
    const uint8_t*  __restrict__ num_children,
    const uint32_t* __restrict__ children_offset,
    const uint32_t* __restrict__ children,
    const uint8_t*  __restrict__ runout_weight,  // Phase 2: per-child orbit size
    const uint32_t* __restrict__ node_offset,    // [N] per-node slot index
    const uint32_t* __restrict__ value_row,      // [N] node → value buffer row
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ strat_src,         // compact [slot * nc]; see StrategyRow
    float* __restrict__ node_values,             // [value_rows * nc]; see value_row
    uint16_t num_canonical,
    int traverser,
    float* __restrict__ regrets,                 // FuseRegrets only
    float pos_disc,
    float neg_disc,
    // Traverser fusion: gridDim.z carries the traverser dimension, so ONE
    // launch covers both backward passes instead of two. blockIdx.z picks the
    // traverser and its own value region. With gridDim.z == 1 (postsolve, and
    // any single-traverser caller) both reduce to what they were.
    size_t value_span,
    DeviceRunoutMaps rm,
    // FuseRegrets only (2026-10-07): the strategy_sum update, fused too.
    float* __restrict__ strategy_sum,
    const float* __restrict__ reach_oop,
    const float* __restrict__ reach_ip,
    float strat_weight,    // STANDARD: ((t+1)/(t+2))^gamma; POSTFLOP: (t'/(t'+1))^3
    int ss_mode)           // dcfr_strategy_sum_mode: 0 accumulate, 1 decay-add, 2 decay-add with reach
{
    const int trav = traverser + static_cast<int>(blockIdx.z);
    node_values += static_cast<size_t>(blockIdx.z) * value_span;

    // 64-bit launch index — see compute_strategy_kernel.
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    const uint32_t local_idx = static_cast<uint32_t>(tid / num_canonical);
    const uint32_t combo     = static_cast<uint32_t>(tid % num_canonical);
    uint32_t n    = level_node_indices[local_idx];

    uint8_t nt = node_types[n];
    if (nt == NT_TERMINAL) return;  // set by terminal kernel

    uint8_t na = num_children[n];
    // B3 inc 1: node_values is no longer indexed by node — terminals live in a
    // compacted region and non-terminals in a 2-level rolling window.
    size_t node_idx = static_cast<size_t>(value_row[n]) * num_canonical + combo;
    uint32_t offset = children_offset[n];

    if (nt == NT_CHANCE) {
        // Phase 2: weighted average over all chance children using the
        // per-child orbit size. Sum of weights equals the undealt-card count;
        // dividing by it keeps the chance node's EV in the same units as the
        // children so backprop is unbiased.
        //
        // weight==0 should NEVER happen — TreeNode default is 1, and
        // GameTreeBuilder always sets either 1 (legacy) or rep.weight (iso).
        // If it slips through, treat as 1 rather than silently re-normalizing
        // (which would bias the average toward the surviving children).
        if (na == 0) {
            node_values[node_idx] = 0.0f;
            return;
        }
        float acc = 0.0f;
        uint32_t total_w = 0;
        for (int k = 0; k < na; ++k) {
            uint32_t child = children[offset + k];
            uint32_t w = runout_weight[child];
            if (w == 0) w = 1;  // guard: treat 0 as 1, not as "skip"
            const size_t child_row =
                static_cast<size_t>(value_row[child]) * num_canonical;
            const uint16_t set = (rm.node_set != nullptr) ? rm.node_set[child] : 0xFFFFu;
            if (set == 0xFFFFu) {
                acc += static_cast<float>(w) * node_values[child_row + combo];
            } else {
                // Exact isomorphism: the other orbit members are this
                // child's world relabelled — gather each one's hand.
                acc += node_values[child_row + combo];
                const size_t first = rm.set_first[set];
                for (uint8_t j = 0; j < rm.set_count[set]; ++j) {
                    acc += node_values[child_row +
                        rm.maps[(first + j) * num_canonical + combo]];
                }
            }
            total_w += w;
        }
        node_values[node_idx] = (total_w > 0)
            ? (acc / static_cast<float>(chance_runout_denominator(total_w))) : 0.0f;
        return;
    }

    if (na == 0) {
        node_values[node_idx] = 0.0f;
        return;
    }

    // Player decision node. Per-action value = the a-th child's node_values
    // row: children are at deeper levels, already aggregated this pass, and
    // node_values is written once per node per pass — so the gather reads
    // exactly what the old lift kernel used to copy into action_values.
    int acting = active_player[n];
    size_t strat_base = static_cast<size_t>(node_offset[n]) * num_canonical
                      + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    auto child_value = [&](int a) -> float {
        uint32_t child = children[offset + a];
        return node_values[static_cast<size_t>(value_row[child]) * num_canonical
                           + combo];
    };

    if (acting == trav) {
        // Traverser-acting branch differs by mode:
        //   BR (postsolve):  per-combo MAX over actions — traverser plays the
        //                    pointwise best response, ignoring averaged strategy.
        //   EV / CFR:        weighted SUM by current_strategy (= averaged strategy
        //                    after finalize, or regret-matched strategy mid-CFR).
        if constexpr (BestResponse) {
            float best = child_value(0);
            for (int a = 1; a < na; ++a) {
                best = fmaxf(best, child_value(a));
            }
            node_values[node_idx] = best;
        } else {
            StrategyRow<SRC> strat(strat_src, strat_base, stride, na);
            float sv[deepsolver::MAX_ACTIONS];
            float sum = 0.0f;
            for (int a = 0; a < na; ++a) {
                float s  = strat(a);
                sv[a] = s;
                sum += s * child_value(a);
            }
            node_values[node_idx] = sum;
            if constexpr (FuseRegrets) {
                // 2026-10-07: the strategy_sum update, fused. It was its own
                // sweep over every node x lane (only this traverser's nodes
                // did work) re-deriving the same strategy. Identical
                // arithmetic to the old update_strategy_sum_kernel, and still
                // ahead of this node's regret update below, so it reads the
                // iteration-start strategy as before.
                if (ss_mode == 1) {
                    // POSTFLOP: strategy_sum = strategy_sum * gamma_t + current_strategy
                    for (int a = 0; a < na; ++a) {
                        const float old = strategy_sum[strat_base + a * stride];
                        strategy_sum[strat_base + a * stride] = old * strat_weight + sv[a];
                    }
                } else {
                    const float* reach_own = (trav == 0) ? reach_oop : reach_ip;
                    const float reach =
                        reach_own[static_cast<size_t>(n) * num_canonical + combo];
                    if (ss_mode == 2) {
                        // DCFR / LINEAR: strategy_sum = strategy_sum * w + reach * s
                        for (int a = 0; a < na; ++a) {
                            const float old = strategy_sum[strat_base + a * stride];
                            strategy_sum[strat_base + a * stride] =
                                old * strat_weight + reach * sv[a];
                        }
                    } else {
                        // STANDARD: strategy_sum += weight * reach * s
                        for (int a = 0; a < na; ++a) {
                            strategy_sum[strat_base + a * stride] +=
                                strat_weight * reach * sv[a];
                        }
                    }
                }
                // Identical arithmetic to the old update_regrets_kernel:
                // instant = action value - node value, applied to the
                // separately-discounted existing regret.
                for (int a = 0; a < na; ++a) {
                    const float instant = child_value(a) - sum;
                    float r = regrets[strat_base + a * stride];
                    r *= (r > 0.0f) ? pos_disc : neg_disc;
                    r += instant;
                    regrets[strat_base + a * stride] = r;
                }
            }
        }
    } else {
        // Opponent's strategy absorbed into reach; just SUM the child values
        float sum = 0.0f;
        for (int a = 0; a < na; ++a) {
            sum += child_value(a);
        }
        node_values[node_idx] = sum;
    }
}

// Both <false> (CFR/EV) and <true> (BR/postsolve) instantiations are
// implicitly produced by the launcher calls in this same TU, so no explicit
// instantiation is needed.

// ============================================================================
// Kernel 5: Regret Update — accumulate instantaneous regret + DCFR discount
// ============================================================================

// ============================================================================
// Host-side launch helpers (called from gpu_backend.cu)
// ============================================================================

static constexpr int DEFAULT_BLOCK_SIZE = 256;

void launch_compute_strategy(
    const float* d_regrets, float* d_current_strategy,
    const uint8_t* d_num_children, const uint8_t* d_node_types,
    const uint32_t* d_node_offset,
    uint32_t num_nodes, uint16_t nc)
{
    const size_t total = static_cast<size_t>(num_nodes) * nc;
    const int grid = static_cast<int>(
        (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
    compute_strategy_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_regrets, d_current_strategy,
        d_num_children, d_node_types, d_node_offset,
        num_nodes, nc);
    CUDA_CHECK(cudaGetLastError());
}

// B1a inc 3: `strat_src_mode` selects how the strategy is derived from
// `d_strat_src` (see StrategyRow). One switch per launch — the mode is
// grid-uniform, so nothing branches inside the kernel.
#define DEEPSOLVER_DISPATCH_STRAT_SRC(mode, KERNEL_CALL)                     \
    do {                                                                     \
        switch (mode) {                                                      \
            case STRAT_SRC_REGRETS: KERNEL_CALL(STRAT_SRC_REGRETS); break;   \
            case STRAT_SRC_SUM:     KERNEL_CALL(STRAT_SRC_SUM);     break;   \
            default:                KERNEL_CALL(STRAT_SRC_MATERIALIZED);     \
        }                                                                    \
        CUDA_CHECK(cudaGetLastError());                                      \
    } while (0)

void launch_propagate_reach(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_node_offset,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_strat_src, int strat_src_mode,
    float* d_reach_oop, float* d_reach_ip, uint16_t nc, int players)
{
    const size_t total = static_cast<size_t>(num_level_nodes) * nc;
    const int grid = static_cast<int>(
        (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
#define DEEPSOLVER_LAUNCH_PROPAGATE(SRC)                                     \
    propagate_reach_forward_kernel<SRC><<<grid, DEFAULT_BLOCK_SIZE>>>(       \
        d_node_types, d_active_player, d_num_children,                       \
        d_children_offset, d_children, d_node_offset,                        \
        d_level_indices, num_level_nodes,                                    \
        d_strat_src,                                                         \
        d_reach_oop, d_reach_ip, nc, players)
    DEEPSOLVER_DISPATCH_STRAT_SRC(strat_src_mode, DEEPSOLVER_LAUNCH_PROPAGATE);
#undef DEEPSOLVER_LAUNCH_PROPAGATE
}

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
    int num_traversers, size_t value_span, DeviceRunoutMaps rm,
    float* d_strategy_sum, const float* d_reach_oop, const float* d_reach_ip,
    float strat_weight, int ss_mode)
{
    const size_t total = static_cast<size_t>(num_level_nodes) * nc;
    const dim3 grid(static_cast<unsigned>(
                        (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
                    1u, static_cast<unsigned>(num_traversers));
    if (d_regrets != nullptr) {
#define DEEPSOLVER_LAUNCH_AGG_CFR(SRC)                                       \
    aggregate_node_values_kernel<false, true, SRC>                           \
        <<<grid, DEFAULT_BLOCK_SIZE>>>(                                      \
        d_node_types, d_active_player, d_num_children,                       \
        d_children_offset, d_children, d_runout_weight, d_node_offset,       \
        d_value_row, d_level_indices, num_level_nodes,                       \
        d_strat_src,                                                         \
        d_node_values, nc, traverser, d_regrets, pos_disc, neg_disc,         \
        value_span, rm,                                                      \
        d_strategy_sum, d_reach_oop, d_reach_ip, strat_weight, ss_mode)
        DEEPSOLVER_DISPATCH_STRAT_SRC(strat_src_mode, DEEPSOLVER_LAUNCH_AGG_CFR);
#undef DEEPSOLVER_LAUNCH_AGG_CFR
        return;
    }
#define DEEPSOLVER_LAUNCH_AGG_EV(SRC)                                        \
    aggregate_node_values_kernel<false, false, SRC>                          \
        <<<grid, DEFAULT_BLOCK_SIZE>>>(                                      \
        d_node_types, d_active_player, d_num_children,                       \
        d_children_offset, d_children, d_runout_weight, d_node_offset,       \
        d_value_row, d_level_indices, num_level_nodes,                       \
        d_strat_src,                                                         \
        d_node_values, nc, traverser, nullptr, 0.0f, 0.0f, value_span, rm,  \
        nullptr, nullptr, nullptr, 0.0f, 0)
    DEEPSOLVER_DISPATCH_STRAT_SRC(strat_src_mode, DEEPSOLVER_LAUNCH_AGG_EV);
#undef DEEPSOLVER_LAUNCH_AGG_EV
}

// Postsolve variant: at traverser's own decision nodes, take max over actions
// instead of strategy-weighted sum. Used by best-response / exploitability
// computation. Opponent and chance nodes behave identically to the EV variant.
void launch_aggregate_node_values_br(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_node_offset, const uint32_t* d_value_row,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_strat_src, int strat_src_mode,
    float* d_node_values,
    uint16_t nc, int traverser, DeviceRunoutMaps rm)
{
    const size_t total = static_cast<size_t>(num_level_nodes) * nc;
    const int grid = static_cast<int>(
        (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
#define DEEPSOLVER_LAUNCH_AGG_BR(SRC)                                        \
    aggregate_node_values_kernel<true, false, SRC>                           \
        <<<grid, DEFAULT_BLOCK_SIZE>>>(                                      \
        d_node_types, d_active_player, d_num_children,                       \
        d_children_offset, d_children, d_runout_weight, d_node_offset,       \
        d_value_row, d_level_indices, num_level_nodes,                       \
        d_strat_src,                                                         \
        d_node_values, nc, traverser, nullptr, 0.0f, 0.0f, 0u, rm,          \
        nullptr, nullptr, nullptr, 0.0f, 0)
    DEEPSOLVER_DISPATCH_STRAT_SRC(strat_src_mode, DEEPSOLVER_LAUNCH_AGG_BR);
#undef DEEPSOLVER_LAUNCH_AGG_BR
}

} // namespace gpu
} // namespace deepsolver
