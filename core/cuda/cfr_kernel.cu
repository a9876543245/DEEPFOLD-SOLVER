/**
 * @file cfr_kernel.cu
 * @brief DCFR kernels in FP32.
 *
 * Contains (all pointwise per node, per combo):
 *   - compute_strategy_kernel          — regret matching
 *   - propagate_reach_forward_kernel   — per-level reach propagation (Phase 4.4)
 *   - update_regrets_kernel            — CFR regret accumulation + DCFR discount
 *   - update_strategy_sum_kernel       — strategy_sum with per-node reach weight
 *
 * Complex backward-pass scheduling (Phase 4.5) lives in the host orchestrator
 * (gpu_backend.cu), which uses these plus the terminal kernels from eval_kernel.cu.
 *
 * Memory layout conventions (matches gpu_backend.cu):
 *   strategy / regrets  → [N][action][combo]   stride = MAX_ACTIONS * nc
 *   reach               → [N][combo]           stride = nc
 *   node_values         → [N][combo]           stride = nc
 *   action_values       → [N][action][combo]   stride = MAX_ACTIONS * nc
 */

#include "util.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace deepsolver {
namespace gpu {

// Mirror of host NodeType values — keep in sync with types.h
constexpr uint8_t NT_PLAYER_OOP = 0;
constexpr uint8_t NT_PLAYER_IP  = 1;
constexpr uint8_t NT_CHANCE     = 2;
constexpr uint8_t NT_TERMINAL   = 3;

// ============================================================================
// Kernel 1: Regret Matching — current_strategy ← positive regrets normalized
// ============================================================================

__global__ void compute_strategy_kernel(
    const float* __restrict__ regrets,         // [N * A * nc]
    float*       __restrict__ current_strategy,// [N * A * nc]
    const uint8_t*  __restrict__ num_children,
    const uint8_t*  __restrict__ node_types,
    uint32_t num_nodes,
    uint16_t num_canonical,
    uint8_t  max_actions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_nodes) * num_canonical;
    if (tid >= total) return;

    int node = tid / num_canonical;
    int combo = tid % num_canonical;

    uint8_t nt = node_types[node];
    if (nt != NT_PLAYER_OOP && nt != NT_PLAYER_IP) return;

    uint8_t na = num_children[node];
    if (na == 0) return;

    size_t base = (static_cast<size_t>(node) * max_actions) * num_canonical
                + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    // Sum positive regrets
    float pos_sum = 0.0f;
    for (int a = 0; a < na; ++a) {
        float r = regrets[base + a * stride];
        if (r > 0.0f) pos_sum += r;
    }

    if (pos_sum > 0.0f) {
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
__global__ void propagate_reach_forward_kernel(
    const uint8_t*  __restrict__ node_types,
    const uint8_t*  __restrict__ active_player,
    const uint8_t*  __restrict__ num_children,
    const uint32_t* __restrict__ children_offset,
    const uint32_t* __restrict__ children,
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ current_strategy,
    uint8_t max_actions,
    float* __restrict__ reach_oop,  // [N * nc]   read parent + write children
    float* __restrict__ reach_ip,   // [N * nc]
    uint16_t num_canonical)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    int local_idx = tid / num_canonical;
    int combo     = tid % num_canonical;
    uint32_t n    = level_node_indices[local_idx];

    uint8_t nt = node_types[n];
    if (nt == NT_TERMINAL) return;

    uint8_t na = num_children[n];
    if (na == 0) return;

    size_t node_reach_idx = static_cast<size_t>(n) * num_canonical + combo;
    float r_oop = reach_oop[node_reach_idx];
    float r_ip  = reach_ip [node_reach_idx];
    uint32_t offset = children_offset[n];

    if (nt == NT_CHANCE) {
        // Phase 2: chance nodes can have multiple children (one per canonical
        // runout). Reach is unchanged by chance dealing — pass parent's reach
        // through to every child. The runout_weight is consumed in the
        // aggregate kernel, NOT here (we want each child to see the full
        // parent reach so its CFR computes counterfactual values correctly).
        for (int k = 0; k < na; ++k) {
            uint32_t child = children[offset + k];
            size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
            reach_oop[child_idx] = r_oop;
            reach_ip [child_idx] = r_ip;
        }
        return;
    }

    // Player decision: scale acting player's reach by their strategy per action
    uint8_t acting = active_player[n];
    size_t strat_base = (static_cast<size_t>(n) * max_actions) * num_canonical
                      + static_cast<size_t>(combo);
    size_t strat_stride = num_canonical;

    for (int a = 0; a < na; ++a) {
        uint32_t child = children[offset + a];
        size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
        float s = current_strategy[strat_base + a * strat_stride];
        if (acting == NT_PLAYER_OOP) {
            reach_oop[child_idx] = r_oop * s;
            reach_ip [child_idx] = r_ip;
        } else {
            reach_oop[child_idx] = r_oop;
            reach_ip [child_idx] = r_ip * s;
        }
    }
}

// ============================================================================
// Kernel 3: Node-Value Aggregation — node_value = Σ strat * action_value
// ============================================================================

/**
 * At each player decision node, compute:
 *   - If acting == traverser:
 *       node_val[c] = Σ_a strat[a][c] * action_val[a][c]
 *   - Else (opponent acting):
 *       node_val[c] = Σ_a action_val[a][c]   (opp strat already in reach)
 *
 * For chance nodes: copy first child's node_value.
 * For terminals: node_value is set separately by terminal kernels (eval_kernel.cu).
 */
template <bool BestResponse>
__global__ void aggregate_node_values_kernel(
    const uint8_t*  __restrict__ node_types,
    const uint8_t*  __restrict__ active_player,
    const uint8_t*  __restrict__ num_children,
    const uint32_t* __restrict__ children_offset,
    const uint32_t* __restrict__ children,
    const uint8_t*  __restrict__ runout_weight,  // Phase 2: per-child orbit size
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ current_strategy,
    uint8_t max_actions,
    const float* __restrict__ action_values,  // [N * A * nc]
    float* __restrict__ node_values,          // [N * nc]
    uint16_t num_canonical,
    int traverser)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    int local_idx = tid / num_canonical;
    int combo     = tid % num_canonical;
    uint32_t n    = level_node_indices[local_idx];

    uint8_t nt = node_types[n];
    if (nt == NT_TERMINAL) return;  // set by terminal kernel

    uint8_t na = num_children[n];
    size_t node_idx = static_cast<size_t>(n) * num_canonical + combo;
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
            size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
            acc += static_cast<float>(w) * node_values[child_idx];
            total_w += w;
        }
        node_values[node_idx] = (total_w > 0)
            ? (acc / static_cast<float>(total_w)) : 0.0f;
        return;
    }

    if (na == 0) {
        node_values[node_idx] = 0.0f;
        return;
    }

    // Player decision node
    int acting = active_player[n];
    size_t av_base = (static_cast<size_t>(n) * max_actions) * num_canonical
                   + static_cast<size_t>(combo);
    size_t strat_base = av_base;
    size_t stride = num_canonical;

    if (acting == traverser) {
        // Traverser-acting branch differs by mode:
        //   BR (postsolve):  per-combo MAX over actions — traverser plays the
        //                    pointwise best response, ignoring averaged strategy.
        //   EV / CFR:        weighted SUM by current_strategy (= averaged strategy
        //                    after finalize, or regret-matched strategy mid-CFR).
        if constexpr (BestResponse) {
            float best = action_values[av_base];
            for (int a = 1; a < na; ++a) {
                float av = action_values[av_base + a * stride];
                best = fmaxf(best, av);
            }
            node_values[node_idx] = best;
        } else {
            float sum = 0.0f;
            for (int a = 0; a < na; ++a) {
                float s  = current_strategy[strat_base + a * stride];
                float av = action_values   [av_base    + a * stride];
                sum += s * av;
            }
            node_values[node_idx] = sum;
        }
    } else {
        // Opponent's strategy absorbed into reach; just SUM action values
        float sum = 0.0f;
        for (int a = 0; a < na; ++a) {
            sum += action_values[av_base + a * stride];
        }
        node_values[node_idx] = sum;
    }
}

// Both <false> (CFR/EV) and <true> (BR/postsolve) instantiations are
// implicitly produced by the launcher calls in this same TU, so no explicit
// instantiation is needed.

// ============================================================================
// Kernel 4: Copy children's node_values into parent's action_values slots
// ============================================================================

/**
 * At each decision node n at this level, for each action a:
 *   action_values[n][a][c] = node_values[child(n,a)][c]
 *
 * Used during backward pass: after processing a deeper level to fill
 * node_values, this kernel lifts those into the parent's action_values
 * so aggregate_node_values_kernel can consume them.
 */
__global__ void lift_child_values_kernel(
    const uint8_t*  __restrict__ node_types,
    const uint8_t*  __restrict__ num_children,
    const uint32_t* __restrict__ children_offset,
    const uint32_t* __restrict__ children,
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ node_values,      // [N * nc]
    float*       __restrict__ action_values,    // [N * A * nc]
    uint8_t max_actions,
    uint16_t num_canonical)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    int local_idx = tid / num_canonical;
    int combo     = tid % num_canonical;
    uint32_t n    = level_node_indices[local_idx];

    uint8_t nt = node_types[n];
    if (nt == NT_TERMINAL) return;

    uint8_t na = num_children[n];
    if (na == 0) return;

    uint32_t offset = children_offset[n];
    size_t av_base = (static_cast<size_t>(n) * max_actions) * num_canonical
                   + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    for (int a = 0; a < na; ++a) {
        uint32_t child = children[offset + a];
        size_t child_idx = static_cast<size_t>(child) * num_canonical + combo;
        action_values[av_base + a * stride] = node_values[child_idx];
    }
}

// ============================================================================
// Kernel 5: Regret Update — accumulate instantaneous regret + DCFR discount
// ============================================================================

/**
 * At each traverser node (acting == traverser), for each action and combo:
 *   instant = action_value[a][c] - node_value[c]
 *   regret[a][c] = (regret * discount) + instant
 *
 * Discount applied to EXISTING regret before adding new (standard DCFR).
 * Positive/negative regrets discounted separately with alpha / beta.
 */
__global__ void update_regrets_kernel(
    float*       __restrict__ regrets,         // [N * A * nc]
    const float* __restrict__ action_values,   // [N * A * nc]
    const float* __restrict__ node_values,     // [N * nc]
    const uint8_t* __restrict__ node_types,
    const uint8_t* __restrict__ active_player,
    const uint8_t* __restrict__ num_children,
    uint32_t num_nodes,
    uint16_t num_canonical,
    uint8_t max_actions,
    int traverser,
    float pos_disc,
    float neg_disc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_nodes) * num_canonical;
    if (tid >= total) return;

    int node  = tid / num_canonical;
    int combo = tid % num_canonical;

    uint8_t nt = node_types[node];
    if (nt != NT_PLAYER_OOP && nt != NT_PLAYER_IP) return;
    if (active_player[node] != traverser) return;

    uint8_t na = num_children[node];
    if (na == 0) return;

    float nv = node_values[static_cast<size_t>(node) * num_canonical + combo];

    size_t reg_base = (static_cast<size_t>(node) * max_actions) * num_canonical
                    + static_cast<size_t>(combo);
    size_t av_base = reg_base;
    size_t stride = num_canonical;

    for (int a = 0; a < na; ++a) {
        float av = action_values[av_base + a * stride];
        float instant = av - nv;
        float r = regrets[reg_base + a * stride];
        r *= (r > 0.0f) ? pos_disc : neg_disc;
        r += instant;
        regrets[reg_base + a * stride] = r;
    }
}

// ============================================================================
// Kernel 6: Strategy Sum Update — strategy_sum += weight * reach * current_strategy
// ============================================================================

/**
 * At each traverser node, add weight-scaled contribution to strategy_sum.
 * Uses per-node reach (from forward pass) — NOT root reach. This is the
 * correct DCFR averaging formula.
 */
__global__ void update_strategy_sum_kernel(
    float*       __restrict__ strategy_sum,     // [N * A * nc]
    const float* __restrict__ current_strategy, // [N * A * nc]
    const float* __restrict__ reach_own,        // [N * nc] — traverser's reach
    const uint8_t* __restrict__ node_types,
    const uint8_t* __restrict__ active_player,
    const uint8_t* __restrict__ num_children,
    uint32_t num_nodes,
    uint16_t num_canonical,
    uint8_t max_actions,
    int traverser,
    float strat_weight,    // STANDARD: ((t+1)/(t+2))^gamma; POSTFLOP: (t'/(t'+1))^3
    int decay_and_add)     // 0 = standard accumulative; 1 = postflop decay-and-add
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = static_cast<int>(num_nodes) * num_canonical;
    if (tid >= total) return;

    int node  = tid / num_canonical;
    int combo = tid % num_canonical;

    uint8_t nt = node_types[node];
    if (nt != NT_PLAYER_OOP && nt != NT_PLAYER_IP) return;
    if (active_player[node] != traverser) return;

    uint8_t na = num_children[node];
    if (na == 0) return;

    size_t base = (static_cast<size_t>(node) * max_actions) * num_canonical
                + static_cast<size_t>(combo);
    size_t stride = num_canonical;

    if (decay_and_add) {
        // POSTFLOP: strategy_sum = strategy_sum * gamma_t + current_strategy
        // No reach weighting (matches postflop-solver). Epoch-reset gamma_t
        // (passed in as strat_weight) makes this an "average over recent
        // iterations of the current epoch."
        for (int a = 0; a < na; ++a) {
            float s = current_strategy[base + a * stride];
            float old = strategy_sum[base + a * stride];
            strategy_sum[base + a * stride] = old * strat_weight + s;
        }
    } else {
        // STANDARD: strategy_sum += weight * reach * current_strategy
        // Textbook DCFR reach-weighted accumulative average.
        float reach = reach_own[static_cast<size_t>(node) * num_canonical + combo];
        for (int a = 0; a < na; ++a) {
            float s = current_strategy[base + a * stride];
            strategy_sum[base + a * stride] += strat_weight * reach * s;
        }
    }
}

// ============================================================================
// Host-side launch helpers (called from gpu_backend.cu)
// ============================================================================

static constexpr int DEFAULT_BLOCK_SIZE = 256;

void launch_compute_strategy(
    const float* d_regrets, float* d_current_strategy,
    const uint8_t* d_num_children, const uint8_t* d_node_types,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions)
{
    int total = static_cast<int>(num_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    compute_strategy_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_regrets, d_current_strategy,
        d_num_children, d_node_types,
        num_nodes, nc, max_actions);
    CUDA_CHECK(cudaGetLastError());
}

void launch_propagate_reach(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_current_strategy, uint8_t max_actions,
    float* d_reach_oop, float* d_reach_ip, uint16_t nc)
{
    int total = static_cast<int>(num_level_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    propagate_reach_forward_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_node_types, d_active_player, d_num_children,
        d_children_offset, d_children,
        d_level_indices, num_level_nodes,
        d_current_strategy, max_actions,
        d_reach_oop, d_reach_ip, nc);
    CUDA_CHECK(cudaGetLastError());
}

void launch_aggregate_node_values(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_current_strategy, uint8_t max_actions,
    const float* d_action_values, float* d_node_values,
    uint16_t nc, int traverser)
{
    int total = static_cast<int>(num_level_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    aggregate_node_values_kernel<false><<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_node_types, d_active_player, d_num_children,
        d_children_offset, d_children, d_runout_weight,
        d_level_indices, num_level_nodes,
        d_current_strategy, max_actions,
        d_action_values, d_node_values, nc, traverser);
    CUDA_CHECK(cudaGetLastError());
}

// Postsolve variant: at traverser's own decision nodes, take max over actions
// instead of strategy-weighted sum. Used by best-response / exploitability
// computation. Opponent and chance nodes behave identically to the EV variant.
void launch_aggregate_node_values_br(
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint8_t*  d_runout_weight,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_current_strategy, uint8_t max_actions,
    const float* d_action_values, float* d_node_values,
    uint16_t nc, int traverser)
{
    int total = static_cast<int>(num_level_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    aggregate_node_values_kernel<true><<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_node_types, d_active_player, d_num_children,
        d_children_offset, d_children, d_runout_weight,
        d_level_indices, num_level_nodes,
        d_current_strategy, max_actions,
        d_action_values, d_node_values, nc, traverser);
    CUDA_CHECK(cudaGetLastError());
}

void launch_lift_child_values(
    const uint8_t* d_node_types, const uint8_t* d_num_children,
    const uint32_t* d_children_offset, const uint32_t* d_children,
    const uint32_t* d_level_indices, uint32_t num_level_nodes,
    const float* d_node_values, float* d_action_values,
    uint8_t max_actions, uint16_t nc)
{
    int total = static_cast<int>(num_level_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    lift_child_values_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_node_types, d_num_children,
        d_children_offset, d_children,
        d_level_indices, num_level_nodes,
        d_node_values, d_action_values, max_actions, nc);
    CUDA_CHECK(cudaGetLastError());
}

void launch_update_regrets(
    float* d_regrets,
    const float* d_action_values, const float* d_node_values,
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions,
    int traverser, float pos_disc, float neg_disc)
{
    int total = static_cast<int>(num_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    update_regrets_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_regrets, d_action_values, d_node_values,
        d_node_types, d_active_player, d_num_children,
        num_nodes, nc, max_actions,
        traverser, pos_disc, neg_disc);
    CUDA_CHECK(cudaGetLastError());
}

void launch_update_strategy_sum(
    float* d_strategy_sum, const float* d_current_strategy,
    const float* d_reach_own,
    const uint8_t* d_node_types, const uint8_t* d_active_player,
    const uint8_t* d_num_children,
    uint32_t num_nodes, uint16_t nc, uint8_t max_actions,
    int traverser, float strat_weight, int decay_and_add)
{
    int total = static_cast<int>(num_nodes) * nc;
    int grid = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    update_strategy_sum_kernel<<<grid, DEFAULT_BLOCK_SIZE>>>(
        d_strategy_sum, d_current_strategy, d_reach_own,
        d_node_types, d_active_player, d_num_children,
        num_nodes, nc, max_actions,
        traverser, strat_weight, decay_and_add);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace deepsolver
