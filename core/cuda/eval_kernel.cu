/**
 * @file eval_kernel.cu
 * @brief Device-side hand evaluation + per-terminal value kernels.
 *
 * Contains:
 *   - Correct variable-board-size hand evaluator (`d_evaluate_board`)
 *     Handles flop (5 cards), turn (6 → best of 6), river (7 → best of 21)
 *     WITHOUT the hole-card padding bug that previously broke flop evaluation.
 *
 *   - `showdown_terminal_kernel`: per-canonical-combo showdown value at a
 *     single terminal. Reads precomputed matchup matrices + opponent reach,
 *     outputs value[c] = Σ_cj reach[cj] · ev[c,cj] · valid[c,cj] · w[cj] · half_pot
 *
 *   - `fold_terminal_kernel`: per-canonical-combo fold value at a terminal.
 *     value[c] = sign · half_pot · Σ_cj reach[cj] · valid[c,cj] · w[cj]
 *
 * These two kernels are called per-terminal during the backward pass (Phase 4.5).
 */

#include "util.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace deepsolver {
namespace gpu {

// ============================================================================
// Device-side hand evaluator using lookup tables
// ============================================================================

/// Rank-to-prime mapping (mirrors CPU side)
__constant__ int d_rank_primes[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

/// Flush lookup table (uploaded from host)
__device__ uint16_t d_flush_table[8192];

/// Unique-5 lookup table
__device__ uint16_t d_unique5_table[8192];

/// Hash lookup table
__device__ uint16_t d_hash_table[65536];

/// Upload lookup tables from host to device __device__ arrays
void upload_eval_tables(const uint16_t* flush_table,
                        const uint16_t* unique5_table,
                        const uint16_t* hash_table) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_flush_table,  flush_table,   8192  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_unique5_table, unique5_table, 8192  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hash_table,   hash_table,    65536 * sizeof(uint16_t)));
}

/// Device-side 5-card evaluation
__device__ __forceinline__ uint16_t d_evaluate5(
    uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4)
{
    uint16_t rank_bits = (1u << (c0 / 4)) | (1u << (c1 / 4)) | (1u << (c2 / 4)) |
                         (1u << (c3 / 4)) | (1u << (c4 / 4));

    uint8_t s0 = c0 % 4;
    bool is_flush = (s0 == (c1 % 4)) && (s0 == (c2 % 4)) &&
                    (s0 == (c3 % 4)) && (s0 == (c4 % 4));

    if (is_flush) return d_flush_table[rank_bits];

    int unique = __popc(rank_bits & 0x1FFF);
    if (unique == 5) return d_unique5_table[rank_bits];

    uint32_t pp = d_rank_primes[c0 / 4] * d_rank_primes[c1 / 4] *
                  d_rank_primes[c2 / 4] * d_rank_primes[c3 / 4] *
                  d_rank_primes[c4 / 4];

    uint32_t idx = pp % 65536;
    while (d_hash_table[idx] == 0) {
        idx = (idx + 1) % 65536;
    }
    return d_hash_table[idx];
}

/// Device-side 7-card evaluation (best of 21 5-card combinations)
__device__ __forceinline__ uint16_t d_evaluate7(
    uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3,
    uint8_t c4, uint8_t c5, uint8_t c6)
{
    uint8_t cards[7] = {c0, c1, c2, c3, c4, c5, c6};
    uint16_t best = 0xFFFF;
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 7; ++j) {
            uint8_t sub[5]; int k = 0;
            for (int m = 0; m < 7; ++m) {
                if (m != i && m != j) sub[k++] = cards[m];
            }
            uint16_t rank = d_evaluate5(sub[0], sub[1], sub[2], sub[3], sub[4]);
            if (rank < best) best = rank;
        }
    }
    return best;
}

/// Device-side 6-card evaluation (best of 6 5-card combinations)
__device__ __forceinline__ uint16_t d_evaluate6(
    uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5)
{
    uint8_t cards[6] = {c0, c1, c2, c3, c4, c5};
    uint16_t best = 0xFFFF;
    for (int skip = 0; skip < 6; ++skip) {
        uint8_t h[5]; int idx = 0;
        for (int j = 0; j < 6; ++j) if (j != skip) h[idx++] = cards[j];
        uint16_t r = d_evaluate5(h[0], h[1], h[2], h[3], h[4]);
        if (r < best) best = r;
    }
    return best;
}

/**
 * @brief Evaluate a hole-card pair against a variable-length board.
 *
 * Replaces the previous implementation that used hole cards as padding for
 * flop boards — that was broken because duplicated cards produced incorrect
 * rank lookups.
 */
__device__ __forceinline__ uint16_t d_evaluate_board(
    uint8_t hole0, uint8_t hole1,
    const uint8_t* __restrict__ board, int board_size)
{
    if (board_size == 5) {
        return d_evaluate7(hole0, hole1,
                           board[0], board[1], board[2], board[3], board[4]);
    }
    if (board_size == 4) {
        return d_evaluate6(hole0, hole1,
                           board[0], board[1], board[2], board[3]);
    }
    // Flop: 5 total cards, no padding needed
    return d_evaluate5(hole0, hole1, board[0], board[1], board[2]);
}

// ============================================================================
// Showdown evaluation kernel (one pair of combos → payoff, [-half_pot,+half_pot])
// ============================================================================

/**
 * @brief Evaluate showdown payoffs for all (OOP combo, IP combo) pairs.
 *
 * @param board         5-card padded board array (only first board_size cells valid)
 * @param board_size    3/4/5 for flop/turn/river
 * @param combo_cards   Original-combo-index → 2 cards, flat [NUM_COMBOS * 2]
 * @param valid_indices Subset of combo indices that are valid in range
 * @param num_valid     Length of valid_indices
 * @param pot           Pot at this terminal
 * @param payoffs_oop   Output: payoff for OOP per pair [num_valid * num_valid]
 *                      +half_pot = OOP wins, -half_pot = IP wins, 0 = tie/conflict
 *
 * Useful for GPU-side precomputation of showdown payoffs if the CPU-side
 * matchup matrix is not uploaded. Currently unused in the main pipeline
 * (CPU precomputes and we upload), kept for future GPU-only matrix builds.
 */
__global__ void showdown_eval_kernel(
    const uint8_t* __restrict__ board,
    int board_size,
    const uint8_t* __restrict__ combo_cards,
    const uint16_t* __restrict__ valid_indices,
    int num_valid,
    float pot,
    __half* __restrict__ payoffs_oop)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid * num_valid) return;

    int oop_idx = idx / num_valid;
    int ip_idx  = idx % num_valid;
    if (oop_idx == ip_idx) {
        payoffs_oop[idx] = __float2half(0.0f);
        return;
    }

    uint16_t oop_combo = valid_indices[oop_idx];
    uint16_t ip_combo  = valid_indices[ip_idx];

    uint8_t oop_c0 = combo_cards[oop_combo * 2];
    uint8_t oop_c1 = combo_cards[oop_combo * 2 + 1];
    uint8_t ip_c0  = combo_cards[ip_combo  * 2];
    uint8_t ip_c1  = combo_cards[ip_combo  * 2 + 1];

    // Card conflict between the two hands
    if (oop_c0 == ip_c0 || oop_c0 == ip_c1 ||
        oop_c1 == ip_c0 || oop_c1 == ip_c1) {
        payoffs_oop[idx] = __float2half(0.0f);
        return;
    }

    uint16_t oop_rank = d_evaluate_board(oop_c0, oop_c1, board, board_size);
    uint16_t ip_rank  = d_evaluate_board(ip_c0,  ip_c1,  board, board_size);

    float half_pot = pot * 0.5f;
    float payoff;
    if      (oop_rank < ip_rank) payoff =  half_pot;   // OOP wins (lower rank = stronger)
    else if (oop_rank > ip_rank) payoff = -half_pot;   // IP wins
    else                         payoff =  0.0f;       // tie

    payoffs_oop[idx] = __float2half(payoff);
}

// ============================================================================
// Per-terminal showdown kernel (used in Phase 4.5 backward pass)
// ============================================================================

/**
 * @brief Compute per-combo showdown counterfactual value at a single terminal.
 *
 * For perspective == 0 (OOP traverser):
 *   value[c] = Σ_cj reach_ip[cj] · matchup_ev[c, cj] · matchup_valid[c, cj]
 *                 · canonical_weight[cj] · half_pot
 *
 * For perspective == 1 (IP traverser):
 *   value[c] = Σ_ci reach_oop[ci] · (-matchup_ev[ci, c]) · matchup_valid[ci, c]
 *                 · canonical_weight[ci] · half_pot
 *
 * One thread per canonical combo. Launch with `nc` threads.
 *
 * Phase 2: callers pass already-offset pointers into the per-runout concat
 * buffer (matchup_ev + matchup_idx[node] * nc * nc), so the kernel itself is
 * unchanged. The offset arithmetic happens in the launch helper.
 *
 * All inputs/outputs use FP32 for numerical safety during the backward pass.
 */
__global__ void showdown_terminal_kernel(
    const float* __restrict__ matchup_ev,        // [nc * nc] (per-runout slice)
    const float* __restrict__ matchup_valid,     // [nc * nc] (per-runout slice)
    const float* __restrict__ canonical_weights, // [nc] (float copy of uint16_t)
    const float* __restrict__ reach_opp,         // [nc] — opponent's reach at terminal
    uint16_t num_canonical,
    float half_pot,
    int perspective,                              // 0 = OOP, 1 = IP
    float* __restrict__ out_values                // [nc]
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_canonical) return;

    float val = 0.0f;
    if (perspective == 0) {
        // OOP traverses: sum over IP combos (cj)
        for (int cj = 0; cj < num_canonical; ++cj) {
            size_t idx = static_cast<size_t>(c) * num_canonical + cj;
            val += reach_opp[cj] * matchup_ev[idx] * matchup_valid[idx]
                 * canonical_weights[cj] * half_pot;
        }
    } else {
        // IP traverses: sum over OOP combos (ci), negate matchup_ev sign
        for (int ci = 0; ci < num_canonical; ++ci) {
            size_t idx = static_cast<size_t>(ci) * num_canonical + c;
            val += reach_opp[ci] * (-matchup_ev[idx]) * matchup_valid[idx]
                 * canonical_weights[ci] * half_pot;
        }
    }
    out_values[c] = val;
}

// ============================================================================
// Per-terminal fold kernel (used in Phase 4.5 backward pass)
// ============================================================================

/**
 * @brief Compute per-combo fold counterfactual value at a single terminal.
 *
 * For fold terminal:
 *   value[c] = sign · half_pot · Σ_cj reach_opp[cj] · matchup_valid[c,cj or cj,c] · w[cj]
 *
 * `sign_for_perspective` encodes:
 *   FOLD_OOP + perspective=0 (OOP loses)    → -1
 *   FOLD_OOP + perspective=1 (IP  wins)     → +1
 *   FOLD_IP  + perspective=0 (OOP wins)     → +1
 *   FOLD_IP  + perspective=1 (IP  loses)    → -1
 *
 * `matrix_orientation` = 0 indexes matchup_valid as [c, cj] (OOP perspective),
 * = 1 indexes as [ci, c] (IP perspective). Matches showdown_terminal_kernel.
 */
__global__ void fold_terminal_kernel(
    const float* __restrict__ matchup_valid,     // [nc * nc]
    const float* __restrict__ canonical_weights, // [nc]
    const float* __restrict__ reach_opp,         // [nc]
    uint16_t num_canonical,
    float half_pot,
    float sign_for_perspective,
    int perspective,                              // 0 = OOP, 1 = IP
    float* __restrict__ out_values                // [nc]
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_canonical) return;

    float opp_total = 0.0f;
    if (perspective == 0) {
        for (int cj = 0; cj < num_canonical; ++cj) {
            size_t idx = static_cast<size_t>(c) * num_canonical + cj;
            opp_total += reach_opp[cj] * matchup_valid[idx] * canonical_weights[cj];
        }
    } else {
        for (int ci = 0; ci < num_canonical; ++ci) {
            size_t idx = static_cast<size_t>(ci) * num_canonical + c;
            opp_total += reach_opp[ci] * matchup_valid[idx] * canonical_weights[ci];
        }
    }
    out_values[c] = sign_for_perspective * half_pot * opp_total;
}

// ============================================================================
// Host-side launch helpers
// ============================================================================

void launch_showdown_terminal(
    const float* d_matchup_ev, const float* d_matchup_valid,
    const float* d_canonical_weights, const float* d_reach_opp,
    uint16_t nc, float half_pot, int perspective,
    float* d_out)
{
    int block = 256;
    int grid  = (nc + block - 1) / block;
    showdown_terminal_kernel<<<grid, block>>>(
        d_matchup_ev, d_matchup_valid, d_canonical_weights, d_reach_opp,
        nc, half_pot, perspective, d_out);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fold_terminal(
    const float* d_matchup_valid, const float* d_canonical_weights,
    const float* d_reach_opp, uint16_t nc, float half_pot,
    float sign_for_perspective, int perspective, float* d_out)
{
    int block = 256;
    int grid  = (nc + block - 1) / block;
    fold_terminal_kernel<<<grid, block>>>(
        d_matchup_valid, d_canonical_weights, d_reach_opp,
        nc, half_pot, sign_for_perspective, perspective, d_out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace deepsolver
