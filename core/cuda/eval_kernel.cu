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

// Mirror host enum values. Keep in sync with types.h.
constexpr uint8_t NT_TERMINAL = 3;
constexpr uint8_t TT_FOLD_OOP = 0;
constexpr uint8_t TT_FOLD_IP  = 1;
constexpr uint8_t TT_SHOWDOWN = 2;

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

/// Parallel key table (prime products) for verified probing. Without it the
/// device probe returns the first non-empty slot, aliasing colliding hands.
__device__ uint32_t d_hash_keys[65536];

/// Upload lookup tables from host to device __device__ arrays
void upload_eval_tables(const uint16_t* flush_table,
                        const uint16_t* unique5_table,
                        const uint16_t* hash_table,
                        const uint32_t* hash_keys) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_flush_table,  flush_table,   8192  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_unique5_table, unique5_table, 8192  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hash_table,   hash_table,    65536 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hash_keys,    hash_keys,     65536 * sizeof(uint32_t)));
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
    // Verify the stored key (see CPU evaluate5): the old keyless probe returned
    // the first non-empty slot, aliasing the 197 colliding paired-hand prime
    // products to each other's rank.
    while (d_hash_table[idx] != 0 && d_hash_keys[idx] != pp) {
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
// Batched per-level terminal kernel
// ============================================================================

__global__ void terminal_level_kernel(
    const uint8_t* __restrict__ node_types,
    const uint8_t* __restrict__ terminal_types,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ parent_indices,
    const float* __restrict__ bet_into,
    const int32_t* __restrict__ matchup_idx,
    const uint32_t* __restrict__ value_row,
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const float* __restrict__ matchup_ev_concat,
    const float* __restrict__ matchup_valid_concat,
    const float* __restrict__ canonical_weights,
    uint32_t num_runouts,
    const float* __restrict__ reach_opp_base,
    // B1b inc 1: the opponent's live canonical combos, ascending. Every loop
    // below sums reach_opp[cj]*...; zero-reach terms contribute exactly +-0.0
    // and x + 0.0 == x, so walking this list instead of [0, nc) is bit-exact
    // and cuts the inner loop to the opponent's real range.
    const uint16_t* __restrict__ opp_live,
    uint16_t num_opp_live,
    uint16_t num_canonical,
    int perspective,
    float rake_rate,
    float rake_cap,
    // 2026-09-09 audit P0: partial-board showdown equity tables, packed
    // [num_equity_tables × nc × nc]; equity_slot[mi] = slot or −1. See
    // SolverContext::matchup_equity_per_runout.
    const float* __restrict__ equity_concat,
    const int32_t* __restrict__ equity_slot,
    float* __restrict__ node_values)
{
    // 64-bit launch index: level_nodes × nc can exceed INT32_MAX on the
    // 4.65M-node roadmap target (see cfr_kernel.cu).
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_level_nodes) * num_canonical;
    if (tid >= total) return;

    const uint32_t local_idx = static_cast<uint32_t>(tid / num_canonical);
    const uint32_t c = static_cast<uint32_t>(tid % num_canonical);
    uint32_t n = level_node_indices[local_idx];
    if (node_types[n] != NT_TERMINAL) return;

    int32_t mi = matchup_idx[n];
    if (mi < 0 || static_cast<uint32_t>(mi) >= num_runouts) mi = 0;
    size_t per_table = static_cast<size_t>(num_canonical) * num_canonical;
    size_t table_off = static_cast<size_t>(mi) * per_table;
    const float* matchup_ev = matchup_ev_concat + table_off;
    const float* matchup_valid = matchup_valid_concat + table_off;
    const float* reach_opp = reach_opp_base + static_cast<size_t>(n) * num_canonical;
    // B3 inc 1: rows are addressed through value_row, not by node index.
    float* out_values =
        node_values + static_cast<size_t>(value_row[n]) * num_canonical;

    float pot_total = pots[n];
    float half_pot = pot_total * 0.5f;
    float rake = fminf(pot_total * rake_rate, rake_cap);
    if (rake < 0.0f) rake = 0.0f;
    float win_payoff  = half_pot - rake;
    float lose_payoff = -half_pot;
    float tie_payoff  = -0.5f * rake;

    uint8_t tt = terminal_types[n];
    if (tt == TT_SHOWDOWN) {
        // 2026-09-09 audit P0: cards still to come → settle on the equity
        // table (expected win − lose over every remaining runout × compat),
        // linear payoff A·eq + B·valid with A = half_pot − rake/2, B = −rake/2.
        const int32_t eq_slot = (equity_slot != nullptr) ? equity_slot[mi] : -1;
        if (eq_slot >= 0) {
            const float* eq = equity_concat + static_cast<size_t>(eq_slot) * per_table;
            const float A = half_pot - 0.5f * rake;
            const float B = -0.5f * rake;
            float s1 = 0.0f, s2 = 0.0f;
            if (perspective == 0) {
                for (int k = 0; k < num_opp_live; ++k) {
                    const int cj = opp_live[k];
                    const size_t idx = static_cast<size_t>(c) * num_canonical + cj;
                    const float w = reach_opp[cj] * canonical_weights[cj];
                    s1 += w * eq[idx];
                    if (B != 0.0f) s2 += w * matchup_valid[idx];
                }
                out_values[c] = A * s1 + B * s2;
            } else {
                for (int k = 0; k < num_opp_live; ++k) {
                    const int ci = opp_live[k];
                    const size_t idx = static_cast<size_t>(ci) * num_canonical + c;
                    const float w = reach_opp[ci] * canonical_weights[ci];
                    s1 += w * eq[idx];
                    if (B != 0.0f) s2 += w * matchup_valid[idx];
                }
                out_values[c] = -A * s1 + B * s2;
            }
            return;
        }
        // Per-pair payoff: branch on signed matchup_ev. m_ev is OOP-perspective
        // (+1 = OOP wins, -1 = OOP loses, 0 = tie).
        float val = 0.0f;
        if (perspective == 0) {
            for (int k = 0; k < num_opp_live; ++k) {
                const int cj = opp_live[k];
                size_t idx = static_cast<size_t>(c) * num_canonical + cj;
                float ev = matchup_ev[idx];
                float valid = matchup_valid[idx];
                if (valid <= 0.0f) continue;
                float payoff;
                if (ev > 0.5f)        payoff = win_payoff;
                else if (ev < -0.5f)  payoff = lose_payoff;
                else                  payoff = tie_payoff;
                val += reach_opp[cj] * valid * canonical_weights[cj] * payoff;
            }
        } else {
            for (int k = 0; k < num_opp_live; ++k) {
                const int ci = opp_live[k];
                size_t idx = static_cast<size_t>(ci) * num_canonical + c;
                float ev = matchup_ev[idx];  // OOP perspective
                float valid = matchup_valid[idx];
                if (valid <= 0.0f) continue;
                float payoff;
                if (ev > 0.5f)        payoff = lose_payoff;  // OOP wins → IP loses
                else if (ev < -0.5f)  payoff = win_payoff;
                else                  payoff = tie_payoff;
                val += reach_opp[ci] * valid * canonical_weights[ci] * payoff;
            }
        }
        out_values[c] = val;
        return;
    }

    uint32_t parent = parent_indices[n];
    float unmatched = bet_into[parent];
    float matched_pot = pot_total - unmatched;
    // Fold rake base = the MATCHED pot (uncalled bet returned first) — mirrors
    // types.h::terminal_rake and the CPU kernels.
    float fold_rake = fminf(matched_pot * rake_rate, rake_cap);
    if (fold_rake < 0.0f) fold_rake = 0.0f;
    float fold_win_gain  = matched_pot * 0.5f - fold_rake;
    float fold_lose_loss = -matched_pot * 0.5f;

    // FOLD_OOP = OOP folded → IP wins. FOLD_IP = IP folded → OOP wins.
    bool i_win = ((perspective == 0 && tt == TT_FOLD_IP) ||
                  (perspective == 1 && tt == TT_FOLD_OOP));
    float self_payoff = i_win ? fold_win_gain : fold_lose_loss;

    float opp_total = 0.0f;
    if (perspective == 0) {
        for (int k = 0; k < num_opp_live; ++k) {
            const int cj = opp_live[k];
            size_t idx = static_cast<size_t>(c) * num_canonical + cj;
            opp_total += reach_opp[cj] * matchup_valid[idx] * canonical_weights[cj];
        }
    } else {
        for (int k = 0; k < num_opp_live; ++k) {
            const int ci = opp_live[k];
            size_t idx = static_cast<size_t>(ci) * num_canonical + c;
            opp_total += reach_opp[ci] * matchup_valid[idx] * canonical_weights[ci];
        }
    }
    out_values[c] = self_payoff * opp_total;
}

// ============================================================================
// Rank-blocker terminal kernel (O(nc·~92) replacement for terminal_level_kernel)
// ============================================================================
//
// For singleton-isomorphism boards (every canonical combo = exactly one
// original, weight 1) the showdown category matrix is just rank comparison
// plus private-card compatibility. This kernel mirrors the CPU
// `showdown_rank_blocker` shortcut on the GPU:
//
//   1. scatter opponent reach into per-rank buckets (atomicAdd into shared mem),
//   2. exclusive prefix-sum over buckets → `stronger`/`weaker`/`same` splits,
//   3. per self-combo c, correct for card removal by iterating ONLY the ~92
//      opponents that share either of c's two cards (card_list CSR).
//
// This is O(nc · ~92) per terminal vs the dense kernel's O(nc²), ~13× fewer
// flops on a rainbow flop (nc≈1176). One CUDA block per terminal.
//
// Perspective-symmetric: the rank comparison is always c-vs-opponent, so the
// same win/lose/tie split serves both OOP and IP traversers — only
// `reach_opp_base` differs (set by the caller, exactly as the dense kernel).

constexpr uint16_t kRbNoBucket = 0xFFFFu;  // mirrors showdown_rank_blocker::Scratch::kNoBucket
constexpr int      kRbBlock    = 256;

// Fixed-point scale (2^42) for the bucket scatter. float atomicAdd resolves
// same-bucket conflicts in warp-scheduling order, which is not reproducible
// across runs — measured as ~1e-4 strategy wobble on node-locked solves
// (locks perturb the uniform iter-0 reach; equal addends had masked the
// order-dependence). 64-bit integer adds are associative, so quantizing to
// fixed point makes the sums bit-identical no matter the scheduling.
// Precision: per-addend error ≤ 2^-43 absolute (reach ≤ ~1e2), total over
// nc ≤ 1326 addends ≲ 1.5e-10 — far below float32 ulp of typical totals.
// Overflow needs a reach sum > 2^64/2^42 ≈ 4.2e6; real sums are ≤ ~1e5.
constexpr float  kRbFixedScale    = 4398046511104.0f;       // 2^42
constexpr double kRbFixedInvScale = 1.0 / 4398046511104.0;  // 2^-42

__global__ void rank_blocker_terminal_kernel(
    const uint8_t* __restrict__ node_types,
    const uint8_t* __restrict__ terminal_types,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ parent_indices,
    const float* __restrict__ bet_into,
    const int32_t* __restrict__ matchup_idx,
    const uint32_t* __restrict__ value_row,
    const uint32_t* __restrict__ level_node_indices,
    uint32_t num_level_nodes,
    const uint16_t* __restrict__ combo_bucket,   // [num_runouts * nc] per-runout
    const uint16_t* __restrict__ bucket_count,   // [num_runouts]
    const uint8_t* __restrict__ combo_card0,     // [nc]
    const uint8_t* __restrict__ combo_card1,     // [nc]
    const uint32_t* __restrict__ card_off,       // [NUM_CARDS+1]
    const uint16_t* __restrict__ card_list,      // [2*nc]
    uint32_t num_runouts,
    const float* __restrict__ reach_opp_base,    // [N * nc]
    float* __restrict__ node_values,             // [value_rows * nc]
    uint16_t num_canonical,
    uint16_t max_bucket_count,
    int perspective,
    float rake_rate,
    float rake_cap,
    // 2026-09-09 audit P0: partial-board showdown equity tables (see
    // terminal_level_kernel). Singleton iso ⇒ every canonical weight is 1.
    const float* __restrict__ equity_concat,
    const int32_t* __restrict__ equity_slot)
{
    const uint32_t blk = blockIdx.x;
    if (blk >= num_level_nodes) return;
    const uint32_t n = level_node_indices[blk];
    if (node_types[n] != NT_TERMINAL) return;   // uniform across the block

    const int nc = static_cast<int>(num_canonical);
    const int tid = threadIdx.x;

    int32_t mi = matchup_idx[n];
    if (mi < 0 || static_cast<uint32_t>(mi) >= num_runouts) mi = 0;
    const uint16_t* __restrict__ mbucket = combo_bucket + static_cast<size_t>(mi) * nc;
    const int B = static_cast<int>(bucket_count[mi]);
    const float* __restrict__ reach_opp = reach_opp_base + static_cast<size_t>(n) * nc;
    float* __restrict__ out_values =
        node_values + static_cast<size_t>(value_row[n]) * nc;

    // Shared memory layout (ull array first for 8-byte alignment):
    // [s_acc : max_B ull][s_total : max_B][s_prefix : max_B+1][s_chunk : blockDim]
    extern __shared__ unsigned long long s_acc[];
    float* smem_f   = reinterpret_cast<float*>(s_acc + max_bucket_count);
    float* s_total  = smem_f;
    float* s_prefix = smem_f + max_bucket_count;
    float* s_chunk  = smem_f + (2 * static_cast<int>(max_bucket_count) + 1);

    // 1) zero bucket totals, then scatter opponent reach into rank buckets.
    //    Accumulate in 64-bit fixed point (see kRbFixedScale) so the result
    //    does not depend on the atomic resolution order.
    for (int i = tid; i < B; i += blockDim.x) s_acc[i] = 0ull;
    __syncthreads();
    for (int o = tid; o < nc; o += blockDim.x) {
        const uint16_t bo = mbucket[o];
        if (bo != kRbNoBucket) {
            const float r = reach_opp[o];
            if (r != 0.0f) {
                const unsigned long long q = __float2ull_rn(r * kRbFixedScale);
                if (q != 0ull) atomicAdd(&s_acc[bo], q);
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < B; i += blockDim.x) {
        s_total[i] = static_cast<float>(
            __ull2double_rn(s_acc[i]) * kRbFixedInvScale);
    }
    __syncthreads();

    // 2) exclusive prefix-sum of s_total[0..B) → s_prefix[0..B]. Chunk scan:
    //    each thread serial-scans a contiguous chunk, thread 0 scans the
    //    per-chunk totals, then each thread adds its chunk offset.
    const int chunk = (B + blockDim.x - 1) / blockDim.x;
    const int start = tid * chunk;
    const int stop  = min(start + chunk, B);
    float local = 0.0f;
    for (int i = start; i < stop; ++i) local += s_total[i];
    s_chunk[tid] = local;
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < blockDim.x; ++t) { float v = s_chunk[t]; s_chunk[t] = acc; acc += v; }
        s_prefix[B] = acc;   // grand total
    }
    __syncthreads();
    {
        float run = s_chunk[tid];
        for (int i = start; i < stop; ++i) { s_prefix[i] = run; run += s_total[i]; }
    }
    __syncthreads();
    const float total = s_prefix[B];

    // 3) per-node payoff constants (uniform across the block).
    const float pot_total = pots[n];
    const float half_pot  = pot_total * 0.5f;
    float rake = fminf(pot_total * rake_rate, rake_cap);
    if (rake < 0.0f) rake = 0.0f;
    const float win_p  = half_pot - rake;
    const float lose_p = -half_pot;
    const float tie_p  = -0.5f * rake;

    const uint8_t tt = terminal_types[n];
    const bool is_showdown = (tt == TT_SHOWDOWN);
    const int32_t eq_slot = (is_showdown && equity_slot != nullptr) ? equity_slot[mi] : -1;
    const float* __restrict__ eq = (eq_slot >= 0)
        ? equity_concat + static_cast<size_t>(eq_slot) * static_cast<size_t>(nc) * nc
        : nullptr;

    // Fold self-payoff (matches terminal_level_kernel / CPU fold_self_payoff).
    const uint32_t parent = parent_indices[n];
    const float matched_pot = pot_total - bet_into[parent];
    // Fold rake base = the MATCHED pot — see terminal_level_kernel.
    float fold_rake = fminf(matched_pot * rake_rate, rake_cap);
    if (fold_rake < 0.0f) fold_rake = 0.0f;
    const float fold_win_gain  = matched_pot * 0.5f - fold_rake;
    const float fold_lose_loss = -matched_pot * 0.5f;
    const bool i_win = ((perspective == 0 && tt == TT_FOLD_IP) ||
                        (perspective == 1 && tt == TT_FOLD_OOP));
    const float self_payoff = i_win ? fold_win_gain : fold_lose_loss;

    // 4) per self-combo value.
    for (int c = tid; c < nc; c += blockDim.x) {
        const uint16_t b16 = mbucket[c];
        if (b16 == kRbNoBucket) { out_values[c] = 0.0f; continue; }
        const int b = b16;
        const uint8_t k0 = combo_card0[c];
        const uint8_t k1 = combo_card1[c];
        const uint32_t lo0 = card_off[k0], hi0 = card_off[k0 + 1];
        const uint32_t lo1 = card_off[k1], hi1 = card_off[k1 + 1];

        if (eq != nullptr) {
            // Equity showdown: A·Σ_j reach[j]·eq[c,j] (OOP) or −A·Σ_ci reach[ci]·eq[ci,c]
            // (IP), plus B × the card-compatible opponent mass on raked solves —
            // the same compatibility sum the fold branch below computes.
            const float A = half_pot - 0.5f * rake;
            const float B = -0.5f * rake;
            float s1 = 0.0f;
            if (perspective == 0) {
                const float* row = eq + static_cast<size_t>(c) * nc;
                for (int j = 0; j < nc; ++j) s1 += reach_opp[j] * row[j];
            } else {
                for (int ci = 0; ci < nc; ++ci) {
                    s1 += reach_opp[ci] * eq[static_cast<size_t>(ci) * nc + c];
                }
                s1 = -s1;
            }
            float val = A * s1;
            if (B != 0.0f) {
                float opp_total = total;
                for (uint32_t k = lo0; k < hi0; ++k) {
                    const uint16_t o = card_list[k];
                    if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
                }
                for (uint32_t k = lo1; k < hi1; ++k) {
                    const uint16_t o = card_list[k];
                    if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
                }
                opp_total += reach_opp[c];
                val += B * opp_total;
            }
            out_values[c] = val;
        } else if (is_showdown) {
            float stronger = s_prefix[b];
            float weaker   = total - s_prefix[b + 1];
            float same     = s_total[b];
            // Subtract every opponent sharing card k0 or k1 (combo c itself
            // appears in BOTH lists → subtracted twice from `same`).
            for (uint32_t k = lo0; k < hi0; ++k) {
                const uint16_t o = card_list[k];
                const uint16_t bo = mbucket[o];
                if (bo == kRbNoBucket) continue;
                const float r = reach_opp[o];
                if      (bo < b) stronger -= r;
                else if (bo > b) weaker   -= r;
                else             same     -= r;
            }
            for (uint32_t k = lo1; k < hi1; ++k) {
                const uint16_t o = card_list[k];
                const uint16_t bo = mbucket[o];
                if (bo == kRbNoBucket) continue;
                const float r = reach_opp[o];
                if      (bo < b) stronger -= r;
                else if (bo > b) weaker   -= r;
                else             same     -= r;
            }
            same += reach_opp[c];   // undo the double subtraction of c
            out_values[c] = win_p * weaker + lose_p * stronger + tie_p * same;
        } else {
            float opp_total = total;
            for (uint32_t k = lo0; k < hi0; ++k) {
                const uint16_t o = card_list[k];
                if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
            }
            for (uint32_t k = lo1; k < hi1; ++k) {
                const uint16_t o = card_list[k];
                if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
            }
            opp_total += reach_opp[c];   // undo the double subtraction of c
            out_values[c] = self_payoff * opp_total;
        }
    }
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
    const float* d_equity_concat,
    const int32_t* d_equity_slot,
    float* d_node_values)
{
    const int block = 256;
    const size_t total = static_cast<size_t>(num_level_nodes) * nc;
    const int grid = static_cast<int>(
        (total + block - 1) / static_cast<size_t>(block));
    if (grid == 0) return;

    terminal_level_kernel<<<grid, block>>>(
        d_node_types, d_terminal_types, d_pots,
        d_parent_indices, d_bet_into, d_matchup_idx,
        d_value_row, d_level_indices, num_level_nodes,
        d_matchup_ev_concat, d_matchup_valid_concat,
        d_canonical_weights, num_runouts,
        d_reach_opp_base, d_opp_live, num_opp_live, nc, perspective,
        rake_rate, rake_cap, d_equity_concat, d_equity_slot, d_node_values);
    CUDA_CHECK(cudaGetLastError());
}

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
    const float* d_equity_concat,
    const int32_t* d_equity_slot,
    float* d_node_values)
{
    if (num_level_nodes == 0) return;
    const int block = kRbBlock;
    const uint32_t grid = num_level_nodes;   // one block per terminal
    // shared: s_acc[max_B] (ull) + s_total[max_B] + s_prefix[max_B+1] + s_chunk[block]
    const size_t shmem =
        static_cast<size_t>(max_bucket_count) * sizeof(unsigned long long) +
        (static_cast<size_t>(2) * max_bucket_count + 1 + block) * sizeof(float);

    rank_blocker_terminal_kernel<<<grid, block, shmem>>>(
        d_node_types, d_terminal_types, d_pots,
        d_parent_indices, d_bet_into, d_matchup_idx,
        d_value_row, d_level_indices, num_level_nodes,
        d_combo_bucket, d_bucket_count,
        d_combo_card0, d_combo_card1, d_card_off, d_card_list,
        num_runouts, d_reach_opp_base, d_node_values,
        nc, max_bucket_count, perspective, rake_rate, rake_cap,
        d_equity_concat, d_equity_slot);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2026-09-10: batched partial-board (equity) showdowns — one GEMM per pass.
//
// A SHOWDOWN terminal whose table still has cards to come settles on the
// equity matrix E (win − lose over every completion × compat; antisymmetric:
// E[j][c] = −E[c][j]). Per terminal that is
//     OOP: out[c] =  A·Σ_j E[c][j]·w[j]
//     IP : out[c] = −A·Σ_j E[j][c]·w[j] = A·Σ_j E[c][j]·w[j]
// — the SAME product for both traversers, with w = reach_opp ⊙ weights and
// A = half_pot − rake/2. The per-terminal branches in the two kernels above
// stream the whole nc² table once per terminal; a collapsed full-menu rainbow
// flop has ~4.8k such terminals on ONE table, which measured 35 ms/iteration
// on an RTX 5090 (27× the pre-audit iteration). Batched, every group is one
// GEMM  OUT[M×nc] = W[M×nc]·Eᵀ  with 64×64×16 shared-memory tiling and 4×4
// outputs per thread — the whole tree's equity terminals in ONE launch, tiles
// indexed through EquityTile. Deterministic: fixed summation order, no
// atomics (CliGpuLockDeterminism).
//
// Raked solves add B·compat[c] (B = −rake/2, compat = card-compatible
// opponent mass, exactly what the per-terminal branches add): on rank-blocker
// boards via equity_rb_compat_add_kernel (card lists), on dense boards as a
// second GEMM against the per-runout `valid` table (coef_kind 1, accumulate).
// ============================================================================

constexpr int kEqTileM   = 64;   // terminals per tile (EquityTile span cap)
constexpr int kEqTileN   = 64;   // hands per tile
constexpr int kEqTileK   = 16;   // K step
constexpr int kEqThreads = 256;  // 16×16 threads × 4×4 outputs
// Row stride of the shared tiles: 68 keeps float4 reads 16-byte aligned and
// limits the transposed store (16 k values into one column) to a 2-way bank
// conflict; a stride of 64 would make it 16-way.
constexpr int kEqSmemStride = kEqTileM + 4;

__global__ void equity_showdown_gemm_kernel(
    const EquityTile* __restrict__ tiles,
    const uint32_t* __restrict__ eq_terminal_order,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ value_row,
    const float* __restrict__ matrix_concat,
    int index_by_runout,          // 0: matrix_concat[eq_slot]; 1: matrix_concat[mi]
    const float* __restrict__ canonical_weights,
    const float* __restrict__ reach_opp_base,
    uint16_t num_canonical,
    float rake_rate,
    float rake_cap,
    int coef_kind,                // 0: A = half_pot − rake/2; 1: B = −rake/2
    int accumulate,               // 0: out = coef·acc; 1: out += coef·acc
    float* __restrict__ node_values)
{
    const EquityTile tile = tiles[blockIdx.x];
    const int nc = static_cast<int>(num_canonical);
    const int c0 = static_cast<int>(blockIdx.y) * kEqTileN;
    const int M  = static_cast<int>(tile.t_end - tile.t_begin);
    const size_t per_table = static_cast<size_t>(nc) * nc;
    const float* __restrict__ mat = matrix_concat
        + static_cast<size_t>(index_by_runout ? tile.mi : tile.eq_slot) * per_table;

    __shared__ __align__(16) float Ws[kEqTileK][kEqSmemStride];   // [k][terminal]
    __shared__ __align__(16) float Ms[kEqTileK][kEqSmemStride];   // [k][hand]
    __shared__ uint32_t s_node[kEqTileM];

    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid & 15;
    const int ty = tid >> 4;
    for (int t = tid; t < kEqTileM; t += kEqThreads) {
        s_node[t] = (t < M) ? eq_terminal_order[tile.t_begin + t] : 0u;
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }

    for (int k0 = 0; k0 < nc; k0 += kEqTileK) {
        // Cooperative loads. Element e ∈ [0, 1024): k = e & 15 varies fastest
        // so 16 consecutive threads read 64 contiguous bytes of one row;
        // r = e >> 4 is the terminal (W) or the hand (matrix).
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e  = tid + i * kEqThreads;
            const int k  = e & (kEqTileK - 1);
            const int r  = e >> 4;
            const int kk = k0 + k;
            float w = 0.0f;
            if (r < M && kk < nc) {
                const uint32_t n = s_node[r];
                w = reach_opp_base[static_cast<size_t>(n) * nc + kk]
                  * canonical_weights[kk];
            }
            Ws[k][r] = w;
            float m = 0.0f;
            const int cc = c0 + r;
            if (cc < nc && kk < nc) {
                m = mat[static_cast<size_t>(cc) * nc + kk];
            }
            Ms[k][r] = m;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < kEqTileK; ++k) {
            const float4 a4 = *reinterpret_cast<const float4*>(&Ws[k][ty * 4]);
            const float4 b4 = *reinterpret_cast<const float4*>(&Ms[k][tx * 4]);
            const float a[4] = {a4.x, a4.y, a4.z, a4.w};
            const float b[4] = {b4.x, b4.y, b4.z, b4.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = ty * 4 + i;
        if (t >= M) continue;
        const uint32_t n = s_node[t];
        const float pot_total = pots[n];
        float rake = fminf(pot_total * rake_rate, rake_cap);
        if (rake < 0.0f) rake = 0.0f;
        const float coef = (coef_kind == 0) ? (pot_total * 0.5f - 0.5f * rake)
                                            : (-0.5f * rake);
        float* __restrict__ out =
            node_values + static_cast<size_t>(value_row[n]) * nc;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + tx * 4 + j;
            if (c >= nc) continue;
            const float v = coef * acc[i][j];
            out[c] = accumulate ? (out[c] + v) : v;
        }
    }
}

/// Raked equity showdowns on rank-blocker boards: out[c] += B·compat[c] with
/// compat[c] = Σ_{o valid, o ∩ c = ∅} reach[o] — the same card-list sum the
/// per-terminal rank_blocker_terminal_kernel adds (weights are 1 on singleton
/// iso). One block per terminal; a fixed-tree block reduction keeps it
/// deterministic. Hands blocked by the board keep the GEMM's exact 0.
__global__ void equity_rb_compat_add_kernel(
    const uint32_t* __restrict__ eq_terminal_order,
    uint32_t num_eq_terminals,
    const float* __restrict__ pots,
    const int32_t* __restrict__ matchup_idx,
    const uint32_t* __restrict__ value_row,
    const uint16_t* __restrict__ combo_bucket,
    const uint8_t* __restrict__ combo_card0,
    const uint8_t* __restrict__ combo_card1,
    const uint32_t* __restrict__ card_off,
    const uint16_t* __restrict__ card_list,
    uint32_t num_runouts,
    const float* __restrict__ reach_opp_base,
    uint16_t num_canonical,
    float rake_rate,
    float rake_cap,
    float* __restrict__ node_values)
{
    const uint32_t blk = blockIdx.x;
    if (blk >= num_eq_terminals) return;
    const uint32_t n = eq_terminal_order[blk];
    const int nc = static_cast<int>(num_canonical);
    const int tid = static_cast<int>(threadIdx.x);

    int32_t mi = matchup_idx[n];
    if (mi < 0 || static_cast<uint32_t>(mi) >= num_runouts) mi = 0;
    const uint16_t* __restrict__ mbucket = combo_bucket + static_cast<size_t>(mi) * nc;
    const float* __restrict__ reach_opp = reach_opp_base + static_cast<size_t>(n) * nc;
    float* __restrict__ out_values =
        node_values + static_cast<size_t>(value_row[n]) * nc;

    const float pot_total = pots[n];
    float rake = fminf(pot_total * rake_rate, rake_cap);
    if (rake < 0.0f) rake = 0.0f;
    const float B = -0.5f * rake;
    if (B == 0.0f) return;   // uniform across the block

    __shared__ float s_red[kRbBlock];
    float local = 0.0f;
    for (int o = tid; o < nc; o += blockDim.x) {
        if (mbucket[o] != kRbNoBucket) local += reach_opp[o];
    }
    s_red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    const float total = s_red[0];

    for (int c = tid; c < nc; c += blockDim.x) {
        if (mbucket[c] == kRbNoBucket) continue;
        const uint8_t k0 = combo_card0[c];
        const uint8_t k1 = combo_card1[c];
        float opp_total = total;
        for (uint32_t k = card_off[k0]; k < card_off[k0 + 1]; ++k) {
            const uint16_t o = card_list[k];
            if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
        }
        for (uint32_t k = card_off[k1]; k < card_off[k1 + 1]; ++k) {
            const uint16_t o = card_list[k];
            if (mbucket[o] != kRbNoBucket) opp_total -= reach_opp[o];
        }
        opp_total += reach_opp[c];   // undo the double subtraction of c
        out_values[c] += B * opp_total;
    }
}

void launch_equity_showdown_gemm(
    const EquityTile* d_tiles,
    uint32_t num_tiles,
    const uint32_t* d_eq_terminal_order,
    const float* d_pots,
    const uint32_t* d_value_row,
    const float* d_matrix_concat,
    int index_by_runout,
    const float* d_canonical_weights,
    const float* d_reach_opp_base,
    uint16_t nc,
    float rake_rate,
    float rake_cap,
    int coef_kind,
    int accumulate,
    float* d_node_values)
{
    if (num_tiles == 0 || nc == 0) return;
    const dim3 grid(num_tiles, (static_cast<unsigned>(nc) + kEqTileN - 1) / kEqTileN);
    equity_showdown_gemm_kernel<<<grid, kEqThreads>>>(
        d_tiles, d_eq_terminal_order, d_pots, d_value_row,
        d_matrix_concat, index_by_runout, d_canonical_weights,
        d_reach_opp_base, nc, rake_rate, rake_cap, coef_kind, accumulate,
        d_node_values);
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// 2026-09-11: full-board showdowns on SignedCount boards as ONE GEMM per
// traverser pass. Same tiling as equity_showdown_gemm_kernel; the matrix is
// the int8 signed pair-count table of the terminal's runout (mi), W is the
// RAW opponent reach, and the epilogue scales by win_p and 1/canonical_weight:
//   out[t][c] = win_p_t * inv_w[c] * sum_k S_mi[c][k] * reach_opp[t][k]
// which is exactly what terminal_level_kernel's category branch computes on
// these boards (payoff * valid * weight folds to +-count / |orbit_c|), and
// what the CPU signed-count kernels compute. S is antisymmetric, so the same
// product form serves both traversers with that traverser's opponent reach.
// Zero-rake boards only (the plan picks SignedCount only then); the rake
// terms are kept in the coefficient so a raked table would still be exact.
// ---------------------------------------------------------------------------
__global__ void signed_count_showdown_gemm_kernel(
    const EquityTile* __restrict__ tiles,         // eq_slot unused; mi = runout
    const uint32_t* __restrict__ terminal_order,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ value_row,
    const int8_t* __restrict__ count_concat,      // [num_runouts x nc x nc]
    const float* __restrict__ canonical_weights,
    const float* __restrict__ reach_opp_base,
    uint16_t num_canonical,
    float rake_rate,
    float rake_cap,
    float* __restrict__ node_values)
{
    const EquityTile tile = tiles[blockIdx.x];
    const int nc = static_cast<int>(num_canonical);
    const int c0 = static_cast<int>(blockIdx.y) * kEqTileN;
    const int M  = static_cast<int>(tile.t_end - tile.t_begin);
    const size_t per_table = static_cast<size_t>(nc) * nc;
    const int8_t* __restrict__ mat = count_concat + static_cast<size_t>(tile.mi) * per_table;

    __shared__ __align__(16) float Ws[kEqTileK][kEqSmemStride];   // [k][terminal]
    __shared__ __align__(16) float Ms[kEqTileK][kEqSmemStride];   // [k][hand]
    __shared__ uint32_t s_node[kEqTileM];

    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid & 15;
    const int ty = tid >> 4;
    for (int t = tid; t < kEqTileM; t += kEqThreads) {
        s_node[t] = (t < M) ? terminal_order[tile.t_begin + t] : 0u;
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }

    for (int k0 = 0; k0 < nc; k0 += kEqTileK) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e  = tid + i * kEqThreads;
            const int k  = e & (kEqTileK - 1);
            const int r  = e >> 4;
            const int kk = k0 + k;
            float w = 0.0f;
            if (r < M && kk < nc) {
                const uint32_t n = s_node[r];
                w = reach_opp_base[static_cast<size_t>(n) * nc + kk];
            }
            Ws[k][r] = w;
            float m = 0.0f;
            const int cc = c0 + r;
            if (cc < nc && kk < nc) {
                m = static_cast<float>(mat[static_cast<size_t>(cc) * nc + kk]);
            }
            Ms[k][r] = m;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < kEqTileK; ++k) {
            const float4 a4 = *reinterpret_cast<const float4*>(&Ws[k][ty * 4]);
            const float4 b4 = *reinterpret_cast<const float4*>(&Ms[k][tx * 4]);
            const float a[4] = {a4.x, a4.y, a4.z, a4.w};
            const float b[4] = {b4.x, b4.y, b4.z, b4.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = ty * 4 + i;
        if (t >= M) continue;
        const uint32_t n = s_node[t];
        const float pot_total = pots[n];
        float rake = fminf(pot_total * rake_rate, rake_cap);
        if (rake < 0.0f) rake = 0.0f;
        const float win_p = pot_total * 0.5f - rake;
        float* __restrict__ out =
            node_values + static_cast<size_t>(value_row[n]) * nc;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + tx * 4 + j;
            if (c >= nc) continue;
            const float w = canonical_weights[c];
            const float inv_w = (w > 0.0f) ? (1.0f / w) : 0.0f;
            out[c] = (win_p * acc[i][j]) * inv_w;
        }
    }
}

void launch_signed_count_showdown_gemm(
    const EquityTile* d_tiles,
    uint32_t num_tiles,
    const uint32_t* d_terminal_order,
    const float* d_pots,
    const uint32_t* d_value_row,
    const int8_t* d_count_concat,
    const float* d_canonical_weights,
    const float* d_reach_opp_base,
    uint16_t nc,
    float rake_rate,
    float rake_cap,
    float* d_node_values)
{
    if (num_tiles == 0 || nc == 0) return;
    const dim3 grid(num_tiles, (static_cast<unsigned>(nc) + kEqTileN - 1) / kEqTileN);
    signed_count_showdown_gemm_kernel<<<grid, kEqThreads>>>(
        d_tiles, d_terminal_order, d_pots, d_value_row, d_count_concat,
        d_canonical_weights, d_reach_opp_base, nc, rake_rate, rake_cap,
        d_node_values);
    CUDA_CHECK(cudaGetLastError());
}



void launch_equity_rb_compat_add(
    const uint32_t* d_eq_terminal_order,
    uint32_t num_eq_terminals,
    const float* d_pots,
    const int32_t* d_matchup_idx,
    const uint32_t* d_value_row,
    const uint16_t* d_combo_bucket,
    const uint8_t* d_combo_card0,
    const uint8_t* d_combo_card1,
    const uint32_t* d_card_off,
    const uint16_t* d_card_list,
    uint32_t num_runouts,
    const float* d_reach_opp_base,
    uint16_t nc,
    float rake_rate,
    float rake_cap,
    float* d_node_values)
{
    if (num_eq_terminals == 0) return;
    equity_rb_compat_add_kernel<<<num_eq_terminals, kRbBlock>>>(
        d_eq_terminal_order, num_eq_terminals, d_pots, d_matchup_idx,
        d_value_row, d_combo_bucket, d_combo_card0, d_combo_card1,
        d_card_off, d_card_list, num_runouts, d_reach_opp_base, nc,
        rake_rate, rake_cap, d_node_values);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2026-09-10: dedicated FOLD terminal kernel for rank-blocker boards.
//
// rank_blocker_terminal_kernel handled folds too, but a fold needs none of
// its rank machinery: the value is self_payoff × (card-compatible opponent
// mass), and that mass is total − S[card0] − S[card1] + reach[c], with S[card]
// the reach of every board-valid opponent hand holding that card. The old
// fold branch still scattered every hand into rank buckets, ran the prefix
// scan, and then walked both card lists (~92 gathers) for EVERY self hand:
// 0.39 ms per launch on the 4,820 folds of a full-menu collapsed rainbow, 28%
// of the iteration. Here the 52 per-card sums are built once per terminal
// (two 64-bit fixed-point atomics per opponent hand — integer adds commute,
// so the result is order-independent, same trick as the rank buckets), the
// total is half their sum (every hand lands in exactly two bins), and each
// self hand costs three loads. Deterministic. Dense-plan (iso) boards keep
// terminal_level_kernel's fold branch: their compatibility is fractional per
// canonical class and needs the per-original data only the dense path has.
// ============================================================================

constexpr int kFoldBlock = 256;
constexpr int kFoldNumCards = 52;   // types.h NUM_CARDS is not visible here

__global__ void fold_blocker_terminal_kernel(
    const uint8_t* __restrict__ terminal_types,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ parent_indices,
    const float* __restrict__ bet_into,
    const int32_t* __restrict__ matchup_idx,
    const uint32_t* __restrict__ value_row,
    const uint32_t* __restrict__ fold_terminal_order,
    uint32_t num_fold_terminals,
    const uint16_t* __restrict__ combo_bucket,   // [num_runouts * nc]; kRbNoBucket = blocked by the board
    const uint8_t* __restrict__ combo_card0,     // [nc]
    const uint8_t* __restrict__ combo_card1,     // [nc]
    uint32_t num_runouts,
    const float* __restrict__ reach_opp_base,    // [N * nc]
    uint16_t num_canonical,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* __restrict__ node_values)
{
    const uint32_t blk = blockIdx.x;
    if (blk >= num_fold_terminals) return;
    const uint32_t n = fold_terminal_order[blk];
    const int nc = static_cast<int>(num_canonical);
    const int tid = static_cast<int>(threadIdx.x);

    int32_t mi = matchup_idx[n];
    if (mi < 0 || static_cast<uint32_t>(mi) >= num_runouts) mi = 0;
    const uint16_t* __restrict__ mbucket = combo_bucket + static_cast<size_t>(mi) * nc;
    const float* __restrict__ reach_opp = reach_opp_base + static_cast<size_t>(n) * nc;
    float* __restrict__ out_values =
        node_values + static_cast<size_t>(value_row[n]) * nc;

    __shared__ unsigned long long s_acc[kFoldNumCards];
    __shared__ float s_card[kFoldNumCards];
    __shared__ unsigned long long s_total_q;
    for (int k = tid; k < kFoldNumCards; k += kFoldBlock) s_acc[k] = 0ull;
    __syncthreads();
    for (int o = tid; o < nc; o += kFoldBlock) {
        if (mbucket[o] == kRbNoBucket) continue;
        const float r = reach_opp[o];
        if (r == 0.0f) continue;
        const unsigned long long q = __float2ull_rn(r * kRbFixedScale);
        if (q == 0ull) continue;
        atomicAdd(&s_acc[combo_card0[o]], q);
        atomicAdd(&s_acc[combo_card1[o]], q);
    }
    __syncthreads();
    if (tid < kFoldNumCards) {
        s_card[tid] = static_cast<float>(
            __ull2double_rn(s_acc[tid]) * kRbFixedInvScale);
    }
    if (tid == 0) {
        unsigned long long t = 0ull;
        for (int k = 0; k < kFoldNumCards; ++k) t += s_acc[k];
        s_total_q = t;   // every hand was added to exactly two bins
    }
    __syncthreads();
    const float total = static_cast<float>(
        __ull2double_rn(s_total_q / 2ull) * kRbFixedInvScale);

    // Fold payoff (matches terminal_level_kernel / CPU fold_self_payoff).
    const float pot_total = pots[n];
    const uint32_t parent = parent_indices[n];
    const float matched_pot = pot_total - bet_into[parent];
    float fold_rake = fminf(matched_pot * rake_rate, rake_cap);
    if (fold_rake < 0.0f) fold_rake = 0.0f;
    const float fold_win_gain  = matched_pot * 0.5f - fold_rake;
    const float fold_lose_loss = -matched_pot * 0.5f;
    const uint8_t tt = terminal_types[n];
    const bool i_win = ((perspective == 0 && tt == TT_FOLD_IP) ||
                        (perspective == 1 && tt == TT_FOLD_OOP));
    const float self_payoff = i_win ? fold_win_gain : fold_lose_loss;

    for (int c = tid; c < nc; c += kFoldBlock) {
        if (mbucket[c] == kRbNoBucket) { out_values[c] = 0.0f; continue; }
        // c sits in both of its card bins: subtracted twice, belongs out once.
        const float compat = total - s_card[combo_card0[c]] - s_card[combo_card1[c]]
                           + reach_opp[c];
        out_values[c] = self_payoff * compat;
    }
}

void launch_fold_blocker_terminals(
    const uint8_t* d_terminal_types,
    const float* d_pots,
    const uint32_t* d_parent_indices,
    const float* d_bet_into,
    const int32_t* d_matchup_idx,
    const uint32_t* d_value_row,
    const uint32_t* d_fold_terminal_order,
    uint32_t num_fold_terminals,
    const uint16_t* d_combo_bucket,
    const uint8_t* d_combo_card0,
    const uint8_t* d_combo_card1,
    uint32_t num_runouts,
    const float* d_reach_opp_base,
    uint16_t nc,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* d_node_values)
{
    if (num_fold_terminals == 0) return;
    fold_blocker_terminal_kernel<<<num_fold_terminals, kFoldBlock>>>(
        d_terminal_types, d_pots, d_parent_indices, d_bet_into, d_matchup_idx,
        d_value_row, d_fold_terminal_order, num_fold_terminals,
        d_combo_bucket, d_combo_card0, d_combo_card1, num_runouts,
        d_reach_opp_base, nc, perspective, rake_rate, rake_cap, d_node_values);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2026-09-11: FOLD terminals on iso (SignedCount) boards. GPU twin of
// fold_blocker::fold_dense_precomputed: a canonical class holds several
// originals, so the per-card opponent mass is built over every board-valid
// original of every opponent hand (two 64-bit fixed-point atomics each,
// order-independent like the rank-blocker kernel), and each self hand sums
// (total - S[card0] - S[card1] + reach_opp[c]) over ITS board-valid originals
// and divides by its orbit size — exactly the dense category kernel's
// payoff * valid * weight for a fold, without streaming the nc^2 valid table
// once per terminal.
// ============================================================================
__global__ void iso_fold_terminal_kernel(
    const uint8_t* __restrict__ terminal_types,
    const float* __restrict__ pots,
    const uint32_t* __restrict__ parent_indices,
    const float* __restrict__ bet_into,
    const int32_t* __restrict__ matchup_idx,
    const uint32_t* __restrict__ value_row,
    const uint32_t* __restrict__ fold_terminal_order,
    uint32_t num_fold_terminals,
    const uint32_t* __restrict__ orig_off,        // [nc + 1] CSR into the original lists
    const uint8_t* __restrict__ orig_card0,       // [total originals]
    const uint8_t* __restrict__ orig_card1,       // [total originals]
    const float* __restrict__ orbit_denom,        // [nc] max(1, |originals|)
    const unsigned long long* __restrict__ board_masks,  // [num_runouts]
    uint32_t num_runouts,
    const float* __restrict__ reach_opp_base,     // [N * nc]
    uint16_t num_canonical,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* __restrict__ node_values)
{
    const uint32_t blk = blockIdx.x;
    if (blk >= num_fold_terminals) return;
    const uint32_t n = fold_terminal_order[blk];
    const int nc = static_cast<int>(num_canonical);
    const int tid = static_cast<int>(threadIdx.x);

    int32_t mi = matchup_idx[n];
    if (mi < 0 || static_cast<uint32_t>(mi) >= num_runouts) mi = 0;
    const unsigned long long bm = board_masks[mi];
    const float* __restrict__ reach_opp = reach_opp_base + static_cast<size_t>(n) * nc;
    float* __restrict__ out_values =
        node_values + static_cast<size_t>(value_row[n]) * nc;

    __shared__ unsigned long long s_acc[kFoldNumCards];
    __shared__ float s_card[kFoldNumCards];
    __shared__ unsigned long long s_total_q;
    for (int k = tid; k < kFoldNumCards; k += kFoldBlock) s_acc[k] = 0ull;
    __syncthreads();
    for (int o = tid; o < nc; o += kFoldBlock) {
        const float r = reach_opp[o];
        if (r == 0.0f) continue;
        const unsigned long long q = __float2ull_rn(r * kRbFixedScale);
        if (q == 0ull) continue;
        const uint32_t pb = orig_off[o], pe = orig_off[o + 1];
        for (uint32_t p = pb; p < pe; ++p) {
            const int c0 = orig_card0[p], c1 = orig_card1[p];
            if (((1ull << c0) | (1ull << c1)) & bm) continue;
            atomicAdd(&s_acc[c0], q);
            atomicAdd(&s_acc[c1], q);
        }
    }
    __syncthreads();
    if (tid < kFoldNumCards) {
        s_card[tid] = static_cast<float>(
            __ull2double_rn(s_acc[tid]) * kRbFixedInvScale);
    }
    if (tid == 0) {
        unsigned long long t = 0ull;
        for (int k = 0; k < kFoldNumCards; ++k) t += s_acc[k];
        s_total_q = t;   // every board-valid original landed in exactly two bins
    }
    __syncthreads();
    const float total = static_cast<float>(
        __ull2double_rn(s_total_q / 2ull) * kRbFixedInvScale);

    // Fold payoff (matches terminal_level_kernel / CPU fold_self_payoff).
    const float pot_total = pots[n];
    const uint32_t parent = parent_indices[n];
    const float matched_pot = pot_total - bet_into[parent];
    float fold_rake = fminf(matched_pot * rake_rate, rake_cap);
    if (fold_rake < 0.0f) fold_rake = 0.0f;
    const float fold_win_gain  = matched_pot * 0.5f - fold_rake;
    const float fold_lose_loss = -matched_pot * 0.5f;
    const uint8_t tt = terminal_types[n];
    const bool i_win = ((perspective == 0 && tt == TT_FOLD_IP) ||
                        (perspective == 1 && tt == TT_FOLD_OOP));
    const float self_payoff = i_win ? fold_win_gain : fold_lose_loss;

    for (int c = tid; c < nc; c += kFoldBlock) {
        const float r_c = reach_opp[c];
        float acc = 0.0f;
        const uint32_t pb = orig_off[c], pe = orig_off[c + 1];
        for (uint32_t p = pb; p < pe; ++p) {
            const int c0 = orig_card0[p], c1 = orig_card1[p];
            if (((1ull << c0) | (1ull << c1)) & bm) continue;
            // the identical original sits in both card bins: subtracted twice,
            // belongs out once (fold_blocker::combo_value)
            acc += ((total - s_card[c0]) - s_card[c1]) + r_c;
        }
        out_values[c] = self_payoff * (acc / orbit_denom[c]);
    }
}

void launch_iso_fold_terminals(
    const uint8_t* d_terminal_types,
    const float* d_pots,
    const uint32_t* d_parent_indices,
    const float* d_bet_into,
    const int32_t* d_matchup_idx,
    const uint32_t* d_value_row,
    const uint32_t* d_fold_terminal_order,
    uint32_t num_fold_terminals,
    const uint32_t* d_orig_off,
    const uint8_t* d_orig_card0,
    const uint8_t* d_orig_card1,
    const float* d_orbit_denom,
    const unsigned long long* d_board_masks,
    uint32_t num_runouts,
    const float* d_reach_opp_base,
    uint16_t nc,
    int perspective,
    float rake_rate,
    float rake_cap,
    float* d_node_values)
{
    if (num_fold_terminals == 0) return;
    iso_fold_terminal_kernel<<<num_fold_terminals, kFoldBlock>>>(
        d_terminal_types, d_pots, d_parent_indices, d_bet_into, d_matchup_idx,
        d_value_row, d_fold_terminal_order, num_fold_terminals,
        d_orig_off, d_orig_card0, d_orig_card1, d_orbit_denom, d_board_masks,
        num_runouts, d_reach_opp_base, nc, perspective, rake_rate, rake_cap,
        d_node_values);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace deepsolver
