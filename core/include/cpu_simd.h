/**
 * @file cpu_simd.h
 * @brief Runtime-dispatched SIMD kernel table for the CPU CFR backend.
 *
 * v1.4.0 — Phase 1 refactor:
 *   This header used to contain inline AVX2 intrinsics gated by a compile-
 *   time `#if DEEPSOLVER_HAS_AVX2`. That forced the whole binary to be built
 *   with `/arch:AVX2`, which in turn made the EXE refuse to start on any
 *   pre-Haswell (2013) Intel / pre-Excavator (2015) AMD CPU — exactly the
 *   "no GPU, old hardware" users this CPU optimization push is meant to
 *   serve. The runtime CPUID dispatch never got a chance to run because
 *   `main()` itself contained AVX2 opcodes.
 *
 *   The fix: move every SIMD implementation into translation units of its
 *   own (`cpu_kernels_scalar.cpp` and `cpu_kernels_avx2.cpp`). Only the
 *   AVX2 `.cpp` is compiled with `/arch:AVX2`. This header just exports
 *   the dispatch table — `Kernels` is a plain struct of function pointers
 *   resolved once at startup based on CPUID.
 *
 *   Cost model: one indirect call per kernel invocation (~3-5 ns), versus
 *   ~200 ns of work for a typical nc=1300 inner loop. Net overhead < 3%.
 *
 *   Backwards-compat shim: previously inline-callable helpers like
 *   `cpu_simd::vec_axpy(...)` are still callable with the same signature,
 *   but now route through the dispatch table.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace deepsolver::cpu_simd {

// ---------------------------------------------------------------------------
// Trivial helpers stay in the header (no SIMD distinction needed).
// ---------------------------------------------------------------------------

struct ActiveRun {
    uint16_t start;
    uint16_t count;
};

inline void vec_set_zero(float* x, std::size_t n) {
    std::memset(x, 0, sizeof(float) * n);
}

inline void vec_copy(float* dst, const float* src, std::size_t n) {
    std::memcpy(dst, src, sizeof(float) * n);
}

// ---------------------------------------------------------------------------
// Kernel table — populated at startup with either the scalar or the AVX2
// implementation, decided by CPUID. All hot-loop work goes through here.
// ---------------------------------------------------------------------------

struct Kernels {
    void  (*vec_mul_in_place)(float* x, const float* y, std::size_t n);
    void  (*vec_mul)(float* dst, const float* a, const float* b, std::size_t n);
    void  (*vec_scale_in_place)(float* x, float s, std::size_t n);
    void  (*vec_add_in_place)(float* dst, const float* src, std::size_t n);
    void  (*vec_axpy)(float* dst, float a, const float* src, std::size_t n);
    void  (*vec_fmadd)(float* dst, const float* a, const float* b, std::size_t n);
    void  (*vec_dcfr_discount)(float* x, float pos, float neg, std::size_t n);
    void  (*vec_pos_add)(float* pos_sum, const float* regret, std::size_t n);
    void  (*vec_pos_normalize)(float* strat, const float* regret,
                               const float* inv_pos_sum,
                               const float* uniform_or_zero, std::size_t n);
    void  (*vec_pos_normalize2)(float* strat0, float* strat1,
                                const float* regret0, const float* regret1,
                                std::size_t n);
    void  (*vec_regret_update)(float* regret, const float* action_val,
                               const float* node_val, std::size_t n);
    void  (*vec_decay_add)(float* dst, float decay, const float* src, std::size_t n);
    void  (*vec_reach_weighted_strat_sum)(float* dst, float sw,
                                          const float* reach, const float* strat,
                                          std::size_t n);
    float (*showdown_oop_inner)(const float* ev_row, const float* valid_row,
                                const float* opp_reach_w,
                                float win_p, float lose_p, float tie_p,
                                std::size_t n);
    void  (*showdown_ip_step)(float* out_vals, const float* ev_row,
                              const float* valid_row,
                              float rw_ci,
                              float win_p, float lose_p, float tie_p,
                              std::size_t n);
    float (*dot_valid_reach)(const float* valid_row, const float* opp_reach_w,
                             std::size_t n);
    float (*dot_valid_reach_active)(const float* valid_row,
                                    const float* opp_reach_w,
                                    const uint16_t* active_indices,
                                    std::size_t active_count);
    float (*dot_valid_reach_active_runs)(const float* valid_row,
                                         const float* opp_reach_w,
                                         const ActiveRun* active_runs,
                                         std::size_t run_count);
    void  (*fold_ip_step)(float* out_vals, const float* valid_row,
                          float rw_ci, const uint8_t* skip_mask,
                          std::size_t n);
    void  (*fold_ip_step_active)(float* out_vals, const float* valid_row,
                                 float rw_ci,
                                 const uint16_t* active_indices,
                                 std::size_t active_count);
    void  (*fold_ip_step_active_runs)(float* out_vals, const float* valid_row,
                                      float rw_ci,
                                      const ActiveRun* active_runs,
                                      std::size_t run_count);

    // v1.8.0 P3-8 spike: full-row variants that fuse the per-c outer loop
    // into the kernel itself. The per-call versions above made
    // evaluate_terminal() pay (a) one dispatch-table lookup, (b) one set of
    // SIMD-constant materializations (vwin/vlose/vtie/v05/vn05), and (c)
    // one function-call prologue/epilogue per c row — that's nc=~1200 of
    // each per terminal × 2 traversers × thousands of terminals.
    //
    // The _full variants take the full ev/valid matrices and write into the
    // entire `out` array in one call. SIMD constants are hoisted out of
    // the c loop. Same numeric output as calling the per-c kernels in
    // sequence — the parity test gates this.
    //
    // Layout: ev/valid_matrix is row-major [c × n + j]. out is n floats.
    //
    // v1.8.1+ optional sparse skip: when `skip_mask` is non-null, output
    // combos where skip_mask[c] != 0 are left at 0 and skipped.
    // The caller is responsible for guaranteeing that skipping is safe —
    // i.e. that out[c] either won't propagate or will only be multiplied
    // by 0 at every ancestor. Safe usage: combos that are 0-reach from
    // root (out-of-range), which propagate as 0 everywhere. UNSAFE: combos
    // with 0-reach at THIS terminal but >0 reach at some ancestor — the
    // ancestor's regret update consumes value[c] and changing it to 0
    // distorts CFR convergence.
    //
    // Pass nullptr for the skip_mask to disable (dense path), preserving
    // the original v1.8.0 behavior for callers without a pre-validated mask.
    //
    // v1.8.2 A2 encoding: category_matrix is a per-cell uint8_t in
    // {0=invalid, 1=win, 2=lose, 3=tie}, pre-thresholded at precompute time
    // from the continuous ev. valid_matrix stays as f32 (exact validity
    // weight). 5 bytes/cell vs the prior 8 bytes/cell — bench shows ~1.97x
    // throughput on cold-cache workloads and bit-identical results.
    void  (*showdown_oop_full)(const uint8_t* category_matrix,
                               const float* valid_matrix,
                               const float* opp_reach_w,
                               const uint8_t* skip_mask,
                               float* out, std::size_t n,
                               float win_p, float lose_p, float tie_p);
    void  (*showdown_ip_full)(const uint8_t* category_matrix,
                              const float* valid_matrix,
                              const float* opp_reach_w,
                              const uint8_t* skip_mask,
                              float* out, std::size_t n,
                              float win_p, float lose_p, float tie_p);
    void  (*showdown_oop_signed_zero_rake)(const float* signed_valid_matrix,
                                           const float* opp_reach_w,
                                           const uint8_t* skip_mask,
                                           float* out, std::size_t n,
                                           float win_p);
    void  (*showdown_ip_signed_zero_rake)(const float* signed_valid_matrix,
                                           const float* opp_reach_w,
                                           const uint8_t* skip_mask,
                                           float* out, std::size_t n,
                                           float win_p);
    void  (*showdown_oop_signed_count_zero_rake)(const int8_t* signed_count_matrix,
                                                 const float* opp_reach,
                                                 const float* inv_weights,
                                                 const uint8_t* skip_mask,
                                                 float* out, std::size_t n,
                                                 float win_p);
    void  (*showdown_ip_signed_count_zero_rake)(const int8_t* signed_count_matrix,
                                                const float* opp_reach,
                                                const float* inv_weights,
                                                const uint8_t* skip_mask,
                                                float* out, std::size_t n,
                                                float win_p);

    // Active-index variants for root-range sparse terminal output. The
    // active list contains canonical output combos whose root reach is
    // non-zero for the traverser. Kernels zero the whole output row first,
    // then compute only those indices. This is stricter than skip_mask:
    // dense/mid-density ranges stay on the contiguous kernels, while very
    // sparse ranges avoid scanning output combos that cannot propagate.
    void  (*showdown_oop_full_active)(const uint8_t* category_matrix,
                                      const float* valid_matrix,
                                      const float* opp_reach_w,
                                      const uint16_t* active_indices,
                                      std::size_t active_count,
                                      float* out, std::size_t n,
                                      float win_p, float lose_p, float tie_p,
                                      bool clear_full_output);
    void  (*showdown_ip_full_active)(const uint8_t* category_matrix,
                                     const float* valid_matrix,
                                     const float* opp_reach_w,
                                     const uint16_t* active_indices,
                                     std::size_t active_count,
                                     float* out, std::size_t n,
                                     float win_p, float lose_p, float tie_p,
                                     bool clear_full_output);
    void  (*showdown_oop_full_active_runs)(const uint8_t* category_matrix,
                                           const float* valid_matrix,
                                           const float* opp_reach_w,
                                           const ActiveRun* active_runs,
                                           std::size_t run_count,
                                           float* out, std::size_t n,
                                           float win_p, float lose_p, float tie_p,
                                           bool clear_full_output);
    void  (*showdown_oop_full_active_opp_blocks)(
                                           const uint8_t* category_matrix,
                                           const float* valid_matrix,
                                           const float* opp_reach_w,
                                           const uint16_t* self_active_indices,
                                           std::size_t self_active_count,
                                           const ActiveRun* opp_blocks,
                                           std::size_t opp_block_count,
                                           float* out, std::size_t n,
                                           float win_p, float lose_p, float tie_p,
                                           bool clear_full_output);
    void  (*showdown_ip_full_active_opp_blocks)(
                                          const uint8_t* category_matrix,
                                          const float* valid_matrix,
                                          const float* opp_reach_w,
                                          const uint16_t* self_active_indices,
                                          std::size_t self_active_count,
                                          const ActiveRun* opp_blocks,
                                          std::size_t opp_block_count,
                                          float* out, std::size_t n,
                                          float win_p, float lose_p, float tie_p,
                                          bool clear_full_output);
    void  (*showdown_ip_full_active_runs)(const uint8_t* category_matrix,
                                          const float* valid_matrix,
                                          const float* opp_reach_w,
                                          const ActiveRun* active_runs,
                                          std::size_t run_count,
                                          float* out, std::size_t n,
                                          float win_p, float lose_p, float tie_p,
                                          bool clear_full_output);
    void  (*showdown_oop_full_active2)(const uint8_t* category_matrix,
                                       const float* valid_matrix,
                                       const float* opp_reach_w,
                                       const uint16_t* self_active_indices,
                                       std::size_t self_active_count,
                                       const uint16_t* opp_active_indices,
                                       std::size_t opp_active_count,
                                       float* out, std::size_t n,
                                       float win_p, float lose_p, float tie_p);
    void  (*showdown_ip_full_active2)(const uint8_t* category_matrix,
                                      const float* valid_matrix,
                                      const float* opp_reach_w,
                                      const uint16_t* self_active_indices,
                                      std::size_t self_active_count,
                                      const uint16_t* opp_active_indices,
                                      std::size_t opp_active_count,
                                      float* out, std::size_t n,
                                      float win_p, float lose_p, float tie_p);

    // POST_OPTIMIZATION_REVIEW Sec 4.3 Phase 2 (kernel-level batching):
    // process M terminal-OOP showdowns that all share the same matchup
    // table, in a single kernel invocation. Each (cat_row, valid_row) is
    // streamed exactly once per c (outer) and reused across all M terminals
    // — the per-call kernel re-streams the matrix M times. On monotone with
    // ~245 terminals per group, the theoretical matrix-bandwidth saving is
    // ~245x; in practice the real saving is bounded by L3 reuse the natural
    // BFS order already gives.
    //
    // Layout / contract:
    //   cat_matrix, valid_matrix : shared [n × n] row-major
    //   skip_mask                : optional [n], applied to ALL terminals
    //                              (root-out-of-range is matchup-invariant)
    //   opp_reach_w_array[t]     : terminal t's opp_reach × canonical_weight
    //                              vector, length n
    //   out_array[t]             : terminal t's output, length n
    //   c_lo, c_hi               : c-range to process (allows the caller to
    //                              parallelize the outer loop across threads
    //                              without each thread re-streaming the same
    //                              matrix rows)
    //
    // Numerically equivalent to calling showdown_oop_full(...) M times with
    // each terminal's (opp_reach_w, win_p, lose_p, tie_p, out) — verified by
    // the per-kernel parity test.
    void (*showdown_oop_full_batch)(
        const uint8_t* category_matrix, const float* valid_matrix,
        std::size_t n,
        std::size_t num_terminals,
        const float* const* opp_reach_w_array,
        const uint8_t* skip_mask,
        float* const* out_array,
        const float* win_p_array,
        const float* lose_p_array,
        const float* tie_p_array,
        std::size_t c_lo, std::size_t c_hi);
};

// Defined in cpu_kernels_scalar.cpp / cpu_kernels_avx2.cpp.
extern const Kernels scalar_kernels;
extern const Kernels avx2_kernels;

enum class SimdMode {
    Scalar,
    Avx2,
};

// Override CPUID detection. Defaults to AUTO which respects the hardware.
// Set before the first kernel use (typically before backend creation).
enum class SimdPolicy {
    Auto,        // Use AVX2 when CPUID + OS support it, else Scalar.
    ForceScalar, // Diagnostic / debugging — force scalar even on AVX2 hosts.
    ForceAvx2,   // Power user — abort startup if AVX2 isn't available.
};

void  set_policy(SimdPolicy policy);
SimdMode active_mode();
const Kernels& kernels();
const char* mode_label();   // "scalar" or "avx2", lifetime forever

// CPUID helpers (exposed for diagnostics / tests).
bool cpuid_supports_avx2_fma_os();

// ---------------------------------------------------------------------------
// Inline call wrappers — keep the previous API ergonomic for cpu_backend.h.
// Each wrapper does one indirect call through the dispatch table. The
// compiler caches the kernels() result in a register across hot loops, so
// the per-call overhead in practice is just the indirect call itself.
// ---------------------------------------------------------------------------

inline void vec_mul_in_place(float* x, const float* y, std::size_t n) {
    kernels().vec_mul_in_place(x, y, n);
}
inline void vec_mul(float* dst, const float* a, const float* b, std::size_t n) {
    kernels().vec_mul(dst, a, b, n);
}
inline void vec_scale_in_place(float* x, float s, std::size_t n) {
    kernels().vec_scale_in_place(x, s, n);
}
inline void vec_add_in_place(float* dst, const float* src, std::size_t n) {
    kernels().vec_add_in_place(dst, src, n);
}
inline void vec_axpy(float* dst, float a, const float* src, std::size_t n) {
    kernels().vec_axpy(dst, a, src, n);
}
inline void vec_fmadd(float* dst, const float* a, const float* b, std::size_t n) {
    kernels().vec_fmadd(dst, a, b, n);
}
inline void vec_dcfr_discount(float* x, float pos, float neg, std::size_t n) {
    kernels().vec_dcfr_discount(x, pos, neg, n);
}
inline void vec_pos_add(float* pos_sum, const float* regret, std::size_t n) {
    kernels().vec_pos_add(pos_sum, regret, n);
}
inline void vec_pos_normalize(
    float* strat, const float* regret,
    const float* inv_pos_sum, const float* uniform_or_zero, std::size_t n)
{
    kernels().vec_pos_normalize(strat, regret, inv_pos_sum, uniform_or_zero, n);
}
inline void vec_pos_normalize2(
    float* strat0, float* strat1,
    const float* regret0, const float* regret1, std::size_t n)
{
    kernels().vec_pos_normalize2(strat0, strat1, regret0, regret1, n);
}
inline void vec_regret_update(
    float* regret, const float* action_val, const float* node_val, std::size_t n)
{
    kernels().vec_regret_update(regret, action_val, node_val, n);
}
inline void vec_decay_add(float* dst, float decay, const float* src, std::size_t n) {
    kernels().vec_decay_add(dst, decay, src, n);
}
inline void vec_reach_weighted_strat_sum(
    float* dst, float sw, const float* reach, const float* strat, std::size_t n)
{
    kernels().vec_reach_weighted_strat_sum(dst, sw, reach, strat, n);
}
inline float showdown_oop_inner(
    const float* ev_row, const float* valid_row, const float* opp_reach_w,
    float win_p, float lose_p, float tie_p, std::size_t n)
{
    return kernels().showdown_oop_inner(
        ev_row, valid_row, opp_reach_w, win_p, lose_p, tie_p, n);
}
inline void showdown_ip_step(
    float* out_vals, const float* ev_row, const float* valid_row,
    float rw_ci, float win_p, float lose_p, float tie_p, std::size_t n)
{
    kernels().showdown_ip_step(
        out_vals, ev_row, valid_row, rw_ci, win_p, lose_p, tie_p, n);
}
inline float dot_valid_reach(
    const float* valid_row, const float* opp_reach_w, std::size_t n)
{
    return kernels().dot_valid_reach(valid_row, opp_reach_w, n);
}
inline float dot_valid_reach_active(
    const float* valid_row, const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count)
{
    return kernels().dot_valid_reach_active(
        valid_row, opp_reach_w, active_indices, active_count);
}
inline float dot_valid_reach_active_runs(
    const float* valid_row, const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count)
{
    return kernels().dot_valid_reach_active_runs(
        valid_row, opp_reach_w, active_runs, run_count);
}
inline void fold_ip_step(
    float* out_vals, const float* valid_row, float rw_ci,
    const uint8_t* skip_mask, std::size_t n)
{
    kernels().fold_ip_step(out_vals, valid_row, rw_ci, skip_mask, n);
}
inline void fold_ip_step_active(
    float* out_vals, const float* valid_row, float rw_ci,
    const uint16_t* active_indices, std::size_t active_count)
{
    kernels().fold_ip_step_active(
        out_vals, valid_row, rw_ci, active_indices, active_count);
}
inline void fold_ip_step_active_runs(
    float* out_vals, const float* valid_row, float rw_ci,
    const ActiveRun* active_runs, std::size_t run_count)
{
    kernels().fold_ip_step_active_runs(
        out_vals, valid_row, rw_ci, active_runs, run_count);
}

inline void showdown_oop_full(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_oop_full(category_matrix, valid_matrix, opp_reach_w,
                                 skip_mask, out, n, win_p, lose_p, tie_p);
}
inline void showdown_ip_full(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_ip_full(category_matrix, valid_matrix, opp_reach_w,
                                skip_mask, out, n, win_p, lose_p, tie_p);
}
inline void showdown_oop_signed_zero_rake(
    const float* signed_valid_matrix, const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    kernels().showdown_oop_signed_zero_rake(
        signed_valid_matrix, opp_reach_w, skip_mask, out, n, win_p);
}
inline void showdown_ip_signed_zero_rake(
    const float* signed_valid_matrix, const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    kernels().showdown_ip_signed_zero_rake(
        signed_valid_matrix, opp_reach_w, skip_mask, out, n, win_p);
}
inline void showdown_oop_signed_count_zero_rake(
    const int8_t* signed_count_matrix, const float* opp_reach,
    const float* inv_weights, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    kernels().showdown_oop_signed_count_zero_rake(
        signed_count_matrix, opp_reach, inv_weights, skip_mask, out, n, win_p);
}
inline void showdown_ip_signed_count_zero_rake(
    const int8_t* signed_count_matrix, const float* opp_reach,
    const float* inv_weights, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    kernels().showdown_ip_signed_count_zero_rake(
        signed_count_matrix, opp_reach, inv_weights, skip_mask, out, n, win_p);
}
inline void showdown_oop_full_active(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_oop_full_active(
        category_matrix, valid_matrix, opp_reach_w,
        active_indices, active_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_ip_full_active(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_ip_full_active(
        category_matrix, valid_matrix, opp_reach_w,
        active_indices, active_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_oop_full_active_runs(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_oop_full_active_runs(
        category_matrix, valid_matrix, opp_reach_w,
        active_runs, run_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_oop_full_active_opp_blocks(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const ActiveRun* opp_blocks, std::size_t opp_block_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_oop_full_active_opp_blocks(
        category_matrix, valid_matrix, opp_reach_w,
        self_active_indices, self_active_count,
        opp_blocks, opp_block_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_ip_full_active_opp_blocks(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const ActiveRun* opp_blocks, std::size_t opp_block_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_ip_full_active_opp_blocks(
        category_matrix, valid_matrix, opp_reach_w,
        self_active_indices, self_active_count,
        opp_blocks, opp_block_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_ip_full_active_runs(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output = true)
{
    kernels().showdown_ip_full_active_runs(
        category_matrix, valid_matrix, opp_reach_w,
        active_runs, run_count, out, n, win_p, lose_p, tie_p,
        clear_full_output);
}
inline void showdown_oop_full_active2(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const uint16_t* opp_active_indices, std::size_t opp_active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_oop_full_active2(
        category_matrix, valid_matrix, opp_reach_w,
        self_active_indices, self_active_count,
        opp_active_indices, opp_active_count,
        out, n, win_p, lose_p, tie_p);
}
inline void showdown_ip_full_active2(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const uint16_t* opp_active_indices, std::size_t opp_active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_ip_full_active2(
        category_matrix, valid_matrix, opp_reach_w,
        self_active_indices, self_active_count,
        opp_active_indices, opp_active_count,
        out, n, win_p, lose_p, tie_p);
}
inline void showdown_oop_full_batch(
    const uint8_t* category_matrix, const float* valid_matrix,
    std::size_t n,
    std::size_t num_terminals,
    const float* const* opp_reach_w_array,
    const uint8_t* skip_mask,
    float* const* out_array,
    const float* win_p_array,
    const float* lose_p_array,
    const float* tie_p_array,
    std::size_t c_lo, std::size_t c_hi)
{
    kernels().showdown_oop_full_batch(
        category_matrix, valid_matrix, n, num_terminals,
        opp_reach_w_array, skip_mask, out_array,
        win_p_array, lose_p_array, tie_p_array, c_lo, c_hi);
}

}  // namespace deepsolver::cpu_simd
