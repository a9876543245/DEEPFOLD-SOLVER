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
#include <cstring>

namespace deepsolver::cpu_simd {

// ---------------------------------------------------------------------------
// Trivial helpers stay in the header (no SIMD distinction needed).
// ---------------------------------------------------------------------------

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
    void  (*vec_scale_in_place)(float* x, float s, std::size_t n);
    void  (*vec_add_in_place)(float* dst, const float* src, std::size_t n);
    void  (*vec_axpy)(float* dst, float a, const float* src, std::size_t n);
    void  (*vec_fmadd)(float* dst, const float* a, const float* b, std::size_t n);
    void  (*vec_dcfr_discount)(float* x, float pos, float neg, std::size_t n);
    void  (*vec_pos_add)(float* pos_sum, const float* regret, std::size_t n);
    void  (*vec_pos_normalize)(float* strat, const float* regret,
                               const float* inv_pos_sum,
                               const float* uniform_or_zero, std::size_t n);
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
    void  (*fold_ip_step)(float* out_vals, const float* valid_row,
                          float rw_ci, std::size_t n);

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
    void  (*showdown_oop_full)(const float* ev_matrix, const float* valid_matrix,
                               const float* opp_reach_w,
                               float* out, std::size_t n,
                               float win_p, float lose_p, float tie_p);
    void  (*showdown_ip_full)(const float* ev_matrix, const float* valid_matrix,
                              const float* opp_reach_w,
                              float* out, std::size_t n,
                              float win_p, float lose_p, float tie_p);
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
inline void fold_ip_step(
    float* out_vals, const float* valid_row, float rw_ci, std::size_t n)
{
    kernels().fold_ip_step(out_vals, valid_row, rw_ci, n);
}

inline void showdown_oop_full(
    const float* ev_matrix, const float* valid_matrix,
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_oop_full(ev_matrix, valid_matrix, opp_reach_w, out, n,
                                 win_p, lose_p, tie_p);
}
inline void showdown_ip_full(
    const float* ev_matrix, const float* valid_matrix,
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    kernels().showdown_ip_full(ev_matrix, valid_matrix, opp_reach_w, out, n,
                                win_p, lose_p, tie_p);
}

}  // namespace deepsolver::cpu_simd
