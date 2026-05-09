/**
 * @file cpu_kernels_scalar.cpp
 * @brief Scalar baseline implementations of the CFR SIMD kernels.
 *
 * Compiled WITHOUT /arch:AVX2 — these run on every x86-64 CPU, no exception.
 * Used as the fallback when CPUID reports no AVX2 support, or when the
 * user forces `--cpu-simd scalar` for debugging / parity testing.
 *
 * Keep this file's instruction set baseline at SSE2 (the x86-64 ABI floor).
 * Even GCC `-O3` will only auto-vectorize up to SSE2 here, so the inner
 * loops emit 4-wide float ops at most. AVX2 yields 8-wide and gets ~2× on
 * the same loop — see cpu_kernels_avx2.cpp.
 */

#include "cpu_simd.h"

#include <cstddef>

namespace deepsolver::cpu_simd::scalar_impl {

static void vec_mul_in_place(float* x, const float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] *= y[i];
}

static void vec_scale_in_place(float* x, float s, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] *= s;
}

static void vec_add_in_place(float* dst, const float* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] += src[i];
}

static void vec_axpy(float* dst, float a, const float* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] += a * src[i];
}

static void vec_fmadd(float* dst, const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] += a[i] * b[i];
}

static void vec_dcfr_discount(float* x, float pos, float neg, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] *= (x[i] > 0.0f) ? pos : neg;
}

static void vec_pos_add(float* pos_sum, const float* regret, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        pos_sum[i] += (regret[i] > 0.0f) ? regret[i] : 0.0f;
}

static void vec_pos_normalize(
    float* strat, const float* regret,
    const float* inv_pos_sum, const float* uniform_or_zero, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float rp = (regret[i] > 0.0f) ? regret[i] : 0.0f;
        strat[i] = rp * inv_pos_sum[i] + uniform_or_zero[i];
    }
}

static void vec_regret_update(
    float* regret, const float* action_val, const float* node_val, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        regret[i] += action_val[i] - node_val[i];
}

static void vec_decay_add(float* dst, float decay, const float* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = dst[i] * decay + src[i];
}

static void vec_reach_weighted_strat_sum(
    float* dst, float sw, const float* reach, const float* strat, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) dst[i] += sw * reach[i] * strat[i];
}

static float showdown_oop_inner(
    const float* ev_row, const float* valid_row, const float* opp_reach_w,
    float win_p, float lose_p, float tie_p, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float ev = ev_row[i];
        float valid = valid_row[i];
        if (valid <= 0.0f) continue;
        float p;
        if (ev >  0.5f) p = win_p;
        else if (ev < -0.5f) p = lose_p;
        else                 p = tie_p;
        sum += opp_reach_w[i] * valid * p;
    }
    return sum;
}

static void showdown_ip_step(
    float* out_vals, const float* ev_row, const float* valid_row,
    float rw_ci, float win_p, float lose_p, float tie_p, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float ev = ev_row[i];
        float valid = valid_row[i];
        if (valid <= 0.0f) continue;
        float p;
        if (ev >  0.5f) p = lose_p;   // OOP wins → IP loses
        else if (ev < -0.5f) p = win_p;
        else                 p = tie_p;
        out_vals[i] += rw_ci * valid * p;
    }
}

static float dot_valid_reach(
    const float* valid_row, const float* opp_reach_w, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += valid_row[i] * opp_reach_w[i];
    return sum;
}

static void fold_ip_step(
    float* out_vals, const float* valid_row, float rw_ci, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) out_vals[i] += rw_ci * valid_row[i];
}

// v1.8.2 A2 encoding: takes a pre-thresholded category byte per cell rather
// than re-bucketing the continuous ev. See cpu_simd.h header + cpu_kernels_avx2.cpp
// for details. Categories: 0=invalid, 1=win, 2=lose, 3=tie.
//
// v1.8.1+ optional skip_mask: rows where skip_mask[c] != 0 set out[c]=0 and
// skip the inner reduction. Caller must ensure skipping is safe (out[c] won't
// be consumed by ancestor regret updates) — see cpu_simd.h header doc.
static void showdown_oop_full(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    for (std::size_t c = 0; c < n; ++c) {
        if (skip_mask && skip_mask[c]) {
            out[c] = 0.0f;
            continue;
        }
        const uint8_t* cat_row   = category_matrix + c * n;
        const float*   valid_row = valid_matrix    + c * n;
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            uint8_t cat = cat_row[i];
            if (cat == 0) continue;
            float p;
            if      (cat == 1) p = win_p;
            else if (cat == 2) p = lose_p;
            else               p = tie_p;
            sum += opp_reach_w[i] * valid_row[i] * p;
        }
        out[c] = sum;
    }
}

static void showdown_ip_full(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    // Initialize output, then accumulate. This matches the order of the
    // per-call version (vec_set_zero followed by per-ci showdown_ip_step
    // calls, with rw_ci=0 ones skipped).
    for (std::size_t i = 0; i < n; ++i) out[i] = 0.0f;
    for (std::size_t ci = 0; ci < n; ++ci) {
        float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        const uint8_t* cat_row   = category_matrix + ci * n;
        const float*   valid_row = valid_matrix    + ci * n;
        for (std::size_t i = 0; i < n; ++i) {
            uint8_t cat = cat_row[i];
            if (cat == 0) continue;
            float p;
            if      (cat == 1) p = lose_p;   // ev > 0.5 → IP loses
            else if (cat == 2) p = win_p;    // ev < -0.5 → IP wins
            else               p = tie_p;
            out[i] += rw_ci * valid_row[i] * p;
        }
    }
}

// v1.8.2 Phase 2 (kernel-level batching): scalar reference. Numerically
// equivalent to calling showdown_oop_full(...) for each terminal in turn —
// the AVX2 version achieves matrix-bandwidth amortization via the fused
// outer-c loop, but parity-wise both implementations must produce the same
// result.
static void showdown_oop_full_batch(
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
    for (std::size_t c = c_lo; c < c_hi; ++c) {
        if (skip_mask && skip_mask[c]) {
            for (std::size_t t = 0; t < num_terminals; ++t) {
                out_array[t][c] = 0.0f;
            }
            continue;
        }
        const uint8_t* cat_row   = category_matrix + c * n;
        const float*   valid_row = valid_matrix    + c * n;
        for (std::size_t t = 0; t < num_terminals; ++t) {
            const float* reach = opp_reach_w_array[t];
            float sum = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                uint8_t cat = cat_row[i];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = win_p_array[t];
                else if (cat == 2) p = lose_p_array[t];
                else               p = tie_p_array[t];
                sum += reach[i] * valid_row[i] * p;
            }
            out_array[t][c] = sum;
        }
    }
}

}  // namespace deepsolver::cpu_simd::scalar_impl

namespace deepsolver::cpu_simd {

const Kernels scalar_kernels = {
    &scalar_impl::vec_mul_in_place,
    &scalar_impl::vec_scale_in_place,
    &scalar_impl::vec_add_in_place,
    &scalar_impl::vec_axpy,
    &scalar_impl::vec_fmadd,
    &scalar_impl::vec_dcfr_discount,
    &scalar_impl::vec_pos_add,
    &scalar_impl::vec_pos_normalize,
    &scalar_impl::vec_regret_update,
    &scalar_impl::vec_decay_add,
    &scalar_impl::vec_reach_weighted_strat_sum,
    &scalar_impl::showdown_oop_inner,
    &scalar_impl::showdown_ip_step,
    &scalar_impl::dot_valid_reach,
    &scalar_impl::fold_ip_step,
    &scalar_impl::showdown_oop_full,
    &scalar_impl::showdown_ip_full,
    &scalar_impl::showdown_oop_full_batch,
};

}  // namespace deepsolver::cpu_simd
