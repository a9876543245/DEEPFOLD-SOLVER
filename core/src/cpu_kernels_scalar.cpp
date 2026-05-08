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

// v1.8.0 P3-8 spike: full-row showdown kernels — see cpu_simd.h for the why.
// Logic is identical to calling showdown_oop_inner / showdown_ip_step n times
// in sequence; just hoists the per-call setup work out of the loop.
static void showdown_oop_full(
    const float* ev_matrix, const float* valid_matrix,
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    for (std::size_t c = 0; c < n; ++c) {
        const float* ev_row    = ev_matrix    + c * n;
        const float* valid_row = valid_matrix + c * n;
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
        out[c] = sum;
    }
}

static void showdown_ip_full(
    const float* ev_matrix, const float* valid_matrix,
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
        const float* ev_row    = ev_matrix    + ci * n;
        const float* valid_row = valid_matrix + ci * n;
        for (std::size_t i = 0; i < n; ++i) {
            float ev = ev_row[i];
            float valid = valid_row[i];
            if (valid <= 0.0f) continue;
            float p;
            if (ev >  0.5f) p = lose_p;
            else if (ev < -0.5f) p = win_p;
            else                 p = tie_p;
            out[i] += rw_ci * valid * p;
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
};

}  // namespace deepsolver::cpu_simd
