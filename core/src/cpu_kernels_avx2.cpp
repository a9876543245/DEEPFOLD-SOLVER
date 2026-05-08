/**
 * @file cpu_kernels_avx2.cpp
 * @brief AVX2 + FMA implementations of the CFR SIMD kernels.
 *
 * COMPILE FLAG REQUIREMENT
 *   This file MUST be compiled with `/arch:AVX2` (MSVC) or `-mavx2 -mfma`
 *   (GCC/Clang) — set per-source-file in CMakeLists.txt. The rest of the
 *   binary is built at the SSE2 baseline so it loads on every x86-64 CPU;
 *   only this translation unit emits AVX2 opcodes.
 *
 * RUNTIME GATING
 *   The functions below are only ever entered through the dispatch table
 *   in `cpu_kernels_dispatch.cpp`, which checks CPUID + OS state for
 *   AVX2/FMA/YMM availability before pointing at `avx2_kernels`. On a
 *   non-AVX2 CPU the dispatch table stays at `scalar_kernels` and these
 *   functions are never called — the AVX2 opcodes inside them are dead
 *   code from the executing CPU's point of view.
 *
 * SAFETY RULE FOR EDITING
 *   - No global ctors or `static` initializers that touch `__m256` here:
 *     those run before main(), before CPUID dispatch, and would crash on
 *     non-AVX2 hardware.
 *   - Don't include AVX2-using headers from this file's _consumers_; only
 *     the function bodies should contain AVX2 codegen.
 */

#include "cpu_simd.h"

#include <cstddef>
#include <immintrin.h>

namespace deepsolver::cpu_simd::avx2_impl {

// Horizontal sum of __m256, used by reductions.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static void vec_mul_in_place(float* x, const float* y, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(a, b));
    }
    for (; i < n; ++i) x[i] *= y[i];
}

static void vec_scale_in_place(float* x, float s, std::size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vs));
    }
    for (; i < n; ++i) x[i] *= s;
}

static void vec_add_in_place(float* dst, const float* src, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, s));
    }
    for (; i < n; ++i) dst[i] += src[i];
}

static void vec_axpy(float* dst, float a, const float* src, std::size_t n) {
    __m256 va = _mm256_set1_ps(a);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        d = _mm256_fmadd_ps(va, s, d);
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) dst[i] += a * src[i];
}

static void vec_fmadd(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d  = _mm256_loadu_ps(dst + i);
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        d = _mm256_fmadd_ps(va, vb, d);
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) dst[i] += a[i] * b[i];
}

static void vec_dcfr_discount(float* x, float pos, float neg, std::size_t n) {
    __m256 vpos = _mm256_set1_ps(pos);
    __m256 vneg = _mm256_set1_ps(neg);
    __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v    = _mm256_loadu_ps(x + i);
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
        __m256 disc = _mm256_blendv_ps(vneg, vpos, mask);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, disc));
    }
    for (; i < n; ++i) x[i] *= (x[i] > 0.0f) ? pos : neg;
}

static void vec_pos_add(float* pos_sum, const float* regret, std::size_t n) {
    __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r  = _mm256_loadu_ps(regret + i);
        __m256 rp = _mm256_max_ps(r, zero);
        __m256 p  = _mm256_loadu_ps(pos_sum + i);
        _mm256_storeu_ps(pos_sum + i, _mm256_add_ps(p, rp));
    }
    for (; i < n; ++i) pos_sum[i] += (regret[i] > 0.0f) ? regret[i] : 0.0f;
}

static void vec_pos_normalize(
    float* strat, const float* regret,
    const float* inv_pos_sum, const float* uniform_or_zero, std::size_t n)
{
    __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r   = _mm256_loadu_ps(regret + i);
        __m256 rp  = _mm256_max_ps(r, zero);
        __m256 inv = _mm256_loadu_ps(inv_pos_sum + i);
        __m256 uoz = _mm256_loadu_ps(uniform_or_zero + i);
        __m256 res = _mm256_fmadd_ps(rp, inv, uoz);
        _mm256_storeu_ps(strat + i, res);
    }
    for (; i < n; ++i) {
        float rp = (regret[i] > 0.0f) ? regret[i] : 0.0f;
        strat[i] = rp * inv_pos_sum[i] + uniform_or_zero[i];
    }
}

static void vec_regret_update(
    float* regret, const float* action_val, const float* node_val, std::size_t n)
{
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r  = _mm256_loadu_ps(regret + i);
        __m256 a  = _mm256_loadu_ps(action_val + i);
        __m256 nv = _mm256_loadu_ps(node_val + i);
        __m256 d  = _mm256_sub_ps(a, nv);
        _mm256_storeu_ps(regret + i, _mm256_add_ps(r, d));
    }
    for (; i < n; ++i) regret[i] += action_val[i] - node_val[i];
}

static void vec_decay_add(float* dst, float decay, const float* src, std::size_t n) {
    __m256 vd = _mm256_set1_ps(decay);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        d = _mm256_fmadd_ps(d, vd, s);
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) dst[i] = dst[i] * decay + src[i];
}

static void vec_reach_weighted_strat_sum(
    float* dst, float sw, const float* reach, const float* strat, std::size_t n)
{
    __m256 vsw = _mm256_set1_ps(sw);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d  = _mm256_loadu_ps(dst + i);
        __m256 r  = _mm256_loadu_ps(reach + i);
        __m256 s  = _mm256_loadu_ps(strat + i);
        __m256 rs = _mm256_mul_ps(r, s);
        d = _mm256_fmadd_ps(vsw, rs, d);
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) dst[i] += sw * reach[i] * strat[i];
}

static float showdown_oop_inner(
    const float* ev_row, const float* valid_row, const float* opp_reach_w,
    float win_p, float lose_p, float tie_p, std::size_t n)
{
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256 v05   = _mm256_set1_ps(0.5f);
    __m256 vn05  = _mm256_set1_ps(-0.5f);
    __m256 acc   = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 ev    = _mm256_loadu_ps(ev_row + i);
        __m256 valid = _mm256_loadu_ps(valid_row + i);
        __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
        __m256 m_win  = _mm256_cmp_ps(ev, v05,  _CMP_GT_OS);
        __m256 m_lose = _mm256_cmp_ps(ev, vn05, _CMP_LT_OS);
        __m256 payoff = _mm256_blendv_ps(vtie, vwin, m_win);
        payoff        = _mm256_blendv_ps(payoff, vlose, m_lose);
        __m256 rv     = _mm256_mul_ps(reach, valid);
        acc           = _mm256_fmadd_ps(rv, payoff, acc);
    }
    float sum = hsum256(acc);
    for (; i < n; ++i) {
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
    __m256 vrw   = _mm256_set1_ps(rw_ci);
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256 v05   = _mm256_set1_ps(0.5f);
    __m256 vn05  = _mm256_set1_ps(-0.5f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 ev    = _mm256_loadu_ps(ev_row + i);
        __m256 valid = _mm256_loadu_ps(valid_row + i);
        __m256 m_win  = _mm256_cmp_ps(ev, v05,  _CMP_GT_OS);
        __m256 m_lose = _mm256_cmp_ps(ev, vn05, _CMP_LT_OS);
        __m256 payoff = _mm256_blendv_ps(vtie, vlose, m_win);
        payoff        = _mm256_blendv_ps(payoff, vwin, m_lose);
        __m256 contrib = _mm256_mul_ps(valid, payoff);
        contrib        = _mm256_mul_ps(contrib, vrw);
        __m256 ov      = _mm256_loadu_ps(out_vals + i);
        _mm256_storeu_ps(out_vals + i, _mm256_add_ps(ov, contrib));
    }
    for (; i < n; ++i) {
        float ev = ev_row[i];
        float valid = valid_row[i];
        if (valid <= 0.0f) continue;
        float p;
        if (ev >  0.5f) p = lose_p;
        else if (ev < -0.5f) p = win_p;
        else                 p = tie_p;
        out_vals[i] += rw_ci * valid * p;
    }
}

static float dot_valid_reach(
    const float* valid_row, const float* opp_reach_w, std::size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(valid_row + i);
        __m256 r = _mm256_loadu_ps(opp_reach_w + i);
        acc = _mm256_fmadd_ps(v, r, acc);
    }
    float sum = hsum256(acc);
    for (; i < n; ++i) sum += valid_row[i] * opp_reach_w[i];
    return sum;
}

static void fold_ip_step(
    float* out_vals, const float* valid_row, float rw_ci, std::size_t n)
{
    __m256 vrw = _mm256_set1_ps(rw_ci);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v  = _mm256_loadu_ps(valid_row + i);
        __m256 ov = _mm256_loadu_ps(out_vals + i);
        ov = _mm256_fmadd_ps(vrw, v, ov);
        _mm256_storeu_ps(out_vals + i, ov);
    }
    for (; i < n; ++i) out_vals[i] += rw_ci * valid_row[i];
}

// v1.8.0 P3-8 spike: full-row showdown kernels. The per-call versions above
// rebuild vwin/vlose/vtie/v05/vn05 on every entry — for nc≈1200 calls per
// terminal that's 6000 wasted broadcast instructions. Hoisting them out of
// the c loop is the only structural change here; the inner loop body is
// byte-identical to showdown_oop_inner / showdown_ip_step.
static void showdown_oop_full(
    const float* ev_matrix, const float* valid_matrix,
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256 v05   = _mm256_set1_ps(0.5f);
    __m256 vn05  = _mm256_set1_ps(-0.5f);

    for (std::size_t c = 0; c < n; ++c) {
        const float* ev_row    = ev_matrix    + c * n;
        const float* valid_row = valid_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 ev    = _mm256_loadu_ps(ev_row + i);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            __m256 m_win  = _mm256_cmp_ps(ev, v05,  _CMP_GT_OS);
            __m256 m_lose = _mm256_cmp_ps(ev, vn05, _CMP_LT_OS);
            __m256 payoff = _mm256_blendv_ps(vtie, vwin, m_win);
            payoff        = _mm256_blendv_ps(payoff, vlose, m_lose);
            __m256 rv     = _mm256_mul_ps(reach, valid);
            acc           = _mm256_fmadd_ps(rv, payoff, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
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
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256 v05   = _mm256_set1_ps(0.5f);
    __m256 vn05  = _mm256_set1_ps(-0.5f);

    // Initialize output (matches vec_set_zero on the per-call path).
    {
        std::size_t i = 0;
        __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    for (std::size_t ci = 0; ci < n; ++ci) {
        float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        __m256 vrw = _mm256_set1_ps(rw_ci);
        const float* ev_row    = ev_matrix    + ci * n;
        const float* valid_row = valid_matrix + ci * n;
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 ev    = _mm256_loadu_ps(ev_row + i);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 m_win  = _mm256_cmp_ps(ev, v05,  _CMP_GT_OS);
            __m256 m_lose = _mm256_cmp_ps(ev, vn05, _CMP_LT_OS);
            __m256 payoff = _mm256_blendv_ps(vtie, vlose, m_win);
            payoff        = _mm256_blendv_ps(payoff, vwin, m_lose);
            __m256 contrib = _mm256_mul_ps(valid, payoff);
            contrib        = _mm256_mul_ps(contrib, vrw);
            __m256 ov      = _mm256_loadu_ps(out + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
        }
        for (; i < n; ++i) {
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

}  // namespace deepsolver::cpu_simd::avx2_impl

namespace deepsolver::cpu_simd {

const Kernels avx2_kernels = {
    &avx2_impl::vec_mul_in_place,
    &avx2_impl::vec_scale_in_place,
    &avx2_impl::vec_add_in_place,
    &avx2_impl::vec_axpy,
    &avx2_impl::vec_fmadd,
    &avx2_impl::vec_dcfr_discount,
    &avx2_impl::vec_pos_add,
    &avx2_impl::vec_pos_normalize,
    &avx2_impl::vec_regret_update,
    &avx2_impl::vec_decay_add,
    &avx2_impl::vec_reach_weighted_strat_sum,
    &avx2_impl::showdown_oop_inner,
    &avx2_impl::showdown_ip_step,
    &avx2_impl::dot_valid_reach,
    &avx2_impl::fold_ip_step,
    &avx2_impl::showdown_oop_full,
    &avx2_impl::showdown_ip_full,
};

}  // namespace deepsolver::cpu_simd
