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

static void vec_mul(float* dst, const float* a, const float* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] * b[i];
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

static void vec_pos_normalize2(
    float* strat0, float* strat1,
    const float* regret0, const float* regret1, std::size_t n)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 r0 = _mm256_loadu_ps(regret0 + i);
        const __m256 r1 = _mm256_loadu_ps(regret1 + i);
        const __m256 r0p = _mm256_max_ps(r0, zero);
        const __m256 r1p = _mm256_max_ps(r1, zero);
        const __m256 pos_sum = _mm256_add_ps(r0p, r1p);
        const __m256 has_pos = _mm256_cmp_ps(pos_sum, zero, _CMP_GT_OS);
        const __m256 safe_sum = _mm256_blendv_ps(one, pos_sum, has_pos);
        const __m256 inv = _mm256_div_ps(one, safe_sum);
        const __m256 norm0 = _mm256_mul_ps(r0p, inv);
        const __m256 norm1 = _mm256_mul_ps(r1p, inv);
        _mm256_storeu_ps(strat0 + i, _mm256_blendv_ps(half, norm0, has_pos));
        _mm256_storeu_ps(strat1 + i, _mm256_blendv_ps(half, norm1, has_pos));
    }
    for (; i < n; ++i) {
        const float r0p = (regret0[i] > 0.0f) ? regret0[i] : 0.0f;
        const float r1p = (regret1[i] > 0.0f) ? regret1[i] : 0.0f;
        const float pos_sum = r0p + r1p;
        if (pos_sum > 0.0f) {
            const float inv = 1.0f / pos_sum;
            strat0[i] = r0p * inv;
            strat1[i] = r1p * inv;
        } else {
            strat0[i] = 0.5f;
            strat1[i] = 0.5f;
        }
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

static float dot_valid_reach_active(
    const float* valid_row, const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count)
{
    float sum = 0.0f;
    for (std::size_t k = 0; k < active_count; ++k) {
        const uint16_t i = active_indices[k];
        sum += valid_row[i] * opp_reach_w[i];
    }
    return sum;
}

static float dot_valid_reach_active_runs(
    const float* valid_row, const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count)
{
    __m256 acc = _mm256_setzero_ps();
    float tail = 0.0f;
    for (std::size_t r = 0; r < run_count; ++r) {
        const std::size_t end =
            static_cast<std::size_t>(active_runs[r].start) + active_runs[r].count;
        std::size_t i = active_runs[r].start;
        for (; i + 8 <= end; i += 8) {
            __m256 v = _mm256_loadu_ps(valid_row + i);
            __m256 rw = _mm256_loadu_ps(opp_reach_w + i);
            acc = _mm256_fmadd_ps(v, rw, acc);
        }
        for (; i < end; ++i) tail += valid_row[i] * opp_reach_w[i];
    }
    return hsum256(acc) + tail;
}

static void fold_ip_step(
    float* out_vals, const float* valid_row, float rw_ci,
    const uint8_t* skip_mask, std::size_t n)
{
    __m256 vrw = _mm256_set1_ps(rw_ci);
    std::size_t i = 0;
    if (skip_mask) {
        const __m256i vzero_i = _mm256_setzero_si256();
        for (; i + 8 <= n; i += 8) {
            __m128i skip8  = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(skip_mask + i));
            __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
            __m256 keep    = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(skip32, vzero_i));
            __m256 v       = _mm256_loadu_ps(valid_row + i);
            __m256 ov      = _mm256_loadu_ps(out_vals + i);
            __m256 contrib = _mm256_and_ps(_mm256_mul_ps(vrw, v), keep);
            _mm256_storeu_ps(out_vals + i, _mm256_add_ps(ov, contrib));
        }
        for (; i < n; ++i) {
            if (!skip_mask[i]) out_vals[i] += rw_ci * valid_row[i];
        }
        return;
    }

    for (; i + 8 <= n; i += 8) {
        __m256 v  = _mm256_loadu_ps(valid_row + i);
        __m256 ov = _mm256_loadu_ps(out_vals + i);
        ov = _mm256_fmadd_ps(vrw, v, ov);
        _mm256_storeu_ps(out_vals + i, ov);
    }
    for (; i < n; ++i) out_vals[i] += rw_ci * valid_row[i];
}

static void fold_ip_step_active(
    float* out_vals, const float* valid_row, float rw_ci,
    const uint16_t* active_indices, std::size_t active_count)
{
    for (std::size_t k = 0; k < active_count; ++k) {
        const uint16_t i = active_indices[k];
        out_vals[i] += rw_ci * valid_row[i];
    }
}

static void fold_ip_step_active_runs(
    float* out_vals, const float* valid_row, float rw_ci,
    const ActiveRun* active_runs, std::size_t run_count)
{
    __m256 vrw = _mm256_set1_ps(rw_ci);
    for (std::size_t r = 0; r < run_count; ++r) {
        const std::size_t end =
            static_cast<std::size_t>(active_runs[r].start) + active_runs[r].count;
        std::size_t i = active_runs[r].start;
        for (; i + 8 <= end; i += 8) {
            __m256 v = _mm256_loadu_ps(valid_row + i);
            __m256 ov = _mm256_loadu_ps(out_vals + i);
            ov = _mm256_fmadd_ps(vrw, v, ov);
            _mm256_storeu_ps(out_vals + i, ov);
        }
        for (; i < end; ++i) out_vals[i] += rw_ci * valid_row[i];
    }
}

// v1.8.2 A2 encoding (POST_OPTIMIZATION_REVIEW Sec 4.3): the showdown kernels
// now read a 1-byte `category` per cell (pre-thresholded at precompute time)
// instead of re-bucketing the continuous ev each iteration. Bandwidth drops
// from 8 bytes/cell to 5 bytes/cell — bench shows ~1.97x speedup on cold
// cache workloads (monotone-shape, where matchup tables blow past L3).
//
// Categories: 0=invalid, 1=win, 2=lose, 3=tie. Mirrors the threshold from
// the prior ev-based code: ev > 0.5 → win, ev < -0.5 → lose, else tie. Cells
// with valid == 0 are stored as invalid; valid stays as f32 to preserve the
// exact validity weight.
//
// v1.8.1+ optional skip_mask — see cpu_simd.h for safety constraints.
static void showdown_oop_full(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    if (tie_p == 0.0f && lose_p == -win_p) {
        __m256 vwin  = _mm256_set1_ps(win_p);
        __m256 vlose = _mm256_set1_ps(lose_p);
        const __m256i v_one = _mm256_set1_epi32(1);
        const __m256i v_two = _mm256_set1_epi32(2);

        for (std::size_t c = 0; c < n; ++c) {
            if (skip_mask && skip_mask[c]) {
                out[c] = 0.0f;
                continue;
            }
            const uint8_t* cat_row   = category_matrix + c * n;
            const float*   valid_row = valid_matrix    + c * n;
            __m256 acc = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
                __m256 rv    = _mm256_mul_ps(reach, valid);
                acc          = _mm256_fmadd_ps(rv, payoff, acc);
            }
            float sum = hsum256(acc);
            for (; i < n; ++i) {
                const uint8_t cat = cat_row[i];
                if      (cat == 1) sum += opp_reach_w[i] * valid_row[i] * win_p;
                else if (cat == 2) sum += opp_reach_w[i] * valid_row[i] * lose_p;
            }
            out[c] = sum;
        }
        return;
    }

    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t c = 0; c < n; ++c) {
        if (skip_mask && skip_mask[c]) {
            out[c] = 0.0f;
            continue;
        }
        const uint8_t* cat_row   = category_matrix + c * n;
        const float*   valid_row = valid_matrix    + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            // Spread 8 category bytes into 8 i32 lanes via cvtepu8_epi32.
            __m128i cat8  = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(cat_row + i));
            __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
            __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
            __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
            __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
            __m256 payoff = _mm256_setzero_ps();
            payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
            payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
            payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            __m256 rv    = _mm256_mul_ps(reach, valid);
            acc          = _mm256_fmadd_ps(rv, payoff, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
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
    const float* opp_reach_w, const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    if (tie_p == 0.0f && lose_p == -win_p) {
        __m256 vwin_for_ip  = _mm256_set1_ps(lose_p);
        __m256 vlose_for_ip = _mm256_set1_ps(win_p);
        const __m256i v_one = _mm256_set1_epi32(1);
        const __m256i v_two = _mm256_set1_epi32(2);

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
            const uint8_t* cat_row   = category_matrix + ci * n;
            const float*   valid_row = valid_matrix    + ci * n;
            std::size_t i = 0;
            if (skip_mask) {
                const __m256i vzero_i = _mm256_setzero_si256();
                for (; i + 8 <= n; i += 8) {
                    __m128i skip8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(skip_mask + i));
                    __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
                    __m256 keep = _mm256_castsi256_ps(
                        _mm256_cmpeq_epi32(skip32, vzero_i));
                    __m128i cat8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(cat_row + i));
                    __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                    __m256 m_win = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                    __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                    __m256 payoff = _mm256_setzero_ps();
                    payoff = _mm256_blendv_ps(payoff, vwin_for_ip,  m_win);
                    payoff = _mm256_blendv_ps(payoff, vlose_for_ip, m_lose);
                    __m256 valid = _mm256_loadu_ps(valid_row + i);
                    __m256 contrib = _mm256_mul_ps(valid, payoff);
                    contrib = _mm256_mul_ps(contrib, vrw);
                    contrib = _mm256_and_ps(contrib, keep);
                    __m256 ov = _mm256_loadu_ps(out + i);
                    _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
                }
                for (; i < n; ++i) {
                    if (skip_mask[i]) continue;
                    const uint8_t cat = cat_row[i];
                    if      (cat == 1) out[i] += rw_ci * valid_row[i] * lose_p;
                    else if (cat == 2) out[i] += rw_ci * valid_row[i] * win_p;
                }
                continue;
            }

            for (; i + 8 <= n; i += 8) {
                __m128i cat8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin_for_ip,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose_for_ip, m_lose);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 contrib = _mm256_mul_ps(valid, payoff);
                contrib = _mm256_mul_ps(contrib, vrw);
                __m256 ov = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
            }
            for (; i < n; ++i) {
                const uint8_t cat = cat_row[i];
                if      (cat == 1) out[i] += rw_ci * valid_row[i] * lose_p;
                else if (cat == 2) out[i] += rw_ci * valid_row[i] * win_p;
            }
        }
        return;
    }

    // IP traverser sees flipped win/lose payoffs: when category==1 (OOP-win
    // from the OOP-perspective ev), IP loses, so the multiplier is lose_p.
    __m256 vwin_for_ip  = _mm256_set1_ps(lose_p);   // category==1 → IP loses
    __m256 vlose_for_ip = _mm256_set1_ps(win_p);    // category==2 → IP wins
    __m256 vtie         = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

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
        const uint8_t* cat_row   = category_matrix + ci * n;
        const float*   valid_row = valid_matrix    + ci * n;
        std::size_t i = 0;
        if (skip_mask) {
            const __m256i vzero_i = _mm256_setzero_si256();
            for (; i + 8 <= n; i += 8) {
                __m128i skip8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(skip_mask + i));
                __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
                __m256 keep    = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(skip32, vzero_i));
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin_for_ip,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose_for_ip, m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,         m_tie);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 contrib = _mm256_mul_ps(valid, payoff);
                contrib        = _mm256_mul_ps(contrib, vrw);
                contrib        = _mm256_and_ps(contrib, keep);
                __m256 ov      = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
            }
            for (; i < n; ++i) {
                if (skip_mask[i]) continue;
                uint8_t cat = cat_row[i];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = lose_p;
                else if (cat == 2) p = win_p;
                else               p = tie_p;
                out[i] += rw_ci * valid_row[i] * p;
            }
            continue;
        }

        for (; i + 8 <= n; i += 8) {
            __m128i cat8  = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(cat_row + i));
            __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
            __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
            __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
            __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
            __m256 payoff = _mm256_setzero_ps();
            payoff = _mm256_blendv_ps(payoff, vwin_for_ip,  m_win);
            payoff = _mm256_blendv_ps(payoff, vlose_for_ip, m_lose);
            payoff = _mm256_blendv_ps(payoff, vtie,         m_tie);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 contrib = _mm256_mul_ps(valid, payoff);
            contrib        = _mm256_mul_ps(contrib, vrw);
            __m256 ov      = _mm256_loadu_ps(out + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
        }
        for (; i < n; ++i) {
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

static void showdown_oop_signed_zero_rake(
    const float* signed_valid_matrix,
    const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    if (!skip_mask) {
        std::size_t c = 0;
        for (; c + 4 <= n; c += 4) {
            const float* row0 = signed_valid_matrix + (c + 0) * n;
            const float* row1 = signed_valid_matrix + (c + 1) * n;
            const float* row2 = signed_valid_matrix + (c + 2) * n;
            const float* row3 = signed_valid_matrix + (c + 3) * n;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + i), reach, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + i), reach, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row2 + i), reach, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row3 + i), reach, acc3);
            }
            float s0 = hsum256(acc0);
            float s1 = hsum256(acc1);
            float s2 = hsum256(acc2);
            float s3 = hsum256(acc3);
            for (; i < n; ++i) {
                const float reach = opp_reach_w[i];
                s0 += row0[i] * reach;
                s1 += row1[i] * reach;
                s2 += row2[i] * reach;
                s3 += row3[i] * reach;
            }
            _mm_storeu_ps(out + c, _mm_setr_ps(
                s0 * win_p, s1 * win_p, s2 * win_p, s3 * win_p));
        }
        for (; c < n; ++c) {
            const float* coeff_row = signed_valid_matrix + c * n;
            __m256 acc = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 coeff = _mm256_loadu_ps(coeff_row + i);
                __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
                acc = _mm256_fmadd_ps(coeff, reach, acc);
            }
            float sum = hsum256(acc);
            for (; i < n; ++i) {
                sum += coeff_row[i] * opp_reach_w[i];
            }
            out[c] = sum * win_p;
        }
        return;
    }

    const __m256 vwin = _mm256_set1_ps(win_p);
    for (std::size_t c = 0; c < n; ++c) {
        if (skip_mask && skip_mask[c]) {
            out[c] = 0.0f;
            continue;
        }
        const float* coeff_row = signed_valid_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 coeff = _mm256_loadu_ps(coeff_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            acc = _mm256_fmadd_ps(coeff, reach, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
            sum += coeff_row[i] * opp_reach_w[i];
        }
        _mm_store_ss(out + c, _mm_mul_ss(_mm_set_ss(sum), _mm256_castps256_ps128(vwin)));
    }
}

static void showdown_ip_signed_zero_rake(
    const float* signed_valid_matrix,
    const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    if (!skip_mask) {
        for (std::size_t ci = 0; ci < n; ++ci) {
            const float scale = -opp_reach_w[ci] * win_p;
            if (scale == 0.0f) continue;
            const __m256 vscale = _mm256_set1_ps(scale);
            const float* coeff_row = signed_valid_matrix + ci * n;
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 coeff = _mm256_loadu_ps(coeff_row + i);
                __m256 ov = _mm256_loadu_ps(out + i);
                ov = _mm256_fmadd_ps(coeff, vscale, ov);
                _mm256_storeu_ps(out + i, ov);
            }
            for (; i < n; ++i) out[i] += coeff_row[i] * scale;
        }
        return;
    }

    for (std::size_t ci = 0; ci < n; ++ci) {
        const float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        const __m256 vscale = _mm256_set1_ps(-rw_ci * win_p);
        const float* coeff_row = signed_valid_matrix + ci * n;
        std::size_t i = 0;
        if (skip_mask) {
            const __m256i vzero_i = _mm256_setzero_si256();
            for (; i + 8 <= n; i += 8) {
                __m128i skip8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(skip_mask + i));
                __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
                __m256 keep = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(skip32, vzero_i));
                __m256 coeff = _mm256_loadu_ps(coeff_row + i);
                __m256 contrib = _mm256_and_ps(_mm256_mul_ps(coeff, vscale), keep);
                __m256 ov = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
            }
            for (; i < n; ++i) {
                if (!skip_mask[i]) out[i] += coeff_row[i] * (-rw_ci * win_p);
            }
            continue;
        }
        for (; i + 8 <= n; i += 8) {
            __m256 coeff = _mm256_loadu_ps(coeff_row + i);
            __m256 ov = _mm256_loadu_ps(out + i);
            ov = _mm256_fmadd_ps(coeff, vscale, ov);
            _mm256_storeu_ps(out + i, ov);
        }
        for (; i < n; ++i) out[i] += coeff_row[i] * (-rw_ci * win_p);
    }
}

static void showdown_oop_signed_count_zero_rake_8row_no_skip(
    const int8_t* signed_count_matrix,
    const float* opp_reach,
    const float* inv_weights,
    float* out, std::size_t n,
    float win_p);

static void showdown_oop_signed_count_zero_rake(
    const int8_t* signed_count_matrix,
    const float* opp_reach,
    const float* inv_weights,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    static constexpr bool kOopSignedCountEightRow = true;
    if constexpr (kOopSignedCountEightRow) {
        if (!skip_mask) {
            showdown_oop_signed_count_zero_rake_8row_no_skip(
                signed_count_matrix, opp_reach, inv_weights, out, n, win_p);
            return;
        }
    }

    for (std::size_t c = 0; c < n; ++c) {
        if (skip_mask && skip_mask[c]) {
            out[c] = 0.0f;
            continue;
        }
        const int8_t* count_row = signed_count_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(count_row + i));
            __m256 count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            __m256 reach = _mm256_loadu_ps(opp_reach + i);
            acc = _mm256_fmadd_ps(count, reach, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
            sum += static_cast<float>(count_row[i]) * opp_reach[i];
        }
        out[c] = sum * win_p * inv_weights[c];
    }
}

static void showdown_ip_signed_count_zero_rake(
    const int8_t* signed_count_matrix,
    const float* opp_reach,
    const float* inv_weights,
    const uint8_t* skip_mask,
    float* out, std::size_t n,
    float win_p)
{
    {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    for (std::size_t ci = 0; ci < n; ++ci) {
        const float scale = -opp_reach[ci] * win_p;
        if (scale == 0.0f) continue;
        const __m256 vscale = _mm256_set1_ps(scale);
        const int8_t* count_row = signed_count_matrix + ci * n;
        std::size_t i = 0;
        if (skip_mask) {
            const __m256i vzero_i = _mm256_setzero_si256();
            for (; i + 8 <= n; i += 8) {
                __m128i skip8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(skip_mask + i));
                __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
                __m256 keep = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(skip32, vzero_i));
                __m128i count8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(count_row + i));
                __m256 count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
                __m256 contrib = _mm256_and_ps(_mm256_mul_ps(count, vscale), keep);
                __m256 ov = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
            }
            for (; i < n; ++i) {
                if (!skip_mask[i]) {
                    out[i] += static_cast<float>(count_row[i]) * scale;
                }
            }
            continue;
        }
        for (; i + 8 <= n; i += 8) {
            __m128i count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(count_row + i));
            __m256 count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            __m256 ov = _mm256_loadu_ps(out + i);
            ov = _mm256_fmadd_ps(count, vscale, ov);
            _mm256_storeu_ps(out + i, ov);
        }
        for (; i < n; ++i) {
            out[i] += static_cast<float>(count_row[i]) * scale;
        }
    }

    std::size_t i = 0;
    if (skip_mask) {
        const __m256i vzero_i = _mm256_setzero_si256();
        for (; i + 8 <= n; i += 8) {
            __m128i skip8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(skip_mask + i));
            __m256i skip32 = _mm256_cvtepu8_epi32(skip8);
            __m256 keep = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(skip32, vzero_i));
            __m256 scaled = _mm256_mul_ps(
                _mm256_loadu_ps(out + i), _mm256_loadu_ps(inv_weights + i));
            _mm256_storeu_ps(out + i, _mm256_and_ps(scaled, keep));
        }
        for (; i < n; ++i) {
            out[i] = skip_mask[i] ? 0.0f : out[i] * inv_weights[i];
        }
        return;
    }
    for (; i + 8 <= n; i += 8) {
        __m256 ov = _mm256_loadu_ps(out + i);
        __m256 iw = _mm256_loadu_ps(inv_weights + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(ov, iw));
    }
    for (; i < n; ++i) out[i] *= inv_weights[i];
}

static void showdown_oop_full_active(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t k = 0; k < active_count; ++k) {
        const uint16_t c = active_indices[k];
        const uint8_t* cat_row =
            category_matrix + static_cast<std::size_t>(c) * n;
        const float* valid_row =
            valid_matrix + static_cast<std::size_t>(c) * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i cat8  = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(cat_row + i));
            __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
            __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
            __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
            __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
            __m256 payoff = _mm256_setzero_ps();
            payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
            payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
            payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            __m256 rv    = _mm256_mul_ps(reach, valid);
            acc          = _mm256_fmadd_ps(rv, payoff, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
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

static void showdown_ip_full_active(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* active_indices, std::size_t active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    } else {
        for (std::size_t k = 0; k < active_count; ++k) {
            out[active_indices[k]] = 0.0f;
        }
    }

    for (std::size_t ci = 0; ci < n; ++ci) {
        float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        const uint8_t* cat_row   = category_matrix + ci * n;
        const float*   valid_row = valid_matrix    + ci * n;
        for (std::size_t k = 0; k < active_count; ++k) {
            const uint16_t i = active_indices[k];
            uint8_t cat = cat_row[i];
            if (cat == 0) continue;
            float p;
            if      (cat == 1) p = lose_p;
            else if (cat == 2) p = win_p;
            else               p = tie_p;
            out[i] += rw_ci * valid_row[i] * p;
        }
    }
}

static void showdown_oop_full_active_runs(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t r = 0; r < run_count; ++r) {
        const std::size_t end =
            static_cast<std::size_t>(active_runs[r].start) + active_runs[r].count;
        for (std::size_t c = active_runs[r].start; c < end; ++c) {
            const uint8_t* cat_row =
                category_matrix + static_cast<std::size_t>(c) * n;
            const float* valid_row =
                valid_matrix + static_cast<std::size_t>(c) * n;
            __m256 acc = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
                __m256 rv    = _mm256_mul_ps(reach, valid);
                acc          = _mm256_fmadd_ps(rv, payoff, acc);
            }
            float sum = hsum256(acc);
            for (; i < n; ++i) {
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
}

static void showdown_ip_full_active_runs(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const ActiveRun* active_runs, std::size_t run_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    } else {
        const __m256 z = _mm256_setzero_ps();
        for (std::size_t r = 0; r < run_count; ++r) {
            const std::size_t end =
                static_cast<std::size_t>(active_runs[r].start) + active_runs[r].count;
            std::size_t i = active_runs[r].start;
            for (; i + 8 <= end; i += 8) _mm256_storeu_ps(out + i, z);
            for (; i < end; ++i) out[i] = 0.0f;
        }
    }

    __m256 vwin_for_ip  = _mm256_set1_ps(win_p);
    __m256 vlose_for_ip = _mm256_set1_ps(lose_p);
    __m256 vtie         = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t ci = 0; ci < n; ++ci) {
        float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        __m256 vrw = _mm256_set1_ps(rw_ci);
        const uint8_t* cat_row =
            category_matrix + static_cast<std::size_t>(ci) * n;
        const float* valid_row =
            valid_matrix + static_cast<std::size_t>(ci) * n;
        for (std::size_t r = 0; r < run_count; ++r) {
            const std::size_t end =
                static_cast<std::size_t>(active_runs[r].start) + active_runs[r].count;
            std::size_t i = active_runs[r].start;
            for (; i + 8 <= end; i += 8) {
                __m128i cat8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vlose_for_ip, m_win);
                payoff = _mm256_blendv_ps(payoff, vwin_for_ip,  m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,         m_tie);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 contrib = _mm256_mul_ps(valid, payoff);
                contrib        = _mm256_mul_ps(contrib, vrw);
                __m256 ov      = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_add_ps(ov, contrib));
            }
            for (; i < end; ++i) {
                uint8_t cat = cat_row[i];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = lose_p;
                else if (cat == 2) p = win_p;
                else               p = tie_p;
                out[i] += rw_ci * valid_row[i] * p;
            }
        }
    }
}

static void showdown_oop_full_active_opp_blocks(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const ActiveRun* opp_blocks, std::size_t opp_block_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t sk = 0; sk < self_active_count; ++sk) {
        const uint16_t c = self_active_indices[sk];
        const uint8_t* cat_row =
            category_matrix + static_cast<std::size_t>(c) * n;
        const float* valid_row =
            valid_matrix + static_cast<std::size_t>(c) * n;
        __m256 acc = _mm256_setzero_ps();
        float tail_sum = 0.0f;
        for (std::size_t b = 0; b < opp_block_count; ++b) {
            const std::size_t end =
                static_cast<std::size_t>(opp_blocks[b].start) + opp_blocks[b].count;
            std::size_t i = opp_blocks[b].start;
            for (; i + 8 <= end; i += 8) {
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
                __m256 valid = _mm256_loadu_ps(valid_row + i);
                __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
                __m256 rv    = _mm256_mul_ps(reach, valid);
                acc          = _mm256_fmadd_ps(rv, payoff, acc);
            }
            for (; i < end; ++i) {
                uint8_t cat = cat_row[i];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = win_p;
                else if (cat == 2) p = lose_p;
                else               p = tie_p;
                tail_sum += opp_reach_w[i] * valid_row[i] * p;
            }
        }
        out[c] = hsum256(acc) + tail_sum;
    }
}

static void showdown_ip_full_active_opp_blocks(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const ActiveRun* opp_blocks, std::size_t opp_block_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p,
    bool clear_full_output)
{
    if (clear_full_output) {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    } else {
        for (std::size_t k = 0; k < self_active_count; ++k) {
            out[self_active_indices[k]] = 0.0f;
        }
    }

    for (std::size_t b = 0; b < opp_block_count; ++b) {
        const std::size_t end =
            static_cast<std::size_t>(opp_blocks[b].start) + opp_blocks[b].count;
        for (std::size_t ci = opp_blocks[b].start; ci < end; ++ci) {
            const float rw_ci = opp_reach_w[ci];
            if (rw_ci == 0.0f) continue;
            const uint8_t* cat_row =
                category_matrix + static_cast<std::size_t>(ci) * n;
            const float* valid_row =
                valid_matrix + static_cast<std::size_t>(ci) * n;
            for (std::size_t sk = 0; sk < self_active_count; ++sk) {
                const uint16_t i = self_active_indices[sk];
                const uint8_t cat = cat_row[i];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = lose_p;
                else if (cat == 2) p = win_p;
                else               p = tie_p;
                out[i] += rw_ci * valid_row[i] * p;
            }
        }
    }
}

static void showdown_oop_full_active2(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const uint16_t* opp_active_indices, std::size_t opp_active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    for (std::size_t sk = 0; sk < self_active_count; ++sk) {
        const uint16_t c = self_active_indices[sk];
        const uint8_t* cat_row =
            category_matrix + static_cast<std::size_t>(c) * n;
        const float* valid_row =
            valid_matrix + static_cast<std::size_t>(c) * n;
        float sum = 0.0f;
        for (std::size_t ok = 0; ok < opp_active_count; ++ok) {
            const uint16_t i = opp_active_indices[ok];
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

static void showdown_ip_full_active2(
    const uint8_t* category_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    const uint16_t* self_active_indices, std::size_t self_active_count,
    const uint16_t* opp_active_indices, std::size_t opp_active_count,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    {
        std::size_t i = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, z);
        for (; i < n; ++i) out[i] = 0.0f;
    }

    for (std::size_t ok = 0; ok < opp_active_count; ++ok) {
        const uint16_t ci = opp_active_indices[ok];
        float rw_ci = opp_reach_w[ci];
        if (rw_ci == 0.0f) continue;
        const uint8_t* cat_row =
            category_matrix + static_cast<std::size_t>(ci) * n;
        const float* valid_row =
            valid_matrix + static_cast<std::size_t>(ci) * n;
        for (std::size_t sk = 0; sk < self_active_count; ++sk) {
            const uint16_t i = self_active_indices[sk];
            uint8_t cat = cat_row[i];
            if (cat == 0) continue;
            float p;
            if      (cat == 1) p = lose_p;
            else if (cat == 2) p = win_p;
            else               p = tie_p;
            out[i] += rw_ci * valid_row[i] * p;
        }
    }
}

static void showdown_oop_full_batch_tiled4(
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
    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t c = c_lo; c < c_hi; ++c) {
        if (skip_mask && skip_mask[c]) {
            for (std::size_t t = 0; t < num_terminals; ++t) {
                out_array[t][c] = 0.0f;
            }
            continue;
        }

        const uint8_t* cat_row   = category_matrix + c * n;
        const float*   valid_row = valid_matrix    + c * n;

        std::size_t t = 0;
        for (; t + 4 <= num_terminals; t += 4) {
            const __m256 vwin0  = _mm256_set1_ps(win_p_array[t + 0]);
            const __m256 vlose0 = _mm256_set1_ps(lose_p_array[t + 0]);
            const __m256 vtie0  = _mm256_set1_ps(tie_p_array[t + 0]);
            const __m256 vwin1  = _mm256_set1_ps(win_p_array[t + 1]);
            const __m256 vlose1 = _mm256_set1_ps(lose_p_array[t + 1]);
            const __m256 vtie1  = _mm256_set1_ps(tie_p_array[t + 1]);
            const __m256 vwin2  = _mm256_set1_ps(win_p_array[t + 2]);
            const __m256 vlose2 = _mm256_set1_ps(lose_p_array[t + 2]);
            const __m256 vtie2  = _mm256_set1_ps(tie_p_array[t + 2]);
            const __m256 vwin3  = _mm256_set1_ps(win_p_array[t + 3]);
            const __m256 vlose3 = _mm256_set1_ps(lose_p_array[t + 3]);
            const __m256 vtie3  = _mm256_set1_ps(tie_p_array[t + 3]);

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 valid8 = _mm256_loadu_ps(valid_row + i);

                __m256 payoff0 = _mm256_setzero_ps();
                payoff0 = _mm256_blendv_ps(payoff0, vwin0,  m_win);
                payoff0 = _mm256_blendv_ps(payoff0, vlose0, m_lose);
                payoff0 = _mm256_blendv_ps(payoff0, vtie0,  m_tie);
                acc0 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(opp_reach_w_array[t + 0] + i), valid8),
                    payoff0, acc0);

                __m256 payoff1 = _mm256_setzero_ps();
                payoff1 = _mm256_blendv_ps(payoff1, vwin1,  m_win);
                payoff1 = _mm256_blendv_ps(payoff1, vlose1, m_lose);
                payoff1 = _mm256_blendv_ps(payoff1, vtie1,  m_tie);
                acc1 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(opp_reach_w_array[t + 1] + i), valid8),
                    payoff1, acc1);

                __m256 payoff2 = _mm256_setzero_ps();
                payoff2 = _mm256_blendv_ps(payoff2, vwin2,  m_win);
                payoff2 = _mm256_blendv_ps(payoff2, vlose2, m_lose);
                payoff2 = _mm256_blendv_ps(payoff2, vtie2,  m_tie);
                acc2 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(opp_reach_w_array[t + 2] + i), valid8),
                    payoff2, acc2);

                __m256 payoff3 = _mm256_setzero_ps();
                payoff3 = _mm256_blendv_ps(payoff3, vwin3,  m_win);
                payoff3 = _mm256_blendv_ps(payoff3, vlose3, m_lose);
                payoff3 = _mm256_blendv_ps(payoff3, vtie3,  m_tie);
                acc3 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(opp_reach_w_array[t + 3] + i), valid8),
                    payoff3, acc3);
            }

            float sum0 = hsum256(acc0);
            float sum1 = hsum256(acc1);
            float sum2 = hsum256(acc2);
            float sum3 = hsum256(acc3);
            for (std::size_t k = i; k < n; ++k) {
                const uint8_t cat = cat_row[k];
                if (cat == 0) continue;
                const float valid = valid_row[k];
                const float p0 = (cat == 1) ? win_p_array[t + 0]
                               : (cat == 2) ? lose_p_array[t + 0]
                                            : tie_p_array[t + 0];
                const float p1 = (cat == 1) ? win_p_array[t + 1]
                               : (cat == 2) ? lose_p_array[t + 1]
                                            : tie_p_array[t + 1];
                const float p2 = (cat == 1) ? win_p_array[t + 2]
                               : (cat == 2) ? lose_p_array[t + 2]
                                            : tie_p_array[t + 2];
                const float p3 = (cat == 1) ? win_p_array[t + 3]
                               : (cat == 2) ? lose_p_array[t + 3]
                                            : tie_p_array[t + 3];
                sum0 += opp_reach_w_array[t + 0][k] * valid * p0;
                sum1 += opp_reach_w_array[t + 1][k] * valid * p1;
                sum2 += opp_reach_w_array[t + 2][k] * valid * p2;
                sum3 += opp_reach_w_array[t + 3][k] * valid * p3;
            }
            out_array[t + 0][c] = sum0;
            out_array[t + 1][c] = sum1;
            out_array[t + 2][c] = sum2;
            out_array[t + 3][c] = sum3;
        }

        for (; t < num_terminals; ++t) {
            const __m256 vwin  = _mm256_set1_ps(win_p_array[t]);
            const __m256 vlose = _mm256_set1_ps(lose_p_array[t]);
            const __m256 vtie  = _mm256_set1_ps(tie_p_array[t]);
            __m256 acc = _mm256_setzero_ps();

            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i cat8  = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(cat_row + i));
                __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
                acc = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(opp_reach_w_array[t] + i),
                                  _mm256_loadu_ps(valid_row + i)),
                    payoff, acc);
            }

            float sum = hsum256(acc);
            for (std::size_t k = i; k < n; ++k) {
                const uint8_t cat = cat_row[k];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = win_p_array[t];
                else if (cat == 2) p = lose_p_array[t];
                else               p = tie_p_array[t];
                sum += opp_reach_w_array[t][k] * valid_row[k] * p;
            }
            out_array[t][c] = sum;
        }
    }
}

// v1.8.2 Phase 2 (kernel-level batching). Processes M terminals in a single
// kernel call, all sharing the same matchup table. Key restructuring:
//
//   loop nesting: outer c -> inner i -> inner-most t
//
//   Within one (c, i) the matrix bytes (cat_row[i..i+8] + valid_row[i..i+8])
//   are loaded ONCE and reused across all M terminals' accumulators. The
//   per-terminal (vwin, vlose, vtie) broadcasts are hoisted to a one-shot
//   pre-pass before the c loop, since they're constant for the whole call.
//
//   Compared to looping showdown_oop_full(...) M times: the FP-op count is
//   identical (same total iters), but the matrix bytes are streamed from
//   memory M× less often. Bench shows the natural BFS terminal order
//   already covers most of the L3 reuse; this kernel pushes the saving
//   into the L1/L2 regime where row data stays hot for all M accumulators.
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
    // Cap how many terminals we'll keep on stack for accumulators / payoff
    // broadcasts. 256 is comfortably above the largest production group
    // (~318 on monotone) — but we fall back to per-call when exceeded so
    // we never overflow the stack on pathological inputs.
    constexpr std::size_t kMaxBatch = 384;
    if (num_terminals == 0) return;
    if (num_terminals > kMaxBatch) {
        for (std::size_t t = 0; t < num_terminals; ++t) {
            // Use the dispatched showdown_oop_full so behavior matches the
            // dispatch table the caller resolved.
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                if (skip_mask && skip_mask[c]) {
                    out_array[t][c] = 0.0f;
                    continue;
                }
                const uint8_t* cat_row   = category_matrix + c * n;
                const float*   valid_row = valid_matrix    + c * n;
                __m256 acc = _mm256_setzero_ps();
                const __m256i v_one   = _mm256_set1_epi32(1);
                const __m256i v_two   = _mm256_set1_epi32(2);
                const __m256i v_three = _mm256_set1_epi32(3);
                __m256 vwin  = _mm256_set1_ps(win_p_array[t]);
                __m256 vlose = _mm256_set1_ps(lose_p_array[t]);
                __m256 vtie  = _mm256_set1_ps(tie_p_array[t]);
                std::size_t i = 0;
                for (; i + 8 <= n; i += 8) {
                    __m128i cat8  = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(cat_row + i));
                    __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
                    __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
                    __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
                    __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
                    __m256 payoff = _mm256_setzero_ps();
                    payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                    payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                    payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);
                    __m256 valid = _mm256_loadu_ps(valid_row + i);
                    __m256 reach = _mm256_loadu_ps(opp_reach_w_array[t] + i);
                    __m256 rv    = _mm256_mul_ps(reach, valid);
                    acc          = _mm256_fmadd_ps(rv, payoff, acc);
                }
                float sum = hsum256(acc);
                for (; i < n; ++i) {
                    uint8_t cat = cat_row[i];
                    if (cat == 0) continue;
                    float p;
                    if      (cat == 1) p = win_p_array[t];
                    else if (cat == 2) p = lose_p_array[t];
                    else               p = tie_p_array[t];
                    sum += opp_reach_w_array[t][i] * valid_row[i] * p;
                }
                out_array[t][c] = sum;
            }
        }
        return;
    }

    showdown_oop_full_batch_tiled4(
        category_matrix, valid_matrix, n, num_terminals,
        opp_reach_w_array, skip_mask, out_array,
        win_p_array, lose_p_array, tie_p_array,
        c_lo, c_hi);
    return;

#if 0
    // Per-terminal broadcasts hoisted out of the c/i loops. 32-byte aligned
    // so we can _mm256_load_ps from them directly.
    alignas(32) float vwin_buf [kMaxBatch * 8];
    alignas(32) float vlose_buf[kMaxBatch * 8];
    alignas(32) float vtie_buf [kMaxBatch * 8];
    for (std::size_t t = 0; t < num_terminals; ++t) {
        _mm256_store_ps(vwin_buf  + t * 8, _mm256_set1_ps(win_p_array[t]));
        _mm256_store_ps(vlose_buf + t * 8, _mm256_set1_ps(lose_p_array[t]));
        _mm256_store_ps(vtie_buf  + t * 8, _mm256_set1_ps(tie_p_array[t]));
    }

    // Per-terminal accumulator bank. Reset to zero at the start of each c.
    alignas(32) float acc_buf[kMaxBatch * 8];

    const __m256i v_one   = _mm256_set1_epi32(1);
    const __m256i v_two   = _mm256_set1_epi32(2);
    const __m256i v_three = _mm256_set1_epi32(3);

    for (std::size_t c = c_lo; c < c_hi; ++c) {
        if (skip_mask && skip_mask[c]) {
            for (std::size_t t = 0; t < num_terminals; ++t) {
                out_array[t][c] = 0.0f;
            }
            continue;
        }
        const uint8_t* cat_row   = category_matrix + c * n;
        const float*   valid_row = valid_matrix    + c * n;

        // Zero accumulators for this c.
        const __m256 zero = _mm256_setzero_ps();
        for (std::size_t t = 0; t < num_terminals; ++t) {
            _mm256_store_ps(acc_buf + t * 8, zero);
        }

        // Vectorized inner over i. Matrix row data loaded once per (c, i),
        // reused across all M terminals.
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i cat8  = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(cat_row + i));
            __m256i cat32 = _mm256_cvtepu8_epi32(cat8);
            __m256 m_win  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_one));
            __m256 m_lose = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_two));
            __m256 m_tie  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(cat32, v_three));
            __m256 valid8 = _mm256_loadu_ps(valid_row + i);

            for (std::size_t t = 0; t < num_terminals; ++t) {
                __m256 vwin  = _mm256_load_ps(vwin_buf  + t * 8);
                __m256 vlose = _mm256_load_ps(vlose_buf + t * 8);
                __m256 vtie  = _mm256_load_ps(vtie_buf  + t * 8);
                __m256 payoff = _mm256_setzero_ps();
                payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
                payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
                payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);

                __m256 reach8 = _mm256_loadu_ps(opp_reach_w_array[t] + i);
                __m256 rv     = _mm256_mul_ps(reach8, valid8);
                __m256 a      = _mm256_load_ps(acc_buf + t * 8);
                a             = _mm256_fmadd_ps(rv, payoff, a);
                _mm256_store_ps(acc_buf + t * 8, a);
            }
        }

        // Tail (i % 8) and per-terminal output write. Tail re-executes the
        // category dispatch in scalar form — typical n=1326 leaves 6 tail
        // iterations so this is cheap.
        for (std::size_t t = 0; t < num_terminals; ++t) {
            __m256 a = _mm256_load_ps(acc_buf + t * 8);
            float sum = hsum256(a);
            for (std::size_t k = i; k < n; ++k) {
                uint8_t cat = cat_row[k];
                if (cat == 0) continue;
                float p;
                if      (cat == 1) p = win_p_array[t];
                else if (cat == 2) p = lose_p_array[t];
                else               p = tie_p_array[t];
                sum += opp_reach_w_array[t][k] * valid_row[k] * p;
            }
            out_array[t][c] = sum;
        }
    }
#endif
}

static void showdown_oop_signed_count_zero_rake_8row_no_skip(
    const int8_t* signed_count_matrix,
    const float* opp_reach,
    const float* inv_weights,
    float* out, std::size_t n,
    float win_p)
{
    std::size_t c = 0;
    const __m128 vwin = _mm_set1_ps(win_p);
    for (; c + 8 <= n; c += 8) {
        const int8_t* row0 = signed_count_matrix + (c + 0) * n;
        const int8_t* row1 = signed_count_matrix + (c + 1) * n;
        const int8_t* row2 = signed_count_matrix + (c + 2) * n;
        const int8_t* row3 = signed_count_matrix + (c + 3) * n;
        const int8_t* row4 = signed_count_matrix + (c + 4) * n;
        const int8_t* row5 = signed_count_matrix + (c + 5) * n;
        const int8_t* row6 = signed_count_matrix + (c + 6) * n;
        const int8_t* row7 = signed_count_matrix + (c + 7) * n;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 reach = _mm256_loadu_ps(opp_reach + i);
            __m128i count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row0 + i));
            __m256 count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc0 = _mm256_fmadd_ps(count, reach, acc0);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row1 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc1 = _mm256_fmadd_ps(count, reach, acc1);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row2 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc2 = _mm256_fmadd_ps(count, reach, acc2);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row3 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc3 = _mm256_fmadd_ps(count, reach, acc3);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row4 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc4 = _mm256_fmadd_ps(count, reach, acc4);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row5 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc5 = _mm256_fmadd_ps(count, reach, acc5);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row6 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc6 = _mm256_fmadd_ps(count, reach, acc6);
            count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row7 + i));
            count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            acc7 = _mm256_fmadd_ps(count, reach, acc7);
        }
        float s0 = hsum256(acc0);
        float s1 = hsum256(acc1);
        float s2 = hsum256(acc2);
        float s3 = hsum256(acc3);
        float s4 = hsum256(acc4);
        float s5 = hsum256(acc5);
        float s6 = hsum256(acc6);
        float s7 = hsum256(acc7);
        for (; i < n; ++i) {
            const float reach = opp_reach[i];
            s0 += static_cast<float>(row0[i]) * reach;
            s1 += static_cast<float>(row1[i]) * reach;
            s2 += static_cast<float>(row2[i]) * reach;
            s3 += static_cast<float>(row3[i]) * reach;
            s4 += static_cast<float>(row4[i]) * reach;
            s5 += static_cast<float>(row5[i]) * reach;
            s6 += static_cast<float>(row6[i]) * reach;
            s7 += static_cast<float>(row7[i]) * reach;
        }
        const __m128 sum_lo = _mm_setr_ps(s0, s1, s2, s3);
        const __m128 sum_hi = _mm_setr_ps(s4, s5, s6, s7);
        const __m128 iw_lo = _mm_loadu_ps(inv_weights + c);
        const __m128 iw_hi = _mm_loadu_ps(inv_weights + c + 4);
        _mm_storeu_ps(out + c,
            _mm_mul_ps(_mm_mul_ps(sum_lo, vwin), iw_lo));
        _mm_storeu_ps(out + c + 4,
            _mm_mul_ps(_mm_mul_ps(sum_hi, vwin), iw_hi));
    }
    for (; c < n; ++c) {
        const int8_t* count_row = signed_count_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i count8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(count_row + i));
            __m256 count = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(count8));
            __m256 reach = _mm256_loadu_ps(opp_reach + i);
            acc = _mm256_fmadd_ps(count, reach, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
            sum += static_cast<float>(count_row[i]) * opp_reach[i];
        }
        out[c] = sum * win_p * inv_weights[c];
    }
}

}  // namespace deepsolver::cpu_simd::avx2_impl

namespace deepsolver::cpu_simd {

const Kernels avx2_kernels = {
    &avx2_impl::vec_mul_in_place,
    &avx2_impl::vec_mul,
    &avx2_impl::vec_scale_in_place,
    &avx2_impl::vec_add_in_place,
    &avx2_impl::vec_axpy,
    &avx2_impl::vec_fmadd,
    &avx2_impl::vec_dcfr_discount,
    &avx2_impl::vec_pos_add,
    &avx2_impl::vec_pos_normalize,
    &avx2_impl::vec_pos_normalize2,
    &avx2_impl::vec_regret_update,
    &avx2_impl::vec_decay_add,
    &avx2_impl::vec_reach_weighted_strat_sum,
    &avx2_impl::showdown_oop_inner,
    &avx2_impl::showdown_ip_step,
    &avx2_impl::dot_valid_reach,
    &avx2_impl::dot_valid_reach_active,
    &avx2_impl::dot_valid_reach_active_runs,
    &avx2_impl::fold_ip_step,
    &avx2_impl::fold_ip_step_active,
    &avx2_impl::fold_ip_step_active_runs,
    &avx2_impl::showdown_oop_full,
    &avx2_impl::showdown_ip_full,
    &avx2_impl::showdown_oop_signed_zero_rake,
    &avx2_impl::showdown_ip_signed_zero_rake,
    &avx2_impl::showdown_oop_signed_count_zero_rake,
    &avx2_impl::showdown_ip_signed_count_zero_rake,
    &avx2_impl::showdown_oop_full_active,
    &avx2_impl::showdown_ip_full_active,
    &avx2_impl::showdown_oop_full_active_runs,
    &avx2_impl::showdown_oop_full_active_opp_blocks,
    &avx2_impl::showdown_ip_full_active_opp_blocks,
    &avx2_impl::showdown_ip_full_active_runs,
    &avx2_impl::showdown_oop_full_active2,
    &avx2_impl::showdown_ip_full_active2,
    &avx2_impl::showdown_oop_full_batch,
};

}  // namespace deepsolver::cpu_simd
