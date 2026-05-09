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
    const float* opp_reach_w, float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
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
    &avx2_impl::showdown_oop_full_batch,
};

}  // namespace deepsolver::cpu_simd
