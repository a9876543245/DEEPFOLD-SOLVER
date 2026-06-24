/**
 * @file test_cpu_kernels.cpp
 * @brief Per-kernel scalar vs AVX2 parity test.
 *
 * v1.8.0 Sprint 0.2 (no-precision-loss guide). Goal: prove that the AVX2
 * kernels in cpu_kernels_avx2.cpp compute the same math as the scalar
 * fallbacks in cpu_kernels_scalar.cpp, byte-for-byte except for FMA
 * rounding noise. Without this guarantee at the kernel level, any solver
 * parity gate (e.g. test_parity.cpp) is comparing two black boxes — if the
 * AVX2 path drifted, we'd attribute the symptom to the higher-level
 * backend instead of localizing it to one kernel.
 *
 * Coverage: every entry in `cpu_simd::Kernels` (the dispatch table). For
 * each kernel:
 *   - sweep representative lengths (1, 2, 7, 8, 9, 31, 32, 1176, 1326)
 *     so we hit (a) the SIMD-aligned fast path, (b) the tail handler,
 *     (c) lengths < 8 that go straight to the scalar fallback inside the
 *     kernel.
 *   - feed inputs that exercise edge cases: zero, negative, near-zero,
 *     large magnitudes, sparse `valid` masks, all-invalid rows.
 *
 * This file compiles with the SSE2 baseline (no /arch:AVX2) — calls into
 * AVX2 implementations go through the `avx2_kernels` dispatch table, so
 * the test target itself needs to link the AVX2 TU but doesn't emit AVX2
 * opcodes outside of it.
 */

#include "cpu_simd.h"
#include "fold_blocker.h"
#include "showdown_rank_blocker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deepsolver;
using namespace deepsolver::cpu_simd;

// ----------------------------------------------------------------------------
// Mini test harness — same shape as test_solver.cpp / test_parity.cpp.
// ----------------------------------------------------------------------------

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(name)                                                         \
    do {                                                                       \
        ++g_tests_run;                                                         \
        std::cout << "[RUN ] " #name << "\n";                                  \
        try {                                                                  \
            name();                                                            \
            ++g_tests_passed;                                                  \
            std::cout << "[PASS] " #name << "\n";                              \
        } catch (const std::exception& e) {                                    \
            std::cout << "[FAIL] " #name ": " << e.what() << "\n";             \
        }                                                                      \
    } while (0)

static void fail(const std::string& msg) { throw std::runtime_error(msg); }

static void assert_close(float a, float b, float abs_tol, float rel_tol,
                          std::size_t n, const char* kernel, std::size_t i)
{
    const float diff = std::fabs(a - b);
    const float scale = std::max(std::fabs(a), std::fabs(b));
    if (diff <= abs_tol) return;
    if (scale > 0.0f && diff / scale <= rel_tol) return;
    std::ostringstream oss;
    oss << kernel << " mismatch at i=" << i << " (n=" << n
        << "): scalar=" << a << " avx2=" << b
        << " diff=" << diff << " abs_tol=" << abs_tol
        << " rel_tol=" << rel_tol;
    fail(oss.str());
}

static void assert_arrays_close(const std::vector<float>& sa,
                                 const std::vector<float>& aa,
                                 float abs_tol, float rel_tol,
                                 const char* kernel)
{
    if (sa.size() != aa.size()) fail("array size mismatch");
    for (std::size_t i = 0; i < sa.size(); ++i) {
        assert_close(sa[i], aa[i], abs_tol, rel_tol, sa.size(), kernel, i);
    }
}
static void assert_arrays_close(const std::vector<float>& sa,
                                 const std::vector<float>& aa,
                                 float abs_tol, float rel_tol,
                                 const std::string& kernel)
{
    assert_arrays_close(sa, aa, abs_tol, rel_tol, kernel.c_str());
}

// ----------------------------------------------------------------------------
// Shared input fixtures. The lengths span (a) sub-SIMD trivial cases, (b)
// exact SIMD-width boundaries (8, 32), (c) tail-handler cases (7, 9, 31,
// 1176, 1326), and (d) the actual production nc range.
// ----------------------------------------------------------------------------

static const std::vector<std::size_t> kLengths = {
    1, 2, 7, 8, 9, 31, 32, 1176, 1326
};

// Per-kernel tolerances. Scalar vs AVX2 differ only by FMA rounding +
// accumulation order; vector ops with no reduction are bit-exact, ones
// that reduce (showdown_*_full / dot_valid_reach) accumulate noise that
// scales with n.
struct Tol { float abs_; float rel_; };
static constexpr Tol kTolElementwise   = { 1e-6f, 1e-6f };
static constexpr Tol kTolReduce        = { 1e-3f, 1e-5f };
static constexpr Tol kTolAccumulate    = { 2e-3f, 2e-5f };

static std::mt19937& rng() {
    static std::mt19937 r(0xCAFEF00DUL);   // deterministic seed → reproducible
    return r;
}

static std::vector<ActiveRun> active_runs_from_active(
    const std::vector<uint16_t>& active);

// Mixed-magnitude payload that hits zero, negative, near-zero, and large.
// Use this for elementwise kernels (no reduction) where catastrophic
// cancellation isn't a concern. Reductions need `range_floats` instead so
// extreme values don't dominate accumulation-order noise.
static std::vector<float> mixed_floats(std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng());
    // sprinkle some explicit edge values
    if (n > 0)  v[0]   = 0.0f;
    if (n > 1)  v[1]   = -0.5f;
    if (n > 5)  v[5]   = 1e-7f;
    if (n > 6)  v[6]   = 1e6f;
    if (n > 7)  v[7]   = -1e6f;
    return v;
}

// Range-clamped payload for inputs feeding into reductions. No 1e6 spikes
// (which would invalidate dot-product comparisons due to FP non-associativity
// rather than any kernel bug). Production values for reach / valid never
// exceed 1.0 anyway.
static std::vector<float> range_floats(std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng());
    return v;
}

// Sparse valid mask: ~70% ones, 30% zeros (matches the typical density of
// the matchup_valid table after dead-card filtering).
static std::vector<float> valid_mask(std::size_t n) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = (dist(rng()) < 0.7f) ? 1.0f : 0.0f;
    return v;
}

// EV row for showdown: roughly equal split of {-1, 0, 1} when valid,
// 0 elsewhere. The category threshold is >0.5 / <-0.5, so this must include
// true win/lose cells instead of only tie cells.
static std::vector<float> ev_row(std::size_t n, const std::vector<float>& valid) {
    std::uniform_int_distribution<int> dist(0, 2);
    std::vector<float> v(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        if (valid[i] <= 0.0f) continue;
        switch (dist(rng())) {
            case 0: v[i] = -1.0f; break;
            case 1: v[i] =  0.0f; break;
            case 2: v[i] =  1.0f; break;
        }
    }
    return v;
}

// ----------------------------------------------------------------------------
// Generic comparison wrapper for an in-place kernel signature
//   void k(float* x, ...args);
// We run scalar and AVX2 on identical copies of the input, then diff.
// ----------------------------------------------------------------------------

template <typename Fn>
static void run_inplace(Fn&& invoke, std::vector<float> base,
                         const Tol& tol, const char* kernel)
{
    std::vector<float> sa = base;
    std::vector<float> aa = base;
    invoke(scalar_kernels, sa.data());
    invoke(avx2_kernels,   aa.data());
    assert_arrays_close(sa, aa, tol.abs_, tol.rel_, kernel);
}

// ----------------------------------------------------------------------------
// Vector kernel tests — each sweeps `kLengths`.
// ----------------------------------------------------------------------------

static void test_vec_mul_in_place() {
    for (auto n : kLengths) {
        auto a = mixed_floats(n, -10.0f, 10.0f);
        auto b = mixed_floats(n, -10.0f, 10.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_mul_in_place(x, b.data(), n); },
            a, kTolElementwise, "vec_mul_in_place");
    }
}

static void test_vec_mul() {
    for (auto n : kLengths) {
        std::vector<float> a(n, 0.0f);
        std::vector<float> b(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 1.25f);
            b[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * -0.75f);
        }
        if (n > 0) a[0] = 0.0f;
        if (n > 1) b[1] = -0.5f;
        if (n > 5) a[5] = 1e-7f;
        if (n > 6) b[6] = 1e6f;
        std::vector<float> ref(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) ref[i] = a[i] * b[i];

        std::vector<float> sa(n, 0.0f);
        std::vector<float> aa(n, 0.0f);
        scalar_kernels.vec_mul(sa.data(), a.data(), b.data(), n);
        avx2_kernels.vec_mul(aa.data(), a.data(), b.data(), n);
        assert_arrays_close(sa, ref, kTolElementwise.abs_, kTolElementwise.rel_,
                            "vec_mul scalar");
        assert_arrays_close(aa, ref, kTolElementwise.abs_, kTolElementwise.rel_,
                            "vec_mul avx2");
    }
}

static void test_vec_scale_in_place() {
    for (auto n : kLengths) {
        auto a = mixed_floats(n, -10.0f, 10.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_scale_in_place(x, 0.317f, n); },
            a, kTolElementwise, "vec_scale_in_place");
    }
}

static void test_vec_add_in_place() {
    for (auto n : kLengths) {
        auto a = mixed_floats(n, -100.0f, 100.0f);
        auto b = mixed_floats(n, -100.0f, 100.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_add_in_place(x, b.data(), n); },
            a, kTolElementwise, "vec_add_in_place");
    }
}

static void test_vec_axpy() {
    for (auto n : kLengths) {
        auto a = mixed_floats(n, -10.0f, 10.0f);
        auto b = mixed_floats(n, -10.0f, 10.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_axpy(x, 2.5f, b.data(), n); },
            a, kTolElementwise, "vec_axpy");
    }
}

static void test_vec_fmadd() {
    for (auto n : kLengths) {
        auto a = mixed_floats(n, -10.0f, 10.0f);
        auto b = mixed_floats(n, -10.0f, 10.0f);
        auto c = mixed_floats(n, -10.0f, 10.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_fmadd(x, b.data(), c.data(), n); },
            a, kTolElementwise, "vec_fmadd");
    }
}

static void test_vec_dcfr_discount() {
    for (auto n : kLengths) {
        // Mix positive and negative to exercise both branches.
        auto a = mixed_floats(n, -100.0f, 100.0f);
        run_inplace(
            [&](const Kernels& k, float* x) { k.vec_dcfr_discount(x, 0.9f, 0.5f, n); },
            a, kTolElementwise, "vec_dcfr_discount");
    }
}

static void test_vec_pos_add() {
    for (auto n : kLengths) {
        auto regret = mixed_floats(n, -50.0f, 50.0f);
        auto pos_sum = mixed_floats(n, 0.0f, 10.0f);
        std::vector<float> sa = pos_sum;
        std::vector<float> aa = pos_sum;
        scalar_kernels.vec_pos_add(sa.data(), regret.data(), n);
        avx2_kernels.vec_pos_add(aa.data(), regret.data(), n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "vec_pos_add");
    }
}

static void test_vec_pos_normalize() {
    for (auto n : kLengths) {
        auto regret = mixed_floats(n, -50.0f, 50.0f);
        auto inv_pos_sum = mixed_floats(n, 0.0f, 1.0f);
        auto uniform_or_zero = mixed_floats(n, 0.0f, 1.0f);
        std::vector<float> sa(n, 0.0f);
        std::vector<float> aa(n, 0.0f);
        scalar_kernels.vec_pos_normalize(
            sa.data(), regret.data(), inv_pos_sum.data(), uniform_or_zero.data(), n);
        avx2_kernels.vec_pos_normalize(
            aa.data(), regret.data(), inv_pos_sum.data(), uniform_or_zero.data(), n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "vec_pos_normalize");
    }
}

static void test_vec_pos_normalize2() {
    for (auto n : kLengths) {
        std::vector<float> regret0(n, 0.0f);
        std::vector<float> regret1(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            regret0[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 3);
            regret1[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 5);
        }
        if (n >= 1) {
            regret0[0] = -1.0f;
            regret1[0] = -2.0f;
        }
        if (n >= 2) {
            regret0[1] = 0.0f;
            regret1[1] = 0.0f;
        }
        if (n >= 3) {
            regret0[2] = 7.0f;
            regret1[2] = -3.0f;
        }

        std::vector<float> ref0(n, 0.0f);
        std::vector<float> ref1(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const float r0p = (regret0[i] > 0.0f) ? regret0[i] : 0.0f;
            const float r1p = (regret1[i] > 0.0f) ? regret1[i] : 0.0f;
            const float pos_sum = r0p + r1p;
            if (pos_sum > 0.0f) {
                const float inv = 1.0f / pos_sum;
                ref0[i] = r0p * inv;
                ref1[i] = r1p * inv;
            } else {
                ref0[i] = 0.5f;
                ref1[i] = 0.5f;
            }
        }

        std::vector<float> sa0(n, 0.0f);
        std::vector<float> sa1(n, 0.0f);
        std::vector<float> aa0(n, 0.0f);
        std::vector<float> aa1(n, 0.0f);
        scalar_kernels.vec_pos_normalize2(
            sa0.data(), sa1.data(), regret0.data(), regret1.data(), n);
        avx2_kernels.vec_pos_normalize2(
            aa0.data(), aa1.data(), regret0.data(), regret1.data(), n);
        assert_arrays_close(sa0, ref0, kTolElementwise.abs_,
                            kTolElementwise.rel_, "vec_pos_normalize2 scalar0");
        assert_arrays_close(sa1, ref1, kTolElementwise.abs_,
                            kTolElementwise.rel_, "vec_pos_normalize2 scalar1");
        assert_arrays_close(sa0, aa0, kTolElementwise.abs_,
                            kTolElementwise.rel_, "vec_pos_normalize2 avx0");
        assert_arrays_close(sa1, aa1, kTolElementwise.abs_,
                            kTolElementwise.rel_, "vec_pos_normalize2 avx1");
    }
}

static void test_vec_regret_update() {
    for (auto n : kLengths) {
        auto action_val = mixed_floats(n, -100.0f, 100.0f);
        auto node_val = mixed_floats(n, -100.0f, 100.0f);
        auto regret = mixed_floats(n, -10.0f, 10.0f);
        std::vector<float> sa = regret;
        std::vector<float> aa = regret;
        scalar_kernels.vec_regret_update(sa.data(), action_val.data(), node_val.data(), n);
        avx2_kernels.vec_regret_update(aa.data(), action_val.data(), node_val.data(), n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "vec_regret_update");
    }
}

static void test_vec_decay_add() {
    for (auto n : kLengths) {
        auto src = mixed_floats(n, -10.0f, 10.0f);
        auto dst = mixed_floats(n, -10.0f, 10.0f);
        std::vector<float> sa = dst;
        std::vector<float> aa = dst;
        scalar_kernels.vec_decay_add(sa.data(), 0.95f, src.data(), n);
        avx2_kernels.vec_decay_add(aa.data(), 0.95f, src.data(), n);
        // dst = dst*decay + src; FMA on AVX2, separate mul+add on scalar.
        // Allow tiny ULP noise.
        assert_arrays_close(sa, aa, 1e-5f, 1e-6f, "vec_decay_add");
    }
}

static void test_vec_reach_weighted_strat_sum() {
    for (auto n : kLengths) {
        auto reach = mixed_floats(n, 0.0f, 1.0f);
        auto strat = mixed_floats(n, 0.0f, 1.0f);
        auto dst = mixed_floats(n, 0.0f, 100.0f);
        std::vector<float> sa = dst;
        std::vector<float> aa = dst;
        scalar_kernels.vec_reach_weighted_strat_sum(sa.data(), 0.85f, reach.data(), strat.data(), n);
        avx2_kernels.vec_reach_weighted_strat_sum(aa.data(), 0.85f, reach.data(), strat.data(), n);
        // FMA rounding on AVX2 vs separate mul+add on scalar — allow ULP-level diff.
        assert_arrays_close(sa, aa, 1e-5f, 1e-6f, "vec_reach_weighted_strat_sum");
    }
}

// ----------------------------------------------------------------------------
// Reduction kernel tests — output is scalar; tolerance widens with n.
// ----------------------------------------------------------------------------

static void test_dot_valid_reach() {
    for (auto n : kLengths) {
        auto valid = valid_mask(n);
        auto reach = range_floats(n, 0.0f, 1.0f);  // production values are probabilities
        float sa = scalar_kernels.dot_valid_reach(valid.data(), reach.data(), n);
        float aa = avx2_kernels.dot_valid_reach(valid.data(), reach.data(), n);
        assert_close(sa, aa, kTolReduce.abs_, kTolReduce.rel_, n,
                     "dot_valid_reach", 0);

        std::vector<uint16_t> active;
        for (std::size_t i = 0; i < n; ++i) {
            if ((i % 5) == 0 || (i % 11) == 3) {
                active.push_back(static_cast<uint16_t>(i));
            }
        }
        float s_active = scalar_kernels.dot_valid_reach_active(
            valid.data(), reach.data(), active.data(), active.size());
        float a_active = avx2_kernels.dot_valid_reach_active(
            valid.data(), reach.data(), active.data(), active.size());
        assert_close(s_active, a_active, kTolReduce.abs_, kTolReduce.rel_, n,
                     "dot_valid_reach_active", 0);

        auto active_runs = active_runs_from_active(active);
        float s_runs = scalar_kernels.dot_valid_reach_active_runs(
            valid.data(), reach.data(), active_runs.data(), active_runs.size());
        float a_runs = avx2_kernels.dot_valid_reach_active_runs(
            valid.data(), reach.data(), active_runs.data(), active_runs.size());
        assert_close(s_runs, s_active, kTolReduce.abs_, kTolReduce.rel_, n,
                     "dot_valid_reach_active_runs (scalar ref)", 0);
        assert_close(a_runs, s_active, kTolReduce.abs_, kTolReduce.rel_, n,
                     "dot_valid_reach_active_runs (avx2 ref)", 0);
    }
}

// ----------------------------------------------------------------------------
// Scalar-output-into-array (showdown_oop_inner / showdown_ip_step) — covered
// by the _full versions below, but also smoke-tested here for the per-c
// path that's still used by tests / future tools.
// ----------------------------------------------------------------------------

static void test_showdown_oop_inner() {
    for (auto n : kLengths) {
        auto valid = valid_mask(n);
        auto ev    = ev_row(n, valid);
        auto reach = mixed_floats(n, 0.0f, 1.0f);
        float sa = scalar_kernels.showdown_oop_inner(
            ev.data(), valid.data(), reach.data(), 50.0f, -50.0f, 0.0f, n);
        float aa = avx2_kernels.showdown_oop_inner(
            ev.data(), valid.data(), reach.data(), 50.0f, -50.0f, 0.0f, n);
        assert_close(sa, aa, kTolReduce.abs_, kTolReduce.rel_, n,
                     "showdown_oop_inner", 0);
    }
}

static void test_showdown_ip_step() {
    for (auto n : kLengths) {
        auto valid = valid_mask(n);
        auto ev    = ev_row(n, valid);
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        scalar_kernels.showdown_ip_step(sa.data(), ev.data(), valid.data(),
                                         0.7f, 50.0f, -50.0f, 0.0f, n);
        avx2_kernels.showdown_ip_step(aa.data(), ev.data(), valid.data(),
                                       0.7f, 50.0f, -50.0f, 0.0f, n);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_ip_step");
    }
}

static void test_fold_ip_step() {
    for (auto n : kLengths) {
        auto valid = valid_mask(n);
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        scalar_kernels.fold_ip_step(sa.data(), valid.data(), 0.42f, nullptr, n);
        avx2_kernels.fold_ip_step(aa.data(), valid.data(), 0.42f, nullptr, n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "fold_ip_step (no-mask)");

        std::vector<uint8_t> skip(n, 0);
        for (std::size_t i = 0; i < n; ++i) skip[i] = ((i % 5) == 1) ? 1 : 0;
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(aa.begin(), aa.end(), 0.0f);
        scalar_kernels.fold_ip_step(sa.data(), valid.data(), 0.42f, skip.data(), n);
        avx2_kernels.fold_ip_step(aa.data(), valid.data(), 0.42f, skip.data(), n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "fold_ip_step (mask)");

        std::vector<float> masked_ref = sa;
        std::vector<uint16_t> active;
        active.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!skip[i]) active.push_back(static_cast<uint16_t>(i));
        }
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(aa.begin(), aa.end(), 0.0f);
        scalar_kernels.fold_ip_step_active(
            sa.data(), valid.data(), 0.42f, active.data(), active.size());
        avx2_kernels.fold_ip_step_active(
            aa.data(), valid.data(), 0.42f, active.data(), active.size());
        assert_arrays_close(sa, masked_ref, kTolElementwise.abs_,
                            kTolElementwise.rel_, "fold_ip_step_active (ref)");
        assert_arrays_close(aa, masked_ref, kTolElementwise.abs_,
                            kTolElementwise.rel_, "fold_ip_step_active (avx2)");

        auto active_runs = active_runs_from_active(active);
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(aa.begin(), aa.end(), 0.0f);
        scalar_kernels.fold_ip_step_active_runs(
            sa.data(), valid.data(), 0.42f, active_runs.data(), active_runs.size());
        avx2_kernels.fold_ip_step_active_runs(
            aa.data(), valid.data(), 0.42f, active_runs.data(), active_runs.size());
        assert_arrays_close(sa, masked_ref, kTolElementwise.abs_,
                            kTolElementwise.rel_, "fold_ip_step_active_runs (ref)");
        assert_arrays_close(aa, masked_ref, kTolElementwise.abs_,
                            kTolElementwise.rel_, "fold_ip_step_active_runs (avx2)");
    }
}

// ----------------------------------------------------------------------------
// Full-matrix kernels (showdown_oop_full / showdown_ip_full). Largest input
// size matters most because the kernel is O(n²); we still sweep a few small
// sizes to exercise the inner-loop tail path.
// ----------------------------------------------------------------------------

static void test_fold_blocker_dense() {
    const auto root_board = parse_board("AsKd7c");
    const IsomorphismMapping iso = compute_isomorphism(
        root_board.data(), static_cast<uint8_t>(root_board.size()));
    const uint16_t nc = iso.num_canonical;
    const auto& combo_table = get_combo_table();

    std::vector<float> reach(nc, 0.0f);
    for (uint16_t c = 0; c < nc; ++c) {
        reach[c] = (c % 13 == 0)
            ? 0.0f
            : static_cast<float>((c * 37u) % 101u) / 100.0f;
    }

    for (const char* board_str : {"AsKd7c", "AsKd7c2h", "AsKd7c2h5d"}) {
        const auto full_board = parse_board(board_str);
        const CardMask board_mask = board_to_mask(
            full_board.data(), static_cast<uint8_t>(full_board.size()));

        std::vector<uint8_t> skip(nc, 0u);
        for (uint16_t c = 0; c < nc; ++c) {
            skip[c] = ((c % 17u) == 3u) ? 1u : 0u;
        }

        std::vector<float> ref(nc, 0.0f), got(nc + 8, -123.0f);
        constexpr float kPayoff = 37.25f;

        for (uint16_t ci = 0; ci < nc; ++ci) {
            if (skip[ci]) {
                ref[ci] = 0.0f;
                continue;
            }
            const auto& originals_i = iso.canonical_to_originals[ci];
            float acc = 0.0f;
            for (uint16_t cj = 0; cj < nc; ++cj) {
                const auto& originals_j = iso.canonical_to_originals[cj];
                int valid = 0;
                for (uint16_t oi : originals_i) {
                    const Combo& combo_i = combo_table[oi];
                    const CardMask mask_i =
                        card_to_mask(combo_i.cards[0])
                        | card_to_mask(combo_i.cards[1]);
                    if (mask_i & board_mask) continue;
                    for (uint16_t oj : originals_j) {
                        const Combo& combo_j = combo_table[oj];
                        const CardMask mask_j =
                            card_to_mask(combo_j.cards[0])
                            | card_to_mask(combo_j.cards[1]);
                        if (mask_j & board_mask) continue;
                        if (mask_i & mask_j) continue;
                        ++valid;
                    }
                }
                const unsigned denom = std::max(
                    1u,
                    static_cast<unsigned>(
                        originals_i.size() * originals_j.size()));
                const float valid_frac =
                    static_cast<float>(valid) / static_cast<float>(denom);
                acc += valid_frac * reach[cj]
                     * static_cast<float>(iso.canonical_weights[cj]);
            }
            ref[ci] = kPayoff * acc;
        }

        fold_blocker::fold_dense(
            iso, board_mask, reach.data(), skip.data(),
            kPayoff, got.data(), got.size());

        assert_arrays_close(
            ref, std::vector<float>(got.begin(), got.begin() + nc),
            2e-3f, 2e-5f,
            std::string("fold_blocker_dense ") + board_str);
        for (std::size_t i = nc; i < got.size(); ++i) {
            if (got[i] != 0.0f) fail("fold_blocker_dense did not zero padding");
        }

        const auto metadata = fold_blocker::build_metadata(iso, board_mask);
        if (!metadata.valid) fail("fold_blocker metadata should be valid");
        std::vector<float> precomputed_got(nc + 8, -123.0f);
        fold_blocker::fold_dense_precomputed(
            metadata, reach.data(), skip.data(),
            kPayoff, precomputed_got.data(), precomputed_got.size());
        assert_arrays_close(
            ref,
            std::vector<float>(
                precomputed_got.begin(), precomputed_got.begin() + nc),
            2e-3f, 2e-5f,
            std::string("fold_blocker_dense_precomputed ") + board_str);
        for (std::size_t i = nc; i < precomputed_got.size(); ++i) {
            if (precomputed_got[i] != 0.0f) {
                fail("fold_blocker_dense_precomputed did not zero padding");
            }
        }

        std::vector<uint16_t> self_active;
        std::vector<uint16_t> opp_active;
        std::vector<uint8_t> self_active_mask(nc, 0u);
        std::vector<float> active_reach(nc, 0.0f);
        for (uint16_t c = 0; c < nc; ++c) {
            if ((c % 7u) == 2u || (c % 29u) == 0u) {
                self_active.push_back(c);
                self_active_mask[c] = 1u;
            }
            if (reach[c] != 0.0f && ((c % 5u) == 1u || (c % 31u) == 0u)) {
                opp_active.push_back(c);
                active_reach[c] = reach[c];
            }
        }

        std::vector<float> active_ref(nc + 8, -123.0f);
        fold_blocker::fold_dense(
            iso, board_mask, active_reach.data(), nullptr,
            kPayoff, active_ref.data(), active_ref.size());

        std::vector<float> active_got(nc + 8, -123.0f);
        fold_blocker::fold_active(
            iso, board_mask, active_reach.data(),
            self_active.data(), self_active.size(),
            opp_active.data(), opp_active.size(),
            true, kPayoff, active_got.data(), active_got.size());

        const std::string active_clear_label =
            std::string("fold_blocker_active (clear) ") + board_str;
        for (uint16_t c = 0; c < nc; ++c) {
            const float want = self_active_mask[c] ? active_ref[c] : 0.0f;
            assert_close(
                active_got[c], want, 2e-3f, 2e-5f,
                nc, active_clear_label.c_str(), c);
        }
        for (std::size_t i = nc; i < active_got.size(); ++i) {
            if (active_got[i] != 0.0f) {
                fail("fold_blocker_active clear did not zero padding");
            }
        }

        std::vector<float> active_precomputed_got(nc + 8, -123.0f);
        fold_blocker::fold_active_precomputed(
            metadata, active_reach.data(),
            self_active.data(), self_active.size(),
            opp_active.data(), opp_active.size(),
            true, kPayoff,
            active_precomputed_got.data(), active_precomputed_got.size());
        for (uint16_t c = 0; c < nc; ++c) {
            const float want = self_active_mask[c] ? active_ref[c] : 0.0f;
            assert_close(
                active_precomputed_got[c], want, 2e-3f, 2e-5f,
                nc, "fold_blocker_active_precomputed (clear)", c);
        }
        for (std::size_t i = nc; i < active_precomputed_got.size(); ++i) {
            if (active_precomputed_got[i] != 0.0f) {
                fail("fold_blocker_active_precomputed clear did not zero padding");
            }
        }

        std::vector<float> active_no_clear(nc + 8, -77.0f);
        fold_blocker::fold_active(
            iso, board_mask, active_reach.data(),
            self_active.data(), self_active.size(),
            opp_active.data(), opp_active.size(),
            false, kPayoff, active_no_clear.data(), active_no_clear.size());

        const std::string active_no_clear_label =
            std::string("fold_blocker_active (no-clear) ") + board_str;
        for (uint16_t c = 0; c < nc; ++c) {
            if (self_active_mask[c]) {
                assert_close(
                    active_no_clear[c], active_ref[c], 2e-3f, 2e-5f,
                    nc, active_no_clear_label.c_str(), c);
            } else if (active_no_clear[c] != -77.0f) {
                fail("fold_blocker_active no-clear touched inactive lane");
            }
        }
        for (std::size_t i = nc; i < active_no_clear.size(); ++i) {
            if (active_no_clear[i] != 0.0f) {
                fail("fold_blocker_active no-clear did not zero padding");
            }
        }

        std::vector<float> active_precomputed_no_clear(nc + 8, -77.0f);
        fold_blocker::fold_active_precomputed(
            metadata, active_reach.data(),
            self_active.data(), self_active.size(),
            opp_active.data(), opp_active.size(),
            false, kPayoff,
            active_precomputed_no_clear.data(),
            active_precomputed_no_clear.size());

        for (uint16_t c = 0; c < nc; ++c) {
            if (self_active_mask[c]) {
                assert_close(
                    active_precomputed_no_clear[c], active_ref[c],
                    2e-3f, 2e-5f,
                    nc, "fold_blocker_active_precomputed (no-clear)", c);
            } else if (active_precomputed_no_clear[c] != -77.0f) {
                fail("fold_blocker_active_precomputed no-clear touched inactive lane");
            }
        }
        for (std::size_t i = nc; i < active_precomputed_no_clear.size(); ++i) {
            if (active_precomputed_no_clear[i] != 0.0f) {
                fail("fold_blocker_active_precomputed no-clear did not zero padding");
            }
        }
    }
}

static void test_showdown_rank_blocker_dense() {
    const auto root_board = parse_board("AsKd7c");
    const IsomorphismMapping iso = compute_isomorphism(
        root_board.data(), static_cast<uint8_t>(root_board.size()));
    if (!showdown_rank_blocker::supports_singleton_iso(iso)) {
        fail("showdown_rank_blocker fixture expected singleton canonical combos");
    }

    const uint16_t nc = iso.num_canonical;
    const auto& combo_table = get_combo_table();
    const CardMask board_mask = board_to_mask(
        root_board.data(), static_cast<uint8_t>(root_board.size()));

    std::vector<uint16_t> ranks(NUM_COMBOS, UINT16_MAX);
    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        if (combo_table[oi].conflicts_with(board_mask)) continue;
        ranks[oi] = static_cast<uint16_t>(1u + ((oi * 37u) % 257u));
    }

    std::vector<float> valid(static_cast<std::size_t>(nc) * nc, 0.0f);
    std::vector<uint8_t> category(static_cast<std::size_t>(nc) * nc, 0u);
    for (uint16_t ci = 0; ci < nc; ++ci) {
        const uint16_t oi = iso.canonical_to_originals[ci][0];
        const Combo& combo_i = combo_table[oi];
        const CardMask mask_i =
            card_to_mask(combo_i.cards[0]) | card_to_mask(combo_i.cards[1]);
        for (uint16_t cj = 0; cj < nc; ++cj) {
            const uint16_t oj = iso.canonical_to_originals[cj][0];
            const Combo& combo_j = combo_table[oj];
            const CardMask mask_j =
                card_to_mask(combo_j.cards[0]) | card_to_mask(combo_j.cards[1]);
            const std::size_t idx = static_cast<std::size_t>(ci) * nc + cj;
            if ((mask_i & board_mask) || (mask_j & board_mask) || (mask_i & mask_j)) {
                continue;
            }
            valid[idx] = 1.0f;
            if (ranks[oi] < ranks[oj]) category[idx] = 1u;
            else if (ranks[oi] > ranks[oj]) category[idx] = 2u;
            else category[idx] = 3u;
        }
    }

    std::vector<float> reach_w(nc, 0.0f);
    std::vector<uint8_t> skip(nc, 0u);
    for (uint16_t c = 0; c < nc; ++c) {
        reach_w[c] = (c % 19u == 0u)
            ? 0.0f
            : static_cast<float>((c * 29u) % 113u) / 113.0f;
        skip[c] = (c % 23u == 7u) ? 1u : 0u;
    }

    showdown_rank_blocker::Scratch scratch;
    const auto metadata = showdown_rank_blocker::build_metadata(iso, ranks);
    if (!metadata.valid) {
        fail("showdown_rank_blocker metadata should be valid");
    }
    showdown_rank_blocker::Scratch metadata_scratch;
    std::vector<float> ref(nc, 0.0f), got(nc + 8, -123.0f);

    scalar_kernels.showdown_oop_full(
        category.data(), valid.data(), reach_w.data(), skip.data(),
        ref.data(), nc, 48.75f, -50.0f, -1.25f);
    showdown_rank_blocker::showdown_dense_singleton(
        iso, ranks, reach_w.data(), skip.data(), got.data(), got.size(),
        48.75f, -50.0f, -1.25f, scratch);
    assert_arrays_close(ref, std::vector<float>(got.begin(), got.begin() + nc),
                        5e-3f, 1e-4f,
                        "showdown_rank_blocker_dense (oop)");

    std::fill(got.begin(), got.end(), -123.0f);
    showdown_rank_blocker::showdown_dense_singleton_precomputed(
        metadata, reach_w.data(), skip.data(), got.data(), got.size(),
        48.75f, -50.0f, -1.25f, metadata_scratch);
    assert_arrays_close(ref, std::vector<float>(got.begin(), got.begin() + nc),
                        5e-3f, 1e-4f,
                        "showdown_rank_blocker_dense_precomputed (oop)");

    std::fill(ref.begin(), ref.end(), 0.0f);
    std::fill(got.begin(), got.end(), -123.0f);
    scalar_kernels.showdown_oop_full(
        category.data(), valid.data(), reach_w.data(), skip.data(),
        ref.data(), nc, 50.0f, -50.0f, 0.0f);
    showdown_rank_blocker::showdown_dense_singleton_precomputed(
        metadata, reach_w.data(), skip.data(), got.data(), got.size(),
        50.0f, -50.0f, 0.0f, metadata_scratch);
    assert_arrays_close(ref, std::vector<float>(got.begin(), got.begin() + nc),
                        5e-3f, 1e-4f,
                        "showdown_rank_blocker_dense_precomputed (zero tie)");

    std::fill(ref.begin(), ref.end(), 0.0f);
    std::fill(got.begin(), got.end(), -123.0f);
    scalar_kernels.showdown_ip_full(
        category.data(), valid.data(), reach_w.data(), skip.data(),
        ref.data(), nc, 48.75f, -50.0f, -1.25f);
    showdown_rank_blocker::showdown_dense_singleton(
        iso, ranks, reach_w.data(), skip.data(), got.data(), got.size(),
        48.75f, -50.0f, -1.25f, scratch);
    assert_arrays_close(ref, std::vector<float>(got.begin(), got.begin() + nc),
                        5e-3f, 1e-4f,
                        "showdown_rank_blocker_dense (ip)");

    for (std::size_t i = nc; i < got.size(); ++i) {
        if (got[i] != 0.0f) fail("showdown_rank_blocker did not zero padding");
    }

    std::vector<uint16_t> self_active;
    std::vector<uint16_t> opp_active;
    std::vector<uint8_t> self_is_active(nc, 0u);
    std::vector<float> active_reach_w(nc, 0.0f);
    for (uint16_t c = 0; c < nc; ++c) {
        if ((c % 7u) == 0u || (c % 31u) == 5u) {
            self_active.push_back(c);
            self_is_active[c] = 1u;
        }
        if ((c % 11u) == 0u || (c % 37u) == 3u) {
            opp_active.push_back(c);
            active_reach_w[c] = reach_w[c];
        }
    }

    std::vector<float> active_ref(nc, 0.0f);
    std::vector<float> active_got(nc + 8, -123.0f);
    scalar_kernels.showdown_oop_full(
        category.data(), valid.data(), active_reach_w.data(), nullptr,
        active_ref.data(), nc, 48.75f, -50.0f, -1.25f);
    for (uint16_t c = 0; c < nc; ++c) {
        if (!self_is_active[c]) active_ref[c] = 0.0f;
    }
    showdown_rank_blocker::showdown_active_singleton_precomputed(
        metadata, active_reach_w.data(),
        self_active.data(), self_active.size(),
        opp_active.data(), opp_active.size(),
        true, active_got.data(), active_got.size(),
        48.75f, -50.0f, -1.25f, metadata_scratch);
    assert_arrays_close(
        active_ref,
        std::vector<float>(active_got.begin(), active_got.begin() + nc),
        5e-3f, 1e-4f,
        "showdown_rank_blocker_active (clear)");
    for (std::size_t i = nc; i < active_got.size(); ++i) {
        if (active_got[i] != 0.0f) {
            fail("showdown_rank_blocker_active did not zero padding");
        }
    }

    std::fill(active_got.begin(), active_got.end(), -123.0f);
    showdown_rank_blocker::showdown_active_singleton_precomputed(
        metadata, active_reach_w.data(),
        self_active.data(), self_active.size(),
        opp_active.data(), opp_active.size(),
        false, active_got.data(), active_got.size(),
        48.75f, -50.0f, -1.25f, metadata_scratch);
    for (uint16_t c = 0; c < nc; ++c) {
        if (self_is_active[c]) {
            assert_close(active_got[c], active_ref[c],
                         5e-3f, 1e-4f, nc,
                         "showdown_rank_blocker_active (no-clear)", c);
        } else if (active_got[c] != -123.0f) {
            fail("showdown_rank_blocker_active touched inactive lane");
        }
    }
}

// v1.8.2 A2: helper to compute the pre-thresholded category byte from an
// (ev, valid) pair, mirroring precompute_matchups()'s bucketing rules:
//   valid == 0           → 0 (invalid)
//   ev > 0.5             → 1 (win)
//   ev < -0.5            → 2 (lose)
//   else                 → 3 (tie)
static inline uint8_t cat_byte(float ev, float valid) {
    if (valid <= 0.0f)   return 0;
    if (ev    >  0.5f)   return 1;
    if (ev    < -0.5f)   return 2;
    return 3;
}
static std::vector<uint8_t> derive_categories(
    const std::vector<float>& ev, const std::vector<float>& valid)
{
    std::vector<uint8_t> cat(ev.size(), 0);
    for (std::size_t i = 0; i < ev.size(); ++i) {
        cat[i] = cat_byte(ev[i], valid[i]);
    }
    return cat;
}

static std::vector<float> derive_signed_coeff(
    const std::vector<float>& ev, const std::vector<float>& valid)
{
    std::vector<float> coeff(ev.size(), 0.0f);
    for (std::size_t i = 0; i < ev.size(); ++i) {
        const uint8_t cat = cat_byte(ev[i], valid[i]);
        if (cat == 1u) coeff[i] = valid[i];
        else if (cat == 2u) coeff[i] = -valid[i];
    }
    return coeff;
}

static std::vector<int8_t> derive_signed_count(
    const std::vector<float>& ev, const std::vector<float>& valid)
{
    std::vector<int8_t> count(ev.size(), 0);
    for (std::size_t i = 0; i < ev.size(); ++i) {
        const uint8_t cat = cat_byte(ev[i], valid[i]);
        if (cat == 1u) count[i] = static_cast<int8_t>(valid[i]);
        else if (cat == 2u) count[i] = static_cast<int8_t>(-valid[i]);
    }
    return count;
}

static std::vector<uint16_t> active_from_skip(const std::vector<uint8_t>& skip) {
    std::vector<uint16_t> active;
    active.reserve(skip.size());
    for (std::size_t i = 0; i < skip.size(); ++i) {
        if (!skip[i]) active.push_back(static_cast<uint16_t>(i));
    }
    return active;
}

static std::vector<ActiveRun> active_runs_from_active(
    const std::vector<uint16_t>& active)
{
    std::vector<ActiveRun> runs;
    if (active.empty()) return runs;
    uint16_t start = active.front();
    uint16_t prev = start;
    uint16_t count = 1;
    for (std::size_t k = 1; k < active.size(); ++k) {
        const uint16_t c = active[k];
        if (c == static_cast<uint16_t>(prev + 1)) {
            prev = c;
            ++count;
            continue;
        }
        runs.push_back({start, count});
        start = prev = c;
        count = 1;
    }
    runs.push_back({start, count});
    return runs;
}

static std::vector<ActiveRun> active_lane_blocks_from_active(
    const std::vector<uint16_t>& active,
    std::size_t n)
{
    std::vector<ActiveRun> blocks;
    if (active.empty()) return blocks;
    constexpr uint16_t kLane = 8;
    uint16_t block_start =
        static_cast<uint16_t>((active.front() / kLane) * kLane);
    uint16_t block_end =
        static_cast<uint16_t>(std::min<std::size_t>(block_start + kLane, n));
    for (uint16_t c : active) {
        const uint16_t start =
            static_cast<uint16_t>((c / kLane) * kLane);
        const uint16_t end =
            static_cast<uint16_t>(std::min<std::size_t>(start + kLane, n));
        if (start <= block_end) {
            if (end > block_end) block_end = end;
            continue;
        }
        blocks.push_back({
            block_start,
            static_cast<uint16_t>(block_end - block_start)
        });
        block_start = start;
        block_end = end;
    }
    blocks.push_back({
        block_start,
        static_cast<uint16_t>(block_end - block_start)
    });
    return blocks;
}

static void assert_active_nozero_output(
    const std::vector<float>& got,
    const std::vector<float>& ref,
    const std::vector<uint8_t>& skip,
    float sentinel,
    float abs_tol,
    float rel_tol,
    const char* kernel)
{
    if (got.size() != ref.size() || got.size() != skip.size()) {
        fail("array size mismatch");
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (skip[i]) {
            assert_close(got[i], sentinel, 0.0f, 0.0f, got.size(), kernel, i);
        } else {
            assert_close(got[i], ref[i], abs_tol, rel_tol, got.size(), kernel, i);
        }
    }
}

static void test_showdown_oop_full() {
    // n² rapidly dwarfs the full sweep; pick a few representative sizes.
    static const std::vector<std::size_t> kFullLengths = {1, 7, 8, 16, 32, 200, 1176};
    for (auto n : kFullLengths) {
        std::vector<float> ev(n * n, 0.0f), valid(n * n, 0.0f);
        for (std::size_t c = 0; c < n; ++c) {
            auto v_row = valid_mask(n);
            auto e_row = ev_row(n, v_row);
            std::copy(v_row.begin(), v_row.end(), valid.begin() + c * n);
            std::copy(e_row.begin(), e_row.end(), ev.begin()    + c * n);
        }
        auto category = derive_categories(ev, valid);
        auto signed_coeff = derive_signed_coeff(ev, valid);
        auto signed_count = derive_signed_count(ev, valid);
        auto reach = range_floats(n, 0.0f, 1.0f);
        std::vector<float> inv_weights(n, 1.0f);
        // v1.8.1+ kernel takes optional skip_mask. Test both the no-mask
        // (dense) path and the with-mask path with ~30% skipped rows.
        std::vector<uint8_t> skip(n, 0);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t c = 0; c < n; ++c) {
            if (dist(rng()) < 0.3f) skip[c] = 1;
        }
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        // No-mask path:
        scalar_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), nullptr,
            sa.data(), n, 50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), nullptr,
            aa.data(), n, 50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_oop_full (no-mask)");
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_oop_signed_zero_rake(
                signed_coeff.data(), reach.data(), nullptr,
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_oop_signed_zero_rake(
                signed_coeff.data(), reach.data(), nullptr,
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_zero_rake (scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_zero_rake (avx2 ref)");
        }
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), nullptr,
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), nullptr,
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count_zero_rake (scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count_zero_rake (avx2 ref)");
        }
        // With-mask path:
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(aa.begin(), aa.end(), 0.0f);
        scalar_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            sa.data(), n, 50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            aa.data(), n, 50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_oop_full (mask)");
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_oop_signed_zero_rake(
                signed_coeff.data(), reach.data(), skip.data(),
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_oop_signed_zero_rake(
                signed_coeff.data(), reach.data(), skip.data(),
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_zero_rake (mask scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_zero_rake (mask avx2 ref)");
        }
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), skip.data(),
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), skip.data(),
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count_zero_rake (mask scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count_zero_rake (mask avx2 ref)");
        }

        // Non-zero tie payoff exercises the generic rake path, while the
        // common 50/-50/0 calls above hit the zero-rake fast path.
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(aa.begin(), aa.end(), 0.0f);
        scalar_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            sa.data(), n, 48.75f, -50.0f, -1.25f);
        avx2_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            aa.data(), n, 48.75f, -50.0f, -1.25f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_oop_full (rake generic)");

        std::vector<float> masked_ref = sa;
        scalar_kernels.showdown_oop_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            masked_ref.data(), n, 50.0f, -50.0f, 0.0f);
        auto active = active_from_skip(skip);
        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_oop_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_oop_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active (scalar ref)");
        assert_arrays_close(aa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active (avx2 ref)");

        auto active_runs = active_runs_from_active(active);
        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_oop_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_oop_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active_runs (scalar ref)");
        assert_arrays_close(aa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active_runs (avx2 ref)");

        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_oop_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, false);
        avx2_kernels.showdown_oop_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, false);
        assert_active_nozero_output(
            sa, masked_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_oop_full_active (nozero scalar)");
        assert_active_nozero_output(
            aa, masked_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_oop_full_active_runs (nozero avx2)");

        std::vector<uint8_t> opp_skip(n, 0);
        for (std::size_t i = 0; i < n; ++i) opp_skip[i] = ((i % 4) == 1) ? 1 : 0;
        auto opp_active = active_from_skip(opp_skip);
        auto reach_opp_active = reach;
        for (std::size_t i = 0; i < n; ++i) {
            if (opp_skip[i]) reach_opp_active[i] = 0.0f;
        }
        auto opp_blocks = active_lane_blocks_from_active(opp_active, n);

        std::vector<float> dual_ref(n, 0.0f), s2(n, 123.0f), a2(n, 123.0f);
        scalar_kernels.showdown_oop_full(
            category.data(), valid.data(), reach_opp_active.data(), skip.data(),
            dual_ref.data(), n, 50.0f, -50.0f, 0.0f);
        scalar_kernels.showdown_oop_full_active2(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), opp_active.data(), opp_active.size(),
            s2.data(), n, 50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_oop_full_active2(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), opp_active.data(), opp_active.size(),
            a2.data(), n, 50.0f, -50.0f, 0.0f);
        assert_arrays_close(s2, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active2 (scalar ref)");
        assert_arrays_close(a2, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active2 (avx2 ref)");

        std::vector<float> sb(n, 123.0f), ab(n, 123.0f);
        scalar_kernels.showdown_oop_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            sb.data(), n, 50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_oop_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            ab.data(), n, 50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sb, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active_opp_blocks (scalar ref)");
        assert_arrays_close(ab, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_oop_full_active_opp_blocks (avx2 ref)");
    }
}

// v1.8.2 Phase 2 parity: showdown_oop_full_batch must produce numerically
// equivalent output to running showdown_oop_full once per terminal. Tests
// both scalar-vs-AVX2 within batch, AND batch-vs-per-call within scalar
// (so a bug in either impl gets caught).
static void test_showdown_oop_full_batch() {
    static const std::vector<std::size_t> kFullLengths = {7, 8, 16, 32, 200, 1176, 1326};
    static const std::vector<std::size_t> kBatchSizes  = {1, 2, 8, 32, 128, 320};
    for (auto n : kFullLengths) {
        // Build a shared category + valid matrix (matches what the production
        // precompute would emit).
        std::vector<float>   ev_mat(n * n, 0.0f);
        std::vector<float>   valid_mat(n * n, 0.0f);
        std::vector<uint8_t> cat_mat(n * n, 0);
        for (std::size_t c = 0; c < n; ++c) {
            auto v_row = valid_mask(n);
            auto e_row = ev_row(n, v_row);
            std::copy(v_row.begin(), v_row.end(), valid_mat.begin() + c * n);
            std::copy(e_row.begin(), e_row.end(), ev_mat.begin()    + c * n);
        }
        // Re-derive category from (ev, valid) mirroring precompute_matchups.
        for (std::size_t k = 0; k < n * n; ++k) {
            cat_mat[k] = cat_byte(ev_mat[k], valid_mat[k]);
        }

        // skip_mask: ~30% of c slots randomly skipped.
        std::vector<uint8_t> skip(n, 0);
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        for (std::size_t c = 0; c < n; ++c) {
            if (dist01(rng()) < 0.3f) skip[c] = 1;
        }

        for (auto M : kBatchSizes) {
            // Build M independent terminals (each with its own opp_reach_w
            // and payoff coefficients).
            std::vector<std::vector<float>> reaches(M);
            std::vector<std::vector<float>> outs_per_call(M);
            std::vector<std::vector<float>> outs_batch_scalar(M);
            std::vector<std::vector<float>> outs_batch_avx2(M);
            std::vector<float> win_p(M), lose_p(M), tie_p(M);
            std::vector<const float*> reach_ptrs(M);
            std::vector<float*>       out_ptrs_scalar(M);
            std::vector<float*>       out_ptrs_avx2(M);
            std::uniform_real_distribution<float> dpay(-100.0f, 100.0f);
            for (std::size_t t = 0; t < M; ++t) {
                // Use range_floats (not mixed_floats) — showdown is a reduction
                // and 1e6 spikes from mixed_floats blow up FP-non-associativity
                // gaps between batch's row-major outer-c order and per-call's
                // c-major order well past the per-cell tolerance, even though
                // the result is mathematically equivalent.
                reaches[t]            = range_floats(n, 0.0f, 1.0f);
                outs_per_call[t]      = std::vector<float>(n, 0.0f);
                outs_batch_scalar[t]  = std::vector<float>(n, 0.0f);
                outs_batch_avx2[t]    = std::vector<float>(n, 0.0f);
                win_p[t]  = dpay(rng());
                lose_p[t] = dpay(rng());
                tie_p[t]  = dpay(rng());
                reach_ptrs[t]      = reaches[t].data();
                out_ptrs_scalar[t] = outs_batch_scalar[t].data();
                out_ptrs_avx2[t]   = outs_batch_avx2[t].data();
            }

            // Reference: per-terminal calls via the existing scalar kernel.
            for (std::size_t t = 0; t < M; ++t) {
                scalar_kernels.showdown_oop_full(
                    cat_mat.data(), valid_mat.data(), reaches[t].data(),
                    skip.data(), outs_per_call[t].data(), n,
                    win_p[t], lose_p[t], tie_p[t]);
            }

            // Batch scalar.
            scalar_kernels.showdown_oop_full_batch(
                cat_mat.data(), valid_mat.data(), n, M,
                reach_ptrs.data(), skip.data(), out_ptrs_scalar.data(),
                win_p.data(), lose_p.data(), tie_p.data(),
                /*c_lo=*/0, /*c_hi=*/n);

            // Batch AVX2.
            avx2_kernels.showdown_oop_full_batch(
                cat_mat.data(), valid_mat.data(), n, M,
                reach_ptrs.data(), skip.data(), out_ptrs_avx2.data(),
                win_p.data(), lose_p.data(), tie_p.data(),
                /*c_lo=*/0, /*c_hi=*/n);

            // Compare both batch impls against the per-call reference.
            for (std::size_t t = 0; t < M; ++t) {
                const std::string ctx_lbl = " (n=" + std::to_string(n)
                                          + " M=" + std::to_string(M)
                                          + " t=" + std::to_string(t) + ")";
                assert_arrays_close(outs_batch_scalar[t], outs_per_call[t],
                                     kTolAccumulate.abs_, kTolAccumulate.rel_,
                                     "showdown_oop_full_batch (scalar vs per-call)" + ctx_lbl);
                assert_arrays_close(outs_batch_avx2[t], outs_per_call[t],
                                     kTolAccumulate.abs_, kTolAccumulate.rel_,
                                     "showdown_oop_full_batch (avx2 vs per-call)" + ctx_lbl);
            }

            // Also exercise the c-range slicing — split [0, n) into halves
            // and verify the union matches the full-range call.
            std::vector<std::vector<float>> outs_split(M);
            std::vector<float*> split_ptrs(M);
            for (std::size_t t = 0; t < M; ++t) {
                outs_split[t]  = std::vector<float>(n, 0.0f);
                split_ptrs[t]  = outs_split[t].data();
            }
            const std::size_t mid = n / 2;
            avx2_kernels.showdown_oop_full_batch(
                cat_mat.data(), valid_mat.data(), n, M,
                reach_ptrs.data(), skip.data(), split_ptrs.data(),
                win_p.data(), lose_p.data(), tie_p.data(),
                /*c_lo=*/0, /*c_hi=*/mid);
            avx2_kernels.showdown_oop_full_batch(
                cat_mat.data(), valid_mat.data(), n, M,
                reach_ptrs.data(), skip.data(), split_ptrs.data(),
                win_p.data(), lose_p.data(), tie_p.data(),
                /*c_lo=*/mid, /*c_hi=*/n);
            for (std::size_t t = 0; t < M; ++t) {
                assert_arrays_close(outs_split[t], outs_per_call[t],
                                     kTolAccumulate.abs_, kTolAccumulate.rel_,
                                     "showdown_oop_full_batch (split c-range)");
            }
        }
    }
}

static void test_showdown_ip_full() {
    static const std::vector<std::size_t> kFullLengths = {1, 7, 8, 16, 32, 200, 1176};
    for (auto n : kFullLengths) {
        std::vector<float> ev(n * n, 0.0f), valid(n * n, 0.0f);
        for (std::size_t c = 0; c < n; ++c) {
            auto v_row = valid_mask(n);
            auto e_row = ev_row(n, v_row);
            std::copy(v_row.begin(), v_row.end(), valid.begin() + c * n);
            std::copy(e_row.begin(), e_row.end(), ev.begin()    + c * n);
        }
        auto category = derive_categories(ev, valid);
        auto signed_coeff = derive_signed_coeff(ev, valid);
        auto signed_count = derive_signed_count(ev, valid);
        auto reach = mixed_floats(n, 0.0f, 1.0f);
        std::vector<float> inv_weights(n, 1.0f);
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        scalar_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), nullptr, sa.data(), n,
            50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), nullptr, aa.data(), n,
            50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_ip_full (no-mask)");
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_ip_signed_zero_rake(
                signed_coeff.data(), reach.data(), nullptr,
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_ip_signed_zero_rake(
                signed_coeff.data(), reach.data(), nullptr,
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_zero_rake (scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_zero_rake (avx2 ref)");
        }
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), nullptr,
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), nullptr,
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count_zero_rake (scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count_zero_rake (avx2 ref)");
        }

        std::vector<uint8_t> skip(n, 0);
        for (std::size_t i = 0; i < n; ++i) skip[i] = ((i % 7) == 2) ? 1 : 0;
        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), skip.data(), sa.data(), n,
            50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), skip.data(), aa.data(), n,
            50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_ip_full (mask)");
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_ip_signed_zero_rake(
                signed_coeff.data(), reach.data(), skip.data(),
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_ip_signed_zero_rake(
                signed_coeff.data(), reach.data(), skip.data(),
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_zero_rake (mask scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_zero_rake (mask avx2 ref)");
        }
        {
            std::vector<float> ss(n, 0.0f), as(n, 0.0f);
            scalar_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), skip.data(),
                ss.data(), n, 50.0f);
            avx2_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), skip.data(),
                as.data(), n, 50.0f);
            assert_arrays_close(ss, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count_zero_rake (mask scalar ref)");
            assert_arrays_close(as, sa, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count_zero_rake (mask avx2 ref)");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (skip[i] && sa[i] != 0.0f) {
                fail("showdown_ip_full masked output should be zero");
            }
        }

        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), skip.data(), sa.data(), n,
            48.75f, -50.0f, -1.25f);
        avx2_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), skip.data(), aa.data(), n,
            48.75f, -50.0f, -1.25f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_ip_full (rake generic)");

        std::vector<float> masked_ref = sa;
        scalar_kernels.showdown_ip_full(
            category.data(), valid.data(), reach.data(), skip.data(),
            masked_ref.data(), n, 50.0f, -50.0f, 0.0f);
        auto active = active_from_skip(skip);
        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_ip_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_ip_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active (scalar ref)");
        assert_arrays_close(aa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active (avx2 ref)");

        auto active_runs = active_runs_from_active(active);
        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_ip_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_ip_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active_runs (scalar ref)");
        assert_arrays_close(aa, masked_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active_runs (avx2 ref)");

        std::fill(sa.begin(), sa.end(), 123.0f);
        std::fill(aa.begin(), aa.end(), 123.0f);
        scalar_kernels.showdown_ip_full_active(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), sa.data(), n,
            50.0f, -50.0f, 0.0f, false);
        avx2_kernels.showdown_ip_full_active_runs(
            category.data(), valid.data(), reach.data(),
            active_runs.data(), active_runs.size(), aa.data(), n,
            50.0f, -50.0f, 0.0f, false);
        assert_active_nozero_output(
            sa, masked_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_ip_full_active (nozero scalar)");
        assert_active_nozero_output(
            aa, masked_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_ip_full_active_runs (nozero avx2)");

        std::vector<uint8_t> opp_skip(n, 0);
        for (std::size_t i = 0; i < n; ++i) opp_skip[i] = ((i % 4) == 1) ? 1 : 0;
        auto opp_active = active_from_skip(opp_skip);
        auto reach_opp_active = reach;
        for (std::size_t i = 0; i < n; ++i) {
            if (opp_skip[i]) reach_opp_active[i] = 0.0f;
        }
        auto opp_blocks = active_lane_blocks_from_active(opp_active, n);

        std::vector<float> dual_ref(n, 0.0f), s2(n, 123.0f), a2(n, 123.0f);
        scalar_kernels.showdown_ip_full(
            category.data(), valid.data(), reach_opp_active.data(), skip.data(),
            dual_ref.data(), n, 50.0f, -50.0f, 0.0f);
        scalar_kernels.showdown_ip_full_active2(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), opp_active.data(), opp_active.size(),
            s2.data(), n, 50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_ip_full_active2(
            category.data(), valid.data(), reach.data(),
            active.data(), active.size(), opp_active.data(), opp_active.size(),
            a2.data(), n, 50.0f, -50.0f, 0.0f);
        assert_arrays_close(s2, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active2 (scalar ref)");
        assert_arrays_close(a2, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active2 (avx2 ref)");

        std::vector<float> sb(n, 123.0f), ab(n, 123.0f);
        scalar_kernels.showdown_ip_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            sb.data(), n, 50.0f, -50.0f, 0.0f, true);
        avx2_kernels.showdown_ip_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            ab.data(), n, 50.0f, -50.0f, 0.0f, true);
        assert_arrays_close(sb, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active_opp_blocks (scalar ref)");
        assert_arrays_close(ab, dual_ref, kTolAccumulate.abs_,
                            kTolAccumulate.rel_,
                            "showdown_ip_full_active_opp_blocks (avx2 ref)");

        std::fill(sb.begin(), sb.end(), 123.0f);
        std::fill(ab.begin(), ab.end(), 123.0f);
        scalar_kernels.showdown_ip_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            sb.data(), n, 50.0f, -50.0f, 0.0f, false);
        avx2_kernels.showdown_ip_full_active_opp_blocks(
            category.data(), valid.data(), reach_opp_active.data(),
            active.data(), active.size(), opp_blocks.data(), opp_blocks.size(),
            ab.data(), n, 50.0f, -50.0f, 0.0f, false);
        assert_active_nozero_output(
            sb, dual_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_ip_full_active_opp_blocks (nozero scalar)");
        assert_active_nozero_output(
            ab, dual_ref, skip, 123.0f,
            kTolAccumulate.abs_, kTolAccumulate.rel_,
            "showdown_ip_full_active_opp_blocks (nozero avx2)");
    }
}

static void test_showdown_signed_count_weighted_formula() {
    static const std::vector<std::size_t> kWeightedLengths = {3, 8, 31, 200};

    for (auto n : kWeightedLengths) {
        std::vector<float> weights(n), inv_weights(n), reach(n), reach_w(n);
        std::vector<uint8_t> skip(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] = static_cast<float>(1 + (i % 4));
            inv_weights[i] = 1.0f / weights[i];
            reach[i] = ((i % 9) == 0)
                ? 0.0f
                : (0.03f + 0.017f * static_cast<float>((i * 7) % 23));
            reach_w[i] = reach[i] * weights[i];
            skip[i] = ((i % 5) == 3) ? 1u : 0u;
        }

        std::vector<uint8_t> category(n * n, 0);
        std::vector<float> valid(n * n, 0.0f);
        std::vector<int8_t> signed_count(n * n, 0);
        for (std::size_t c = 0; c < n; ++c) {
            for (std::size_t i = 0; i < n; ++i) {
                if (((c * 17 + i * 11) % 19) == 0) continue;

                const int max_count = static_cast<int>(weights[c] * weights[i]);
                const int valid_count = 1
                    + static_cast<int>((c * 5 + i * 7 + 3) % max_count);
                const std::size_t idx = c * n + i;
                valid[idx] = static_cast<float>(valid_count)
                    / (weights[c] * weights[i]);

                switch ((c * 3 + i * 5) % 4) {
                    case 0:
                        category[idx] = 1u;
                        signed_count[idx] = static_cast<int8_t>(valid_count);
                        break;
                    case 1:
                        category[idx] = 2u;
                        signed_count[idx] = static_cast<int8_t>(-valid_count);
                        break;
                    default:
                        category[idx] = 3u;
                        break;
                }
            }
        }

        for (int pass = 0; pass < 2; ++pass) {
            const uint8_t* mask = (pass == 0) ? nullptr : skip.data();
            const bool masked = (pass != 0);
            const std::string suffix = masked ? " (mask)" : " (no-mask)";

            std::vector<float> ref_oop(n, 0.0f), sc_oop(n, 0.0f), ac_oop(n, 0.0f);
            scalar_kernels.showdown_oop_full(
                category.data(), valid.data(), reach_w.data(), mask,
                ref_oop.data(), n, 50.0f, -50.0f, 0.0f);
            scalar_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), mask,
                sc_oop.data(), n, 50.0f);
            avx2_kernels.showdown_oop_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), mask,
                ac_oop.data(), n, 50.0f);
            assert_arrays_close(sc_oop, ref_oop, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count weighted scalar" + suffix);
            assert_arrays_close(ac_oop, ref_oop, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_oop_signed_count weighted avx2" + suffix);

            std::vector<float> ref_ip(n, 0.0f), sc_ip(n, 0.0f), ac_ip(n, 0.0f);
            scalar_kernels.showdown_ip_full(
                category.data(), valid.data(), reach_w.data(), mask,
                ref_ip.data(), n, 50.0f, -50.0f, 0.0f);
            scalar_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), mask,
                sc_ip.data(), n, 50.0f);
            avx2_kernels.showdown_ip_signed_count_zero_rake(
                signed_count.data(), reach.data(), inv_weights.data(), mask,
                ac_ip.data(), n, 50.0f);
            assert_arrays_close(sc_ip, ref_ip, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count weighted scalar" + suffix);
            assert_arrays_close(ac_ip, ref_ip, kTolAccumulate.abs_,
                                kTolAccumulate.rel_,
                                "showdown_ip_signed_count weighted avx2" + suffix);
        }
    }
}

// ----------------------------------------------------------------------------
// Edge cases — very small n, all-zero inputs, all-invalid masks.
// ----------------------------------------------------------------------------

static void test_edge_case_zero_input() {
    // All zeros: every kernel must produce all zeros (or stable scalar).
    const std::size_t n = 32;
    std::vector<float> z(n, 0.0f);
    {
        std::vector<float> sa = z, aa = z;
        scalar_kernels.vec_dcfr_discount(sa.data(), 0.9f, 0.5f, n);
        avx2_kernels.vec_dcfr_discount(aa.data(), 0.9f, 0.5f, n);
        assert_arrays_close(sa, aa, 0.0f, 0.0f, "edge: vec_dcfr_discount(0)");
    }
    {
        // dot of zero × zero = 0, no FP issues
        float sa = scalar_kernels.dot_valid_reach(z.data(), z.data(), n);
        float aa = avx2_kernels.dot_valid_reach(z.data(), z.data(), n);
        if (sa != 0.0f || aa != 0.0f) fail("dot_valid_reach(0,0) != 0");
    }
}

static void test_edge_case_all_invalid_showdown() {
    const std::size_t n = 64;
    std::vector<float> valid(n, 0.0f);    // all invalid
    auto ev = ev_row(n, valid);            // all 0 since valid is 0
    auto reach = mixed_floats(n, 0.0f, 1.0f);
    float sa = scalar_kernels.showdown_oop_inner(
        ev.data(), valid.data(), reach.data(), 50.0f, -50.0f, 0.0f, n);
    float aa = avx2_kernels.showdown_oop_inner(
        ev.data(), valid.data(), reach.data(), 50.0f, -50.0f, 0.0f, n);
    if (sa != 0.0f || aa != 0.0f) fail("showdown_oop_inner(all-invalid) != 0");
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "=== DeepSolver CPU kernel parity (scalar vs AVX2) ===\n\n";

    if (!cpuid_supports_avx2_fma_os()) {
        std::cout << "host has no AVX2/FMA — test is a no-op (PASS)\n";
        return 0;
    }

    RUN_TEST(test_vec_mul_in_place);
    RUN_TEST(test_vec_mul);
    RUN_TEST(test_vec_scale_in_place);
    RUN_TEST(test_vec_add_in_place);
    RUN_TEST(test_vec_axpy);
    RUN_TEST(test_vec_fmadd);
    RUN_TEST(test_vec_dcfr_discount);
    RUN_TEST(test_vec_pos_add);
    RUN_TEST(test_vec_pos_normalize);
    RUN_TEST(test_vec_pos_normalize2);
    RUN_TEST(test_vec_regret_update);
    RUN_TEST(test_vec_decay_add);
    RUN_TEST(test_vec_reach_weighted_strat_sum);
    RUN_TEST(test_dot_valid_reach);
    RUN_TEST(test_showdown_oop_inner);
    RUN_TEST(test_showdown_ip_step);
    RUN_TEST(test_fold_ip_step);
    RUN_TEST(test_fold_blocker_dense);
    RUN_TEST(test_showdown_rank_blocker_dense);
    RUN_TEST(test_showdown_oop_full);
    RUN_TEST(test_showdown_ip_full);
    RUN_TEST(test_showdown_signed_count_weighted_formula);
    RUN_TEST(test_showdown_oop_full_batch);
    RUN_TEST(test_edge_case_zero_input);
    RUN_TEST(test_edge_case_all_invalid_showdown);

    std::cout << "\n=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
