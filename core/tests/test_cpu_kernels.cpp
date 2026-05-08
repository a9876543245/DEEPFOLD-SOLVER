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
static constexpr Tol kTolAccumulate    = { 1e-3f, 1e-5f };

static std::mt19937& rng() {
    static std::mt19937 r(0xCAFEF00DUL);   // deterministic seed → reproducible
    return r;
}

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

// EV row for showdown: roughly equal split of {-0.5, 0, 0.5} when valid,
// 0 elsewhere (matches scalar+SIMD comparison thresholds in the kernel).
static std::vector<float> ev_row(std::size_t n, const std::vector<float>& valid) {
    std::uniform_int_distribution<int> dist(0, 2);
    std::vector<float> v(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        if (valid[i] <= 0.0f) continue;
        switch (dist(rng())) {
            case 0: v[i] = -0.5f; break;
            case 1: v[i] =  0.0f; break;
            case 2: v[i] =  0.5f; break;
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
        scalar_kernels.fold_ip_step(sa.data(), valid.data(), 0.42f, n);
        avx2_kernels.fold_ip_step(aa.data(), valid.data(), 0.42f, n);
        assert_arrays_close(sa, aa, kTolElementwise.abs_, kTolElementwise.rel_,
                            "fold_ip_step");
    }
}

// ----------------------------------------------------------------------------
// Full-matrix kernels (showdown_oop_full / showdown_ip_full). Largest input
// size matters most because the kernel is O(n²); we still sweep a few small
// sizes to exercise the inner-loop tail path.
// ----------------------------------------------------------------------------

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
        auto reach = mixed_floats(n, 0.0f, 1.0f);
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        scalar_kernels.showdown_oop_full(
            ev.data(), valid.data(), reach.data(), sa.data(), n,
            50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_oop_full(
            ev.data(), valid.data(), reach.data(), aa.data(), n,
            50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_oop_full");
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
        auto reach = mixed_floats(n, 0.0f, 1.0f);
        std::vector<float> sa(n, 0.0f), aa(n, 0.0f);
        scalar_kernels.showdown_ip_full(
            ev.data(), valid.data(), reach.data(), sa.data(), n,
            50.0f, -50.0f, 0.0f);
        avx2_kernels.showdown_ip_full(
            ev.data(), valid.data(), reach.data(), aa.data(), n,
            50.0f, -50.0f, 0.0f);
        assert_arrays_close(sa, aa, kTolAccumulate.abs_, kTolAccumulate.rel_,
                            "showdown_ip_full");
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
    RUN_TEST(test_vec_scale_in_place);
    RUN_TEST(test_vec_add_in_place);
    RUN_TEST(test_vec_axpy);
    RUN_TEST(test_vec_fmadd);
    RUN_TEST(test_vec_dcfr_discount);
    RUN_TEST(test_vec_pos_add);
    RUN_TEST(test_vec_pos_normalize);
    RUN_TEST(test_vec_regret_update);
    RUN_TEST(test_vec_decay_add);
    RUN_TEST(test_vec_reach_weighted_strat_sum);
    RUN_TEST(test_dot_valid_reach);
    RUN_TEST(test_showdown_oop_inner);
    RUN_TEST(test_showdown_ip_step);
    RUN_TEST(test_fold_ip_step);
    RUN_TEST(test_showdown_oop_full);
    RUN_TEST(test_showdown_ip_full);
    RUN_TEST(test_edge_case_zero_input);
    RUN_TEST(test_edge_case_all_invalid_showdown);

    std::cout << "\n=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
