/**
 * @file test_parity.cpp
 * @brief Cross-configuration parity tests for the CPU CFR backend.
 *
 * Why this exists (v1.6.0): the levelized CPU backend is about 5x faster than
 * reference but for now is opt-in. Before flipping the GUI default to
 * levelized, we need a cheap CI gate that catches latent divergence between:
 *
 *   1. reference (recursive scratch arena) vs. levelized (BFS-flat) backends
 *   2. scalar vs. AVX2 SIMD policies
 *   3. cpu_threads=1 vs. cpu_threads=N (smoke check; until Step C lands the
 *      thread cap is mostly informational, but the test exercises the API
 *      surface and will exercise real concurrency once Step C clamps OMP).
 *
 * Tolerances are loose enough to absorb FP non-determinism from OMP reduction
 * order and SIMD vs scalar ordering, but tight enough to catch real bugs:
 *   - EV: 1.0 chip absolute (root EV magnitude is ~5-30 chips)
 *   - global_strategy action freq: 2.0% absolute (out of 100%)
 *
 * Iteration count is intentionally small (60). Strategy doesn't need to be
 * converged — it needs to be *the same* up to numerical noise across the two
 * configurations. Convergence is checked by test_solver.cpp.
 */

#include "solver.h"
#include "card.h"
#include "hand_evaluator.h"
#include "cpu_simd.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deepsolver;

// ----------------------------------------------------------------------------
// Mini test framework (mirrors test_solver.cpp's style for consistency).
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
            std::cout << "[PASS] " #name << "\n\n";                            \
        } catch (const std::exception& e) {                                    \
            std::cout << "[FAIL] " #name ": " << e.what() << "\n\n";           \
        } catch (...) {                                                        \
            std::cout << "[FAIL] " #name ": unknown exception\n\n";            \
        }                                                                      \
    } while (0)

static void assert_true(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error("assertion failed: " + msg);
}

static void assert_near(float a, float b, float tol, const std::string& msg) {
    if (std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << msg << " (a=" << a << " b=" << b << " tol=" << tol << ")";
        throw std::runtime_error(oss.str());
    }
}

// ----------------------------------------------------------------------------
// Shared setup. Tiny rainbow flop, 60 iter — fast, deterministic enough for
// parity assertions, well under the 10-second smoke budget.
// ----------------------------------------------------------------------------

static SolverConfig make_parity_config() {
    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 500.0f;
    config.board_size = 3;
    auto board = parse_board("AsKd2c");
    for (size_t i = 0; i < 3; ++i) config.board[i] = board[i];
    config.max_iterations = 60;
    config.target_exploitability = 0.0f;       // run all 60 iters, no early stop
    config.exploitability_check_interval = 1000;  // disable interim checks
    return config;
}

// Build a label→pct map from a solver result so two configs can be compared
// even if action ordering shifts.
static std::map<std::string, float> strategy_map(
    const std::vector<std::pair<std::string, float>>& gs)
{
    std::map<std::string, float> m;
    for (const auto& [k, v] : gs) m[k] = v;
    return m;
}

// Compare two global_strategy maps. Every action present in either side must
// match within `tol` percentage points, and the action sets must agree.
static void assert_strategy_close(
    const std::map<std::string, float>& a,
    const std::map<std::string, float>& b,
    float tol_pct,
    const std::string& label)
{
    if (a.size() != b.size()) {
        std::ostringstream oss;
        oss << label << ": action set size mismatch (a=" << a.size()
            << " b=" << b.size() << ")";
        throw std::runtime_error(oss.str());
    }
    for (const auto& [action, pct_a] : a) {
        auto it = b.find(action);
        if (it == b.end()) {
            throw std::runtime_error(
                label + ": action '" + action + "' missing in second result");
        }
        assert_near(pct_a, it->second, tol_pct,
                    label + " — action '" + action + "' diverges");
    }
}

// ----------------------------------------------------------------------------
// Parity 1: reference vs levelized backend. Same SIMD, same threads.
//
// Catches: any algorithmic divergence between the two CFR implementations
// (reach propagation order, regret update timing, terminal payoff calc).
// ----------------------------------------------------------------------------

static void test_parity_reference_vs_levelized() {
    auto& eval = get_evaluator();
    eval.initialize();

    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    // Reference run.
    auto cfg_ref = make_parity_config();
    cfg_ref.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_ref(cfg_ref);
    auto r_ref = s_ref.solve();
    auto akh_ref = s_ref.analyze_combo("AhKh");

    // Levelized run.
    auto cfg_lvl = make_parity_config();
    cfg_lvl.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    Solver s_lvl(cfg_lvl);
    auto r_lvl = s_lvl.solve();
    auto akh_lvl = s_lvl.analyze_combo("AhKh");

    std::cout << "  reference  exploit=" << r_ref.exploitability_pct
              << "%  AhKh ev=" << akh_ref.ev << "\n";
    std::cout << "  levelized  exploit=" << r_lvl.exploitability_pct
              << "%  AhKh ev=" << akh_lvl.ev << "\n";

    assert_near(akh_ref.ev, akh_lvl.ev, 1.0f,
                "AhKh EV should match between reference and levelized");

    auto m_ref = strategy_map(r_ref.global_strategy);
    auto m_lvl = strategy_map(r_lvl.global_strategy);
    assert_strategy_close(m_ref, m_lvl, 2.0f, "ref vs levelized");
}

// ----------------------------------------------------------------------------
// Parity 2: scalar vs AVX2 SIMD policy. Same backend, same threads.
//
// Catches: a divergent kernel implementation in cpu_kernels_avx2.cpp.
// Only meaningful when the host actually supports AVX2 — on a non-AVX2 CPU
// ForceAvx2 would abort, so we skip in that case.
// ----------------------------------------------------------------------------

static void test_parity_scalar_vs_avx2() {
    auto& eval = get_evaluator();
    eval.initialize();

    if (!cpu_simd::cpuid_supports_avx2_fma_os()) {
        std::cout << "  skipped — host does not support AVX2/FMA\n";
        return;
    }

    // Force scalar.
    cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceScalar);
    auto cfg_s = make_parity_config();
    cfg_s.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_scalar(cfg_s);
    auto r_scalar = s_scalar.solve();
    auto akh_s = s_scalar.analyze_combo("AhKh");

    // Force AVX2.
    cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceAvx2);
    auto cfg_a = make_parity_config();
    cfg_a.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_avx(cfg_a);
    auto r_avx = s_avx.solve();
    auto akh_a = s_avx.analyze_combo("AhKh");

    // Restore default for any subsequent test.
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    std::cout << "  scalar  AhKh ev=" << akh_s.ev << "\n";
    std::cout << "  avx2    AhKh ev=" << akh_a.ev << "\n";

    assert_near(akh_s.ev, akh_a.ev, 1.0f,
                "AhKh EV should match between scalar and avx2");

    auto m_s = strategy_map(r_scalar.global_strategy);
    auto m_a = strategy_map(r_avx.global_strategy);
    assert_strategy_close(m_s, m_a, 2.0f, "scalar vs avx2");
}

// ----------------------------------------------------------------------------
// Parity 3: thread-cap consistency on the levelized backend.
//
// Today (pre-Step-C) cpu_threads is plumbed through SolverConfig but
// LevelizedCpuBackend doesn't yet apply num_threads(...) to its OMP loops, so
// 1 and N both fall back to omp_get_max_threads(). This test still passes
// (the strategy is deterministic-up-to-OMP-noise across runs of the same
// thread count) and starts to mean something the moment Step C clamps OMP.
// Tolerance is the same 2.0% — within OMP reduction noise once threading
// actually varies.
// ----------------------------------------------------------------------------

static void test_parity_levelized_thread_cap() {
    auto& eval = get_evaluator();
    eval.initialize();

    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    auto cfg_one = make_parity_config();
    cfg_one.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    cfg_one.cpu_threads = 1;
    Solver s_one(cfg_one);
    auto r_one = s_one.solve();
    auto akh_one = s_one.analyze_combo("AhKh");

    auto cfg_many = make_parity_config();
    cfg_many.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    cfg_many.cpu_threads = 4;
    Solver s_many(cfg_many);
    auto r_many = s_many.solve();
    auto akh_many = s_many.analyze_combo("AhKh");

    std::cout << "  levelized 1T  AhKh ev=" << akh_one.ev
              << " (threads_eff=" << r_one.resources.cpu_threads_effective
              << ")\n";
    std::cout << "  levelized 4T  AhKh ev=" << akh_many.ev
              << " (threads_eff=" << r_many.resources.cpu_threads_effective
              << ")\n";

    assert_near(akh_one.ev, akh_many.ev, 1.0f,
                "AhKh EV should match across thread counts on levelized");

    auto m1 = strategy_map(r_one.global_strategy);
    auto m4 = strategy_map(r_many.global_strategy);
    assert_strategy_close(m1, m4, 2.0f, "levelized 1T vs 4T");
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "=== DeepSolver CPU parity test suite ===\n\n";

    RUN_TEST(test_parity_reference_vs_levelized);
    RUN_TEST(test_parity_scalar_vs_avx2);
    RUN_TEST(test_parity_levelized_thread_cap);

    std::cout << "=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
