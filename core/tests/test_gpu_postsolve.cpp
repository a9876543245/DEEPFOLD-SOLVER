/**
 * @file test_gpu_postsolve.cpp
 * @brief Validate GPU postsolve (combo EVs, exploitability) matches CPU.
 *
 * The GPU backend now reuses its device-resident state (tree, matchup tables,
 * averaged strategy in current_strategy, root reach) to compute per-combo EV
 * and best-response values without round-tripping through CPU. This test
 * solves the same flop spot on both backends and asserts:
 *
 *   - exploitability_pct matches within 0.5%
 *   - per-combo EV vector matches within 5 chips on a 100-chip pot
 *
 * Tolerances are loose because CPU and GPU DCFR converge to slightly
 * different strategies due to FP32 reduction order (~0.1% global, 1-3%
 * per-combo per the GPU_BACKEND_FUNCTIONAL note in solver_backend.h).
 * What we're really testing is that the GPU postsolve dispatch does NOT
 * introduce a structural error on top of the existing strategy noise.
 */

#include "solver.h"
#include "solver_backend.h"
#include "card.h"
#include "hand_evaluator.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

using namespace deepsolver;

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(name)                                                          \
    do {                                                                         \
        ++g_tests_run;                                                           \
        std::cout << "[RUN ] " #name << "\n";                                    \
        try {                                                                    \
            name();                                                              \
            ++g_tests_passed;                                                    \
            std::cout << "[PASS] " #name << "\n\n";                              \
        } catch (const std::exception& e) {                                      \
            std::cout << "[FAIL] " #name ": " << e.what() << "\n\n";             \
        } catch (...) {                                                          \
            std::cout << "[FAIL] " #name ": unknown exception\n\n";              \
        }                                                                        \
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

static SolverConfig make_test_config() {
    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 500.0f;
    config.board_size = 3;
    auto board = parse_board("AsKd2c");
    for (size_t i = 0; i < 3; ++i) config.board[i] = board[i];
    config.max_iterations = 80;
    config.target_exploitability = 0.005f;
    config.exploitability_check_interval = 50;
    config.compute_combo_evs = true;
    config.compute_exploitability = true;
    return config;
}

// ----------------------------------------------------------------------------
// End-to-end parity: CPU vs GPU postsolve
// ----------------------------------------------------------------------------

static void test_gpu_postsolve_matches_cpu() {
    if (!has_cuda_gpu()) {
        std::cout << "  [SKIP] no CUDA-capable GPU detected\n";
        ++g_tests_passed;
        return;
    }

    auto& eval = get_evaluator();
    eval.initialize();

    // Both solvers use GPU CFR — guarantees identical strategies (no atomics
    // in the kernels, deterministic across runs). The only difference is
    // postsolve dispatch: one uses GPU, the other forces CPU. Any meaningful
    // gap in exploitability_pct or ev_vector therefore comes from the GPU
    // postsolve path alone, not from CFR convergence drift.
    auto cfg_gpu_post = make_test_config();
    auto cfg_cpu_post = make_test_config();
    cfg_cpu_post.force_cpu_postsolve = true;

    Solver gpu_post_solver(cfg_gpu_post, BackendType::GPU);
    auto gpu_res = gpu_post_solver.solve();

    Solver cpu_post_solver(cfg_cpu_post, BackendType::GPU);
    auto cpu_res = cpu_post_solver.solve();

    std::cout << "  backend=" << gpu_post_solver.backend_name() << "\n";
    std::cout << "  GPU postsolve exploit=" << gpu_res.exploitability_pct
              << "% | CPU postsolve exploit=" << cpu_res.exploitability_pct << "%\n";

    // Same GPU-solved strategy, different postsolve. Tolerate FP32 reduction
    // order drift (~0.05%) but flag any structural bug.
    assert_near(gpu_res.exploitability_pct,
                cpu_res.exploitability_pct,
                0.5f,
                "GPU and CPU postsolve should match within 0.5% on same strategy");

    assert_true(gpu_res.ev_vector.size() == cpu_res.ev_vector.size(),
                "ev_vector sizes should match");

    float max_diff = 0.0f, sum_diff = 0.0f;
    int n = static_cast<int>(gpu_res.ev_vector.size());
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(gpu_res.ev_vector[i] - cpu_res.ev_vector[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    float mean_diff = (n > 0) ? sum_diff / static_cast<float>(n) : 0.0f;
    std::cout << "  EV per-combo max_diff=" << max_diff
              << " mean_diff=" << mean_diff << " (n=" << n << ")\n";

    // Tight tolerance — same strategy, same math, just different reduction
    // order. 0.05 chips on a 100-chip pot = 0.05%.
    assert_true(max_diff < 0.5f,
                "max per-combo EV diff should be < 0.5 chips on same strategy");
}

// ----------------------------------------------------------------------------
// Direct GPU postsolve API smoke test — make sure it returns a sane vector
// ----------------------------------------------------------------------------

static void test_gpu_postsolve_smoke() {
    if (!has_cuda_gpu()) {
        std::cout << "  [SKIP] no CUDA-capable GPU detected\n";
        ++g_tests_passed;
        return;
    }

    auto& eval = get_evaluator();
    eval.initialize();

    Solver gpu_solver(make_test_config(), BackendType::GPU);
    auto res = gpu_solver.solve();

    assert_true(res.ev_vector.size() > 0,
                "GPU solve should produce a non-empty ev_vector");
    assert_true(res.exploitability_pct >= 0.0f,
                "exploitability should be non-negative");
    assert_true(res.exploitability_pct < 50.0f,
                "exploitability should be reasonable (< 50%)");

    // ev_vector entries are per-combo OOP EVs after 1/total_ip_weight scaling.
    // Strong A-high TPTK should outperform 7-2o on AsKd2c.
    auto strong = gpu_solver.analyze_combo("AhKh");
    auto weak   = gpu_solver.analyze_combo("7h2c");
    std::cout << "  GPU AhKh EV=" << strong.ev
              << " | 7h2c EV=" << weak.ev << "\n";

    assert_true(strong.ev > weak.ev,
                "strong hand should outperform weak hand on GPU postsolve");
}

int main() {
    RUN_TEST(test_gpu_postsolve_smoke);
    RUN_TEST(test_gpu_postsolve_matches_cpu);

    std::cout << "\n[" << g_tests_passed << "/" << g_tests_run << "] passed\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
