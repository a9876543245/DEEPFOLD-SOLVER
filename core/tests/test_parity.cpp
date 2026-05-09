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

#include <chrono>
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
        const auto _t0 = std::chrono::steady_clock::now();                     \
        try {                                                                  \
            name();                                                            \
            ++g_tests_passed;                                                  \
            const auto _t1 = std::chrono::steady_clock::now();                 \
            const double _ms = std::chrono::duration<double, std::milli>(      \
                                   _t1 - _t0).count();                         \
            std::cout << "[PASS] " #name << " (" << _ms << " ms)\n\n";         \
        } catch (const std::exception& e) {                                    \
            const auto _t1 = std::chrono::steady_clock::now();                 \
            const double _ms = std::chrono::duration<double, std::milli>(      \
                                   _t1 - _t0).count();                         \
            std::cout << "[FAIL] " #name ": " << e.what() << " (" << _ms       \
                      << " ms)\n\n";                                           \
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
// v1.8.0 Sprint 0.3: tightened deterministic fixtures.
//
// The original AsKd2c flop fixture above runs the full default bet-sizing
// menu, so the tree branches widely; combined with 60 iter (not converged)
// the tolerances had to stay loose at ±2.0pp / ±1.0 chip. To catch finer
// drift we add two tighter fixtures:
//
//   1. river_no_chance — 5-card board, no further dealing, smaller tree,
//      130 iter. Tightest reproducibility because there's zero chance
//      enumeration.
//   2. flop_limited_sizing — same AsKd2c flop but with a single bet size
//      (Bet_75 only). Smaller branching factor → faster convergence →
//      tighter tolerances at fewer iter.
//
// Tolerance budget for these:
//   AhKh EV: 0.05 chip (vs 1.0 chip on the loose fixture)
//   global_strategy: 0.5pp (vs 2.0pp on the loose fixture)
//
// Why both backends should be bit-equal on these: levelized and reference
// implement the same DCFR update; differences come from FP ordering
// (forward/backward pass aggregation order). With small nc, fewer
// reductions, and no chance integration, those ordering differences are
// vanishing.
// ----------------------------------------------------------------------------

static void test_parity_river_no_chance() {
    auto& eval = get_evaluator();
    eval.initialize();
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    auto make_river = []() {
        SolverConfig sc;
        sc.pot = 100.0f;
        sc.effective_stack = 200.0f;   // shorter SPR → faster convergence
        sc.board_size = 5;
        auto board = parse_board("AsKd7c2h5d");
        for (size_t i = 0; i < 5; ++i) sc.board[i] = board[i];
        sc.max_iterations = 130;
        sc.target_exploitability = 0.0f;
        sc.exploitability_check_interval = 1000;
        return sc;
    };

    auto cfg_ref = make_river();
    cfg_ref.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_ref(cfg_ref);
    auto r_ref = s_ref.solve();
    auto akh_ref = s_ref.analyze_combo("AhKh");

    auto cfg_lvl = make_river();
    cfg_lvl.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    Solver s_lvl(cfg_lvl);
    auto r_lvl = s_lvl.solve();
    auto akh_lvl = s_lvl.analyze_combo("AhKh");

    std::cout << "  river ref  exploit=" << r_ref.exploitability_pct
              << "%  AhKh ev=" << akh_ref.ev << "\n";
    std::cout << "  river lvl  exploit=" << r_lvl.exploitability_pct
              << "%  AhKh ev=" << akh_lvl.ev << "\n";

    assert_near(akh_ref.ev, akh_lvl.ev, 0.05f,
                "river: AhKh EV must match within 0.05 chip");

    auto m_ref = strategy_map(r_ref.global_strategy);
    auto m_lvl = strategy_map(r_lvl.global_strategy);
    assert_strategy_close(m_ref, m_lvl, 0.5f, "river ref vs levelized");
}

static void test_parity_flop_limited_sizing() {
    auto& eval = get_evaluator();
    eval.initialize();
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    auto make_flop = []() {
        SolverConfig sc;
        sc.pot = 100.0f;
        sc.effective_stack = 200.0f;
        sc.board_size = 3;
        auto board = parse_board("AsKd2c");
        for (size_t i = 0; i < 3; ++i) sc.board[i] = board[i];
        sc.max_iterations = 100;
        sc.target_exploitability = 0.0f;
        sc.exploitability_check_interval = 1000;
        // Constrain to a single sizing on flop (skip turn/river — board is
        // 3-card so they're not in the tree anyway). Smaller branching →
        // strategy converges faster, tolerance can tighten.
        sc.bet_sizing.flop_sizes = {0.75f};
        return sc;
    };

    auto cfg_ref = make_flop();
    cfg_ref.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_ref(cfg_ref);
    auto r_ref = s_ref.solve();
    auto akh_ref = s_ref.analyze_combo("AhKh");

    auto cfg_lvl = make_flop();
    cfg_lvl.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    Solver s_lvl(cfg_lvl);
    auto r_lvl = s_lvl.solve();
    auto akh_lvl = s_lvl.analyze_combo("AhKh");

    std::cout << "  flop ltd ref  exploit=" << r_ref.exploitability_pct
              << "%  AhKh ev=" << akh_ref.ev << "\n";
    std::cout << "  flop ltd lvl  exploit=" << r_lvl.exploitability_pct
              << "%  AhKh ev=" << akh_lvl.ev << "\n";

    assert_near(akh_ref.ev, akh_lvl.ev, 0.05f,
                "flop limited: AhKh EV must match within 0.05 chip");

    auto m_ref = strategy_map(r_ref.global_strategy);
    auto m_lvl = strategy_map(r_lvl.global_strategy);
    assert_strategy_close(m_ref, m_lvl, 0.5f, "flop limited ref vs levelized");
}

// ----------------------------------------------------------------------------
// v1.8.1+ out-of-range skip parity. The new optimization skips terminal
// evaluation for combos that are 0-reach from root (excluded from the
// user's range). Mathematically safe because those combos propagate as 0
// throughout the tree. This fixture uses a narrow range (~6% of combos
// in range) where the optimization fires hard, and asserts the result
// matches the dense path (reference backend with full range disabled
// for the OOP combos would naturally produce 0-reach behavior, but we
// just need ref vs lvl agreement to gate against drift).
// ----------------------------------------------------------------------------

static void test_parity_narrow_range_skip() {
    auto& eval = get_evaluator();
    eval.initialize();
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    auto make_cfg = []() {
        SolverConfig sc;
        sc.pot = 100.0f;
        sc.effective_stack = 500.0f;
        sc.board_size = 3;
        auto board = parse_board("AsKd7c");
        for (size_t i = 0; i < 3; ++i) sc.board[i] = board[i];
        sc.max_iterations = 60;
        sc.target_exploitability = 0.0f;
        sc.exploitability_check_interval = 1000;

        // Narrow ranges — set most combos to 0 weight so the skip mask
        // fires on the bulk of the canonical-combo space.
        sc.has_custom_ranges = true;
        sc.oop_range_weights.fill(0.0f);
        sc.ip_range_weights.fill(0.0f);
        const auto& combo_table = get_combo_table();
        auto in = [&](const std::string& label) {
            for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
                if (combo_to_grid_label(combo_table[i]) == label) {
                    sc.oop_range_weights[i] = 1.0f;
                    sc.ip_range_weights[i]  = 1.0f;
                }
            }
        };
        for (auto& l : { std::string("AA"), std::string("KK"), std::string("QQ"),
                         std::string("JJ"), std::string("TT"), std::string("AKs"),
                         std::string("AKo") }) in(l);
        return sc;
    };

    auto cfg_ref = make_cfg();
    cfg_ref.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
    Solver s_ref(cfg_ref);
    auto r_ref = s_ref.solve();
    auto akh_ref = s_ref.analyze_combo("AhKh");

    auto cfg_lvl = make_cfg();
    cfg_lvl.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    Solver s_lvl(cfg_lvl);
    auto r_lvl = s_lvl.solve();
    auto akh_lvl = s_lvl.analyze_combo("AhKh");

    std::cout << "  narrow ref  exploit=" << r_ref.exploitability_pct
              << "%  AhKh ev=" << akh_ref.ev << "\n";
    std::cout << "  narrow lvl  exploit=" << r_lvl.exploitability_pct
              << "%  AhKh ev=" << akh_lvl.ev << "\n";

    assert_near(akh_ref.ev, akh_lvl.ev, 0.05f,
                "narrow range: AhKh EV must match within 0.05 chip");

    auto m_ref = strategy_map(r_ref.global_strategy);
    auto m_lvl = strategy_map(r_lvl.global_strategy);
    assert_strategy_close(m_ref, m_lvl, 0.5f, "narrow range ref vs levelized");
}

// ----------------------------------------------------------------------------
// v1.8.0 Sprint 3: persistent OMP team flag parity.
//
// `cpu_persistent_omp` toggles between (a) one OMP parallel-region per
// level (the v1.5.0 baseline) and (b) one OMP parallel-region per pass
// with `omp for` distributing each level (the candidate). Same node
// processing order, same SIMD kernels, same level dependency via
// implicit barriers — must produce bit-identical results.
//
// This test is what gates Sprint 3 from drifting parity-wise. If the
// candidate ever diverges from the baseline on these fixtures, the most
// likely cause is a missed barrier or a per-thread state leak between
// levels.
// ----------------------------------------------------------------------------

static void test_parity_persistent_omp_toggle() {
    auto& eval = get_evaluator();
    eval.initialize();
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    auto make_cfg = [](bool persistent) {
        SolverConfig sc;
        sc.pot = 100.0f;
        sc.effective_stack = 500.0f;
        sc.board_size = 3;
        auto board = parse_board("AsKd2c");
        for (size_t i = 0; i < 3; ++i) sc.board[i] = board[i];
        sc.max_iterations = 60;
        sc.target_exploitability = 0.0f;
        sc.exploitability_check_interval = 1000;
        sc.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
        sc.cpu_threads = 4;        // multi-thread to actually exercise the diff
        sc.cpu_persistent_omp = persistent;
        return sc;
    };

    Solver s_off(make_cfg(false));
    auto r_off = s_off.solve();
    auto akh_off = s_off.analyze_combo("AhKh");

    Solver s_on(make_cfg(true));
    auto r_on = s_on.solve();
    auto akh_on = s_on.analyze_combo("AhKh");

    std::cout << "  persistent=off  exploit=" << r_off.exploitability_pct
              << "%  AhKh ev=" << akh_off.ev << "\n";
    std::cout << "  persistent=on   exploit=" << r_on.exploitability_pct
              << "%  AhKh ev=" << akh_on.ev << "\n";

    assert_near(akh_off.ev, akh_on.ev, 0.05f,
                "persistent_omp toggle: AhKh EV must match within 0.05 chip");

    auto m_off = strategy_map(r_off.global_strategy);
    auto m_on  = strategy_map(r_on.global_strategy);
    assert_strategy_close(m_off, m_on, 0.5f, "persistent_omp 0 vs 1");
}

// ----------------------------------------------------------------------------
// Main
//
// Supports --suite=fast|extended|all (default all). Split is data-driven from
// measured per-test wall time — the slowest fixtures (forced-scalar SIMD,
// thread-cap sweep, limited-sizing tight tolerance) live in extended so the
// smoke loop stays under ~10s while still gating on the core algorithmic /
// persistent-OMP / narrow-range correctness paths.
//
//   fast      — ref-vs-lvl, persistent-OMP toggle, river_no_chance,
//               narrow_range_skip. ~5–6s wall time.
//   extended  — scalar_vs_avx2 (slow scalar path), levelized_thread_cap,
//               flop_limited_sizing. ~17s wall time.
//   all       — both subsets back-to-back.
// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string suite = "all";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--suite=", 0) == 0) {
            suite = a.substr(8);
        } else if (a == "--suite" && i + 1 < argc) {
            suite = argv[++i];
        }
    }
    if (suite != "fast" && suite != "extended" && suite != "all") {
        std::cerr << "unknown --suite=" << suite
                  << " (valid: fast | extended | all)\n";
        return 2;
    }

    std::cout << "=== DeepSolver CPU parity test suite (suite=" << suite
              << ") ===\n\n";

    if (suite == "fast" || suite == "all") {
        RUN_TEST(test_parity_reference_vs_levelized);
        RUN_TEST(test_parity_persistent_omp_toggle);
        RUN_TEST(test_parity_river_no_chance);
        RUN_TEST(test_parity_narrow_range_skip);
    }
    if (suite == "extended" || suite == "all") {
        RUN_TEST(test_parity_scalar_vs_avx2);
        RUN_TEST(test_parity_levelized_thread_cap);
        RUN_TEST(test_parity_flop_limited_sizing);
    }

    std::cout << "=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
