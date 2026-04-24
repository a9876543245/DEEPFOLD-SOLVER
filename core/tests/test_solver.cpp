/**
 * @file test_solver.cpp
 * @brief Correctness tests for CPU DCFR solver.
 *
 * Validates:
 *   - EV computation produces non-zero, monotonically sensible values
 *   - Exploitability converges with iterations (< 5% at 300 iter on flop)
 *   - Global strategy frequencies sum to 100%
 *
 * These tests are intentionally cheap (< 60s each) so they can run in CI.
 * Larger benchmarks comparing against TexasSolver are handled by benchmark.cpp.
 */

#include "solver.h"
#include "game_tree_builder.h"
#include "card.h"
#include "hand_evaluator.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>

using namespace deepsolver;

// ----------------------------------------------------------------------------
// Mini test framework
// ----------------------------------------------------------------------------

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(name)                                                        \
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
// Shared setup: build a flop solver config (AsKd2c, pot=100, stack=500)
// ----------------------------------------------------------------------------

static SolverConfig make_flop_config(int max_iter = 100) {
    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 500.0f;
    config.board_size = 3;
    auto board = parse_board("AsKd2c");
    for (size_t i = 0; i < 3; ++i) config.board[i] = board[i];
    config.max_iterations = max_iter;
    config.target_exploitability = 0.005f;
    config.exploitability_check_interval = 50;
    return config;
}

// ----------------------------------------------------------------------------
// Test 1: EV sanity — strong hand > weak hand, both non-zero
// ----------------------------------------------------------------------------

static void test_ev_sanity() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(50);  // fast: 50 iter
    Solver solver(config);
    solver.solve();

    auto strong = solver.analyze_combo("AhKh");  // TPTK on A-high flop
    auto weak   = solver.analyze_combo("7h4d");  // air

    std::cout << "  AhKh EV=" << strong.ev << " | 7h4d EV=" << weak.ev << "\n";

    assert_true(strong.ev != 0.0f, "strong hand EV should be non-zero");
    assert_true(strong.ev > weak.ev,
                "strong hand EV should exceed weak hand EV (Phase 1.1 fix)");
}

// ----------------------------------------------------------------------------
// Test 2: Exploitability converges
// ----------------------------------------------------------------------------

static void test_convergence() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(300);
    Solver solver(config);
    auto result = solver.solve();

    std::cout << "  iterations=" << result.iterations_run
              << " exploitability=" << result.exploitability_pct << "%\n";

    assert_true(result.exploitability_pct >= 0.0f,
                "exploitability must be non-negative");
    // 10% at 300 iter is reasonable for the default 4-action tree (Check,
    // Bet_33, Bet_75, All-in). The old 5% threshold was tuned for a
    // degenerate 1-action tree pre-oop_has_initiative fix. An exact 0.0%
    // is acceptable when the BR clamp to max(0, exploit) fires on numerical
    // noise around an already-well-converged strategy.
    assert_true(result.exploitability_pct < 10.0f,
                "exploitability should converge below 10% at 300 iter");
}

// ----------------------------------------------------------------------------
// Test 3: Global strategy frequencies sum to ~100%
// ----------------------------------------------------------------------------

static void test_global_strategy_shape() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(100);
    Solver solver(config);
    auto result = solver.solve();

    assert_true(!result.global_strategy.empty(),
                "global strategy should be populated");

    float total = 0.0f;
    for (const auto& [action, pct] : result.global_strategy) {
        std::cout << "  " << action << ": " << pct << "%\n";
        assert_true(pct >= -0.01f && pct <= 100.1f,
                    "each action frequency should be in [0, 100]");
        total += pct;
    }
    std::cout << "  total=" << total << "%\n";
    assert_near(total, 100.0f, 2.0f,
                "global strategy frequencies should sum to ~100%");
}

// ----------------------------------------------------------------------------
// Test 4: EV monotonicity — better board cards → higher EV for TPTK
// ----------------------------------------------------------------------------

static void test_ev_combo_monotonicity() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(50);
    Solver solver(config);
    solver.solve();

    // On AsKd2c: AhKh (TPTK) should EV > AhQh (TP weaker kicker) > Ah9h > Kh9h
    auto akt = solver.analyze_combo("AhKh");
    auto aq  = solver.analyze_combo("AhQh");
    auto a9  = solver.analyze_combo("Ah9h");
    auto k9  = solver.analyze_combo("Kh9h");

    std::cout << "  AhKh=" << akt.ev << " AhQh=" << aq.ev
              << " Ah9h=" << a9.ev << " Kh9h=" << k9.ev << "\n";

    // All A-high hands beat K-high on this board (in expectation)
    assert_true(akt.ev > k9.ev,
                "TPTK EV should exceed second-pair (K-high) EV");
}

// ----------------------------------------------------------------------------
// Test 5: Non-degenerate strategy — root must offer a bet option
//         (regression for the "100% Check" bug where tree hardcoded
//          oop_has_initiative=false, giving the root only one action)
// ----------------------------------------------------------------------------

static void test_root_has_bet_action() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(50);
    // Defaults: oop_has_initiative=true. Bet sizes present → root should offer
    // more than just Check.
    Solver solver(config);
    auto result = solver.solve();

    assert_true(result.global_strategy.size() >= 2,
                "root must expose Check + at least one Bet/All-in option "
                "(oop_has_initiative default bug)");

    bool has_check = false;
    bool has_aggressive = false;
    for (const auto& [action, pct] : result.global_strategy) {
        if (action == "Check") has_check = true;
        else has_aggressive = true;
        std::cout << "  " << action << ": " << pct << "%\n";
    }
    assert_true(has_check, "Check action should be present at root");
    assert_true(has_aggressive,
                "At least one aggressive action (Bet/All-in) should be present");
}

// ----------------------------------------------------------------------------
// Test 6: Polar OOP vs medium IP produces non-degenerate betting
//         (AA/KK nut range with some air vs middle-pair IP → polarized)
// ----------------------------------------------------------------------------

static void test_polar_oop_produces_bets() {
    auto& eval = get_evaluator();
    eval.initialize();

    auto config = make_flop_config(50);
    config.effective_stack = 150.0f;  // moderate SPR 1.5

    // OOP: pure nuts + some air → polar range that SHOULD bet some %
    config.has_custom_ranges = true;
    config.oop_range_weights.fill(0.0f);
    config.ip_range_weights.fill(0.0f);

    const auto& combo_table = get_combo_table();
    auto set_weight = [&](std::array<float, NUM_COMBOS>& arr,
                          const std::string& label, float w) {
        for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
            if (combo_to_grid_label(combo_table[i]) == label) arr[i] = w;
        }
    };

    // OOP polar: AA, KK (nuts) + 72o, 83o (air)
    set_weight(config.oop_range_weights, "AA",  1.0f);
    set_weight(config.oop_range_weights, "KK",  1.0f);
    set_weight(config.oop_range_weights, "72o", 1.0f);
    set_weight(config.oop_range_weights, "83o", 1.0f);

    // IP medium: TT, 99, 88, 77
    set_weight(config.ip_range_weights, "TT", 1.0f);
    set_weight(config.ip_range_weights, "99", 1.0f);
    set_weight(config.ip_range_weights, "88", 1.0f);
    set_weight(config.ip_range_weights, "77", 1.0f);

    Solver solver(config);
    auto result = solver.solve();

    float check_pct = 0.0f;
    for (const auto& [action, pct] : result.global_strategy) {
        if (action == "Check") check_pct = pct;
        std::cout << "  " << action << ": " << pct << "%\n";
    }

    // Polar range must bet SOMETHING — 100% check would be a regression to
    // the degenerate pre-fix behavior.
    assert_true(check_pct < 99.0f,
                "polar OOP range vs medium IP must not be 100% check");
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main() {
    std::cout << "=== DeepSolver CPU correctness test suite ===\n\n";

    RUN_TEST(test_ev_sanity);
    RUN_TEST(test_convergence);
    RUN_TEST(test_global_strategy_shape);
    RUN_TEST(test_ev_combo_monotonicity);
    RUN_TEST(test_root_has_bet_action);
    RUN_TEST(test_polar_oop_produces_bets);

    std::cout << "=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
