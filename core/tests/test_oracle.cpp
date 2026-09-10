/**
 * @file test_oracle.cpp
 * @brief Independent brute-force oracles for the game model (2026-09-09 audit).
 *
 * Every other suite compares two implementations of THIS engine with each
 * other (reference vs levelized, scalar vs AVX2, CPU vs GPU). Agreement proves
 * they compute the same thing, not that the thing is No-Limit Hold'em. These
 * tests instead enumerate the game by hand — opponent hands × runouts × 7-card
 * evaluation — and check the solver's conditional EVs, payoffs and
 * exploitability against that enumeration:
 *
 *   1. chance conditioning      — a runout is drawn from the cards NEITHER
 *                                 player holds (1/44 on a turn, not 1/48)
 *   2. conditional denominators — a hand's EV is conditional on the opponent
 *                                 holding a hand that can be dealt alongside
 *                                 it; exploitability is an expectation over the
 *                                 legal joint deal
 *   3. raked payoffs            — postsolve uses the CFR payoff model: win =
 *                                 half_pot − rake, lose = −half_pot, tie =
 *                                 −rake/2; folds rake the MATCHED pot
 *   4. all-in runout equity     — a pre-river all-in is settled over every
 *                                 remaining runout, not on the current board
 *
 * Each fixture is small enough to enumerate in well under a second. Where a
 * fixed strategy is needed (all-in call, bet/fold), node locks pin it, so the
 * expected value has a closed form the solver cannot influence.
 *
 * Each test was run against the pre-fix engine and failed there (ROADMAP §5
 * rule 2): the old chance rule, denominators and unraked postsolve each move
 * the numbers by 8-16%, far outside the tolerances below.
 */

#include "solver.h"
#include "card.h"
#include "hand_evaluator.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deepsolver;

// ----------------------------------------------------------------------------
// Mini test framework (same shape as test_solver.cpp / test_parity.cpp).
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

static void assert_near(double a, double b, double tol, const std::string& msg) {
    if (std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << msg << " (got=" << a << " expected=" << b << " tol=" << tol << ")";
        throw std::runtime_error(oss.str());
    }
}

// ----------------------------------------------------------------------------
// Fixture helpers
// ----------------------------------------------------------------------------

static void set_board(SolverConfig& cfg, const char* text) {
    auto board = parse_board(text);
    cfg.board_size = static_cast<uint8_t>(board.size());
    for (size_t i = 0; i < board.size(); ++i) cfg.board[i] = board[i];
}

/// No betting anywhere: every node is Check → the hand is checked down.
static void make_checkdown(SolverConfig& cfg) {
    cfg.bet_sizing.flop_sizes.clear();
    cfg.bet_sizing.turn_sizes.clear();
    cfg.bet_sizing.river_sizes.clear();
    cfg.bet_sizing.flop_allin = cfg.bet_sizing.turn_allin = cfg.bet_sizing.river_allin = false;
}

static void fixed_iterations(SolverConfig& cfg, int iters) {
    cfg.max_iterations = iters;
    cfg.target_exploitability = 0.0f;
    cfg.exploitability_check_interval = 1000;
    cfg.dcfr_schedule = SolverConfig::DcfrSchedule::STANDARD;
}

static uint16_t combo_index(const char* text) {
    Card c0 = parse_card(std::string(text).substr(0, 2));
    Card c1 = parse_card(std::string(text).substr(2, 2));
    return Combo(c0, c1).index();
}

static void set_range(std::array<float, NUM_COMBOS>& w,
                      std::initializer_list<const char*> combos, float weight = 1.0f) {
    w.fill(0.0f);
    for (const char* c : combos) w[combo_index(c)] = weight;
}

/// Lock `combo` (or every combo when combo_idx == UINT16_MAX is not allowed by
/// the engine, so callers pass one lock per combo) to a pure action.
static void lock_all_combos(SolverConfig& cfg, const std::string& history,
                            const std::array<float, NUM_COMBOS>& range,
                            size_t action_index, size_t num_actions_upper_bound) {
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        if (range[i] <= 0.0f) continue;
        NodeLockEntry lock;
        lock.history   = history;
        lock.combo_idx = i;
        lock.combo_str = get_combo_table()[i].to_string();
        lock.strategy.assign(num_actions_upper_bound, 0.0f);
        lock.strategy[action_index] = 1.0f;
        cfg.node_locks.push_back(std::move(lock));
    }
}

// ----------------------------------------------------------------------------
// Brute-force oracle: conditional check-down EV of hand (c0,c1) against a
// weighted opponent range on `board` with `board_size` cards, settling the
// showdown over EVERY completion of the board (0, 1 or 2 more cards) that
// neither hand blocks. Payoffs: win = half_pot − rake, lose = −half_pot,
// tie = −rake/2, rake on the full pot. The result is divided by the mass of
// the opponent hands that can actually be dealt alongside (c0,c1).
// ----------------------------------------------------------------------------

struct Oracle {
    const HandEvaluator& eval;
    std::vector<Card> board;

    uint16_t rank_with(Card h0, Card h1, const std::vector<Card>& extra) const {
        Card cards[7];
        int n = 0;
        cards[n++] = h0; cards[n++] = h1;
        for (Card b : board) cards[n++] = b;
        for (Card e : extra) cards[n++] = e;
        if (n != 7) throw std::runtime_error("oracle needs a 5-card final board");
        return eval.evaluate(cards[0], cards[1], cards[2], cards[3],
                             cards[4], cards[5], cards[6]);
    }

    /// Σ over completions of (wins − loses) and ties, for one opponent hand.
    void tally(Card c0, Card c1, Card j0, Card j1,
               long& wins, long& loses, long& ties, long& runouts) const {
        CardMask used = card_to_mask(c0) | card_to_mask(c1) |
                        card_to_mask(j0) | card_to_mask(j1);
        for (Card b : board) used |= card_to_mask(b);
        const int to_come = 5 - static_cast<int>(board.size());
        auto settle = [&](const std::vector<Card>& extra) {
            const uint16_t rc = rank_with(c0, c1, extra);
            const uint16_t rj = rank_with(j0, j1, extra);
            if (rc < rj) ++wins; else if (rc > rj) ++loses; else ++ties;
            ++runouts;
        };
        if (to_come == 0) { settle({}); return; }
        for (Card r1 = 0; r1 < NUM_CARDS; ++r1) {
            if (used & card_to_mask(r1)) continue;
            if (to_come == 1) { settle({r1}); continue; }
            for (Card r2 = static_cast<Card>(r1 + 1); r2 < NUM_CARDS; ++r2) {
                if (used & card_to_mask(r2)) continue;
                settle({r1, r2});
            }
        }
    }

    double conditional_ev(Card c0, Card c1,
                          const std::array<float, NUM_COMBOS>& opp_weights,
                          double pot, double rake_rate, double rake_cap) const {
        const double half_pot = pot * 0.5;
        double rake = std::min(pot * rake_rate, rake_cap);
        if (rake < 0.0) rake = 0.0;
        const double win_p = half_pot - rake, lose_p = -half_pot, tie_p = -0.5 * rake;

        CardMask dead = card_to_mask(c0) | card_to_mask(c1);
        for (Card b : board) dead |= card_to_mask(b);
        const auto& combos = get_combo_table();
        double acc = 0.0, mass = 0.0;
        for (uint16_t j = 0; j < NUM_COMBOS; ++j) {
            const double w = opp_weights[j];
            if (w <= 0.0) continue;
            if (combos[j].conflicts_with(dead)) continue;
            long wins = 0, loses = 0, ties = 0, runouts = 0;
            tally(c0, c1, combos[j].cards[0], combos[j].cards[1], wins, loses, ties, runouts);
            const double payoff = (wins * win_p + loses * lose_p + ties * tie_p) / runouts;
            acc += w * payoff;
            mass += w;
        }
        if (mass <= 0.0) throw std::runtime_error("oracle: no compatible opponent hand");
        return acc / mass;
    }
};

static Oracle make_oracle(const SolverConfig& cfg) {
    Oracle o{get_evaluator(), {}};
    for (uint8_t i = 0; i < cfg.board_size; ++i) o.board.push_back(cfg.board[i]);
    return o;
}

static void expect_hand_evs(Solver& solver, const SolverConfig& cfg,
                            std::initializer_list<const char*> hands,
                            double tol, const char* label) {
    Oracle oracle = make_oracle(cfg);
    for (const char* h : hands) {
        const uint16_t idx = combo_index(h);
        const Combo& combo = get_combo_table()[idx];
        const double expected = oracle.conditional_ev(
            combo.cards[0], combo.cards[1], cfg.ip_range_weights,
            cfg.pot, cfg.rake_rate, cfg.rake_cap);
        const auto analysis = solver.analyze_combo(h);
        std::cout << "  " << label << " " << h << ": solver=" << analysis.ev
                  << " oracle=" << expected << "\n";
        assert_near(analysis.ev, expected, tol,
                    std::string(label) + ": conditional EV of " + h);
    }
}

// ----------------------------------------------------------------------------
// 1. Turn check-down: runout probability is 1/44 (both hands' cards excluded)
//    AND the EV is conditional on a dealable opponent hand.
// ----------------------------------------------------------------------------

static void test_turn_checkdown_conditional_ev() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "AsKd7c2h");
    make_checkdown(cfg);
    fixed_iterations(cfg, 1);

    Solver solver(cfg, BackendType::CPU);
    auto result = solver.solve();
    assert_true(!result.runout_approximated, "turn board must enumerate its rivers");
    // Nobody has a decision, so best response == EV: exploitability is 0.
    assert_near(result.exploitability_pct, 0.0, 1e-3,
                "check-down exploitability must be zero");
    expect_hand_evs(solver, cfg, {"AhKh", "QcJc", "7h7d", "5s4s", "2c2d"},
                    0.05, "turn check-down");
}

// ----------------------------------------------------------------------------
// 2. River check-down with rake: postsolve must use the raked payoffs.
// ----------------------------------------------------------------------------

static void test_river_checkdown_raked_ev() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "AsKd7c2h9s");
    make_checkdown(cfg);
    fixed_iterations(cfg, 1);
    cfg.rake_rate = 0.05f;   // 5 chips on this pot (cap does not bind)
    cfg.rake_cap  = 100.0f;

    Solver solver(cfg, BackendType::CPU);
    auto result = solver.solve();
    assert_near(result.exploitability_pct, 0.0, 1e-3,
                "raked check-down exploitability must be zero");
    expect_hand_evs(solver, cfg, {"AhKh", "QcJc", "7h7d", "5h4h", "3c3d"},
                    0.02, "river raked check-down");
}

// ----------------------------------------------------------------------------
// 3. Fold rake base: OOP bets half pot, IP always folds (locked). The uncalled
//    bet is returned, so the rake is 5% of the MATCHED 100-chip pot (= 5), not
//    of the 150 in the middle (= 7.5). EV(OOP) = 50 − 5 = 45 for every hand.
// ----------------------------------------------------------------------------

static void test_fold_rake_uses_matched_pot() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "AsKd7c2h9s");
    cfg.bet_sizing.river_sizes = {0.5f};
    cfg.bet_sizing.river_allin = false;
    cfg.raise_cap = 1;
    fixed_iterations(cfg, 1);
    cfg.rake_rate = 0.05f;
    cfg.rake_cap  = 100.0f;

    // Root actions: [Check, Bet_50]. IP facing the bet: [Fold, Call, Raise_*].
    lock_all_combos(cfg, "", cfg.oop_range_weights, /*Bet_50=*/1, 2);
    lock_all_combos(cfg, "Bet_50", cfg.ip_range_weights, /*Fold=*/0, 3);

    Solver solver(cfg, BackendType::CPU);
    auto result = solver.solve();
    assert_true(result.action_labels.size() == 2 && result.action_labels[1] == "Bet_50",
                "root must offer exactly Check and Bet_50");
    for (const char* h : {"AhKh", "7h7d", "3c3d"}) {
        const auto a = solver.analyze_combo(h);
        std::cout << "  bet/fold " << h << ": solver=" << a.ev << " expected=45\n";
        assert_near(a.ev, 45.0, 1e-3, std::string("fold rake on the matched pot for ") + h);
    }
}

// ----------------------------------------------------------------------------
// 4. Blocked opponent hands must not dilute a hand's conditional EV.
//    OOP = AhKh; IP = {AhQh (impossible), 2c2d, 3c3d}. Both live IP hands beat
//    AK-high on TsJd4c5s9h, so EV(AhKh) = −50 exactly.
// ----------------------------------------------------------------------------

static void test_conditional_ev_excludes_blocked_opponent_hands() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "TsJd4c5s9h");
    make_checkdown(cfg);
    fixed_iterations(cfg, 1);
    cfg.has_custom_ranges = true;
    set_range(cfg.oop_range_weights, {"AhKh"});
    set_range(cfg.ip_range_weights, {"AhQh", "2c2d", "3c3d"});

    Solver solver(cfg, BackendType::CPU);
    solver.solve();
    const auto a = solver.analyze_combo("AhKh");
    std::cout << "  AhKh vs {AhQh(blocked), 22, 33}: solver=" << a.ev << " expected=-50\n";
    assert_near(a.ev, -50.0, 1e-3, "blocked IP hand must not dilute the EV");
    expect_hand_evs(solver, cfg, {"AhKh"}, 1e-3, "blocked-opponent oracle");
}

// ----------------------------------------------------------------------------
// 5. Exploitability is an expectation over the LEGAL joint deal: adding IP
//    hands that every OOP hand blocks cannot change it (they are never dealt).
//    A river spot with real decisions, solved for a few iterations so the
//    exploitability is comfortably non-zero.
// ----------------------------------------------------------------------------

static SolverConfig make_river_betting_spot() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 100.0f;
    set_board(cfg, "TsJd4c5s9h");
    cfg.bet_sizing.river_sizes = {0.5f};
    cfg.bet_sizing.river_allin = true;
    cfg.raise_cap = 1;
    fixed_iterations(cfg, 20);
    cfg.has_custom_ranges = true;
    set_range(cfg.oop_range_weights, {"AhKh"});
    set_range(cfg.ip_range_weights, {"2c2d", "3c3d", "6h6d"});
    return cfg;
}

static void test_exploitability_ignores_blocked_opponent_hands() {
    auto cfg_a = make_river_betting_spot();
    Solver sa(cfg_a, BackendType::CPU);
    auto ra = sa.solve();

    auto cfg_b = make_river_betting_spot();
    cfg_b.ip_range_weights[combo_index("AhQd")] = 1.0f;   // blocked by AhKh
    cfg_b.ip_range_weights[combo_index("KhQd")] = 1.0f;   // blocked by AhKh
    Solver sb(cfg_b, BackendType::CPU);
    auto rb = sb.solve();

    std::cout << "  exploit without blocked hands=" << ra.exploitability_pct
              << "%  with=" << rb.exploitability_pct << "%\n";
    assert_true(ra.exploitability_pct > 0.5f,
                "fixture must have measurable exploitability at 20 iterations");
    assert_near(rb.exploitability_pct, ra.exploitability_pct, 1e-3,
                "blocked IP hands must not change exploitability");
    assert_near(sb.analyze_combo("AhKh").ev, sa.analyze_combo("AhKh").ev, 1e-3,
                "blocked IP hands must not change AhKh's EV");
}

// ----------------------------------------------------------------------------
// 6. Scaling one range uniformly is a no-op for every conditional quantity.
// ----------------------------------------------------------------------------

static void test_range_scale_invariance() {
    auto cfg_a = make_river_betting_spot();
    Solver sa(cfg_a, BackendType::CPU);
    auto ra = sa.solve();

    auto cfg_b = make_river_betting_spot();
    for (auto& w : cfg_b.ip_range_weights) w *= 0.5f;
    Solver sb(cfg_b, BackendType::CPU);
    auto rb = sb.solve();

    std::cout << "  exploit ×1=" << ra.exploitability_pct
              << "%  ×0.5=" << rb.exploitability_pct << "%\n";
    assert_near(rb.exploitability_pct, ra.exploitability_pct, 1e-3,
                "uniformly scaling IP's range must not change exploitability");
    assert_near(sb.analyze_combo("AhKh").ev, sa.analyze_combo("AhKh").ev, 1e-3,
                "uniformly scaling IP's range must not change AhKh's EV");
}

// ----------------------------------------------------------------------------
// 7. Two ranges that can never both be dealt are rejected, not reported as a
//    perfectly converged 0%.
// ----------------------------------------------------------------------------

static void test_empty_legal_range_rejected() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "TsJd4c5s9h");
    make_checkdown(cfg);
    fixed_iterations(cfg, 1);
    cfg.has_custom_ranges = true;
    set_range(cfg.oop_range_weights, {"AhKh"});
    set_range(cfg.ip_range_weights, {"AhQd", "KhQd"});

    bool threw = false;
    try {
        Solver solver(cfg, BackendType::CPU);
        solver.solve();
    } catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "  rejected: " << e.what() << "\n";
    }
    assert_true(threw, "a solve with no legal matchup must throw");
}

// ----------------------------------------------------------------------------
// 8/9. All-in before the river settles over every remaining runout. OOP is
//      locked to jam, IP to call, so EV(OOP hand) is the equity difference
//      against IP's range × (pot/2 + stack) — enumerated by the oracle.
// ----------------------------------------------------------------------------

static void run_allin_equity_case(const char* board_text, float stack,
                                  std::initializer_list<const char*> hands,
                                  const char* label) {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = stack;
    set_board(cfg, board_text);
    cfg.bet_sizing.flop_sizes.clear();
    cfg.bet_sizing.turn_sizes.clear();
    cfg.bet_sizing.river_sizes.clear();
    cfg.bet_sizing.flop_allin = cfg.bet_sizing.turn_allin = cfg.bet_sizing.river_allin = true;
    fixed_iterations(cfg, 1);
    // The default budget lets a rainbow flop enumerate (49 turn tables, each
    // with its own all-in equity table); that is the production shape, and
    // the locked line ends in the FLOP all-in showdown regardless.

    // Root: [Check, All-in]; IP facing the jam: [Fold, Call].
    lock_all_combos(cfg, "", cfg.oop_range_weights, /*All-in=*/1, 2);
    lock_all_combos(cfg, "All-in", cfg.ip_range_weights, /*Call=*/1, 2);

    Solver solver(cfg, BackendType::CPU);
    auto result = solver.solve();
    assert_true(result.action_labels.size() == 2 && result.action_labels[1] == "All-in",
                std::string(label) + ": root must offer exactly Check and All-in");

    // The all-in showdown pot is 100 + 2 × stack, so the oracle's "pot" is that.
    SolverConfig oracle_cfg = cfg;
    oracle_cfg.pot = 100.0f + 2.0f * stack;
    expect_hand_evs(solver, oracle_cfg, hands, 0.05, label);
}

static void test_flop_allin_runout_equity() {
    run_allin_equity_case("AsKd7c", 500.0f, {"AhKh", "QhJh", "5h4h", "2d2h"},
                          "flop all-in");
}

static void test_turn_allin_runout_equity() {
    run_allin_equity_case("AsKd7c2h", 300.0f, {"AhKh", "QhJh", "5s4s", "7d7h"},
                          "turn all-in");
}

// ----------------------------------------------------------------------------
// 10. The CFR kernels and the postsolve sweep must play the SAME game: solve a
//     turn spot with real betting to convergence and confirm the best response
//     finds little to exploit. If the kernels normalized chance nodes by 48
//     while the sweep used 44 (or vice versa), the "equilibrium" would be for
//     a different game and this would stall at several percent of the pot.
// ----------------------------------------------------------------------------

static void test_turn_cfr_and_postsolve_share_one_game() {
    for (auto kind : {SolverConfig::CpuBackendKind::REFERENCE,
                      SolverConfig::CpuBackendKind::LEVELIZED}) {
        SolverConfig cfg;
        cfg.pot = 100.0f;
        cfg.effective_stack = 100.0f;
        set_board(cfg, "AsKd7c2h");
        cfg.bet_sizing.turn_sizes  = {0.5f};
        cfg.bet_sizing.river_sizes = {0.5f};
        cfg.raise_cap = 1;
        cfg.cpu_backend_kind = kind;
        const char* label = (kind == SolverConfig::CpuBackendKind::REFERENCE)
                                ? "reference" : "levelized";

        // Measured on this fixture with STANDARD DCFR: 1.05% at 400 iterations,
        // 0.35% at 1600. A CFR/postsolve mismatch plateaus instead of falling.
        fixed_iterations(cfg, 400);
        Solver s_short(cfg, BackendType::CPU);
        auto r_short = s_short.solve();
        fixed_iterations(cfg, 1600);
        Solver s_long(cfg, BackendType::CPU);
        auto r_long = s_long.solve();
        std::cout << "  " << label << " turn spot: exploit=" << r_short.exploitability_pct
                  << "% @400 -> " << r_long.exploitability_pct << "% @1600\n";
        assert_true(!r_long.runout_approximated, "turn spot must enumerate rivers");
        assert_true(r_long.exploitability_pct < 0.6f,
                    std::string(label) + ": CFR and postsolve disagree on the game "
                    "(exploitability plateaus above 0.6% of pot at 1600 iterations)");
        assert_true(r_long.exploitability_pct < 0.5f * r_short.exploitability_pct,
                    std::string(label) + ": exploitability must keep falling with "
                    "more iterations (a plateau means CFR and the best response "
                    "play different games)");
    }
}

// ----------------------------------------------------------------------------
// 11. Suit isomorphism must respect the RANGES (and locks), not just the
//     board. On A♠K♠7♥ the ♣↔♦ swap is a board symmetry, but an IP range that
//     holds Q♣J♣ and not Q♦J♦ is not symmetric under it. Pre-2026-09 the two
//     shared a canonical slot and reach-init took the MAX weight over the
//     slot, so Q♦J♦ was silently dealt into IP's range — doubling the QJ mass
//     relative to the singleton-orbit 5♥5♠ and moving every OOP EV.
// ----------------------------------------------------------------------------

static SolverConfig make_two_tone_checkdown() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;
    set_board(cfg, "AsKs7h");
    make_checkdown(cfg);
    fixed_iterations(cfg, 1);
    cfg.has_custom_ranges = true;
    return cfg;
}

static void test_isomorphism_respects_asymmetric_ranges() {
    // Control: a suit-symmetric IP range keeps the full board group, so the
    // two-tone flop compresses to its usual 721 canonical hands.
    {
        auto cfg = make_two_tone_checkdown();
        set_range(cfg.ip_range_weights, {"QcJc", "QdJd", "5h5s"});
        Solver solver(cfg, BackendType::CPU);
        auto r = solver.solve();
        std::cout << "  symmetric IP range: canonical_combos=" << r.resources.canonical_combos << "\n";
        assert_true(r.resources.canonical_combos == 721,
                    "a suit-symmetric range must keep the two-tone board's 721-hand compression");
        expect_hand_evs(solver, cfg, {"AhAd", "7c7d", "Tc9c"}, 0.05, "symmetric-range check-down");
    }
    // Asymmetric: Q♣J♣ without Q♦J♦ breaks the ♣↔♦ symmetry. The canonical
    // space must fall back to the identity (1176 board-live hands) and every
    // OOP EV must match the brute-force oracle over IP's ACTUAL range.
    {
        auto cfg = make_two_tone_checkdown();
        set_range(cfg.ip_range_weights, {"QcJc", "5h5s"});
        Solver solver(cfg, BackendType::CPU);
        auto r = solver.solve();
        std::cout << "  asymmetric IP range: canonical_combos=" << r.resources.canonical_combos << "\n";
        assert_true(r.resources.canonical_combos == 1176,
                    "an asymmetric range must not be bucketed under a symmetry it breaks");
        expect_hand_evs(solver, cfg, {"AhAd", "7c7d", "Tc9c"}, 0.05, "asymmetric-range check-down");
    }
    // A node lock on ONE suit combination is an asymmetry too.
    {
        auto cfg = make_two_tone_checkdown();
        set_range(cfg.ip_range_weights, {"QcJc", "QdJd", "5h5s"});
        NodeLockEntry lock;
        lock.history   = "Check";          // IP's first decision
        lock.combo_idx = combo_index("QcJc");
        lock.combo_str = "QcJc";
        lock.strategy  = {1.0f};
        cfg.node_locks.push_back(lock);
        Solver solver(cfg, BackendType::CPU);
        auto r = solver.solve();
        std::cout << "  lock on QcJc only: canonical_combos=" << r.resources.canonical_combos << "\n";
        assert_true(r.resources.canonical_combos == 1176,
                    "a lock on one suit combination must break that suit symmetry");
    }
}

// ----------------------------------------------------------------------------

int main() {
    auto& eval = get_evaluator();
    if (!eval.is_initialized()) eval.initialize();

    std::cout << "=== DeepSolver game-model oracle suite ===\n\n";

    RUN_TEST(test_turn_checkdown_conditional_ev);
    RUN_TEST(test_river_checkdown_raked_ev);
    RUN_TEST(test_fold_rake_uses_matched_pot);
    RUN_TEST(test_conditional_ev_excludes_blocked_opponent_hands);
    RUN_TEST(test_exploitability_ignores_blocked_opponent_hands);
    RUN_TEST(test_range_scale_invariance);
    RUN_TEST(test_empty_legal_range_rejected);
    RUN_TEST(test_flop_allin_runout_equity);
    RUN_TEST(test_turn_allin_runout_equity);
    RUN_TEST(test_turn_cfr_and_postsolve_share_one_game);
    RUN_TEST(test_isomorphism_respects_asymmetric_ranges);

    std::cout << "=== " << g_tests_passed << " / " << g_tests_run
              << " tests passed ===\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
