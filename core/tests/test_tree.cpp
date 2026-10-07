/**
 * @file test_tree.cpp
 * @brief Unit tests for the game tree builder.
 */

#include "game_tree_builder.h"
#include "isomorphism.h"
#include "card.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace deepsolver;

void test_basic_tree() {
    std::cout << "Testing basic tree construction... ";

    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 200.0f;
    config.board_size = 3;

    auto board = parse_board("AsKd7c");
    for (size_t i = 0; i < board.size(); ++i) {
        config.board[i] = board[i];
    }

    GameTreeBuilder builder(config);
    auto tree = builder.build();

    // Basic sanity checks
    assert(tree.total_nodes > 0);
    assert(tree.total_edges > 0);
    assert(tree.node_types.size() == tree.total_nodes);
    assert(tree.pots.size() == tree.total_nodes);
    assert(tree.children.size() == tree.total_edges);

    // Root node should be OOP player
    assert(tree.node_types[0] == static_cast<uint8_t>(NodeType::PLAYER_OOP));
    assert(tree.pots[0] == 100.0f);
    assert(tree.stacks[0] == 200.0f);

    // Root should have children (at least Check)
    assert(tree.num_children[0] > 0);

    std::cout << "PASSED (nodes=" << tree.total_nodes
              << " edges=" << tree.total_edges << ")\n";
}

void test_tree_pruning() {
    std::cout << "Testing tree pruning rules... ";

    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 500.0f;
    config.board_size = 3;
    config.raise_cap = 3;
    config.allin_threshold = 0.12f;
    config.allow_donk_bet = false;

    auto board = parse_board("AsKd7c");
    for (size_t i = 0; i < board.size(); ++i) {
        config.board[i] = board[i];
    }

    GameTreeBuilder builder(config);
    auto tree = builder.build();

    // Count terminal nodes
    uint32_t terminal_count = 0;
    uint32_t player_count = 0;
    for (uint32_t i = 0; i < tree.total_nodes; ++i) {
        if (tree.node_types[i] == static_cast<uint8_t>(NodeType::TERMINAL)) {
            terminal_count++;
        }
        if (tree.node_types[i] <= 1) { // PLAYER_OOP or PLAYER_IP
            player_count++;
        }
    }

    assert(terminal_count > 0);
    assert(player_count > 0);

    // Root (OOP without initiative): should NOT have bet options (no donk)
    // Only Check should be available
    uint8_t root_actions = tree.num_children[0];
    bool has_check = false;
    uint32_t root_off = tree.children_offset[0];
    for (uint8_t i = 0; i < root_actions; ++i) {
        auto at = static_cast<ActionType>(tree.child_action_types[root_off + i]);
        if (at == ActionType::CHECK) has_check = true;
    }
    assert(has_check);

    std::cout << "PASSED (terminals=" << terminal_count
              << " players=" << player_count << ")\n";
}

void test_short_stack_allin() {
    std::cout << "Testing short stack all-in forcing... ";

    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 15.0f;  // Very short stack (SPR=0.15)
    config.board_size = 3;
    config.allin_threshold = 0.12f;

    auto board = parse_board("AsKd7c");
    for (size_t i = 0; i < board.size(); ++i) {
        config.board[i] = board[i];
    }

    GameTreeBuilder builder(config);
    auto tree = builder.build();

    // With such a short stack, most bets should collapse to all-in
    bool found_allin = false;
    for (uint32_t i = 0; i < tree.total_edges; ++i) {
        if (tree.child_action_types[i] == static_cast<uint8_t>(ActionType::ALLIN)) {
            found_allin = true;
            break;
        }
    }
    assert(found_allin);

    // Tree should be relatively small with short stack
    assert(tree.total_nodes < 100);

    std::cout << "PASSED (nodes=" << tree.total_nodes << ")\n";
}

void test_node_integrity() {
    std::cout << "Testing node integrity... ";

    SolverConfig config;
    config.pot = 80.0f;
    config.effective_stack = 300.0f;
    config.board_size = 3;

    auto board = parse_board("Jh9s4c");
    for (size_t i = 0; i < board.size(); ++i) {
        config.board[i] = board[i];
    }

    GameTreeBuilder builder(config);
    auto tree = builder.build();

    // Verify all parent indices are valid
    for (uint32_t i = 1; i < tree.total_nodes; ++i) {
        assert(tree.parent_indices[i] < tree.total_nodes);
    }

    // Verify child indices are valid
    for (uint32_t i = 0; i < tree.total_edges; ++i) {
        assert(tree.children[i] < tree.total_nodes);
    }

    // Every non-terminal node should have at least 1 child
    for (uint32_t i = 0; i < tree.total_nodes; ++i) {
        auto nt = static_cast<NodeType>(tree.node_types[i]);
        if (nt != NodeType::TERMINAL) {
            assert(tree.num_children[i] > 0);
        }
    }

    std::cout << "PASSED\n";
}

// ============================================================================
// Phase 2: Suit isomorphism for runout enumeration
// ============================================================================

static uint8_t weight_sum(const CanonicalRunouts& cr) {
    uint8_t s = 0;
    for (const auto& r : cr.reps) s = static_cast<uint8_t>(s + r.weight);
    return s;
}

// Inline test runner — used directly because static-inline helpers were
// observed to be optimized away by MSVC /O2 in this translation unit.
#define RUN_ISO_CASE(BOARD_STR, EXPECTED, TAG)                                  \
    do {                                                                        \
        auto board = parse_board(BOARD_STR);                                    \
        auto cr = enumerate_canonical_runouts(                                  \
            board.data(), static_cast<uint8_t>(board.size()));                  \
        uint8_t expected_total =                                                \
            static_cast<uint8_t>(NUM_CARDS - board.size());                     \
        if (weight_sum(cr) != expected_total) {                                 \
            std::cerr << "FAIL " TAG ": weight " << int(weight_sum(cr))         \
                      << " != " << int(expected_total) << "\n";                 \
            std::exit(1);                                                       \
        }                                                                       \
        if (cr.reps.size() != static_cast<size_t>(EXPECTED)) {                  \
            std::cerr << "FAIL " TAG ": got " << cr.reps.size()                 \
                      << " classes, expected " << (EXPECTED) << "\n";           \
            std::exit(1);                                                       \
        }                                                                       \
        std::cout << "  " TAG " (" BOARD_STR "): "                              \
                  << cr.reps.size() << " classes (sum_w="                       \
                  << int(weight_sum(cr)) << ")\n";                              \
    } while (0)

void test_runout_iso_flop_textures() {
    std::cout << "Testing runout iso on flop textures...\n";

    // Rainbow: G = {id}. 49 undealt -> 49 classes.
    RUN_ISO_CASE("KsAd7c", 49, "rainbow");
    // Two-tone (suit-suit-other): KsAd3s, ^=spade^2, dia^1, hearts/clubs unused.
    //   spades alive 11, dia alive 12, {h,c} pairs 13 -> 36 classes.
    RUN_ISO_CASE("KsAd3s", 36, "two-tone");
    // Monotone: KsAs3s, S_3 over hearts/diamonds/clubs.
    //   spades alive 10, {h,d,c} orbit-3 x 13 ranks -> 23 classes.
    RUN_ISO_CASE("KsAs3s", 23, "monotone");
    // Paired with two suits on the pair: KsKd7c.
    //   {spade<->dia} swap fixes board; hearts and clubs stay.
    //   See test header math: 37 classes.
    RUN_ISO_CASE("KsKd7c", 37, "paired-two-suit");

    std::cout << "  PASSED\n";
}

void test_runout_iso_turn_textures() {
    std::cout << "Testing runout iso on turn textures...\n";

    // Rainbow turn: G={id}. 48 -> 48.
    RUN_ISO_CASE("KsAd7c2h", 48, "rainbow-turn");
    // 3-of-suit turn KsAs3s2h: spade^3 hearts^1, dia/clubs unused.
    //   spades alive 10, hearts alive 12, {d,c} pairs 13 -> 35 classes.
    RUN_ISO_CASE("KsAs3s2h", 35, "3-of-suit-turn");

    std::cout << "  PASSED\n";
}

// Independent ledger: reconstruct both players' contributions from edges,
// never from the builder's stack/bet_into calculations. Keep these checks live
// in Release too -- assert() alone is disabled by NDEBUG in production builds.
static void require_rule(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_chip_ledger(const FlatGameTree& tree, uint32_t node,
                             float starting_pot, float starting_stack,
                             float oop_paid, float ip_paid) {
    const auto close = [](float a, float b) { return std::abs(a - b) < 0.01f; };
    require_rule(close(tree.pots[node], starting_pot + oop_paid + ip_paid),
                 "pot must equal starting pot plus both contributions");
    require_rule(oop_paid <= starting_stack + 0.01f &&
                 ip_paid <= starting_stack + 0.01f,
                 "a player cannot invest more than their starting stack");
    const auto type = static_cast<NodeType>(tree.node_types[node]);
    if (type == NodeType::TERMINAL) {
        require_rule(tree.stacks[node] >= -0.01f, "terminal stack cannot be negative");
        if (static_cast<TerminalType>(tree.terminal_types[node]) == TerminalType::SHOWDOWN)
            require_rule(close(oop_paid, ip_paid), "showdown contributions must match");
        return;
    }
    if (type == NodeType::CHANCE) {
        require_rule(close(oop_paid, ip_paid), "a street can end only with matched contributions");
        const auto off = tree.children_offset[node];
        for (uint8_t a = 0; a < tree.num_children[node]; ++a)
            check_chip_ledger(tree, tree.children[off + a], starting_pot, starting_stack,
                              oop_paid, ip_paid);
        return;
    }
    const int actor = tree.active_player[node];
    const float own = actor == 0 ? oop_paid : ip_paid;
    const float opp = actor == 0 ? ip_paid : oop_paid;
    const float to_call = opp - own;
    require_rule(close(tree.stacks[node], starting_stack - own),
                 "decision stack must belong to the player about to act");
    require_rule(close(tree.bet_into[node], to_call),
                 "amount to call must be the difference in contributions");
    const auto off = tree.children_offset[node];
    for (uint8_t a = 0; a < tree.num_children[node]; ++a) {
        const auto action = static_cast<ActionType>(tree.child_action_types[off + a]);
        const float amount = tree.child_action_amts[off + a];
        float paid = 0.0f;
        if (action == ActionType::CALL) {
            require_rule(close(amount, to_call), "call must pay only the outstanding amount");
            paid = amount;
        } else if (action == ActionType::BET || action == ActionType::RAISE ||
                   action == ActionType::ALLIN) {
            require_rule(amount > to_call, "aggression must increase the wager");
            require_rule(to_call <= 0.0f || amount + 0.01f >= 2.0f * to_call ||
                         close(amount, starting_stack - own),
                         "a non-all-in raise must at least double the outstanding increment");
            paid = amount;
        }
        check_chip_ledger(tree, tree.children[off + a], starting_pot, starting_stack,
                          oop_paid + (actor == 0 ? paid : 0.0f),
                          ip_paid + (actor == 1 ? paid : 0.0f));
    }
}

static void test_betting_chip_conservation() {
    for (const char* board_text : {"AsKd7c", "AsKd7c2h", "AsKd7c2h3s"}) {
        for (float stack : {15.0f, 100.0f, 500.0f}) {
            SolverConfig config;
            config.pot = 100.0f;
            config.effective_stack = stack;
            const auto board = parse_board(board_text);
            config.board_size = static_cast<uint8_t>(board.size());
            std::copy(board.begin(), board.end(), config.board.begin());
            config.bet_sizing.river_sizes = {0.01f, 0.5f, 1.5f};
            config.raise_cap = 3;
            GameTreeBuilder builder(config);
            // Test betting state across streets without making this rules test
            // dependent on runout compression or memory estimates.
            builder.set_force_runout_collapse(true);
            const auto tree = builder.build();
            check_chip_ledger(tree, 0, config.pot, stack, 0.0f, 0.0f);
        }
    }
    std::cout << "Betting chip conservation PASSED\n";
}

static uint32_t action_child(const FlatGameTree& tree, uint32_t node,
                             ActionType action, float amount) {
    const auto off = tree.children_offset[node];
    for (uint8_t a = 0; a < tree.num_children[node]; ++a) {
        if (tree.child_action_types[off + a] == static_cast<uint8_t>(action) &&
            std::abs(tree.child_action_amts[off + a] - amount) < 0.01f)
            return tree.children[off + a];
    }
    throw std::runtime_error("expected betting action is missing");
}

static void test_raise_and_allin_examples() {
    SolverConfig config;
    config.pot = 100.0f;
    config.effective_stack = 500.0f;
    config.board_size = 5;
    const auto board = parse_board("AsKd7c2h3s");
    std::copy(board.begin(), board.end(), config.board.begin());
    config.bet_sizing.river_sizes = {0.5f};
    config.allin_threshold = 0.0f;
    GameTreeBuilder builder(config);
    const auto tree = builder.build();
    const auto bet = action_child(tree, 0, ActionType::BET, 50.0f);
    const auto raise = action_child(tree, bet, ActionType::RAISE, 150.0f);
    const auto call = action_child(tree, raise, ActionType::CALL, 100.0f);
    require_rule(tree.pots[call] == 400.0f && tree.stacks[call] == 350.0f,
                 "bet 50 / raise 150 / call 100 must leave pot 400 and stacks 350");
    const auto shove = action_child(tree, 0, ActionType::ALLIN, 500.0f);
    const auto called = action_child(tree, shove, ActionType::CALL, 500.0f);
    require_rule(tree.pots[called] == 1100.0f && tree.stacks[called] == 0.0f,
                 "500-chip shove and call must leave pot 1100 and no negative stack");

    // A 75-chip all-in over a 50-chip bet is allowed despite being smaller
    // than a full raise. The bettor has 25 left and can only fold or call.
    config.effective_stack = 75.0f;
    GameTreeBuilder short_builder(config);
    const auto short_tree = short_builder.build();
    const auto short_bet = action_child(short_tree, 0, ActionType::BET, 50.0f);
    const auto short_raise = action_child(short_tree, short_bet, ActionType::ALLIN, 75.0f);
    require_rule(short_tree.num_children[short_raise] == 2,
                 "an all-in raise must leave only fold and call");
    const auto short_call = action_child(short_tree, short_raise, ActionType::CALL, 25.0f);
    require_rule(short_tree.pots[short_call] == 250.0f && short_tree.stacks[short_call] == 0.0f,
                 "short all-in call must pay only the remaining 25 chips");
    std::cout << "Raise and all-in ledger examples PASSED\n";
}

// ----------------------------------------------------------------------------
// 2026-10-06 audit + custom bet sizing
// ----------------------------------------------------------------------------

static SolverConfig river_config(float pot, float stack) {
    SolverConfig config;
    config.pot = pot;
    config.effective_stack = stack;
    const auto board = parse_board("AsKd7c2h3s");
    config.board_size = 5;
    std::copy(board.begin(), board.end(), config.board.begin());
    return config;
}

static std::vector<std::pair<uint8_t, float>> actions_at(const FlatGameTree& tree, uint32_t node) {
    std::vector<std::pair<uint8_t, float>> out;
    const auto off = tree.children_offset[node];
    for (uint8_t a = 0; a < tree.num_children[node]; ++a) {
        out.push_back({tree.child_action_types[off + a], tree.child_action_amts[off + a]});
    }
    return out;
}

static int count_type(const FlatGameTree& tree, uint32_t node, ActionType t) {
    int n = 0;
    for (const auto& [type, amt] : actions_at(tree, node)) n += (type == static_cast<uint8_t>(t));
    return n;
}

static void test_menu_order_and_duplicates() {
    // Menu order must not matter: a `break` after the first forced all-in
    // used to drop every later (smaller) size — [Check, All-in] for 1.5,0.33.
    auto a = river_config(100.0f, 150.0f);
    a.bet_sizing.river_sizes = {1.5f, 0.33f};
    auto b = river_config(100.0f, 150.0f);
    b.bet_sizing.river_sizes = {0.33f, 1.5f};
    GameTreeBuilder ba(a), bb(b);
    const auto ta = ba.build(), tb = bb.build();
    require_rule(actions_at(ta, 0) == actions_at(tb, 0), "menu order must not change the root actions");
    require_rule(count_type(ta, 0, ActionType::BET) == 1 && count_type(ta, 0, ActionType::ALLIN) == 1,
                 "1.5,0.33 with stack 150 must keep the 33% bet and one all-in");

    // Donk + forced all-in used to append a SECOND all-in after the donk bet.
    auto d = river_config(100.0f, 100.0f);
    d.bet_sizing.river_sizes = {0.75f};
    d.oop_has_initiative = false;
    d.allow_donk_bet = true;
    GameTreeBuilder bd(d);
    const auto td = bd.build();
    require_rule(count_type(td, 0, ActionType::ALLIN) == 1, "a node carries at most one all-in");

    // Six sizes + check + all-in: the old MAX_ACTIONS = 6 cut the all-in off.
    auto six = river_config(100.0f, 2000.0f);
    six.bet_sizing.river_sizes = {0.25f, 0.33f, 0.5f, 0.75f, 1.0f, 1.5f};
    GameTreeBuilder bs(six);
    const auto ts = bs.build();
    require_rule(count_type(ts, 0, ActionType::BET) == 6 && count_type(ts, 0, ActionType::ALLIN) == 1,
                 "six bet sizes must all be kept, plus the all-in");

    // A non-positive size is ignored instead of recursing forever.
    auto z = river_config(100.0f, 200.0f);
    z.bet_sizing.river_sizes = {0.0f, 0.5f};
    GameTreeBuilder bz(z);
    const auto tz = bz.build();
    require_rule(count_type(tz, 0, ActionType::BET) == 1, "a 0% size must not become a bet");
    std::cout << "Menu order / duplicate / cap rules PASSED\n";
}

static void test_custom_menus() {
    // Per-player river menus: OOP bets 50%, IP raises "2.5x" (TO 2.5 times
    // OOP's 50-chip bet = 125) or 100% (50 call + 100% of the 200-chip pot
    // after calling = 250).
    auto c = river_config(100.0f, 1000.0f);
    c.allin_threshold = 0.0f;
    c.bet_sizing.custom = true;
    c.bet_sizing.player[0][2].bet = {{BetSize::Kind::PotFraction, 0.5f}};
    c.bet_sizing.player[1][2].raise = {{BetSize::Kind::Multiplier, 2.5f},
                                       {BetSize::Kind::PotFraction, 1.0f}};
    c.bet_sizing.player[0][2].allin = false;
    c.bet_sizing.player[1][2].allin = false;
    GameTreeBuilder builder(c);
    const auto tree = builder.build();
    const auto bet = action_child(tree, 0, ActionType::BET, 50.0f);
    action_child(tree, bet, ActionType::RAISE, 125.0f);
    const auto big = action_child(tree, bet, ActionType::RAISE, 250.0f);
    // OOP has no raise menu: fold or call only.
    require_rule(tree.num_children[big] == 2, "an empty raise menu leaves fold and call");
    require_rule(count_type(tree, 0, ActionType::ALLIN) == 0, "all-in off means no all-in");

    // Re-raise "x" sizing counts the raiser's own street wager: OOP bet 50,
    // IP raised to 125, so a 3x re-raise is TO 375 = 325 more.
    auto r = c;
    r.bet_sizing.player[0][2].raise = {{BetSize::Kind::Multiplier, 3.0f}};
    GameTreeBuilder rb(r);
    const auto rt = rb.build();
    const auto rbet = action_child(rt, 0, ActionType::BET, 50.0f);
    const auto rraise = action_child(rt, rbet, ActionType::RAISE, 125.0f);
    action_child(rt, rraise, ActionType::RAISE, 325.0f);
    std::cout << "Custom per-player menus PASSED\n";
}

static void test_donk_menu() {
    // Turn root, OOP without initiative: OOP's turn lead takes the donk menu
    // (25% here); after IP bets the turn and OOP calls, OOP's river lead is a
    // donk again; after a checked-through turn it is a normal bet.
    SolverConfig c;
    c.pot = 100.0f;
    c.effective_stack = 1000.0f;
    const auto board = parse_board("AsKd7c2h");
    c.board_size = 4;
    std::copy(board.begin(), board.end(), c.board.begin());
    c.oop_has_initiative = false;
    c.allin_threshold = 0.0f;
    c.bet_sizing.custom = true;
    for (int st = 1; st < 3; ++st) {
        c.bet_sizing.player[0][st].bet  = {{BetSize::Kind::PotFraction, 0.75f}};
        c.bet_sizing.player[0][st].donk = {{BetSize::Kind::PotFraction, 0.25f}};
        c.bet_sizing.player[1][st].bet  = {{BetSize::Kind::PotFraction, 0.5f}};
        c.bet_sizing.player[0][st].allin = c.bet_sizing.player[1][st].allin = false;
    }
    GameTreeBuilder builder(c);
    builder.set_force_runout_collapse(true);
    const auto tree = builder.build();
    action_child(tree, 0, ActionType::BET, 25.0f);           // turn donk
    const auto chk = action_child(tree, 0, ActionType::CHECK, 0.0f);
    const auto ip_bet = action_child(tree, chk, ActionType::BET, 50.0f);
    const auto call = action_child(tree, ip_bet, ActionType::CALL, 50.0f);
    const uint32_t river_after_bet = tree.children[tree.children_offset[call]];
    action_child(tree, river_after_bet, ActionType::BET, 50.0f);   // 25% of 200: donk
    const auto ip_chk = action_child(tree, chk, ActionType::CHECK, 0.0f);
    const uint32_t river_after_check = tree.children[tree.children_offset[ip_chk]];
    action_child(tree, river_after_check, ActionType::BET, 75.0f); // 75% of 100: bet
    std::cout << "Donk menu PASSED\n";
}

static void test_legacy_lists_equal_custom_menus() {
    // The legacy lists and the menus they resolve to build the same tree.
    SolverConfig legacy;
    legacy.pot = 55.0f;
    legacy.effective_stack = 975.0f;
    const auto board = parse_board("Td9d6h2c");
    legacy.board_size = 4;
    std::copy(board.begin(), board.end(), legacy.board.begin());
    legacy.bet_sizing.turn_sizes = {0.33f, 0.66f, 1.0f};
    legacy.bet_sizing.river_sizes = {0.5f, 1.25f};
    legacy.oop_has_initiative = false;
    legacy.allow_donk_bet = true;
    SolverConfig custom = legacy;
    const SizingMenus menus = resolve_sizing_menus(legacy);
    custom.bet_sizing.custom = true;
    for (int p = 0; p < 2; ++p)
        for (int st = 0; st < 3; ++st) custom.bet_sizing.player[p][st] = menus[p][st];
    GameTreeBuilder bl(legacy), bc(custom);
    bl.set_force_runout_collapse(true);
    bc.set_force_runout_collapse(true);
    const auto tl = bl.build(), tc = bc.build();
    require_rule(tl.total_nodes == tc.total_nodes &&
                 tl.child_action_amts == tc.child_action_amts &&
                 tl.child_action_types == tc.child_action_types,
                 "legacy lists and their resolved menus must build one tree");
    std::cout << "Legacy lists == resolved menus PASSED (" << tl.total_nodes << " nodes)\n";
}

int main() {
    try {
        std::cout << "=== DeepSolver Game Tree Builder Tests ===\n";

        test_basic_tree();
        test_tree_pruning();
        test_short_stack_allin();
        test_node_integrity();
        test_runout_iso_flop_textures();
        test_runout_iso_turn_textures();
        test_betting_chip_conservation();
        test_raise_and_allin_examples();
        test_menu_order_and_duplicates();
        test_custom_menus();
        test_donk_menu();
        test_legacy_lists_equal_custom_menus();

        std::cout << "\nAll tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Game tree rule failure: " << e.what() << '\n';
        return 1;
    }
}
