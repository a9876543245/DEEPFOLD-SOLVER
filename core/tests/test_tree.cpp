/**
 * @file test_tree.cpp
 * @brief Unit tests for the game tree builder.
 */

#include "game_tree_builder.h"
#include "isomorphism.h"
#include "card.h"
#include <iostream>
#include <cassert>

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

int main() {
    std::cout << "=== DeepSolver Game Tree Builder Tests ===\n";

    test_basic_tree();
    test_tree_pruning();
    test_short_stack_allin();
    test_node_integrity();
    test_runout_iso_flop_textures();
    test_runout_iso_turn_textures();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
