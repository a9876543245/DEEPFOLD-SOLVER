/**
 * @file test_evaluator.cpp
 * @brief Unit tests for the hand evaluator.
 */

#include "hand_evaluator.h"
#include "card.h"
#include <iostream>
#include <cassert>

using namespace deepsolver;

void test_card_parsing() {
    std::cout << "Testing card parsing... ";

    // Test specific cards
    assert(parse_card("2c") == make_card(RANK_2, CLUBS));
    assert(parse_card("As") == make_card(RANK_A, SPADES));
    assert(parse_card("Kh") == make_card(RANK_K, HEARTS));
    assert(parse_card("Td") == make_card(RANK_T, DIAMONDS));

    // Test roundtrip
    for (Card c = 0; c < NUM_CARDS; ++c) {
        std::string s = card_to_string(c);
        Card parsed = parse_card(s);
        assert(parsed == c);
    }

    // Test board parsing
    auto board = parse_board("AsKd7c");
    assert(board.size() == 3);
    assert(board[0] == parse_card("As"));
    assert(board[1] == parse_card("Kd"));
    assert(board[2] == parse_card("7c"));

    std::cout << "PASSED\n";
}

void test_combo_table() {
    std::cout << "Testing combo table... ";

    const auto& table = get_combo_table();

    // Check total count
    int count = 0;
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        assert(table[i].cards[0] < table[i].cards[1]);
        assert(table[i].index() == i);
        count++;
    }
    assert(count == NUM_COMBOS);

    // Check no duplicates
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        for (uint16_t j = i + 1; j < NUM_COMBOS; ++j) {
            assert(table[i].cards[0] != table[j].cards[0] ||
                   table[i].cards[1] != table[j].cards[1]);
        }
    }

    std::cout << "PASSED\n";
}

void test_hand_rankings() {
    std::cout << "Testing hand rankings... ";

    HandEvaluator eval;
    eval.initialize();

    // Test that Royal Flush < Full House < High Card
    // Royal Flush: As Ks Qs Js Ts (board) + two random non-spade cards
    Card rf_board[] = {
        parse_card("As"), parse_card("Ks"), parse_card("Qs"),
        parse_card("Js"), parse_card("Ts")
    };
    uint16_t rf_rank = eval.evaluate5(rf_board[0], rf_board[1], rf_board[2],
                                       rf_board[3], rf_board[4]);
    assert(rf_rank >= 1 && rf_rank <= 10); // Straight flush range

    // Full House: AAA KK
    Card fh[] = {
        parse_card("Ac"), parse_card("Ad"), parse_card("Ah"),
        parse_card("Kc"), parse_card("Kd")
    };
    uint16_t fh_rank = eval.evaluate5(fh[0], fh[1], fh[2], fh[3], fh[4]);
    assert(fh_rank > rf_rank); // Full house is weaker

    // High Card: Ac Kd Qs Jh 9c
    Card hc[] = {
        parse_card("Ac"), parse_card("Kd"), parse_card("Qs"),
        parse_card("Jh"), parse_card("9c")
    };
    uint16_t hc_rank = eval.evaluate5(hc[0], hc[1], hc[2], hc[3], hc[4]);
    assert(hc_rank > fh_rank); // High card is weaker

    std::cout << "PASSED (RF=" << rf_rank << " FH=" << fh_rank
              << " HC=" << hc_rank << ")\n";
}

void test_7card_evaluation() {
    std::cout << "Testing 7-card evaluation... ";

    HandEvaluator eval;
    eval.initialize();

    // AhKh on board AsKd7c 2d 3s → Two pair (AK) should be decent
    Card c0 = parse_card("Ah");
    Card c1 = parse_card("Kh");
    Card b0 = parse_card("As");
    Card b1 = parse_card("Kd");
    Card b2 = parse_card("7c");
    Card b3 = parse_card("2d");
    Card b4 = parse_card("3s");

    uint16_t rank = eval.evaluate(c0, c1, b0, b1, b2, b3, b4);
    // Should be Two Pair range
    assert(rank > 0 && rank <= NUM_HAND_RANKS);

    // Compare: 77 on same board → Set of 7s (should be stronger, lower rank)
    Card s0 = parse_card("7d");
    Card s1 = parse_card("7h");
    uint16_t set_rank = eval.evaluate(s0, s1, b0, b1, b2, b3, b4);

    // Full house 77 on A K 7 → 777 AK = full house
    assert(set_rank < rank); // Set/full house should beat two pair

    std::cout << "PASSED (AK=" << rank << " 77=" << set_rank << ")\n";
}

void test_hand_categories() {
    std::cout << "Testing hand categories... ";

    assert(std::string(hand_category_name(1)) == "Straight Flush");
    assert(std::string(hand_category_name(100)) == "Four of a Kind");
    assert(std::string(hand_category_name(200)) == "Full House");
    assert(std::string(hand_category_name(500)) == "Flush");
    assert(std::string(hand_category_name(1605)) == "Straight");
    assert(std::string(hand_category_name(2000)) == "Three of a Kind");
    assert(std::string(hand_category_name(3000)) == "Two Pair");
    assert(std::string(hand_category_name(5000)) == "One Pair");
    assert(std::string(hand_category_name(7000)) == "High Card");

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== DeepSolver Hand Evaluator Tests ===\n";

    test_card_parsing();
    test_combo_table();
    test_hand_rankings();
    test_7card_evaluation();
    test_hand_categories();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
