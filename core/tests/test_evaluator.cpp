/**
 * @file test_evaluator.cpp
 * @brief Unit tests for the hand evaluator.
 */

#include "hand_evaluator.h"
#include "card.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <string>
#include <utility>

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

// Build a concrete 5-card hand from a rank-multiset (list of (rank, count)).
// Occurrences of a rank get suits 0,1,2,3; this never makes a 5-flush (a pair
// already repeats a rank so popcount<5 → hash path, which is what we test).
static void build_hand(const std::vector<std::pair<int,int>>& multiset, Card out[5]) {
    int n = 0;
    for (auto& [r, cnt] : multiset)
        for (int s = 0; s < cnt; ++s)
            out[n++] = make_card(static_cast<Rank>(r), static_cast<Suit>(s));
}

// Regression for the hash-collision evaluator bug: open addressing without a
// stored key returned the FIRST non-empty slot, so the 197 paired-hand prime
// products that collide mod 65536 returned each other's rank. We assert the
// evaluator is INJECTIVE over every paired rank-multiset (collisions alias two
// multisets to one rank) and that each rank lands in the correct category band.
void test_hash_collisions() {
    std::cout << "Testing evaluator injectivity (hash-collision regression)... ";
    HandEvaluator eval;
    eval.initialize();

    std::map<uint16_t, std::string> seen;  // rank -> first hand description
    auto check = [&](const std::vector<std::pair<int,int>>& ms,
                     uint16_t lo, uint16_t hi, const std::string& desc) {
        Card h[5]; build_hand(ms, h);
        uint16_t r = eval.evaluate5(h[0], h[1], h[2], h[3], h[4]);
        assert(r >= lo && r <= hi);          // correct category band
        auto it = seen.find(r);
        if (it != seen.end()) {
            std::cout << "\n  COLLISION: '" << desc << "' and '" << it->second
                      << "' both rank " << r << "\n";
            assert(false);
        }
        seen[r] = desc;
    };

    // One pair: 13 pair ranks x C(12,3) kicker sets
    for (int p = 12; p >= 0; --p)
        for (int k1 = 12; k1 >= 0; --k1) { if (k1==p) continue;
        for (int k2 = k1-1; k2 >= 0; --k2) { if (k2==p) continue;
        for (int k3 = k2-1; k3 >= 0; --k3) { if (k3==p) continue;
            check({{p,2},{k1,1},{k2,1},{k3,1}}, ONE_PAIR_MAX-2859, ONE_PAIR_MAX,
                  "pair");
        }}}
    // Two pair: C(13,2) pairs x 11 kickers
    for (int p1 = 12; p1 >= 1; --p1)
        for (int p2 = p1-1; p2 >= 0; --p2)
        for (int k = 12; k >= 0; --k) { if (k==p1||k==p2) continue;
            check({{p1,2},{p2,2},{k,1}}, TWO_PAIR_MAX-857, TWO_PAIR_MAX, "twopair");
        }
    // Trips: 13 x C(12,2)
    for (int t = 12; t >= 0; --t)
        for (int k1 = 12; k1 >= 0; --k1) { if (k1==t) continue;
        for (int k2 = k1-1; k2 >= 0; --k2) { if (k2==t) continue;
            check({{t,3},{k1,1},{k2,1}}, THREE_OF_A_KIND_MAX-857,
                  THREE_OF_A_KIND_MAX, "trips");
        }}
    // Full house: 13 x 12
    for (int t = 12; t >= 0; --t)
        for (int p = 12; p >= 0; --p) { if (p==t) continue;
            check({{t,3},{p,2}}, FULL_HOUSE_MAX-155, FULL_HOUSE_MAX, "boat");
        }
    // Quads: 13 x 12
    for (int q = 12; q >= 0; --q)
        for (int k = 12; k >= 0; --k) { if (k==q) continue;
            check({{q,4},{k,1}}, FOUR_OF_A_KIND_MAX-155, FOUR_OF_A_KIND_MAX, "quads");
        }

    std::cout << "PASSED (" << seen.size() << " distinct paired hands, all injective)\n";
}

// Regression for the exact symptom the bug produced in solves: on Ks9d4h2c7s,
// a higher pair must out-rank a lower pair, and a set must out-rank an overpair.
void test_pocket_pair_ordering() {
    std::cout << "Testing pocket-pair ordering on Ks9d4h2c7s... ";
    HandEvaluator eval;
    eval.initialize();
    Card b0=parse_card("Ks"), b1=parse_card("9d"), b2=parse_card("4h"),
         b3=parse_card("2c"), b4=parse_card("7s");
    auto pp = [&](const char* a, const char* b) {
        return eval.evaluate(parse_card(a), parse_card(b), b0, b1, b2, b3, b4);
    };
    uint16_t r88 = pp("8c","8d"), r33 = pp("3c","3d");
    uint16_t r66 = pp("6c","6d"), r55 = pp("5c","5d");
    uint16_t rAA = pp("Ac","Ad"), rKK = pp("Kc","Kd"); // KK = set of kings
    assert(r88 < r33);  // pair of 8s beats pair of 3s (lower rank = better)
    assert(r66 < r33);
    assert(r55 < r33);
    assert(r88 < r66 && r66 < r55); // strict ladder 88>66>55
    assert(rKK < rAA);  // set of kings beats overpair aces
    std::cout << "PASSED (88=" << r88 << " 66=" << r66 << " 55=" << r55
              << " 33=" << r33 << " KKset=" << rKK << " AA=" << rAA << ")\n";
}

int main() {
    std::cout << "=== DeepSolver Hand Evaluator Tests ===\n";

    test_card_parsing();
    test_combo_table();
    test_hand_rankings();
    test_7card_evaluation();
    test_hand_categories();
    test_hash_collisions();
    test_pocket_pair_ordering();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
