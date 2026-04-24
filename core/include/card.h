/**
 * @file card.h
 * @brief Card utility functions: parsing, formatting, combo enumeration.
 */

#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <stdexcept>

namespace deepsolver {

// ============================================================================
// Card String Conversion
// ============================================================================

/// Rank characters: "23456789TJQKA"
constexpr char RANK_CHARS[NUM_RANKS + 1] = "23456789TJQKA";

/// Suit characters: "cdhs"
constexpr char SUIT_CHARS[NUM_SUITS + 1] = "cdhs";

/// Suit symbols for display (ASCII-safe)
constexpr const char* SUIT_SYMBOLS[NUM_SUITS] = {"c", "d", "h", "s"};
constexpr const char* SUIT_UNICODE[NUM_SUITS] = {"\u2663", "\u2666", "\u2665", "\u2660"};

/// Parse a two-character card string (e.g., "As") to Card value
inline Card parse_card(const std::string& s) {
    if (s.size() != 2) {
        throw std::invalid_argument("Card string must be 2 characters: " + s);
    }

    Rank r = NUM_RANKS;
    for (uint8_t i = 0; i < NUM_RANKS; ++i) {
        if (s[0] == RANK_CHARS[i]) { r = static_cast<Rank>(i); break; }
    }
    if (r == NUM_RANKS) {
        throw std::invalid_argument("Invalid rank character: " + std::string(1, s[0]));
    }

    Suit su = NUM_SUITS;
    for (uint8_t i = 0; i < NUM_SUITS; ++i) {
        if (s[1] == SUIT_CHARS[i]) { su = static_cast<Suit>(i); break; }
    }
    if (su == NUM_SUITS) {
        throw std::invalid_argument("Invalid suit character: " + std::string(1, s[1]));
    }

    return make_card(r, su);
}

/// Convert Card to two-character string (e.g., Card(50) -> "As")
inline std::string card_to_string(Card c) {
    if (c >= NUM_CARDS) return "??";
    std::string s(2, ' ');
    s[0] = RANK_CHARS[card_rank(c)];
    s[1] = SUIT_CHARS[card_suit(c)];
    return s;
}

/// Parse a board string like "AsKd7c" into array of Cards
inline std::vector<Card> parse_board(const std::string& s) {
    if (s.size() % 2 != 0 || s.size() < 6 || s.size() > 10) {
        throw std::invalid_argument("Board must be 3-5 cards (6-10 chars): " + s);
    }
    std::vector<Card> board;
    board.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) {
        board.push_back(parse_card(s.substr(i, 2)));
    }
    return board;
}

/// Convert board array to string
inline std::string board_to_string(const Card* board, uint8_t size) {
    std::string s;
    s.reserve(size * 2);
    for (uint8_t i = 0; i < size; ++i) {
        s += card_to_string(board[i]);
    }
    return s;
}

// ============================================================================
// Combo Enumeration
// ============================================================================

/// Combo = ordered pair of cards (card1 < card2)
struct Combo {
    Card cards[2];

    Combo() : cards{0, 0} {}
    Combo(Card c1, Card c2) {
        if (c1 < c2) { cards[0] = c1; cards[1] = c2; }
        else         { cards[0] = c2; cards[1] = c1; }
    }

    /// Get a unique index in [0, 1325] for this combo
    /// Formula: index = c2*(c2-1)/2 + c1  (where c1 < c2)
    uint16_t index() const {
        return static_cast<uint16_t>(cards[1]) * (cards[1] - 1) / 2 + cards[0];
    }

    /// Convert to display string (e.g., "AhKh")
    std::string to_string() const {
        return card_to_string(cards[0]) + card_to_string(cards[1]);
    }

    /// Check if this combo conflicts with a card mask (dead cards)
    bool conflicts_with(CardMask mask) const {
        return (card_to_mask(cards[0]) & mask) || (card_to_mask(cards[1]) & mask);
    }
};

/// Lookup table: combo_index -> Combo (pre-computed at init)
/// combo_index_table[i] gives the Combo with index i.
inline const std::array<Combo, NUM_COMBOS>& get_combo_table() {
    static bool initialized = false;
    static std::array<Combo, NUM_COMBOS> table{};
    if (!initialized) {
        uint16_t idx = 0;
        for (Card c2 = 1; c2 < NUM_CARDS; ++c2) {
            for (Card c1 = 0; c1 < c2; ++c1) {
                table[idx] = Combo(c1, c2);
                ++idx;
            }
        }
        initialized = true;
    }
    return table;
}

/// Get all valid combos given dead cards (board + opponent cards)
inline std::vector<uint16_t> get_valid_combos(CardMask dead_cards) {
    const auto& table = get_combo_table();
    std::vector<uint16_t> valid;
    valid.reserve(NUM_COMBOS);
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        if (!table[i].conflicts_with(dead_cards)) {
            valid.push_back(i);
        }
    }
    return valid;
}

/// Build dead card mask from board
inline CardMask board_to_mask(const Card* board, uint8_t size) {
    CardMask mask = 0;
    for (uint8_t i = 0; i < size; ++i) {
        mask |= card_to_mask(board[i]);
    }
    return mask;
}

// ============================================================================
// Hand Category Labels (for 13x13 grid display)
// ============================================================================

/// Returns the grid label for a combo (e.g., "AA", "AKs", "AKo")
inline std::string combo_to_grid_label(const Combo& combo) {
    Rank r1 = card_rank(combo.cards[0]);
    Rank r2 = card_rank(combo.cards[1]);
    Suit s1 = card_suit(combo.cards[0]);
    Suit s2 = card_suit(combo.cards[1]);

    // Ensure higher rank first for display
    Rank high = (r1 >= r2) ? r1 : r2;
    Rank low  = (r1 >= r2) ? r2 : r1;

    std::string label;
    label += RANK_CHARS[high];
    if (high == low) {
        label += RANK_CHARS[low];  // Pair: "AA"
    } else {
        label += RANK_CHARS[low];
        label += (s1 == s2) ? 's' : 'o';  // "AKs" or "AKo"
    }
    return label;
}

} // namespace deepsolver
