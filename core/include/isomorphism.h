/**
 * @file isomorphism.h
 * @brief Suit isomorphism mapper for reducing the number of canonical combos.
 *
 * Given a board texture, many starting hand combos are strategically equivalent
 * due to suit symmetry. For example, on a rainbow flop (3 different suits),
 * the fourth suit is identical in value to any of the three board suits that
 * don't interact with the board.
 *
 * This module computes the canonical mapping: 1326 combos → ~940 canonical combos
 * (on flop), saving ~30% memory.
 */

#pragma once

#include "types.h"
#include "card.h"
#include <array>
#include <vector>
#include <algorithm>
#include <map>

namespace deepsolver {

// ============================================================================
// Suit Isomorphism Result
// ============================================================================

/// Maps original combo indices to canonical indices
struct IsomorphismMapping {
    /// For each of 1326 combos: its canonical index (in compressed space)
    std::array<uint16_t, NUM_COMBOS> original_to_canonical{};

    /// Total number of canonical combos (< NUM_COMBOS)
    uint16_t num_canonical = 0;

    /// For each canonical index: list of original combo indices that map to it
    std::vector<std::vector<uint16_t>> canonical_to_originals;

    /// Weight of each canonical combo (number of originals it represents)
    std::vector<uint16_t> canonical_weights;
};

// ============================================================================
// Suit Isomorphism Computations
// ============================================================================

/**
 * @brief Compute the canonical suit mapping for a given board.
 *
 * Algorithm:
 *   1. Analyze the board's suit pattern.
 *   2. For each starting combo, compute its "canonical form" by finding the
 *      lexicographically smallest suit permutation that preserves board structure.
 *   3. Group equivalent combos under the same canonical index.
 *
 * @param board Board cards
 * @param board_size Number of board cards (3-5)
 * @return IsomorphismMapping
 */
inline IsomorphismMapping compute_isomorphism(const Card* board, uint8_t board_size) {
    IsomorphismMapping result;

    // Step 1: Determine which suits appear on the board
    uint8_t suit_count[4] = {};
    for (uint8_t i = 0; i < board_size; ++i) {
        suit_count[card_suit(board[i])]++;
    }

    // Step 2: Build suit equivalence classes
    // Suits with the same count on the board can be permuted freely
    // Example: Board = As Kd 7c → suit_count = {1, 1, 0, 1}
    //   Hearts(0 on board) is different from Clubs/Diamonds/Spades(1 each)
    //   But C, D, S can be permuted among themselves

    // Group suits by their board count (and rank pattern for more precision)
    struct SuitSignature {
        uint8_t count;                 ///< How many board cards of this suit
        std::vector<Rank> board_ranks; ///< Which ranks appear (sorted)
    };

    std::array<SuitSignature, 4> suit_sigs;
    for (int s = 0; s < 4; ++s) {
        suit_sigs[s].count = suit_count[s];
        for (uint8_t i = 0; i < board_size; ++i) {
            if (card_suit(board[i]) == s) {
                suit_sigs[s].board_ranks.push_back(card_rank(board[i]));
            }
        }
        std::sort(suit_sigs[s].board_ranks.begin(), suit_sigs[s].board_ranks.end());
    }

    // Generate all valid suit permutations (those that preserve the board)
    // A permutation p[0..3] maps suit i → suit p[i]
    std::vector<std::array<uint8_t, 4>> valid_perms;
    std::array<uint8_t, 4> perm = {0, 1, 2, 3};

    do {
        bool valid = true;
        // Check: board under this permutation must equal the original board
        // (up to card reordering)
        CardMask original_board = 0;
        CardMask permuted_board = 0;

        for (uint8_t i = 0; i < board_size; ++i) {
            original_board |= card_to_mask(board[i]);
            Card permuted_card = make_card(card_rank(board[i]),
                                            static_cast<Suit>(perm[card_suit(board[i])]));
            permuted_board |= card_to_mask(permuted_card);
        }

        if (original_board == permuted_board) {
            valid_perms.push_back(perm);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    // Step 3: For each combo, find its canonical form
    // Canonical = lexicographically smallest combo after applying all valid permutations

    CardMask dead_mask = board_to_mask(board, board_size);
    const auto& combo_table = get_combo_table();

    // Map canonical combo index → set of originals
    std::map<uint32_t, uint16_t> canonical_map; // canonical_key → canonical_id
    uint16_t next_canonical = 0;

    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        const Combo& combo = combo_table[i];

        // Skip combos that conflict with the board
        if (combo.conflicts_with(dead_mask)) {
            result.original_to_canonical[i] = UINT16_MAX; // Invalid
            continue;
        }

        // Find the smallest equivalent combo under all valid permutations
        uint32_t min_key = UINT32_MAX;

        for (const auto& p : valid_perms) {
            Card c0 = make_card(card_rank(combo.cards[0]),
                                static_cast<Suit>(p[card_suit(combo.cards[0])]));
            Card c1 = make_card(card_rank(combo.cards[1]),
                                static_cast<Suit>(p[card_suit(combo.cards[1])]));

            // Ensure c0 < c1 for canonical ordering
            if (c0 > c1) std::swap(c0, c1);

            // Check the permuted combo doesn't conflict with board
            if ((card_to_mask(c0) & dead_mask) || (card_to_mask(c1) & dead_mask)) {
                continue;
            }

            uint32_t key = static_cast<uint32_t>(c0) * 52 + c1;
            min_key = std::min(min_key, key);
        }

        // Map to canonical index
        auto it = canonical_map.find(min_key);
        if (it == canonical_map.end()) {
            canonical_map[min_key] = next_canonical;
            result.original_to_canonical[i] = next_canonical;
            next_canonical++;
        } else {
            result.original_to_canonical[i] = it->second;
        }
    }

    result.num_canonical = next_canonical;

    // Build reverse mapping
    result.canonical_to_originals.resize(next_canonical);
    result.canonical_weights.resize(next_canonical, 0);

    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = result.original_to_canonical[i];
        if (ci != UINT16_MAX) {
            result.canonical_to_originals[ci].push_back(i);
            result.canonical_weights[ci]++;
        }
    }

    return result;
}

// ============================================================================
// Phase 2: Canonical Runout Enumeration (PioSolver / GTO+ style)
// ============================================================================

/// One canonical next-card class to deal at a CHANCE node.
struct CanonicalRunout {
    Card card;       ///< Lex-min representative of this orbit
    uint8_t weight;  ///< Orbit size (= number of undealt cards this rep covers)
};

/// Result of canonical runout enumeration.
struct CanonicalRunouts {
    std::vector<CanonicalRunout> reps;  ///< Distinct equivalence classes
    uint8_t total_weight = 0;           ///< Sum of weights (= undealt-card count)
};

/// Compute the suit-permutation group that fixes a given multiset of board
/// cards. Returns up to 4!=24 permutations, each represented as a length-4
/// array `p` where suit `i` maps to suit `p[i]`. This is the building block
/// for both `compute_isomorphism()` (combo space) and
/// `enumerate_canonical_runouts()` (deck space).
inline std::vector<std::array<uint8_t, 4>>
suit_perms_fixing_board(const Card* board, uint8_t board_size) {
    std::vector<std::array<uint8_t, 4>> valid;
    std::array<uint8_t, 4> perm = {0, 1, 2, 3};

    do {
        CardMask original = 0;
        CardMask permuted = 0;
        for (uint8_t i = 0; i < board_size; ++i) {
            original |= card_to_mask(board[i]);
            Card c = make_card(card_rank(board[i]),
                               static_cast<Suit>(perm[card_suit(board[i])]));
            permuted |= card_to_mask(c);
        }
        if (original == permuted) {
            valid.push_back(perm);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return valid;
}

/**
 * @brief Enumerate canonical next-card deals for the current board.
 *
 * This is the PioSolver / GTO+ style runout iso: given the current full board
 * (config flop ∪ runout cards dealt so far), find the suit-permutation group
 * G that fixes the board, then partition the undealt deck into G-orbits.
 * Emit one representative per orbit (lex-min) with weight = orbit size.
 *
 * Invariants:
 *   - sum(weights) == 52 - full_size  (every undealt card covered exactly once)
 *   - For any two cards in the same orbit, the resulting subgame is isomorphic
 *     under a suit-symmetric range, so the CFR strategies + EVs are identical.
 *     The chance-node weighted average therefore reproduces the un-iso'd EV.
 *
 * Speedup by board texture (per chance level):
 *   - Rainbow (all distinct suits): 1.0x  (G = {id})
 *   - Two-tone (two suits used, one twice): up to 2x
 *   - Monotone flop / three-of-suit turn: up to 3x (G = S_3 over unused suits)
 *   - Single-rank flops with paired suits: 2x  (one suit-swap fixes board)
 *
 * @param full_board Current full board (config flop + cards dealt so far)
 * @param full_size  Number of cards in full_board (3..5)
 */
inline CanonicalRunouts enumerate_canonical_runouts(
    const Card* full_board, uint8_t full_size)
{
    CanonicalRunouts out;
    auto perms = suit_perms_fixing_board(full_board, full_size);

    bool dead[NUM_CARDS] = {};
    for (uint8_t i = 0; i < full_size; ++i) dead[full_board[i]] = true;

    // For each undealt card C, compute its orbit under G and emit only if C
    // is the lex-min of its orbit. This naturally yields one rep per orbit
    // without needing an "already emitted" bitmap.
    for (uint8_t card = 0; card < NUM_CARDS; ++card) {
        if (dead[card]) continue;

        // Compute orbit by applying every perm; dedupe.
        // |perms| <= 24, so the orbit fits in a tiny stack array.
        Card orbit[24];
        uint8_t orbit_size = 0;
        Card lex_min = card;
        for (const auto& p : perms) {
            Card img = make_card(card_rank(card),
                                 static_cast<Suit>(p[card_suit(card)]));
            // Linear-search dedupe (orbit is at most 24 entries)
            bool seen = false;
            for (uint8_t k = 0; k < orbit_size; ++k) {
                if (orbit[k] == img) { seen = true; break; }
            }
            if (!seen) {
                orbit[orbit_size++] = img;
                if (img < lex_min) lex_min = img;
            }
        }

        if (card == lex_min) {
            out.reps.push_back({card, orbit_size});
            out.total_weight = static_cast<uint8_t>(out.total_weight + orbit_size);
        }
    }

    return out;
}

} // namespace deepsolver
