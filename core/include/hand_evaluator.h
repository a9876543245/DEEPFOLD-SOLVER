/**
 * @file hand_evaluator.h
 * @brief High-performance 7-card poker hand evaluator using two-step lookup.
 *
 * Clean-room implementation inspired by Cactus Kev's approach:
 *   Step 1: Hash 7 cards into a unique key via prime products
 *   Step 2: Look up the hand rank from a pre-computed table
 *
 * Lower rank values = stronger hands (1 = Royal Flush).
 * The lookup table is ~130 KB and fits in GPU shared memory.
 */

#pragma once

#include "types.h"
#include <array>
#include <algorithm>
#include <cstring>

namespace deepsolver {

// ============================================================================
// Constants
// ============================================================================

/// Total distinct 7-card hand equivalence classes
/// In standard poker, there are 4,824 distinct 5-card hand ranks.
/// For 7-card evaluation, we choose the best 5 from 7.
constexpr int NUM_HAND_RANKS = 7462;

/// Hand category boundaries (lower rank = stronger)
enum HandCategory : uint16_t {
    STRAIGHT_FLUSH_MAX  = 10,
    FOUR_OF_A_KIND_MAX  = 166,
    FULL_HOUSE_MAX      = 322,
    FLUSH_MAX           = 1599,
    STRAIGHT_MAX        = 1609,
    THREE_OF_A_KIND_MAX = 2467,
    TWO_PAIR_MAX        = 3325,
    ONE_PAIR_MAX        = 6185,
    HIGH_CARD_MAX       = 7462,
};

/// Get hand category string from rank
inline const char* hand_category_name(uint16_t rank) {
    if (rank <= STRAIGHT_FLUSH_MAX)  return "Straight Flush";
    if (rank <= FOUR_OF_A_KIND_MAX)  return "Four of a Kind";
    if (rank <= FULL_HOUSE_MAX)      return "Full House";
    if (rank <= FLUSH_MAX)           return "Flush";
    if (rank <= STRAIGHT_MAX)        return "Straight";
    if (rank <= THREE_OF_A_KIND_MAX) return "Three of a Kind";
    if (rank <= TWO_PAIR_MAX)        return "Two Pair";
    if (rank <= ONE_PAIR_MAX)        return "One Pair";
    return "High Card";
}

// ============================================================================
// Prime number mapping for rank hashing
// ============================================================================

/// Each rank maps to a unique prime for multiplicative hashing
constexpr int RANK_PRIMES[13] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41
};

// ============================================================================
// Lookup Tables
// ============================================================================

/**
 * @brief Hand evaluator with pre-computed lookup tables.
 *
 * Usage:
 *   HandEvaluator eval;
 *   eval.initialize();  // Call once at startup
 *   uint16_t rank = eval.evaluate(card1, card2, card3, card4, card5, card6, card7);
 *   // Lower rank = better hand. 1 = best (Royal Flush).
 */
class HandEvaluator {
public:
    /// Initialize all lookup tables. MUST be called before any evaluation.
    void initialize();

    /// Evaluate the best 5-card hand from exactly 7 cards.
    /// Returns a rank in [1, 7462]. Lower = stronger.
    uint16_t evaluate(Card c0, Card c1, Card c2, Card c3,
                      Card c4, Card c5, Card c6) const;

    /// Evaluate 5-card hand directly.
    uint16_t evaluate5(Card c0, Card c1, Card c2, Card c3, Card c4) const;

    /// Get the flush table for GPU upload (read-only pointer)
    const uint16_t* get_flush_table() const { return flush_table_.data(); }

    /// Get the unique5 table for GPU upload
    const uint16_t* get_unique5_table() const { return unique5_table_.data(); }

    /// Get the hash-based table for GPU upload
    const uint16_t* get_hash_table() const { return hash_table_.data(); }

    /// Sizes for GPU memory allocation
    static constexpr size_t FLUSH_TABLE_SIZE = 8192;
    static constexpr size_t UNIQUE5_TABLE_SIZE = 8192;
    static constexpr size_t HASH_TABLE_SIZE = 65536;

    bool is_initialized() const { return initialized_; }

private:
    /// Flush lookup: indexed by OR of rank bit patterns when flush detected
    std::array<uint16_t, FLUSH_TABLE_SIZE> flush_table_{};

    /// Unique-5 lookup: for hands with all distinct ranks (straights, high cards)
    std::array<uint16_t, UNIQUE5_TABLE_SIZE> unique5_table_{};

    /// General hash lookup: indexed by prime product hash for remaining cases
    std::array<uint16_t, HASH_TABLE_SIZE> hash_table_{};

    /// Helper: generate the rank bitmask for a card
    static uint16_t rank_bit(Card c) { return 1u << card_rank(c); }

    /// Helper: get prime for a card's rank
    static int rank_prime(Card c) { return RANK_PRIMES[card_rank(c)]; }

    /// Internal: populate flush table
    void init_flush_table();

    /// Internal: populate unique-5 table
    void init_unique5_table();

    /// Internal: populate hash table
    void init_hash_table();

    /// Internal: evaluate exactly 5 cards (core logic)
    uint16_t eval5_internal(uint16_t rank_bits, int prime_product, bool is_flush) const;

    bool initialized_ = false;
};

// ============================================================================
// Global Instance
// ============================================================================

/// Global hand evaluator instance (initialized once in main/init)
HandEvaluator& get_evaluator();

} // namespace deepsolver
