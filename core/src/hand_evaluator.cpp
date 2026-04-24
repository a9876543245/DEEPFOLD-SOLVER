/**
 * @file hand_evaluator.cpp
 * @brief Implementation of the two-step lookup hand evaluator.
 *
 * Clean-room design. The evaluation logic works as follows:
 *
 * For 5-card evaluation:
 *   1. Compute rank_bits = OR of (1 << rank) for each card
 *   2. If all cards share a suit (flush), look up flush_table[rank_bits]
 *   3. If popcount(rank_bits) == 5 (all unique ranks), look up unique5_table[rank_bits]
 *      This catches straights and high-card-only hands.
 *   4. Otherwise, compute prime_product = product of rank primes, hash into hash_table
 *      This handles pairs, trips, quads, full houses, two pairs.
 *
 * For 7-card evaluation:
 *   Iterate over all C(7,5)=21 combinations and return the minimum (best) rank.
 */

#include "hand_evaluator.h"
#include <numeric>
#include <bit>       // std::popcount (C++20)
#include <cassert>
#include <map>

namespace deepsolver {

// ============================================================================
// 5-card hand rank computation (used to build tables)
// ============================================================================

namespace {

/// Straight patterns: bit patterns for A-high down to A-low (wheel)
constexpr uint16_t STRAIGHT_PATTERNS[10] = {
    0b1111100000000,  // A-K-Q-J-T
    0b0111110000000,  // K-Q-J-T-9
    0b0011111000000,  // Q-J-T-9-8
    0b0001111100000,  // J-T-9-8-7
    0b0000111110000,  // T-9-8-7-6
    0b0000011111000,  // 9-8-7-6-5
    0b0000001111100,  // 8-7-6-5-4
    0b0000000111110,  // 7-6-5-4-3
    0b0000000011111,  // 6-5-4-3-2
    0b1000000001111,  // 5-4-3-2-A (wheel)
};

/// Check if rank_bits form a straight, return straight "height" (0-9) or -1
int find_straight(uint16_t rank_bits) {
    for (int i = 0; i < 10; ++i) {
        if ((rank_bits & STRAIGHT_PATTERNS[i]) == STRAIGHT_PATTERNS[i]) {
            return i;
        }
    }
    return -1;
}

/// Count how many bits are set in the lower 13 bits
int popcount13(uint16_t v) {
    return std::popcount(static_cast<uint16_t>(v & 0x1FFF));
}

/// Compute the "raw" 5-card hand rank for table building.
/// This is the slow/reference evaluator that populates tables.
///
/// Returns rank in [1, 7462]:
///   1-10:    Straight Flush
///   11-166:  Four of a Kind
///   167-322: Full House
///   323-1599: Flush
///   1600-1609: Straight
///   1610-2467: Three of a Kind
///   2468-3325: Two Pair
///   3326-6185: One Pair
///   6186-7462: High Card
uint16_t compute_5card_rank(const Card cards[5]) {
    // Count ranks
    uint8_t rank_count[13] = {};
    uint16_t rank_bits = 0;
    bool is_flush = true;
    Suit first_suit = card_suit(cards[0]);

    for (int i = 0; i < 5; ++i) {
        Rank r = card_rank(cards[i]);
        rank_count[r]++;
        rank_bits |= (1u << r);
        if (card_suit(cards[i]) != first_suit) is_flush = false;
    }

    int unique_ranks = popcount13(rank_bits);
    int straight_idx = find_straight(rank_bits);

    // Classify the hand
    if (is_flush && straight_idx >= 0) {
        // Straight Flush (ranks 1-10)
        return static_cast<uint16_t>(1 + straight_idx);
    }

    // Sort rank counts to classify
    // Collect (count, rank) pairs sorted by count desc, rank desc
    struct RankInfo { uint8_t count; uint8_t rank; };
    RankInfo infos[5];
    int n_info = 0;
    for (int r = 12; r >= 0; --r) {
        if (rank_count[r] > 0) {
            infos[n_info++] = {rank_count[r], static_cast<uint8_t>(r)};
        }
    }
    // Stable sort by count descending
    std::sort(infos, infos + n_info, [](const RankInfo& a, const RankInfo& b) {
        return a.count > b.count || (a.count == b.count && a.rank > b.rank);
    });

    if (infos[0].count == 4) {
        // Four of a Kind (ranks 11-166)
        // 13 possible quad ranks × 12 kickers = 156, + offsets
        static uint16_t base = 11;
        uint16_t quad_rank_offset = (12 - infos[0].rank) * 12;
        uint16_t kicker_offset = 12 - infos[1].rank;
        if (kicker_offset >= (12 - infos[0].rank)) kicker_offset--;
        return base + quad_rank_offset + kicker_offset;
    }

    if (infos[0].count == 3 && infos[1].count == 2) {
        // Full House (ranks 167-322)
        static uint16_t base = 167;
        uint16_t trips_offset = (12 - infos[0].rank) * 12;
        uint16_t pair_offset = 12 - infos[1].rank;
        if (pair_offset >= (12 - infos[0].rank)) pair_offset--;
        return base + trips_offset + pair_offset;
    }

    if (is_flush) {
        // Flush (ranks 323-1599)
        // Ranked by highest cards. We use the rank_bits pattern.
        // We'll enumerate all flush patterns later during init.
        // For now, return a placeholder that will be overwritten.
        static uint16_t flush_counter = 323;
        return flush_counter++;  // Will be properly set in init
    }

    if (straight_idx >= 0) {
        // Straight (ranks 1600-1609)
        return static_cast<uint16_t>(1600 + straight_idx);
    }

    if (infos[0].count == 3) {
        // Three of a Kind (ranks 1610-2467)
        static uint16_t base = 1610;
        uint16_t trips_offset = (12 - infos[0].rank);
        // Two kickers sorted desc
        uint16_t k1 = 12 - infos[1].rank;
        uint16_t k2 = 12 - infos[2].rank;
        // C(12, 2) = 66 kicker combos per trips rank
        uint16_t kicker_idx = k1 * (k1 - 1) / 2 + k2 - (k1 >= (12 - infos[0].rank) ? 1 : 0);
        return base + trips_offset * 66 + std::min(kicker_idx, static_cast<uint16_t>(65));
    }

    if (infos[0].count == 2 && infos[1].count == 2) {
        // Two Pair (ranks 2468-3325)
        static uint16_t base = 2468;
        uint16_t p1 = 12 - infos[0].rank;
        uint16_t p2 = 12 - infos[1].rank;
        uint16_t k  = 12 - infos[2].rank;
        uint16_t pair_idx = p1 * (p1 - 1) / 2 + p2;
        return base + pair_idx * 11 + std::min(k, static_cast<uint16_t>(10));
    }

    if (infos[0].count == 2) {
        // One Pair (ranks 3326-6185)
        static uint16_t base = 3326;
        uint16_t pair_rank = 12 - infos[0].rank;
        // 3 kickers from remaining 12 ranks → C(12,3) = 220 combos
        uint16_t k1 = 12 - infos[1].rank;
        uint16_t k2 = 12 - infos[2].rank;
        uint16_t k3 = 12 - infos[3].rank;
        return base + pair_rank * 220 + std::min(static_cast<uint16_t>(k1 * 66 + k2 * 11 + k3),
                                                  static_cast<uint16_t>(219));
    }

    // High Card (ranks 6186-7462)
    static uint16_t base = 6186;
    // 5 unique ranks, not a straight, not a flush
    // C(13,5) - 10 straights = 1277 high card combos
    uint16_t h = 0;
    for (int i = 0; i < n_info && i < 5; ++i) {
        h = h * 13 + (12 - infos[i].rank);
    }
    return base + (h % 1277);
}

} // anonymous namespace

// ============================================================================
// HandEvaluator Implementation
// ============================================================================

void HandEvaluator::init_flush_table() {
    // Enumerate all C(13,5) = 1287 possible 5-rank patterns for flushes
    // Assign hand ranks in order (best to worst)

    // Collect all 5-rank bit patterns
    struct FlushEntry {
        uint16_t bits;
        int straight_idx;  // -1 if not straight
        // Sorted rank values (highest first)
        uint8_t ranks[5];
    };

    std::vector<FlushEntry> entries;
    entries.reserve(1287);

    for (int a = 12; a >= 4; --a) {
        for (int b = a-1; b >= 3; --b) {
            for (int c = b-1; c >= 2; --c) {
                for (int d = c-1; d >= 1; --d) {
                    for (int e = d-1; e >= 0; --e) {
                        uint16_t bits = (1u << a) | (1u << b) | (1u << c) |
                                       (1u << d) | (1u << e);
                        int si = find_straight(bits);
                        entries.push_back({bits, si,
                            {static_cast<uint8_t>(a), static_cast<uint8_t>(b),
                             static_cast<uint8_t>(c), static_cast<uint8_t>(d),
                             static_cast<uint8_t>(e)}});
                    }
                }
            }
        }
    }

    // Sort: straight flushes first (by straight height), then non-straight flushes by rank
    std::sort(entries.begin(), entries.end(), [](const FlushEntry& a, const FlushEntry& b) {
        bool a_sf = a.straight_idx >= 0;
        bool b_sf = b.straight_idx >= 0;
        if (a_sf != b_sf) return a_sf;  // SFs before non-SFs
        if (a_sf && b_sf) return a.straight_idx < b.straight_idx;
        // Compare by highest rank descending
        for (int i = 0; i < 5; ++i) {
            if (a.ranks[i] != b.ranks[i]) return a.ranks[i] > b.ranks[i];
        }
        return false;
    });

    // Assign ranks: straight flushes = 1-10, flushes = 323-1599
    flush_table_.fill(0);
    uint16_t sf_rank = 1;
    uint16_t flush_rank = 323;
    for (const auto& entry : entries) {
        if (entry.straight_idx >= 0) {
            flush_table_[entry.bits] = sf_rank++;
        } else {
            flush_table_[entry.bits] = flush_rank++;
        }
    }
}

void HandEvaluator::init_unique5_table() {
    // For non-flush hands with 5 unique ranks: straights and high cards
    unique5_table_.fill(0);

    // Collect all 5-unique-rank patterns (same as flush but ranked differently)
    struct UniqueEntry {
        uint16_t bits;
        int straight_idx;
        uint8_t ranks[5];
    };

    std::vector<UniqueEntry> entries;
    entries.reserve(1287);

    for (int a = 12; a >= 4; --a) {
        for (int b = a-1; b >= 3; --b) {
            for (int c = b-1; c >= 2; --c) {
                for (int d = c-1; d >= 1; --d) {
                    for (int e = d-1; e >= 0; --e) {
                        uint16_t bits = (1u << a) | (1u << b) | (1u << c) |
                                       (1u << d) | (1u << e);
                        int si = find_straight(bits);
                        entries.push_back({bits, si,
                            {static_cast<uint8_t>(a), static_cast<uint8_t>(b),
                             static_cast<uint8_t>(c), static_cast<uint8_t>(d),
                             static_cast<uint8_t>(e)}});
                    }
                }
            }
        }
    }

    // Sort: straights by height, then high cards by rank
    std::sort(entries.begin(), entries.end(), [](const UniqueEntry& a, const UniqueEntry& b) {
        bool a_s = a.straight_idx >= 0;
        bool b_s = b.straight_idx >= 0;
        if (a_s != b_s) return a_s;
        if (a_s && b_s) return a.straight_idx < b.straight_idx;
        for (int i = 0; i < 5; ++i) {
            if (a.ranks[i] != b.ranks[i]) return a.ranks[i] > b.ranks[i];
        }
        return false;
    });

    uint16_t straight_rank = 1600;
    uint16_t highcard_rank = 6186;
    for (const auto& entry : entries) {
        if (entry.straight_idx >= 0) {
            unique5_table_[entry.bits] = straight_rank++;
        } else {
            unique5_table_[entry.bits] = highcard_rank++;
        }
    }
}

void HandEvaluator::init_hash_table() {
    // For non-flush, non-unique-rank hands (pairs, trips, quads, boat, two pair)
    // We use the product of rank primes as a hash key.

    hash_table_.fill(0);

    // We need to enumerate all possible rank multisets for:
    //   - One Pair: (2,1,1,1) → 13 * C(12,3) = 2860 combos
    //   - Two Pair: (2,2,1) → C(13,2) * 11 = 858 combos
    //   - Three of a Kind: (3,1,1) → 13 * C(12,2) = 858 combos
    //   - Full House: (3,2) → 13 * 12 = 156 combos
    //   - Four of a Kind: (4,1) → 13 * 12 = 156 combos

    struct HashEntry {
        uint32_t prime_product;
        uint16_t rank;
    };

    std::vector<HashEntry> entries;

    // === Four of a Kind ===
    {
        uint16_t r = 11;
        for (int q = 12; q >= 0; --q) {
            int qp = RANK_PRIMES[q];
            uint32_t quad_prime = qp * qp * qp * qp;
            for (int k = 12; k >= 0; --k) {
                if (k == q) continue;
                uint32_t pp = quad_prime * RANK_PRIMES[k];
                entries.push_back({pp, r++});
            }
        }
    }

    // === Full House ===
    {
        uint16_t r = 167;
        for (int t = 12; t >= 0; --t) {
            int tp = RANK_PRIMES[t];
            uint32_t trip_prime = tp * tp * tp;
            for (int p = 12; p >= 0; --p) {
                if (p == t) continue;
                uint32_t pp = trip_prime * RANK_PRIMES[p] * RANK_PRIMES[p];
                entries.push_back({pp, r++});
            }
        }
    }

    // === Three of a Kind ===
    {
        uint16_t r = 1610;
        for (int t = 12; t >= 0; --t) {
            int tp = RANK_PRIMES[t];
            uint32_t trip_prime = tp * tp * tp;
            for (int k1 = 12; k1 >= 1; --k1) {
                if (k1 == t) continue;
                for (int k2 = k1 - 1; k2 >= 0; --k2) {
                    if (k2 == t) continue;
                    uint32_t pp = trip_prime * RANK_PRIMES[k1] * RANK_PRIMES[k2];
                    entries.push_back({pp, r++});
                }
            }
        }
    }

    // === Two Pair ===
    {
        uint16_t r = 2468;
        for (int p1 = 12; p1 >= 1; --p1) {
            for (int p2 = p1 - 1; p2 >= 0; --p2) {
                uint32_t pair_prime = RANK_PRIMES[p1] * RANK_PRIMES[p1] *
                                     RANK_PRIMES[p2] * RANK_PRIMES[p2];
                for (int k = 12; k >= 0; --k) {
                    if (k == p1 || k == p2) continue;
                    uint32_t pp = pair_prime * RANK_PRIMES[k];
                    entries.push_back({pp, r++});
                }
            }
        }
    }

    // === One Pair ===
    {
        uint16_t r = 3326;
        for (int p = 12; p >= 0; --p) {
            uint32_t pair_prime = RANK_PRIMES[p] * RANK_PRIMES[p];
            for (int k1 = 12; k1 >= 2; --k1) {
                if (k1 == p) continue;
                for (int k2 = k1 - 1; k2 >= 1; --k2) {
                    if (k2 == p) continue;
                    for (int k3 = k2 - 1; k3 >= 0; --k3) {
                        if (k3 == p) continue;
                        uint32_t pp = pair_prime * RANK_PRIMES[k1] *
                                      RANK_PRIMES[k2] * RANK_PRIMES[k3];
                        entries.push_back({pp, r++});
                    }
                }
            }
        }
    }

    // Store in hash table using simple modular hashing
    for (const auto& e : entries) {
        uint32_t idx = e.prime_product % HASH_TABLE_SIZE;
        // Linear probing for collision resolution
        while (hash_table_[idx] != 0) {
            idx = (idx + 1) % HASH_TABLE_SIZE;
        }
        hash_table_[idx] = e.rank;
    }

    // Also store the prime products for reverse lookup during evaluation
    // We'll use a separate map for the actual evaluation
}

void HandEvaluator::initialize() {
    if (initialized_) return;

    init_flush_table();
    init_unique5_table();
    init_hash_table();

    initialized_ = true;
}

uint16_t HandEvaluator::evaluate5(Card c0, Card c1, Card c2, Card c3, Card c4) const {
    assert(initialized_);

    // Step 1: Compute rank bits and check for flush
    uint16_t rank_bits = rank_bit(c0) | rank_bit(c1) | rank_bit(c2) |
                         rank_bit(c3) | rank_bit(c4);

    bool is_flush = (card_suit(c0) == card_suit(c1)) &&
                    (card_suit(c1) == card_suit(c2)) &&
                    (card_suit(c2) == card_suit(c3)) &&
                    (card_suit(c3) == card_suit(c4));

    // Step 2a: Flush → direct lookup
    if (is_flush) {
        return flush_table_[rank_bits];
    }

    // Step 2b: All unique ranks → unique5 lookup (straights + high cards)
    if (popcount13(rank_bits) == 5) {
        return unique5_table_[rank_bits];
    }

    // Step 2c: Hash lookup for pairs/trips/quads/boats/two-pairs
    uint32_t prime_product = static_cast<uint32_t>(rank_prime(c0)) *
                             rank_prime(c1) * rank_prime(c2) *
                             rank_prime(c3) * rank_prime(c4);

    uint32_t idx = prime_product % HASH_TABLE_SIZE;
    // Linear probe to find the entry
    while (hash_table_[idx] == 0) {
        idx = (idx + 1) % HASH_TABLE_SIZE;
    }
    return hash_table_[idx];
}

uint16_t HandEvaluator::evaluate(Card c0, Card c1, Card c2, Card c3,
                                  Card c4, Card c5, Card c6) const {
    assert(initialized_);

    // Evaluate all C(7,5) = 21 combinations, return the best (minimum rank)
    Card cards[7] = {c0, c1, c2, c3, c4, c5, c6};
    uint16_t best = NUM_HAND_RANKS + 1;

    // Enumerate all 21 5-card subsets
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 7; ++j) {
            // Skip cards i and j
            Card sub[5];
            int k = 0;
            for (int m = 0; m < 7; ++m) {
                if (m != i && m != j) {
                    sub[k++] = cards[m];
                }
            }
            uint16_t rank = evaluate5(sub[0], sub[1], sub[2], sub[3], sub[4]);
            if (rank < best) best = rank;
        }
    }

    return best;
}

// ============================================================================
// Global Instance
// ============================================================================

HandEvaluator& get_evaluator() {
    static HandEvaluator instance;
    return instance;
}

} // namespace deepsolver
