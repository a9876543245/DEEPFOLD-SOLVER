/**
 * @file showdown_rank_blocker.h
 * @brief Rank-prefix + blocker shortcut for dense singleton showdown terminals.
 *
 * The dense matrix showdown path computes, for each self combo c:
 *
 *   sum_j compatible(c,j) * opp_reach_w[j] * payoff(rank(c), rank(j))
 *
 * When every canonical combo is a single original combo, the category matrix is
 * exactly rank comparison plus private-card compatibility. We can replace the
 * O(nc^2) table scan with:
 *
 *   1. aggregate opponent reach by hand rank,
 *   2. build rank prefixes globally and per blocker card,
 *   3. answer each combo with prefix differences and two card-blocker
 *      subtractions.
 *
 * Canonical buckets with multiple originals deliberately fall back to the
 * matrix path because their category is a majority bucket over original pairs.
 */

#pragma once

#include "card.h"
#include "hand_evaluator.h"
#include "isomorphism.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace deepsolver::showdown_rank_blocker {

struct Scratch {
    static constexpr uint16_t kNoBucket = std::numeric_limits<uint16_t>::max();

    std::vector<uint16_t> rank_to_bucket;
    std::vector<uint16_t> touched_ranks;
    std::vector<float> bucket_total;
    std::vector<float> bucket_prefix;
    std::vector<float> card_bucket_blocked;
    std::vector<float> card_bucket_prefix;
    std::size_t bucket_capacity = 0;

    void ensure(std::size_t max_buckets) {
        if (rank_to_bucket.empty()) {
            rank_to_bucket.assign(NUM_HAND_RANKS + 1, kNoBucket);
        }
        if (bucket_capacity < max_buckets) {
            bucket_capacity = max_buckets;
            bucket_total.resize(bucket_capacity);
            bucket_prefix.resize(bucket_capacity + 1);
            card_bucket_blocked.resize(
                static_cast<std::size_t>(NUM_CARDS) * bucket_capacity);
            card_bucket_prefix.resize(
                static_cast<std::size_t>(NUM_CARDS) * (bucket_capacity + 1));
        }
        touched_ranks.clear();
    }
};

struct Metadata {
    bool valid = false;
    std::size_t bucket_count = 0;
    std::vector<uint16_t> combo_bucket;
    std::vector<Card> combo_card0;
    std::vector<Card> combo_card1;
};

inline bool supports_singleton_iso(const IsomorphismMapping& iso) {
    for (uint16_t c = 0; c < iso.num_canonical; ++c) {
        if (iso.canonical_to_originals[c].size() != 1u) return false;
        if (iso.canonical_weights[c] != 1u) return false;
    }
    return true;
}

inline Metadata build_metadata(
    const IsomorphismMapping& iso,
    const std::vector<uint16_t>& original_ranks)
{
    Metadata meta;
    const uint16_t nc = iso.num_canonical;
    if (original_ranks.size() < NUM_COMBOS || nc == 0) return meta;
    if (!supports_singleton_iso(iso)) return meta;

    std::vector<uint16_t> rank_to_bucket(
        NUM_HAND_RANKS + 1, Scratch::kNoBucket);
    std::vector<uint16_t> touched_ranks;
    touched_ranks.reserve(nc);

    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }
        if (rank_to_bucket[rank] == Scratch::kNoBucket) {
            rank_to_bucket[rank] = 0;
            touched_ranks.push_back(rank);
        }
    }

    std::sort(touched_ranks.begin(), touched_ranks.end());
    if (touched_ranks.empty()) return meta;
    for (std::size_t b = 0; b < touched_ranks.size(); ++b) {
        rank_to_bucket[touched_ranks[b]] = static_cast<uint16_t>(b);
    }

    const auto& combo_table = get_combo_table();
    meta.combo_bucket.assign(nc, Scratch::kNoBucket);
    meta.combo_card0.assign(nc, 0);
    meta.combo_card1.assign(nc, 0);
    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }
        meta.combo_bucket[c] = rank_to_bucket[rank];
        const Combo& combo = combo_table[oi];
        meta.combo_card0[c] = combo.cards[0];
        meta.combo_card1[c] = combo.cards[1];
    }
    meta.bucket_count = touched_ranks.size();
    meta.valid = true;
    return meta;
}

inline void build_prefixes(
    const Metadata& meta,
    const float* opp_reach_w,
    const uint16_t* opp_active,
    std::size_t opp_count,
    Scratch& scratch)
{
    const std::size_t B = meta.bucket_count;
    scratch.ensure(B);

    std::fill(scratch.bucket_total.begin(),
              scratch.bucket_total.begin() + static_cast<std::ptrdiff_t>(B),
              0.0f);

    const std::size_t bucket_stride = scratch.bucket_capacity;
    const std::size_t prefix_stride = scratch.bucket_capacity + 1;
    for (Card card = 0; card < NUM_CARDS; ++card) {
        float* blocked = scratch.card_bucket_blocked.data()
            + static_cast<std::size_t>(card) * bucket_stride;
        std::fill(blocked, blocked + static_cast<std::ptrdiff_t>(B), 0.0f);
    }

    for (std::size_t k = 0; k < opp_count; ++k) {
        const uint16_t c = opp_active
            ? opp_active[k]
            : static_cast<uint16_t>(k);
        const uint16_t b = meta.combo_bucket[c];
        if (b == Scratch::kNoBucket) continue;
        const float r = opp_reach_w[c];
        if (r == 0.0f) continue;

        scratch.bucket_total[b] += r;
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(meta.combo_card0[c]) * bucket_stride + b] += r;
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(meta.combo_card1[c]) * bucket_stride + b] += r;
    }

    scratch.bucket_prefix[0] = 0.0f;
    for (std::size_t b = 0; b < B; ++b) {
        scratch.bucket_prefix[b + 1] =
            scratch.bucket_prefix[b] + scratch.bucket_total[b];
    }

    for (Card card = 0; card < NUM_CARDS; ++card) {
        float acc = 0.0f;
        float* prefix = scratch.card_bucket_prefix.data()
            + static_cast<std::size_t>(card) * prefix_stride;
        const float* blocked = scratch.card_bucket_blocked.data()
            + static_cast<std::size_t>(card) * bucket_stride;
        prefix[0] = 0.0f;
        for (std::size_t b = 0; b < B; ++b) {
            acc += blocked[b];
            prefix[b + 1] = acc;
        }
    }
}

inline float card_prefix_sum(
    const Scratch& scratch,
    Card card,
    std::size_t lo,
    std::size_t hi)
{
    const std::size_t prefix_stride = scratch.bucket_capacity + 1;
    const float* prefix = scratch.card_bucket_prefix.data()
        + static_cast<std::size_t>(card) * prefix_stride;
    return prefix[hi] - prefix[lo];
}

inline float combo_value(
    const Metadata& meta,
    const float* opp_reach_w,
    uint16_t c,
    float win_p,
    float lose_p,
    float tie_p,
    const Scratch& scratch)
{
    const uint16_t b16 = meta.combo_bucket[c];
    if (b16 == Scratch::kNoBucket) return 0.0f;

    const std::size_t b = b16;
    const std::size_t B = meta.bucket_count;
    const Card c0 = meta.combo_card0[c];
    const Card c1 = meta.combo_card1[c];
    const float total_reach = scratch.bucket_prefix[B];

    const float stronger =
        scratch.bucket_prefix[b]
        - card_prefix_sum(scratch, c0, 0, b)
        - card_prefix_sum(scratch, c1, 0, b);
    const float weaker =
        (total_reach - scratch.bucket_prefix[b + 1])
        - card_prefix_sum(scratch, c0, b + 1, B)
        - card_prefix_sum(scratch, c1, b + 1, B);
    const float same =
        scratch.bucket_total[b]
        - card_prefix_sum(scratch, c0, b, b + 1)
        - card_prefix_sum(scratch, c1, b, b + 1)
        + opp_reach_w[c];

    return win_p * weaker + lose_p * stronger + tie_p * same;
}

inline float combo_value_no_tie(
    const Metadata& meta,
    const float* opp_reach_w,
    uint16_t c,
    float win_p,
    float lose_p,
    const Scratch& scratch)
{
    (void)opp_reach_w;
    const uint16_t b16 = meta.combo_bucket[c];
    if (b16 == Scratch::kNoBucket) return 0.0f;

    const std::size_t b = b16;
    const std::size_t B = meta.bucket_count;
    const Card c0 = meta.combo_card0[c];
    const Card c1 = meta.combo_card1[c];
    const float total_reach = scratch.bucket_prefix[B];

    const float stronger =
        scratch.bucket_prefix[b]
        - card_prefix_sum(scratch, c0, 0, b)
        - card_prefix_sum(scratch, c1, 0, b);
    const float weaker =
        (total_reach - scratch.bucket_prefix[b + 1])
        - card_prefix_sum(scratch, c0, b + 1, B)
        - card_prefix_sum(scratch, c1, b + 1, B);

    return win_p * weaker + lose_p * stronger;
}

inline void showdown_dense_singleton_precomputed(
    const Metadata& meta,
    const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out,
    std::size_t out_stride,
    float win_p,
    float lose_p,
    float tie_p,
    Scratch& scratch)
{
    const uint16_t nc = static_cast<uint16_t>(meta.combo_bucket.size());
    if (!meta.valid || nc == 0) {
        std::fill(out, out + out_stride, 0.0f);
        return;
    }

    build_prefixes(meta, opp_reach_w, nullptr, nc, scratch);
    if (tie_p == 0.0f) {
        for (uint16_t c = 0; c < nc; ++c) {
            if (skip_mask != nullptr && skip_mask[c]) {
                out[c] = 0.0f;
                continue;
            }
            out[c] = combo_value_no_tie(
                meta, opp_reach_w, c, win_p, lose_p, scratch);
        }
    } else {
        for (uint16_t c = 0; c < nc; ++c) {
            if (skip_mask != nullptr && skip_mask[c]) {
                out[c] = 0.0f;
                continue;
            }
            out[c] = combo_value(
                meta, opp_reach_w, c, win_p, lose_p, tie_p, scratch);
        }
    }
    for (std::size_t i = nc; i < out_stride; ++i) {
        out[i] = 0.0f;
    }
}

inline void showdown_active_singleton_precomputed(
    const Metadata& meta,
    const float* opp_reach_w,
    const uint16_t* self_active,
    std::size_t self_count,
    const uint16_t* opp_active,
    std::size_t opp_count,
    bool clear_output,
    float* out,
    std::size_t out_stride,
    float win_p,
    float lose_p,
    float tie_p,
    Scratch& scratch)
{
    auto clear_active_or_all = [&]() {
        if (clear_output) {
            std::fill(out, out + out_stride, 0.0f);
        } else if (self_active != nullptr) {
            for (std::size_t k = 0; k < self_count; ++k) {
                out[self_active[k]] = 0.0f;
            }
        }
    };

    if (!meta.valid || self_active == nullptr || opp_active == nullptr) {
        clear_active_or_all();
        return;
    }
    if (clear_output) {
        std::fill(out, out + out_stride, 0.0f);
    }

    build_prefixes(meta, opp_reach_w, opp_active, opp_count, scratch);
    if (tie_p == 0.0f) {
        for (std::size_t k = 0; k < self_count; ++k) {
            const uint16_t c = self_active[k];
            out[c] = combo_value_no_tie(
                meta, opp_reach_w, c, win_p, lose_p, scratch);
        }
    } else {
        for (std::size_t k = 0; k < self_count; ++k) {
            const uint16_t c = self_active[k];
            out[c] = combo_value(
                meta, opp_reach_w, c, win_p, lose_p, tie_p, scratch);
        }
    }
}

inline void showdown_dense_singleton(
    const IsomorphismMapping& iso,
    const std::vector<uint16_t>& original_ranks,
    const float* opp_reach_w,
    const uint8_t* skip_mask,
    float* out,
    std::size_t out_stride,
    float win_p,
    float lose_p,
    float tie_p,
    Scratch& scratch)
{
    const uint16_t nc = iso.num_canonical;
    scratch.ensure(nc);

    if (original_ranks.size() < NUM_COMBOS || nc == 0) {
        std::fill(out, out + out_stride, 0.0f);
        return;
    }

    const auto& combo_table = get_combo_table();

    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }
        if (scratch.rank_to_bucket[rank] == Scratch::kNoBucket) {
            scratch.rank_to_bucket[rank] = 0;
            scratch.touched_ranks.push_back(rank);
        }
    }

    std::sort(scratch.touched_ranks.begin(), scratch.touched_ranks.end());
    const std::size_t B = scratch.touched_ranks.size();
    if (B == 0) {
        std::fill(out, out + out_stride, 0.0f);
        return;
    }

    for (std::size_t b = 0; b < B; ++b) {
        scratch.rank_to_bucket[scratch.touched_ranks[b]] =
            static_cast<uint16_t>(b);
    }

    std::fill(scratch.bucket_total.begin(),
              scratch.bucket_total.begin() + static_cast<std::ptrdiff_t>(B),
              0.0f);
    std::fill(scratch.card_bucket_blocked.begin(),
              scratch.card_bucket_blocked.begin()
                  + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(NUM_CARDS) * scratch.bucket_capacity),
              0.0f);

    const std::size_t bucket_stride = scratch.bucket_capacity;
    const std::size_t prefix_stride = scratch.bucket_capacity + 1;

    for (uint16_t c = 0; c < nc; ++c) {
        const float r = opp_reach_w[c];
        if (r == 0.0f) continue;

        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }

        const uint16_t b = scratch.rank_to_bucket[rank];
        scratch.bucket_total[b] += r;

        const Combo& combo = combo_table[oi];
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(combo.cards[0]) * bucket_stride + b] += r;
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(combo.cards[1]) * bucket_stride + b] += r;
    }

    scratch.bucket_prefix[0] = 0.0f;
    for (std::size_t b = 0; b < B; ++b) {
        scratch.bucket_prefix[b + 1] =
            scratch.bucket_prefix[b] + scratch.bucket_total[b];
    }

    for (Card card = 0; card < NUM_CARDS; ++card) {
        float acc = 0.0f;
        float* prefix = scratch.card_bucket_prefix.data()
            + static_cast<std::size_t>(card) * prefix_stride;
        const float* blocked = scratch.card_bucket_blocked.data()
            + static_cast<std::size_t>(card) * bucket_stride;
        prefix[0] = 0.0f;
        for (std::size_t b = 0; b < B; ++b) {
            acc += blocked[b];
            prefix[b + 1] = acc;
        }
    }

    auto card_prefix_sum = [&](Card card, std::size_t lo, std::size_t hi) {
        const float* prefix = scratch.card_bucket_prefix.data()
            + static_cast<std::size_t>(card) * prefix_stride;
        return prefix[hi] - prefix[lo];
    };

    const float total_reach = scratch.bucket_prefix[B];
    for (uint16_t c = 0; c < nc; ++c) {
        if (skip_mask != nullptr && skip_mask[c]) {
            out[c] = 0.0f;
            continue;
        }

        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            out[c] = 0.0f;
            continue;
        }

        const std::size_t b = scratch.rank_to_bucket[rank];
        const Combo& combo = combo_table[oi];
        const Card c0 = combo.cards[0];
        const Card c1 = combo.cards[1];

        const float stronger =
            scratch.bucket_prefix[b]
            - card_prefix_sum(c0, 0, b)
            - card_prefix_sum(c1, 0, b);
        const float weaker =
            (total_reach - scratch.bucket_prefix[b + 1])
            - card_prefix_sum(c0, b + 1, B)
            - card_prefix_sum(c1, b + 1, B);
        const float same =
            scratch.bucket_total[b]
            - card_prefix_sum(c0, b, b + 1)
            - card_prefix_sum(c1, b, b + 1)
            + opp_reach_w[c];

        out[c] = win_p * weaker + lose_p * stronger + tie_p * same;
    }

    for (std::size_t i = nc; i < out_stride; ++i) {
        out[i] = 0.0f;
    }

    for (uint16_t rank : scratch.touched_ranks) {
        scratch.rank_to_bucket[rank] = Scratch::kNoBucket;
    }
}

inline void showdown_active_singleton(
    const IsomorphismMapping& iso,
    const std::vector<uint16_t>& original_ranks,
    const float* opp_reach_w,
    const uint16_t* self_active,
    std::size_t self_count,
    const uint16_t* opp_active,
    std::size_t opp_count,
    bool clear_output,
    float* out,
    std::size_t out_stride,
    float win_p,
    float lose_p,
    float tie_p,
    Scratch& scratch)
{
    const uint16_t nc = iso.num_canonical;
    scratch.ensure(nc);

    auto clear_active_or_all = [&]() {
        if (clear_output) {
            std::fill(out, out + out_stride, 0.0f);
        } else if (self_active != nullptr) {
            for (std::size_t k = 0; k < self_count; ++k) {
                out[self_active[k]] = 0.0f;
            }
        }
    };

    if (original_ranks.size() < NUM_COMBOS || nc == 0
        || self_active == nullptr || opp_active == nullptr) {
        clear_active_or_all();
        return;
    }
    if (clear_output) {
        std::fill(out, out + out_stride, 0.0f);
    }

    const auto& combo_table = get_combo_table();

    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }
        if (scratch.rank_to_bucket[rank] == Scratch::kNoBucket) {
            scratch.rank_to_bucket[rank] = 0;
            scratch.touched_ranks.push_back(rank);
        }
    }

    std::sort(scratch.touched_ranks.begin(), scratch.touched_ranks.end());
    const std::size_t B = scratch.touched_ranks.size();
    if (B == 0) {
        clear_active_or_all();
        return;
    }

    for (std::size_t b = 0; b < B; ++b) {
        scratch.rank_to_bucket[scratch.touched_ranks[b]] =
            static_cast<uint16_t>(b);
    }

    std::fill(scratch.bucket_total.begin(),
              scratch.bucket_total.begin() + static_cast<std::ptrdiff_t>(B),
              0.0f);
    std::fill(scratch.card_bucket_blocked.begin(),
              scratch.card_bucket_blocked.begin()
                  + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(NUM_CARDS) * scratch.bucket_capacity),
              0.0f);

    const std::size_t bucket_stride = scratch.bucket_capacity;
    const std::size_t prefix_stride = scratch.bucket_capacity + 1;

    for (std::size_t k = 0; k < opp_count; ++k) {
        const uint16_t c = opp_active[k];
        const float r = opp_reach_w[c];
        if (r == 0.0f) continue;

        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            continue;
        }

        const uint16_t b = scratch.rank_to_bucket[rank];
        scratch.bucket_total[b] += r;

        const Combo& combo = combo_table[oi];
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(combo.cards[0]) * bucket_stride + b] += r;
        scratch.card_bucket_blocked[
            static_cast<std::size_t>(combo.cards[1]) * bucket_stride + b] += r;
    }

    scratch.bucket_prefix[0] = 0.0f;
    for (std::size_t b = 0; b < B; ++b) {
        scratch.bucket_prefix[b + 1] =
            scratch.bucket_prefix[b] + scratch.bucket_total[b];
    }

    for (Card card = 0; card < NUM_CARDS; ++card) {
        float acc = 0.0f;
        float* prefix = scratch.card_bucket_prefix.data()
            + static_cast<std::size_t>(card) * prefix_stride;
        const float* blocked = scratch.card_bucket_blocked.data()
            + static_cast<std::size_t>(card) * bucket_stride;
        prefix[0] = 0.0f;
        for (std::size_t b = 0; b < B; ++b) {
            acc += blocked[b];
            prefix[b + 1] = acc;
        }
    }

    auto card_prefix_sum = [&](Card card, std::size_t lo, std::size_t hi) {
        const float* prefix = scratch.card_bucket_prefix.data()
            + static_cast<std::size_t>(card) * prefix_stride;
        return prefix[hi] - prefix[lo];
    };

    const float total_reach = scratch.bucket_prefix[B];
    for (std::size_t k = 0; k < self_count; ++k) {
        const uint16_t c = self_active[k];
        const uint16_t oi = iso.canonical_to_originals[c][0];
        const uint16_t rank = original_ranks[oi];
        if (rank == std::numeric_limits<uint16_t>::max() || rank == 0
            || rank > NUM_HAND_RANKS) {
            out[c] = 0.0f;
            continue;
        }

        const std::size_t b = scratch.rank_to_bucket[rank];
        const Combo& combo = combo_table[oi];
        const Card c0 = combo.cards[0];
        const Card c1 = combo.cards[1];

        const float stronger =
            scratch.bucket_prefix[b]
            - card_prefix_sum(c0, 0, b)
            - card_prefix_sum(c1, 0, b);
        const float weaker =
            (total_reach - scratch.bucket_prefix[b + 1])
            - card_prefix_sum(c0, b + 1, B)
            - card_prefix_sum(c1, b + 1, B);
        const float same =
            scratch.bucket_total[b]
            - card_prefix_sum(c0, b, b + 1)
            - card_prefix_sum(c1, b, b + 1)
            + opp_reach_w[c];

        out[c] = win_p * weaker + lose_p * stronger + tie_p * same;
    }

    for (uint16_t rank : scratch.touched_ranks) {
        scratch.rank_to_bucket[rank] = Scratch::kNoBucket;
    }
}

}  // namespace deepsolver::showdown_rank_blocker
