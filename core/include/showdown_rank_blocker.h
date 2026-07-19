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
 * Multi-original canonical buckets (iso boards) use the same sweep run over
 * ORIGINAL combos, averaged back to canonical -- the fold_blocker.h argument
 * with a rank payoff inserted. Two facts make that exact rather than an
 * approximation, and neither is visible from this file alone:
 *
 *   - A canonical bucket is an orbit of the suit permutations that fix the
 *     board (isomorphism.h:98-115). Hand evaluation is invariant under a
 *     global suit relabel, so every original in a bucket has the SAME rank.
 *     Rank therefore stays per-canonical; only the blocker cards need the
 *     per-original expansion.
 *   - Because rank is constant per bucket, compute_matchup_for_board's
 *     ev_val = (oop_wins - ip_wins) / valid is always exactly -1, 0 or +1, so
 *     the category thresholding at solver.h:1734-1736 is lossless and the
 *     dense cell it feeds is the plain orbit-pair average. Expanding to
 *     originals and averaging back reproduces it up to FP summation order.
 *
 * Dense divides the pair count by |Oi|*|Oj| and the kernel multiplies
 * canonical_weights[cj] = |Oj| back in (eval_kernel.cu:386), so the opponent
 * orbit size cancels and `denom` below is the SELF orbit size only. denom uses
 * the full flop-based orbit size while the CSR lists only runout-live
 * originals -- matching fold_blocker.h:65-75 and dense's valid_val.
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
    /// Rank bucket per canonical combo. Well-defined even for multi-original
    /// buckets: rank is constant within an orbit (see file header).
    std::vector<uint16_t> combo_bucket;
    /// Singleton fast path only (valid when `singleton`): the bucket's sole
    /// original's two cards, indexed by canonical.
    std::vector<Card> combo_card0;
    std::vector<Card> combo_card1;

    /// General (iso) path: CSR from canonical to its runout-live originals.
    /// `orig_card*[bucket_offsets[c] .. bucket_offsets[c+1])` are the cards of
    /// the originals represented by canonical c.
    std::vector<uint32_t> bucket_offsets;
    std::vector<Card> orig_card0;
    std::vector<Card> orig_card1;
    /// Self orbit size (= iso.canonical_weights), the divisor that averages the
    /// per-original sweep back to canonical. Full flop orbit size, NOT the live
    /// count -- dense normalizes by the same thing.
    std::vector<float> denom;
    /// True when every canonical is exactly one original of weight 1, i.e. the
    /// shipped singleton kernels apply verbatim.
    bool singleton = false;
};

inline bool supports_singleton_iso(const IsomorphismMapping& iso) {
    for (uint16_t c = 0; c < iso.num_canonical; ++c) {
        if (iso.canonical_to_originals[c].size() != 1u) return false;
        if (iso.canonical_weights[c] != 1u) return false;
    }
    return true;
}

inline bool rank_is_live(uint16_t rank) {
    return rank != std::numeric_limits<uint16_t>::max() && rank != 0
        && rank <= NUM_HAND_RANKS;
}

inline Metadata build_metadata(
    const IsomorphismMapping& iso,
    const std::vector<uint16_t>& original_ranks)
{
    Metadata meta;
    const uint16_t nc = iso.num_canonical;
    if (original_ranks.size() < NUM_COMBOS || nc == 0) return meta;

    // Rank is constant within an orbit, so a canonical's rank is that of ANY
    // live original. It must be the first LIVE one, not originals[0] -- on a
    // runout board the lex-min original can be blocked while its orbit mates
    // are not. (Singleton buckets make the two coincide, which is why the
    // pre-iso version could index [0] directly.)
    std::vector<uint16_t> canonical_rank(
        nc, std::numeric_limits<uint16_t>::max());
    for (uint16_t c = 0; c < nc; ++c) {
        for (uint16_t oi : iso.canonical_to_originals[c]) {
            const uint16_t r = original_ranks[oi];
            if (rank_is_live(r)) { canonical_rank[c] = r; break; }
        }
    }

    std::vector<uint16_t> rank_to_bucket(
        NUM_HAND_RANKS + 1, Scratch::kNoBucket);
    std::vector<uint16_t> touched_ranks;
    touched_ranks.reserve(nc);

    for (uint16_t c = 0; c < nc; ++c) {
        const uint16_t rank = canonical_rank[c];
        if (!rank_is_live(rank)) continue;
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
    meta.singleton = supports_singleton_iso(iso);
    meta.combo_bucket.assign(nc, Scratch::kNoBucket);
    meta.combo_card0.assign(nc, 0);
    meta.combo_card1.assign(nc, 0);
    meta.bucket_offsets.assign(static_cast<std::size_t>(nc) + 1u, 0u);
    meta.denom.assign(nc, 1.0f);

    std::size_t total_originals = 0;
    for (uint16_t c = 0; c < nc; ++c) {
        total_originals += iso.canonical_to_originals[c].size();
    }
    meta.orig_card0.reserve(total_originals);
    meta.orig_card1.reserve(total_originals);

    for (uint16_t c = 0; c < nc; ++c) {
        meta.bucket_offsets[c] = static_cast<uint32_t>(meta.orig_card0.size());
        const auto& originals = iso.canonical_to_originals[c];
        meta.denom[c] = static_cast<float>(
            std::max<std::size_t>(std::size_t{1}, originals.size()));

        const uint16_t rank = canonical_rank[c];
        if (!rank_is_live(rank)) continue;  // dead bucket: kNoBucket, empty span
        meta.combo_bucket[c] = rank_to_bucket[rank];

        for (uint16_t oi : originals) {
            if (!rank_is_live(original_ranks[oi])) continue;  // runout-blocked
            const Combo& combo = combo_table[oi];
            meta.orig_card0.push_back(combo.cards[0]);
            meta.orig_card1.push_back(combo.cards[1]);
        }
        if (meta.singleton) {
            const Combo& combo = combo_table[originals[0]];
            meta.combo_card0[c] = combo.cards[0];
            meta.combo_card1[c] = combo.cards[1];
        }
    }
    meta.bucket_offsets[nc] = static_cast<uint32_t>(meta.orig_card0.size());
    meta.bucket_count = touched_ranks.size();
    meta.valid = true;
    return meta;
}

/// Size the scratch for `meta` and zero the accumulators. Shared head of the
/// singleton and iso sweeps.
inline void reset_buckets(const Metadata& meta, Scratch& scratch) {
    const std::size_t B = meta.bucket_count;
    scratch.ensure(B);

    std::fill(scratch.bucket_total.begin(),
              scratch.bucket_total.begin() + static_cast<std::ptrdiff_t>(B),
              0.0f);

    const std::size_t bucket_stride = scratch.bucket_capacity;
    for (Card card = 0; card < NUM_CARDS; ++card) {
        float* blocked = scratch.card_bucket_blocked.data()
            + static_cast<std::size_t>(card) * bucket_stride;
        std::fill(blocked, blocked + static_cast<std::ptrdiff_t>(B), 0.0f);
    }
}

/// Turn the accumulated per-bucket reach into the global and per-card rank
/// prefixes the combo_value queries read. Shared tail of both sweeps.
inline void finalize_prefixes(const Metadata& meta, Scratch& scratch) {
    const std::size_t B = meta.bucket_count;
    const std::size_t bucket_stride = scratch.bucket_capacity;
    const std::size_t prefix_stride = scratch.bucket_capacity + 1;

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

inline void build_prefixes(
    const Metadata& meta,
    const float* opp_reach_w,
    const uint16_t* opp_active,
    std::size_t opp_count,
    Scratch& scratch)
{
    reset_buckets(meta, scratch);
    const std::size_t bucket_stride = scratch.bucket_capacity;

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

    finalize_prefixes(meta, scratch);
}

/// Iso-board opponent aggregation: same rank bucketing as `build_prefixes`, but
/// each canonical fans out to its live originals so the blocker corrections are
/// exact.
///
/// !! REACH CONVENTION DIFFERS FROM THE SINGLETON PATH !!
/// `build_prefixes` takes reach ALREADY multiplied by canonical_weights (the
/// `opp_reach_w` the CPU terminal builds at cpu_backend_levelized.h:1640). The
/// iso sweep takes RAW canonical reach, because fanning each canonical out to
/// its live originals already contributes the |Oj| factor -- passing weighted
/// reach here would apply the orbit size twice. This mirrors fold_blocker.h:96,
/// which is raw for the same reason. The two conventions coincide on singleton
/// boards (all weights 1), which is why the shipped path can take either.
inline void build_prefixes_iso(
    const Metadata& meta,
    const float* opp_reach,
    const uint16_t* opp_active,
    std::size_t opp_count,
    Scratch& scratch)
{
    reset_buckets(meta, scratch);
    const std::size_t bucket_stride = scratch.bucket_capacity;

    for (std::size_t k = 0; k < opp_count; ++k) {
        const uint16_t c = opp_active
            ? opp_active[k]
            : static_cast<uint16_t>(k);
        const uint16_t b = meta.combo_bucket[c];
        if (b == Scratch::kNoBucket) continue;
        const float r = opp_reach[c];
        if (r == 0.0f) continue;

        // Rank is orbit-constant, so `b` is hoisted: only the cards fan out.
        const uint32_t begin = meta.bucket_offsets[c];
        const uint32_t end =
            meta.bucket_offsets[static_cast<std::size_t>(c) + 1u];
        for (uint32_t p = begin; p < end; ++p) {
            scratch.bucket_total[b] += r;
            scratch.card_bucket_blocked[
                static_cast<std::size_t>(meta.orig_card0[p]) * bucket_stride + b] += r;
            scratch.card_bucket_blocked[
                static_cast<std::size_t>(meta.orig_card1[p]) * bucket_stride + b] += r;
        }
    }

    finalize_prefixes(meta, scratch);
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

/// Iso-board per-canonical showdown value: run the singleton query once per
/// live original of `ci`, then average by the orbit size. Reduces to
/// `combo_value` exactly when the orbit is a singleton (denom == 1).
/// `opp_reach` is RAW canonical reach -- see build_prefixes_iso.
inline float combo_value_iso(
    const Metadata& meta,
    const float* opp_reach,
    uint16_t ci,
    float win_p,
    float lose_p,
    float tie_p,
    const Scratch& scratch)
{
    const uint16_t b16 = meta.combo_bucket[ci];
    if (b16 == Scratch::kNoBucket) return 0.0f;

    const std::size_t b = b16;
    const std::size_t B = meta.bucket_count;
    const float total_reach = scratch.bucket_prefix[B];
    // The only opponent original holding BOTH of oi's cards is oi itself, so
    // each card term subtracts it once -- add it back. Every original in the
    // orbit maps to canonical ci, making this loop-invariant, and it lands in
    // the tie term because rank(oi) == rank(ci).
    const float self_reach = opp_reach[ci];

    const uint32_t begin = meta.bucket_offsets[ci];
    const uint32_t end = meta.bucket_offsets[static_cast<std::size_t>(ci) + 1u];
    float acc = 0.0f;
    for (uint32_t p = begin; p < end; ++p) {
        const Card c0 = meta.orig_card0[p];
        const Card c1 = meta.orig_card1[p];
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
            + self_reach;
        acc += win_p * weaker + lose_p * stronger + tie_p * same;
    }
    return acc / meta.denom[ci];
}

inline float combo_value_no_tie_iso(
    const Metadata& meta,
    const float* opp_reach,
    uint16_t ci,
    float win_p,
    float lose_p,
    const Scratch& scratch)
{
    (void)opp_reach;
    const uint16_t b16 = meta.combo_bucket[ci];
    if (b16 == Scratch::kNoBucket) return 0.0f;

    const std::size_t b = b16;
    const std::size_t B = meta.bucket_count;
    const float total_reach = scratch.bucket_prefix[B];

    const uint32_t begin = meta.bucket_offsets[ci];
    const uint32_t end = meta.bucket_offsets[static_cast<std::size_t>(ci) + 1u];
    float acc = 0.0f;
    for (uint32_t p = begin; p < end; ++p) {
        const Card c0 = meta.orig_card0[p];
        const Card c1 = meta.orig_card1[p];
        const float stronger =
            scratch.bucket_prefix[b]
            - card_prefix_sum(scratch, c0, 0, b)
            - card_prefix_sum(scratch, c1, 0, b);
        const float weaker =
            (total_reach - scratch.bucket_prefix[b + 1])
            - card_prefix_sum(scratch, c0, b + 1, B)
            - card_prefix_sum(scratch, c1, b + 1, B);
        acc += win_p * weaker + lose_p * stronger;
    }
    return acc / meta.denom[ci];
}

/// `opp_reach` is RAW canonical reach -- see build_prefixes_iso.
inline void showdown_dense_iso_precomputed(
    const Metadata& meta,
    const float* opp_reach,
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

    build_prefixes_iso(meta, opp_reach, nullptr, nc, scratch);
    if (tie_p == 0.0f) {
        for (uint16_t c = 0; c < nc; ++c) {
            if (skip_mask != nullptr && skip_mask[c]) {
                out[c] = 0.0f;
                continue;
            }
            out[c] = combo_value_no_tie_iso(
                meta, opp_reach, c, win_p, lose_p, scratch);
        }
    } else {
        for (uint16_t c = 0; c < nc; ++c) {
            if (skip_mask != nullptr && skip_mask[c]) {
                out[c] = 0.0f;
                continue;
            }
            out[c] = combo_value_iso(
                meta, opp_reach, c, win_p, lose_p, tie_p, scratch);
        }
    }
    for (std::size_t i = nc; i < out_stride; ++i) {
        out[i] = 0.0f;
    }
}

/// `opp_reach` is RAW canonical reach -- see build_prefixes_iso.
inline void showdown_active_iso_precomputed(
    const Metadata& meta,
    const float* opp_reach,
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

    build_prefixes_iso(meta, opp_reach, opp_active, opp_count, scratch);
    if (tie_p == 0.0f) {
        for (std::size_t k = 0; k < self_count; ++k) {
            const uint16_t c = self_active[k];
            out[c] = combo_value_no_tie_iso(
                meta, opp_reach, c, win_p, lose_p, scratch);
        }
    } else {
        for (std::size_t k = 0; k < self_count; ++k) {
            const uint16_t c = self_active[k];
            out[c] = combo_value_iso(
                meta, opp_reach, c, win_p, lose_p, tie_p, scratch);
        }
    }
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
