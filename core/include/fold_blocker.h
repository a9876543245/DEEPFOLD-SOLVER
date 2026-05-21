/**
 * @file fold_blocker.h
 * @brief Blocker-sum shortcut for fold terminal validity reductions.
 *
 * Fold payoffs do not depend on hand strength, only on whether the opponent
 * combo is card-compatible with the traverser's combo on the current board.
 * The dense matrix path computes:
 *
 *   out[c] = payoff * sum_j matchup_valid[c,j] * opp_reach[j] * weight[j]
 *
 * With canonical compression, matchup_valid[c,j] is the fraction of original
 * combo pairs in canonical buckets (c,j) that are card-compatible. That is
 * exactly equal to averaging, over each original combo represented by c, the
 * opponent original-combo reach that is not blocked by either private card.
 *
 * This turns the fold terminal from O(nc^2) matrix-vector work into O(1326)
 * blocker aggregation plus O(1326) output averaging.
 */

#pragma once

#include "card.h"
#include "isomorphism.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deepsolver::fold_blocker {

struct Metadata {
    bool valid = false;
    uint16_t num_canonical = 0;
    std::vector<uint32_t> bucket_offsets;
    std::vector<uint8_t> bucket_card0;
    std::vector<uint8_t> bucket_card1;
    std::vector<float> bucket_denom;
};

inline Metadata build_metadata(
    const IsomorphismMapping& iso,
    CardMask board_mask)
{
    Metadata metadata;
    const uint16_t nc = iso.num_canonical;
    const auto& combo_table = get_combo_table();

    metadata.num_canonical = nc;
    metadata.bucket_offsets.assign(static_cast<std::size_t>(nc) + 1u, 0u);
    metadata.bucket_denom.assign(nc, 1.0f);

    std::size_t total_originals = 0;
    for (uint16_t c = 0; c < nc; ++c) {
        total_originals += iso.canonical_to_originals[c].size();
    }
    metadata.bucket_card0.reserve(total_originals);
    metadata.bucket_card1.reserve(total_originals);

    for (uint16_t c = 0; c < nc; ++c) {
        metadata.bucket_offsets[c] =
            static_cast<uint32_t>(metadata.bucket_card0.size());
        const auto& originals = iso.canonical_to_originals[c];
        metadata.bucket_denom[c] = static_cast<float>(
            std::max<std::size_t>(std::size_t{1}, originals.size()));
        for (uint16_t oi : originals) {
            const Combo& combo = combo_table[oi];
            const Card c0 = combo.cards[0];
            const Card c1 = combo.cards[1];
            const CardMask mask = card_to_mask(c0) | card_to_mask(c1);
            if (mask & board_mask) continue;
            metadata.bucket_card0.push_back(c0);
            metadata.bucket_card1.push_back(c1);
        }
    }
    metadata.bucket_offsets[nc] =
        static_cast<uint32_t>(metadata.bucket_card0.size());
    metadata.valid = metadata.bucket_offsets.size() ==
            static_cast<std::size_t>(nc) + 1u
        && metadata.bucket_card0.size() == metadata.bucket_card1.size()
        && metadata.bucket_denom.size() == nc;
    return metadata;
}

inline void accumulate_opponent(
    const Metadata& metadata,
    const float* opp_reach,
    std::array<float, NUM_CARDS>& blocked_by_card,
    float& total_reach)
{
    blocked_by_card.fill(0.0f);
    total_reach = 0.0f;
    const uint16_t nc = metadata.num_canonical;
    for (uint16_t cj = 0; cj < nc; ++cj) {
        const float r = opp_reach[cj];
        if (r == 0.0f) continue;
        const uint32_t begin = metadata.bucket_offsets[cj];
        const uint32_t end = metadata.bucket_offsets[static_cast<std::size_t>(cj) + 1u];
        for (uint32_t p = begin; p < end; ++p) {
            const uint8_t c0 = metadata.bucket_card0[p];
            const uint8_t c1 = metadata.bucket_card1[p];
            total_reach += r;
            blocked_by_card[c0] += r;
            blocked_by_card[c1] += r;
        }
    }
}

inline void accumulate_opponent_active(
    const Metadata& metadata,
    const float* opp_reach,
    const uint16_t* opp_active_indices,
    std::size_t opp_active_count,
    std::array<float, NUM_CARDS>& blocked_by_card,
    float& total_reach)
{
    blocked_by_card.fill(0.0f);
    total_reach = 0.0f;
    for (std::size_t k = 0; k < opp_active_count; ++k) {
        const uint16_t cj = opp_active_indices[k];
        const float r = opp_reach[cj];
        if (r == 0.0f) continue;
        const uint32_t begin = metadata.bucket_offsets[cj];
        const uint32_t end = metadata.bucket_offsets[static_cast<std::size_t>(cj) + 1u];
        for (uint32_t p = begin; p < end; ++p) {
            const uint8_t c0 = metadata.bucket_card0[p];
            const uint8_t c1 = metadata.bucket_card1[p];
            total_reach += r;
            blocked_by_card[c0] += r;
            blocked_by_card[c1] += r;
        }
    }
}

inline float combo_value(
    const Metadata& metadata,
    const float* opp_reach,
    const std::array<float, NUM_CARDS>& blocked_by_card,
    float total_reach,
    uint16_t ci)
{
    float acc = 0.0f;
    const float exact_combo_reach = opp_reach[ci];
    const uint32_t begin = metadata.bucket_offsets[ci];
    const uint32_t end = metadata.bucket_offsets[static_cast<std::size_t>(ci) + 1u];
    for (uint32_t p = begin; p < end; ++p) {
        const uint8_t c0 = metadata.bucket_card0[p];
        const uint8_t c1 = metadata.bucket_card1[p];
        acc += total_reach
             - blocked_by_card[c0]
             - blocked_by_card[c1]
             + exact_combo_reach;
    }
    return acc / metadata.bucket_denom[ci];
}

inline void fold_dense_precomputed(
    const Metadata& metadata,
    const float* opp_reach,
    const uint8_t* skip_mask,
    float self_payoff,
    float* out,
    std::size_t out_stride)
{
    const uint16_t nc = metadata.num_canonical;
    std::array<float, NUM_CARDS> blocked_by_card{};
    float total_reach = 0.0f;
    accumulate_opponent(metadata, opp_reach, blocked_by_card, total_reach);

    for (uint16_t ci = 0; ci < nc; ++ci) {
        if (skip_mask != nullptr && skip_mask[ci]) {
            out[ci] = 0.0f;
            continue;
        }
        out[ci] = self_payoff * combo_value(
            metadata, opp_reach, blocked_by_card, total_reach, ci);
    }
    for (std::size_t i = nc; i < out_stride; ++i) {
        out[i] = 0.0f;
    }
}

inline void fold_active_precomputed(
    const Metadata& metadata,
    const float* opp_reach,
    const uint16_t* self_active_indices,
    std::size_t self_active_count,
    const uint16_t* opp_active_indices,
    std::size_t opp_active_count,
    bool clear_full_output,
    float self_payoff,
    float* out,
    std::size_t out_stride)
{
    const uint16_t nc = metadata.num_canonical;
    if (clear_full_output) {
        std::fill(out, out + out_stride, 0.0f);
    } else if (out_stride > nc) {
        std::fill(out + nc, out + out_stride, 0.0f);
    }

    std::array<float, NUM_CARDS> blocked_by_card{};
    float total_reach = 0.0f;
    accumulate_opponent_active(
        metadata, opp_reach, opp_active_indices, opp_active_count,
        blocked_by_card, total_reach);

    for (std::size_t k = 0; k < self_active_count; ++k) {
        const uint16_t ci = self_active_indices[k];
        out[ci] = self_payoff * combo_value(
            metadata, opp_reach, blocked_by_card, total_reach, ci);
    }
}

inline void fold_dense(
    const IsomorphismMapping& iso,
    CardMask board_mask,
    const float* opp_reach,
    const uint8_t* skip_mask,
    float self_payoff,
    float* out,
    std::size_t out_stride)
{
    const uint16_t nc = iso.num_canonical;
    const auto& combo_table = get_combo_table();

    std::array<float, NUM_CARDS> blocked_by_card{};
    float total_reach = 0.0f;

    for (uint16_t cj = 0; cj < nc; ++cj) {
        const float r = opp_reach[cj];
        if (r == 0.0f) continue;
        for (uint16_t oj : iso.canonical_to_originals[cj]) {
            const Combo& combo = combo_table[oj];
            const Card c0 = combo.cards[0];
            const Card c1 = combo.cards[1];
            const CardMask mask = card_to_mask(c0) | card_to_mask(c1);
            if (mask & board_mask) continue;

            total_reach += r;
            blocked_by_card[c0] += r;
            blocked_by_card[c1] += r;
        }
    }

    for (uint16_t ci = 0; ci < nc; ++ci) {
        if (skip_mask != nullptr && skip_mask[ci]) {
            out[ci] = 0.0f;
            continue;
        }

        const auto& originals = iso.canonical_to_originals[ci];
        if (originals.empty()) {
            out[ci] = 0.0f;
            continue;
        }

        float acc = 0.0f;
        const float exact_combo_reach = opp_reach[ci];
        for (uint16_t oi : originals) {
            const Combo& combo = combo_table[oi];
            const Card c0 = combo.cards[0];
            const Card c1 = combo.cards[1];
            const CardMask mask = card_to_mask(c0) | card_to_mask(c1);
            if (mask & board_mask) continue;

            acc += total_reach
                 - blocked_by_card[c0]
                 - blocked_by_card[c1]
                 + exact_combo_reach;
        }

        const float denom = static_cast<float>(
            std::max<std::size_t>(std::size_t{1}, originals.size()));
        out[ci] = self_payoff * (acc / denom);
    }

    for (std::size_t i = nc; i < out_stride; ++i) {
        out[i] = 0.0f;
    }
}

inline void fold_active(
    const IsomorphismMapping& iso,
    CardMask board_mask,
    const float* opp_reach,
    const uint16_t* self_active_indices,
    std::size_t self_active_count,
    const uint16_t* opp_active_indices,
    std::size_t opp_active_count,
    bool clear_full_output,
    float self_payoff,
    float* out,
    std::size_t out_stride)
{
    const uint16_t nc = iso.num_canonical;
    const auto& combo_table = get_combo_table();

    if (clear_full_output) {
        std::fill(out, out + out_stride, 0.0f);
    } else if (out_stride > nc) {
        std::fill(out + nc, out + out_stride, 0.0f);
    }

    std::array<float, NUM_CARDS> blocked_by_card{};
    float total_reach = 0.0f;

    for (std::size_t k = 0; k < opp_active_count; ++k) {
        const uint16_t cj = opp_active_indices[k];
        const float r = opp_reach[cj];
        if (r == 0.0f) continue;
        for (uint16_t oj : iso.canonical_to_originals[cj]) {
            const Combo& combo = combo_table[oj];
            const Card c0 = combo.cards[0];
            const Card c1 = combo.cards[1];
            const CardMask mask = card_to_mask(c0) | card_to_mask(c1);
            if (mask & board_mask) continue;

            total_reach += r;
            blocked_by_card[c0] += r;
            blocked_by_card[c1] += r;
        }
    }

    for (std::size_t k = 0; k < self_active_count; ++k) {
        const uint16_t ci = self_active_indices[k];
        const auto& originals = iso.canonical_to_originals[ci];
        if (originals.empty()) {
            out[ci] = 0.0f;
            continue;
        }

        float acc = 0.0f;
        const float exact_combo_reach = opp_reach[ci];
        for (uint16_t oi : originals) {
            const Combo& combo = combo_table[oi];
            const Card c0 = combo.cards[0];
            const Card c1 = combo.cards[1];
            const CardMask mask = card_to_mask(c0) | card_to_mask(c1);
            if (mask & board_mask) continue;

            acc += total_reach
                 - blocked_by_card[c0]
                 - blocked_by_card[c1]
                 + exact_combo_reach;
        }

        const float denom = static_cast<float>(
            std::max<std::size_t>(std::size_t{1}, originals.size()));
        out[ci] = self_payoff * (acc / denom);
    }
}

}  // namespace deepsolver::fold_blocker
