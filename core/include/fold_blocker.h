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
#include "cpu_simd.h"

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
    // 2026-09-11 vector layout of the same buckets (cpu_simd::fold_dense_slots):
    //   owner[p]          canonical index of live original p (flat accumulate)
    //   slot_card0/1      [num_slots][slot_stride], 52 marks an empty slot
    std::vector<uint16_t> owner;
    std::vector<uint8_t> slot_card0;
    std::vector<uint8_t> slot_card1;
    std::size_t slot_stride = 0;
    std::size_t num_slots = 0;
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
            metadata.owner.push_back(c);
        }
    }
    metadata.bucket_offsets[nc] =
        static_cast<uint32_t>(metadata.bucket_card0.size());
    // Slot layout: slot s of canonical c is its s-th live original.
    metadata.slot_stride = (static_cast<std::size_t>(nc) + 7u) & ~static_cast<std::size_t>(7u);
    metadata.num_slots = 0;
    for (uint16_t c = 0; c < nc; ++c) {
        const std::size_t sz = metadata.bucket_offsets[c + 1u] - metadata.bucket_offsets[c];
        if (sz > metadata.num_slots) metadata.num_slots = sz;
    }
    metadata.slot_card0.assign(metadata.num_slots * metadata.slot_stride, 52u);
    metadata.slot_card1.assign(metadata.num_slots * metadata.slot_stride, 52u);
    for (uint16_t c = 0; c < nc; ++c) {
        const uint32_t begin = metadata.bucket_offsets[c];
        const uint32_t end = metadata.bucket_offsets[c + 1u];
        for (uint32_t p = begin; p < end; ++p) {
            const std::size_t at = static_cast<std::size_t>(p - begin) * metadata.slot_stride + c;
            metadata.slot_card0[at] = metadata.bucket_card0[p];
            metadata.slot_card1[at] = metadata.bucket_card1[p];
        }
    }
    metadata.valid = metadata.bucket_offsets.size() ==
            static_cast<std::size_t>(nc) + 1u
        && metadata.bucket_card0.size() == metadata.bucket_card1.size()
        && metadata.bucket_denom.size() == nc;
    return metadata;
}

// Flat over the live originals (same order as the bucket loop it replaces,
// so bit-identical), which removes the data-dependent inner loop that
// mispredicted on every bucket. Zero reach is still skipped: on a river
// most hands are dead and the three dependent adds per original are the
// whole cost.
// `blocked_by_card` needs NUM_CARDS + 1 entries; [52] stays 0 for the empty
// slot of the vector kernel.
inline void accumulate_opponent(
    const Metadata& metadata,
    const float* opp_reach,
    float* blocked_by_card,
    float& total_reach)
{
    for (std::size_t k = 0; k <= NUM_CARDS; ++k) blocked_by_card[k] = 0.0f;
    total_reach = 0.0f;
    const std::size_t total_originals = metadata.owner.size();
    for (std::size_t p = 0; p < total_originals; ++p) {
        const float r = opp_reach[metadata.owner[p]];
        if (r == 0.0f) continue;
        total_reach += r;
        blocked_by_card[metadata.bucket_card0[p]] += r;
        blocked_by_card[metadata.bucket_card1[p]] += r;
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
    alignas(32) float blocked_by_card[56];   // 53 used, 56 readable (vector lookup)
    float total_reach = 0.0f;
    accumulate_opponent(metadata, opp_reach, blocked_by_card, total_reach);

    if (skip_mask == nullptr && !metadata.slot_card0.empty()) {
        cpu_simd::fold_dense_slots(
            metadata.slot_card0.data(), metadata.slot_card1.data(),
            metadata.slot_stride, metadata.num_slots,
            metadata.bucket_denom.data(), blocked_by_card, total_reach,
            opp_reach, self_payoff, out, nc);
    } else {
        std::array<float, NUM_CARDS> blocked{};
        for (std::size_t k = 0; k < NUM_CARDS; ++k) blocked[k] = blocked_by_card[k];
        for (uint16_t ci = 0; ci < nc; ++ci) {
            if (skip_mask != nullptr && skip_mask[ci]) {
                out[ci] = 0.0f;
                continue;
            }
            out[ci] = self_payoff * combo_value(
                metadata, opp_reach, blocked, total_reach, ci);
        }
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

/// Card-compatible opponent mass per canonical hand on `board_mask`:
///
///   out[c] = Σ_j opp_reach[j] · weight[j] · valid_board[c, j]
///
/// i.e. exactly the fold terminal's reduction with payoff 1 (valid is pure
/// card compatibility, averaged over c's originals). This is the denominator
/// every conditional quantity needs (2026-09-09 audit): a hand's EV is
/// conditional on the opponent holding a hand it CAN hold, and the
/// exploitability is an expectation over the LEGAL joint deal, not over the
/// product of the two unconditional range masses. `opp_reach` is the RAW
/// per-canonical reach (fold_dense weights each original itself).
inline void compatible_opponent_mass(
    const IsomorphismMapping& iso,
    CardMask board_mask,
    const float* opp_reach,
    float* out)
{
    fold_dense(iso, board_mask, opp_reach, /*skip_mask=*/nullptr,
               /*self_payoff=*/1.0f, out, iso.num_canonical);
}

/// Σ over card-compatible ORIGINAL hand pairs of self_reach · opp_reach —
/// the mass of the legal joint deal. Zero means the two ranges cannot both be
/// dealt on this board (every pair shares a card, or a range is empty).
inline double legal_joint_mass(
    const IsomorphismMapping& iso,
    CardMask board_mask,
    const std::vector<float>& self_reach,
    const std::vector<float>& opp_reach)
{
    const uint16_t nc = iso.num_canonical;
    if (nc == 0 || self_reach.size() < nc || opp_reach.size() < nc) return 0.0;
    std::vector<float> compat(nc, 0.0f);
    compatible_opponent_mass(iso, board_mask, opp_reach.data(), compat.data());
    double mass = 0.0;
    for (uint16_t c = 0; c < nc; ++c) {
        mass += static_cast<double>(self_reach[c])
              * static_cast<double>(iso.canonical_weights[c])
              * static_cast<double>(compat[c]);
    }
    return mass;
}

}  // namespace deepsolver::fold_blocker
