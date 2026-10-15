#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "cpu_simd.h"

namespace deepsolver {

/// 2026-10-08: the averaged strategy every postsolve pass and output reads —
/// per player node a row [a * nc + c], summing to 1 over actions per hand.
///
/// The reference, GPU and decomposition backends hand over materialized rows.
/// LevelizedCpuBackend hands over a VIEW of its cumulative strategy sums
/// instead, normalized per (node, hand) on read with the arithmetic its
/// finalize() used to materialize them with. The rows were one full float
/// copy of the largest host object, alive next to the CFR state: 9.7 of the
/// 31.4 GB the lean BTN-vs-BB flop needed.
///
/// A view borrows the backend's buffers, so it is valid until that backend
/// prepares again or is freed — the Solver drops it before either, and
/// release_backend() hands the buffers over first (adopt()).
class FinalStrategy {
public:
    /// Resolved node locks: (node, canonical hand) -> frequencies per action.
    using Locks = std::map<std::pair<uint32_t, uint16_t>, std::vector<float>>;

    FinalStrategy() = default;
    explicit FinalStrategy(std::vector<std::vector<float>> rows)
        : rows_(std::move(rows)) {}
    // Moves keep an adopted buffer's address (vector moves steal it); a copy
    // would still point into the source's buffers.
    FinalStrategy(FinalStrategy&&) = default;
    FinalStrategy& operator=(FinalStrategy&&) = default;
    FinalStrategy(const FinalStrategy&) = delete;
    FinalStrategy& operator=(const FinalStrategy&) = delete;

    /// `sums` laid out [offsets[node] + a * stride + c]; offsets[node] ==
    /// SIZE_MAX for nodes without a decision. 16-bit sums (one scale per
    /// node, which normalization cancels) need no scales here.
    static FinalStrategy view(const float* sums, const std::size_t* offsets,
                              std::size_t stride, const uint8_t* num_children,
                              uint16_t nc) {
        FinalStrategy f = view_of(offsets, stride, num_children, nc);
        f.sums_ = sums;
        return f;
    }
    static FinalStrategy view(const uint16_t* sums, const std::size_t* offsets,
                              std::size_t stride, const uint8_t* num_children,
                              uint16_t nc) {
        FinalStrategy f = view_of(offsets, stride, num_children, nc);
        f.sums16_ = sums;
        return f;
    }

    bool empty() const { return rows_.empty() && !view_; }

    /// Node `node`'s row (na * nc floats), or nullptr when it has none.
    /// Materialized rows are returned in place; a view decodes into `scratch`.
    const float* row(uint32_t node, std::vector<float>& scratch) const {
        if (!view_) {
            if (node >= rows_.size() || rows_[node].empty()) return nullptr;
            return rows_[node].data();
        }
        const std::size_t off = offsets_[node];
        if (off == static_cast<std::size_t>(-1)) return nullptr;
        const uint8_t na = num_children_[node];
        scratch.assign(static_cast<std::size_t>(na) * nc_, 0.0f);
        if (sums16_ != nullptr) normalize(sums16_ + off, na, scratch.data());
        else                    normalize(sums_ + off, na, scratch.data());
        for (auto it = locks_.lower_bound(Locks::key_type{node, uint16_t{0}});
             it != locks_.end() && it->first.first == node; ++it) {
            const auto& forced = it->second;
            for (uint8_t a = 0; a < na && a < forced.size(); ++a) {
                scratch[static_cast<std::size_t>(a) * nc_ + it->first.second] = forced[a];
            }
        }
        return scratch.data();
    }

    /// Locked hands play their lock: written over materialized rows, applied
    /// on read by a view.
    void apply_locks(const Locks& locks, uint16_t nc) {
        if (view_) { locks_ = locks; return; }
        for (const auto& [key, forced] : locks) {
            const auto [node_idx, canonical_idx] = key;
            if (node_idx >= rows_.size()) continue;
            for (std::size_t a = 0; a < forced.size(); ++a) {
                const std::size_t idx = a * nc + canonical_idx;
                if (idx < rows_[node_idx].size()) rows_[node_idx][idx] = forced[a];
            }
        }
    }

    /// Does this view read `sums`?
    bool views(const float* sums) const { return view_ && sums16_ == nullptr && sums_ == sums; }
    bool views(const uint16_t* sums) const { return view_ && sums_ == nullptr && sums16_ == sums; }

    /// Take over the buffers a view reads, so it outlives their backend.
    void adopt(std::vector<float>&& sums, std::vector<std::size_t>&& offsets) {
        owned_sums_ = std::move(sums);
        owned_offsets_ = std::move(offsets);
        sums_ = owned_sums_.data();
        offsets_ = owned_offsets_.data();
    }
    void adopt(std::vector<uint16_t>&& sums, std::vector<std::size_t>&& offsets) {
        owned_sums16_ = std::move(sums);
        owned_offsets_ = std::move(offsets);
        sums16_ = owned_sums16_.data();
        offsets_ = owned_offsets_.data();
    }

private:
    static FinalStrategy view_of(const std::size_t* offsets, std::size_t stride,
                                 const uint8_t* num_children, uint16_t nc) {
        FinalStrategy f;
        f.view_ = true;
        f.offsets_ = offsets;
        f.stride_ = stride;
        f.num_children_ = num_children;
        f.nc_ = nc;
        return f;
    }

    /// One node's sums -> rows [a * nc + c], each hand normalized over its
    /// actions; uniform when its sum is below kMinRegretSum (16-bit: zero).
    template <typename T>
    void normalize(const T* sum_base, uint8_t na, float* out) const {
        const uint16_t nc = nc_;
        for (uint16_t c = 0; c < nc; ++c) {
            float total = 0.0f;
            for (uint8_t a = 0; a < na; ++a) {
                total += static_cast<float>(sum_base[static_cast<std::size_t>(a) * stride_ + c]);
            }
            if (total >= kMinRegretSum) {
                float inv = 1.0f / total;
                for (uint8_t a = 0; a < na; ++a) {
                    out[static_cast<std::size_t>(a) * nc + c] =
                        static_cast<float>(sum_base[static_cast<std::size_t>(a) * stride_ + c]) * inv;
                }
            } else {
                float uniform = 1.0f / na;
                for (uint8_t a = 0; a < na; ++a) {
                    out[static_cast<std::size_t>(a) * nc + c] = uniform;
                }
            }
        }
    }

    std::vector<std::vector<float>> rows_;
    bool view_ = false;
    const float* sums_ = nullptr;
    const uint16_t* sums16_ = nullptr;
    const std::size_t* offsets_ = nullptr;
    std::size_t stride_ = 0;
    const uint8_t* num_children_ = nullptr;
    uint16_t nc_ = 0;
    Locks locks_;
    std::vector<float> owned_sums_;
    std::vector<uint16_t> owned_sums16_;
    std::vector<std::size_t> owned_offsets_;
};

}  // namespace deepsolver
