#pragma once

// GPU B3 inc 1 — the single definition of how many rows the device value
// buffer needs.
//
// `node_values` used to be [N][nc], one row per node. It is now two regions:
//
//   TERMINAL rows        one per terminal, packed. A terminal's value depends
//                        only on reach, so all of them are written in one
//                        launch after the forward pass; each is then read by
//                        its parent during the backward ascent. That span is
//                        the whole ascent, so these rows cannot be windowed.
//   NON-TERMINAL rows    a two-level rolling window keyed on (depth parity,
//                        slot within level). A non-terminal's value is written
//                        when its level is aggregated and read one level up,
//                        so levels L and L+1 must coexist and nothing else
//                        must. Level L-1 shares parity with L+1 and overwrites
//                        exactly the rows that just went dead.
//
// The estimator and GpuBackend::prepare() must agree on the resulting row
// count to the byte — Phase 0's "estimate == allocation" contract is what
// makes the VRAM gate trustworthy — so both call THIS function rather than
// each deriving it. prepare() additionally hard-checks its built table against
// the returned count.
//
// Measured on the 4.65M-node target (m0_b6_standard): 2,950,659 terminal rows
// (63.4% of N) + a 954,081-row non-terminal window (20.5%) = 83.9% of N.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "types.h"

namespace deepsolver {

/// Depth from root for every node (root = 0, every child = parent + 1).
/// Requires the BFS-flattened tree (child index > parent index), which
/// GameTreeBuilder guarantees.
inline std::vector<uint32_t> gpu_depth_from_root(const FlatGameTree& tree) {
    const uint32_t N = tree.total_nodes;
    std::vector<uint32_t> depth(N, 0);
    for (uint32_t n = 0; n < N; ++n) {
        const uint8_t na = tree.num_children[n];
        for (uint8_t a = 0; a < na; ++a) {
            const uint32_t child = tree.children[tree.children_offset[n] + a];
            if (child < N) depth[child] = depth[n] + 1;
        }
    }
    return depth;
}

/// Rows in the device value buffer: terminals (all of them) plus the widest
/// non-terminal level of each depth parity.
inline uint64_t gpu_value_rows(const FlatGameTree& tree) {
    const uint32_t N = tree.total_nodes;
    if (N == 0) return 0;

    const std::vector<uint32_t> depth = gpu_depth_from_root(tree);
    uint32_t num_levels = 0;
    for (uint32_t d : depth) num_levels = std::max(num_levels, d + 1u);

    std::vector<uint32_t> nt_width(num_levels, 0);
    uint64_t terminals = 0;
    for (uint32_t n = 0; n < N; ++n) {
        if (static_cast<NodeType>(tree.node_types[n]) == NodeType::TERMINAL) {
            ++terminals;
        } else {
            nt_width[depth[n]]++;
        }
    }

    uint32_t parity_width[2] = {0, 0};
    for (uint32_t L = 0; L < num_levels; ++L) {
        parity_width[L & 1u] = std::max(parity_width[L & 1u], nt_width[L]);
    }
    return terminals + parity_width[0] + parity_width[1];
}

}  // namespace deepsolver
