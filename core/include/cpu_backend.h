/**
 * @file cpu_backend.h
 * @brief CPU reference implementation of ISolverBackend.
 *
 * Uses recursive CFR traversal with per-combo DCFR regret matching.
 * This is the correctness reference — GPU results are validated against
 * this backend with a tolerance of < 1% max deviation.
 */

#pragma once

#include "solver_backend.h"
#include "types.h"
#include "isomorphism.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace deepsolver {

class CpuBackend final : public ISolverBackend {
public:
    CpuBackend() = default;
    ~CpuBackend() override = default;

    void prepare(const SolverContext& ctx) override;
    void iterate(int iteration) override;
    void finalize() override;
    const std::vector<std::vector<float>>& strategy() const override { return strategy_; }
    const char* name() const override { return "CPU-DCFR"; }

private:
    SolverContext ctx_{};

    // Per-iteration DCFR state (allocated in prepare())
    std::vector<std::vector<float>> regrets_;           // [node][action * nc + combo]
    std::vector<std::vector<float>> strategy_sum_;      // [node][action * nc + combo]
    std::vector<std::vector<float>> current_strategy_;  // [node][action * nc + combo]

    // Final averaged strategy (populated in finalize())
    std::vector<std::vector<float>> strategy_;

    // ---- Internal methods ----
    void compute_strategy();
    void apply_dcfr_discount(int iteration);
    std::vector<float> cfr_traverse(uint32_t node_idx, int traverser, int iteration,
                                     std::vector<float>& reach_oop,
                                     std::vector<float>& reach_ip);
};

// ============================================================================
// prepare: allocate state based on tree size and canonical combo count
// ============================================================================

inline void CpuBackend::prepare(const SolverContext& ctx) {
    ctx_ = ctx;
    uint16_t nc = ctx.iso->num_canonical;
    uint32_t n = ctx.tree->total_nodes;

    regrets_.assign(n, {});
    strategy_sum_.assign(n, {});
    current_strategy_.assign(n, {});
    strategy_.clear();

    for (uint32_t i = 0; i < n; ++i) {
        uint8_t na = ctx.tree->num_children[i];
        auto nt = static_cast<NodeType>(ctx.tree->node_types[i]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        size_t sz = static_cast<size_t>(na) * nc;
        regrets_[i].assign(sz, 0.0f);
        strategy_sum_[i].assign(sz, 0.0f);
        float uniform = (na > 0) ? 1.0f / na : 0.0f;
        current_strategy_[i].assign(sz, uniform);
    }
}

// ============================================================================
// compute_strategy: regret matching → current strategy
// ============================================================================

inline void CpuBackend::compute_strategy() {
    uint16_t nc = ctx_.iso->num_canonical;
    const auto& resolved_locks = *ctx_.resolved_locks;

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;

        uint8_t na = ctx_.tree->num_children[n];
        if (na == 0) continue;

        for (uint16_t c = 0; c < nc; ++c) {
            auto lock_it = resolved_locks.find({n, c});
            if (lock_it != resolved_locks.end()) {
                const auto& forced = lock_it->second;
                for (uint8_t a = 0; a < na && a < forced.size(); ++a) {
                    current_strategy_[n][a * nc + c] = forced[a];
                }
                continue;
            }

            float pos_sum = 0.0f;
            for (uint8_t a = 0; a < na; ++a) {
                float r = regrets_[n][a * nc + c];
                if (r > 0) pos_sum += r;
            }

            if (pos_sum > 0) {
                for (uint8_t a = 0; a < na; ++a) {
                    float r = regrets_[n][a * nc + c];
                    current_strategy_[n][a * nc + c] = (r > 0) ? r / pos_sum : 0.0f;
                }
            } else {
                float uniform = 1.0f / na;
                for (uint8_t a = 0; a < na; ++a) {
                    current_strategy_[n][a * nc + c] = uniform;
                }
            }
        }
    }
}

// ============================================================================
// DCFR discount applied to existing regrets before new regrets are added
// ============================================================================

inline void CpuBackend::apply_dcfr_discount(int iteration) {
    if (iteration <= 0) return;
    uint16_t nc = ctx_.iso->num_canonical;
    float t = static_cast<float>(iteration);
    float pos_disc = std::pow(t / (t + 1.0f), ctx_.config->dcfr_alpha);
    float neg_disc = std::pow(t / (t + 1.0f), ctx_.config->dcfr_beta);

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        uint8_t na = ctx_.tree->num_children[n];
        for (uint8_t a = 0; a < na; ++a) {
            for (uint16_t c = 0; c < nc; ++c) {
                float& r = regrets_[n][a * nc + c];
                r *= (r > 0) ? pos_disc : neg_disc;
            }
        }
    }
}

// ============================================================================
// Recursive CFR traversal. Correctly handles:
//  - Terminal (showdown + fold) per-combo counterfactual values
//  - Chance: passes through
//  - Traverser's own decision nodes: strategy-weighted aggregation + regret update
//  - Opponent's decision nodes: just SUM children (opp strat already in reach)
// Also updates strategy_sum using PER-NODE reach (proper DCFR weighting).
// ============================================================================

inline std::vector<float> CpuBackend::cfr_traverse(
    uint32_t node_idx, int traverser, int iteration,
    std::vector<float>& reach_oop, std::vector<float>& reach_ip)
{
    uint16_t nc = ctx_.iso->num_canonical;
    const auto& tree = *ctx_.tree;
    auto nt = static_cast<NodeType>(tree.node_types[node_idx]);

    // ---- Terminal ----
    if (nt == NodeType::TERMINAL) {
        std::vector<float> values(nc, 0.0f);
        auto tt = static_cast<TerminalType>(tree.terminal_types[node_idx]);
        float half_pot = tree.pots[node_idx] / 2.0f;
        // Pick the matchup table that corresponds to this terminal's full
        // board (root flop + any runout cards dealt by chance ancestors).
        // Falls back to root matchup if the per-runout tables haven't been
        // computed (e.g. legacy code paths or river-only solves).
        int32_t mi = (node_idx < tree.matchup_idx.size()) ? tree.matchup_idx[node_idx] : 0;
        const auto& matchup_ev_table =
            (ctx_.matchup_ev_per_runout && mi >= 0 &&
             static_cast<size_t>(mi) < ctx_.matchup_ev_per_runout->size())
                ? (*ctx_.matchup_ev_per_runout)[mi]
                : *ctx_.matchup_ev;
        const auto& matchup_valid_table =
            (ctx_.matchup_valid_per_runout && mi >= 0 &&
             static_cast<size_t>(mi) < ctx_.matchup_valid_per_runout->size())
                ? (*ctx_.matchup_valid_per_runout)[mi]
                : *ctx_.matchup_valid;
        const auto& matchup_ev    = matchup_ev_table;
        const auto& matchup_valid = matchup_valid_table;
        const auto& canonical_weights = ctx_.iso->canonical_weights;

        if (tt == TerminalType::SHOWDOWN) {
            for (uint16_t c = 0; c < nc; ++c) {
                float val = 0.0f;
                if (traverser == 0) {
                    for (uint16_t cj = 0; cj < nc; ++cj) {
                        size_t idx = static_cast<size_t>(c) * nc + cj;
                        val += reach_ip[cj] * matchup_ev[idx] * matchup_valid[idx]
                               * static_cast<float>(canonical_weights[cj]) * half_pot;
                    }
                } else {
                    for (uint16_t ci = 0; ci < nc; ++ci) {
                        size_t idx = static_cast<size_t>(ci) * nc + c;
                        val += reach_oop[ci] * (-matchup_ev[idx]) * matchup_valid[idx]
                               * static_cast<float>(canonical_weights[ci]) * half_pot;
                    }
                }
                values[c] = val;
            }
        } else {
            // CRITICAL FIX: at a fold terminal the pot is *asymmetric* — the
            // bettor put in more than the folder. The winner only gains what
            // the loser actually committed, NOT half of the (inflated) pot.
            //
            // Old (buggy): gain = pot / 2.
            //   For an OOP shove of 890 into a 220 pot followed by IP fold,
            //   pot[fold] = 1110, half_pot = 555. CFR thought OOP gained 555
            //   from the fold, when the real gain is only 110 (IP's prior
            //   contribution to the matched part of the pot). That ~5x
            //   inflation made all-in dominate every spot in the regret
            //   matching update.
            //
            // The matched portion of the pot equals (pot at fold node) minus
            // (the unmatched bet that triggered the fold). The unmatched bet
            // is recorded as `bet_into` on the *parent* of the fold node
            // (the facing-bet decision node).
            uint32_t parent = tree.parent_indices[node_idx];
            float unmatched_bet = (parent < tree.total_nodes)
                ? tree.bet_into[parent] : 0.0f;
            float matched_pot = tree.pots[node_idx] - unmatched_bet;
            float gain = matched_pot * 0.5f;

            float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
            float sign = (traverser == 0) ? sign_oop : -sign_oop;
            for (uint16_t c = 0; c < nc; ++c) {
                float opp_total = 0.0f;
                if (traverser == 0) {
                    for (uint16_t cj = 0; cj < nc; ++cj) {
                        size_t idx = static_cast<size_t>(c) * nc + cj;
                        opp_total += reach_ip[cj] * matchup_valid[idx]
                                   * static_cast<float>(canonical_weights[cj]);
                    }
                } else {
                    for (uint16_t ci = 0; ci < nc; ++ci) {
                        size_t idx = static_cast<size_t>(ci) * nc + c;
                        opp_total += reach_oop[ci] * matchup_valid[idx]
                                   * static_cast<float>(canonical_weights[ci]);
                    }
                }
                values[c] = sign * gain * opp_total;
            }
        }
        return values;
    }

    // ---- Chance (enumerate runouts) ----
    // Iterate every child (one per dealt card under Phase 1, or per iso
    // representative under Phase 2). Sum child values weighted by runout
    // weight, then divide by total weight to get the expected value over a
    // uniformly-drawn deal. This is the fix for Spot-C-class spots where
    // polar OOP must bet because draws have over-runout equity.
    //
    // Subtlety on reach masking: when a card is dealt that matches one of
    // a player's hole cards (e.g. AhKh hole and Ah turn), that combo is
    // dead in the subtree. matchup_valid_per_runout already encodes the
    // dead-card constraint at the terminal, so reach can stay unchanged
    // for the recursive call — invalid pairs contribute 0 to terminal val.
    if (nt == NodeType::CHANCE) {
        uint8_t nch = tree.num_children[node_idx];
        if (nch == 0) return std::vector<float>(nc, 0.0f);

        std::vector<float> avg(nc, 0.0f);
        uint32_t total_weight = 0;
        uint32_t off = tree.children_offset[node_idx];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree.children[off + k];
            // weight==0 should never occur (TreeNode default is 1); treat as
            // 1 rather than dropping the child, which would bias the avg.
            uint32_t weight = (child < tree.runout_weight.size())
                                ? tree.runout_weight[child] : 1;
            if (weight == 0) weight = 1;
            std::vector<float> child_vals = cfr_traverse(
                child, traverser, iteration, reach_oop, reach_ip);
            for (uint16_t c = 0; c < nc; ++c) {
                avg[c] += static_cast<float>(weight) * child_vals[c];
            }
            total_weight += weight;
        }
        if (total_weight > 0) {
            float inv = 1.0f / static_cast<float>(total_weight);
            for (uint16_t c = 0; c < nc; ++c) avg[c] *= inv;
        }
        return avg;
    }

    // ---- Player decision node ----
    int acting = tree.active_player[node_idx];
    uint8_t na = tree.num_children[node_idx];
    if (na == 0) return std::vector<float>(nc, 0.0f);
    auto& strat = current_strategy_[node_idx];
    auto& acting_reach = (acting == 0) ? reach_oop : reach_ip;

    // strategy_sum update at traverser nodes (per-node reach weighting)
    if (acting == traverser) {
        float sw = std::pow((iteration + 1.0f) / (iteration + 2.0f),
                             ctx_.config->dcfr_gamma);
        for (uint8_t a = 0; a < na; ++a) {
            for (uint16_t c = 0; c < nc; ++c) {
                strategy_sum_[node_idx][a * nc + c] +=
                    sw * acting_reach[c] * strat[a * nc + c];
            }
        }
    }

    // Recurse into each action
    std::vector<std::vector<float>> action_vals(na);
    for (uint8_t a = 0; a < na; ++a) {
        uint32_t child = tree.children[tree.children_offset[node_idx] + a];
        std::vector<float> saved(nc);
        for (uint16_t c = 0; c < nc; ++c) {
            saved[c] = acting_reach[c];
            acting_reach[c] *= strat[a * nc + c];
        }
        action_vals[a] = cfr_traverse(child, traverser, iteration, reach_oop, reach_ip);
        for (uint16_t c = 0; c < nc; ++c) acting_reach[c] = saved[c];
    }

    std::vector<float> node_vals(nc, 0.0f);
    if (acting == traverser) {
        for (uint8_t a = 0; a < na; ++a) {
            for (uint16_t c = 0; c < nc; ++c) {
                node_vals[c] += strat[a * nc + c] * action_vals[a][c];
            }
        }
        for (uint8_t a = 0; a < na; ++a) {
            for (uint16_t c = 0; c < nc; ++c) {
                float instant = action_vals[a][c] - node_vals[c];
                regrets_[node_idx][a * nc + c] += instant;
            }
        }
    } else {
        // Opponent node: strat already in reach, just sum
        for (uint8_t a = 0; a < na; ++a) {
            for (uint16_t c = 0; c < nc; ++c) {
                node_vals[c] += action_vals[a][c];
            }
        }
    }
    return node_vals;
}

// ============================================================================
// iterate: one DCFR iteration (two traversals, one per player)
// ============================================================================

inline void CpuBackend::iterate(int iteration) {
    compute_strategy();
    apply_dcfr_discount(iteration);

    // Traversal for OOP — accumulates OOP regrets and OOP strategy_sum
    {
        auto roop = *ctx_.oop_reach;
        auto rip  = *ctx_.ip_reach;
        cfr_traverse(0, 0, iteration, roop, rip);
    }
    // Traversal for IP
    {
        auto roop = *ctx_.oop_reach;
        auto rip  = *ctx_.ip_reach;
        cfr_traverse(0, 1, iteration, roop, rip);
    }
}

// ============================================================================
// finalize: strategy_sum → normalized strategy
// ============================================================================

inline void CpuBackend::finalize() {
    uint16_t nc = ctx_.iso->num_canonical;
    uint32_t n = ctx_.tree->total_nodes;
    strategy_.assign(n, {});

    for (uint32_t i = 0; i < n; ++i) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[i]);
        uint8_t na = ctx_.tree->num_children[i];

        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            strategy_[i].assign(static_cast<size_t>(na) * nc, 0.0f);
            for (uint16_t c = 0; c < nc; ++c) {
                float total = 0.0f;
                for (uint8_t a = 0; a < na; ++a) {
                    total += strategy_sum_[i][a * nc + c];
                }
                if (total > 1e-7f) {
                    for (uint8_t a = 0; a < na; ++a) {
                        strategy_[i][a * nc + c] = strategy_sum_[i][a * nc + c] / total;
                    }
                } else {
                    float uniform = 1.0f / na;
                    for (uint8_t a = 0; a < na; ++a) {
                        strategy_[i][a * nc + c] = uniform;
                    }
                }
            }
        } else {
            strategy_[i].assign(static_cast<size_t>(na) * nc, 0.0f);
        }
    }
}

} // namespace deepsolver
