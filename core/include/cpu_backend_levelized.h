/**
 * @file cpu_backend_levelized.h
 * @brief Multi-threaded CPU CFR backend using a level-based traversal.
 *
 * Phase 4 of the CPU optimization plan. The recursive `CpuBackend` only
 * scales to 2 threads (OOP || IP via OMP parallel sections). To get past
 * that we need the same shape of work the GPU kernels already use:
 *
 *   1. Compute each node's depth (distance to nearest leaf).
 *   2. Bucket nodes into level arrays, level 0 = terminals.
 *   3. Process levels in order:
 *        forward pass, root → leaves: propagate reach + update strategy_sum
 *        backward pass, leaves → root: compute value + update regrets
 *   4. Within each level, nodes are independent → `#pragma omp parallel for`.
 *
 * The recursion-and-arena pattern in CpuBackend can't do this because the
 * arena's bump pointer is shared serial state across the recursion frames.
 * Levelization replaces it with per-node arrays of fixed offset.
 *
 * Memory cost:
 *   reach_oop / reach_ip / value_traverser:  N × nc × 4 B each
 *   For N ≈ 10000, nc ≈ 1300 → ≈ 50 MB per array, ≈ 200 MB total. The
 *   reference backend's regrets / strategy_sum / current_strategy already
 *   take 3× that, so the marginal cost is small relative to the existing
 *   per-iter state.
 *
 * Correctness:
 *   - Reach is purely a function of root reach × strat path, so the
 *     forward pass is shared across both traversers (run once per iter).
 *   - Value depends on traverser identity (terminal payoff sign), so the
 *     backward pass runs twice — once per traverser. Each traverser
 *     writes to disjoint slices of regrets_ (its own acting==traverser
 *     player nodes), so the two backward passes can also overlap if
 *     desired (see `iterate()`).
 *
 * The reference `CpuBackend` is kept as the parity oracle. Selectable via
 * `--cpu-backend reference|levelized`; `levelized` is the new default
 * once parity tests pass on rainbow / monotone / turn / river / locked /
 * rake spots.
 */

#pragma once

#include "solver_backend.h"
#include "types.h"
#include "isomorphism.h"
#include "cpu_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace deepsolver {

class LevelizedCpuBackend final : public ISolverBackend {
public:
    LevelizedCpuBackend() = default;
    ~LevelizedCpuBackend() override = default;

    void prepare(const SolverContext& ctx) override;
    void iterate(int iteration) override;
    void finalize() override;
    const std::vector<std::vector<float>>& strategy() const override { return strategy_; }
    const char* name() const override {
        return (cpu_simd::active_mode() == cpu_simd::SimdMode::Avx2)
            ? "CPU-Levelized-AVX2"
            : "CPU-Levelized-scalar";
    }
    uint32_t cpu_threads_effective() const override { return cpu_threads_effective_; }

private:
    SolverContext ctx_{};

    // Resolved in prepare() from config.cpu_threads:
    //   0 → auto: omp_get_max_threads() (env-controlled, typically all cores)
    //   1 → serial: one OMP team thread, no parallel-for fan-out
    //   N → clamp to [1, hardware_concurrency()]
    // Used as `num_threads(cpu_threads_effective_)` on every parallel-for in
    // forward_pass / backward_pass. Avoids global omp_set_num_threads() so we
    // don't leak a thread cap into postsolve / estimate-only / GPU paths
    // sharing this process.
    uint32_t cpu_threads_effective_ = 0;

    // Per-iteration DCFR state — same shape as the reference backend so
    // node locks / dcfr_schedule tests carry over unchanged.
    std::vector<std::vector<float>> regrets_;
    std::vector<std::vector<float>> strategy_sum_;
    std::vector<std::vector<float>> current_strategy_;

    // Final averaged strategy (populated in finalize()).
    std::vector<std::vector<float>> strategy_;

    // Cached float copy of canonical_weights for SIMD use.
    std::vector<float> canonical_weights_f_;

    // ---- Levelization ----
    // node_order_[level_offsets_[L] .. level_offsets_[L+1]) lists every
    // node at depth L. depth = distance to nearest leaf, so level 0 holds
    // terminals; level max_depth holds the root.
    std::vector<uint32_t> node_order_;
    std::vector<uint32_t> level_offsets_;
    uint32_t max_depth_  = 0;
    uint32_t num_levels_ = 0;

    // ---- Per-node persistent buffers ----
    // Flat [N × nc] in row-major (node, combo). Allocated once in
    // prepare() and reused across iters. Reach is shared across both
    // traverser passes; value is per-traverser (computed in each
    // backward pass).
    std::vector<float> reach_oop_;
    std::vector<float> reach_ip_;
    std::vector<float> value_;

    // Per-node scratch for compute_strategy() (reused across nodes).
    std::vector<float> pos_sum_scratch_;
    std::vector<float> inv_pos_sum_scratch_;
    std::vector<float> uniform_or_zero_scratch_;

    // ---- Internal methods ----
    void compute_strategy();
    void apply_dcfr_discount(int iteration);
    void build_level_schedule();

    // One forward pass: propagate reach top-down, update strategy_sum.
    // Reach propagation is independent of traverser, so this runs ONCE
    // per iter rather than per-traverser.
    void forward_pass(int iteration);

    // One backward pass for the given traverser. Computes value_[node]
    // bottom-up, updates regrets_[node] at acting == traverser nodes.
    void backward_pass(int traverser);

    // Per-node terminal payoff helper. Writes nc floats to `out`.
    void evaluate_terminal(uint32_t node_idx, int traverser, float* out);

    // Returns base offset of node n's nc-wide row in a flat [N × nc] array.
    inline std::size_t row_off(uint32_t n) const {
        return static_cast<std::size_t>(n) * ctx_.iso->num_canonical;
    }
};

// ============================================================================
// Levelization (depth = distance to nearest leaf, bottom-up by level)
// ============================================================================

inline void LevelizedCpuBackend::build_level_schedule() {
    const uint32_t N = ctx_.tree->total_nodes;
    if (N == 0) {
        max_depth_ = 0;
        num_levels_ = 0;
        node_order_.clear();
        level_offsets_.clear();
        return;
    }

    // Single reverse-pass works because GameTreeBuilder emits children
    // with index > parent (BFS-flattened tree).
    std::vector<uint32_t> depth(N, 0);
    for (int64_t i = static_cast<int64_t>(N) - 1; i >= 0; --i) {
        uint8_t na = ctx_.tree->num_children[i];
        if (na == 0) {
            depth[i] = 0;
            continue;
        }
        uint32_t max_child_d = 0;
        for (uint8_t a = 0; a < na; ++a) {
            uint32_t child = ctx_.tree->children[ctx_.tree->children_offset[i] + a];
            if (child < N && depth[child] + 1 > max_child_d) {
                max_child_d = depth[child] + 1;
            }
        }
        depth[i] = max_child_d;
    }

    max_depth_  = 0;
    for (uint32_t d : depth) if (d > max_depth_) max_depth_ = d;
    num_levels_ = max_depth_ + 1;

    std::vector<uint32_t> count(num_levels_, 0);
    for (uint32_t d : depth) count[d]++;

    level_offsets_.assign(num_levels_ + 1, 0);
    for (uint32_t L = 0; L < num_levels_; ++L) {
        level_offsets_[L + 1] = level_offsets_[L] + count[L];
    }

    node_order_.assign(N, 0);
    std::vector<uint32_t> cursor(level_offsets_);
    for (uint32_t n = 0; n < N; ++n) {
        node_order_[cursor[depth[n]]++] = n;
    }
}

// ============================================================================
// prepare: allocate state, build level schedule
// ============================================================================

inline void LevelizedCpuBackend::prepare(const SolverContext& ctx) {
    ctx_ = ctx;
    const uint16_t nc = ctx.iso->num_canonical;
    const uint32_t N = ctx.tree->total_nodes;

    // Resolve effective thread count via the shared helper so estimate_only()
    // and the live backend agree on what `--cpu-threads N` means.
    {
        const uint32_t requested = (ctx.config != nullptr) ? ctx.config->cpu_threads : 0u;
        uint32_t hw = 1u;
#ifdef _OPENMP
        hw = static_cast<uint32_t>(omp_get_max_threads());
#endif
        cpu_threads_effective_ = resolve_cpu_threads(requested, hw);
    }

    regrets_.assign(N, {});
    strategy_sum_.assign(N, {});
    current_strategy_.assign(N, {});
    strategy_.clear();

    for (uint32_t i = 0; i < N; ++i) {
        uint8_t na = ctx.tree->num_children[i];
        auto nt = static_cast<NodeType>(ctx.tree->node_types[i]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        std::size_t sz = static_cast<std::size_t>(na) * nc;
        regrets_[i].assign(sz, 0.0f);
        strategy_sum_[i].assign(sz, 0.0f);
        float uniform = (na > 0) ? 1.0f / na : 0.0f;
        current_strategy_[i].assign(sz, uniform);
    }

    canonical_weights_f_.resize(nc);
    for (uint16_t k = 0; k < nc; ++k) {
        canonical_weights_f_[k] = static_cast<float>(ctx.iso->canonical_weights[k]);
    }

    // Per-node flat buffers — N × nc floats each.
    const std::size_t flat_sz = static_cast<std::size_t>(N) * nc;
    reach_oop_.assign(flat_sz, 0.0f);
    reach_ip_.assign(flat_sz, 0.0f);
    value_.assign(flat_sz, 0.0f);

    pos_sum_scratch_.assign(nc, 0.0f);
    inv_pos_sum_scratch_.assign(nc, 0.0f);
    uniform_or_zero_scratch_.assign(nc, 0.0f);

    build_level_schedule();
}

// ============================================================================
// compute_strategy / apply_dcfr_discount — same as CpuBackend, just
// re-implemented here so the two backends are independent.
// ============================================================================

inline void LevelizedCpuBackend::compute_strategy() {
    const uint16_t nc = ctx_.iso->num_canonical;
    const auto& resolved_locks = *ctx_.resolved_locks;

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;

        uint8_t na = ctx_.tree->num_children[n];
        if (na == 0) continue;
        const float uniform = 1.0f / na;
        const auto& regret_vec = regrets_[n];
        auto&       strat_vec  = current_strategy_[n];

        cpu_simd::vec_set_zero(pos_sum_scratch_.data(), nc);
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_pos_add(
                pos_sum_scratch_.data(),
                regret_vec.data() + static_cast<std::size_t>(a) * nc,
                nc);
        }

        for (uint16_t c = 0; c < nc; ++c) {
            if (pos_sum_scratch_[c] > 0.0f) {
                inv_pos_sum_scratch_[c] = 1.0f / pos_sum_scratch_[c];
                uniform_or_zero_scratch_[c] = 0.0f;
            } else {
                inv_pos_sum_scratch_[c] = 0.0f;
                uniform_or_zero_scratch_[c] = uniform;
            }
        }

        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_pos_normalize(
                strat_vec.data() + static_cast<std::size_t>(a) * nc,
                regret_vec.data() + static_cast<std::size_t>(a) * nc,
                inv_pos_sum_scratch_.data(),
                uniform_or_zero_scratch_.data(),
                nc);
        }

        if (!resolved_locks.empty()) {
            for (uint16_t c = 0; c < nc; ++c) {
                auto lock_it = resolved_locks.find({n, c});
                if (lock_it == resolved_locks.end()) continue;
                const auto& forced = lock_it->second;
                for (uint8_t a = 0; a < na && a < forced.size(); ++a) {
                    strat_vec[static_cast<std::size_t>(a) * nc + c] = forced[a];
                }
            }
        }
    }
}

inline void LevelizedCpuBackend::apply_dcfr_discount(int iteration) {
    float pos_disc, neg_disc, strat_weight;
    compute_dcfr_factors(iteration, *ctx_.config, pos_disc, neg_disc, strat_weight);
    (void)strat_weight;
    const uint16_t nc = ctx_.iso->num_canonical;

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        uint8_t na = ctx_.tree->num_children[n];
        std::size_t total = static_cast<std::size_t>(na) * nc;
        cpu_simd::vec_dcfr_discount(regrets_[n].data(), pos_disc, neg_disc, total);
    }
}

// ============================================================================
// Terminal payoff (same formulas as CpuBackend, written for the levelized
// indexing. `out` is a pointer to nc floats.)
// ============================================================================

inline void LevelizedCpuBackend::evaluate_terminal(
    uint32_t node_idx, int traverser, float* out)
{
    const auto& tree = *ctx_.tree;
    const uint16_t nc = ctx_.iso->num_canonical;
    auto tt = static_cast<TerminalType>(tree.terminal_types[node_idx]);

    float pot_total = tree.pots[node_idx];
    float half_pot  = pot_total * 0.5f;
    float rake = std::min(pot_total * ctx_.config->rake_rate,
                          ctx_.config->rake_cap);
    if (rake < 0.0f) rake = 0.0f;
    float win_payoff  = half_pot - rake;
    float lose_payoff = -half_pot;
    float tie_payoff  = -0.5f * rake;

    int32_t mi = (node_idx < tree.matchup_idx.size()) ? tree.matchup_idx[node_idx] : 0;
    const auto& matchup_ev_table =
        (ctx_.matchup_ev_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_ev_per_runout->size())
            ? (*ctx_.matchup_ev_per_runout)[mi]
            : *ctx_.matchup_ev;
    const auto& matchup_valid_table =
        (ctx_.matchup_valid_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_valid_per_runout->size())
            ? (*ctx_.matchup_valid_per_runout)[mi]
            : *ctx_.matchup_valid;
    const float* matchup_ev    = matchup_ev_table.data();
    const float* matchup_valid = matchup_valid_table.data();

    // opp_reach × canonical_weight, computed once.
    std::vector<float> opp_reach_w(nc);
    const float* opp_reach = (traverser == 0)
        ? &reach_ip_[row_off(node_idx)]
        : &reach_oop_[row_off(node_idx)];
    for (uint16_t k = 0; k < nc; ++k) {
        opp_reach_w[k] = opp_reach[k] * canonical_weights_f_[k];
    }

    if (tt == TerminalType::SHOWDOWN) {
        if (traverser == 0) {
            for (uint16_t c = 0; c < nc; ++c) {
                const float* ev_row    = matchup_ev    + static_cast<std::size_t>(c) * nc;
                const float* valid_row = matchup_valid + static_cast<std::size_t>(c) * nc;
                out[c] = cpu_simd::showdown_oop_inner(
                    ev_row, valid_row, opp_reach_w.data(),
                    win_payoff, lose_payoff, tie_payoff, nc);
            }
        } else {
            cpu_simd::vec_set_zero(out, nc);
            for (uint16_t ci = 0; ci < nc; ++ci) {
                float rw_ci = opp_reach_w[ci];
                if (rw_ci == 0.0f) continue;
                const float* ev_row    = matchup_ev    + static_cast<std::size_t>(ci) * nc;
                const float* valid_row = matchup_valid + static_cast<std::size_t>(ci) * nc;
                cpu_simd::showdown_ip_step(
                    out, ev_row, valid_row, rw_ci,
                    win_payoff, lose_payoff, tie_payoff, nc);
            }
        }
    } else {
        // Fold terminal — asymmetric pot fix from CpuBackend.
        uint32_t parent = tree.parent_indices[node_idx];
        float unmatched_bet = (parent < tree.total_nodes)
            ? tree.bet_into[parent] : 0.0f;
        float matched_pot = pot_total - unmatched_bet;
        float fold_win_gain  = matched_pot * 0.5f - rake;
        float fold_lose_loss = -matched_pot * 0.5f;

        float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
        float self_payoff;
        if ((traverser == 0 && sign_oop > 0) || (traverser == 1 && sign_oop < 0)) {
            self_payoff = fold_win_gain;
        } else {
            self_payoff = fold_lose_loss;
        }

        if (traverser == 0) {
            for (uint16_t c = 0; c < nc; ++c) {
                const float* valid_row =
                    matchup_valid + static_cast<std::size_t>(c) * nc;
                float opp_total =
                    cpu_simd::dot_valid_reach(valid_row, opp_reach_w.data(), nc);
                out[c] = self_payoff * opp_total;
            }
        } else {
            cpu_simd::vec_set_zero(out, nc);
            for (uint16_t ci = 0; ci < nc; ++ci) {
                float rw_ci = opp_reach_w[ci];
                if (rw_ci == 0.0f) continue;
                const float* valid_row =
                    matchup_valid + static_cast<std::size_t>(ci) * nc;
                cpu_simd::fold_ip_step(out, valid_row, rw_ci, nc);
            }
            cpu_simd::vec_scale_in_place(out, self_payoff, nc);
        }
    }
}

// ============================================================================
// Forward pass — propagate reach top-down, update strategy_sum at
// acting-player nodes. Reach is independent of traverser identity, so we
// run this ONCE per iter even though we'll then run two backward passes.
//
// Strategy_sum is updated for both players as we descend: at every player
// node, the acting player's reach is what gets used (by the standard
// schedule's reach-weighted sum) before being multiplied by strat[a] for
// the children.
// ============================================================================

inline void LevelizedCpuBackend::forward_pass(int iteration) {
    const auto& tree = *ctx_.tree;
    const uint16_t nc = ctx_.iso->num_canonical;
    const uint32_t N = tree.total_nodes;
    if (N == 0) return;

    // Initialize root reach (node 0 == root by GameTreeBuilder convention).
    std::memcpy(reach_oop_.data(), ctx_.oop_reach->data(), sizeof(float) * nc);
    std::memcpy(reach_ip_.data(),  ctx_.ip_reach->data(),  sizeof(float) * nc);

    float pos_unused, neg_unused, sw;
    compute_dcfr_factors(iteration, *ctx_.config, pos_unused, neg_unused, sw);
    const bool decay_and_add = (ctx_.config->dcfr_schedule ==
                                SolverConfig::DcfrSchedule::POSTFLOP_STYLE);

    // Walk levels from root down to the level just above terminals (level 1).
    // Level 0 contains terminals — no reach propagation from there.
    if (num_levels_ <= 1) return;
    for (int64_t L = static_cast<int64_t>(num_levels_) - 1; L >= 1; --L) {
        const uint32_t lo = level_offsets_[L];
        const uint32_t hi = level_offsets_[L + 1];

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 8) num_threads(static_cast<int>(cpu_threads_effective_))
        #endif
        for (int64_t idx = static_cast<int64_t>(lo);
             idx < static_cast<int64_t>(hi); ++idx)
        {
            uint32_t n = node_order_[static_cast<std::size_t>(idx)];
            auto nt = static_cast<NodeType>(tree.node_types[n]);
            if (nt == NodeType::TERMINAL) continue;

            const float* parent_oop = &reach_oop_[row_off(n)];
            const float* parent_ip  = &reach_ip_[row_off(n)];

            if (nt == NodeType::CHANCE) {
                // Reach passes through chance nodes unchanged. The expected
                // value over chance is handled at the BACKWARD pass via
                // weighted child averaging.
                uint8_t nch = tree.num_children[n];
                uint32_t off = tree.children_offset[n];
                for (uint8_t k = 0; k < nch; ++k) {
                    uint32_t child = tree.children[off + k];
                    std::memcpy(&reach_oop_[row_off(child)], parent_oop, sizeof(float) * nc);
                    std::memcpy(&reach_ip_[row_off(child)],  parent_ip,  sizeof(float) * nc);
                }
                continue;
            }

            // PLAYER_OOP / PLAYER_IP
            int acting = tree.active_player[n];
            uint8_t na = tree.num_children[n];
            if (na == 0) continue;
            const auto& strat = current_strategy_[n];

            // Strategy_sum update at this node — uses reach of the ACTING
            // player. Reference backend does this only at acting==traverser
            // nodes, but since we run a backward pass per traverser and
            // strategy_sum entries are disjoint per acting player, doing it
            // once during the shared forward pass is equivalent and saves a
            // pass.
            const float* acting_reach = (acting == 0) ? parent_oop : parent_ip;
            if (decay_and_add) {
                for (uint8_t a = 0; a < na; ++a) {
                    cpu_simd::vec_decay_add(
                        strategy_sum_[n].data() + static_cast<std::size_t>(a) * nc,
                        sw,
                        strat.data() + static_cast<std::size_t>(a) * nc,
                        nc);
                }
            } else {
                for (uint8_t a = 0; a < na; ++a) {
                    cpu_simd::vec_reach_weighted_strat_sum(
                        strategy_sum_[n].data() + static_cast<std::size_t>(a) * nc,
                        sw,
                        acting_reach,
                        strat.data() + static_cast<std::size_t>(a) * nc,
                        nc);
                }
            }

            // Reach propagation to children.
            //   Child's acting-player-reach = parent's acting reach × strat[a]
            //   Child's other reach         = parent's other reach (unchanged)
            uint32_t off = tree.children_offset[n];
            for (uint8_t a = 0; a < na; ++a) {
                uint32_t child = tree.children[off + a];
                float* child_oop = &reach_oop_[row_off(child)];
                float* child_ip  = &reach_ip_[row_off(child)];
                if (acting == 0) {
                    // Acting OOP: scale OOP by strat[a], copy IP straight.
                    for (uint16_t c = 0; c < nc; ++c) {
                        child_oop[c] = parent_oop[c] *
                            strat[static_cast<std::size_t>(a) * nc + c];
                    }
                    std::memcpy(child_ip, parent_ip, sizeof(float) * nc);
                } else {
                    // Acting IP.
                    for (uint16_t c = 0; c < nc; ++c) {
                        child_ip[c] = parent_ip[c] *
                            strat[static_cast<std::size_t>(a) * nc + c];
                    }
                    std::memcpy(child_oop, parent_oop, sizeof(float) * nc);
                }
            }
        }
    }
}

// ============================================================================
// Backward pass for one traverser. Walks levels 0 → max_depth.
// ============================================================================

inline void LevelizedCpuBackend::backward_pass(int traverser) {
    const auto& tree = *ctx_.tree;
    const uint16_t nc = ctx_.iso->num_canonical;
    const uint32_t N = tree.total_nodes;
    if (N == 0) return;

    for (uint32_t L = 0; L < num_levels_; ++L) {
        const uint32_t lo = level_offsets_[L];
        const uint32_t hi = level_offsets_[L + 1];

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 8) num_threads(static_cast<int>(cpu_threads_effective_))
        #endif
        for (int64_t idx = static_cast<int64_t>(lo);
             idx < static_cast<int64_t>(hi); ++idx)
        {
            uint32_t n = node_order_[static_cast<std::size_t>(idx)];
            auto nt = static_cast<NodeType>(tree.node_types[n]);
            float* out = &value_[row_off(n)];

            if (nt == NodeType::TERMINAL) {
                evaluate_terminal(n, traverser, out);
                continue;
            }

            if (nt == NodeType::CHANCE) {
                uint8_t nch = tree.num_children[n];
                if (nch == 0) {
                    cpu_simd::vec_set_zero(out, nc);
                    continue;
                }
                cpu_simd::vec_set_zero(out, nc);
                uint32_t total_weight = 0;
                uint32_t off = tree.children_offset[n];
                for (uint8_t k = 0; k < nch; ++k) {
                    uint32_t child = tree.children[off + k];
                    uint32_t weight = (child < tree.runout_weight.size())
                                        ? tree.runout_weight[child] : 1;
                    if (weight == 0) weight = 1;
                    cpu_simd::vec_axpy(
                        out, static_cast<float>(weight),
                        &value_[row_off(child)], nc);
                    total_weight += weight;
                }
                if (total_weight > 0) {
                    cpu_simd::vec_scale_in_place(
                        out, 1.0f / static_cast<float>(total_weight), nc);
                }
                continue;
            }

            // Player decision node.
            int acting = tree.active_player[n];
            uint8_t na = tree.num_children[n];
            if (na == 0) {
                cpu_simd::vec_set_zero(out, nc);
                continue;
            }
            const auto& strat = current_strategy_[n];
            uint32_t off = tree.children_offset[n];

            if (acting == traverser) {
                // node_val = Σ_a strat[a] · child_val[a]
                cpu_simd::vec_set_zero(out, nc);
                for (uint8_t a = 0; a < na; ++a) {
                    uint32_t child = tree.children[off + a];
                    cpu_simd::vec_fmadd(
                        out,
                        strat.data() + static_cast<std::size_t>(a) * nc,
                        &value_[row_off(child)],
                        nc);
                }
                // regret[a, c] += child_val[a, c] − node_val[c]
                for (uint8_t a = 0; a < na; ++a) {
                    uint32_t child = tree.children[off + a];
                    cpu_simd::vec_regret_update(
                        regrets_[n].data() + static_cast<std::size_t>(a) * nc,
                        &value_[row_off(child)],
                        out,
                        nc);
                }
            } else {
                // Opp node: opp's strat is already baked into reach by the
                // forward pass, so just sum the children's values.
                cpu_simd::vec_set_zero(out, nc);
                for (uint8_t a = 0; a < na; ++a) {
                    uint32_t child = tree.children[off + a];
                    cpu_simd::vec_add_in_place(
                        out, &value_[row_off(child)], nc);
                }
            }
        }
    }
}

// ============================================================================
// iterate: one DCFR iteration.
// ============================================================================

inline void LevelizedCpuBackend::iterate(int iteration) {
    compute_strategy();
    apply_dcfr_discount(iteration);

    // Single forward pass updates reach + strategy_sum for both traversers.
    forward_pass(iteration);

    // Backward passes — one per traverser. They write to disjoint slices
    // of regrets_ (acting==traverser nodes are different sets), so they
    // could in principle run on separate threads, but each backward pass
    // is already heavily multi-threaded inside via the per-level OMP. We
    // keep them serial to avoid oversubscribing the thread pool.
    backward_pass(0);
    backward_pass(1);
}

// ============================================================================
// finalize — same as CpuBackend; normalize strategy_sum into strategy.
// ============================================================================

inline void LevelizedCpuBackend::finalize() {
    const uint16_t nc = ctx_.iso->num_canonical;
    const uint32_t N = ctx_.tree->total_nodes;
    strategy_.assign(N, {});

    for (uint32_t i = 0; i < N; ++i) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[i]);
        uint8_t na = ctx_.tree->num_children[i];

        if ((nt == NodeType::PLAYER_OOP || nt == NodeType::PLAYER_IP) && na > 0) {
            strategy_[i].assign(static_cast<std::size_t>(na) * nc, 0.0f);
            for (uint16_t c = 0; c < nc; ++c) {
                float total = 0.0f;
                for (uint8_t a = 0; a < na; ++a) {
                    total += strategy_sum_[i][static_cast<std::size_t>(a) * nc + c];
                }
                if (total > 1e-7f) {
                    float inv = 1.0f / total;
                    for (uint8_t a = 0; a < na; ++a) {
                        strategy_[i][static_cast<std::size_t>(a) * nc + c] =
                            strategy_sum_[i][static_cast<std::size_t>(a) * nc + c] * inv;
                    }
                } else {
                    float uniform = 1.0f / na;
                    for (uint8_t a = 0; a < na; ++a) {
                        strategy_[i][static_cast<std::size_t>(a) * nc + c] = uniform;
                    }
                }
            }
        } else {
            strategy_[i].assign(static_cast<std::size_t>(na) * nc, 0.0f);
        }
    }
}

}  // namespace deepsolver
