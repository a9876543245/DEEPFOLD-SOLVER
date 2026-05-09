/**
 * @file cpu_backend.h
 * @brief CPU reference implementation of ISolverBackend.
 *
 * Uses recursive CFR traversal with per-combo DCFR regret matching.
 * This is the correctness reference — GPU results are validated against
 * this backend with a tolerance of < 1% max deviation.
 *
 * v1.4.0 performance pass:
 *   - Hot inner loops use AVX2/FMA via cpu_simd.h (8-wide float).
 *   - cfr_traverse() runs against a per-traverser ScratchArena instead of
 *     allocating fresh std::vector<float> per recursion. Removes the
 *     malloc/free dominated profile that previously made nc≈1300 traversals
 *     spend more time in the allocator than in CFR math.
 *   - The two traversers (OOP + IP) execute in parallel via OpenMP. They
 *     write to disjoint slices of regrets_/strategy_sum_ (each writes only
 *     at nodes where acting==traverser), so no synchronization is needed.
 *   - Showdown/fold terminals for the IP traverser were loop-swapped so the
 *     inner loop reads matchup matrices contiguously instead of striding.
 *
 * Combined effect on a typical flop spot is ~10-20× wall-clock per iter on
 * a Skylake-class laptop, taking the engine from "needs 15h on CPU" into
 * the same minute-scale ballpark as commercial solvers.
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

// ============================================================================
// ScratchArena — bump allocator used by cfr_traverse to avoid std::vector
// allocations on the hot recursion path.
//
// Why this matters: with nc≈1300 and a tree of ~10000 nodes traversed twice
// per iter, the previous code did ~60k std::vector<float>(nc) allocations
// per iter (~300MB of malloc/free). That was the bottleneck on CPU, often
// exceeding the actual CFR math by 2–3×.
//
// Usage protocol:
//   - reserve()'d once at iterate() start with enough capacity for the
//     deepest recursion path, so alloc() never resizes (which would
//     invalidate outstanding pointers).
//   - Each recursion frame does its allocs, then arena.rewind(mark) to
//     pop the frame.
// ============================================================================
struct ScratchArena {
    std::vector<float> data;
    std::size_t top = 0;

    void reset_for_iter() { top = 0; }

    void reserve(std::size_t n) {
        if (data.size() < n) data.resize(n);
    }

    float* alloc(std::size_t n) {
        // PRECONDITION: caller has reserved enough capacity. We do not resize
        // on overflow because that would invalidate every outstanding pointer
        // held by parent recursion frames.
        if (top + n > data.size()) {
            // Grow generously — we're between traversal frames so this only
            // ever happens during the first call when our initial estimate
            // was too low. Subsequent iterations stay in the resized buffer.
            data.resize((top + n) * 2 + 4096);
        }
        float* p = data.data() + top;
        top += n;
        return p;
    }

    std::size_t mark() const { return top; }
    void rewind(std::size_t m) { top = m; }
};

class CpuBackend final : public ISolverBackend {
public:
    CpuBackend() = default;
    ~CpuBackend() override = default;

    void prepare(const SolverContext& ctx) override;
    void iterate(int iteration) override;
    void finalize() override;
    const std::vector<std::vector<float>>& strategy() const override { return strategy_; }
    const char* name() const override {
        // Runtime label — the dispatch table chose AVX2 vs scalar at startup
        // based on CPUID. See cpu_kernels_dispatch.cpp.
        return (cpu_simd::active_mode() == cpu_simd::SimdMode::Avx2)
            ? "CPU-DCFR-AVX2"
            : "CPU-DCFR-scalar";
    }
    /// Reference backend's only intra-iter parallelism is the OOP||IP
    /// `parallel sections` block, which fans out to at most 2 OMP threads
    /// regardless of `omp_get_max_threads()`. So the effective worker count
    /// here is 1 (when user explicitly asked for serial) or 2 otherwise.
    uint32_t cpu_threads_effective() const override {
        const uint32_t requested =
            (ctx_.config != nullptr) ? ctx_.config->cpu_threads : 0u;
        return (requested == 1u) ? 1u : 2u;
    }

private:
    SolverContext ctx_{};

    // Per-iteration DCFR state (allocated in prepare())
    std::vector<std::vector<float>> regrets_;           // [node][action * nc + combo]
    std::vector<std::vector<float>> strategy_sum_;      // [node][action * nc + combo]
    std::vector<std::vector<float>> current_strategy_;  // [node][action * nc + combo]

    // Final averaged strategy (populated in finalize())
    std::vector<std::vector<float>> strategy_;

    // canonical_weights as float (precomputed once in prepare()) so SIMD
    // doesn't have to convert from uint16_t every time.
    std::vector<float> canonical_weights_f_;

    // Per-traverser arenas: lets OOP and IP traversals run on separate
    // threads without contending on a shared bump pointer. Capacity is
    // sized in prepare() based on tree shape.
    ScratchArena arena_oop_;
    ScratchArena arena_ip_;

    // Per-traverser reach scratch (mutated during traversal, restored at
    // each step). Allocated once per prepare() to avoid per-iter copies.
    std::vector<float> reach_oop_thread0_;
    std::vector<float> reach_ip_thread0_;
    std::vector<float> reach_oop_thread1_;
    std::vector<float> reach_ip_thread1_;

    // Root output buffer for each traversal (we don't need the value but
    // cfr_traverse must have somewhere to write).
    std::vector<float> root_out_thread0_;
    std::vector<float> root_out_thread1_;

    // Reusable per-node scratch for compute_strategy regret-matching pass.
    std::vector<float> pos_sum_;
    std::vector<float> inv_pos_sum_;
    std::vector<float> uniform_or_zero_;

    // v1.8.1+ out-of-range skip masks (same semantics as in
    // LevelizedCpuBackend — see that file for the correctness note).
    std::vector<uint8_t> oop_out_of_range_mask_;
    std::vector<uint8_t> ip_out_of_range_mask_;

    // ---- Internal methods ----
    void compute_strategy();
    void apply_dcfr_discount(int iteration);
    void cfr_traverse(uint32_t node_idx, int traverser, int iteration,
                      float* reach_oop, float* reach_ip, float* out_vals,
                      ScratchArena& arena);
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

    uint8_t max_actions = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t na = ctx.tree->num_children[i];
        auto nt = static_cast<NodeType>(ctx.tree->node_types[i]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        if (na > max_actions) max_actions = na;
        std::size_t sz = static_cast<std::size_t>(na) * nc;
        regrets_[i].assign(sz, 0.0f);
        strategy_sum_[i].assign(sz, 0.0f);
        float uniform = (na > 0) ? 1.0f / na : 0.0f;
        current_strategy_[i].assign(sz, uniform);
    }

    // Precompute canonical_weights as float for SIMD reach * weight ops.
    canonical_weights_f_.resize(nc);
    for (uint16_t k = 0; k < nc; ++k) {
        canonical_weights_f_[k] = static_cast<float>(ctx.iso->canonical_weights[k]);
    }

    // Reach scratch buffers: one OOP + one IP for each potential traverser
    // thread. Even when not running OMP we use the per-traverser pair to
    // avoid implicit allocations in iterate().
    reach_oop_thread0_.assign(nc, 0.0f);
    reach_ip_thread0_.assign(nc, 0.0f);
    reach_oop_thread1_.assign(nc, 0.0f);
    reach_ip_thread1_.assign(nc, 0.0f);
    root_out_thread0_.assign(nc, 0.0f);
    root_out_thread1_.assign(nc, 0.0f);

    pos_sum_.assign(nc, 0.0f);
    inv_pos_sum_.assign(nc, 0.0f);
    uniform_or_zero_.assign(nc, 0.0f);

    // v1.8.1+ out-of-range skip masks built from root reach.
    oop_out_of_range_mask_.assign(nc, 0);
    ip_out_of_range_mask_.assign(nc, 0);
    if (ctx.oop_reach != nullptr) {
        for (uint16_t c = 0; c < nc; ++c) {
            if ((*ctx.oop_reach)[c] == 0.0f) oop_out_of_range_mask_[c] = 1;
        }
    }
    if (ctx.ip_reach != nullptr) {
        for (uint16_t c = 0; c < nc; ++c) {
            if ((*ctx.ip_reach)[c] == 0.0f) ip_out_of_range_mask_[c] = 1;
        }
    }

    // Reserve scratch arena capacity. Worst case per recursion frame at a
    // player decision node is (max_actions * nc) for action_vals + (nc) for
    // saved reach. Tree depth is bounded by ~30 in practice; double for
    // safety, plus padding for terminal/chance scratch.
    const std::size_t depth_max = 32;
    const std::size_t per_frame =
        static_cast<std::size_t>(max_actions + 2) * nc + 4 * nc;
    const std::size_t reserve_floats = depth_max * per_frame + 16 * nc;
    arena_oop_.reserve(reserve_floats);
    arena_ip_.reserve(reserve_floats);
}

// ============================================================================
// compute_strategy: regret matching → current strategy
//
// Two-pass formulation to enable SIMD over the canonical-combo dimension:
//   Pass 1: pos_sum[c] = Σ_a max(regrets[a*nc+c], 0)
//   Pass 2: strat[a*nc+c] = max(regrets[a*nc+c], 0) / pos_sum[c]
//                         (or 1/na if pos_sum[c] == 0)
// Pass 2 is rewritten branchless as one FMA via precomputed inv_pos_sum and
// uniform_or_zero scratch — see cpu_simd::vec_pos_normalize.
// ============================================================================

inline void CpuBackend::compute_strategy() {
    uint16_t nc = ctx_.iso->num_canonical;
    const auto& resolved_locks = *ctx_.resolved_locks;

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;

        uint8_t na = ctx_.tree->num_children[n];
        if (na == 0) continue;
        const float uniform = 1.0f / na;
        const auto& regret_vec = regrets_[n];
        auto&       strat_vec  = current_strategy_[n];

        // Pass 1: pos_sum[c] = Σ_a max(regret[a*nc+c], 0)
        cpu_simd::vec_set_zero(pos_sum_.data(), nc);
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_pos_add(
                pos_sum_.data(),
                regret_vec.data() + static_cast<std::size_t>(a) * nc,
                nc);
        }

        // Build inv_pos_sum_ and uniform_or_zero_ scratch (per-c branch lifted).
        for (uint16_t c = 0; c < nc; ++c) {
            if (pos_sum_[c] > 0.0f) {
                inv_pos_sum_[c] = 1.0f / pos_sum_[c];
                uniform_or_zero_[c] = 0.0f;
            } else {
                inv_pos_sum_[c] = 0.0f;
                uniform_or_zero_[c] = uniform;
            }
        }

        // Pass 2: strat[a*nc+c] = max(regret, 0) * inv_pos_sum + uniform_or_zero
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_pos_normalize(
                strat_vec.data() + static_cast<std::size_t>(a) * nc,
                regret_vec.data() + static_cast<std::size_t>(a) * nc,
                inv_pos_sum_.data(),
                uniform_or_zero_.data(),
                nc);
        }

        // Apply locks (rare path, scalar overwrite is fine).
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

// ============================================================================
// DCFR discount: r *= (r > 0) ? pos_disc : neg_disc, applied to all regrets
// ============================================================================

inline void CpuBackend::apply_dcfr_discount(int iteration) {
    float pos_disc, neg_disc, strat_weight;
    compute_dcfr_factors(iteration, *ctx_.config, pos_disc, neg_disc, strat_weight);
    (void)strat_weight;  // applied separately in cfr_traverse
    uint16_t nc = ctx_.iso->num_canonical;

    for (uint32_t n = 0; n < ctx_.tree->total_nodes; ++n) {
        auto nt = static_cast<NodeType>(ctx_.tree->node_types[n]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        uint8_t na = ctx_.tree->num_children[n];
        std::size_t total = static_cast<std::size_t>(na) * nc;
        cpu_simd::vec_dcfr_discount(regrets_[n].data(), pos_disc, neg_disc, total);
    }
}

// ============================================================================
// Recursive CFR traversal. Writes nc floats to out_vals; uses arena for any
// transient buffers (action_vals, saved reach, child_vals, weighted reach).
// ============================================================================

inline void CpuBackend::cfr_traverse(
    uint32_t node_idx, int traverser, int iteration,
    float* reach_oop, float* reach_ip, float* out_vals,
    ScratchArena& arena)
{
    uint16_t nc = ctx_.iso->num_canonical;
    const auto& tree = *ctx_.tree;
    auto nt = static_cast<NodeType>(tree.node_types[node_idx]);

    // ---------------- Terminal ----------------
    if (nt == NodeType::TERMINAL) {
        auto tt = static_cast<TerminalType>(tree.terminal_types[node_idx]);
        float pot_total = tree.pots[node_idx];
        float half_pot = pot_total * 0.5f;
        float rake = std::min(pot_total * ctx_.config->rake_rate,
                              ctx_.config->rake_cap);
        if (rake < 0.0f) rake = 0.0f;
        float win_payoff  = half_pot - rake;
        float lose_payoff = -half_pot;
        float tie_payoff  = -0.5f * rake;

        int32_t mi = (node_idx < tree.matchup_idx.size()) ? tree.matchup_idx[node_idx] : 0;
        const auto& matchup_valid_table =
            (ctx_.matchup_valid_per_runout && mi >= 0 &&
             static_cast<std::size_t>(mi) < ctx_.matchup_valid_per_runout->size())
                ? (*ctx_.matchup_valid_per_runout)[mi]
                : *ctx_.matchup_valid;
        // v1.8.2 A2 encoding — see cpu_backend_levelized.h for the why.
        const auto& matchup_category_table =
            (ctx_.matchup_category_per_runout && mi >= 0 &&
             static_cast<std::size_t>(mi) < ctx_.matchup_category_per_runout->size())
                ? (*ctx_.matchup_category_per_runout)[mi]
                : *ctx_.matchup_category;
        const auto* matchup_valid    = matchup_valid_table.data();
        const auto* matchup_category = matchup_category_table.data();

        // Precompute opp_reach * canonical_weights once per terminal so
        // the inner SIMD loop only does reach_w * valid * payoff.
        std::size_t mark = arena.mark();
        float* opp_reach_w = arena.alloc(nc);
        const float* opp_reach = (traverser == 0) ? reach_ip : reach_oop;
        for (uint16_t k = 0; k < nc; ++k) {
            opp_reach_w[k] = opp_reach[k] * canonical_weights_f_[k];
        }

        // v1.8.1+ out-of-range skip mask — see cpu_backend_levelized.h for
        // the correctness note.
        const uint8_t* skip_mask = (traverser == 0)
            ? oop_out_of_range_mask_.data()
            : ip_out_of_range_mask_.data();

        if (tt == TerminalType::SHOWDOWN) {
            // v1.8.0 P3-8 spike: full-matrix kernels — see cpu_simd.h for
            // why. Same numeric output as the per-c loop (parity test
            // gates this); just hoists SIMD-constant setup out of the c
            // loop so we don't rebuild vwin/vlose/vtie nc times per call.
            if (traverser == 0) {
                cpu_simd::showdown_oop_full(
                    matchup_category, matchup_valid, opp_reach_w, skip_mask,
                    out_vals, nc, win_payoff, lose_payoff, tie_payoff);
            } else {
                cpu_simd::showdown_ip_full(
                    matchup_category, matchup_valid, opp_reach_w, out_vals, nc,
                    win_payoff, lose_payoff, tie_payoff);
            }
        } else {
            // Fold terminal — asymmetric pot. Winner only gets the matched
            // portion of what loser actually committed; rake comes off the
            // winner's gain. See git history for the v1.0.x bugfix.
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
                // For OOP: out_vals[c] = self_payoff * Σ_cj (reach_w[cj] * valid[c, cj])
                // Inner cj is contiguous — single SIMD dot product per c.
                // v1.8.1+ out-of-range skip.
                for (uint16_t c = 0; c < nc; ++c) {
                    if (skip_mask[c]) {
                        out_vals[c] = 0.0f;
                        continue;
                    }
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(c) * nc;
                    float opp_total =
                        cpu_simd::dot_valid_reach(valid_row, opp_reach_w, nc);
                    out_vals[c] = self_payoff * opp_total;
                }
            } else {
                // For IP: loop-swap to ci-outer so the inner valid_row scan is
                // contiguous. Final scale by self_payoff after the loop.
                cpu_simd::vec_set_zero(out_vals, nc);
                for (uint16_t ci = 0; ci < nc; ++ci) {
                    float rw_ci = opp_reach_w[ci];
                    if (rw_ci == 0.0f) continue;
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(ci) * nc;
                    cpu_simd::fold_ip_step(out_vals, valid_row, rw_ci, nc);
                }
                cpu_simd::vec_scale_in_place(out_vals, self_payoff, nc);
            }
        }

        arena.rewind(mark);
        return;
    }

    // ---------------- Chance ----------------
    if (nt == NodeType::CHANCE) {
        uint8_t nch = tree.num_children[node_idx];
        if (nch == 0) {
            cpu_simd::vec_set_zero(out_vals, nc);
            return;
        }
        cpu_simd::vec_set_zero(out_vals, nc);
        std::size_t mark = arena.mark();
        float* child_vals = arena.alloc(nc);
        uint32_t total_weight = 0;
        uint32_t off = tree.children_offset[node_idx];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree.children[off + k];
            uint32_t weight = (child < tree.runout_weight.size())
                                ? tree.runout_weight[child] : 1;
            if (weight == 0) weight = 1;
            std::size_t inner_mark = arena.mark();
            cfr_traverse(child, traverser, iteration,
                         reach_oop, reach_ip, child_vals, arena);
            arena.rewind(inner_mark);
            cpu_simd::vec_axpy(out_vals, static_cast<float>(weight), child_vals, nc);
            total_weight += weight;
        }
        if (total_weight > 0) {
            float inv = 1.0f / static_cast<float>(total_weight);
            cpu_simd::vec_scale_in_place(out_vals, inv, nc);
        }
        arena.rewind(mark);
        return;
    }

    // ---------------- Player decision ----------------
    int acting = tree.active_player[node_idx];
    uint8_t na = tree.num_children[node_idx];
    if (na == 0) {
        cpu_simd::vec_set_zero(out_vals, nc);
        return;
    }
    auto& strat = current_strategy_[node_idx];
    float* acting_reach = (acting == 0) ? reach_oop : reach_ip;

    // strategy_sum update at traverser nodes (must happen BEFORE recursion
    // mutates acting_reach).
    if (acting == traverser) {
        float pos_unused, neg_unused, sw;
        compute_dcfr_factors(iteration, *ctx_.config, pos_unused, neg_unused, sw);

        if (ctx_.config->dcfr_schedule ==
            SolverConfig::DcfrSchedule::POSTFLOP_STYLE) {
            // s = s * sw + strat
            for (uint8_t a = 0; a < na; ++a) {
                cpu_simd::vec_decay_add(
                    strategy_sum_[node_idx].data() + static_cast<std::size_t>(a) * nc,
                    sw,
                    strat.data() + static_cast<std::size_t>(a) * nc,
                    nc);
            }
        } else {
            // s += sw * acting_reach * strat
            for (uint8_t a = 0; a < na; ++a) {
                cpu_simd::vec_reach_weighted_strat_sum(
                    strategy_sum_[node_idx].data() + static_cast<std::size_t>(a) * nc,
                    sw,
                    acting_reach,
                    strat.data() + static_cast<std::size_t>(a) * nc,
                    nc);
            }
        }
    }

    // Allocate scratch for action_vals (na rows of nc) + saved reach.
    std::size_t mark = arena.mark();
    float* action_vals = arena.alloc(static_cast<std::size_t>(na) * nc);
    float* saved       = arena.alloc(nc);

    for (uint8_t a = 0; a < na; ++a) {
        uint32_t child = tree.children[tree.children_offset[node_idx] + a];
        // saved = acting_reach;  acting_reach *= strat[a]
        cpu_simd::vec_copy(saved, acting_reach, nc);
        cpu_simd::vec_mul_in_place(
            acting_reach,
            strat.data() + static_cast<std::size_t>(a) * nc,
            nc);
        std::size_t inner_mark = arena.mark();
        cfr_traverse(child, traverser, iteration,
                     reach_oop, reach_ip,
                     action_vals + static_cast<std::size_t>(a) * nc,
                     arena);
        arena.rewind(inner_mark);
        cpu_simd::vec_copy(acting_reach, saved, nc);
    }

    if (acting == traverser) {
        // node_vals = Σ_a strat[a] * action_vals[a]
        cpu_simd::vec_set_zero(out_vals, nc);
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_fmadd(
                out_vals,
                strat.data() + static_cast<std::size_t>(a) * nc,
                action_vals + static_cast<std::size_t>(a) * nc,
                nc);
        }
        // regret[a, c] += action_val[a, c] - node_val[c]
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_regret_update(
                regrets_[node_idx].data() + static_cast<std::size_t>(a) * nc,
                action_vals + static_cast<std::size_t>(a) * nc,
                out_vals,
                nc);
        }
    } else {
        // Opponent node: opp's strat already in reach, just sum.
        cpu_simd::vec_set_zero(out_vals, nc);
        for (uint8_t a = 0; a < na; ++a) {
            cpu_simd::vec_add_in_place(
                out_vals,
                action_vals + static_cast<std::size_t>(a) * nc,
                nc);
        }
    }

    arena.rewind(mark);
}

// ============================================================================
// iterate: one DCFR iteration (compute strat, discount, two traversals).
//
// The two traversals are independent (they write to disjoint nodes — OOP
// traversal updates only OOP-acting nodes' regrets, and vice versa) so we
// run them on separate threads via OpenMP. Each thread has its own arena
// and reach scratch buffers, so there's no synchronization cost.
// ============================================================================

inline void CpuBackend::iterate(int iteration) {
    compute_strategy();
    apply_dcfr_discount(iteration);

    // Reset arenas for this iter (capacity reserved in prepare()).
    arena_oop_.reset_for_iter();
    arena_ip_.reset_for_iter();

    uint16_t nc = ctx_.iso->num_canonical;
    const auto& base_oop_reach = *ctx_.oop_reach;
    const auto& base_ip_reach  = *ctx_.ip_reach;

    // Snapshot reach buffers per traversal (mutated during recursion).
    std::memcpy(reach_oop_thread0_.data(), base_oop_reach.data(), sizeof(float) * nc);
    std::memcpy(reach_ip_thread0_.data(),  base_ip_reach.data(),  sizeof(float) * nc);
    std::memcpy(reach_oop_thread1_.data(), base_oop_reach.data(), sizeof(float) * nc);
    std::memcpy(reach_ip_thread1_.data(),  base_ip_reach.data(),  sizeof(float) * nc);

    // v1.4.0 Phase 2: cpu_threads honors --cpu-threads.
    //   0 (auto) or >=2 → run OOP and IP traversers in parallel via OMP
    //   1            → serial (used for benchmarking / parity testing)
    const uint32_t cfg_threads = ctx_.config->cpu_threads;
    const bool run_parallel = (cfg_threads != 1);

#if defined(_OPENMP)
    if (run_parallel) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                cfr_traverse(0, /*traverser=*/0, iteration,
                             reach_oop_thread0_.data(), reach_ip_thread0_.data(),
                             root_out_thread0_.data(), arena_oop_);
            }
            #pragma omp section
            {
                cfr_traverse(0, /*traverser=*/1, iteration,
                             reach_oop_thread1_.data(), reach_ip_thread1_.data(),
                             root_out_thread1_.data(), arena_ip_);
            }
        }
    } else {
        cfr_traverse(0, 0, iteration,
                     reach_oop_thread0_.data(), reach_ip_thread0_.data(),
                     root_out_thread0_.data(), arena_oop_);
        cfr_traverse(0, 1, iteration,
                     reach_oop_thread1_.data(), reach_ip_thread1_.data(),
                     root_out_thread1_.data(), arena_ip_);
    }
#else
    (void)run_parallel;
    cfr_traverse(0, 0, iteration,
                 reach_oop_thread0_.data(), reach_ip_thread0_.data(),
                 root_out_thread0_.data(), arena_oop_);
    cfr_traverse(0, 1, iteration,
                 reach_oop_thread1_.data(), reach_ip_thread1_.data(),
                 root_out_thread1_.data(), arena_ip_);
#endif
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

} // namespace deepsolver
