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
 *        forward pass, root ??leaves: propagate reach + update strategy_sum
 *        backward pass, leaves ??root: compute value + update regrets
 *   4. Within each level, nodes are independent ??`#pragma omp parallel for`.
 *
 * The recursion-and-arena pattern in CpuBackend can't do this because the
 * arena's bump pointer is shared serial state across the recursion frames.
 * Levelization replaces it with per-node arrays of fixed offset.
 *
 * Memory cost:
 *   reach_oop / reach_ip / value_traverser:  N ? nc ? 4 B each
 *   For N ??10000, nc ??1300 ????50 MB per array, ??200 MB total. The
 *   reference backend's regrets / strategy_sum / current_strategy already
 *   take 3? that, so the marginal cost is small relative to the existing
 *   per-iter state.
 *
 * Correctness:
 *   - Reach is purely a function of root reach ? strat path, so the
 *     forward pass is shared across both traversers (run once per iter).
 *   - Value depends on traverser identity (terminal payoff sign), so the
 *     backward pass runs twice ??once per traverser. Each traverser
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
#include "fold_blocker.h"
#include "showdown_rank_blocker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
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

    // v1.8.1+ phase-timing accessors. Each call returns the cumulative
    // wall-clock time (in ms) spent in that iteration phase across the
    // whole solve. Solver pulls these into SolverTiming after the iter
    // loop completes. Always-on instrumentation; the per-phase
    // omp_get_wtime() calls add ~5 ? ~50ns per iter on Windows ??well
    // below 0.1% of typical iter time.
    double phase_compute_strategy_ms()  const { return phase_compute_strategy_ms_;  }
    double phase_apply_discount_ms()    const { return phase_apply_discount_ms_;    }
    double phase_forward_pass_ms()      const { return phase_forward_pass_ms_;      }
    double phase_backward_pass_oop_ms() const { return phase_backward_pass_oop_ms_; }
    double phase_backward_pass_ip_ms()  const { return phase_backward_pass_ip_ms_;  }
    double phase_backward_showdown_ms() const {
        double s = 0.0;
        for (double v : showdown_acc_per_thread_) s += v;
        return s * 1000.0;
    }
    double phase_backward_fold_ms() const {
        double s = 0.0;
        for (double v : fold_acc_per_thread_) s += v;
        return s * 1000.0;
    }

    CpuBackendDiagnostics cpu_diagnostics() const override {
        CpuBackendDiagnostics d;
        const uint32_t nc = (ctx_.iso != nullptr) ? ctx_.iso->num_canonical : 0u;
        d.available = (nc != 0u);
        d.canonical_combos = nc;
        d.oop_active_count =
            static_cast<uint32_t>(oop_terminal_active_indices_.size());
        d.ip_active_count =
            static_cast<uint32_t>(ip_terminal_active_indices_.size());
        d.oop_active_run_count =
            static_cast<uint32_t>(oop_terminal_active_runs_.size());
        d.ip_active_run_count =
            static_cast<uint32_t>(ip_terminal_active_runs_.size());
        d.oop_active_block_count =
            static_cast<uint32_t>(oop_terminal_active_blocks_.size());
        d.ip_active_block_count =
            static_cast<uint32_t>(ip_terminal_active_blocks_.size());
        auto span_sum = [](const std::vector<cpu_simd::ActiveRun>& blocks) {
            uint32_t total = 0u;
            for (const auto& block : blocks) total += block.count;
            return total;
        };
        d.oop_active_block_span = span_sum(oop_terminal_active_blocks_);
        d.ip_active_block_span = span_sum(ip_terminal_active_blocks_);
        if (nc != 0u) {
            d.oop_active_density =
                static_cast<float>(d.oop_active_count) / static_cast<float>(nc);
            d.ip_active_density =
                static_cast<float>(d.ip_active_count) / static_cast<float>(nc);
        }
        auto avg_run_len = [](const std::vector<cpu_simd::ActiveRun>& runs) {
            uint32_t total = 0u;
            for (const auto& run : runs) total += run.count;
            return runs.empty()
                ? 0.0f
                : static_cast<float>(total) / static_cast<float>(runs.size());
        };
        d.oop_avg_run_length = avg_run_len(oop_terminal_active_runs_);
        d.ip_avg_run_length = avg_run_len(ip_terminal_active_runs_);
        d.oop_terminal_active_list = oop_use_terminal_active_list_;
        d.ip_terminal_active_list = ip_use_terminal_active_list_;
        d.oop_active_runs = oop_use_active_runs_;
        d.ip_active_runs = ip_use_active_runs_;
        d.oop_active_blocks = oop_use_active_blocks_;
        d.ip_active_blocks = ip_use_active_blocks_;
        d.oop_sparse_traversal = oop_use_sparse_traversal_;
        d.ip_sparse_traversal = ip_use_sparse_traversal_;
        d.oop_block_strategy = use_block_strategy_for_player(0);
        d.ip_block_strategy = use_block_strategy_for_player(1);
        d.oop_block_strategy_sum = use_block_strategy_sum_for_player(0);
        d.ip_block_strategy_sum = use_block_strategy_sum_for_player(1);
        d.oop_block_traversal = use_block_traversal_for_player(0);
        d.ip_block_traversal = use_block_traversal_for_player(1);
        d.oop_terminal_active2 = oop_use_terminal_active2_;
        d.ip_terminal_active2 = ip_use_terminal_active2_;
        d.ip_terminal_output_skip = ip_use_terminal_output_skip_;
        d.oop_sparse_opp_reach_build =
            use_sparse_opp_reach_build_for_traverser(0);
        d.ip_sparse_opp_reach_build =
            use_sparse_opp_reach_build_for_traverser(1);
        d.fold_blocker_shortcut = kFoldBlockerShortcutEnabled;
        d.fold_blocker_precomputed = fold_precomputed_enabled_;
        d.showdown_rank_blocker_shortcut =
            use_rank_blocker_showdown_for_traverser(0)
            || use_rank_blocker_showdown_for_traverser(1)
            || use_active_rank_blocker_showdown_for_traverser(0)
            || use_active_rank_blocker_showdown_for_traverser(1);
        d.showdown_signed_coeff_shortcut =
            use_signed_coeff_showdown_for_traverser(0)
            || use_signed_coeff_showdown_for_traverser(1);
        d.sparse_terminal_no_full_clear_enabled =
            kSparseTerminalNoFullClearEnabled;
        populate_matchup_category_diagnostics(
            d, ctx_.matchup_category_per_runout,
            ctx_.matchup_category, nc);
        return d;
    }

private:
    SolverContext ctx_{};

    // Resolved in prepare() from config.cpu_threads:
    //   0 ??auto: omp_get_max_threads() (env-controlled, typically all cores)
    //   1 ??serial: one OMP team thread, no parallel-for fan-out
    //   N ??clamp to [1, hardware_concurrency()]
    // Used as `num_threads(cpu_threads_effective_)` on every parallel-for in
    // forward_pass / backward_pass. Avoids global omp_set_num_threads() so we
    // don't leak a thread cap into postsolve / estimate-only / GPU paths
    // sharing this process.
    uint32_t cpu_threads_effective_ = 0;

    // Per-iteration DCFR state for player decision nodes.
    // Flat SoA layout:
    //   field[node_state_offset_[node] + action * action_stride_ + combo]
    // action_stride_ is num_canonical padded to an AVX2 lane boundary.
    static constexpr std::size_t kNoDecisionState =
        static_cast<std::size_t>(-1);
    static constexpr std::size_t kActionLane = 8;

    std::vector<float> regrets_;
    std::vector<float> strategy_sum_;
    std::vector<float> current_strategy_;
    std::vector<std::size_t> node_state_offset_;
    std::size_t action_stride_ = 0;
    std::size_t row_stride_ = 0;

    // Final averaged strategy (populated in finalize()).
    std::vector<std::vector<float>> strategy_;

    // Cached float copies of canonical weights and inverse weights for SIMD use.
    std::vector<float> canonical_weights_f_;
    std::vector<float> canonical_inv_weights_f_;

    // ---- Levelization ----
    // node_order_[level_offsets_[L] .. level_offsets_[L+1]) lists every
    // node at depth L. depth = distance to nearest leaf, so level 0 holds
    // terminals; level max_depth holds the root.
    std::vector<uint32_t> node_order_;
    std::vector<uint32_t> level_offsets_;
    uint32_t max_depth_  = 0;
    uint32_t num_levels_ = 0;

    // Player decision nodes with at least one child. Built once in prepare()
    // so per-iteration phases do not rescan terminals/chance nodes.
    std::vector<uint32_t> player_nodes_;

    // POST_OPTIMIZATION_REVIEW Sec 4.3 Phase 2: level-0 terminals grouped
    // by matchup_idx. Matrix benchmark on monotone shows ~245 terminals share
    // each of the 446 matchup tables. Built once at prepare() from level-0
    // nodes. Used by backward_pass to feed showdown_oop_full_batch with all
    // SHOWDOWN terminals from one table per call ??the matrix rows stream
    // from cache once for the whole group instead of once per terminal call.
    std::vector<std::vector<uint32_t>> terminals_by_table_;

    // SHOWDOWN-only subset of terminals_by_table_ (split out so the batch
    // kernel doesn't have to filter inside its hot path). FOLD terminals at
    // level 0 keep the per-call path because the FOLD kernel's structure
    // (nested OOP/IP loops with rw_ci skips) doesn't benefit from batching
    // and refactoring it just for FOLD risks regressing the dominant
    // SHOWDOWN path's win.
    std::vector<std::vector<uint32_t>> showdowns_by_table_;
    std::vector<uint32_t> non_showdown_l0_;     // FOLD + any other types

    // Per-thread scratch for the batch kernel call. Grouped here so we
    // don't re-allocate per group (or per iter). Sized to the largest group
    // ? per-terminal scratch needs in prepare().
    std::vector<float>             batch_opp_reach_w_;     // [max_group ? nc]
    std::vector<float*>            batch_out_ptrs_;        // [max_group]
    std::vector<const float*>      batch_reach_ptrs_;      // [max_group]
    std::vector<float>             batch_win_payoff_;      // [max_group]
    std::vector<float>             batch_lose_payoff_;     // [max_group]
    std::vector<float>             batch_tie_payoff_;      // [max_group]
    std::size_t                    max_showdown_group_size_ = 0;

    // ---- Per-node persistent buffers ----
    // Flat [N ? nc] in row-major (node, combo). Allocated once in
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

    // v1.8.0 P2-5: per-thread scratch for evaluate_terminal()'s `opp_reach ? weight`
    // pre-computation. Layout: [thread_id ? nc, thread_id ? nc + nc).
    // Sized to cpu_threads_effective_ ? nc in prepare() so the OMP parallel-for
    // inside backward_pass() can call evaluate_terminal() without each call
    // heap-allocating a fresh nc-wide vector. On a 549-node tree there are
    // ~270 terminals ? 2 traversers = ~540 mallocs/iter under the old code;
    // this lifts that into a single prepare()-time allocation.
    std::vector<float> terminal_opp_reach_w_;
    std::vector<showdown_rank_blocker::Scratch> showdown_rank_scratch_;
    fold_blocker::Metadata fold_fallback_metadata_;
    std::vector<fold_blocker::Metadata> fold_metadata_per_runout_;
    std::vector<showdown_rank_blocker::Metadata> showdown_rank_metadata_per_runout_;
    inline float* terminal_scratch_for_thread(int tid) {
        const uint32_t cap = cpu_threads_effective_;
        const uint32_t safe_tid =
            (tid < 0 || static_cast<uint32_t>(tid) >= cap) ? 0u
                                                           : static_cast<uint32_t>(tid);
        return terminal_opp_reach_w_.data()
             + static_cast<std::size_t>(safe_tid) * ctx_.iso->num_canonical;
    }

    // v1.8.1+ phase timing accumulators (cumulative across whole solve, ms).
    // Reset to 0 in prepare(); accumulated in iterate(). Solver reads them
    // after the iter loop and stores into SolverTiming.
    double phase_compute_strategy_ms_  = 0.0;
    double phase_apply_discount_ms_    = 0.0;
    double phase_forward_pass_ms_      = 0.0;
    double phase_backward_pass_oop_ms_ = 0.0;
    double phase_backward_pass_ip_ms_  = 0.0;

    // POST_OPTIMIZATION_REVIEW Sec 4.2: showdown vs fold split inside
    // evaluate_terminal(). Per-thread accumulators sized in prepare() so
    // OMP threads add into their own slot without a synchronization point
    // on the inner-most hot path. Held in seconds; phase_backward_*_ms()
    // converts to ms ? thread-sum (= CPU-seconds across all workers).
    std::vector<double> showdown_acc_per_thread_;
    std::vector<double> fold_acc_per_thread_;

    // v1.8.1+ out-of-range skip masks. Built once in prepare() from the
    // root reach vectors. mask[c] = 1 means "this combo is 0-reach from
    // root" ??the user's range excludes it, so out[c] for this combo at
    // any terminal can be set to 0 without affecting CFR convergence
    // (it propagates as 0 everywhere). UNSAFE to derive a per-node
    // "skip when local reach is 0" because that breaks regret updates
    // at ancestors with non-zero reach (a CFR signal we need).
    //
    // 70-75% of canonical combos are out-of-range on standard preflop
    // ranges ??~3? speedup on OOP showdown / fold paths.
    std::vector<uint8_t> oop_out_of_range_mask_;
    std::vector<uint8_t> ip_out_of_range_mask_;
    std::vector<uint16_t> oop_terminal_active_indices_;
    std::vector<uint16_t> ip_terminal_active_indices_;
    std::vector<cpu_simd::ActiveRun> oop_terminal_active_runs_;
    std::vector<cpu_simd::ActiveRun> ip_terminal_active_runs_;
    std::vector<cpu_simd::ActiveRun> oop_terminal_active_blocks_;
    std::vector<cpu_simd::ActiveRun> ip_terminal_active_blocks_;
    bool oop_has_out_of_range_ = false;
    bool ip_has_out_of_range_ = false;
    bool ip_use_terminal_output_skip_ = false;
    bool oop_use_terminal_active_list_ = false;
    bool ip_use_terminal_active_list_ = false;
    bool oop_use_active_runs_ = false;
    bool ip_use_active_runs_ = false;
    bool oop_use_active_blocks_ = false;
    bool ip_use_active_blocks_ = false;
    bool oop_use_terminal_active2_ = false;
    bool ip_use_terminal_active2_ = false;
    bool oop_use_sparse_traversal_ = false;
    bool ip_use_sparse_traversal_ = false;
    bool showdown_rank_blocker_supported_ = false;
    bool showdown_signed_coeff_supported_ = false;
    bool fold_precomputed_enabled_ = false;

    // Active sparse terminals only need to produce values for root-range
    // combos: subsequent sparse traversal reads exactly those lanes, and
    // postsolve EV/exploitability recomputes from finalized strategy rather
    // than reading backend value_ scratch. Skipping the full row clear avoids
    // writing nc floats per terminal on narrow custom ranges.
    static constexpr bool kSparseTerminalNoFullClearEnabled = true;
    // Full block traversal still loses too much to per-block overhead in
    // forward/backward/discount. Keep it off, but let compute_strategy use
    // the lane-aligned blocks independently where the measured win is clean.
    // Strategy_sum is also isolated: it can skip inactive lanes without
    // changing reach propagation or backward traversal.
    static constexpr bool kEnableBlockTraversal = false;
    static constexpr bool kEnableBlockStrategy = true;
    static constexpr bool kEnableBlockStrategySum = true;
    static constexpr bool kFoldBlockerShortcutEnabled = true;
    static constexpr bool kFoldBlockerPrecomputedEnabled = true;
    static constexpr std::size_t kFoldBlockerPrecomputeMinRunouts = 64;
    static constexpr bool kShowdownRankBlockerShortcutEnabled = true;

    inline const std::vector<uint16_t>& active_indices_for_player(int player) const {
        return (player == 0) ? oop_terminal_active_indices_
                             : ip_terminal_active_indices_;
    }
    inline const std::vector<cpu_simd::ActiveRun>& active_runs_for_player(int player) const {
        return (player == 0) ? oop_terminal_active_runs_
                             : ip_terminal_active_runs_;
    }
    inline bool use_sparse_traversal_for_player(int player) const {
        return (player == 0) ? oop_use_sparse_traversal_
                             : ip_use_sparse_traversal_;
    }
    inline bool use_active_runs_for_player(int player) const {
        return (player == 0) ? oop_use_active_runs_
                             : ip_use_active_runs_;
    }
    inline bool use_active_blocks_for_player(int player) const {
        return (player == 0) ? oop_use_active_blocks_
                             : ip_use_active_blocks_;
    }
    inline bool use_block_traversal_for_player(int player) const {
        return kEnableBlockTraversal
            && use_active_blocks_for_player(player)
            && !use_sparse_traversal_for_player(player);
    }
    inline bool use_block_strategy_for_player(int player) const {
        return kEnableBlockStrategy
            && use_active_blocks_for_player(player)
            && !use_sparse_traversal_for_player(player);
    }
    inline bool use_block_strategy_sum_for_player(int player) const {
        return kEnableBlockStrategySum
            && use_active_blocks_for_player(player)
            && !use_sparse_traversal_for_player(player);
    }
    inline const std::vector<cpu_simd::ActiveRun>& active_blocks_for_player(int player) const {
        return (player == 0) ? oop_terminal_active_blocks_
                             : ip_terminal_active_blocks_;
    }
    inline bool use_terminal_active_list_for_player(int player) const {
        return (player == 0) ? oop_use_terminal_active_list_
                             : ip_use_terminal_active_list_;
    }
    inline bool use_sparse_opp_reach_build_for_traverser(int traverser) const {
        const int opp = 1 - traverser;
        return use_sparse_traversal_for_player(opp)
            || use_active_blocks_for_player(opp)
            || (use_terminal_active_list_for_player(opp)
                && use_active_runs_for_player(opp));
    }
    inline bool use_rank_blocker_showdown_for_traverser(int traverser) const {
        return kShowdownRankBlockerShortcutEnabled
            && showdown_rank_blocker_supported_
            && !use_terminal_active_list_for_player(traverser)
            && !use_sparse_traversal_for_player(traverser);
    }
    inline bool use_active_rank_blocker_showdown_for_traverser(int traverser) const {
        return kShowdownRankBlockerShortcutEnabled
            && showdown_rank_blocker_supported_
            && use_terminal_active_list_for_player(traverser)
            && !use_sparse_traversal_for_player(traverser);
    }
    inline bool use_signed_coeff_showdown_for_traverser(int traverser) const {
        return showdown_signed_coeff_supported_
            && !use_terminal_active_list_for_player(traverser)
            && !use_rank_blocker_showdown_for_traverser(traverser);
    }
    static inline void build_active_runs(
        const std::vector<uint16_t>& active,
        std::vector<cpu_simd::ActiveRun>& runs)
    {
        runs.clear();
        runs.reserve(active.size());
        if (active.empty()) return;
        uint16_t start = active.front();
        uint16_t prev = start;
        uint16_t count = 1;
        for (std::size_t k = 1; k < active.size(); ++k) {
            const uint16_t c = active[k];
            if (c == static_cast<uint16_t>(prev + 1)) {
                prev = c;
                ++count;
                continue;
            }
            runs.push_back({start, count});
            start = prev = c;
            count = 1;
        }
        runs.push_back({start, count});
    }
    static inline void build_active_lane_blocks(
        const std::vector<uint16_t>& active,
        std::vector<cpu_simd::ActiveRun>& blocks,
        uint32_t n)
    {
        blocks.clear();
        blocks.reserve(active.size());
        if (active.empty()) return;
        constexpr uint16_t kLane = 8;
        uint16_t block_start =
            static_cast<uint16_t>((active.front() / kLane) * kLane);
        uint16_t block_end =
            static_cast<uint16_t>(std::min<uint32_t>(block_start + kLane, n));
        for (uint16_t c : active) {
            const uint16_t start =
                static_cast<uint16_t>((c / kLane) * kLane);
            const uint16_t end =
                static_cast<uint16_t>(std::min<uint32_t>(start + kLane, n));
            if (start <= block_end) {
                if (end > block_end) block_end = end;
                continue;
            }
            blocks.push_back({
                block_start,
                static_cast<uint16_t>(block_end - block_start)
            });
            block_start = start;
            block_end = end;
        }
        blocks.push_back({
            block_start,
            static_cast<uint16_t>(block_end - block_start)
        });
    }
    static inline uint32_t active_block_span(
        const std::vector<cpu_simd::ActiveRun>& blocks)
    {
        uint32_t total = 0u;
        for (const auto& block : blocks) total += block.count;
        return total;
    }
    static inline bool active_blocks_are_worth_using(
        const std::vector<uint16_t>& active,
        const std::vector<cpu_simd::ActiveRun>& blocks,
        uint32_t n)
    {
        if (active.empty() || blocks.empty()) return false;
        const uint32_t span = active_block_span(blocks);
        return span < n && span * 4u <= n * 3u;
    }
    static inline bool active_runs_are_worth_using(
        const std::vector<uint16_t>& active,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        return !runs.empty() && (runs.size() * 4u <= active.size());
    }
    static inline void vec_set_zero_active(
        float* dst, const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] = 0.0f;
    }
    static inline void vec_set_zero_active_runs(
        float* dst, const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_set_zero(dst + run.start, run.count);
        }
    }
    static inline void vec_copy_active(
        float* dst, const float* src, const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] = src[c];
    }
    static inline void vec_copy_active_runs(
        float* dst, const float* src, const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_copy(dst + run.start, src + run.start, run.count);
        }
    }
    static inline void vec_mul_active(
        float* dst, const float* src, const float* mul,
        const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] = src[c] * mul[c];
    }
    static inline void vec_mul_active_runs(
        float* dst, const float* src, const float* mul,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_copy(dst + run.start, src + run.start, run.count);
            cpu_simd::vec_mul_in_place(dst + run.start, mul + run.start, run.count);
        }
    }
    static inline void vec_add_active(
        float* dst, const float* src, const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] += src[c];
    }
    static inline void vec_add_active_runs(
        float* dst, const float* src, const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_add_in_place(dst + run.start, src + run.start, run.count);
        }
    }
    static inline void vec_axpy_active(
        float* dst, float a, const float* src,
        const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] += a * src[c];
    }
    static inline void vec_axpy_active_runs(
        float* dst, float a, const float* src,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_axpy(dst + run.start, a, src + run.start, run.count);
        }
    }
    static inline void vec_scale_active(
        float* dst, float s, const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] *= s;
    }
    static inline void vec_scale_active_runs(
        float* dst, float s, const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_scale_in_place(dst + run.start, s, run.count);
        }
    }
    static inline void vec_fmadd_active(
        float* dst, const float* a, const float* b,
        const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) dst[c] += a[c] * b[c];
    }
    static inline void vec_fmadd_active_runs(
        float* dst, const float* a, const float* b,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_fmadd(
                dst + run.start, a + run.start, b + run.start, run.count);
        }
    }
    static inline void vec_regret_update_active(
        float* regret, const float* action_val, const float* node_val,
        const std::vector<uint16_t>& active)
    {
        for (uint16_t c : active) regret[c] += action_val[c] - node_val[c];
    }
    static inline void vec_regret_update_active_runs(
        float* regret, const float* action_val, const float* node_val,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_regret_update(
                regret + run.start,
                action_val + run.start,
                node_val + run.start,
                run.count);
        }
    }
    static inline void vec_dcfr_discount_active_runs(
        float* regret, float pos_disc, float neg_disc,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_dcfr_discount(
                regret + run.start, pos_disc, neg_disc, run.count);
        }
    }
    static inline void vec_reach_weighted_strat_sum_active_runs(
        float* dst, float sw, const float* reach, const float* strat,
        const std::vector<cpu_simd::ActiveRun>& runs)
    {
        for (const auto& run : runs) {
            cpu_simd::vec_reach_weighted_strat_sum(
                dst + run.start, sw,
                reach + run.start,
                strat + run.start,
                run.count);
        }
    }
    inline void sparse_set_zero(float* dst, int player) const {
        if (use_active_runs_for_player(player)) {
            vec_set_zero_active_runs(dst, active_runs_for_player(player));
        } else {
            vec_set_zero_active(dst, active_indices_for_player(player));
        }
    }
    inline void sparse_copy(float* dst, const float* src, int player) const {
        if (use_active_runs_for_player(player)) {
            vec_copy_active_runs(dst, src, active_runs_for_player(player));
        } else {
            vec_copy_active(dst, src, active_indices_for_player(player));
        }
    }
    inline void sparse_mul(float* dst, const float* src, const float* mul,
                           int player) const {
        if (use_active_runs_for_player(player)) {
            vec_mul_active_runs(dst, src, mul, active_runs_for_player(player));
        } else {
            vec_mul_active(dst, src, mul, active_indices_for_player(player));
        }
    }
    inline void sparse_add(float* dst, const float* src, int player) const {
        if (use_active_runs_for_player(player)) {
            vec_add_active_runs(dst, src, active_runs_for_player(player));
        } else {
            vec_add_active(dst, src, active_indices_for_player(player));
        }
    }
    inline void sparse_axpy(float* dst, float a, const float* src,
                            int player) const {
        if (use_active_runs_for_player(player)) {
            vec_axpy_active_runs(dst, a, src, active_runs_for_player(player));
        } else {
            vec_axpy_active(dst, a, src, active_indices_for_player(player));
        }
    }
    inline void sparse_scale(float* dst, float s, int player) const {
        if (use_active_runs_for_player(player)) {
            vec_scale_active_runs(dst, s, active_runs_for_player(player));
        } else {
            vec_scale_active(dst, s, active_indices_for_player(player));
        }
    }
    inline void sparse_fmadd(float* dst, const float* a, const float* b,
                             int player) const {
        if (use_active_runs_for_player(player)) {
            vec_fmadd_active_runs(dst, a, b, active_runs_for_player(player));
        } else {
            vec_fmadd_active(dst, a, b, active_indices_for_player(player));
        }
    }
    inline void sparse_regret_update(float* regret, const float* action_val,
                                     const float* node_val, int player) const {
        if (use_active_runs_for_player(player)) {
            vec_regret_update_active_runs(
                regret, action_val, node_val, active_runs_for_player(player));
        } else {
            vec_regret_update_active(
                regret, action_val, node_val, active_indices_for_player(player));
        }
    }
    inline void sparse_reach_weighted_strat_sum(
        float* dst, float sw, const float* reach, const float* strat,
        int player) const
    {
        if (use_active_runs_for_player(player)) {
            vec_reach_weighted_strat_sum_active_runs(
                dst, sw, reach, strat, active_runs_for_player(player));
        } else {
            const auto& active = active_indices_for_player(player);
            for (uint16_t c : active) dst[c] += sw * reach[c] * strat[c];
        }
    }
    inline void block_set_zero(float* dst, int player) const {
        vec_set_zero_active_runs(dst, active_blocks_for_player(player));
    }
    inline void block_copy(float* dst, const float* src, int player) const {
        vec_copy_active_runs(dst, src, active_blocks_for_player(player));
    }
    inline void block_mul(float* dst, const float* src, const float* mul,
                          int player) const {
        vec_mul_active_runs(dst, src, mul, active_blocks_for_player(player));
    }
    inline void block_add(float* dst, const float* src, int player) const {
        vec_add_active_runs(dst, src, active_blocks_for_player(player));
    }
    inline void block_axpy(float* dst, float a, const float* src,
                           int player) const {
        vec_axpy_active_runs(dst, a, src, active_blocks_for_player(player));
    }
    inline void block_scale(float* dst, float s, int player) const {
        vec_scale_active_runs(dst, s, active_blocks_for_player(player));
    }
    inline void block_fmadd(float* dst, const float* a, const float* b,
                            int player) const {
        vec_fmadd_active_runs(dst, a, b, active_blocks_for_player(player));
    }
    inline void block_regret_update(float* regret, const float* action_val,
                                    const float* node_val, int player) const {
        vec_regret_update_active_runs(
            regret, action_val, node_val, active_blocks_for_player(player));
    }
    inline void block_reach_weighted_strat_sum(
        float* dst, float sw, const float* reach, const float* strat,
        int player) const
    {
        vec_reach_weighted_strat_sum_active_runs(
            dst, sw, reach, strat, active_blocks_for_player(player));
    }

    // ---- Internal methods ----
    void compute_strategy();

    // v1.8.2 Phase 2: process all SHOWDOWN terminals in `group` (which all
    // share the same matchup_idx) for the OOP traverser via the fused
    // batch kernel. The matrix table (cat + valid for this mi) streams
    // from cache once per c-row across all M terminals' accumulators.
    // Gather is single-threaded (1.3 MB written per group, ~50µs at DRAM
    // bandwidth); the kernel call is parallelized across c-slices via
    // OMP so 8 threads share the matrix-row reads.
    void process_showdown_group_oop(const std::vector<uint32_t>& group);
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

    // Returns base offset of node n's nc-wide row in a flat [N ? nc] array.
    inline std::size_t row_off(uint32_t n) const {
        return static_cast<std::size_t>(n) * row_stride_;
    }

    static inline std::size_t round_up_to_lane(std::size_t n) {
        return (n + kActionLane - 1) & ~(kActionLane - 1);
    }

    inline std::size_t state_off(uint32_t n, uint8_t action = 0) const {
        return node_state_offset_[n]
             + static_cast<std::size_t>(action) * action_stride_;
    }

    inline float* regret_ptr(uint32_t n, uint8_t action = 0) {
        return regrets_.data() + state_off(n, action);
    }
    inline const float* regret_ptr(uint32_t n, uint8_t action = 0) const {
        return regrets_.data() + state_off(n, action);
    }
    inline float* strategy_sum_ptr(uint32_t n, uint8_t action = 0) {
        return strategy_sum_.data() + state_off(n, action);
    }
    inline const float* strategy_sum_ptr(uint32_t n, uint8_t action = 0) const {
        return strategy_sum_.data() + state_off(n, action);
    }
    inline float* current_strategy_ptr(uint32_t n, uint8_t action = 0) {
        return current_strategy_.data() + state_off(n, action);
    }
    inline const float* current_strategy_ptr(uint32_t n, uint8_t action = 0) const {
        return current_strategy_.data() + state_off(n, action);
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

    action_stride_ = round_up_to_lane(nc);
    row_stride_ = action_stride_;
    node_state_offset_.assign(N, kNoDecisionState);
    strategy_.clear();
    player_nodes_.clear();
    player_nodes_.reserve(N / 4);

    std::size_t decision_state_floats = 0;
    for (uint32_t i = 0; i < N; ++i) {
        uint8_t na = ctx.tree->num_children[i];
        auto nt = static_cast<NodeType>(ctx.tree->node_types[i]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) continue;
        if (na == 0) continue;
        player_nodes_.push_back(i);
        node_state_offset_[i] = decision_state_floats;
        decision_state_floats += static_cast<std::size_t>(na) * action_stride_;
    }

    regrets_.assign(decision_state_floats, 0.0f);
    strategy_sum_.assign(decision_state_floats, 0.0f);
    current_strategy_.assign(decision_state_floats, 0.0f);
    for (uint32_t i : player_nodes_) {
        const uint8_t na = ctx.tree->num_children[i];
        const float uniform = 1.0f / na;
        for (uint8_t a = 0; a < na; ++a) {
            float* strat = current_strategy_ptr(i, a);
            std::fill(strat, strat + action_stride_, uniform);
        }
    }

    canonical_weights_f_.resize(nc);
    canonical_inv_weights_f_.resize(nc);
    for (uint16_t k = 0; k < nc; ++k) {
        const float w = static_cast<float>(ctx.iso->canonical_weights[k]);
        canonical_weights_f_[k] = w;
        canonical_inv_weights_f_[k] = (w > 0.0f) ? (1.0f / w) : 0.0f;
    }

    // Per-node flat buffers ??N ? nc floats each.
    const std::size_t flat_sz = static_cast<std::size_t>(N) * row_stride_;
    reach_oop_.assign(flat_sz, 0.0f);
    reach_ip_.assign(flat_sz, 0.0f);
    value_.assign(flat_sz, 0.0f);

    pos_sum_scratch_.assign(action_stride_, 0.0f);
    inv_pos_sum_scratch_.assign(action_stride_, 0.0f);
    uniform_or_zero_scratch_.assign(action_stride_, 0.0f);

    // v1.8.0 P2-5: pre-allocate the terminal scratch so the inner OMP loop
    // doesn't heap-allocate per terminal visit. cpu_threads_effective_ is
    // already resolved earlier in prepare(). Total = threads ? nc floats ??    // tiny next to reach_oop_/reach_ip_/value_ which are N ? nc each.
    terminal_opp_reach_w_.assign(
        static_cast<std::size_t>(cpu_threads_effective_) * nc, 0.0f);
    showdown_rank_scratch_.resize(cpu_threads_effective_);

    // v1.8.1+ reset phase timers for this solve.
    phase_compute_strategy_ms_  = 0.0;
    phase_apply_discount_ms_    = 0.0;
    phase_forward_pass_ms_      = 0.0;
    phase_backward_pass_oop_ms_ = 0.0;
    phase_backward_pass_ip_ms_  = 0.0;

    // POST_OPTIMIZATION_REVIEW Sec 4.2: per-thread showdown/fold accumulators.
    // Sized to cpu_threads_effective_ so evaluate_terminal() can index by
    // omp_get_thread_num() without bounds checks. Reset to 0 each prepare().
    showdown_acc_per_thread_.assign(cpu_threads_effective_, 0.0);
    fold_acc_per_thread_.assign(cpu_threads_effective_, 0.0);

    // v1.8.1+ build the out-of-range skip masks from root reach.
    // mask[c] = 1 iff that combo has 0-reach at root (i.e. excluded
    // from the user's range, so propagates as 0 throughout the tree
    // and out[c]=0 at any terminal is mathematically safe).
    oop_out_of_range_mask_.assign(nc, 0);
    ip_out_of_range_mask_.assign(nc, 0);
    oop_terminal_active_indices_.clear();
    ip_terminal_active_indices_.clear();
    oop_terminal_active_runs_.clear();
    ip_terminal_active_runs_.clear();
    oop_terminal_active_blocks_.clear();
    ip_terminal_active_blocks_.clear();
    oop_terminal_active_indices_.reserve(nc);
    ip_terminal_active_indices_.reserve(nc);
    oop_has_out_of_range_ = false;
    ip_has_out_of_range_ = false;
    oop_use_terminal_active_list_ = false;
    ip_use_terminal_active_list_ = false;
    oop_use_active_runs_ = false;
    ip_use_active_runs_ = false;
    oop_use_active_blocks_ = false;
    ip_use_active_blocks_ = false;
    oop_use_terminal_active2_ = false;
    ip_use_terminal_active2_ = false;
    oop_use_sparse_traversal_ = false;
    ip_use_sparse_traversal_ = false;
    showdown_signed_coeff_supported_ = false;
    fold_precomputed_enabled_ = false;
    fold_fallback_metadata_ = fold_blocker::Metadata{};
    fold_metadata_per_runout_.clear();
    if constexpr (kFoldBlockerShortcutEnabled && kFoldBlockerPrecomputedEnabled) {
        if (ctx.matchup_board_masks != nullptr
            && ctx.matchup_board_masks->size() >= kFoldBlockerPrecomputeMinRunouts) {
            const auto& board_masks = *ctx.matchup_board_masks;
            fold_metadata_per_runout_.reserve(board_masks.size());
            for (CardMask board_mask : board_masks) {
                fold_metadata_per_runout_.push_back(
                    fold_blocker::build_metadata(*ctx.iso, board_mask));
            }
            fold_precomputed_enabled_ = !fold_metadata_per_runout_.empty();
        }
    }
    showdown_rank_metadata_per_runout_.clear();
    showdown_rank_blocker_supported_ =
        kShowdownRankBlockerShortcutEnabled
        && ctx.matchup_original_ranks_per_runout != nullptr
        && showdown_rank_blocker::supports_singleton_iso(*ctx.iso);
    if (showdown_rank_blocker_supported_) {
        const auto& rank_tables = *ctx.matchup_original_ranks_per_runout;
        showdown_rank_metadata_per_runout_.reserve(rank_tables.size());
        bool any_rank_metadata = false;
        for (const auto& ranks : rank_tables) {
            auto metadata =
                showdown_rank_blocker::build_metadata(*ctx.iso, ranks);
            any_rank_metadata = any_rank_metadata || metadata.valid;
            showdown_rank_metadata_per_runout_.push_back(std::move(metadata));
        }
        showdown_rank_blocker_supported_ = any_rank_metadata;
    }
    showdown_signed_coeff_supported_ =
        ctx.matchup_showdown_count_per_runout != nullptr
        && !ctx.matchup_showdown_count_per_runout->empty();
    uint16_t ip_out_of_range_count = 0;
    if (ctx.oop_reach != nullptr) {
        for (uint16_t c = 0; c < nc; ++c) {
            if ((*ctx.oop_reach)[c] == 0.0f) {
                oop_out_of_range_mask_[c] = 1;
                oop_has_out_of_range_ = true;
            } else {
                oop_terminal_active_indices_.push_back(c);
            }
        }
    } else {
        for (uint16_t c = 0; c < nc; ++c) {
            oop_terminal_active_indices_.push_back(c);
        }
    }
    if (ctx.ip_reach != nullptr) {
        for (uint16_t c = 0; c < nc; ++c) {
            if ((*ctx.ip_reach)[c] == 0.0f) {
                ip_out_of_range_mask_[c] = 1;
                ip_has_out_of_range_ = true;
                ++ip_out_of_range_count;
            } else {
                ip_terminal_active_indices_.push_back(c);
            }
        }
    } else {
        for (uint16_t c = 0; c < nc; ++c) {
            ip_terminal_active_indices_.push_back(c);
        }
    }
    build_active_runs(oop_terminal_active_indices_, oop_terminal_active_runs_);
    build_active_runs(ip_terminal_active_indices_, ip_terminal_active_runs_);
    build_active_lane_blocks(
        oop_terminal_active_indices_, oop_terminal_active_blocks_, nc);
    build_active_lane_blocks(
        ip_terminal_active_indices_, ip_terminal_active_blocks_, nc);
    ip_use_terminal_output_skip_ =
        (static_cast<uint32_t>(ip_out_of_range_count) * 8u
         >= static_cast<uint32_t>(nc) * 7u);
    constexpr uint32_t kActiveListDensityDen = 4u; // active <= 25%
    oop_use_terminal_active_list_ =
        (static_cast<uint32_t>(oop_terminal_active_indices_.size())
            * kActiveListDensityDen
         <= static_cast<uint32_t>(nc));
    ip_use_terminal_active_list_ =
        (static_cast<uint32_t>(ip_terminal_active_indices_.size())
            * kActiveListDensityDen
         <= static_cast<uint32_t>(nc));
    oop_use_active_runs_ =
        oop_use_terminal_active_list_
        && active_runs_are_worth_using(
            oop_terminal_active_indices_, oop_terminal_active_runs_);
    ip_use_active_runs_ =
        ip_use_terminal_active_list_
        && active_runs_are_worth_using(
            ip_terminal_active_indices_, ip_terminal_active_runs_);
    oop_use_active_blocks_ =
        oop_use_terminal_active_list_
        && active_blocks_are_worth_using(
            oop_terminal_active_indices_, oop_terminal_active_blocks_, nc);
    ip_use_active_blocks_ =
        ip_use_terminal_active_list_
        && active_blocks_are_worth_using(
            ip_terminal_active_indices_, ip_terminal_active_blocks_, nc);
    // Keep medium_sparse (14.4%) on the dense traversal path for now: exact
    // active-index traversal was parity-safe but regressed 100 iters from
    // ~104ms to ~219ms because forward/backward become scalar gather loops.
    constexpr uint32_t kSparseTraversalDensityDen = 8u; // active <= 12.5%
    oop_use_sparse_traversal_ =
        (static_cast<uint32_t>(oop_terminal_active_indices_.size())
            * kSparseTraversalDensityDen
         <= static_cast<uint32_t>(nc));
    ip_use_sparse_traversal_ =
        (static_cast<uint32_t>(ip_terminal_active_indices_.size())
            * kSparseTraversalDensityDen
         <= static_cast<uint32_t>(nc));
    // Active2 terminal kernels are implemented and parity-tested, but left
    // disabled at runtime for now: the different reduction order causes CFR
    // trajectory drift against the reference backend, and the current scalar
    // indexed inner loop did not beat the one-sided active path wall-clock.
    constexpr bool kEnableTerminalActive2 = false;
    oop_use_terminal_active2_ = kEnableTerminalActive2
        && (static_cast<uint32_t>(oop_terminal_active_indices_.size())
            * kSparseTraversalDensityDen
            <= static_cast<uint32_t>(nc));
    ip_use_terminal_active2_ = kEnableTerminalActive2
        && (static_cast<uint32_t>(ip_terminal_active_indices_.size())
            * kSparseTraversalDensityDen
            <= static_cast<uint32_t>(nc));

    build_level_schedule();

    // Phase 2: bucket level-0 terminals by their matchup_idx so backward_pass
    // can iterate groups serially (parallel within group) and amortize the
    // per-table DRAM streaming across all terminals that share that table.
    // Reuse stats on monotone: 446 tables ? ~245 terminals each.
    terminals_by_table_.clear();
    showdowns_by_table_.clear();
    non_showdown_l0_.clear();
    max_showdown_group_size_ = 0;
    if (num_levels_ > 0) {
        const uint32_t lo = level_offsets_[0];
        const uint32_t hi = level_offsets_[1];
        int32_t max_mi = -1;
        for (uint32_t k = lo; k < hi; ++k) {
            uint32_t n = node_order_[k];
            if (n < ctx.tree->matchup_idx.size()) {
                int32_t mi = ctx.tree->matchup_idx[n];
                if (mi > max_mi) max_mi = mi;
            }
        }
        if (max_mi >= 0) {
            terminals_by_table_.resize(static_cast<std::size_t>(max_mi) + 1);
            showdowns_by_table_.resize(static_cast<std::size_t>(max_mi) + 1);
            for (uint32_t k = lo; k < hi; ++k) {
                uint32_t n = node_order_[k];
                int32_t mi = (n < ctx.tree->matchup_idx.size())
                    ? ctx.tree->matchup_idx[n] : -1;
                if (mi >= 0) {
                    const std::size_t bin = static_cast<std::size_t>(mi);
                    terminals_by_table_[bin].push_back(n);
                    auto tt = static_cast<TerminalType>(ctx.tree->terminal_types[n]);
                    if (tt == TerminalType::SHOWDOWN) {
                        showdowns_by_table_[bin].push_back(n);
                    } else {
                        non_showdown_l0_.push_back(n);
                    }
                } else {
                    non_showdown_l0_.push_back(n);
                }
            }
            for (const auto& g : showdowns_by_table_) {
                if (g.size() > max_showdown_group_size_) {
                    max_showdown_group_size_ = g.size();
                }
            }
        }
    }
    // Pre-size batch scratch to the largest group so the hot path never
    // resizes. opp_reach_w buffer is [max_group ? nc] flat.
    if (max_showdown_group_size_ > 0) {
        batch_opp_reach_w_.assign(
            max_showdown_group_size_ * static_cast<std::size_t>(nc), 0.0f);
        batch_out_ptrs_.assign(max_showdown_group_size_, nullptr);
        batch_reach_ptrs_.assign(max_showdown_group_size_, nullptr);
        batch_win_payoff_.assign(max_showdown_group_size_, 0.0f);
        batch_lose_payoff_.assign(max_showdown_group_size_, 0.0f);
        batch_tie_payoff_.assign(max_showdown_group_size_, 0.0f);
    }
}

// ============================================================================
// compute_strategy / apply_dcfr_discount ??same as CpuBackend, just
// re-implemented here so the two backends are independent.
// ============================================================================

// Note (v1.8.0 Sprint 2): the no-precision-loss guide proposed
// parallelizing this function across player nodes via per-thread scratch.
// Implemented + measured + reverted because both the parallel and serial
// variants of the new structure regressed standard-benchmark 8T
// throughput by ~10-15%, past the doc's >5% stop condition. Theory: 216
// player nodes ? ~6µs serial work = ~1.3ms / call; OMP team setup on
// MSVC is comparable, and the bigger per-thread scratch (8nc ? 3 buffers
// = 113KB) appears to perturb heap layout enough that even the serial
// path slows. Sprint 3's persistent-team refactor is the right
// structural fix; revisit parallelization here once that lands.
inline void LevelizedCpuBackend::compute_strategy() {
    const uint16_t nc = ctx_.iso->num_canonical;
    const std::size_t stride = action_stride_;
    const auto& resolved_locks = *ctx_.resolved_locks;
    const bool allow_sparse_strategy =
        (ctx_.config->dcfr_schedule != SolverConfig::DcfrSchedule::POSTFLOP_STYLE);

    for (uint32_t n : player_nodes_) {
        uint8_t na = ctx_.tree->num_children[n];
        const float uniform = 1.0f / na;
        const float* regret_base = regret_ptr(n);
        float*       strat_base  = current_strategy_ptr(n);
        const int acting = ctx_.tree->active_player[n];

        const bool use_sparse_strategy =
            allow_sparse_strategy && use_sparse_traversal_for_player(acting);
        const bool use_block_strategy =
            allow_sparse_strategy && use_block_strategy_for_player(acting);
        if (use_block_strategy) {
            const auto& blocks = active_blocks_for_player(acting);
            vec_set_zero_active_runs(pos_sum_scratch_.data(), blocks);
            for (uint8_t a = 0; a < na; ++a) {
                const float* regret_a =
                    regret_base + static_cast<std::size_t>(a) * stride;
                for (const auto& block : blocks) {
                    cpu_simd::vec_pos_add(
                        pos_sum_scratch_.data() + block.start,
                        regret_a + block.start,
                        block.count);
                }
            }

            for (const auto& block : blocks) {
                const uint16_t end =
                    static_cast<uint16_t>(block.start + block.count);
                for (uint16_t c = block.start; c < end; ++c) {
                    if (pos_sum_scratch_[c] > 0.0f) {
                        inv_pos_sum_scratch_[c] =
                            1.0f / pos_sum_scratch_[c];
                        uniform_or_zero_scratch_[c] = 0.0f;
                    } else {
                        inv_pos_sum_scratch_[c] = 0.0f;
                        uniform_or_zero_scratch_[c] = uniform;
                    }
                }
            }

            for (uint8_t a = 0; a < na; ++a) {
                const float* regret_a =
                    regret_base + static_cast<std::size_t>(a) * stride;
                float* strat_a =
                    strat_base + static_cast<std::size_t>(a) * stride;
                for (const auto& block : blocks) {
                    cpu_simd::vec_pos_normalize(
                        strat_a + block.start,
                        regret_a + block.start,
                        inv_pos_sum_scratch_.data() + block.start,
                        uniform_or_zero_scratch_.data() + block.start,
                        block.count);
                }
            }
        } else if (use_sparse_strategy) {
            auto update_combo = [&](uint16_t c) {
                float pos_sum = 0.0f;
                for (uint8_t a = 0; a < na; ++a) {
                    const float r =
                        regret_base[static_cast<std::size_t>(a) * stride + c];
                    if (r > 0.0f) pos_sum += r;
                }
                if (pos_sum > 0.0f) {
                    const float inv = 1.0f / pos_sum;
                    for (uint8_t a = 0; a < na; ++a) {
                        const float r =
                            regret_base[static_cast<std::size_t>(a) * stride + c];
                        strat_base[static_cast<std::size_t>(a) * stride + c] =
                            (r > 0.0f) ? r * inv : 0.0f;
                    }
                } else {
                    for (uint8_t a = 0; a < na; ++a) {
                        strat_base[static_cast<std::size_t>(a) * stride + c] =
                            uniform;
                    }
                }
            };
            if (use_active_runs_for_player(acting)) {
                const auto& runs = active_runs_for_player(acting);
                for (const auto& run : runs) {
                    const uint16_t end =
                        static_cast<uint16_t>(run.start + run.count);
                    for (uint16_t c = run.start; c < end; ++c) update_combo(c);
                }
            } else {
                const auto& active = active_indices_for_player(acting);
                for (uint16_t c : active) update_combo(c);
            }
        } else {
            if (na == 2) {
                cpu_simd::vec_pos_normalize2(
                    strat_base,
                    strat_base + stride,
                    regret_base,
                    regret_base + stride,
                    stride);
            } else {
                cpu_simd::vec_set_zero(pos_sum_scratch_.data(), stride);
                for (uint8_t a = 0; a < na; ++a) {
                    cpu_simd::vec_pos_add(
                        pos_sum_scratch_.data(),
                        regret_base + static_cast<std::size_t>(a) * stride,
                        stride);
                }

                for (std::size_t c = 0; c < stride; ++c) {
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
                        strat_base + static_cast<std::size_t>(a) * stride,
                        regret_base + static_cast<std::size_t>(a) * stride,
                        inv_pos_sum_scratch_.data(),
                        uniform_or_zero_scratch_.data(),
                        stride);
                }
            }
        }

        if (!resolved_locks.empty()) {
            for (uint16_t c = 0; c < nc; ++c) {
                auto lock_it = resolved_locks.find({n, c});
                if (lock_it == resolved_locks.end()) continue;
                const auto& forced = lock_it->second;
                for (uint8_t a = 0; a < na && a < forced.size(); ++a) {
                    strat_base[static_cast<std::size_t>(a) * stride + c] = forced[a];
                }
            }
        }
    }
}

inline void LevelizedCpuBackend::apply_dcfr_discount(int iteration) {
    float pos_disc, neg_disc, strat_weight;
    compute_dcfr_factors(iteration, *ctx_.config, pos_disc, neg_disc, strat_weight);
    (void)strat_weight;
    const std::size_t stride = action_stride_;

    auto discount_node = [&](uint32_t n) {
        uint8_t na = ctx_.tree->num_children[n];
        const int acting = ctx_.tree->active_player[n];
        if (use_sparse_traversal_for_player(acting)) {
            float* regret_base = regret_ptr(n);
            for (uint8_t a = 0; a < na; ++a) {
                float* regret = regret_base + static_cast<std::size_t>(a) * stride;
                if (use_active_runs_for_player(acting)) {
                    vec_dcfr_discount_active_runs(
                        regret, pos_disc, neg_disc, active_runs_for_player(acting));
                } else {
                    const auto& active = active_indices_for_player(acting);
                    for (uint16_t c : active) {
                        const float r = regret[c];
                        regret[c] = (r > 0.0f) ? r * pos_disc : r * neg_disc;
                    }
                }
            }
        } else if (use_block_traversal_for_player(acting)) {
            float* regret_base = regret_ptr(n);
            for (uint8_t a = 0; a < na; ++a) {
                float* regret = regret_base + static_cast<std::size_t>(a) * stride;
                vec_dcfr_discount_active_runs(
                    regret, pos_disc, neg_disc, active_blocks_for_player(acting));
            }
        } else {
            std::size_t total = static_cast<std::size_t>(na) * stride;
            cpu_simd::vec_dcfr_discount(regret_ptr(n), pos_disc, neg_disc, total);
        }
    };

#if defined(_OPENMP)
    if (cpu_threads_effective_ > 1 && player_nodes_.size() >= 4096) {
        #pragma omp parallel for schedule(static) num_threads(static_cast<int>(cpu_threads_effective_))
        for (int64_t i = 0; i < static_cast<int64_t>(player_nodes_.size()); ++i) {
            discount_node(player_nodes_[static_cast<std::size_t>(i)]);
        }
        return;
    }
#endif

    for (uint32_t n : player_nodes_) {
        discount_node(n);
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
    const auto& matchup_valid_table =
        (ctx_.matchup_valid_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_valid_per_runout->size())
            ? (*ctx_.matchup_valid_per_runout)[mi]
            : *ctx_.matchup_valid;
    // v1.8.2 A2 encoding: showdown reads pre-thresholded category bytes
    // instead of the continuous ev. Falls back to the root-board single
    // category table when this terminal sits on the root board (no chance
    // enumeration in the tree).
    const auto& matchup_category_table =
        (ctx_.matchup_category_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_category_per_runout->size())
            ? (*ctx_.matchup_category_per_runout)[mi]
            : *ctx_.matchup_category;
    const std::vector<int8_t>* matchup_showdown_count_table = nullptr;
    if (ctx_.matchup_showdown_count_per_runout && mi >= 0 &&
        static_cast<std::size_t>(mi) < ctx_.matchup_showdown_count_per_runout->size()) {
        matchup_showdown_count_table =
            &(*ctx_.matchup_showdown_count_per_runout)[mi];
    } else {
        matchup_showdown_count_table = ctx_.matchup_showdown_count;
    }
    const showdown_rank_blocker::Metadata* matchup_rank_metadata = nullptr;
    if (showdown_rank_blocker_supported_ && mi >= 0 &&
        static_cast<std::size_t>(mi) < showdown_rank_metadata_per_runout_.size()) {
        const auto& metadata =
            showdown_rank_metadata_per_runout_[static_cast<std::size_t>(mi)];
        if (metadata.valid) matchup_rank_metadata = &metadata;
    }
    const fold_blocker::Metadata* matchup_fold_metadata = nullptr;
    if constexpr (kFoldBlockerPrecomputedEnabled) {
        if (fold_precomputed_enabled_) {
            if (mi >= 0 &&
                static_cast<std::size_t>(mi) < fold_metadata_per_runout_.size()) {
                const auto& metadata =
                    fold_metadata_per_runout_[static_cast<std::size_t>(mi)];
                if (metadata.valid) matchup_fold_metadata = &metadata;
            }
            if (matchup_fold_metadata == nullptr && fold_fallback_metadata_.valid) {
                matchup_fold_metadata = &fold_fallback_metadata_;
            }
        }
    }
    const float*   matchup_valid    = matchup_valid_table.data();
    const uint8_t* matchup_category = matchup_category_table.data();
    const int8_t* matchup_showdown_count =
        (matchup_showdown_count_table != nullptr &&
         matchup_showdown_count_table->size() >=
             static_cast<std::size_t>(nc) * nc)
            ? matchup_showdown_count_table->data()
            : nullptr;

    const bool use_active_list = (traverser == 0)
        ? oop_use_terminal_active_list_
        : ip_use_terminal_active_list_;
    const std::vector<uint16_t>& active_indices = (traverser == 0)
        ? oop_terminal_active_indices_
        : ip_terminal_active_indices_;
    const std::vector<uint16_t>& opp_active_indices = (traverser == 0)
        ? ip_terminal_active_indices_
        : oop_terminal_active_indices_;
    const std::vector<cpu_simd::ActiveRun>& active_runs = (traverser == 0)
        ? oop_terminal_active_runs_
        : ip_terminal_active_runs_;
    const std::vector<cpu_simd::ActiveRun>& opp_active_runs = (traverser == 0)
        ? ip_terminal_active_runs_
        : oop_terminal_active_runs_;
    const std::vector<cpu_simd::ActiveRun>& opp_active_blocks = (traverser == 0)
        ? ip_terminal_active_blocks_
        : oop_terminal_active_blocks_;
    const bool use_active_runs =
        use_active_list && use_active_runs_for_player(traverser);
    const bool use_opp_active_runs = use_active_runs_for_player(1 - traverser);
    const bool use_opp_active_blocks =
        use_active_blocks_for_player(1 - traverser);
    // With sparse traversal, inactive combo lanes are provably ignored by
    // later value/regret operations. The active kernels are also tested in a
    // no-zero mode, so narrow-range terminals can skip the full-row memset.
    const bool clear_active_terminal_output =
        !(kSparseTerminalNoFullClearEnabled
          && use_sparse_traversal_for_player(traverser));
    const bool use_dual_active = use_active_list
        && ((traverser == 0) ? ip_use_terminal_active2_
                             : oop_use_terminal_active2_);
    const bool use_dual_active_showdown = use_dual_active;
    const bool use_dual_active_fold = use_dual_active;
    const bool has_skip = (traverser == 0)
        ? oop_has_out_of_range_
        : ip_use_terminal_output_skip_;
    const uint8_t* skip_mask = (!use_active_list && has_skip)
        ? ((traverser == 0)
            ? oop_out_of_range_mask_.data()
            : ip_out_of_range_mask_.data())
        : nullptr;
    // opp_reach ? canonical_weight, computed once into thread-local scratch.
    // The pre-v1.8.0 code allocated a fresh `std::vector<float>(nc)` here on
    // every terminal visit ??under OMP that's hundreds of mallocs per iter
    // contending on the heap allocator. Now we slot into a fixed buffer
    // sized to cpu_threads_effective_ ? nc in prepare().
    const float* opp_reach = (traverser == 0)
        ? &reach_ip_[row_off(node_idx)]
        : &reach_oop_[row_off(node_idx)];
    const bool use_signed_coeff_showdown =
        tt == TerminalType::SHOWDOWN
        && use_signed_coeff_showdown_for_traverser(traverser)
        && matchup_showdown_count != nullptr
        && tie_payoff == 0.0f
        && lose_payoff == -win_payoff;

    auto current_board_mask = [&]() {
        if (ctx_.matchup_board_masks && mi >= 0 &&
            static_cast<std::size_t>(mi) < ctx_.matchup_board_masks->size()) {
            return (*ctx_.matchup_board_masks)[static_cast<std::size_t>(mi)];
        }
        return (ctx_.config != nullptr)
            ? board_to_mask(ctx_.config->board.data(), ctx_.config->board_size)
            : CardMask{0};
    };
    auto fold_self_payoff = [&]() {
        uint32_t parent = tree.parent_indices[node_idx];
        float unmatched_bet = (parent < tree.total_nodes)
            ? tree.bet_into[parent] : 0.0f;
        float matched_pot = pot_total - unmatched_bet;
        float fold_win_gain  = matched_pot * 0.5f - rake;
        float fold_lose_loss = -matched_pot * 0.5f;
        float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
        return ((traverser == 0 && sign_oop > 0)
                || (traverser == 1 && sign_oop < 0))
            ? fold_win_gain
            : fold_lose_loss;
    };

    const bool use_active_fold_blocker =
        use_active_list && !use_sparse_traversal_for_player(traverser);
    if (tt != TerminalType::SHOWDOWN
        && kFoldBlockerShortcutEnabled
        && (!use_active_list || use_active_fold_blocker)) {
#ifdef _OPENMP
        const double _fb_t0 = omp_get_wtime();
#endif
        if (use_active_fold_blocker) {
            if constexpr (kFoldBlockerPrecomputedEnabled) {
                if (matchup_fold_metadata != nullptr) {
                    fold_blocker::fold_active_precomputed(
                        *matchup_fold_metadata, opp_reach,
                        active_indices.data(), active_indices.size(),
                        opp_active_indices.data(), opp_active_indices.size(),
                        clear_active_terminal_output,
                        fold_self_payoff(), out, row_stride_);
                } else {
                    fold_blocker::fold_active(
                        *ctx_.iso, current_board_mask(), opp_reach,
                        active_indices.data(), active_indices.size(),
                        opp_active_indices.data(), opp_active_indices.size(),
                        clear_active_terminal_output,
                        fold_self_payoff(), out, row_stride_);
                }
            } else {
                fold_blocker::fold_active(
                    *ctx_.iso, current_board_mask(), opp_reach,
                    active_indices.data(), active_indices.size(),
                    opp_active_indices.data(), opp_active_indices.size(),
                    clear_active_terminal_output,
                    fold_self_payoff(), out, row_stride_);
            }
        } else {
            if constexpr (kFoldBlockerPrecomputedEnabled) {
                if (matchup_fold_metadata != nullptr) {
                    fold_blocker::fold_dense_precomputed(
                        *matchup_fold_metadata, opp_reach, skip_mask,
                        fold_self_payoff(), out, row_stride_);
                } else {
                    fold_blocker::fold_dense(
                        *ctx_.iso, current_board_mask(), opp_reach, skip_mask,
                        fold_self_payoff(), out, row_stride_);
                }
            } else {
                fold_blocker::fold_dense(
                    *ctx_.iso, current_board_mask(), opp_reach, skip_mask,
                    fold_self_payoff(), out, row_stride_);
            }
        }
#ifdef _OPENMP
        {
            const double _et_dt = omp_get_wtime() - _fb_t0;
            const int _et_tid = omp_get_thread_num();
            const uint32_t _et_safe = (_et_tid < 0
                || static_cast<uint32_t>(_et_tid) >= cpu_threads_effective_)
                    ? 0u : static_cast<uint32_t>(_et_tid);
            fold_acc_per_thread_[_et_safe] += _et_dt;
        }
#endif
        return;
    }

    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    float* opp_reach_w = nullptr;
    const bool use_sparse_opp_reach_build =
        use_sparse_opp_reach_build_for_traverser(traverser);
    if (!use_signed_coeff_showdown) {
        opp_reach_w = terminal_scratch_for_thread(tid);
        if (use_sparse_opp_reach_build && use_opp_active_runs) {
            for (const auto& run : opp_active_runs) {
                cpu_simd::vec_copy(
                    opp_reach_w + run.start,
                    opp_reach + run.start,
                    run.count);
                cpu_simd::vec_mul_in_place(
                    opp_reach_w + run.start,
                    canonical_weights_f_.data() + run.start,
                    run.count);
            }
        } else if (use_sparse_opp_reach_build) {
            for (uint16_t c : opp_active_indices) {
                opp_reach_w[c] = opp_reach[c] * canonical_weights_f_[c];
            }
        } else if (use_dual_active) {
            if (use_opp_active_runs) {
                for (const auto& run : opp_active_runs) {
                    const uint16_t end =
                        static_cast<uint16_t>(run.start + run.count);
                    for (uint16_t c = run.start; c < end; ++c) {
                        opp_reach_w[c] = opp_reach[c] * canonical_weights_f_[c];
                    }
                }
            } else {
                for (uint16_t c : opp_active_indices) {
                    opp_reach_w[c] = opp_reach[c] * canonical_weights_f_[c];
                }
            }
        } else {
            cpu_simd::vec_copy(opp_reach_w, opp_reach, nc);
            cpu_simd::vec_mul_in_place(opp_reach_w, canonical_weights_f_.data(), nc);
        }
    }

    // v1.8.1+ out-of-range skip: skip rows for combos that have 0 reach
    // at root (user's range excludes them). These combos propagate as 0
    // throughout the tree, so out[c]=0 at any terminal is provably safe.
    // We do NOT skip combos with 0 reach at THIS terminal but >0 reach
    // at some ancestor ??that would break regret updates at the ancestor.
    // The mask is built once in prepare() from root reach.
    // POST_OPTIMIZATION_REVIEW Sec 4.2: time showdown vs fold separately.
    // omp_get_wtime() resolution on Windows is ~1繕s which is > the call cost
    // for tiny terminals ??but called once per terminal, the noise averages
    // out across the iter loop and the per-call overhead is well under 1%.
#ifdef _OPENMP
    const double _et_t0 = omp_get_wtime();
#endif
    if (tt == TerminalType::SHOWDOWN) {
        // v1.8.0 P3-8 spike: full-matrix kernels fold the per-c outer loop
        // into the kernel itself, hoisting SIMD constants out of the hot
        // path and saving nc dispatch-table lookups per terminal call.
        // v1.8.2 A2: kernels now read category bytes (1 B/cell) instead of
        // ev floats (4 B/cell) ??see SolverContext::matchup_category.
        const bool use_rank_blocker_showdown =
            use_rank_blocker_showdown_for_traverser(traverser)
            && matchup_rank_metadata != nullptr;
        const bool use_active_rank_blocker_showdown =
            use_active_rank_blocker_showdown_for_traverser(traverser)
            && matchup_rank_metadata != nullptr;
        if (use_active_rank_blocker_showdown) {
            const uint32_t safe_tid =
                (tid < 0 || static_cast<uint32_t>(tid) >= cpu_threads_effective_)
                    ? 0u : static_cast<uint32_t>(tid);
            showdown_rank_blocker::showdown_active_singleton_precomputed(
                *matchup_rank_metadata, opp_reach_w,
                active_indices.data(), active_indices.size(),
                opp_active_indices.data(), opp_active_indices.size(),
                clear_active_terminal_output,
                out, row_stride_, win_payoff, lose_payoff, tie_payoff,
                showdown_rank_scratch_[safe_tid]);
        } else if (use_rank_blocker_showdown) {
            const uint32_t safe_tid =
                (tid < 0 || static_cast<uint32_t>(tid) >= cpu_threads_effective_)
                    ? 0u : static_cast<uint32_t>(tid);
            showdown_rank_blocker::showdown_dense_singleton_precomputed(
                *matchup_rank_metadata, opp_reach_w,
                skip_mask,
                out, row_stride_, win_payoff, lose_payoff, tie_payoff,
                showdown_rank_scratch_[safe_tid]);
        } else if (use_signed_coeff_showdown) {
            if (traverser == 0) {
                cpu_simd::showdown_oop_signed_count_zero_rake(
                    matchup_showdown_count, opp_reach,
                    canonical_inv_weights_f_.data(), skip_mask,
                    out, nc, win_payoff);
            } else {
                cpu_simd::showdown_ip_signed_count_zero_rake(
                    matchup_showdown_count, opp_reach,
                    canonical_inv_weights_f_.data(), skip_mask,
                    out, nc, win_payoff);
            }
        } else if (traverser == 0) {
            if (use_dual_active_showdown) {
                cpu_simd::showdown_oop_full_active2(
                    matchup_category, matchup_valid, opp_reach_w,
                    active_indices.data(), active_indices.size(),
                    opp_active_indices.data(), opp_active_indices.size(),
                    out, nc, win_payoff, lose_payoff, tie_payoff);
            } else if (use_active_list) {
                if (use_opp_active_blocks) {
                    cpu_simd::showdown_oop_full_active_opp_blocks(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_indices.data(), active_indices.size(),
                        opp_active_blocks.data(), opp_active_blocks.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                } else if (use_active_runs) {
                    cpu_simd::showdown_oop_full_active_runs(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_runs.data(), active_runs.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                } else {
                    cpu_simd::showdown_oop_full_active(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_indices.data(), active_indices.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                }
            } else {
                cpu_simd::showdown_oop_full(
                    matchup_category, matchup_valid, opp_reach_w, skip_mask, out, nc,
                    win_payoff, lose_payoff, tie_payoff);
            }
        } else {
            if (use_dual_active_showdown) {
                cpu_simd::showdown_ip_full_active2(
                    matchup_category, matchup_valid, opp_reach_w,
                    active_indices.data(), active_indices.size(),
                    opp_active_indices.data(), opp_active_indices.size(),
                    out, nc, win_payoff, lose_payoff, tie_payoff);
            } else if (use_active_list) {
                if (use_opp_active_blocks && !use_active_runs) {
                    cpu_simd::showdown_ip_full_active_opp_blocks(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_indices.data(), active_indices.size(),
                        opp_active_blocks.data(), opp_active_blocks.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                } else if (use_active_runs) {
                    cpu_simd::showdown_ip_full_active_runs(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_runs.data(), active_runs.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                } else {
                    cpu_simd::showdown_ip_full_active(
                        matchup_category, matchup_valid, opp_reach_w,
                        active_indices.data(), active_indices.size(),
                        out, nc, win_payoff, lose_payoff, tie_payoff,
                        clear_active_terminal_output);
                }
            } else {
                cpu_simd::showdown_ip_full(
                    matchup_category, matchup_valid, opp_reach_w, skip_mask, out, nc,
                    win_payoff, lose_payoff, tie_payoff);
            }
        }
    } else {
        // Fold terminal ??asymmetric pot fix from CpuBackend.
        const float self_payoff = fold_self_payoff();

        if (traverser == 0) {
            // v1.8.1+ out-of-range skip ??same skip_mask as showdown.
            if (use_active_list) {
                if (clear_active_terminal_output) {
                    cpu_simd::vec_set_zero(out, nc);
                }
                auto eval_fold_combo = [&](uint16_t c) {
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(c) * nc;
                    float opp_total;
                    if (use_dual_active_fold) {
                        opp_total = use_opp_active_runs
                            ? cpu_simd::dot_valid_reach_active_runs(
                                valid_row, opp_reach_w,
                                opp_active_runs.data(), opp_active_runs.size())
                            : cpu_simd::dot_valid_reach_active(
                                valid_row, opp_reach_w,
                                opp_active_indices.data(), opp_active_indices.size());
                    } else if (use_opp_active_blocks) {
                        opp_total = cpu_simd::dot_valid_reach_active_runs(
                            valid_row, opp_reach_w,
                            opp_active_blocks.data(), opp_active_blocks.size());
                    } else {
                        opp_total = cpu_simd::dot_valid_reach(valid_row, opp_reach_w, nc);
                    }
                    out[c] = self_payoff * opp_total;
                };
                if (use_active_runs) {
                    for (const auto& run : active_runs) {
                        const uint16_t end =
                            static_cast<uint16_t>(run.start + run.count);
                        for (uint16_t c = run.start; c < end; ++c) {
                            eval_fold_combo(c);
                        }
                    }
                } else {
                    for (uint16_t c : active_indices) eval_fold_combo(c);
                }
            } else {
                for (uint16_t c = 0; c < nc; ++c) {
                    if (skip_mask && skip_mask[c]) {
                        out[c] = 0.0f;
                        continue;
                    }
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(c) * nc;
                    float opp_total =
                        cpu_simd::dot_valid_reach(valid_row, opp_reach_w, nc);
                    out[c] = self_payoff * opp_total;
                }
            }
        } else {
            if (use_active_list && !clear_active_terminal_output) {
                sparse_set_zero(out, traverser);
            } else {
                cpu_simd::vec_set_zero(out, nc);
            }
            if (use_dual_active_fold) {
                for (uint16_t ci : opp_active_indices) {
                    float rw_ci = opp_reach_w[ci];
                    if (rw_ci == 0.0f) continue;
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(ci) * nc;
                    if (use_active_runs) {
                        cpu_simd::fold_ip_step_active_runs(
                            out, valid_row, rw_ci,
                            active_runs.data(), active_runs.size());
                    } else {
                        cpu_simd::fold_ip_step_active(
                            out, valid_row, rw_ci,
                            active_indices.data(), active_indices.size());
                    }
                }
            } else {
                auto apply_fold_ci = [&](uint16_t ci) {
                    const float rw_ci = opp_reach_w[ci];
                    if (rw_ci == 0.0f) return;
                    const float* valid_row =
                        matchup_valid + static_cast<std::size_t>(ci) * nc;
                    if (use_active_list) {
                        if (use_active_runs) {
                            cpu_simd::fold_ip_step_active_runs(
                                out, valid_row, rw_ci,
                                active_runs.data(), active_runs.size());
                        } else {
                            cpu_simd::fold_ip_step_active(
                                out, valid_row, rw_ci,
                                active_indices.data(), active_indices.size());
                        }
                    } else {
                        cpu_simd::fold_ip_step(out, valid_row, rw_ci, skip_mask, nc);
                    }
                };
                if (use_opp_active_blocks) {
                    for (const auto& block : opp_active_blocks) {
                        const uint16_t end =
                            static_cast<uint16_t>(block.start + block.count);
                        for (uint16_t ci = block.start; ci < end; ++ci) {
                            apply_fold_ci(ci);
                        }
                    }
                } else {
                    for (uint16_t ci = 0; ci < nc; ++ci) {
                        apply_fold_ci(ci);
                    }
                }
            }
            if (use_active_list) {
                if (use_active_runs) {
                    vec_scale_active_runs(out, self_payoff, active_runs);
                } else {
                    vec_scale_active(out, self_payoff, active_indices);
                }
            } else {
                cpu_simd::vec_scale_in_place(out, self_payoff, nc);
            }
        }
    }

    if (row_stride_ > nc) {
        std::fill(out + nc, out + row_stride_, 0.0f);
    }

#ifdef _OPENMP
    {
        const double _et_dt = omp_get_wtime() - _et_t0;
        const int _et_tid = omp_get_thread_num();
        const uint32_t _et_safe = (_et_tid < 0
            || static_cast<uint32_t>(_et_tid) >= cpu_threads_effective_)
                ? 0u : static_cast<uint32_t>(_et_tid);
        if (tt == TerminalType::SHOWDOWN) {
            showdown_acc_per_thread_[_et_safe] += _et_dt;
        } else {
            fold_acc_per_thread_[_et_safe] += _et_dt;
        }
    }
#endif
}
// ============================================================================
// v1.8.2 Phase 2: fused-kernel showdown for an OOP-traverser group.
// Bypasses the per-call evaluate_terminal path and feeds the batch kernel
// with M terminals' opp_reach_w + payoff coefficients gathered into the
// pre-sized scratch buffers from prepare(). The matrix (cat + valid for
// the group's shared matchup_idx) then streams from L1/L2 once per
// c-row instead of once per terminal call.
// ============================================================================
inline void LevelizedCpuBackend::process_showdown_group_oop(
    const std::vector<uint32_t>& group)
{
    if (group.empty()) return;
    const auto& tree = *ctx_.tree;
    const uint16_t nc = ctx_.iso->num_canonical;
    const std::size_t M = group.size();

    // All members share the same matchup_idx by construction.
    const int32_t mi = (group[0] < tree.matchup_idx.size())
        ? tree.matchup_idx[group[0]] : 0;

    // Resolve matchup tables for this mi. Same fallback rule as
    // evaluate_terminal ??the per-runout vectors are populated when chance
    // enumeration is engaged; otherwise we fall through to the root-board
    // single-table path.
    const auto& matchup_valid_table =
        (ctx_.matchup_valid_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_valid_per_runout->size())
            ? (*ctx_.matchup_valid_per_runout)[mi]
            : *ctx_.matchup_valid;
    const auto& matchup_category_table =
        (ctx_.matchup_category_per_runout && mi >= 0 &&
         static_cast<std::size_t>(mi) < ctx_.matchup_category_per_runout->size())
            ? (*ctx_.matchup_category_per_runout)[mi]
            : *ctx_.matchup_category;
    const float*   valid_ptr = matchup_valid_table.data();
    const uint8_t* cat_ptr   = matchup_category_table.data();

    // Pre-resolve per-terminal pointers (cheap: M scalar copies).
    for (std::size_t t = 0; t < M; ++t) {
        const uint32_t n = group[t];
        batch_reach_ptrs_[t] = batch_opp_reach_w_.data()
                             + t * static_cast<std::size_t>(nc);
        batch_out_ptrs_[t]   = &value_[row_off(n)];
    }

    const uint8_t* skip_mask = oop_has_out_of_range_
        ? oop_out_of_range_mask_.data()
        : nullptr;

    // Charge the entire group's wall time to the showdown phase counter
    // (best-effort ??finer per-thread breakdown isn't needed for the
    // diagnostic and would require timing inside each c-slice).
#ifdef _OPENMP
    const double _bg_t0 = omp_get_wtime();
#endif

    // Single OMP region per group: each thread (a) gathers a slice of
    // terminals' opp_reach_w + payoff coefficients, (b) hits a barrier so
    // every gather is visible, then (c) processes a c-slice of the fused
    // showdown kernel. Both phases parallelize at the same time without
    // paying two separate fork/joins. With 446 monotone groups, that
    // halves the OMP overhead vs the naive "gather then parallel kernel"
    // version and recovers the 1.3 MB-per-group gather cost (was
    // serializing into a ~1s/iter cliff).
#if defined(_OPENMP)
    if (cpu_threads_effective_ > 1) {
        #pragma omp parallel num_threads(static_cast<int>(cpu_threads_effective_))
        {
            const int tid   = omp_get_thread_num();
            const int nthr  = omp_get_num_threads();

            // Phase A: gather. Each thread covers a slice of terminals.
            const std::size_t t_lo = (M * tid) / nthr;
            const std::size_t t_hi = (M * (tid + 1)) / nthr;
            for (std::size_t t = t_lo; t < t_hi; ++t) {
                const uint32_t n = group[t];
                const float pot_total = tree.pots[n];
                float rake = std::min(pot_total * ctx_.config->rake_rate,
                                      ctx_.config->rake_cap);
                if (rake < 0.0f) rake = 0.0f;
                const float half_pot = pot_total * 0.5f;
                batch_win_payoff_[t]  = half_pot - rake;
                batch_lose_payoff_[t] = -half_pot;
                batch_tie_payoff_[t]  = -0.5f * rake;

                float* opp_w_buf = batch_opp_reach_w_.data()
                                 + t * static_cast<std::size_t>(nc);
                const float* opp_reach = &reach_ip_[row_off(n)];
                cpu_simd::vec_copy(opp_w_buf, opp_reach, nc);
                cpu_simd::vec_mul_in_place(opp_w_buf, canonical_weights_f_.data(), nc);
            }

            #pragma omp barrier

            // Phase B: kernel. Each thread processes a slice of the c outer.
            const std::size_t c_lo =
                (static_cast<std::size_t>(nc) * tid) / nthr;
            const std::size_t c_hi =
                (static_cast<std::size_t>(nc) * (tid + 1)) / nthr;
            cpu_simd::showdown_oop_full_batch(
                cat_ptr, valid_ptr,
                static_cast<std::size_t>(nc), M,
                batch_reach_ptrs_.data(), skip_mask, batch_out_ptrs_.data(),
                batch_win_payoff_.data(),  batch_lose_payoff_.data(),
                batch_tie_payoff_.data(),
                c_lo, c_hi);
        }
    } else
#endif
    {
        // Single-thread fallback: gather then call.
        for (std::size_t t = 0; t < M; ++t) {
            const uint32_t n = group[t];
            const float pot_total = tree.pots[n];
            float rake = std::min(pot_total * ctx_.config->rake_rate,
                                  ctx_.config->rake_cap);
            if (rake < 0.0f) rake = 0.0f;
            const float half_pot = pot_total * 0.5f;
            batch_win_payoff_[t]  = half_pot - rake;
            batch_lose_payoff_[t] = -half_pot;
            batch_tie_payoff_[t]  = -0.5f * rake;

            float* opp_w_buf = batch_opp_reach_w_.data()
                             + t * static_cast<std::size_t>(nc);
            const float* opp_reach = &reach_ip_[row_off(n)];
            cpu_simd::vec_copy(opp_w_buf, opp_reach, nc);
            cpu_simd::vec_mul_in_place(opp_w_buf, canonical_weights_f_.data(), nc);
        }
        cpu_simd::showdown_oop_full_batch(
            cat_ptr, valid_ptr,
            static_cast<std::size_t>(nc), M,
            batch_reach_ptrs_.data(), skip_mask, batch_out_ptrs_.data(),
            batch_win_payoff_.data(),  batch_lose_payoff_.data(),
            batch_tie_payoff_.data(),
            0, static_cast<std::size_t>(nc));
    }

    if (row_stride_ > nc) {
        for (std::size_t t = 0; t < M; ++t) {
            std::fill(batch_out_ptrs_[t] + nc,
                      batch_out_ptrs_[t] + row_stride_, 0.0f);
        }
    }

#ifdef _OPENMP
    {
        const double _bg_dt = omp_get_wtime() - _bg_t0;
        // Charge against thread 0 ??the OMP region above stays inside the
        // batch kernel call so we can't easily attribute per-thread cost
        // without instrumenting the kernel internals.
        if (!showdown_acc_per_thread_.empty()) {
            // Multiply by num threads to keep the CPU-seconds semantic
            // consistent with the per-call path's per-thread accumulator.
            showdown_acc_per_thread_[0] += _bg_dt
                * static_cast<double>(cpu_threads_effective_);
        }
    }
#endif
}

// ============================================================================
// Forward pass ??propagate reach top-down, update strategy_sum at
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
    if (row_stride_ > nc) {
        std::fill(reach_oop_.data() + nc, reach_oop_.data() + row_stride_, 0.0f);
        std::fill(reach_ip_.data() + nc,  reach_ip_.data() + row_stride_,  0.0f);
    }

    float pos_unused, neg_unused, sw;
    compute_dcfr_factors(iteration, *ctx_.config, pos_unused, neg_unused, sw);
    const bool decay_and_add = (ctx_.config->dcfr_schedule ==
                                SolverConfig::DcfrSchedule::POSTFLOP_STYLE);

    // Level 0 contains terminals ??no reach propagation from there. Walk
    // levels root ??leaves (decreasing depth, level 1 inclusive).
    if (num_levels_ <= 1) return;

    // Per-node body, shared by both OMP variants. Captures by reference so
    // the inner SIMD-kernel calls can stay on hot paths; MSVC inlines this
    // through the lambda for both call sites in measurement.
    auto process_node = [&](uint32_t n) {
        auto nt = static_cast<NodeType>(tree.node_types[n]);
        if (nt == NodeType::TERMINAL) return;

        const float* parent_oop = &reach_oop_[row_off(n)];
        const float* parent_ip  = &reach_ip_[row_off(n)];

        if (nt == NodeType::CHANCE) {
            // Reach passes through chance unchanged; EV averaging happens
            // on the backward pass.
            uint8_t nch = tree.num_children[n];
            uint32_t off = tree.children_offset[n];
            for (uint8_t k = 0; k < nch; ++k) {
                uint32_t child = tree.children[off + k];
                float* child_oop = &reach_oop_[row_off(child)];
                float* child_ip  = &reach_ip_[row_off(child)];
                if (oop_use_sparse_traversal_) {
                    sparse_copy(child_oop, parent_oop, 0);
                } else if (use_block_traversal_for_player(0)) {
                    block_copy(child_oop, parent_oop, 0);
                } else {
                    std::memcpy(child_oop, parent_oop, sizeof(float) * row_stride_);
                }
                if (ip_use_sparse_traversal_) {
                    sparse_copy(child_ip, parent_ip, 1);
                } else if (use_block_traversal_for_player(1)) {
                    block_copy(child_ip, parent_ip, 1);
                } else {
                    std::memcpy(child_ip, parent_ip, sizeof(float) * row_stride_);
                }
            }
            return;
        }

        // PLAYER_OOP / PLAYER_IP
        int acting = tree.active_player[n];
        uint8_t na = tree.num_children[n];
        if (na == 0) return;
        const float* strat = current_strategy_ptr(n);
        const std::size_t stride = action_stride_;

        // Strategy_sum update at this node ??uses reach of the ACTING
        // player. Reference backend does this only at acting==traverser
        // nodes, but since we run a backward pass per traverser and
        // strategy_sum entries are disjoint per acting player, doing it
        // once during the shared forward pass is equivalent and saves a
        // pass.
        const float* acting_reach = (acting == 0) ? parent_oop : parent_ip;
        if (decay_and_add) {
            for (uint8_t a = 0; a < na; ++a) {
                cpu_simd::vec_decay_add(
                    strategy_sum_ptr(n, a),
                    sw,
                    strat + static_cast<std::size_t>(a) * stride,
                    stride);
            }
        } else if (use_sparse_traversal_for_player(acting)) {
            for (uint8_t a = 0; a < na; ++a) {
                const float* strat_a = strat + static_cast<std::size_t>(a) * stride;
                sparse_reach_weighted_strat_sum(
                    strategy_sum_ptr(n, a), sw, acting_reach, strat_a, acting);
            }
        } else if (use_block_strategy_sum_for_player(acting)) {
            for (uint8_t a = 0; a < na; ++a) {
                const float* strat_a = strat + static_cast<std::size_t>(a) * stride;
                block_reach_weighted_strat_sum(
                    strategy_sum_ptr(n, a), sw, acting_reach, strat_a, acting);
            }
        } else {
            for (uint8_t a = 0; a < na; ++a) {
                cpu_simd::vec_reach_weighted_strat_sum(
                    strategy_sum_ptr(n, a),
                    sw,
                    acting_reach,
                    strat + static_cast<std::size_t>(a) * stride,
                    stride);
            }
        }

        // Reach propagation to children.
        //   Child's acting-player-reach = parent's acting reach ? strat[a]
        //   Child's other reach         = parent's other reach (unchanged)
        uint32_t off = tree.children_offset[n];
        for (uint8_t a = 0; a < na; ++a) {
            uint32_t child = tree.children[off + a];
            float* child_oop = &reach_oop_[row_off(child)];
            float* child_ip  = &reach_ip_[row_off(child)];
            const float* strat_a = strat + static_cast<std::size_t>(a) * stride;
            if (acting == 0) {
                if (oop_use_sparse_traversal_) {
                    sparse_mul(child_oop, parent_oop, strat_a, 0);
                } else if (use_block_traversal_for_player(0)) {
                    block_mul(child_oop, parent_oop, strat_a, 0);
                } else {
                    cpu_simd::vec_mul(child_oop, parent_oop, strat_a, stride);
                }
                if (ip_use_sparse_traversal_) {
                    sparse_copy(child_ip, parent_ip, 1);
                } else if (use_block_traversal_for_player(1)) {
                    block_copy(child_ip, parent_ip, 1);
                } else {
                    std::memcpy(child_ip, parent_ip, sizeof(float) * row_stride_);
                }
            } else {
                if (ip_use_sparse_traversal_) {
                    sparse_mul(child_ip, parent_ip, strat_a, 1);
                } else if (use_block_traversal_for_player(1)) {
                    block_mul(child_ip, parent_ip, strat_a, 1);
                } else {
                    cpu_simd::vec_mul(child_ip, parent_ip, strat_a, stride);
                }
                if (oop_use_sparse_traversal_) {
                    sparse_copy(child_oop, parent_oop, 0);
                } else if (use_block_traversal_for_player(0)) {
                    block_copy(child_oop, parent_oop, 0);
                } else {
                    std::memcpy(child_oop, parent_oop, sizeof(float) * row_stride_);
                }
            }
        }
    };

#if defined(_OPENMP)
    // v1.8.0 Sprint 3: persistent OMP team variant. Single fork
    // wraps the level loop; `omp for` distributes nodes within each
    // level. Implicit barrier at end of each `omp for` keeps the
    // parent-before-child invariant intact (no `nowait`). Saves K-1
    // team creations per pass (K = num_levels) on big trees where the
    // per-level fork overhead measurably hurts scaling.
    if (ctx_.config != nullptr && ctx_.config->cpu_persistent_omp
        && cpu_threads_effective_ > 1) {
        #pragma omp parallel num_threads(static_cast<int>(cpu_threads_effective_))
        {
            for (int64_t L = static_cast<int64_t>(num_levels_) - 1; L >= 1; --L) {
                const uint32_t lo = level_offsets_[L];
                const uint32_t hi = level_offsets_[L + 1];
                #pragma omp for schedule(dynamic, 8)
                for (int64_t idx = static_cast<int64_t>(lo);
                     idx < static_cast<int64_t>(hi); ++idx)
                {
                    process_node(node_order_[static_cast<std::size_t>(idx)]);
                }
                // implicit barrier here is required for level dependency
            }
        }
        return;
    }
#endif

    // Default: per-level parallel-for fork. Same shape as v1.5.0 ??kept
    // as the safe baseline until persistent-team variant has clean
    // paired-benchmark evidence.
    for (int64_t L = static_cast<int64_t>(num_levels_) - 1; L >= 1; --L) {
        const uint32_t lo = level_offsets_[L];
        const uint32_t hi = level_offsets_[L + 1];

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 8) num_threads(static_cast<int>(cpu_threads_effective_))
        #endif
        for (int64_t idx = static_cast<int64_t>(lo);
             idx < static_cast<int64_t>(hi); ++idx)
        {
            process_node(node_order_[static_cast<std::size_t>(idx)]);
        }
    }
}

// ============================================================================
// Backward pass for one traverser. Walks levels 0 ??max_depth.
// ============================================================================

inline void LevelizedCpuBackend::backward_pass(int traverser) {
    const auto& tree = *ctx_.tree;
    const uint32_t N = tree.total_nodes;
    if (N == 0) return;
    const bool sparse_value = use_sparse_traversal_for_player(traverser);
    const bool block_value = use_block_traversal_for_player(traverser);
    const bool sparse_opp_reach =
        use_sparse_opp_reach_build_for_traverser(traverser);
    if (sparse_opp_reach) {
        cpu_simd::vec_set_zero(
            terminal_opp_reach_w_.data(),
            terminal_opp_reach_w_.size());
    }

    // Per-node body, shared by both OMP variants. See forward_pass for
    // why this lambda pattern preserves both correctness (read-only
    // captures + per-node disjoint writes) and codegen (MSVC inlines).
    auto process_node = [&](uint32_t n) {
        auto nt = static_cast<NodeType>(tree.node_types[n]);
        float* out = &value_[row_off(n)];

        if (nt == NodeType::TERMINAL) {
            evaluate_terminal(n, traverser, out);
            return;
        }

        if (nt == NodeType::CHANCE) {
            uint8_t nch = tree.num_children[n];
            if (nch == 0) {
                if (sparse_value) {
                    sparse_set_zero(out, traverser);
                } else if (block_value) {
                    block_set_zero(out, traverser);
                } else {
                    cpu_simd::vec_set_zero(out, row_stride_);
                }
                return;
            }
            if (sparse_value) {
                sparse_set_zero(out, traverser);
            } else if (block_value) {
                block_set_zero(out, traverser);
            } else {
                cpu_simd::vec_set_zero(out, row_stride_);
            }
            uint32_t total_weight = 0;
            uint32_t off = tree.children_offset[n];
            for (uint8_t k = 0; k < nch; ++k) {
                uint32_t child = tree.children[off + k];
                uint32_t weight = (child < tree.runout_weight.size())
                                    ? tree.runout_weight[child] : 1;
                if (weight == 0) weight = 1;
                if (sparse_value) {
                    sparse_axpy(
                        out, static_cast<float>(weight),
                        &value_[row_off(child)], traverser);
                } else if (block_value) {
                    block_axpy(
                        out, static_cast<float>(weight),
                        &value_[row_off(child)], traverser);
                } else {
                    cpu_simd::vec_axpy(
                        out, static_cast<float>(weight),
                        &value_[row_off(child)], row_stride_);
                }
                total_weight += weight;
            }
            if (total_weight > 0) {
                if (sparse_value) {
                    sparse_scale(
                        out, 1.0f / static_cast<float>(total_weight),
                        traverser);
                } else if (block_value) {
                    block_scale(
                        out, 1.0f / static_cast<float>(total_weight),
                        traverser);
                } else {
                    cpu_simd::vec_scale_in_place(
                        out, 1.0f / static_cast<float>(total_weight), row_stride_);
                }
            }
            return;
        }

        // Player decision node.
        int acting = tree.active_player[n];
        uint8_t na = tree.num_children[n];
        if (na == 0) {
            if (sparse_value) {
                sparse_set_zero(out, traverser);
            } else if (block_value) {
                block_set_zero(out, traverser);
            } else {
                cpu_simd::vec_set_zero(out, row_stride_);
            }
            return;
        }
        const float* strat = current_strategy_ptr(n);
        const std::size_t stride = action_stride_;
        uint32_t off = tree.children_offset[n];

        if (acting == traverser) {
            // node_val = Σ_a strat[a] · child_val[a]
            if (sparse_value) {
                sparse_set_zero(out, traverser);
            } else if (block_value) {
                block_set_zero(out, traverser);
            } else {
                cpu_simd::vec_set_zero(out, row_stride_);
            }
            for (uint8_t a = 0; a < na; ++a) {
                uint32_t child = tree.children[off + a];
                const float* strat_a = strat + static_cast<std::size_t>(a) * stride;
                if (sparse_value) {
                    sparse_fmadd(
                        out,
                        strat_a,
                        &value_[row_off(child)],
                        traverser);
                } else if (block_value) {
                    block_fmadd(
                        out,
                        strat_a,
                        &value_[row_off(child)],
                        traverser);
                } else {
                    cpu_simd::vec_fmadd(
                        out,
                        strat_a,
                        &value_[row_off(child)],
                        stride);
                }
            }
            // regret[a, c] += child_val[a, c] ??node_val[c]
            for (uint8_t a = 0; a < na; ++a) {
                uint32_t child = tree.children[off + a];
                if (sparse_value) {
                    sparse_regret_update(
                        regret_ptr(n, a),
                        &value_[row_off(child)],
                        out,
                        traverser);
                } else if (block_value) {
                    block_regret_update(
                        regret_ptr(n, a),
                        &value_[row_off(child)],
                        out,
                        traverser);
                } else {
                    cpu_simd::vec_regret_update(
                        regret_ptr(n, a),
                        &value_[row_off(child)],
                        out,
                        stride);
                }
            }
        } else {
            // Opp node: opp's strat is already baked into reach by the
            // forward pass, so just sum the children's values.
            if (sparse_value) {
                sparse_set_zero(out, traverser);
            } else if (block_value) {
                block_set_zero(out, traverser);
            } else {
                cpu_simd::vec_set_zero(out, row_stride_);
            }
            for (uint8_t a = 0; a < na; ++a) {
                uint32_t child = tree.children[off + a];
                if (sparse_value) {
                    sparse_add(out, &value_[row_off(child)], traverser);
                } else if (block_value) {
                    block_add(out, &value_[row_off(child)], traverser);
                } else {
                    cpu_simd::vec_add_in_place(
                        out, &value_[row_off(child)], row_stride_);
                }
            }
        }
    };

#if defined(_OPENMP)
    // v1.8.0 Sprint 3: persistent OMP team variant. Same
    // structural argument as forward_pass ??single fork wraps the
    // level loop, implicit barrier between `omp for` iterations
    // preserves the child-before-parent dependency that the backward
    // pass relies on (parent reads child `value_[row_off(child)]`).
    if (ctx_.config != nullptr && ctx_.config->cpu_persistent_omp
        && cpu_threads_effective_ > 1) {
        #pragma omp parallel num_threads(static_cast<int>(cpu_threads_effective_))
        {
            for (uint32_t L = 0; L < num_levels_; ++L) {
                const uint32_t lo = level_offsets_[L];
                const uint32_t hi = level_offsets_[L + 1];
                #pragma omp for schedule(dynamic, 8)
                for (int64_t idx = static_cast<int64_t>(lo);
                     idx < static_cast<int64_t>(hi); ++idx)
                {
                    process_node(node_order_[static_cast<std::size_t>(idx)]);
                }
                // implicit barrier here is required for level dependency
            }
        }
        return;
    }
#endif

    // v1.8.2 Phase 2 experiment: kernel-level batching (process_showdown_group_oop)
    // was implemented but disabled ??measured regression on monotone (-13%
    // 8T) outweighed the standard win (+4% noise). Root cause: each per-t
    // inner-loop iteration reintroduces memory traffic via acc_buf
    // load/store, which cancels the matrix-row-reuse savings the batch
    // shape was supposed to capture. Natural BFS terminal order already
    // covers most of the available cache locality. The kernel + parity
    // test stay so a future revisit (different CPU / workload / acc-in-
    // register variant) can re-enable with one line.
    for (uint32_t L = 0; L < num_levels_; ++L) {
        const uint32_t lo = level_offsets_[L];
        const uint32_t hi = level_offsets_[L + 1];

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 8) num_threads(static_cast<int>(cpu_threads_effective_))
        #endif
        for (int64_t idx = static_cast<int64_t>(lo);
             idx < static_cast<int64_t>(hi); ++idx)
        {
            process_node(node_order_[static_cast<std::size_t>(idx)]);
        }
    }
}

// ============================================================================
// iterate: one DCFR iteration.
// ============================================================================

inline void LevelizedCpuBackend::iterate(int iteration) {
    // v1.8.1+ phase timing. omp_get_wtime resolution is ~µs on Windows /
    // sub-µs on Linux; per-iter cost is negligible. The if-guards on
    // _OPENMP keep the build clean if anyone disables OMP (then phase
    // timers stay at 0 ??caller treats that as "unavailable").
    auto now = []() -> double {
        #if defined(_OPENMP)
        return omp_get_wtime();
        #else
        return 0.0;
        #endif
    };

    const double t0 = now();
    compute_strategy();
    const double t1 = now();
    apply_dcfr_discount(iteration);
    const double t2 = now();

    // Single forward pass updates reach + strategy_sum for both traversers.
    forward_pass(iteration);
    const double t3 = now();

    // Backward passes ??one per traverser. They write to disjoint slices
    // of regrets_ (acting==traverser nodes are different sets), so they
    // could in principle run on separate threads, but each backward pass
    // is already heavily multi-threaded inside via the per-level OMP. We
    // keep them serial to avoid oversubscribing the thread pool.
    backward_pass(0);
    const double t4 = now();
    backward_pass(1);
    const double t5 = now();

    phase_compute_strategy_ms_  += (t1 - t0) * 1000.0;
    phase_apply_discount_ms_    += (t2 - t1) * 1000.0;
    phase_forward_pass_ms_      += (t3 - t2) * 1000.0;
    phase_backward_pass_oop_ms_ += (t4 - t3) * 1000.0;
    phase_backward_pass_ip_ms_  += (t5 - t4) * 1000.0;
}

// ============================================================================
// finalize ??same as CpuBackend; normalize strategy_sum into strategy.
// ============================================================================

inline void LevelizedCpuBackend::finalize() {
    const uint16_t nc = ctx_.iso->num_canonical;
    const uint32_t N = ctx_.tree->total_nodes;
    strategy_.assign(N, {});

    auto finalize_node = [&](uint32_t i) {
        uint8_t na = ctx_.tree->num_children[i];
        strategy_[i].assign(static_cast<std::size_t>(na) * nc, 0.0f);
        const std::size_t stride = action_stride_;
        const float* sum_base = strategy_sum_ptr(i);
        for (uint16_t c = 0; c < nc; ++c) {
            float total = 0.0f;
            for (uint8_t a = 0; a < na; ++a) {
                total += sum_base[static_cast<std::size_t>(a) * stride + c];
            }
            if (total > 1e-7f) {
                float inv = 1.0f / total;
                for (uint8_t a = 0; a < na; ++a) {
                    strategy_[i][static_cast<std::size_t>(a) * nc + c] =
                        sum_base[static_cast<std::size_t>(a) * stride + c] * inv;
                }
            } else {
                float uniform = 1.0f / na;
                for (uint8_t a = 0; a < na; ++a) {
                    strategy_[i][static_cast<std::size_t>(a) * nc + c] = uniform;
                }
            }
        }
    };

#if defined(_OPENMP)
    if (cpu_threads_effective_ > 1 && player_nodes_.size() >= 4096) {
        #pragma omp parallel for schedule(static) num_threads(static_cast<int>(cpu_threads_effective_))
        for (int64_t idx = 0; idx < static_cast<int64_t>(player_nodes_.size()); ++idx) {
            finalize_node(player_nodes_[static_cast<std::size_t>(idx)]);
        }
        return;
    }
#endif

    for (uint32_t i : player_nodes_) {
        finalize_node(i);
    }
}

}  // namespace deepsolver
