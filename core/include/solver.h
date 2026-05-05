/**
 * @file solver.h
 * @brief Host-side solver orchestrator.
 *
 * The Solver class:
 *   1. Builds the game tree and computes suit isomorphism
 *   2. Precomputes the showdown matchup matrix
 *   3. Initializes per-combo reach probabilities
 *   4. Resolves node locks
 *   5. Delegates DCFR iteration to an ISolverBackend (CPU or GPU)
 *   6. Runs post-solve CPU passes: per-combo EV and exploitability (BR)
 *   7. Extracts and returns results
 *
 * The backend surface is intentionally small — only the DCFR iteration
 * hot path goes through it. EV and BR post-passes are always CPU-side
 * (cheap compared to hundreds of iterations).
 */

#pragma once

#include "types.h"
#include "card.h"
#include "hand_evaluator.h"
#include "game_tree_builder.h"
#include "isomorphism.h"
#include "solver_backend.h"
#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace deepsolver {

// ============================================================================
// Progress callback
// ============================================================================

using ProgressCallback = std::function<void(int iteration, float exploitability, float elapsed_ms)>;

namespace detail {

inline uint32_t effective_postsolve_threads(uint32_t requested) {
    uint32_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    uint32_t limit = (requested == 0) ? hw : requested;
    return std::max<uint32_t>(1, std::min<uint32_t>(limit, hw));
}

class PostsolveThreadPool {
public:
    explicit PostsolveThreadPool(uint32_t thread_count) {
        thread_count = std::max<uint32_t>(1, thread_count);
        workers_.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~PostsolveThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    PostsolveThreadPool(const PostsolveThreadPool&) = delete;
    PostsolveThreadPool& operator=(const PostsolveThreadPool&) = delete;

    uint32_t thread_count() const {
        return static_cast<uint32_t>(workers_.size());
    }

    template <typename F>
    auto enqueue(F&& f) -> std::future<std::invoke_result_t<F>> {
        using ReturnT = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<ReturnT()>>(std::forward<F>(f));
        auto fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("postsolve thread pool is stopping");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

inline PostsolveThreadPool& postsolve_pool() {
    static PostsolveThreadPool pool(effective_postsolve_threads(0));
    return pool;
}

template <typename Fn>
inline void postsolve_parallel_for(
    uint32_t total,
    bool enabled,
    uint32_t requested_threads,
    Fn&& fn)
{
    if (!enabled || total < 32) {
        for (uint32_t i = 0; i < total; ++i) fn(i);
        return;
    }

    auto& pool = postsolve_pool();
    uint32_t threads = effective_postsolve_threads(requested_threads);
    threads = std::min<uint32_t>(threads, pool.thread_count());
    threads = std::min<uint32_t>(threads, total);
    if (threads <= 1) {
        for (uint32_t i = 0; i < total; ++i) fn(i);
        return;
    }

    uint32_t chunk = (total + threads - 1) / threads;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (uint32_t begin = 0; begin < total; begin += chunk) {
        uint32_t end = std::min<uint32_t>(total, begin + chunk);
        futures.emplace_back(pool.enqueue([begin, end, &fn]() {
            for (uint32_t i = begin; i < end; ++i) fn(i);
        }));
    }
    for (auto& fut : futures) fut.get();
}

} // namespace detail

inline bool should_auto_select_gpu(
    const SolverConfig& config,
    const FlatGameTree& tree,
    size_t matchup_table_count)
{
    if (!GPU_BACKEND_FUNCTIONAL || !has_cuda_gpu()) return false;

    (void)matchup_table_count;

    // With the current CUDA backend, real solves on turn/river spots already
    // beat the CPU path on supported NVIDIA hardware. Keep AUTO on CPU only
    // for diagnostics/no-iteration calls where GPU upload would be pure fixed
    // overhead. Explicit --backend cpu remains available for parity testing.
    if (config.max_iterations <= 0) return false;
    if (tree.total_nodes < 128 && config.max_iterations < 25) return false;

    return true;
}

// ============================================================================
// Solver (orchestrator)
// ============================================================================

class Solver {
public:
    explicit Solver(const SolverConfig& config,
                    BackendType backend_type = BackendType::AUTO);
    ~Solver() = default;

    /// Run the full solve. Returns final result.
    SolverResult solve(ProgressCallback progress_cb = nullptr);

    /// Analyze a specific target combo after solve()
    ComboAnalysis analyze_combo(const std::string& combo_str) const;

    uint32_t tree_node_count() const { return tree_.total_nodes; }
    uint32_t navigate_to_node(const std::string& history) const;
    std::vector<std::pair<std::string, float>> extract_global_strategy_at(uint32_t node_idx) const;
    std::vector<std::string> get_action_labels_at(uint32_t node_idx) const;

    /// Per-grid-label strategy (e.g. "AA" → {"Check": 0.62, "Bet_33": 0.38}).
    /// Aggregates all canonical combos that share a grid label, weighted by the
    /// acting player's reach so hands outside the range (weight 0) are ignored.
    /// Each action frequency is in [0, 1] (NOT a percentage, matching the UI's
    /// ComboStrategy = Record<string, number> contract).
    std::vector<std::pair<std::string,
        std::vector<std::pair<std::string, float>>>>
        extract_combo_strategies_at(uint32_t node_idx) const;

    /// Extract the opponent's (non-acting player's) range at a given decision
    /// node, computed as reach probability propagated from root through the
    /// path taken to reach node_idx.
    /// Returns list of (grid_label, weight) pairs where weight is the max
    /// reach across the label's canonical slots, normalized to the largest
    /// label weight at this node (so the heaviest label = 1.0).
    /// Also returns who the "opponent" is so the UI can label it.
    struct OpponentRangeResult {
        std::string opponent;  // "OOP" or "IP"
        std::vector<std::pair<std::string, float>> labels;  // sorted by weight desc
    };
    OpponentRangeResult extract_opponent_range_at(uint32_t node_idx) const;

    /// Which player is acting at this node ("OOP" | "IP" | "" if not a
    /// player-decision node).
    std::string acting_player_at(uint32_t node_idx) const;

    /// One canonical runout representative for the runout-selector UI.
    struct RunoutOption {
        uint8_t card;     // 0..51 (rank * 4 + suit)
        uint8_t weight;   // orbit size (sum across reps = undealt card count)
    };

    /// One node entry in the cached strategy tree. Mirrors the per-node
    /// fields of SolverResult so the frontend can render a node from this
    /// alone, no extra solve required.
    struct StrategyTreeEntry {
        std::string acting;                                  // "OOP" | "IP"
        std::vector<std::string> action_labels;
        std::vector<std::pair<std::string, float>> global_strategy;
        // Per-grid-label only (not per-specific-combo); keeps cache size
        // manageable. If the UI needs per-variant strategy at some node it
        // can still trigger an explicit solve.
        std::vector<std::pair<std::string,
            std::vector<std::pair<std::string, float>>>> combo_strategies;
        std::string opponent_side;                           // "OOP" | "IP"
        std::vector<std::pair<std::string, float>> opponent_range;

        // Per-grid-label EV (in chips, from acting player's perspective)
        // at THIS node. Aggregated across canonical slots weighted by reach.
        // Empty for terminal/chance fallthrough nodes.
        std::vector<std::pair<std::string, float>> combo_evs;

        // Path B (per-runout cache):
        // Cumulative cards dealt from root to this node via the path's
        // chance-skip jumps. Empty if no chance preceded this node. Lets the
        // UI display "Strategies for runout: 2♣" instead of silently showing
        // lex-min canonical's strategies.
        std::vector<uint8_t> dealt_cards;
        // If the IMMEDIATE PRIOR chance (the chance step that produced this
        // sub-tree's root) had multiple canonical runouts, list them all so
        // the UI can render a runout picker. Empty means no choice was
        // available (rainbow degraded, terminal, or no chance preceded).
        // The currently-selected runout is the LAST entry of `dealt_cards`.
        std::vector<RunoutOption> runout_options;
    };

    /// DFS the solved tree from root, keyed by player-action history (the
    /// same string format `navigate_to_node` accepts). Auto-skips CHANCE to
    /// first child (lex-min canonical runout) so paths only contain player
    /// actions, matching what the UI sends back. Limited to
    /// `max_player_depth` consecutive player decisions to bound output size.
    ///
    /// **CACHE-HONESTY CAVEAT**: when iso enumeration is engaged (typically
    /// monotone/two-tone/3-of-suit boards), the chance auto-skip lands on
    /// the SMALLEST canonical card (e.g. 2♣ over 2♦ over 2♥ over 2♠). The
    /// strategies cached for any post-chance node are therefore the
    /// strategies on THAT specific runout, not an average over the orbit.
    /// On rainbow boards (single-child fallback) the cached strategies use
    /// the root board (no per-runout equity) — the same as before Phase 2.
    /// UI consumers that show a board image with the cached strategies must
    /// either (a) display the actual lex-min runout cards, or (b) trigger a
    /// fresh solve when the user wants strategies for a specific runout.
    ///
    /// Phase 3 (10-point plan) — `StrategyTreeEvMode` controls how much memory
    /// the EV cache spends. `VISIBLE` (recommended default) only computes EVs
    /// for the nodes actually emitted in the strategy tree, instead of every
    /// inner node. `NONE` skips EV computation entirely. `FULL` is the legacy
    /// behavior — caches every visited node × 2 perspectives, which can blow
    /// host RAM on deep multi-bet trees (this is the `solver.h:1739-1745` flag
    /// the maturity plan called out).
    enum class StrategyTreeEvMode : uint8_t {
        NONE    = 0, ///< Don't compute combo EVs. Smallest response.
        VISIBLE = 1, ///< Compute EVs only for emitted nodes (default).
        FULL    = 2, ///< Cache every visited node (legacy behavior).
    };

    /// Sprint 1 (market-beating plan): `max_nodes` caps the emitted set.
    /// 0 = use `config.memory_budget.strategy_tree_max_nodes`. When the cap
    /// is hit, the walker returns without descending deeper into the
    /// remaining branches and `*out_truncated` (if non-null) is set to true
    /// so the caller can surface the truncation in the response payload.
    std::map<std::string, StrategyTreeEntry>
        build_strategy_tree(int max_player_depth = 8,
                            StrategyTreeEvMode ev_mode = StrategyTreeEvMode::VISIBLE,
                            uint32_t max_nodes = 0,
                            bool* out_truncated = nullptr) const;

    /// Backward-compat overload — `include_combo_evs=true` ⇒ VISIBLE,
    /// `false` ⇒ NONE. Existing call sites keep working.
    std::map<std::string, StrategyTreeEntry>
        build_strategy_tree(int max_player_depth, bool include_combo_evs) const {
        return build_strategy_tree(max_player_depth,
            include_combo_evs ? StrategyTreeEvMode::VISIBLE
                              : StrategyTreeEvMode::NONE);
    }

    /// Name of the active backend (for logging / UI indicator)
    const char* backend_name() const { return backend_ ? backend_->name() : "none"; }

private:
    SolverConfig config_;
    BackendType  backend_type_;
    FlatGameTree tree_;
    IsomorphismMapping iso_;
    bool solved_ = false;
    int actual_iterations_run_ = 0;

    // Final results (populated after solve)
    std::vector<std::vector<float>> strategy_;
    std::vector<float> ev_;
    float exploitability_pct_ = 0.0f;

    // Precomputed (shared between backend and post-solve passes)
    std::vector<float> ip_reach_;
    std::vector<float> oop_reach_;
    /// Legacy single matchup table — kept as the table for `matchup_idx == 0`
    /// (i.e. the root board). All other runouts live in matchup_ev_per_runout_.
    std::vector<float> matchup_ev_;
    std::vector<float> matchup_valid_;
    /// Phase 1 chance-enumeration: per-runout matchup tables. Indexed by
    /// FlatGameTree.matchup_idx[node]. Entry [0] mirrors matchup_ev_.
    std::vector<std::vector<float>> matchup_ev_per_runout_;
    std::vector<std::vector<float>> matchup_valid_per_runout_;
    std::map<std::pair<uint32_t, uint16_t>, std::vector<float>> resolved_locks_;

    // Backend (CPU or GPU) handles DCFR iteration
    std::unique_ptr<ISolverBackend> backend_;

    // ---- Setup (CPU, shared by any backend) ----
    void initialize_reach_probs();
    void resolve_node_locks();
    void apply_node_locks();
    void precompute_matchups();

    // ---- Post-solve CPU passes (operate on strategy_) ----
    float compute_exploitability();
    std::vector<float> cpu_best_response_traverse(uint32_t node_idx, int player,
                                                   std::vector<float>& reach_oop,
                                                   std::vector<float>& reach_ip) const;
    std::vector<float> cpu_ev_traverse(uint32_t node_idx, int perspective,
                                        std::vector<float>& reach_oop,
                                        std::vector<float>& reach_ip,
                                        std::map<uint32_t, std::vector<float>>* out_node_values = nullptr,
                                        const std::set<uint32_t>* visible_filter = nullptr) const;
    void compute_combo_evs();

    // ---- Helpers ----
    uint16_t evaluate_combo(const Combo& combo) const;
    static std::string action_to_label(ActionType type, float amount, float pot);
    /// Trivial accessor used by build_strategy_tree's lambda — wraps
    /// tree_.children_offset[node] so the walk reads cleaner.
    uint32_t off_of(uint32_t node) const;
};

// ============================================================================
// Constructor / helpers
// ============================================================================

inline Solver::Solver(const SolverConfig& config, BackendType backend_type)
    : config_(config), backend_type_(backend_type) {
    auto& eval = get_evaluator();
    if (!eval.is_initialized()) eval.initialize();
}

inline std::string Solver::action_to_label(ActionType type, float amount, float pot) {
    switch (type) {
        case ActionType::FOLD:  return "Fold";
        case ActionType::CHECK: return "Check";
        case ActionType::CALL:  return "Call";
        case ActionType::ALLIN: return "All-in";
        case ActionType::BET:
        case ActionType::RAISE: {
            float pct = (pot > 0) ? (amount / pot * 100.0f) : 0;
            std::ostringstream oss;
            oss << ((type == ActionType::BET) ? "Bet_" : "Raise_")
                << static_cast<int>(pct);
            return oss.str();
        }
        default: return "Unknown";
    }
}

inline uint16_t Solver::evaluate_combo(const Combo& combo) const {
    auto& eval = get_evaluator();
    Card c0 = combo.cards[0], c1 = combo.cards[1];
    if (config_.board_size == 5) {
        return eval.evaluate(c0, c1, config_.board[0], config_.board[1],
                             config_.board[2], config_.board[3], config_.board[4]);
    } else if (config_.board_size == 4) {
        Card cards[6] = {c0, c1, config_.board[0], config_.board[1],
                         config_.board[2], config_.board[3]};
        uint16_t best = UINT16_MAX;
        for (int skip = 0; skip < 6; ++skip) {
            Card h[5]; int idx = 0;
            for (int j = 0; j < 6; ++j) if (j != skip) h[idx++] = cards[j];
            best = std::min(best, eval.evaluate5(h[0], h[1], h[2], h[3], h[4]));
        }
        return best;
    } else {
        return eval.evaluate5(c0, c1, config_.board[0], config_.board[1], config_.board[2]);
    }
}

// ============================================================================
// Solve entry point
// ============================================================================

inline SolverResult Solver::solve(ProgressCallback progress_cb) {
    using Clock = std::chrono::high_resolution_clock;

    auto start_time = Clock::now();
    auto stage_start = start_time;
    SolverTiming timing;
    timing.postsolve_threads = config_.parallel_postsolve
        ? detail::effective_postsolve_threads(config_.postsolve_threads)
        : 1;
    actual_iterations_run_ = 0;

    auto elapsed_since = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<float, std::milli>(end - begin).count();
    };

    // Phase 2 (10-point plan): isomorphism first, so the tree builder can
    // make a byte-based runout-enumeration decision (needs nc to estimate
    // matchup table size). compute_isomorphism is pure and only reads the
    // flop/turn/river board cards already in config_, so reordering is safe.
    stage_start = Clock::now();
    iso_ = compute_isomorphism(config_.board.data(), config_.board_size);
    auto stage_end = Clock::now();
    timing.isomorphism_ms = elapsed_since(stage_start, stage_end);

    // Step 1-2: tree (now with memory-budget-aware runout cap)
    stage_start = Clock::now();
    GameTreeBuilder builder(config_);
    builder.set_memory_policy(iso_.num_canonical, config_.memory_budget);
    tree_ = builder.build();
    stage_end = Clock::now();
    timing.tree_build_ms = elapsed_since(stage_start, stage_end);
    timing.tree_nodes = tree_.total_nodes;
    timing.tree_edges = tree_.total_edges;

    // Step 3: precompute matchup matrix + reach probs + node locks
    stage_start = Clock::now();
    precompute_matchups();
    stage_end = Clock::now();
    timing.precompute_matchups_ms = elapsed_since(stage_start, stage_end);
    timing.matchup_tables = static_cast<uint32_t>(matchup_ev_per_runout_.size());

    stage_start = Clock::now();
    initialize_reach_probs();
    stage_end = Clock::now();
    timing.reach_init_ms = elapsed_since(stage_start, stage_end);

    stage_start = Clock::now();
    resolve_node_locks();
    stage_end = Clock::now();
    timing.node_locks_ms = elapsed_since(stage_start, stage_end);

    // ----------------------------------------------------------------
    // Sprint 1 (resource policy guide): pre-backend budget gate.
    // This is the single point where we say "we will / won't / can't"
    // BEFORE the backend allocates the regrets / strategy_sum / device
    // arrays that dominate per-iteration memory. Three outcomes:
    //   - OK         → keep selected_backend, proceed to prepare()
    //   - downgrade  → AUTO+GPU → CPU, record fallback_reason
    //   - reject     → throw structured runtime_error (main.cpp emits JSON)
    // ----------------------------------------------------------------
    SolveFootprintEstimate gate_est;
    {
        const uint64_t nc      = iso_.num_canonical;
        const uint64_t total_n = tree_.total_nodes;
        uint32_t player_nodes = 0;
        for (uint32_t n = 0; n < tree_.total_nodes; ++n) {
            auto t = static_cast<NodeType>(tree_.node_types[n]);
            if (t == NodeType::PLAYER_OOP || t == NodeType::PLAYER_IP) ++player_nodes;
        }
        gate_est.matchup_tables_bytes   = bytes_for_matchup_tables(
            matchup_ev_per_runout_.size(), nc);
        gate_est.cpu_state_bytes        = bytes_for_cpu_state(player_nodes, MAX_ACTIONS, nc);
        gate_est.gpu_state_bytes        = bytes_for_gpu_state(total_n, MAX_ACTIONS, nc);
        gate_est.strategy_tree_ev_bytes = bytes_for_strategy_tree_ev_cache(player_nodes, nc);
        gate_est.json_response_bytes    = bytes_for_json_response(
            std::min<uint64_t>(player_nodes,
                               config_.memory_budget.strategy_tree_max_nodes));
    }

    std::string fallback_reason;  // surfaced into result.resources below.

    // Step 4: create backend. AUTO uses a size-aware heuristic because the
    // CUDA path has a meaningful fixed cost on small/medium solves.
    stage_start = Clock::now();
    BackendType selected_backend = backend_type_;
    if (backend_type_ == BackendType::AUTO) {
        selected_backend = should_auto_select_gpu(
            config_, tree_, matchup_ev_per_runout_.size())
            ? BackendType::GPU
            : BackendType::CPU;
    }

    // GPU budget check. `gpu_bytes == 0` means "let backend probe at
    // runtime" — we don't reject in that case (host can't predict free
    // VRAM accurately for arbitrary devices). When a budget IS set:
    //   - Explicit `--backend gpu` + over budget → reject hard.
    //   - AUTO + over budget → downgrade to CPU, record reason.
    if (selected_backend == BackendType::GPU &&
        config_.memory_budget.gpu_bytes > 0 &&
        gate_est.gpu_state_bytes > config_.memory_budget.gpu_bytes) {
        char msg[384];
        snprintf(msg, sizeof(msg),
            "GPU state estimate %.2f GB exceeds VRAM budget %.2f GB "
            "(N=%u nodes, A=%u actions, nc=%u canonical). ",
            static_cast<double>(gate_est.gpu_state_bytes) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(config_.memory_budget.gpu_bytes) / (1024.0 * 1024.0 * 1024.0),
            tree_.total_nodes, static_cast<unsigned>(MAX_ACTIONS),
            static_cast<unsigned>(iso_.num_canonical));

        if (backend_type_ == BackendType::AUTO) {
            // Downgrade. Don't fail the solve.
            selected_backend = BackendType::CPU;
            fallback_reason = std::string(msg) + "Auto-downgraded to CPU.";
        } else {
            // Explicit --backend gpu. Refuse so the user can decide.
            throw std::runtime_error(std::string(msg) +
                "Use --backend cpu, --backend auto, or raise --gpu-memory-mb.");
        }
    }

    // CPU host budget check applies whenever we end up on CPU (either by
    // request or via the AUTO-downgrade above). Matchup tables already
    // exist at this point — they were the FIRST line of defence in
    // precompute_matchups. Now we add the per-iteration CPU state and
    // strategy-tree EV cache + JSON response on top.
    if (selected_backend == BackendType::CPU &&
        config_.memory_budget.host_bytes > 0) {
        const uint64_t total_host =
              gate_est.matchup_tables_bytes
            + gate_est.cpu_state_bytes
            + gate_est.strategy_tree_ev_bytes
            + gate_est.json_response_bytes;
        if (total_host > config_.memory_budget.host_bytes) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                "Host RAM estimate %.2f GB exceeds budget %.2f GB "
                "(matchup=%.1f MB, cpu_state=%.1f MB, strategy_tree=%.1f MB, "
                "json=%.1f MB). Raise --host-memory-mb, reduce bet sizes, or "
                "use --strategy-tree-evs none.",
                static_cast<double>(total_host) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(config_.memory_budget.host_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(gate_est.matchup_tables_bytes) / (1024.0 * 1024.0),
                static_cast<double>(gate_est.cpu_state_bytes) / (1024.0 * 1024.0),
                static_cast<double>(gate_est.strategy_tree_ev_bytes) / (1024.0 * 1024.0),
                static_cast<double>(gate_est.json_response_bytes) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }
    }

    backend_ = create_backend(selected_backend);
    if (!backend_) backend_ = std::make_unique<CpuBackend>();

    SolverContext ctx;
    ctx.tree                       = &tree_;
    ctx.iso                        = &iso_;
    ctx.config                     = &config_;
    ctx.matchup_ev                 = &matchup_ev_;
    ctx.matchup_valid              = &matchup_valid_;
    ctx.matchup_ev_per_runout      = &matchup_ev_per_runout_;
    ctx.matchup_valid_per_runout   = &matchup_valid_per_runout_;
    ctx.ip_reach                   = &ip_reach_;
    ctx.oop_reach                  = &oop_reach_;
    ctx.resolved_locks             = &resolved_locks_;
    backend_->prepare(ctx);
    stage_end = Clock::now();
    timing.backend_prepare_ms = elapsed_since(stage_start, stage_end);

    // Step 5: run DCFR iterations through backend
    stage_start = Clock::now();
    for (int t = 0; t < config_.max_iterations; ++t) {
        backend_->iterate(t);
        actual_iterations_run_ = t + 1;

        if (progress_cb && (t + 1) % config_.exploitability_check_interval == 0) {
            auto now = Clock::now();
            float elapsed = elapsed_since(start_time, now);
            progress_cb(t + 1, 0.0f, elapsed);
        }
    }
    stage_end = Clock::now();
    timing.iterations_ms = elapsed_since(stage_start, stage_end);

    // Step 6: finalize (normalize strategy_sum → strategy)
    stage_start = Clock::now();
    backend_->finalize();
    strategy_ = backend_->strategy();  // copy out for post-solve passes
    apply_node_locks();
    solved_ = true;
    stage_end = Clock::now();
    timing.finalize_ms = elapsed_since(stage_start, stage_end);

    // Step 7: optional post-solve CPU passes (EV + exploitability)
    stage_start = Clock::now();
    if (config_.parallel_postsolve &&
        config_.compute_combo_evs &&
        config_.compute_exploitability) {
        auto ev_future = std::async(std::launch::async, [&]() {
            auto pass_start = Clock::now();
            compute_combo_evs();
            auto pass_end = Clock::now();
            return elapsed_since(pass_start, pass_end);
        });
        auto exploit_future = std::async(std::launch::async, [&]() {
            auto pass_start = Clock::now();
            float exploit = compute_exploitability();
            auto pass_end = Clock::now();
            return std::make_pair(exploit, elapsed_since(pass_start, pass_end));
        });

        timing.combo_evs_ms = ev_future.get();
        auto exploit_result = exploit_future.get();
        exploitability_pct_ = exploit_result.first;
        timing.exploitability_ms = exploit_result.second;
    } else {
        if (config_.compute_combo_evs) {
            auto pass_start = Clock::now();
            compute_combo_evs();
            auto pass_end = Clock::now();
            timing.combo_evs_ms = elapsed_since(pass_start, pass_end);
        } else {
            ev_.assign(iso_.num_canonical, 0.0f);
        }

        if (config_.compute_exploitability) {
            auto pass_start = Clock::now();
            exploitability_pct_ = compute_exploitability();
            auto pass_end = Clock::now();
            timing.exploitability_ms = elapsed_since(pass_start, pass_end);
        } else {
            exploitability_pct_ = 0.0f;
        }
    }
    stage_end = Clock::now();
    timing.postsolve_ms = elapsed_since(stage_start, stage_end);
    timing.total_ms = elapsed_since(start_time, stage_end);

    SolverResult result;
    result.iterations_run      = actual_iterations_run_;
    result.exploitability_pct  = exploitability_pct_;
    result.combo_evs_computed  = config_.compute_combo_evs;
    result.exploitability_computed = config_.compute_exploitability;
    result.action_labels       = get_action_labels_at(0);
    result.global_strategy     = extract_global_strategy_at(0);
    result.combo_strategies    = extract_combo_strategies_at(0);
    if (config_.compute_combo_evs) {
        result.ev_vector = ev_;  // expose per-canonical-combo OOP EVs
    }
    result.timing              = timing;

    // Sprint 1 (market-beating plan): every solve fills the resources block
    // with byte estimates and the budget decision so the UI / benchmark can
    // diagnose "where did the RAM go" without re-running with profiling.
    // Reuses gate_est computed before backend allocation.
    {
        SolveResources r;
        r.canonical_combos = iso_.num_canonical;
        uint32_t player_nodes = 0;
        for (uint32_t n = 0; n < tree_.total_nodes; ++n) {
            auto t = static_cast<NodeType>(tree_.node_types[n]);
            if (t == NodeType::PLAYER_OOP || t == NodeType::PLAYER_IP) ++player_nodes;
        }
        r.player_nodes = player_nodes;

        r.estimated_matchup_bytes        = gate_est.matchup_tables_bytes;
        r.estimated_cpu_state_bytes      = gate_est.cpu_state_bytes;
        r.estimated_gpu_state_bytes      = gate_est.gpu_state_bytes;
        r.estimated_strategy_tree_bytes  = gate_est.strategy_tree_ev_bytes;
        r.estimated_json_bytes           = gate_est.json_response_bytes;

        r.host_budget_bytes        = config_.memory_budget.host_bytes;
        r.gpu_budget_bytes         = config_.memory_budget.gpu_bytes;
        r.strategy_tree_max_nodes  = config_.memory_budget.strategy_tree_max_nodes;

        BudgetDecision d = evaluate_budget(gate_est, config_.memory_budget);
        r.budget_decision            = budget_decision_str(d);
        if (d != BudgetDecision::OK) {
            r.diagnostic = format_budget_failure(d, gate_est, config_.memory_budget);
        }
        // Sprint 1: surface the AUTO→CPU downgrade reason recorded by the
        // pre-backend gate. Empty string means no fallback happened.
        r.fallback_reason = fallback_reason;
        result.resources = std::move(r);
    }
    return result;
}

// ============================================================================
// Precompute: reach probabilities and showdown matchup matrix
// ============================================================================

inline void Solver::initialize_reach_probs() {
    ip_reach_.assign(iso_.num_canonical, 0.0f);
    oop_reach_.assign(iso_.num_canonical, 0.0f);

    const auto& combo_table = get_combo_table();
    CardMask dead = board_to_mask(config_.board.data(), config_.board_size);

    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        if (combo_table[i].conflicts_with(dead)) continue;
        ip_reach_[ci]  = std::max(ip_reach_[ci],  config_.ip_range_weights[i]);
        oop_reach_[ci] = std::max(oop_reach_[ci], config_.oop_range_weights[i]);
    }
}

/// Helper: compute one (matchup_ev, matchup_valid) pair for a specific board.
/// `board_cards` is the FULL board to evaluate at this terminal (config flop +
/// any runout cards dealt from chance nodes along the path).
inline void compute_matchup_for_board(
    const Solver& solver_unused,
    const IsomorphismMapping& iso,
    const Card* board_cards,
    uint8_t board_size,
    std::function<uint16_t(const Combo&)> eval_for_board,
    std::vector<float>& out_ev,
    std::vector<float>& out_valid)
{
    (void)solver_unused;
    uint16_t nc = iso.num_canonical;
    out_ev.assign(static_cast<size_t>(nc) * nc, 0.0f);
    out_valid.assign(static_cast<size_t>(nc) * nc, 0.0f);

    const auto& combo_table = get_combo_table();
    CardMask board_mask = board_to_mask(board_cards, board_size);

    std::array<CardMask, NUM_COMBOS> combo_masks{};
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        combo_masks[i] = card_to_mask(combo_table[i].cards[0])
                       | card_to_mask(combo_table[i].cards[1]);
    }

    std::vector<uint16_t> ranks(NUM_COMBOS, UINT16_MAX);
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        if (combo_table[i].conflicts_with(board_mask)) continue;
        ranks[i] = eval_for_board(combo_table[i]);
    }

    for (uint16_t ci = 0; ci < nc; ++ci) {
        const auto& originals_i = iso.canonical_to_originals[ci];
        for (uint16_t cj = 0; cj < nc; ++cj) {
            const auto& originals_j = iso.canonical_to_originals[cj];
            int oop_wins = 0, ip_wins = 0, valid = 0;
            for (uint16_t oi : originals_i) {
                if (ranks[oi] == UINT16_MAX) continue;
                CardMask mask_i = combo_masks[oi];
                for (uint16_t oj : originals_j) {
                    if (ranks[oj] == UINT16_MAX) continue;
                    CardMask mask_j = combo_masks[oj];
                    if (mask_i & mask_j) continue;
                    ++valid;
                    if (ranks[oi] < ranks[oj]) ++oop_wins;
                    else if (ranks[oi] > ranks[oj]) ++ip_wins;
                }
            }
            size_t idx = static_cast<size_t>(ci) * nc + cj;
            if (valid > 0) {
                out_ev[idx] = static_cast<float>(oop_wins - ip_wins) / valid;
                out_valid[idx] = static_cast<float>(valid)
                    / std::max(1u, static_cast<unsigned>(originals_i.size() * originals_j.size()));
            }
        }
    }
}

inline void Solver::precompute_matchups() {
    uint16_t nc = iso_.num_canonical;
    auto& eval = get_evaluator();

    // Build a hand-evaluator closure for an arbitrary board. Uses the same
    // evaluate_combo logic but for a runtime board (not config_.board).
    auto make_eval_for = [&eval](const Card* bd, uint8_t bs) {
        return [&eval, bd, bs](const Combo& combo) -> uint16_t {
            Card c0 = combo.cards[0], c1 = combo.cards[1];
            if (bs == 5) {
                return eval.evaluate(c0, c1, bd[0], bd[1], bd[2], bd[3], bd[4]);
            } else if (bs == 4) {
                Card cards[6] = {c0, c1, bd[0], bd[1], bd[2], bd[3]};
                uint16_t best = UINT16_MAX;
                for (int skip = 0; skip < 6; ++skip) {
                    Card h[5]; int idx = 0;
                    for (int j = 0; j < 6; ++j) if (j != skip) h[idx++] = cards[j];
                    best = std::min(best, eval.evaluate5(h[0], h[1], h[2], h[3], h[4]));
                }
                return best;
            } else {
                return eval.evaluate5(c0, c1, bd[0], bd[1], bd[2]);
            }
        };
    };

    // ---- Walk the tree to discover every distinct board reachable at any
    // node. Use DFS, accumulating runout cards from chance children.
    // Board signature = sorted tuple of (config.board ∪ runout_cards).
    matchup_ev_per_runout_.clear();
    matchup_valid_per_runout_.clear();
    std::map<std::vector<uint8_t>, int32_t> sig_to_idx;

    std::vector<int32_t>& matchup_idx = tree_.matchup_idx;
    matchup_idx.assign(tree_.total_nodes, 0);

    // DFS with explicit stack: (node, runout_cards_so_far).
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> stk;
    {
        std::vector<uint8_t> empty;
        stk.push_back({0u, empty});
    }

    auto signature_for = [&](const std::vector<uint8_t>& runouts) {
        std::vector<uint8_t> sig;
        sig.reserve(config_.board_size + runouts.size());
        for (uint8_t i = 0; i < config_.board_size; ++i) sig.push_back(config_.board[i]);
        for (uint8_t c : runouts) sig.push_back(c);
        std::sort(sig.begin(), sig.end());
        return sig;
    };

    // Sprint 1 (market-beating plan): hard byte cap at the actual allocation
    // site. The tree builder's gate is the FIRST line of defence (refuses to
    // enumerate runouts that wouldn't fit). This is the SECOND — even if the
    // tree somehow ends up with more matchup signatures than budgeted, refuse
    // to allocate. Throws a structured runtime_error which main.cpp formats
    // as JSON for the UI.
    const uint64_t per_table_bytes =
        2ULL * static_cast<uint64_t>(nc) * static_cast<uint64_t>(nc) * sizeof(float);
    // Reserve half the host budget for matchup tables (same split as the
    // tree builder gate). Floor of 3 GB matches the builder's safety floor.
    const uint64_t matchup_byte_cap = (config_.memory_budget.host_bytes > 0)
        ? (config_.memory_budget.host_bytes / 2ULL)
        : (3ULL * 1024 * 1024 * 1024);

    auto register_signature = [&](const std::vector<uint8_t>& runouts) -> int32_t {
        auto sig = signature_for(runouts);
        auto it = sig_to_idx.find(sig);
        if (it != sig_to_idx.end()) return it->second;
        int32_t new_idx = static_cast<int32_t>(matchup_ev_per_runout_.size());

        // Bytes already allocated + this new table. Refuse before push_back
        // so a mid-allocation bad_alloc can't kill the process.
        const uint64_t cur_bytes  = static_cast<uint64_t>(new_idx) * per_table_bytes;
        const uint64_t next_bytes = cur_bytes + per_table_bytes;
        if (next_bytes > matchup_byte_cap) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "precompute_matchups would allocate %.2f GB (cap %.2f GB, "
                "%d tables × %u canonical² × 8 B). Reduce flop runout enumeration, "
                "use --strategy-tree-evs none, or raise --host-memory-mb.",
                static_cast<double>(next_bytes)  / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(matchup_byte_cap) / (1024.0 * 1024.0 * 1024.0),
                new_idx + 1, static_cast<unsigned>(nc));
            throw std::runtime_error(buf);
        }

        sig_to_idx[sig] = new_idx;
        // Compute matchup table for this board signature.
        std::vector<float> ev, valid;
        uint8_t bs = static_cast<uint8_t>(sig.size());
        compute_matchup_for_board(*this, iso_, sig.data(), bs,
            make_eval_for(sig.data(), bs), ev, valid);
        matchup_ev_per_runout_.push_back(std::move(ev));
        matchup_valid_per_runout_.push_back(std::move(valid));
        return new_idx;
    };

    while (!stk.empty()) {
        auto [node, runouts] = std::move(stk.back());
        stk.pop_back();

        // Tag this node with the matchup index for its current board.
        matchup_idx[node] = register_signature(runouts);

        auto nt = static_cast<NodeType>(tree_.node_types[node]);
        if (nt == NodeType::TERMINAL) continue;

        uint32_t off = tree_.children_offset[node];
        uint8_t nch = tree_.num_children[node];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree_.children[off + k];
            std::vector<uint8_t> child_runouts = runouts;
            // If the child was produced by a chance deal, append the dealt card.
            if (tree_.dealt_card[child] != 0xFFu) {
                child_runouts.push_back(tree_.dealt_card[child]);
            }
            stk.push_back({child, std::move(child_runouts)});
        }
    }

    // Keep matchup_ev_/_valid_ pointing at the root board (idx 0) for
    // backward-compat with code paths that haven't been migrated yet.
    if (!matchup_ev_per_runout_.empty()) {
        matchup_ev_    = matchup_ev_per_runout_[0];
        matchup_valid_ = matchup_valid_per_runout_[0];
    } else {
        matchup_ev_.assign(static_cast<size_t>(nc) * nc, 0.0f);
        matchup_valid_.assign(static_cast<size_t>(nc) * nc, 0.0f);
    }
}

// ============================================================================
// Node locks
// ============================================================================

inline void Solver::resolve_node_locks() {
    resolved_locks_.clear();
    for (const auto& lock : config_.node_locks) {
        uint32_t node_idx = navigate_to_node(lock.history);
        if (node_idx >= tree_.total_nodes) continue;
        uint16_t ci = UINT16_MAX;
        if (lock.combo_idx < NUM_COMBOS) {
            ci = iso_.original_to_canonical[lock.combo_idx];
        }
        if (ci == UINT16_MAX) continue;
        resolved_locks_[{node_idx, ci}] = lock.strategy;
    }
}

inline void Solver::apply_node_locks() {
    if (resolved_locks_.empty()) return;
    for (const auto& [key, forced_strat] : resolved_locks_) {
        auto [node_idx, canonical_idx] = key;
        if (node_idx >= strategy_.size()) continue;
        uint8_t n_act = tree_.num_children[node_idx];
        if (n_act == 0) continue;
        for (uint8_t a = 0; a < n_act && a < forced_strat.size(); ++a) {
            size_t idx = a * iso_.num_canonical + canonical_idx;
            if (idx < strategy_[node_idx].size()) {
                strategy_[node_idx][idx] = forced_strat[a];
            }
        }
    }
}

// ============================================================================
// EV forward pass (post-solve, CPU, operates on final strategy_)
// ============================================================================

inline std::vector<float> Solver::cpu_ev_traverse(
    uint32_t node_idx, int perspective,
    std::vector<float>& reach_oop, std::vector<float>& reach_ip,
    std::map<uint32_t, std::vector<float>>* out_node_values,
    const std::set<uint32_t>* visible_filter) const
{
    // Helper: write the per-combo values to the out-map (if requested) and
    // then return them. Used at every return point so the map records every
    // visited node's values during a single root traversal.
    //
    // Phase 3 (10-point plan): when `visible_filter` is non-null we only
    // record nodes the strategy tree will actually emit. The traversal still
    // recurses through every node — we need their EVs to compute parents
    // correctly — but we drop the per-combo vector once we've used it,
    // saving O(visited_nodes × nc × 4 B) of host RAM for the typical
    // 8-deep tree where only a few hundred nodes get emitted.
    auto record = [&](std::vector<float>&& vals) -> std::vector<float> {
        if (out_node_values) {
            const bool keep = (visible_filter == nullptr) ||
                              (visible_filter->find(node_idx) != visible_filter->end());
            if (keep) (*out_node_values)[node_idx] = vals;
        }
        return std::move(vals);
    };
    uint16_t nc = iso_.num_canonical;
    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);

    if (nt == NodeType::TERMINAL) {
        std::vector<float> values(nc, 0.0f);
        auto tt = static_cast<TerminalType>(tree_.terminal_types[node_idx]);
        float half_pot = tree_.pots[node_idx] / 2.0f;

        // Phase 1: pick per-runout matchup table.
        int32_t mi = (node_idx < tree_.matchup_idx.size()) ? tree_.matchup_idx[node_idx] : 0;
        const auto& m_ev = (mi >= 0 && static_cast<size_t>(mi) < matchup_ev_per_runout_.size())
            ? matchup_ev_per_runout_[mi] : matchup_ev_;
        const auto& m_valid = (mi >= 0 && static_cast<size_t>(mi) < matchup_valid_per_runout_.size())
            ? matchup_valid_per_runout_[mi] : matchup_valid_;

        if (tt == TerminalType::SHOWDOWN) {
            if (perspective == 0) {
                std::vector<float> weighted_ip(nc, 0.0f);
                for (uint16_t cj = 0; cj < nc; ++cj) {
                    weighted_ip[cj] = reach_ip[cj] *
                        static_cast<float>(iso_.canonical_weights[cj]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float val = 0.0f;
                        for (uint16_t cj = 0; cj < nc; ++cj) {
                            size_t idx = static_cast<size_t>(c) * nc + cj;
                            val += weighted_ip[cj] * m_ev[idx] * m_valid[idx];
                        }
                        values[c] = val * half_pot;
                    });
            } else {
                std::vector<float> weighted_oop(nc, 0.0f);
                for (uint16_t ci = 0; ci < nc; ++ci) {
                    weighted_oop[ci] = reach_oop[ci] *
                        static_cast<float>(iso_.canonical_weights[ci]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float val = 0.0f;
                        for (uint16_t ci = 0; ci < nc; ++ci) {
                            size_t idx = static_cast<size_t>(ci) * nc + c;
                            val += weighted_oop[ci] * (-m_ev[idx]) * m_valid[idx];
                        }
                        values[c] = val * half_pot;
                    });
            }
        } else {
            uint32_t parent = tree_.parent_indices[node_idx];
            float unmatched_bet = (parent < tree_.total_nodes)
                ? tree_.bet_into[parent] : 0.0f;
            float gain = (tree_.pots[node_idx] - unmatched_bet) * 0.5f;

            float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
            float sign = (perspective == 0) ? sign_oop : -sign_oop;
            if (perspective == 0) {
                std::vector<float> weighted_ip(nc, 0.0f);
                for (uint16_t cj = 0; cj < nc; ++cj) {
                    weighted_ip[cj] = reach_ip[cj] *
                        static_cast<float>(iso_.canonical_weights[cj]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float opp_total = 0.0f;
                        for (uint16_t cj = 0; cj < nc; ++cj) {
                            size_t idx = static_cast<size_t>(c) * nc + cj;
                            opp_total += weighted_ip[cj] * m_valid[idx];
                        }
                        values[c] = sign * gain * opp_total;
                    });
            } else {
                std::vector<float> weighted_oop(nc, 0.0f);
                for (uint16_t ci = 0; ci < nc; ++ci) {
                    weighted_oop[ci] = reach_oop[ci] *
                        static_cast<float>(iso_.canonical_weights[ci]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float opp_total = 0.0f;
                        for (uint16_t ci = 0; ci < nc; ++ci) {
                            size_t idx = static_cast<size_t>(ci) * nc + c;
                            opp_total += weighted_oop[ci] * m_valid[idx];
                        }
                        values[c] = sign * gain * opp_total;
                    });
            }
        }
        return record(std::move(values));
    }

    if (nt == NodeType::CHANCE) {
        // Enumerate runouts: weighted average over all chance children.
        // weight==0 should never occur (TreeNode default is 1); treat as 1
        // rather than silently dropping the child, which would bias the
        // average toward survivors.
        uint8_t nch = tree_.num_children[node_idx];
        if (nch == 0) return record(std::vector<float>(nc, 0.0f));
        std::vector<float> avg(nc, 0.0f);
        uint32_t total_weight = 0;
        uint32_t off = tree_.children_offset[node_idx];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree_.children[off + k];
            uint32_t weight = (child < tree_.runout_weight.size())
                                ? tree_.runout_weight[child] : 1;
            if (weight == 0) weight = 1;
            std::vector<float> child_vals = cpu_ev_traverse(
                child, perspective, reach_oop, reach_ip, out_node_values, visible_filter);
            for (uint16_t c = 0; c < nc; ++c) {
                avg[c] += static_cast<float>(weight) * child_vals[c];
            }
            total_weight += weight;
        }
        if (total_weight > 0) {
            float inv = 1.0f / static_cast<float>(total_weight);
            for (uint16_t c = 0; c < nc; ++c) avg[c] *= inv;
        }
        return record(std::move(avg));
    }

    int acting = tree_.active_player[node_idx];
    uint8_t na = tree_.num_children[node_idx];
    if (node_idx >= strategy_.size() || strategy_[node_idx].empty()) {
        return record(std::vector<float>(nc, 0.0f));
    }
    const auto& strat = strategy_[node_idx];

    std::vector<float> node_vals(nc, 0.0f);
    uint32_t action_offset = tree_.children_offset[node_idx];
    if (acting == perspective) {
        for (uint8_t a = 0; a < na; ++a) {
            uint32_t child = tree_.children[action_offset + a];
            std::vector<float> child_vals =
                cpu_ev_traverse(child, perspective, reach_oop, reach_ip, out_node_values, visible_filter);
            for (uint16_t c = 0; c < nc; ++c) {
                node_vals[c] += strat[a * nc + c] * child_vals[c];
            }
        }
    } else {
        auto& acting_reach = (acting == 0) ? reach_oop : reach_ip;
        std::vector<float> saved(nc);
        for (uint8_t a = 0; a < na; ++a) {
            uint32_t child = tree_.children[action_offset + a];
            for (uint16_t c = 0; c < nc; ++c) {
                saved[c] = acting_reach[c];
                acting_reach[c] *= strat[a * nc + c];
            }
            std::vector<float> child_vals =
                cpu_ev_traverse(child, perspective, reach_oop, reach_ip, out_node_values, visible_filter);
            for (uint16_t c = 0; c < nc; ++c) acting_reach[c] = saved[c];
            for (uint16_t c = 0; c < nc; ++c) node_vals[c] += child_vals[c];
        }
    }
    return record(std::move(node_vals));
}

inline void Solver::compute_combo_evs() {
    uint16_t nc = iso_.num_canonical;
    ev_.assign(nc, 0.0f);
    if (strategy_.empty() || nc == 0) return;

    // Prefer GPU postsolve when the backend supports it. Falls back to CPU
    // traversal on empty / size-mismatched return (treated as "not supported").
    std::vector<float> oop_evs_raw;
    if (!config_.force_cpu_postsolve &&
        backend_ && backend_->supports_gpu_postsolve()) {
        oop_evs_raw = backend_->compute_combo_evs_gpu();
    }
    if (oop_evs_raw.size() != nc) {
        auto r_oop = oop_reach_;
        auto r_ip  = ip_reach_;
        oop_evs_raw = cpu_ev_traverse(0, 0, r_oop, r_ip);
    }

    float total_ip_weight = 0.0f;
    for (uint16_t cj = 0; cj < nc; ++cj) {
        total_ip_weight += ip_reach_[cj] * static_cast<float>(iso_.canonical_weights[cj]);
    }
    float scale = (total_ip_weight > 1e-6f) ? (1.0f / total_ip_weight) : 0.0f;
    for (uint16_t c = 0; c < nc; ++c) {
        ev_[c] = oop_evs_raw[c] * scale;
    }
}

// ============================================================================
// Exploitability via per-combo Best Response
// ============================================================================

inline std::vector<float> Solver::cpu_best_response_traverse(
    uint32_t node_idx, int player,
    std::vector<float>& reach_oop, std::vector<float>& reach_ip) const
{
    uint16_t nc = iso_.num_canonical;
    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);

    if (nt == NodeType::TERMINAL) {
        std::vector<float> values(nc, 0.0f);
        auto tt = static_cast<TerminalType>(tree_.terminal_types[node_idx]);
        float half_pot = tree_.pots[node_idx] / 2.0f;

        // Phase 1: per-runout matchup table.
        int32_t mi = (node_idx < tree_.matchup_idx.size()) ? tree_.matchup_idx[node_idx] : 0;
        const auto& m_ev = (mi >= 0 && static_cast<size_t>(mi) < matchup_ev_per_runout_.size())
            ? matchup_ev_per_runout_[mi] : matchup_ev_;
        const auto& m_valid = (mi >= 0 && static_cast<size_t>(mi) < matchup_valid_per_runout_.size())
            ? matchup_valid_per_runout_[mi] : matchup_valid_;

        if (tt == TerminalType::SHOWDOWN) {
            if (player == 0) {
                std::vector<float> weighted_ip(nc, 0.0f);
                for (uint16_t cj = 0; cj < nc; ++cj) {
                    weighted_ip[cj] = reach_ip[cj] *
                        static_cast<float>(iso_.canonical_weights[cj]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float val = 0.0f;
                        for (uint16_t cj = 0; cj < nc; ++cj) {
                            size_t idx = static_cast<size_t>(c) * nc + cj;
                            val += weighted_ip[cj] * m_ev[idx] * m_valid[idx];
                        }
                        values[c] = val * half_pot;
                    });
            } else {
                std::vector<float> weighted_oop(nc, 0.0f);
                for (uint16_t ci = 0; ci < nc; ++ci) {
                    weighted_oop[ci] = reach_oop[ci] *
                        static_cast<float>(iso_.canonical_weights[ci]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float val = 0.0f;
                        for (uint16_t ci = 0; ci < nc; ++ci) {
                            size_t idx = static_cast<size_t>(ci) * nc + c;
                            val += weighted_oop[ci] * (-m_ev[idx]) * m_valid[idx];
                        }
                        values[c] = val * half_pot;
                    });
            }
        } else {
            uint32_t parent = tree_.parent_indices[node_idx];
            float unmatched_bet = (parent < tree_.total_nodes)
                ? tree_.bet_into[parent] : 0.0f;
            float gain = (tree_.pots[node_idx] - unmatched_bet) * 0.5f;

            float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
            float sign = (player == 0) ? sign_oop : -sign_oop;
            if (player == 0) {
                std::vector<float> weighted_ip(nc, 0.0f);
                for (uint16_t cj = 0; cj < nc; ++cj) {
                    weighted_ip[cj] = reach_ip[cj] *
                        static_cast<float>(iso_.canonical_weights[cj]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float opp_total = 0.0f;
                        for (uint16_t cj = 0; cj < nc; ++cj) {
                            size_t idx = static_cast<size_t>(c) * nc + cj;
                            opp_total += weighted_ip[cj] * m_valid[idx];
                        }
                        values[c] = sign * gain * opp_total;
                    });
            } else {
                std::vector<float> weighted_oop(nc, 0.0f);
                for (uint16_t ci = 0; ci < nc; ++ci) {
                    weighted_oop[ci] = reach_oop[ci] *
                        static_cast<float>(iso_.canonical_weights[ci]);
                }
                detail::postsolve_parallel_for(
                    static_cast<uint32_t>(nc),
                    config_.parallel_postsolve,
                    config_.postsolve_threads,
                    [&](uint32_t cidx) {
                        uint16_t c = static_cast<uint16_t>(cidx);
                        float opp_total = 0.0f;
                        for (uint16_t ci = 0; ci < nc; ++ci) {
                            size_t idx = static_cast<size_t>(ci) * nc + c;
                            opp_total += weighted_oop[ci] * m_valid[idx];
                        }
                        values[c] = sign * gain * opp_total;
                    });
            }
        }
        return values;
    }

    if (nt == NodeType::CHANCE) {
        // Enumerate runouts: weighted average over all chance children.
        // weight==0 should never occur; treat as 1 (see cpu_ev_traverse).
        uint8_t nch = tree_.num_children[node_idx];
        if (nch == 0) return std::vector<float>(nc, 0.0f);
        std::vector<float> avg(nc, 0.0f);
        uint32_t total_weight = 0;
        uint32_t off = tree_.children_offset[node_idx];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree_.children[off + k];
            uint32_t weight = (child < tree_.runout_weight.size())
                                ? tree_.runout_weight[child] : 1;
            if (weight == 0) weight = 1;
            std::vector<float> child_vals = cpu_best_response_traverse(
                child, player, reach_oop, reach_ip);
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

    int acting = tree_.active_player[node_idx];
    uint8_t na = tree_.num_children[node_idx];
    if (na == 0) return std::vector<float>(nc, 0.0f);

    if (acting == player) {
        // BR: each combo picks its best action independently
        uint32_t action_offset = tree_.children_offset[node_idx];
        uint32_t first_child = tree_.children[action_offset];
        std::vector<float> best =
            cpu_best_response_traverse(first_child, player, reach_oop, reach_ip);
        for (uint8_t a = 1; a < na; ++a) {
            uint32_t child = tree_.children[action_offset + a];
            std::vector<float> child_vals =
                cpu_best_response_traverse(child, player, reach_oop, reach_ip);
            for (uint16_t c = 0; c < nc; ++c) {
                best[c] = std::max(best[c], child_vals[c]);
            }
        }
        return best;
    }

    // Opponent: play averaged strategy, absorbed into reach, then sum children
    bool has_strat = (node_idx < strategy_.size() && !strategy_[node_idx].empty());
    std::vector<float> node_vals(nc, 0.0f);
    uint32_t action_offset = tree_.children_offset[node_idx];
    auto& acting_reach = (acting == 0) ? reach_oop : reach_ip;
    std::vector<float> saved(nc);
    for (uint8_t a = 0; a < na; ++a) {
        uint32_t child = tree_.children[action_offset + a];
        for (uint16_t c = 0; c < nc; ++c) {
            saved[c] = acting_reach[c];
            if (has_strat) {
                acting_reach[c] *= strategy_[node_idx][a * nc + c];
            }
        }
        std::vector<float> child_vals =
            cpu_best_response_traverse(child, player, reach_oop, reach_ip);
        for (uint16_t c = 0; c < nc; ++c) acting_reach[c] = saved[c];
        for (uint16_t c = 0; c < nc; ++c) {
            node_vals[c] += child_vals[c];
        }
    }
    return node_vals;
}

inline float Solver::compute_exploitability() {
    if (!solved_ || strategy_.empty()) return 0.0f;
    uint16_t nc = iso_.num_canonical;

    std::vector<float> br_oop_per_combo;
    std::vector<float> br_ip_per_combo;

    // Prefer GPU postsolve. Both BR vectors must be size==nc to count as a
    // success — anything else falls back to CPU traversal.
    if (!config_.force_cpu_postsolve &&
        backend_ && backend_->supports_gpu_postsolve()) {
        br_oop_per_combo = backend_->compute_best_response_gpu(0);
        br_ip_per_combo  = backend_->compute_best_response_gpu(1);
    }
    bool gpu_ok = (br_oop_per_combo.size() == nc &&
                   br_ip_per_combo.size()  == nc);

    if (!gpu_ok) {
        if (config_.parallel_postsolve) {
            auto oop_future = std::async(std::launch::async, [&]() {
                auto r_oop = oop_reach_;
                auto r_ip  = ip_reach_;
                return cpu_best_response_traverse(0, 0, r_oop, r_ip);
            });
            auto ip_future = std::async(std::launch::async, [&]() {
                auto r_oop = oop_reach_;
                auto r_ip  = ip_reach_;
                return cpu_best_response_traverse(0, 1, r_oop, r_ip);
            });
            br_oop_per_combo = oop_future.get();
            br_ip_per_combo = ip_future.get();
        } else {
            auto r_oop = oop_reach_;
            auto r_ip  = ip_reach_;
            br_oop_per_combo = cpu_best_response_traverse(0, 0, r_oop, r_ip);

            r_oop = oop_reach_;
            r_ip  = ip_reach_;
            br_ip_per_combo = cpu_best_response_traverse(0, 1, r_oop, r_ip);
        }
    }

    float br_oop_total = 0.0f, br_ip_total = 0.0f;
    float total_oop_w = 0.0f,  total_ip_w  = 0.0f;
    for (uint16_t c = 0; c < nc; ++c) {
        float w = static_cast<float>(iso_.canonical_weights[c]);
        br_oop_total += oop_reach_[c] * w * br_oop_per_combo[c];
        br_ip_total  += ip_reach_[c]  * w * br_ip_per_combo[c];
        total_oop_w  += oop_reach_[c] * w;
        total_ip_w   += ip_reach_[c]  * w;
    }

    float denom = total_oop_w * total_ip_w;
    float avg_oop = (denom > 1e-6f) ? (br_oop_total / denom) : 0.0f;
    float avg_ip  = (denom > 1e-6f) ? (br_ip_total  / denom) : 0.0f;
    float exploit_chips = (avg_oop + avg_ip) / 2.0f;
    float exploit = exploit_chips / std::max(config_.pot, 1.0f) * 100.0f;
    return std::max(0.0f, exploit);
}

// ============================================================================
// Tree navigation & result extraction
// ============================================================================

inline std::vector<std::string> Solver::get_action_labels_at(uint32_t node_idx) const {
    std::vector<std::string> labels;
    if (node_idx >= tree_.total_nodes) return labels;
    uint32_t offset = tree_.children_offset[node_idx];
    uint8_t num_children = tree_.num_children[node_idx];
    float pot = tree_.pots[node_idx];
    for (uint8_t i = 0; i < num_children; ++i) {
        auto at = static_cast<ActionType>(tree_.child_action_types[offset + i]);
        float amt = tree_.child_action_amts[offset + i];
        labels.push_back(action_to_label(at, amt, pot));
    }
    return labels;
}

inline std::vector<std::pair<std::string, float>> Solver::extract_global_strategy_at(uint32_t node_idx) const {
    std::vector<std::pair<std::string, float>> global_strat;
    if (!solved_ || node_idx >= tree_.total_nodes || strategy_.empty() || strategy_[node_idx].empty()) {
        return global_strat;
    }
    uint8_t n_act = tree_.num_children[node_idx];
    if (n_act == 0) return global_strat;

    std::vector<float> totals(n_act, 0.0f);
    float grand_total = 0;
    auto labels = get_action_labels_at(node_idx);

    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);
    bool is_ip = (nt == NodeType::PLAYER_IP);
    const auto& reach = is_ip ? ip_reach_ : oop_reach_;

    for (uint16_t c = 0; c < iso_.num_canonical; ++c) {
        float weight = static_cast<float>(iso_.canonical_weights[c]);
        float reach_w = (c < reach.size()) ? reach[c] : 1.0f;
        weight *= reach_w;
        for (uint8_t a = 0; a < n_act; ++a) {
            size_t idx = a * iso_.num_canonical + c;
            if (idx < strategy_[node_idx].size()) {
                float val = strategy_[node_idx][idx] * weight;
                totals[a] += val;
                grand_total += val;
            }
        }
    }

    for (uint8_t a = 0; a < n_act; ++a) {
        float pct = (grand_total > 0) ? (totals[a] / grand_total * 100.0f) : 0;
        global_strat.push_back({labels[a], pct});
    }
    return global_strat;
}

inline std::vector<std::pair<std::string,
    std::vector<std::pair<std::string, float>>>>
    Solver::extract_combo_strategies_at(uint32_t node_idx) const
{
    std::vector<std::pair<std::string,
        std::vector<std::pair<std::string, float>>>> out;
    if (!solved_ || node_idx >= tree_.total_nodes ||
        strategy_.empty() || strategy_[node_idx].empty()) {
        return out;
    }
    uint8_t n_act = tree_.num_children[node_idx];
    if (n_act == 0) return out;

    auto labels = get_action_labels_at(node_idx);
    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);
    bool is_ip = (nt == NodeType::PLAYER_IP);
    const auto& reach = is_ip ? ip_reach_ : oop_reach_;

    const auto& combo_table = get_combo_table();

    // Per-grid-label aggregation. Each grid label (e.g. "AKs") can map to
    // multiple canonical slots (different suit orientations post-flop). We
    // sum action * reach across those slots, then normalize per label.
    std::map<std::string, std::vector<float>> totals;  // label → per-action sum
    std::map<std::string, float> weights;              // label → summed reach

    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        std::string label = combo_to_grid_label(combo_table[i]);
        float reach_w = (ci < reach.size()) ? reach[ci] : 0.0f;
        if (reach_w <= 0.0f) continue;  // outside range → skip

        auto& arr = totals[label];
        if (arr.empty()) arr.assign(n_act, 0.0f);
        for (uint8_t a = 0; a < n_act; ++a) {
            size_t idx = a * iso_.num_canonical + ci;
            if (idx < strategy_[node_idx].size()) {
                arr[a] += strategy_[node_idx][idx] * reach_w;
            }
        }
        weights[label] += reach_w;
    }

    for (auto& [label, arr] : totals) {
        float w = weights[label];
        if (w <= 0.0f) continue;
        std::vector<std::pair<std::string, float>> entry;
        entry.reserve(n_act);
        for (uint8_t a = 0; a < n_act; ++a) {
            entry.push_back({labels[a], arr[a] / w});  // [0, 1]
        }
        out.push_back({label, std::move(entry)});
    }

    // Also emit per-specific-combo strategies (e.g. "AhKs", "AsKh") so the
    // UI's combo-variants panel can show per-variant frequencies. Different
    // suit orientations may land in different canonical slots post-flop
    // (e.g. flush-draw vs no-FD vs board-blocker), so the strategy can
    // legitimately differ between variants of the same grid label.
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        float reach_w = (ci < reach.size()) ? reach[ci] : 0.0f;
        if (reach_w <= 0.0f) continue;

        const Combo& cb = combo_table[i];
        // Build combo string like "AhKs" — cards sorted by descending rank
        // to match the UI's canonical form.
        auto card_str = [](Card c) -> std::string {
            static const char ranks[] = "23456789TJQKA";
            static const char suits[] = "cdhs";
            std::string r;
            r += ranks[card_rank(c)];
            r += suits[card_suit(c)];
            return r;
        };
        Card c0 = cb.cards[0], c1 = cb.cards[1];
        if (card_rank(c1) > card_rank(c0) ||
            (card_rank(c1) == card_rank(c0) && card_suit(c1) > card_suit(c0))) {
            std::swap(c0, c1);
        }
        std::string combo_key = card_str(c0) + card_str(c1);

        std::vector<std::pair<std::string, float>> entry;
        entry.reserve(n_act);
        for (uint8_t a = 0; a < n_act; ++a) {
            size_t idx = a * iso_.num_canonical + ci;
            float v = (idx < strategy_[node_idx].size())
                        ? strategy_[node_idx][idx] : 0.0f;
            entry.push_back({labels[a], v});
        }
        out.push_back({std::move(combo_key), std::move(entry)});
    }
    return out;
}

inline uint32_t Solver::navigate_to_node(const std::string& history) const {
    if (history.empty()) return 0;
    std::vector<std::string> steps;
    size_t start = 0, comma;
    while ((comma = history.find(',', start)) != std::string::npos) {
        if (comma > start) steps.push_back(history.substr(start, comma - start));
        start = comma + 1;
    }
    if (start < history.length()) steps.push_back(history.substr(start));

    // --- Fuzzy step parser ---
    // The frontend may send labels in formats that don't exactly match our
    // internal action_to_label() output — e.g. "Bet 75%" vs "Bet_75" (space
    // vs underscore), or "Raise 3x" (multiplier of facing bet) when our
    // tree has "Raise_N" with N as fraction-of-pot. We parse each step into
    // (ActionType, optional target amount) and then match to the closest
    // child of the right action type. This is robust to label-format drift.
    struct ParsedStep {
        ActionType type;
        bool has_amount;
        float target_amount;  // absolute bet/raise-to amount in chips
    };
    auto parse_step = [&](const std::string& raw, float node_pot,
                          float node_bet_into) -> ParsedStep {
        ParsedStep ps{ActionType::CHECK, false, 0.0f};
        // Normalize: trim whitespace
        std::string s = raw;
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();

        // Prefix-based action type detection (case-sensitive, matches labels
        // we generate: Check, Fold, Call, All-in, Bet, Raise).
        auto starts_with = [&](const std::string& prefix) {
            return s.size() >= prefix.size() &&
                   s.compare(0, prefix.size(), prefix) == 0;
        };
        if (starts_with("Check")) { ps.type = ActionType::CHECK; return ps; }
        if (starts_with("Fold"))  { ps.type = ActionType::FOLD;  return ps; }
        if (starts_with("Call"))  { ps.type = ActionType::CALL;  return ps; }
        if (starts_with("All-in") || starts_with("Allin")) {
            ps.type = ActionType::ALLIN;
            return ps;
        }

        bool is_bet   = starts_with("Bet");
        bool is_raise = starts_with("Raise");
        if (!is_bet && !is_raise) {
            // Unknown label — fall through as CHECK; exact match fallback will
            // try regardless of parsed type.
            return ps;
        }
        ps.type = is_bet ? ActionType::BET : ActionType::RAISE;

        // Extract the numeric payload (could be "33", "75%", "3x", " 33 "…)
        std::string tail = s.substr(is_bet ? 3 : 5);
        // Strip separators
        for (char& c : tail) if (c == '_' || c == ' ' || c == '\t') c = ' ';
        // Find first digit
        size_t i = 0;
        while (i < tail.size() && (tail[i] < '0' || tail[i] > '9')) ++i;
        if (i >= tail.size()) return ps;
        float num = 0.0f;
        bool has_dot = false;
        float frac = 0.1f;
        for (; i < tail.size(); ++i) {
            char c = tail[i];
            if (c >= '0' && c <= '9') {
                if (!has_dot) num = num * 10.0f + (c - '0');
                else          { num += (c - '0') * frac; frac *= 0.1f; }
            } else if (c == '.') {
                has_dot = true;
            } else {
                break;
            }
        }
        // Figure out the suffix ("%" / "x" / neither) to interpret the unit.
        char suffix = 0;
        while (i < tail.size() && (tail[i] == ' ' || tail[i] == '\t')) ++i;
        if (i < tail.size()) suffix = tail[i];

        if (is_bet) {
            // Bet X or Bet X% → X percent of node_pot
            ps.target_amount = node_pot * (num / 100.0f);
            ps.has_amount = true;
        } else {
            // Raise X / Raise X% → X percent of node_pot (backend native)
            // Raise Xx → X multiplied by the facing bet (frontend convention)
            if (suffix == 'x' || suffix == 'X') {
                ps.target_amount = node_bet_into * num;
            } else {
                ps.target_amount = node_pot * (num / 100.0f);
            }
            ps.has_amount = true;
        }
        return ps;
    };

    // Path B: parse optional "#<card>" runout token off each step. The card
    // tells us which canonical chance child to descend into when we cross a
    // chance node. Without it, we fall back to lex-min (legacy behavior).
    auto parse_card_token = [](const std::string& tok) -> int {
        if (tok.size() < 2) return -1;
        static const char RANK_CH[] = "23456789TJQKA";
        static const char SUIT_CH[] = "cdhs";
        int r = -1, s = -1;
        for (int i = 0; i < 13; ++i) if (tok[0] == RANK_CH[i]) { r = i; break; }
        for (int i = 0; i < 4;  ++i) if (tok[1] == SUIT_CH[i]) { s = i; break; }
        if (r < 0 || s < 0) return -1;
        return r * 4 + s;
    };

    // pending_runout_card carries a runout choice from the PREVIOUS step
    // through the next chance-skip. -1 = use lex-min default.
    auto skip_chance_with_runout = [&](uint32_t& node, int runout_card) {
        while (static_cast<NodeType>(tree_.node_types[node]) == NodeType::CHANCE) {
            uint8_t nch = tree_.num_children[node];
            if (nch == 0) break;
            uint32_t off = tree_.children_offset[node];
            uint32_t pick = tree_.children[off];  // lex-min default
            if (runout_card >= 0) {
                for (uint8_t k = 0; k < nch; ++k) {
                    uint32_t child = tree_.children[off + k];
                    uint8_t dc = (child < tree_.dealt_card.size())
                        ? tree_.dealt_card[child] : 0xFFu;
                    if (dc == static_cast<uint8_t>(runout_card)) {
                        pick = child;
                        break;
                    }
                }
                runout_card = -1;  // consume the choice on the first chance level only
            }
            node = pick;
        }
    };

    uint32_t current_node = 0;
    int pending_runout_card = -1;
    for (const std::string& step_raw : steps) {
        // Split off "#<card>" suffix if present. The card applies to the
        // NEXT chance-skip (which happens at the top of the NEXT iteration
        // when current_node is CHANCE), so save it as `pending_runout_card`.
        std::string step = step_raw;
        int step_runout_card = -1;
        size_t hash = step_raw.find('#');
        if (hash != std::string::npos) {
            step = step_raw.substr(0, hash);
            step_runout_card = parse_card_token(step_raw.substr(hash + 1));
        }

        if (static_cast<NodeType>(tree_.node_types[current_node]) == NodeType::TERMINAL) break;
        // Skip any CHANCE nodes here, applying the pending runout choice from
        // the PRIOR step (if it carried "#<card>"). On the first chance level
        // we use the choice; subsequent chance levels still go lex-min.
        skip_chance_with_runout(current_node, pending_runout_card);
        pending_runout_card = step_runout_card;
        auto labels = get_action_labels_at(current_node);

        // Attempt 1: exact string match (preserves compat with history strings
        // we ourselves generate, e.g. from node locks).
        bool found = false;
        for (uint8_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == step) {
                current_node = tree_.children[tree_.children_offset[current_node] + i];
                found = true;
                break;
            }
        }
        if (found) continue;

        // Attempt 2: fuzzy match by parsed (action_type, amount). Walk all
        // children of the current node and pick the one whose action type
        // matches and whose amount is closest to the parsed target.
        float node_pot      = tree_.pots[current_node];
        float node_bet_into = tree_.bet_into[current_node];
        ParsedStep ps = parse_step(step, node_pot, node_bet_into);

        uint8_t nc = tree_.num_children[current_node];
        uint32_t offset = tree_.children_offset[current_node];
        int best_i = -1;
        float best_diff = 0.0f;
        for (uint8_t i = 0; i < nc; ++i) {
            auto ct = static_cast<ActionType>(tree_.child_action_types[offset + i]);
            if (ct != ps.type) {
                // For non-amount actions we require the type to match exactly.
                // For Bet/Raise mismatches (e.g. "Bet 75%" sent, only Raise
                // options here because we're facing a bet), skip.
                continue;
            }
            float amt = tree_.child_action_amts[offset + i];
            float diff = std::fabs(amt - ps.target_amount);
            if (best_i < 0 || diff < best_diff) {
                best_i = i;
                best_diff = diff;
            }
        }
        if (best_i >= 0) {
            current_node = tree_.children[offset + best_i];
            continue;
        }

        // No match at all — bail out rather than silently navigate wrong.
        break;
    }
    // Final chance-skip — consume any trailing pending_runout_card.
    skip_chance_with_runout(current_node, pending_runout_card);
    return current_node;
}

inline std::string Solver::acting_player_at(uint32_t node_idx) const {
    if (node_idx >= tree_.total_nodes) return "";
    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);
    if (nt == NodeType::PLAYER_OOP) return "OOP";
    if (nt == NodeType::PLAYER_IP)  return "IP";
    return "";
}

inline Solver::OpponentRangeResult
Solver::extract_opponent_range_at(uint32_t node_idx) const
{
    OpponentRangeResult out{};
    if (!solved_ || node_idx >= tree_.total_nodes) return out;
    auto nt = static_cast<NodeType>(tree_.node_types[node_idx]);
    if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) return out;

    // Opponent = the non-acting player
    bool acting_is_ip = (nt == NodeType::PLAYER_IP);
    int opponent = acting_is_ip ? 0 : 1;  // 0=OOP, 1=IP
    out.opponent = acting_is_ip ? "OOP" : "IP";

    // Reconstruct path from root to node_idx via parent_indices.
    // Each entry = (parent_node, child_index_at_parent).
    std::vector<std::pair<uint32_t, uint8_t>> path;
    uint32_t cur = node_idx;
    while (cur != 0) {
        uint32_t parent = tree_.parent_indices[cur];
        if (parent >= tree_.total_nodes) break;
        uint8_t nchild = tree_.num_children[parent];
        uint32_t off   = tree_.children_offset[parent];
        uint8_t found_i = 0xFF;
        for (uint8_t i = 0; i < nchild; ++i) {
            if (tree_.children[off + i] == cur) { found_i = i; break; }
        }
        if (found_i == 0xFF) break;  // tree inconsistency — bail
        path.push_back({parent, found_i});
        cur = parent;
    }
    std::reverse(path.begin(), path.end());

    // Walk forward from root, compounding each player's reach according to
    // their strategy at their own nodes (opponent's strategy shapes opponent's
    // reach; traverser's strategy shapes traverser's reach — see CFR convention).
    uint16_t nc = iso_.num_canonical;
    std::vector<float> roop = oop_reach_;  // prior
    std::vector<float> rip  = ip_reach_;
    for (auto& [n, a] : path) {
        auto ntn = static_cast<NodeType>(tree_.node_types[n]);
        if (ntn == NodeType::CHANCE) continue;
        if (ntn != NodeType::PLAYER_OOP && ntn != NodeType::PLAYER_IP) continue;
        bool is_ip_node = (ntn == NodeType::PLAYER_IP);
        auto& reach = is_ip_node ? rip : roop;
        if (n >= strategy_.size() || strategy_[n].empty()) continue;
        for (uint16_t c = 0; c < nc; ++c) {
            size_t idx = static_cast<size_t>(a) * nc + c;
            if (idx < strategy_[n].size()) {
                reach[c] *= strategy_[n][idx];
            }
        }
    }

    const auto& reach = (opponent == 0) ? roop : rip;

    // Aggregate per grid-label: max reach across that label's canonical slots.
    // Using max (not mean) keeps the display intuitive — a label shows full
    // color if ANY of its suited/offsuit variants still reaches.
    const auto& combo_table = get_combo_table();
    std::map<std::string, float> label_w;
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        float w = (ci < reach.size()) ? reach[ci] : 0.0f;
        if (w <= 0.0f) continue;
        std::string label = combo_to_grid_label(combo_table[i]);
        auto it = label_w.find(label);
        if (it == label_w.end() || w > it->second) label_w[label] = w;
    }

    // Normalize to the heaviest label at this node so the UI always has a
    // clear strongest color = 1.0.
    float max_w = 0.0f;
    for (auto& [l, w] : label_w) if (w > max_w) max_w = w;
    if (max_w <= 0.0f) return out;

    out.labels.reserve(label_w.size());
    for (auto& [l, w] : label_w) {
        out.labels.push_back({l, w / max_w});
    }
    std::sort(out.labels.begin(), out.labels.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    return out;
}

inline ComboAnalysis Solver::analyze_combo(const std::string& combo_str) const {
    ComboAnalysis analysis;
    analysis.combo_str = combo_str;
    if (!solved_ || combo_str.size() != 4) return analysis;

    try {
        Card c0 = parse_card(combo_str.substr(0, 2));
        Card c1 = parse_card(combo_str.substr(2, 2));
        Combo combo(c0, c1);
        uint16_t combo_idx = combo.index();
        uint16_t canonical = iso_.original_to_canonical[combo_idx];
        if (canonical == UINT16_MAX) return analysis;

        analysis.combo = {c0, c1};
        uint8_t n_act = tree_.num_children[0];
        auto labels = get_action_labels_at(0);
        float best_freq = -1;

        for (uint8_t a = 0; a < n_act; ++a) {
            size_t idx = a * iso_.num_canonical + canonical;
            float freq = (idx < strategy_[0].size()) ? strategy_[0][idx] : 0;
            analysis.strategy_mix.push_back({labels[a], freq * 100.0f});
            if (freq > best_freq) {
                best_freq = freq;
                analysis.best_action = labels[a];
            }
        }

        if (canonical < ev_.size()) {
            analysis.ev = ev_[canonical];
        }
    } catch (...) {}

    return analysis;
}

// ============================================================================
// build_strategy_tree: cache full per-node strategies for client-side nav
// ============================================================================

inline std::map<std::string, Solver::StrategyTreeEntry>
Solver::build_strategy_tree(int max_player_depth, StrategyTreeEvMode ev_mode,
                             uint32_t max_nodes, bool* out_truncated) const {
    std::map<std::string, StrategyTreeEntry> out;
    if (out_truncated) *out_truncated = false;
    if (!solved_) return out;

    const bool need_evs = (ev_mode != StrategyTreeEvMode::NONE);

    // Sprint 1: resolve the emitted-node cap. 0 = inherit from config.
    // Final 0 = unlimited (legacy behavior). We compare against `out.size()`
    // inside the walk lambda below.
    uint32_t effective_max_nodes = max_nodes;
    if (effective_max_nodes == 0) {
        effective_max_nodes = config_.memory_budget.strategy_tree_max_nodes;
    }
    bool truncated = false;

    // Phase 3 (10-point plan): cap the EV cache size by computing the set of
    // nodes the strategy tree will actually emit BEFORE running the EV pass,
    // then telling cpu_ev_traverse to drop everything else. The pre-walk
    // mirrors the recursion structure of the main `walk` below but only
    // collects node ids — no allocation per node — so it's effectively free.
    //
    // FULL mode skips this and records every visited node, matching the
    // pre-Phase-3 behavior (kept as an escape hatch for tests / debugging).
    std::set<uint32_t> visible_nodes;
    if (need_evs && ev_mode == StrategyTreeEvMode::VISIBLE) {
        std::function<void(uint32_t, int, int)> precount;
        precount = [&](uint32_t node, int player_depth, int chance_levels_seen) {
            // Mirror the chance-handling in `walk`: at first chance level
            // enumerate every runout option; afterwards auto-skip to lex-min.
            int chance_seen = chance_levels_seen;
            while (node < tree_.total_nodes &&
                   static_cast<NodeType>(tree_.node_types[node]) == NodeType::CHANCE) {
                uint8_t nch = tree_.num_children[node];
                if (nch == 0) return;
                if (chance_seen == 0) {
                    uint32_t off = tree_.children_offset[node];
                    bool any_real = false;
                    for (uint8_t k = 0; k < nch; ++k) {
                        uint32_t child = tree_.children[off + k];
                        uint8_t dc = (child < tree_.dealt_card.size())
                            ? tree_.dealt_card[child] : 0xFFu;
                        if (dc == 0xFFu) continue;
                        any_real = true;
                        precount(child, player_depth, chance_seen + 1);
                    }
                    if (any_real) return;
                    // single-option chance: fall through to auto-skip
                    uint32_t child = tree_.children[off];
                    node = child;
                    ++chance_seen;
                    continue;
                }
                uint32_t child = tree_.children[tree_.children_offset[node]];
                node = child;
                ++chance_seen;
            }
            if (node >= tree_.total_nodes) return;
            auto nt = static_cast<NodeType>(tree_.node_types[node]);
            if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) return;
            visible_nodes.insert(node);
            if (player_depth >= max_player_depth) return;
            uint8_t na = tree_.num_children[node];
            uint32_t off = tree_.children_offset[node];
            for (uint8_t a = 0; a < na; ++a) {
                precount(tree_.children[off + a], player_depth + 1, chance_seen);
            }
        };
        precount(0, 0, 0);
    }

    // Pre-compute per-node EVs from each player's perspective by running
    // cpu_ev_traverse from root with capture maps. One pass per perspective
    // gets values for the visible-or-all set — depending on ev_mode.
    std::map<uint32_t, std::vector<float>> node_vals_oop;
    std::map<uint32_t, std::vector<float>> node_vals_ip;
    if (need_evs) {
        const std::set<uint32_t>* filter = nullptr;
        if (ev_mode == StrategyTreeEvMode::VISIBLE) {
            filter = &visible_nodes;
        }
        auto roop = oop_reach_, rip = ip_reach_;
        cpu_ev_traverse(0, 0, roop, rip, &node_vals_oop, filter);
        roop = oop_reach_; rip = ip_reach_;
        cpu_ev_traverse(0, 1, roop, rip, &node_vals_ip, filter);
    }

    // Aggregate per-canonical-combo values into per-grid-label EVs at one
    // node. Uses the acting player's reach as weights so labels with no
    // reach at this node contribute zero (rather than skewing the mean).
    // Normalize raw cpu_ev_traverse values: they're scaled by total opponent
    // reach × canonical_weight (so a wider opponent range inflates the
    // numbers proportionally). Divide by that total to get chips per hand.
    auto opp_total_weight = [&](bool acting_is_ip) -> float {
        const auto& opp = acting_is_ip ? oop_reach_ : ip_reach_;
        float total = 0.0f;
        for (uint16_t c = 0; c < iso_.num_canonical; ++c) {
            float w = static_cast<float>(iso_.canonical_weights[c]);
            total += opp[c] * w;
        }
        return total;
    };
    float oop_norm = 1.0f / std::max(1e-6f, opp_total_weight(false)); // OOP acting → IP opp
    float ip_norm  = 1.0f / std::max(1e-6f, opp_total_weight(true));  // IP acting  → OOP opp

    auto evs_at = [&](uint32_t node) -> std::vector<std::pair<std::string, float>> {
        if (!need_evs) return {};
        auto nt = static_cast<NodeType>(tree_.node_types[node]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) return {};
        bool acting_is_ip = (nt == NodeType::PLAYER_IP);
        const auto& vals_map = acting_is_ip ? node_vals_ip : node_vals_oop;
        auto it = vals_map.find(node);
        if (it == vals_map.end()) return {};
        const std::vector<float>& vals = it->second;
        float norm = acting_is_ip ? ip_norm : oop_norm;

        const auto& reach = acting_is_ip ? ip_reach_ : oop_reach_;
        const auto& combo_table = get_combo_table();
        std::map<std::string, float> sum_w_val;
        std::map<std::string, float> sum_w;
        for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
            uint16_t ci = iso_.original_to_canonical[i];
            if (ci == UINT16_MAX) continue;
            float r = (ci < reach.size()) ? reach[ci] : 0.0f;
            if (r <= 0.0f) continue;
            std::string label = combo_to_grid_label(combo_table[i]);
            float v = (ci < vals.size()) ? vals[ci] : 0.0f;
            sum_w_val[label] += r * v * norm;
            sum_w[label]     += r;
        }
        std::vector<std::pair<std::string, float>> out_pairs;
        for (auto& [label, sw] : sum_w) {
            if (sw <= 0.0f) continue;
            out_pairs.push_back({label, sum_w_val[label] / sw});
        }
        return out_pairs;
    };

    // Path token formatter: cards encode as "<rank><suit>" lowercase
    // (e.g. card 0 = "2c"). Used in the cache key for runout selection.
    // Format chosen to be human-readable in JSON output and easy to parse
    // by the frontend.
    static const char RANK_CH[] = "23456789TJQKA";
    static const char SUIT_CH[] = "cdhs";
    auto card_token = [](uint8_t c) -> std::string {
        std::string s; s += RANK_CH[c / 4]; s += SUIT_CH[c % 4]; return s;
    };

    // Helper: for a CHANCE node, build the list of canonical runout options
    // the UI can present. Reads dealt_card + runout_weight from the SoA
    // tree (populated by GameTreeBuilder during iso enumeration). Skips
    // children with dealt_card == 0xFF (legacy single-child fallback).
    auto chance_options = [&](uint32_t chance_node) -> std::vector<RunoutOption> {
        std::vector<RunoutOption> opts;
        uint8_t nch = tree_.num_children[chance_node];
        uint32_t off = tree_.children_offset[chance_node];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = tree_.children[off + k];
            uint8_t dc = (child < tree_.dealt_card.size()) ? tree_.dealt_card[child] : 0xFFu;
            if (dc == 0xFFu) continue;  // legacy fallback: not a real runout choice
            uint8_t w  = (child < tree_.runout_weight.size()) ? tree_.runout_weight[child] : 1;
            opts.push_back({dc, w});
        }
        return opts;
    };

    // Recursive walker. `path` is the cache key built from player-action
    // labels (comma-separated) plus optional "#<card>" tokens marking the
    // user's runout choice at the FIRST chance level. Subsequent chance
    // levels still auto-skip to lex-min (cache size: enumerating all chance
    // levels would explode for flop solves with iso engaged).
    //
    // `dealt_cards_so_far` accumulates cards picked along this path; copied
    // into each cached entry for UI disclosure.
    // `runouts_for_first_chance` is a snapshot of options at THIS sub-tree's
    // root chance step, also copied into each entry so the runout picker UI
    // can render from any node within the same runout subtree.
    // `chance_levels_seen` controls when to enumerate vs auto-skip.
    std::function<void(uint32_t, const std::string&,
                       const std::vector<uint8_t>&,
                       const std::vector<RunoutOption>&,
                       int, int)> walk;
    walk = [&](uint32_t node, const std::string& path,
               const std::vector<uint8_t>& dealt_cards_so_far,
               const std::vector<RunoutOption>& runouts_for_first_chance,
               int player_depth, int chance_levels_seen) {
        // Process CHANCE nodes inline.
        std::vector<uint8_t> dealt_now = dealt_cards_so_far;
        std::vector<RunoutOption> runouts_now = runouts_for_first_chance;
        int chance_seen = chance_levels_seen;

        while (node < tree_.total_nodes &&
               static_cast<NodeType>(tree_.node_types[node]) == NodeType::CHANCE) {
            uint8_t nch = tree_.num_children[node];
            if (nch == 0) return;

            // First chance encountered (per sub-tree from root): enumerate
            // every canonical runout option. Lex-min canonical (the FIRST
            // child in tree order) is also emitted at the NO-SUFFIX key so
            // existing nav code that doesn't know about runout selection
            // still hits cache. Other reps get "#<card>" suffix keys.
            if (chance_seen == 0) {
                auto opts = chance_options(node);
                if (opts.size() > 1) {
                    uint32_t off = tree_.children_offset[node];
                    bool is_lex_min = true;  // first iter = first child = lex-min
                    for (uint8_t k = 0; k < nch; ++k) {
                        uint32_t child = tree_.children[off + k];
                        uint8_t dc = (child < tree_.dealt_card.size())
                            ? tree_.dealt_card[child] : 0xFFu;
                        if (dc == 0xFFu) continue;
                        std::vector<uint8_t> d = dealt_now;
                        d.push_back(dc);
                        std::string new_path = is_lex_min
                            ? path
                            : (path + "#" + card_token(dc));
                        walk(child, new_path, d, opts,
                             player_depth, chance_seen + 1);
                        is_lex_min = false;
                    }
                    return;
                }
                // Single-option chance: degrade to auto-skip with the one
                // child. dealt_card may be 0xFF (legacy) — don't append.
                uint32_t child = tree_.children[off_of(node)];
                uint8_t dc = (child < tree_.dealt_card.size())
                    ? tree_.dealt_card[child] : 0xFFu;
                if (dc != 0xFFu) dealt_now.push_back(dc);
                node = child;
                ++chance_seen;
                continue;
            }
            // Subsequent chance levels: auto-skip to first (lex-min) child.
            uint32_t child = tree_.children[off_of(node)];
            uint8_t dc = (child < tree_.dealt_card.size())
                ? tree_.dealt_card[child] : 0xFFu;
            if (dc != 0xFFu) dealt_now.push_back(dc);
            node = child;
            ++chance_seen;
        }
        if (node >= tree_.total_nodes) return;

        auto nt = static_cast<NodeType>(tree_.node_types[node]);
        if (nt != NodeType::PLAYER_OOP && nt != NodeType::PLAYER_IP) return;
        if (out.count(path)) return;  // already cached (shared sub-path)

        // Sprint 1: emitted-node cap. Once we've cached `effective_max_nodes`
        // unique entries, refuse to add more — any further `walk` recursion
        // returns immediately. We mark `truncated` so the caller can show
        // "tree truncated at N nodes" in the UI instead of silently dropping
        // branches.
        if (effective_max_nodes > 0 && out.size() >= effective_max_nodes) {
            truncated = true;
            return;
        }

        StrategyTreeEntry entry;
        entry.acting          = acting_player_at(node);
        entry.action_labels   = get_action_labels_at(node);
        entry.global_strategy = extract_global_strategy_at(node);
        auto all_combo = extract_combo_strategies_at(node);
        for (auto& [label, mix] : all_combo) {
            if (label.size() <= 3) {
                entry.combo_strategies.push_back({label, mix});
            }
        }
        auto opp = extract_opponent_range_at(node);
        entry.opponent_side  = opp.opponent;
        entry.opponent_range = std::move(opp.labels);
        entry.combo_evs      = evs_at(node);
        entry.dealt_cards    = dealt_now;
        entry.runout_options = runouts_now;

        std::vector<std::string> labels_local = entry.action_labels;
        out[path] = std::move(entry);

        if (player_depth >= max_player_depth) return;

        uint8_t na = tree_.num_children[node];
        uint32_t off = tree_.children_offset[node];
        for (uint8_t a = 0; a < na && a < labels_local.size(); ++a) {
            uint32_t child = tree_.children[off + a];
            const std::string& alabel = labels_local[a];
            std::string new_path = path.empty() ? alabel : (path + "," + alabel);
            walk(child, new_path, dealt_now, runouts_now,
                 player_depth + 1, chance_seen);
        }
    };

    std::vector<uint8_t> empty_cards;
    std::vector<RunoutOption> empty_runouts;
    walk(0, "", empty_cards, empty_runouts, 0, 0);
    if (out_truncated) *out_truncated = truncated;
    return out;
}

// Tiny helper used only inside the walk lambda. children_offset(node) is
// what we want — wrapped here because the lambda accesses tree_ a lot and
// it's clearer with a name.
inline uint32_t Solver::off_of(uint32_t node) const {
    return tree_.children_offset[node];
}

} // namespace deepsolver
