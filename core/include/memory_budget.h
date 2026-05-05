/**
 * @file memory_budget.h
 * @brief Centralized memory budget + footprint estimation.
 *
 * Phase 1 of the 10-point maturity plan. Before this header existed each
 * subsystem (tree builder, matchup precompute, GPU backend, strategy-tree
 * cache) made its own ad-hoc memory decision (e.g. game_tree_builder.h had
 * a hardcoded `projected <= 2000` runout-table cap). This file is the one
 * place where we say:
 *
 *   - How much host RAM are we allowed to use
 *   - How much GPU VRAM are we allowed to use
 *   - How big can a JSON response be
 *   - How many emitted strategy-tree nodes are tolerable
 *
 * And it provides the `bytes_for_*` helpers that every caller should run
 * BEFORE allocating, so failures become structured errors instead of
 * `std::bad_alloc` aborts.
 *
 * Header-only on purpose: every estimator is constexpr-ish or a tiny pure
 * function, so the cost of including is small. If we eventually grow
 * runtime policy logic (probe free VRAM, read env vars, etc.) we should
 * split it into memory_budget.cpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace deepsolver {

// ============================================================================
// Constants — tune here, not at the call site
// ============================================================================

namespace memory_budget {

/// Default host RAM budget when no override is given. 6 GB. Chosen so a
/// solve fits comfortably alongside the Tauri webview, the OS, and a
/// browser tab full of GTO Wizard reference data on a 16 GB laptop.
constexpr uint64_t kDefaultHostBudgetBytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;

/// Default cap on emitted strategy-tree nodes. Each emitted node carries
/// per-grid-label arrays (~169 entries) so 2000 nodes ≈ 338k entries of
/// JSON which still serializes in well under 100 ms.
constexpr uint32_t kDefaultStrategyTreeMaxNodes = 2000;

/// Default cap on the JSON response size we'll happily emit. 100 MB is
/// enormous — the real ceiling is the Tauri IPC bridge, which grinds on
/// payloads larger than ~40 MB. Keeping the cap conservative avoids
/// silent UI freezes.
constexpr uint64_t kDefaultJsonResponseBudgetBytes = 100ULL * 1024ULL * 1024ULL;

/// Fraction of free VRAM we're willing to claim when we can probe it.
/// 75% leaves headroom for cuBLAS workspaces, kernel scratch, and the
/// driver itself.
constexpr float kGpuVramReservedFraction = 0.75f;

/// Each matchup table is a pair of float matrices: EV (nc × nc) and valid
/// (nc × nc). We track the "× 2" explicitly so the formula is auditable.
constexpr uint64_t kMatchupMatricesPerRunout = 2;

/// CPU backend keeps three float arrays per (player_node × max_actions × nc):
/// regrets, strategy_sum, current_strategy. The accumulating action_values
/// is a fourth, but we pessimistically count three for the steady-state,
/// transient Phase-1 buffers, plus a small slack.
constexpr uint64_t kCpuStateArraysPerNode = 3;

/// GPU backend keeps regrets, strategy_sum, current_strategy, action_values
/// (4 N×A×nc), reach_scratch_oop, reach_scratch_ip (2 N×nc), node_values
/// (N×nc). We sum these explicitly in `gpu_state_bytes`.

} // namespace memory_budget

// ============================================================================
// Budget — what we're allowed to spend
// ============================================================================

struct MemoryBudget {
    /// Host RAM ceiling (bytes). 0 = unlimited (don't gate on host RAM).
    uint64_t host_bytes = memory_budget::kDefaultHostBudgetBytes;

    /// GPU VRAM ceiling (bytes). 0 = let GPU backend probe at runtime.
    uint64_t gpu_bytes = 0;

    /// JSON response ceiling (bytes). Soft warning, not a hard error.
    uint64_t json_bytes = memory_budget::kDefaultJsonResponseBudgetBytes;

    /// Cap on emitted strategy-tree nodes (Phase 3 ties into this).
    uint32_t strategy_tree_max_nodes = memory_budget::kDefaultStrategyTreeMaxNodes;

    /// Built-in defaults. Convenience constructor.
    static MemoryBudget defaults() { return MemoryBudget{}; }
};

// ============================================================================
// Footprint estimate — what a planned solve would actually spend
// ============================================================================

struct SolveFootprintEstimate {
    uint64_t matchup_tables_bytes      = 0;
    uint64_t cpu_state_bytes           = 0;
    uint64_t gpu_state_bytes           = 0;
    uint64_t strategy_tree_ev_bytes    = 0;
    uint64_t json_response_bytes       = 0;

    /// Sum of all host-side estimates (matchup + CPU state + strategy-tree
    /// EV cache + JSON). We exclude GPU state — that lives in VRAM.
    uint64_t total_host_bytes() const {
        return matchup_tables_bytes
             + cpu_state_bytes
             + strategy_tree_ev_bytes
             + json_response_bytes;
    }
};

// ============================================================================
// Estimators — tiny pure functions, all in bytes
// ============================================================================

/// Bytes required to hold all per-runout matchup tables.
///
///   runout_tables × nc² × 2 (EV+valid) × sizeof(float)
///
/// `runout_tables` is the number of distinct board signatures the tree
/// will reference. `nc` is iso.num_canonical (canonical combo count).
/// Pass these from the caller — this header doesn't see the IsomorphismMapping.
inline uint64_t bytes_for_matchup_tables(uint64_t runout_tables, uint64_t nc) {
    return runout_tables
         * nc * nc
         * memory_budget::kMatchupMatricesPerRunout
         * sizeof(float);
}

/// Bytes for the CPU CFR state. Counts regrets + strategy_sum + current_strategy
/// (three arrays). action_values is a transient stack-allocated vector in
/// CPU traversal, not heap.
inline uint64_t bytes_for_cpu_state(uint64_t player_nodes, uint64_t max_actions,
                                     uint64_t nc) {
    return player_nodes * max_actions * nc
         * memory_budget::kCpuStateArraysPerNode
         * sizeof(float);
}

/// Bytes for the GPU CFR state. Lives in VRAM. Includes:
///   - regrets, strategy_sum, current_strategy, action_values: each N×A×nc floats
///   - reach_scratch_oop, reach_scratch_ip: each N×nc floats
///   - node_values: N×nc floats
/// Total = (4·A + 3) · N · nc · sizeof(float)
inline uint64_t bytes_for_gpu_state(uint64_t total_nodes, uint64_t max_actions,
                                     uint64_t nc) {
    const uint64_t per_node = (4ULL * max_actions + 3ULL) * nc;
    return per_node * total_nodes * sizeof(float);
}

/// Bytes for the strategy-tree EV cache. The current implementation
/// stores `<node_id, vector<float>>` for every visited inner node ×
/// 2 perspectives (OOP-acting and IP-acting). The vector length is `nc`.
/// We approximate the std::map overhead at 2x the raw vector data —
/// red-black tree nodes plus heap headers.
inline uint64_t bytes_for_strategy_tree_ev_cache(uint64_t cached_nodes, uint64_t nc,
                                                  uint64_t perspectives = 2) {
    const uint64_t raw = cached_nodes * nc * perspectives * sizeof(float);
    return raw * 2; // ×2 to approximate map<uint32_t, vector<float>> overhead.
}

/// Rough JSON response size estimator. Strategy tree dominates: each
/// emitted node carries ~5 lists × 169 grid labels × ~30 bytes/entry.
inline uint64_t bytes_for_json_response(uint64_t emitted_nodes,
                                         uint64_t opponent_range_entries = 169,
                                         uint64_t combo_strategies_count = 169) {
    constexpr uint64_t kBytesPerEntry = 32;       // "AA":"50.0%",  rough average
    constexpr uint64_t kNodeOverhead  = 256;      // path string + acting + dealt_cards
    const uint64_t per_node = kNodeOverhead
                            + opponent_range_entries * kBytesPerEntry
                            + combo_strategies_count * kBytesPerEntry * 2; // strategy + EV
    return emitted_nodes * per_node;
}

// ============================================================================
// Decisions
// ============================================================================

enum class BudgetDecision : uint8_t {
    OK              = 0, ///< All estimates within budget — proceed.
    REDUCE_RUNOUTS  = 1, ///< Matchup tables would blow host budget.
    REDUCE_TREE     = 2, ///< Strategy tree emitted nodes too large.
    REDUCE_JSON     = 3, ///< Final JSON response would exceed cap.
    GPU_OOM_LIKELY  = 4, ///< GPU state estimate exceeds VRAM budget.
    HOST_OOM_LIKELY = 5, ///< Host total exceeds host budget.
};

inline const char* budget_decision_str(BudgetDecision d) {
    switch (d) {
        case BudgetDecision::OK:              return "ok";
        case BudgetDecision::REDUCE_RUNOUTS:  return "reduce_runouts";
        case BudgetDecision::REDUCE_TREE:     return "reduce_tree";
        case BudgetDecision::REDUCE_JSON:     return "reduce_json";
        case BudgetDecision::GPU_OOM_LIKELY:  return "gpu_oom_likely";
        case BudgetDecision::HOST_OOM_LIKELY: return "host_oom_likely";
    }
    return "unknown";
}

/// Decide whether the planned solve fits the budget.
///
/// Order of severity (returned first wins): GPU > matchup > host total
/// > strategy tree > JSON. We return the most severe offender so the
/// caller's error message points at the actual blocker.
inline BudgetDecision evaluate_budget(const SolveFootprintEstimate& est,
                                       const MemoryBudget& budget) {
    if (budget.gpu_bytes > 0 && est.gpu_state_bytes > budget.gpu_bytes) {
        return BudgetDecision::GPU_OOM_LIKELY;
    }
    if (budget.host_bytes > 0 && est.matchup_tables_bytes > budget.host_bytes) {
        // Single-component blowout — flag it specifically so the caller
        // can offer "fewer runouts / fewer bet sizes" rather than a generic
        // "use less RAM".
        return BudgetDecision::REDUCE_RUNOUTS;
    }
    if (budget.host_bytes > 0 && est.total_host_bytes() > budget.host_bytes) {
        return BudgetDecision::HOST_OOM_LIKELY;
    }
    if (est.strategy_tree_ev_bytes > 0 &&
        budget.strategy_tree_max_nodes > 0 &&
        est.strategy_tree_ev_bytes > 4ULL * 1024 * 1024 * 1024) {
        // Hard guard: 4 GB of EV cache is never reasonable.
        return BudgetDecision::REDUCE_TREE;
    }
    if (budget.json_bytes > 0 && est.json_response_bytes > budget.json_bytes) {
        return BudgetDecision::REDUCE_JSON;
    }
    return BudgetDecision::OK;
}

/// Build a human-readable diagnostic line. Shape matches the format the
/// CLI's error JSON expects so the message can be embedded directly:
///
///   "Matchup precompute would require 5.2 GB, exceeding host budget 3.0 GB."
inline std::string format_budget_failure(BudgetDecision d,
                                          const SolveFootprintEstimate& est,
                                          const MemoryBudget& budget) {
    auto gb = [](uint64_t b) -> std::string {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
        return std::string(buf);
    };
    auto mb = [](uint64_t b) -> std::string {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(b) / (1024.0 * 1024.0));
        return std::string(buf);
    };

    switch (d) {
        case BudgetDecision::OK:
            return "ok";
        case BudgetDecision::REDUCE_RUNOUTS:
            return "Matchup precompute would require " + gb(est.matchup_tables_bytes)
                 + ", exceeding host budget " + gb(budget.host_bytes)
                 + ". Use fewer runouts (skip flop-level enumeration), fewer bet sizes, or a smaller iso-bucket count.";
        case BudgetDecision::HOST_OOM_LIKELY:
            return "Host RAM estimate " + gb(est.total_host_bytes())
                 + " exceeds budget " + gb(budget.host_bytes)
                 + " (matchup=" + gb(est.matchup_tables_bytes)
                 + ", cpu_state=" + gb(est.cpu_state_bytes)
                 + ", strategy_tree=" + gb(est.strategy_tree_ev_bytes)
                 + ", json=" + mb(est.json_response_bytes) + ").";
        case BudgetDecision::GPU_OOM_LIKELY:
            return "GPU state estimate " + gb(est.gpu_state_bytes)
                 + " exceeds VRAM budget " + gb(budget.gpu_bytes)
                 + ". Falling back to CPU is recommended.";
        case BudgetDecision::REDUCE_TREE:
            return "Strategy-tree EV cache estimate " + gb(est.strategy_tree_ev_bytes)
                 + " is unreasonably large. Use --strategy-tree-evs visible or none.";
        case BudgetDecision::REDUCE_JSON:
            return "JSON response estimate " + mb(est.json_response_bytes)
                 + " exceeds budget " + mb(budget.json_bytes)
                 + ". Reduce strategy-tree depth or disable combo EVs.";
    }
    return "unknown";
}

} // namespace deepsolver
