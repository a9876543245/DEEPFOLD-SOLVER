/**
 * @file terminal_plan.h
 * @brief TerminalRepresentationPlan — the ONE decision for how showdown
 *        terminals are represented, shared by every consumer.
 *
 * FABLE_SOLVER_OPTIMIZATION_REPORT §5.1 / PR-4. Before this header the same
 * question — "does this solve use the rank-blocker, the signed-count matrix,
 * or dense category/valid tables?" — was answered independently in four
 * places, and they disagreed in exactly the way the Phase-0 review round 2
 * predicted:
 *
 *   - GpuBackend::prepare (rank_blocker_activatable → dense-upload skip)
 *   - matchup_bytes_per_cell (host 9 vs 10 B/cell)
 *   - LevelizedCpuBackend::prepare (per-traverser kernel shortcuts)
 *   - the memory estimators, which assumed DENSE everywhere — a ~6× device
 *     over-estimate on singleton-iso boards (AsKd7c2h: est 593 MB vs 98 MB
 *     measured) that can falsely downgrade AUTO solves.
 *
 * This header does NOT change any math (report PR-4 contract). It moves the
 * decision to one function, lets the estimators consult it, and gives the
 * backends a hard consistency check so plan and behavior can never drift.
 * The production dense-path removal (host tables) is PR-5 / A4-host.
 *
 * Two-phase evaluation, because the inputs arrive at different times:
 *
 *   1. plan_terminal_representation(config, iso) — STATIC plan. Available
 *      at estimate time (needs only the iso mapping + rake config).
 *   2. refine_terminal_plan(plan, iso, ranks_per_runout) — refined at solve
 *      time after precompute_matchups(), when per-runout rank tables exist.
 *      Mirrors GpuBackend's rank_blocker_activatable() term "at least one
 *      runout has valid blocker metadata": refinement can only DEMOTE
 *      RankBlockerOnly back to the dense/signed representation (the rare
 *      no-valid-metadata edge), never promote — so the static plan is the
 *      optimistic-but-honest estimate and the refined plan is exact.
 *
 * CPU note: the levelized backend keeps richer PER-TRAVERSER kernel choices
 * (active-density gates, iso-metadata paths). Those pick HOW to consume the
 * tables this plan describes; they do not change which tables exist, so the
 * plan intentionally does not micro-manage them.
 */

#pragma once

#include "types.h"
#include "isomorphism.h"
#include "memory_budget.h"
#include "showdown_rank_blocker.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace deepsolver {

enum class TerminalRepresentation : uint8_t {
    /// Singleton-iso boards: the O(nc·~92) rank-blocker serves every
    /// showdown terminal. No dense nc² EV/valid upload on the GPU.
    RankBlockerOnly = 0,
    /// Zero-rake iso-engaged boards: host builds the signed pair-count
    /// matrix (+1 byte/cell) so CPU showdown runs a signed dot product.
    SignedCount = 1,
    /// General dense path: category/valid tables, dense GPU upload.
    DenseCategoryValid = 2,
    /// DEEPSOLVER_RB_SELFCHECK=1: rank-blocker AND dense tables both
    /// resident so the deterministic self-check can diff them. Debug only.
    DebugDenseSelfCheck = 3,
};

inline const char* terminal_representation_label(TerminalRepresentation r) {
    switch (r) {
        case TerminalRepresentation::RankBlockerOnly:     return "rank_blocker_only";
        case TerminalRepresentation::SignedCount:         return "signed_count";
        case TerminalRepresentation::DenseCategoryValid:  return "dense_category_valid";
        case TerminalRepresentation::DebugDenseSelfCheck: return "debug_dense_selfcheck";
    }
    return "unknown";
}

struct TerminalRepresentationPlan {
    TerminalRepresentation representation =
        TerminalRepresentation::DenseCategoryValid;

    /// GPU uploads the dense per-runout EV+valid float matrices
    /// (tables × nc² × 8 B). False only when the rank-blocker serves every
    /// terminal — the A4-GPU dense-upload skip.
    bool device_dense_upload = true;

    /// Host bytes per matchup cell (EV + valid + category, plus the int8
    /// signed count under SignedCount). Exactly what matchup_bytes_per_cell
    /// returned before this header existed.
    uint64_t host_bytes_per_cell = memory_budget::kBaseMatchupBytesPerCell;

    /// False until refine_terminal_plan() ran with real rank tables. The
    /// estimators run on the static plan; gates and backends on the refined
    /// one.
    bool refined_with_ranks = false;

    const char* label() const {
        return terminal_representation_label(representation);
    }
};

/// The DEEPSOLVER_RB_SELFCHECK escape hatch, read once. Kept here so every
/// consumer sees the same answer (GpuBackend used to read it privately).
inline bool terminal_selfcheck_forced() {
    static const bool forced = (std::getenv("DEEPSOLVER_RB_SELFCHECK") != nullptr);
    return forced;
}

/// STATIC plan — estimate-time decision from config + iso alone. Decision
/// table (mirrors the pre-PR-4 scattered logic exactly):
///
///   singleton-iso                → RankBlockerOnly   (no dense upload)
///     ... + RB_SELFCHECK env     → DebugDenseSelfCheck (dense upload forced)
///   rake == 0 (iso-engaged)      → SignedCount       (host +1 B/cell)
///   otherwise                    → DenseCategoryValid
///
/// host_bytes_per_cell reproduces matchup_bytes_per_cell():
/// signed ⇒ kSignedCountMatchupBytesPerCell, else kBaseMatchupBytesPerCell.
inline TerminalRepresentationPlan plan_terminal_representation(
    const SolverConfig& config,
    const IsomorphismMapping& iso)
{
    TerminalRepresentationPlan p;
    const bool singleton = showdown_rank_blocker::supports_singleton_iso(iso);
    const bool zero_rake = (config.rake_rate == 0.0f && config.rake_cap == 0.0f);

    if (singleton) {
        p.representation = terminal_selfcheck_forced()
            ? TerminalRepresentation::DebugDenseSelfCheck
            : TerminalRepresentation::RankBlockerOnly;
        p.device_dense_upload =
            (p.representation == TerminalRepresentation::DebugDenseSelfCheck);
        p.host_bytes_per_cell = memory_budget::kBaseMatchupBytesPerCell;
    } else if (zero_rake) {
        p.representation      = TerminalRepresentation::SignedCount;
        p.device_dense_upload = true;
        p.host_bytes_per_cell = memory_budget::kSignedCountMatchupBytesPerCell;
    } else {
        p.representation      = TerminalRepresentation::DenseCategoryValid;
        p.device_dense_upload = true;
        p.host_bytes_per_cell = memory_budget::kBaseMatchupBytesPerCell;
    }
    return p;
}

/// Refined plan — solve-time, after precompute_matchups() built the
/// per-runout rank tables. Mirrors rank_blocker_activatable()'s remaining
/// terms: the blocker only serves a solve when the rank tables exist for
/// every runout AND at least one runout yields valid blocker metadata.
/// When that fails on a RankBlockerOnly/SelfCheck plan, demote to the
/// representation the non-singleton branch would have picked (the dense
/// upload then really happens, and the estimators' static optimism is
/// caught by the backend pre-flight).
inline TerminalRepresentationPlan refine_terminal_plan(
    TerminalRepresentationPlan p,
    const IsomorphismMapping& iso,
    const std::vector<std::vector<uint16_t>>& ranks_per_runout,
    uint32_t num_runouts)
{
    p.refined_with_ranks = true;
    const bool blocker_planned =
        p.representation == TerminalRepresentation::RankBlockerOnly ||
        p.representation == TerminalRepresentation::DebugDenseSelfCheck;
    if (!blocker_planned) return p;

    bool activatable = false;
    if (iso.num_canonical != 0 && ranks_per_runout.size() == num_runouts) {
        for (const auto& ranks : ranks_per_runout) {
            auto meta = showdown_rank_blocker::build_metadata(iso, ranks);
            if (meta.valid && meta.bucket_count > 0) { activatable = true; break; }
        }
    }
    if (!activatable) {
        // Rare edge (no valid blocker metadata despite singleton iso):
        // fall back exactly like GpuBackend's materialize_dense=true path.
        p.representation      = TerminalRepresentation::DenseCategoryValid;
        p.device_dense_upload = true;
        p.host_bytes_per_cell = memory_budget::kBaseMatchupBytesPerCell;
    }
    return p;
}

} // namespace deepsolver
