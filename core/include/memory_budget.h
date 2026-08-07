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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

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
/// v1.8.3: zero-rake compressed CPU solves pass 3 because they also keep a
/// signed showdown coefficient matrix. The category byte matrix is slack.
constexpr uint64_t kBaseMatchupMatricesPerRunout = 2;
constexpr uint64_t kSignedMatchupMatricesPerRunout = 3;
constexpr uint64_t kMatchupMatricesPerRunout = kBaseMatchupMatricesPerRunout;
// Byte-accurate estimator constants used by bytes_for_matchup_tables().
constexpr uint64_t kMatchupCategoryBytesPerCell = sizeof(uint8_t);
constexpr uint64_t kBaseMatchupBytesPerCell =
    2ULL * sizeof(float) + kMatchupCategoryBytesPerCell;
constexpr uint64_t kSignedCountMatchupBytesPerCell =
    kBaseMatchupBytesPerCell + sizeof(int8_t);
constexpr uint64_t kMatchupBytesPerCell = kBaseMatchupBytesPerCell;

/// CPU backend keeps three strategy-shaped float arrays: regrets,
/// strategy_sum, current_strategy. Sized by ACTUAL per-node action slots
/// (Σ num_children over player nodes), not MAX_ACTIONS — see
/// bytes_for_cpu_state_compact.
constexpr uint64_t kCpuStateArraysPerNode = 3;

/// LevelizedCpuBackend pads every nc-wide row up to a SIMD lane multiple
/// (action_stride_ = round_up_to_lane(nc)). MUST stay equal to
/// LevelizedCpuBackend::kActionLane — the estimator mirrors the backend's
/// real allocation, and a drift here silently re-opens the estimate ≠
/// allocation gap this constant exists to close.
constexpr uint64_t kCpuActionLaneFloats = 8;

/// Peak-host lifetime model constants (GPT Phase-0 review P1-1). The state
/// estimator being byte-exact is NOT the same as the solve's host PEAK:
/// finalize materializes the averaged strategy TWICE (backend nested copy +
/// Solver::strategy_ deep copy at solver.h `strategy_ = backend_->strategy()`;
/// on GPU the flat download staging overlaps the nested repack at the same
/// size). Model validated 2026-07-21 against measured peak RSS:
///   mono CPU  .528 matchup + 1.470 state + 2×.236 strategy + base ≈ 2.52 GB
///             vs 2.433 GB measured   (+3.6%)
///   mono GPU  .528 + 2×.236 + context ≈ 1.16 GB vs 1.166 GB measured
///   std  GPU  .012 + .007 + context ≈ .138 GB vs .144 GB measured
/// Process baseline (CRT, iso tables, code) — small but real. It used to be
/// described as covering the tree arrays too; it does not, and cannot — see
/// `bytes_for_flat_tree` below.
constexpr uint64_t kHostProcessOverheadBytes = 96ULL * 1024ULL * 1024ULL;

/// The host-resident FlatGameTree. Thirteen per-node arrays (31 B/node) and
/// three per-edge arrays (9 B/edge), from the struct in types.h.
///
/// This was folded into the flat 96 MB process baseline until 2026-08-03,
/// which was harmless while trees were thousands of nodes and wrong once B1b
/// inc 2 let 4.65M-node trees through: 4,654,449 nodes is 186 MB of arrays
/// before the builder's push_back growth, and the 4.65M gate measured 393 MB
/// of unexplained peak RSS. The builder does not reserve, so capacity can sit
/// at up to 2× size — which is both the honest bound and what that gap says
/// (393 measured vs 372 modeled). On a 627k-node rainbow the same term
/// accounts for the CPU spot's residual −0.6% (20.6 MB measured vs 50 modeled,
/// i.e. conservative, which is the direction the gate needs).
constexpr uint64_t kFlatTreeBytesPerNode = 31;
constexpr uint64_t kFlatTreeBytesPerEdge = 9;
constexpr double   kFlatTreeGrowthFactor = 2.0;
inline uint64_t bytes_for_flat_tree(uint64_t nodes, uint64_t edges) {
    return static_cast<uint64_t>(
        static_cast<double>(nodes * kFlatTreeBytesPerNode
                          + edges * kFlatTreeBytesPerEdge)
        * kFlatTreeGrowthFactor);
}
/// CUDA runtime/context host-side RSS (driver pools, pinned staging).
/// Measured 125–140 MB on CUDA 13.2 / Win11 / RTX 5090; padded slightly.
constexpr uint64_t kGpuHostContextBytes = 160ULL * 1024ULL * 1024ULL;
/// A4-host inc 4: the CUDA context cost is NOT flat — on Windows/WDDM the
/// driver keeps a host-side backing store for device allocations (VRAM is
/// pageable), so host RSS grows with the device footprint. Measured after
/// prepare(), host RSS minus process baseline and matchup tables, on CUDA
/// 13.2 / Win11 / RTX 5090:
///     device   98 MB →   34 MB   (fixed context dominates)
///     device 1.82 GB →  477 MB   (26%)
///     device 16.3 GB → 2.73 GB   (17%)
/// 16% of device total, ON TOP of the flat context above, stays on the
/// conservative side of all three. It only started to matter when inc 4 let
/// multi-GB enumerated trees reach the GPU: at 16 GB of VRAM the missing
/// term was 2.7 GB of real host RAM the gate never charged.
constexpr double kGpuDeviceHostMirrorFraction = 0.16;
inline uint64_t bytes_for_gpu_host_overhead(uint64_t device_total_bytes) {
    return kGpuHostContextBytes
         + static_cast<uint64_t>(
               static_cast<double>(device_total_bytes)
               * kGpuDeviceHostMirrorFraction);
}

/// Per-node index tables the GPU backend keeps HOST-side for the whole solve:
/// `host_node_order[N]` (the level schedule, walked on the host to launch per
/// level) and `host_value_row[N]` (B3 inc 1's node → value-row indirection),
/// 4 B each, plus the `depth[N]` array that is live inside prepare(). Not the
/// device copies — these are separate host vectors, exactly sized.
///
/// 12 B/node is invisible until the trees get large, and then it is not: the
/// 4.65M-node spot retains 37 MB of it, which was exactly the residual left
/// after the flat-tree term was added (measured est 4505.4 vs 4544.4 MB).
constexpr uint64_t kGpuHostIndexBytesPerNode = 12;
inline uint64_t bytes_for_gpu_host_index_tables(uint64_t nodes) {
    return nodes * kGpuHostIndexBytesPerNode;
}

/// How many copies of the finalized strategy are alive at peak. Measured as
/// (finalize RSS delta ÷ one copy) across turn/mono/rainbow spots, 2026-07-27:
///     CPU  2.05  2.09  2.11  2.13   — the backend's nested copy and
///                                     Solver::strategy_ both stay live
///     GPU  1.24  1.28  1.36         — the flat device download is released
///                                     as the nested repack forms
/// The flat 2× was calibrated on CPU and over-charged GPU solves by a full
/// copy: 1.3 GB on the enumerated rainbow, which is the difference between
/// "enumerate" and "collapse" at a 8 GB budget. 1.5 stays above every GPU
/// measurement without charging a copy that never exists.
constexpr double kFinalStrategyCopiesCpu = 2.0;
constexpr double kFinalStrategyCopiesGpu = 1.5;

/// ...and that calibration measured the wrong moment. It divided the FINALIZE
/// RSS delta by one copy, which is blind to the mid-loop exploitability probe:
/// the probe calls finalize() too, so on a GPU solve it downloads and repacks
/// a full strategy long before the real finalize does, and the allocator's
/// high-water mark keeps it. Measured 2026-08-03 on narrow ranges, peak RSS
/// with the probe on vs off (`node scripts/peakhost.mjs`):
///     rainbow flop 627k nodes   1424.0 vs 1057.8 MB   (+366 = 0.98 copies)
///     paired  flop 395k nodes    985.3 vs  803.7 MB   (+182 = ~1 copy)
///     mono    flop 185k nodes    302.1 vs  274.8 MB   (+27  = ~1 copy)
/// The CPU is unaffected (422.5 vs 422.2, 3233.6 vs 3233.7): its finalize
/// already keeps two copies live, so the probe reuses that same heap.
///
/// This only became visible after B1b inc 2 shrank every nc-scaled term — the
/// old over-charge was hiding a full strategy copy, and on the 627k rainbow
/// that showed up as a 17.9% UNDER-estimate on the gate the solve has to pass.
constexpr double kFinalStrategyProbeCopiesGpu = 1.0;
inline uint64_t bytes_for_live_final_strategy(uint64_t one_copy_bytes,
                                              bool gpu_backend,
                                              bool exploit_probe_runs = false) {
    double copies = gpu_backend ? kFinalStrategyCopiesGpu
                                : kFinalStrategyCopiesCpu;
    if (gpu_backend && exploit_probe_runs) copies += kFinalStrategyProbeCopiesGpu;
    return static_cast<uint64_t>(static_cast<double>(one_copy_bytes) * copies);
}
/// Result JSON floor when the navigation strategy tree is NOT emitted
/// (--no-strategy-tree): root strategies + diagnostics + resources, no
/// per-node cache. Measured ~0.3–2.7 MB depending on nc.
constexpr uint64_t kNoTreeJsonFloorBytes = 4ULL * 1024ULL * 1024ULL;

/// One value region per traverser. The two backward passes write different
/// values to the same rows, so a shared region forced them to run one after
/// the other — 26 of the 42 kernel launches per iteration were 13 depth levels
/// × 2 traversers. Giving each its own region lets one grid carry both
/// (`blockIdx.z`), which is the launch-count lever measured in ROADMAP §2(4):
/// the two hot kernels average 2.6 µs and 2.1 µs to run against 3.26 µs to
/// issue. The cost is one extra value buffer — 348 MB on the 627k-node
/// enumerated rainbow, 1.47 GB on the 4.65M-node target.
constexpr uint64_t kGpuValueRegions = 2;

/// GPU backend keeps regrets, strategy_sum, current_strategy (3 compact
/// strat-shaped buffers of Σ-player-actions × nc; B1a inc 2 dropped
/// action_values), plus reach_scratch_oop and reach_scratch_ip (2 full-tree
/// N×nc buffers) and node_values (B3 inc 1: value_rows × nc, smaller than
/// N × nc — see gpu_value_layout.h). Summed in `bytes_for_gpu_state_compact`.

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
    /// Device TOTAL a GPU solve would claim (state + matchup upload + tree
    /// metadata + levels + reserve). This — not gpu_state_bytes — is what
    /// the VRAM ceiling compares (review round 2 P1-3: a dense turn solve
    /// measurably exceeded the user's --gpu-memory-mb while the state-only
    /// check said "ok"). 0 on CPU-final footprints.
    uint64_t device_total_bytes        = 0;
    /// Peak-host lifetime terms (P1-1): 2× the materialized final strategy
    /// (backend nested copy + Solver deep copy, both alive after finalize)
    /// plus process/CUDA-context overhead. Without these the host gate
    /// passed solves whose finalize then blew the budget (measured: mono
    /// CPU est 2.00 GB vs 2.43 GB actual peak).
    uint64_t final_strategy_bytes      = 0;   ///< ONE copy; see gpu_backend.
    uint64_t process_overhead_bytes    = 0;
    /// Host-resident FlatGameTree (bytes_for_flat_tree). Separate from the
    /// process baseline because it scales with the tree, and at 4.65M nodes
    /// it is hundreds of MB.
    uint64_t flat_tree_bytes           = 0;
    uint64_t strategy_tree_ev_bytes    = 0;
    uint64_t json_response_bytes       = 0;

    /// Peak host-side estimate across the whole solve lifetime (prepare →
    /// iterate → finalize → serialize). Excludes GPU state — that lives in
    /// VRAM.
    /// True when this footprint describes a GPU-executing solve. Only affects
    /// how many finalized-strategy copies are charged (see
    /// kFinalStrategyCopies*) — the device terms above already say which
    /// backend this is, but they can legitimately be 0, so it is explicit.
    bool gpu_backend = false;

    /// True when the solve will run the mid-loop exploitability probe. The
    /// probe calls finalize(), so on GPU it costs an extra live strategy copy
    /// at peak (see kFinalStrategyProbeCopiesGpu).
    bool exploit_probe_runs = false;

    uint64_t total_host_bytes() const {
        return matchup_tables_bytes
             + cpu_state_bytes
             + memory_budget::bytes_for_live_final_strategy(
                   final_strategy_bytes, gpu_backend, exploit_probe_runs)
             + process_overhead_bytes
             + flat_tree_bytes
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
/// `bytes_per_cell` already includes element sizes.
inline uint64_t bytes_for_matchup_tables(
    uint64_t runout_tables,
    uint64_t nc,
    uint64_t bytes_per_cell = memory_budget::kMatchupBytesPerCell)
{
    return runout_tables
         * nc * nc
         * bytes_per_cell;
}

/// Lane-rounded row width the levelized CPU backend actually allocates for
/// every nc-wide row (mirrors LevelizedCpuBackend::round_up_to_lane).
inline uint64_t cpu_lane_stride(uint64_t nc) {
    return (nc + memory_budget::kCpuActionLaneFloats - 1)
         & ~(memory_budget::kCpuActionLaneFloats - 1);
}

/// Bytes for the CPU CFR state. Counts regrets + strategy_sum +
/// current_strategy (three arrays), each `player_action_slots ×
/// lane-rounded nc` floats, where player_action_slots = Σ num_children over
/// PLAYER nodes. This mirrors LevelizedCpuBackend::prepare()'s compact
/// allocation exactly (node_state_offset_ prefix sum over actual action
/// counts). The reference recursive backend allocates the same slot count as
/// per-node vectors (un-padded), so this is a ≤1% overestimate there.
///
/// Renamed (not just re-derived) from bytes_for_cpu_state(player_nodes,
/// MAX_ACTIONS, nc): the old formula priced every player node at MAX_ACTIONS
/// (6) instead of its real action count — a ~77% overestimate on the
/// enumerated monotone fixture (2.61 GB reported vs 1.47 GB allocated) that
/// caused false host-gate collapses. Any caller still passing MAX_ACTIONS
/// now fails to compile instead of silently mis-estimating.
inline uint64_t bytes_for_cpu_state_compact(uint64_t player_action_slots,
                                            uint64_t nc) {
    return player_action_slots * cpu_lane_stride(nc)
         * memory_budget::kCpuStateArraysPerNode
         * sizeof(float);
}

/// Host bytes for ONE materialized final-strategy copy: every player node's
/// na × nc row as an UN-padded nested vector (LevelizedCpuBackend::finalize
/// writes na × nc, not na × stride; GPU finalize repacks to the same shape).
/// Plus per-row vector headers, which stop being noise at 100k+ nodes.
/// The peak-host model charges TWO of these (backend copy + Solver copy —
/// both alive after `strategy_ = backend_->strategy()`).
inline uint64_t bytes_for_final_strategy(uint64_t player_action_slots,
                                         uint64_t nc,
                                         uint64_t player_nodes) {
    return player_action_slots * nc * sizeof(float)
         + player_nodes * 32ULL;  // vector header + allocator slack per row
}

/// v1.7.0: extra heap held by the levelized CPU backend on top of the
/// reference state. LevelizedCpuBackend allocates three flat [N × nc] float
/// buffers — `reach_oop_`, `reach_ip_`, and `value_` — that the recursive
/// reference backend doesn't need (it carries reach on the call stack).
///
///   total = 3 × total_nodes × lane-rounded nc × sizeof(float)
///
/// On a 200k-node tree with nc=1326 that lands at ~3.2 GB, easily enough to
/// blow a tight host-RAM budget. Add this to `cpu_state_bytes` whenever the
/// solve will run on the levelized backend so the host gate can reject
/// before LevelizedCpuBackend::prepare() heap-allocates.
///
/// `total_nodes` is `tree.total_nodes` (every node, not just player nodes —
/// reach is propagated through chance nodes too). `nc` is iso.num_canonical;
/// the backend pads each row to row_stride_ = round_up_to_lane(nc), so the
/// estimator uses the same lane-rounded width.
inline uint64_t bytes_for_levelized_cpu_extra(uint64_t total_nodes, uint64_t nc) {
    return total_nodes * cpu_lane_stride(nc) * 3ULL * sizeof(float);
}

/// Bytes for the GPU CFR state. Lives in VRAM. Mirrors the COMPACT layout
/// GpuBackend's alloc_solver_state() actually allocates (B1a increments 1+2):
///   - regrets, strategy_sum, current_strategy: each
///     [player_action_slots × nc] floats, where player_action_slots =
///     Σ num_children over PLAYER nodes (the per-node slot table; chance and
///     terminal nodes own no strat-shaped rows). Increment 2 dropped the
///     4th buffer (action_values) — aggregate/update_regrets gather child
///     node_values directly. Keep this factor in sync with
///     alloc_solver_state's own pre-flight.
///   - reach_scratch_oop, reach_scratch_ip, node_values: each N×nc floats
///   - node_offset slot table: N uint32
/// Total = (3·slots + 3·N) · nc · sizeof(float) + N · sizeof(uint32)
///
/// The pre-B1a dense formula ((4·MAX_ACTIONS + 3) · N · nc) over-counted
/// enumerated monotone trees ~2.9× (7,453 MB estimated vs 2,587 MB measured
/// on Ah9h4h) which made the ① collapse gate reject boards that actually
/// fit. Renamed (not just re-derived) so any caller still passing
/// MAX_ACTIONS fails to compile instead of silently mis-estimating.
/// B1a inc 3: `materialize_strategy` = does this solve keep a device-side
/// current_strategy? Only node-locked solves do (the lock override is a write
/// into it); everything else derives the strategy from regrets or
/// strategy_sum inside each consumer, so the strat-shaped term is 2 buffers,
/// not 3. Passed explicitly rather than defaulted so a caller that has not
/// thought about locks fails to compile.
/// B3 inc 1: `value_rows` is the device value buffer's row count — terminals
/// plus a two-level non-terminal window, NOT total_nodes. Callers get it from
/// `gpu_value_rows(tree)` (gpu_value_layout.h), the same function
/// GpuBackend::prepare() builds its table from, so the estimate keeps matching
/// the allocation to the byte. Passing 0 falls back to total_nodes, which is
/// the pre-B3 behavior and a safe over-estimate.
inline uint64_t bytes_for_gpu_state_compact(uint64_t total_nodes,
                                            uint64_t player_action_slots,
                                            uint64_t nc,
                                            bool materialize_strategy,
                                            uint64_t value_rows) {
    const uint64_t strat_buffers = materialize_strategy ? 3ULL : 2ULL;
    const uint64_t values = value_rows ? value_rows : total_nodes;
    return (strat_buffers * player_action_slots + 2ULL * total_nodes
              + memory_budget::kGpuValueRegions * values)
             * nc * sizeof(float)
         + 2ULL * total_nodes * sizeof(uint32_t);  // node_offset + value_row
}

/// Device-side TOTAL for a GPU solve (review round 2, P1-3): the VRAM
/// ceiling must compare what prepare() actually claims, not just solver
/// state. Components mirror GpuBackend::prepare():
///   - solver state (compact, bytes_for_gpu_state_compact)
///   - dense matchup upload (EV+valid floats) — ONLY when the
///     TerminalRepresentationPlan says the dense tables actually upload
///     (`device_dense_upload`). PR-4 made this plan-aware: the
///     dense-always assumption was a ~6× over-estimate on singleton-iso
///     boards (AsKd7c2h: 593 MB est vs 98 MB measured) because the
///     rank-blocker skips the upload there.
///   - tree metadata uploads (upload_tree: ~24 B/node + 5 B/edge)
///   - level schedule (~4 B/node)
///   - allocator-granularity + small-buffer reserve (reach/locks/rank
///     tables): 8 MiB + ~3% of the two big terms.
/// Validated against measured cudaMemGetInfo prepare deltas:
///   mono AsKsQs  (dense)      est ~1875 MiB vs 1822 MiB measured (+2.9%)
///   dense turn   (Ah9h4h2c)   est 183.6 MiB vs 182 MiB measured  (+0.9%)
///   singleton turn (AsKd7c2h, no upload) — re-validate after PR-4 wiring.
inline uint64_t bytes_for_gpu_device_total(uint64_t total_nodes,
                                           uint64_t total_edges,
                                           uint64_t player_action_slots,
                                           uint64_t matchup_tables,
                                           uint64_t nc,
                                           bool device_dense_upload,
                                           bool materialize_strategy,
                                           uint64_t value_rows) {
    const uint64_t state   = bytes_for_gpu_state_compact(
        total_nodes, player_action_slots, nc, materialize_strategy, value_rows);
    const uint64_t matchup = device_dense_upload
        ? bytes_for_matchup_tables(matchup_tables, nc, 2ULL * sizeof(float))
        : 0ULL;
    const uint64_t tree    = 24ULL * total_nodes + 5ULL * total_edges;
    const uint64_t levels  = 4ULL * total_nodes + 4096ULL;
    const uint64_t reserve = 8ULL * 1024ULL * 1024ULL + (state + matchup) / 32ULL;
    return state + matchup + tree + levels + reserve;
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
// v1.2.2: solve-time estimate (so UI can show ETA pre-iteration)
// ============================================================================
//
// Cost model: dominant per-iter work is the cfr regret update — for every
// player node, we touch every action × every canonical combo × every opponent
// canonical combo (the matchup matrix multiply). So:
//   ops_per_iter ≈ player_nodes × MAX_ACTIONS × nc × nc
//
// Throughput is a hardcoded backend table calibrated against the
// `--benchmark standard` scenario on a few reference machines. Goal is NOT
// 10% accuracy — goal is to distinguish "30 seconds" from "30 minutes" so
// the user knows whether to commit. Real measurements often within 2× of the
// estimate, sometimes 3-4× off on edge cases (very deep stacks with lots of
// bet sizes have higher constant overhead).
//
// CC-aware GPU rate: Pascal (6.x) is ~10× slower than Ada (8.9) at the kind
// of fp32 + atomicAdd workload our kernels do. Don't lump them together.

inline uint64_t ops_per_solve_iteration(uint64_t player_nodes,
                                         uint64_t max_actions,
                                         uint64_t canonical_combos)
{
    return player_nodes * max_actions * canonical_combos * canonical_combos;
}

/// Returns rough ops/second throughput for the named backend.
///   - backend_label_lc is the lowercased backend name as it appears in the
///     SolverResult.backend field, e.g. "cpu-dcfr-avx2", "cpu-levelized-avx2",
///     "cpu-dcfr-scalar", or "cuda (...)" with the device name attached.
///   - cuda_compute_capability is the device CC×10 (so 89 = Ada, 61 = Pascal,
///     90 = Hopper). 0 means "unknown — use a conservative middle value".
///   - cpu_threads_effective is the resolved OMP team size for CPU backends
///     (output of `resolve_cpu_threads()`). Ignored on GPU. Pass 0 to fall
///     back to `hardware_concurrency()` — historically callers did this and
///     we want the helper to keep working when wired into older code.
///
/// v1.3.0: rates **recalibrated** against actual benchmarks. The pre-1.3.0
/// table was 50× pessimistic on the GPU side (calibrated against a
/// hand-wave estimate, not real measurements). RTX 5090 measurement:
/// `--benchmark standard` reports 568 Gops/s sustained — the previous
/// CC-12 entry of 10 Gops/s would have estimated 11 minutes for a spot
/// that actually ran in ~12 seconds. CPU rates also bumped (unmeasured
/// but parallel logic — atomicAdd is faster than I assumed).
///
/// v1.7.1: CPU model split by backend variant (reference vs levelized) and
/// threads. The pre-v1.7.1 model returned 1.5e8 × min(8, cores) regardless
/// of which CPU backend ran, which made `--estimate-only` say "150 seconds"
/// for a spot that levelized 8T finishes in 0.7s — 200× pessimistic, big
/// enough that users were ignoring the banner. Calibrated against
/// `--benchmark standard` (AsKd7c, 216 player nodes, 1176 nc, 1.79e9 ops/iter)
/// across reference {1T, 2T} and levelized {1T, 2T, 4T, 8T}; the per-core
/// avx2 base of 4.0e10 ops/s comes in ~12% pessimistic on this calibration
/// machine, which is the safer side for an ETA banner.
inline double estimated_throughput_ops_per_sec(const std::string& backend_label_lc,
                                                int cuda_compute_capability,
                                                uint32_t cpu_threads_effective = 0u)
{
    if (backend_label_lc.rfind("cuda", 0) == 0) {  // starts with "cuda"
        switch (cuda_compute_capability / 10) {
            case 6:  return 5.0e9;   // Pascal (GTX 10): bumped from 0.5 → 5 Gops
            case 7:  return 2.0e10;  // Volta/Turing: bumped from 1.5 → 20 Gops
            case 8:  return 1.0e11;  // Ampere/Ada: bumped from 5 → 100 Gops
            case 9:  return 3.0e11;  // Hopper: bumped from 8 → 300 Gops
            case 12: return 5.0e11;  // Blackwell (RTX 5090): MEASURED 568 Gops/s
            default: return 5.0e10;  // unknown CC: middle modern-GPU estimate
        }
    }

    // ---- CPU model (v1.7.1) ----
    //
    // Per-core base rate at 1 thread:
    //   AVX2 + FMA path: ~4.0e10 ops/s on a typical modern x86 core.
    //     (measured ~4.5e10 on the calibration machine; we shave 12% to
    //     stay slightly pessimistic for safer ETAs on slower CPUs.)
    //   Scalar fallback: ~1.6e10 ops/s. AVX2 wins are dominated by
    //     vec_pos_normalize / vec_regret_update which SIMD cleanly.
    //
    // Multi-thread scaling depends on the CFR backend variant:
    //   reference (CpuBackend): only intra-iter parallelism is the
    //     OOP||IP `parallel sections` block — caps at 2 OMP threads, and
    //     the second thread only buys ~10% over single-thread (both
    //     traversers share the matchup table and saturate L2).
    //   levelized (LevelizedCpuBackend): per-level `parallel for` over
    //     all nodes, scales linearly up to the physical core count, then
    //     ~40% per logical thread past that (HT diminishing).
    //
    // Calibration constants chosen to match `--benchmark standard` within
    // ~10% across both backends and 1/2/4/8 threads. Real GUI solves on
    // turn/river spots have larger trees, so the per-core rate is a
    // moderate over-estimate there — but still 5-10× more accurate than
    // the pre-v1.7.1 single-rate model.
    const bool is_avx2     = backend_label_lc.find("avx2") != std::string::npos;
    const bool is_levelized = backend_label_lc.find("levelized") != std::string::npos;
    const double per_core   = is_avx2 ? 4.0e10 : 1.6e10;

    uint32_t threads = cpu_threads_effective;
    if (threads == 0u) {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }

    if (is_levelized) {
        // Linear up to 4 cores, then 40% per additional logical thread.
        // Tuned against an 8-logical / 4-physical Intel laptop CPU; rough
        // but better than ignoring threads entirely.
        const double t = static_cast<double>(threads);
        const double eff = (t <= 4.0) ? t : 4.0 + (t - 4.0) * 0.4;
        return per_core * eff;
    }

    // Reference backend: cap at 2-thread parallel sections, +10% from 2nd thread.
    return per_core * (threads >= 2u ? 1.1 : 1.0);
}

inline double estimate_solve_seconds(uint64_t player_nodes,
                                     uint64_t max_actions,
                                     uint64_t canonical_combos,
                                     int max_iterations,
                                     const std::string& backend_label_lc,
                                     int cuda_compute_capability,
                                     uint32_t cpu_threads_effective = 0u)
{
    if (max_iterations <= 0 || player_nodes == 0 || canonical_combos == 0) return 0.0;
    const uint64_t ops = ops_per_solve_iteration(player_nodes, max_actions, canonical_combos);
    const double rate = estimated_throughput_ops_per_sec(
        backend_label_lc, cuda_compute_capability, cpu_threads_effective);
    if (rate <= 0.0) return 0.0;
    // Add a fixed 0.5s for backend prepare + final postsolve so very small
    // spots don't show "0.01s" — too good to be true.
    return (static_cast<double>(ops) * static_cast<double>(max_iterations)) / rate + 0.5;
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
    // Device check uses the TOTAL device claim when the footprint carries
    // one (review round 2 P1-3); state-only is the legacy fallback for
    // callers that haven't priced the full upload.
    const uint64_t device_needed = est.device_total_bytes > 0
        ? est.device_total_bytes : est.gpu_state_bytes;
    if (budget.gpu_bytes > 0 && device_needed > budget.gpu_bytes) {
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
