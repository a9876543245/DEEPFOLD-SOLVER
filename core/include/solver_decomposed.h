/**
 * @file solver_decomposed.h
 * @brief Streaming / subgame (chance-node) decomposition orchestrator.
 *
 * Splits a flop spot at the TURN chance node:
 *   - TRUNK   = flop betting whose turn chance children are leaves
 *               (built via GameTreeBuilder::set_truncate_at_chance). Tiny.
 *   - SUBGAME = the turn betting + river game rooted at one turn card. Built
 *               and solved independently by the EXISTING engine (Solver),
 *               forced into the PARENT flop's canonical space so trunk and
 *               subgames share one iso and need NO reach bridge — mirroring
 *               how the monolithic tree keeps a single (flop) iso across all
 *               per-runout boards.
 *
 * The OUTER loop is CFR over the trunk (Stage 1 validated model B: re-solve
 * each subgame seeded with the current trunk reach, NO CFR-D gadget — it
 * converges to monolithic equilibrium quality).
 *
 * Stage 2 (speed): a two-phase outer iteration —
 *   1. forward-reach pass collects the entering reach at every leaf,
 *   2. all leaf subgames are solved IN PARALLEL (independent; each forced
 *      single-threaded so leaf-level parallelism doesn't oversubscribe),
 *   3. two backward CFR passes read the cached per-leaf values.
 * Results are identical to the serial lazy-solve (each leaf is solved
 * independently and deterministically), just N× faster on a multi-core box.
 *
 * Self-contained: the only hot-core touch is the opt-in builder flag plus the
 * opt-in Solver forced-iso / root-value accessors.
 */

#pragma once

#include "types.h"
#include "isomorphism.h"
#include "game_tree_builder.h"
#include "solver.h"           // Solver, compute_matchup_for_board, get_evaluator
#include "solver_backend.h"   // compute_dcfr_factors, BackendType

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace deepsolver {

struct DecompositionOptions {
    int  outer_iterations = 200;   ///< Trunk CFR iterations.
    int  inner_iterations = 200;   ///< Per-subgame CFR iterations (each solve).
    bool verbose          = false;
    /// VALIDATION-HARNESS speedup (opt-in, default off). When true, each leaf
    /// keeps a persistent Solver: built once, then Solver::resolve()'d each
    /// outer iter (reusing the cached tree + matchup tables, re-running only
    /// CFR). Big speedup BUT holds every subgame's tree+matchups resident —
    /// fine for small validation boards (AsKsQs ≈ a few GB), NOT the bounded-
    /// memory production path. Production rainbow flops use GPU streaming
    /// (Stage 3), which keeps only a few subgames resident at a time. Default
    /// false = bounded streaming (rebuild each outer iter).
    bool cache_subgames   = false;

    /// Backend for the per-leaf subgame solves (Stage 3).
    ///   CPU (default) — the omp-parallel, optionally host-cached path above.
    ///   GPU           — stream each turn subgame through a BOUNDED resident
    ///                   pool on the GpuBackend. Subgames are solved one at a
    ///                   time (a single turn solve already saturates the GPU),
    ///                   so this path is sequential, NOT omp-parallel.
    BackendType subgame_backend = BackendType::CPU;

    /// GPU resident-pool capacity (only read when subgame_backend == GPU):
    ///   0  → keep EVERY subgame resident (full cross-outer-iter reuse via
    ///        resolve(); fastest, but VRAM grows with the turn-card count —
    ///        only for boards whose subgames collectively fit).
    ///   >0 → bounded streaming: at most this many GPU solvers are alive at
    ///        once; the LRU one is freed before a new one is built, so peak
    ///        VRAM is capped independent of the turn-card count. This is the
    ///        production rainbow-flop path (the monolithic solve is 76 GB; a
    ///        few resident turn subgames are a few GB).
    int gpu_resident_cap = 0;

    /// Hybrid quality knob (only meaningful when subgame_backend == GPU): run
    /// the OUTER iterations on the GPU (fast value generation — only subgame EV
    /// feeds the trunk, and GPU≈CPU on EV), but solve the FINAL want_br pass on
    /// the CPU reference backend so the DELIVERED strategies + reported
    /// exploitability are CPU-quality. Closes the ~0.13pp GPU-vs-CPU backend
    /// equilibrium-quality offset (see memo) at the cost of one extra CPU pass
    /// over the leaves. Default off.
    bool cpu_final_pass = false;

    /// Stage 5 — build the stitched UI navigation strategy tree. When true the
    /// FINAL pass also extracts each turn subgame's Solver::build_strategy_tree
    /// (river auto-skipped via initial_chance_levels_seen=1) and the driver
    /// stitches them under the trunk's flop betting into
    /// DecomposedResult::strategy_tree — keyed EXACTLY like a monolithic
    /// enumerated-board tree (turn = first chance enumerated: lex-min at the
    /// no-suffix key, other canonical turn cards at "...#<card>"; river
    /// auto-skipped to lex-min). The sidecar JSON + frontend nav then consume
    /// it unchanged. Default off (pure-perf decomposition skips it).
    bool     build_nav = false;
    int      nav_max_player_depth = 8;   ///< matches main.cpp's build_strategy_tree depth.
    uint32_t nav_max_nodes = 0;          ///< 0 = unlimited; else global cap on emitted entries.

    /// Roadmap ③ — WARM-START the per-leaf subgames across outer sweeps
    /// (opt-in, default off; env DEEPSOLVER_DECOMP_WARMSTART=1).
    ///
    /// Default (off) re-solves every subgame from ZERO regrets each sweep, so
    /// a subgame is never deeper than `inner_iterations` however many sweeps
    /// run — and the final want_br pass, which fixes the DELIVERED strategy and
    /// the reported exploitability, is likewise a fresh `inner_iterations`
    /// solve. That is the measured cause of decomposition's high residual
    /// exploitability on real-size spots, not the outer coupling.
    ///
    /// On, a leaf whose Solver instance persists (GPU-pinned, LRU-resident, or
    /// CPU-cached) continues its CFR run: regrets + strategy_sum are kept and
    /// the DCFR iteration index continues (sweep t runs iterations
    /// [t·inner, (t+1)·inner)), so the subgame accumulates outer×inner depth.
    /// Leaves that do NOT persist (streamed past the VRAM pin budget) are
    /// rebuilt and stay cold — hence `warm_start_leaves` in the result, which
    /// reports how many actually warm-started rather than assuming all did.
    ///
    /// Approximate by construction: regrets carried across a sweep were
    /// accumulated under the previous sweep's reach. Off by default until the
    /// convergence sweep says what it buys.
    bool warm_start_subgames = false;

    /// Roadmap ③ — trunk CFR iterations per subgame refresh (default 1 = the
    /// historical behaviour, exactly).
    ///
    /// `outer_iterations` conflates two things that have wildly different
    /// costs: how many times the trunk runs a CFR iteration, and how many
    /// times all N leaf subgames are re-solved. They were 1:1, so outer=30
    /// gives the trunk **30 CFR iterations** — while the monolithic solve of
    /// the same flop uses 3000 to reach 0.5%. And each of those 30 costs a
    /// full 805-subgame sweep (~12 min), so you cannot buy more.
    ///
    /// They don't need to be coupled: at a chance node the trunk CFR reads the
    /// CACHED per-leaf value vector (inj_*) and never recurses into a subgame,
    /// so a trunk iteration is O(trunk_nodes × nc) — microseconds against the
    /// minutes a subgame sweep costs. Setting this to K runs K trunk CFR
    /// iterations against each refreshed leaf value function, with a
    /// continuous DCFR index (sweep t covers [t·K, (t+1)·K)).
    ///
    /// The approximation is the one this decomposition already makes: leaf
    /// values are a value function captured at the sweep's entering reach, and
    /// go stale as the trunk strategy moves within the sweep. K trades that
    /// staleness for trunk convergence.
    int trunk_iterations_per_sweep = 1;
};

struct DecomposedResult {
    bool  ok = false;
    /// Per-canonical (flop iso) OOP counterfactual root value — same quantity
    /// and scaling as Solver::root_values(0), so directly comparable to a
    /// monolithic solve of the same spot.
    std::vector<float> root_value_oop;
    float exploitability_pct = 0.0f;   ///< True full-game exploitability.
    int   outer_iterations_run = 0;
    /// Roadmap ③: total trunk CFR iterations = outer_iterations_run ×
    /// trunk_iterations_per_sweep. Reported separately because it is the
    /// number comparable to a monolithic solve's `iterations_run` — the outer
    /// count only says how often the leaf value functions were refreshed.
    int   trunk_iterations_run = 0;
    int   subgame_solves = 0;          ///< Total Solver::solve() calls.
    uint32_t trunk_nodes = 0;
    uint32_t leaf_count  = 0;          ///< Distinct (chance,turn-card) leaves.
    uint16_t num_canonical = 0;
    /// Max GPU subgame solvers held resident simultaneously (0 on the CPU
    /// path). The bounded-VRAM guarantee: this never exceeds gpu_resident_cap
    /// (or, on the adaptive path, the VRAM-derived K + 1 transient).
    int gpu_peak_resident = 0;
    /// Adaptive path only: the pin count K chosen from measured free VRAM
    /// (how many turn subgames stayed GPU-resident across outer sweeps). 0 on
    /// the fixed-cap / CPU paths.
    int gpu_cap_used = 0;
    /// Roadmap ③: how many leaves actually warm-started on the last sweep
    /// (persistent solver + a warm-start-capable backend). 0 when
    /// opts.warm_start_subgames is off. Less than leaf_count means the rest
    /// were streamed and stayed cold — the number to report rather than
    /// claiming a warm-start that only partly happened.
    int warm_start_leaves = 0;

    /// Stitched UI nav cache (only when opts.build_nav; empty otherwise). Same
    /// type/keys as Solver::build_strategy_tree so the existing sidecar JSON
    /// serializer + frontend navigation consume it with NO changes.
    std::map<std::string, Solver::StrategyTreeEntry> strategy_tree;
    bool     strategy_tree_truncated = false;
    uint32_t strategy_tree_nodes = 0;
};

// ============================================================================
// TrunkDecomposition — holds trunk state + drives the nested CFR.
// ============================================================================

class TrunkDecomposition {
public:
    TrunkDecomposition(const SolverConfig& cfg, const DecompositionOptions& opts)
        : base_cfg_(cfg), opts_(opts) {}

    DecomposedResult run();

private:
    SolverConfig          base_cfg_;
    DecompositionOptions  opts_;

    IsomorphismMapping    iso_;          ///< Flop iso, shared by trunk+subgames.
    FlatGameTree          trunk_;
    uint16_t              nc_ = 0;

    // Flop-board matchup (all trunk terminals are on the flop board).
    std::vector<float>    mev_, mvalid_;
    std::vector<float>    cw_;           ///< canonical weights as float.

    // Root entering reach (flop canonical), from base_cfg_ ranges.
    std::vector<float>    root_reach_oop_, root_reach_ip_;

    // Trunk CFR state, indexed by trunk node. Only player nodes are sized.
    std::vector<std::vector<float>> regrets_, strat_sum_, cur_strat_, avg_strat_;

    // Leaf bookkeeping. A "leaf" is a turn-card child of a (truncated) chance
    // node — num_children == 0 and dealt_card != 0xFF. Indexed 0..leaf_count-1.
    std::vector<uint32_t> leaves_;       ///< leaf index → trunk node id.
    std::vector<int32_t>  leaf_idx_;      ///< trunk node id → leaf index (-1 if not).

    // Per-leaf entering reach (filled by the forward-reach pass) and the
    // subgame values / BR values (filled by the parallel solve). Indexed by
    // leaf index so the parallel solve writes disjoint slots (lock-free).
    std::vector<std::vector<float>> leaf_reach_oop_, leaf_reach_ip_;
    std::vector<std::vector<float>> inj_oop_, inj_ip_;
    std::vector<std::vector<float>> br_oop_, br_ip_;

    // Persistent per-leaf Solvers (only when opts_.cache_subgames). Built once,
    // then resolve()'d each outer iter — reuses cached tree + matchup tables.
    std::vector<std::unique_ptr<Solver>> leaf_solvers_;

    // ---- GPU subgame streaming pool (Stage 3; subgame_backend == GPU) ----
    // A bounded resident pool keyed by leaf index. Each slot owns one solved
    // GPU Solver; the LRU slot is freed BEFORE a new solver is built, so peak
    // resident solvers (hence peak VRAM) never exceeds gpu_cap_, independent of
    // the turn-card count. Accessed sequentially only (the GPU path never runs
    // under omp — a single turn subgame already saturates the device).
    std::vector<int32_t>                 gpu_slot_leaf_;   ///< leaf in slot (-1 = empty)
    std::vector<std::unique_ptr<Solver>> gpu_slot_solver_;
    std::vector<uint64_t>                gpu_slot_lru_;    ///< last-use tick per slot
    uint64_t gpu_tick_ = 0;
    int      gpu_cap_  = 0;             ///< effective pool capacity (set in build_trunk)
    int      gpu_peak_resident_ = 0;

    // ---- Stage 3.5: adaptive pin-first-K pool (subgame_backend==GPU,
    // gpu_resident_cap < 0). Pin leaves 0..K-1 resident for the whole run (built
    // once; resolve_keep_board() each outer sweep → skips the matchup re-upload,
    // the dominant per-leaf cost); stream leaves >= K (build+solve+free each
    // visit). K is MEASURED from real free VRAM after leaf 0 so it adapts to the
    // user's GPU and never OOMs — not every machine has 32 GB.
    bool     gpu_adaptive_     = false;
    bool     gpu_cap_resolved_ = false;
    uint64_t gpu_free_start_   = 0;
    std::vector<std::unique_ptr<Solver>> gpu_pinned_;

    // ---- Stage 5 nav stitching (only when opts_.build_nav) ----
    // Each turn subgame's extracted Solver::build_strategy_tree, filled in the
    // FINAL pass right after the subgame solves (before any GPU eviction), keyed
    // by leaf index. Merged under the trunk's flop betting in build_nav_tree().
    std::vector<std::map<std::string, Solver::StrategyTreeEntry>> leaf_nav_;
    // Per-trunk-node captured EVs for flop-node combo_evs, acting player's
    // perspective (OOP map ← perspective-0 ev pass, IP map ← perspective-1).
    // node_opp_* hold the opponent reach mass at the node for PIO normalization
    // — same convention as Solver::build_strategy_tree's evs_at.
    std::map<uint32_t, std::vector<float>> tnode_vals_oop_, tnode_vals_ip_;
    std::map<uint32_t, float>              tnode_opp_oop_,  tnode_opp_ip_;

    int subgame_solves_ = 0;

    // ---- setup ----
    void build_trunk();
    void init_reach();
    void alloc_state();

    // ---- trunk CFR machinery (reference math) ----
    void regret_match();
    void apply_discount(int iteration);
    void terminal_value(uint32_t node, int who,
                        const std::vector<float>& roop,
                        const std::vector<float>& rip,
                        std::vector<float>& out) const;

    // Forward-reach pass: thread `use_avg ? avg : current` strategy down the
    // trunk and record the entering reach at every leaf.
    void forward_reach(uint32_t node, std::vector<float> roop,
                       std::vector<float> rip, bool use_avg);

    // Solve ALL leaf subgames from leaf_reach_*; fills inj_* (and br_* when
    // want_br). CPU: omp-parallel over leaves (each forced single-threaded).
    // GPU: sequential streaming through the bounded resident pool.
    void solve_all_subgames(bool want_br, bool force_cpu = false, bool want_nav = false);
    void solve_one_subgame(int li, bool want_br, bool want_nav = false);     ///< CPU path.
    void solve_one_subgame_gpu(int li, bool want_br, bool want_nav = false); ///< GPU LRU pool path.
    void solve_one_subgame_gpu_pinned(int li, bool want_br, bool want_nav = false); ///< adaptive pin-first-K.
    int  compute_adaptive_K(uint64_t free_start) const;   ///< K from measured free VRAM.

    /// Re-solve a leaf whose Solver PERSISTED since the last sweep. Warm-start
    /// (keep regrets + continue the DCFR schedule) when enabled and the backend
    /// supports it; otherwise the cold same-board re-solve that has always run
    /// here. `keep_board` marks a GPU-resident board (skip the matchup
    /// re-upload); the CPU-cached path passes false.
    void resolve_persistent(Solver* s, bool keep_board) {
        if (opts_.warm_start_subgames && s->backend_supports_warm_start()) {
            s->resolve_keep_state(sweep_ * opts_.inner_iterations);
            ++warm_leaves_;
        } else if (keep_board) {
            s->resolve_keep_board();
        } else {
            s->resolve();
        }
    }

    /// Outer sweep index; the warm-start iteration offset is
    /// sweep_ × inner_iterations. The final want_br pass uses
    /// sweep_ = outer_iterations so its schedule continues too.
    int sweep_ = 0;
    /// Leaves that actually warm-started on the current sweep. Atomic: the CPU
    /// path solves leaves in an omp-parallel loop.
    std::atomic<int> warm_leaves_{0};

    // Shared between the CPU and GPU paths.
    void leaf_ranges(int li, std::array<float, NUM_COMBOS>& oop_w,
                     std::array<float, NUM_COMBOS>& ip_w) const;
    SolverConfig make_sub_cfg(int li,
                              const std::array<float, NUM_COMBOS>& oop_w,
                              const std::array<float, NUM_COMBOS>& ip_w) const;
    void read_subgame_values(Solver* s, int li, bool want_br, bool want_nav = false);

    // CFR pass: updates regrets/strat_sum at traverser nodes; reads cached
    // subgame values (inj_*) at chance leaves.
    void cfr(uint32_t node, int traverser, int iteration,
             std::vector<float> roop, std::vector<float> rip,
             std::vector<float>& out);

    // EV pass (averaged strategy) using cached inj_* at leaves. When cap_vals /
    // cap_opp are non-null, records the acting==perspective node's value vector
    // and the opponent reach mass there (for flop-node combo_evs, Stage 5).
    void ev(uint32_t node, int perspective,
            std::vector<float> roop, std::vector<float> rip,
            std::vector<float>& out,
            std::map<uint32_t, std::vector<float>>* cap_vals = nullptr,
            std::map<uint32_t, float>* cap_opp = nullptr) const;

    // Best-response pass for `player` using cached br_* at leaves.
    void br(uint32_t node, int player,
            std::vector<float> roop, std::vector<float> rip,
            std::vector<float>& out) const;

    void finalize_strategy();

    // ---- Stage 5: stitch the UI navigation strategy tree ----
    // Capture per-flop-node EVs (two averaged-strategy ev passes), then walk the
    // trunk emitting StrategyTreeEntry for each flop player node and merge each
    // turn subgame's extracted nav (from leaf_nav_) under it. Fills r.* fields.
    void build_nav_tree(DecomposedResult& r);
    // Per-flop-node StrategyTreeEntry field producers (mirror Solver::extract_*
    // over the trunk arrays + avg_strat_ + root reach). `roop`/`rip` are the
    // reach AT this node (threaded by the walk) for the opponent-range field.
    std::vector<std::pair<std::string, float>> trunk_global_strategy(uint32_t n) const;
    std::vector<std::pair<std::string,
        std::vector<std::pair<std::string, float>>>> trunk_combo_strategies(uint32_t n) const;
    std::vector<std::string> trunk_action_labels(uint32_t n) const;
    Solver::OpponentRangeResult trunk_opponent_range(uint32_t n,
        const std::vector<float>& roop, const std::vector<float>& rip) const;
    std::vector<std::pair<std::string, float>> trunk_evs(uint32_t n) const;

    bool is_player(uint32_t n) const {
        auto t = static_cast<NodeType>(trunk_.node_types[n]);
        return t == NodeType::PLAYER_OOP || t == NodeType::PLAYER_IP;
    }
};

// ============================================================================
// Setup
// ============================================================================

inline void TrunkDecomposition::build_trunk() {
    iso_ = compute_isomorphism(base_cfg_.board.data(), base_cfg_.board_size);
    nc_  = iso_.num_canonical;

    GameTreeBuilder builder(base_cfg_);
    builder.set_truncate_at_chance(true);
    trunk_ = builder.build();

    cw_.assign(nc_, 0.0f);
    for (uint16_t c = 0; c < nc_; ++c)
        cw_[c] = static_cast<float>(iso_.canonical_weights[c]);

    // Flop-board matchup (board_size == 3) in flop-canonical space.
    auto& eval = get_evaluator();
    if (!eval.is_initialized()) eval.initialize();
    auto eval_for = [&eval](const Card* bd, uint8_t bs) {
        return [&eval, bd, bs](const Combo& combo) -> uint16_t {
            return eval.evaluate5(combo.cards[0], combo.cards[1], bd[0], bd[1], bd[2]);
        };
    };
    std::vector<uint8_t> cat; std::vector<float> coeff; std::vector<int8_t> cnt;
    std::vector<uint16_t> ranks;
    Solver dummy(base_cfg_, BackendType::CPU);  // compute_matchup_for_board ignores it
    compute_matchup_for_board(dummy, iso_, base_cfg_.board.data(),
                              base_cfg_.board_size,
                              eval_for(base_cfg_.board.data(), base_cfg_.board_size),
                              mev_, mvalid_, cat, coeff, cnt, false, ranks);

    // Identify leaves (turn-card children of truncated chance nodes).
    leaf_idx_.assign(trunk_.total_nodes, -1);
    for (uint32_t n = 0; n < trunk_.total_nodes; ++n) {
        if (trunk_.num_children[n] == 0 &&
            n < trunk_.dealt_card.size() &&
            trunk_.dealt_card[n] != 0xFFu) {
            leaf_idx_[n] = static_cast<int32_t>(leaves_.size());
            leaves_.push_back(n);
        }
    }
    const size_t L = leaves_.size();
    leaf_reach_oop_.assign(L, {}); leaf_reach_ip_.assign(L, {});
    inj_oop_.assign(L, {}); inj_ip_.assign(L, {});
    br_oop_.assign(L, {}); br_ip_.assign(L, {});
    leaf_solvers_.clear();
    if (opts_.cache_subgames) leaf_solvers_.resize(L);  // null until first solve
    leaf_nav_.clear();
    if (opts_.build_nav) leaf_nav_.assign(L, {});       // filled in the final pass

    // Size the GPU resident pool. cap < 0 ⇒ ADAPTIVE pin-first-K (K measured
    // from free VRAM after leaf 0; production path). cap 0 ⇒ all leaves resident
    // (one LRU slot each). cap > 0 ⇒ bounded LRU streaming with that many slots.
    if (opts_.subgame_backend == BackendType::GPU) {
        gpu_adaptive_      = (opts_.gpu_resident_cap < 0);
        gpu_peak_resident_ = 0;
        if (gpu_adaptive_) {
            gpu_pinned_.clear();
            gpu_pinned_.resize(L);
            gpu_cap_          = 1;      // provisional: leaf 0 always pinned.
            gpu_cap_resolved_ = false;
            gpu_free_start_   = 0;
        } else {
            gpu_cap_ = (opts_.gpu_resident_cap > 0)
                           ? std::min<int>(opts_.gpu_resident_cap, static_cast<int>(L))
                           : static_cast<int>(L);
            if (gpu_cap_ < 1) gpu_cap_ = 1;
            gpu_slot_leaf_.assign(gpu_cap_, -1);
            gpu_slot_solver_.clear();
            gpu_slot_solver_.resize(gpu_cap_);
            gpu_slot_lru_.assign(gpu_cap_, 0);
            gpu_tick_ = 0;
        }
    }
}

inline void TrunkDecomposition::init_reach() {
    root_reach_oop_.assign(nc_, 0.0f);
    root_reach_ip_.assign(nc_, 0.0f);
    const auto& combo_table = get_combo_table();
    CardMask dead = board_to_mask(base_cfg_.board.data(), base_cfg_.board_size);
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        if (combo_table[i].conflicts_with(dead)) continue;
        root_reach_ip_[ci]  = std::max(root_reach_ip_[ci],  base_cfg_.ip_range_weights[i]);
        root_reach_oop_[ci] = std::max(root_reach_oop_[ci], base_cfg_.oop_range_weights[i]);
    }
}

inline void TrunkDecomposition::alloc_state() {
    uint32_t N = trunk_.total_nodes;
    regrets_.assign(N, {});
    strat_sum_.assign(N, {});
    cur_strat_.assign(N, {});
    avg_strat_.assign(N, {});
    for (uint32_t n = 0; n < N; ++n) {
        if (!is_player(n)) continue;
        uint8_t na = trunk_.num_children[n];
        if (na == 0) continue;  // a leaf placeholder typed PLAYER but childless
        std::size_t sz = static_cast<std::size_t>(na) * nc_;
        regrets_[n].assign(sz, 0.0f);
        strat_sum_[n].assign(sz, 0.0f);
        cur_strat_[n].assign(sz, 1.0f / na);
        avg_strat_[n].assign(sz, 1.0f / na);
    }
}

// ============================================================================
// Reference terminal evaluation (mirrors Solver::cpu_ev_traverse). `who` is
// the perspective/traverser (0 = OOP, 1 = IP). All trunk terminals are on the
// flop board, so the single mev_/mvalid_ table applies.
// ============================================================================

inline void TrunkDecomposition::terminal_value(
    uint32_t node, int who,
    const std::vector<float>& roop, const std::vector<float>& rip,
    std::vector<float>& out) const
{
    out.assign(nc_, 0.0f);
    auto tt = static_cast<TerminalType>(trunk_.terminal_types[node]);
    float half_pot = trunk_.pots[node] * 0.5f;

    if (tt == TerminalType::SHOWDOWN) {
        if (who == 0) {
            std::vector<float> wip(nc_);
            for (uint16_t cj = 0; cj < nc_; ++cj) wip[cj] = rip[cj] * cw_[cj];
            for (uint16_t c = 0; c < nc_; ++c) {
                const std::size_t base = static_cast<std::size_t>(c) * nc_;
                float v = 0.0f;
                for (uint16_t cj = 0; cj < nc_; ++cj)
                    v += wip[cj] * mev_[base + cj] * mvalid_[base + cj];
                out[c] = v * half_pot;
            }
        } else {
            std::vector<float> woop(nc_);
            for (uint16_t ci = 0; ci < nc_; ++ci) woop[ci] = roop[ci] * cw_[ci];
            for (uint16_t ci = 0; ci < nc_; ++ci) {
                float s = woop[ci];
                if (s == 0.0f) continue;
                s *= half_pot;
                const std::size_t base = static_cast<std::size_t>(ci) * nc_;
                for (uint16_t c = 0; c < nc_; ++c)
                    out[c] -= s * mev_[base + c] * mvalid_[base + c];
            }
        }
        return;
    }

    // Fold terminal: winner gets only the matched portion.
    uint32_t parent = trunk_.parent_indices[node];
    float unmatched = (parent < trunk_.total_nodes) ? trunk_.bet_into[parent] : 0.0f;
    float gain = (trunk_.pots[node] - unmatched) * 0.5f;
    float sign_oop = (tt == TerminalType::FOLD_OOP) ? -1.0f : 1.0f;
    float sign = (who == 0) ? sign_oop : -sign_oop;
    if (who == 0) {
        std::vector<float> wip(nc_);
        for (uint16_t cj = 0; cj < nc_; ++cj) wip[cj] = rip[cj] * cw_[cj];
        for (uint16_t c = 0; c < nc_; ++c) {
            const std::size_t base = static_cast<std::size_t>(c) * nc_;
            float opp = 0.0f;
            for (uint16_t cj = 0; cj < nc_; ++cj) opp += wip[cj] * mvalid_[base + cj];
            out[c] = sign * gain * opp;
        }
    } else {
        std::vector<float> woop(nc_);
        for (uint16_t ci = 0; ci < nc_; ++ci) woop[ci] = roop[ci] * cw_[ci];
        const float sg = sign * gain;
        for (uint16_t ci = 0; ci < nc_; ++ci) {
            float s = woop[ci];
            if (s == 0.0f) continue;
            s *= sg;
            const std::size_t base = static_cast<std::size_t>(ci) * nc_;
            for (uint16_t c = 0; c < nc_; ++c) out[c] += s * mvalid_[base + c];
        }
    }
}

// ============================================================================
// Forward-reach pass + parallel subgame solve.
// ============================================================================

inline void TrunkDecomposition::forward_reach(
    uint32_t node, std::vector<float> roop, std::vector<float> rip, bool use_avg)
{
    auto nt = static_cast<NodeType>(trunk_.node_types[node]);
    if (nt == NodeType::TERMINAL) return;
    uint32_t off = trunk_.children_offset[node];
    if (nt == NodeType::CHANCE) {
        uint8_t nch = trunk_.num_children[node];
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = trunk_.children[off + k];
            int32_t li = leaf_idx_[child];
            if (li >= 0) { leaf_reach_oop_[li] = roop; leaf_reach_ip_[li] = rip; }
        }
        return;
    }
    int acting = trunk_.active_player[node];
    uint8_t na = trunk_.num_children[node];
    const auto& strat = use_avg ? avg_strat_[node] : cur_strat_[node];
    std::vector<float>& ar = (acting == 0) ? roop : rip;
    for (uint8_t a = 0; a < na; ++a) {
        std::vector<float> cr_oop = roop, cr_ip = rip;
        std::vector<float>& car = (acting == 0) ? cr_oop : cr_ip;
        for (uint16_t c = 0; c < nc_; ++c) car[c] = ar[c] * strat[a * nc_ + c];
        forward_reach(trunk_.children[off + a], cr_oop, cr_ip, use_avg);
    }
}

// Entering ranges for a leaf (flop-canonical reach → 1326 originals). With a
// forced flop iso the subgame re-buckets these back to the SAME canonical
// space, so this is an identity bridge (turn symmetry ⊆ flop symmetry).
inline void TrunkDecomposition::leaf_ranges(
    int li, std::array<float, NUM_COMBOS>& oop_w,
    std::array<float, NUM_COMBOS>& ip_w) const
{
    const std::vector<float>& roop = leaf_reach_oop_[li];
    const std::vector<float>& rip  = leaf_reach_ip_[li];
    oop_w.fill(0.0f);
    ip_w.fill(0.0f);
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        oop_w[i] = roop[ci];
        ip_w[i]  = rip[ci];
    }
}

// Subgame config for a leaf: board = flop + this turn card, pot/stack from the
// chance node, ranges = entering reach. A turn solve, fixed iteration count.
inline SolverConfig TrunkDecomposition::make_sub_cfg(
    int li, const std::array<float, NUM_COMBOS>& oop_w,
    const std::array<float, NUM_COMBOS>& ip_w) const
{
    uint32_t leaf = leaves_[static_cast<size_t>(li)];
    SolverConfig sub = base_cfg_;
    sub.board[base_cfg_.board_size] = trunk_.dealt_card[leaf];
    sub.board_size = static_cast<uint8_t>(base_cfg_.board_size + 1);
    sub.pot = trunk_.pots[leaf];
    sub.effective_stack = trunk_.stacks[leaf];
    sub.oop_has_initiative = true;            // post-chance: OOP acts, has init.
    sub.has_custom_ranges = true;
    sub.oop_range_weights = oop_w;
    sub.ip_range_weights  = ip_w;
    sub.max_iterations = opts_.inner_iterations;
    sub.target_exploitability = 0.0f;         // fixed iteration count.
    sub.time_budget_seconds = 0;
    sub.compute_combo_evs = false;
    sub.compute_exploitability = false;
    sub.cpu_threads = 1;                       // CPU: leaf-level parallelism instead.
    sub.parallel_postsolve = false;
    return sub;
}

// Read the solved subgame's root EV (+ BR when wanted) into the leaf's slots.
// When want_nav, also extract the subgame's UI strategy tree NOW (before any
// GPU eviction) for the Stage-5 stitch.
inline void TrunkDecomposition::read_subgame_values(Solver* s, int li, bool want_br,
                                                    bool want_nav) {
    inj_oop_[li] = s->root_values(0);
    inj_ip_[li]  = s->root_values(1);
    if (inj_oop_[li].size() != nc_) inj_oop_[li].assign(nc_, 0.0f);
    if (inj_ip_[li].size()  != nc_) inj_ip_[li].assign(nc_, 0.0f);
    if (want_br) {
        br_oop_[li] = s->root_br_values(0);
        br_ip_[li]  = s->root_br_values(1);
        if (br_oop_[li].size() != nc_) br_oop_[li].assign(nc_, 0.0f);
        if (br_ip_[li].size()  != nc_) br_ip_[li].assign(nc_, 0.0f);
    }
    if (want_nav && li < static_cast<int>(leaf_nav_.size())) {
        // initial_chance_levels_seen=1: in the full game the turn (dealt by the
        // trunk) is the first chance, so the subgame's river is a SUBSEQUENT
        // chance — auto-skip it to lex-min, matching the monolithic
        // enumerated-board contract (only the turn is enumerated).
        bool trunc = false;
        leaf_nav_[li] = s->build_strategy_tree(
            opts_.nav_max_player_depth,
            Solver::StrategyTreeEvMode::VISIBLE,
            /*max_nodes=*/0, &trunc, /*initial_chance_levels_seen=*/1);
    }
}

inline void TrunkDecomposition::solve_one_subgame(int li, bool want_br, bool want_nav) {
    uint32_t leaf = leaves_[static_cast<size_t>(li)];
    std::array<float, NUM_COMBOS> oop_w, ip_w;
    leaf_ranges(li, oop_w, ip_w);

    Solver* s = nullptr;
    std::unique_ptr<Solver> local;  // owns the solver in non-cached mode.
    const bool cached = opts_.cache_subgames && leaf < trunk_.total_nodes;
    if (cached && leaf_solvers_[li]) {
        // Fast path: reuse the cached tree+matchups, just re-solve with new
        // ranges. Skips the dominant tree-build + matchup-precompute cost.
        s = leaf_solvers_[li].get();
        s->set_ranges(oop_w, ip_w);
        resolve_persistent(s, /*keep_board=*/false);
    } else {
        SolverConfig sub = make_sub_cfg(li, oop_w, ip_w);
        auto owned = std::make_unique<Solver>(sub, BackendType::CPU);
        owned->set_forced_iso(&iso_);
        owned->solve();
        s = owned.get();
        if (cached) leaf_solvers_[li] = std::move(owned);
        else        local = std::move(owned);
    }
    read_subgame_values(s, li, want_br, want_nav);
}

// GPU path: solve one leaf through the bounded resident pool. If the leaf's
// solver is still resident, reuse it (set_ranges + resolve — reuses the device-
// resident tree + matchup buffers). Otherwise free the LRU slot (releasing its
// VRAM) and build a fresh GPU solve there. Peak residency is thus capped at
// gpu_cap_ regardless of how many turn cards the flop has.
inline void TrunkDecomposition::solve_one_subgame_gpu(int li, bool want_br, bool want_nav) {
    std::array<float, NUM_COMBOS> oop_w, ip_w;
    leaf_ranges(li, oop_w, ip_w);

    int slot = -1;
    for (int k = 0; k < gpu_cap_; ++k)
        if (gpu_slot_leaf_[k] == li) { slot = k; break; }

    if (slot >= 0) {
        Solver* s = gpu_slot_solver_[slot].get();
        s->set_ranges(oop_w, ip_w);
        // same board resident → skip the matchup re-upload (and, when
        // warm-start is on, the regret reset too).
        resolve_persistent(s, /*keep_board=*/true);
        gpu_slot_lru_[slot] = ++gpu_tick_;
        read_subgame_values(s, li, want_br, want_nav);
        return;
    }

    // Not resident: pick an empty slot, else the LRU one.
    slot = 0;
    for (int k = 0; k < gpu_cap_; ++k) {
        if (gpu_slot_leaf_[k] < 0) { slot = k; break; }
        if (gpu_slot_lru_[k] < gpu_slot_lru_[slot]) slot = k;
    }
    // Free the evicted solver's VRAM BEFORE allocating the new one so peak
    // residency stays exactly gpu_cap_, not gpu_cap_+1.
    if (gpu_slot_leaf_[slot] >= 0) {
        gpu_slot_solver_[slot].reset();
        gpu_slot_leaf_[slot] = -1;
    }
    SolverConfig sub = make_sub_cfg(li, oop_w, ip_w);
    auto owned = std::make_unique<Solver>(sub, BackendType::GPU);
    owned->set_forced_iso(&iso_);
    owned->solve();
    gpu_slot_solver_[slot] = std::move(owned);
    gpu_slot_leaf_[slot]   = li;
    gpu_slot_lru_[slot]    = ++gpu_tick_;
    read_subgame_values(gpu_slot_solver_[slot].get(), li, want_br, want_nav);

    int filled = 0;
    for (int k = 0; k < gpu_cap_; ++k) if (gpu_slot_leaf_[k] >= 0) ++filled;
    if (filled > gpu_peak_resident_) gpu_peak_resident_ = filled;
}

// Adaptive path: how many turn subgames to PIN resident on the GPU. Measured
// from real free VRAM (after leaf 0 is resident) so it adapts to the user's
// card and never OOMs — leaf 0's footprint is one resident subgame's VRAM.
inline int TrunkDecomposition::compute_adaptive_K(uint64_t free_start) const {
    const int L = static_cast<int>(leaves_.size());
    uint64_t free_after = gpu_free_bytes();
    // No CUDA (the "GPU" subgames fell back to CpuBackend → host RAM, not VRAM):
    // cap the pin count so we don't try to hold every leaf's host precompute.
    if (free_start == 0 || free_after == 0) return std::min(L, 8);
    if (free_after >= free_start)           return std::min(L, 4);   // measurement noise
    uint64_t per = free_start - free_after;                          // one resident leaf's VRAM
    if (per < (16ull << 20))                return std::min(L, 4);   // implausibly small → safe
    uint64_t avail = free_after;            // real VRAM remaining after leaf 0.
    // Optional override: simulate a smaller card (test the partial-pin / no-OOM
    // path) or cap GPU use. DEEPSOLVER_DECOMP_VRAM_MB = TOTAL budget in MB; leaf
    // 0 already consumed `per`, so the remaining simulated budget is cap - per.
    if (const char* e = std::getenv("DEEPSOLVER_DECOMP_VRAM_MB")) {
        uint64_t capb = (static_cast<uint64_t>(std::max(0, std::atoi(e)))) << 20;
        uint64_t remain = (capb > per) ? (capb - per) : 0;
        avail = std::min(avail, remain);
    }
    // Headroom: 25% of free VRAM, floored at 3 GB. The desktop app SHARES the
    // GPU with its WebView renderer — pinning into the last ~1.5 GB starves it
    // until the page crash-reloads mid-solve (observed on the 2026-07-18
    // desktop e2e: plateau at 30.7/32.6 GB → WebView reload at leaf ~290).
    // Also reserve one `per` for the lone transient streamed solver that is
    // briefly alive while solving a non-pinned leaf.
    uint64_t headroom = std::max<uint64_t>(
        static_cast<uint64_t>(static_cast<double>(avail) * 0.25), 3ull << 30);
    uint64_t usable = (avail > headroom) ? (avail - headroom) : 0;
    int more = (usable > per) ? static_cast<int>((usable - per) / per) : 0;
    int K = 1 + more;                                               // + leaf 0 (already resident)
    if (K < 1) K = 1;
    if (K > L) K = L;
    return K;
}

// Adaptive path: pin leaves 0..gpu_cap_-1 (persistent solver each — built once,
// resolve_keep_board() on later sweeps skips the matchup upload); stream the
// rest (transient solver, built+solved+freed) so peak VRAM = K pinned + 1.
inline void TrunkDecomposition::solve_one_subgame_gpu_pinned(int li, bool want_br,
                                                            bool want_nav) {
    std::array<float, NUM_COMBOS> oop_w, ip_w;
    leaf_ranges(li, oop_w, ip_w);

    if (li < gpu_cap_) {
        std::unique_ptr<Solver>& slot = gpu_pinned_[static_cast<size_t>(li)];
        if (!slot) {
            SolverConfig sub = make_sub_cfg(li, oop_w, ip_w);
            try {
                auto s = std::make_unique<Solver>(sub, BackendType::GPU);
                s->set_forced_iso(&iso_);
                s->solve();            // first visit: full upload (board → device).
                slot = std::move(s);
            } catch (const std::exception& e) {
                // Pin overshot free VRAM. compute_adaptive_K sizes K from
                // leaf 0's footprint, which under-measures when another
                // process frees/takes VRAM during that sample (seen with the
                // desktop WebView) or when later lines' subtrees are bigger.
                // Stop pinning here — this and all later leaves stream.
                gpu_cap_ = li;
                std::fprintf(stderr,
                             "[decompose] pin stopped at leaf %d (%s); "
                             "streaming the rest\n", li, e.what());
            }
        } else {
            slot->set_ranges(oop_w, ip_w);
            // later sweeps: reuse the resident board (and, with warm-start,
            // continue this leaf's CFR run instead of restarting it).
            resolve_persistent(slot.get(), /*keep_board=*/true);
        }
        if (slot) {
            read_subgame_values(slot.get(), li, want_br, want_nav);
            int resident = 0;
            for (const auto& s : gpu_pinned_) if (s) ++resident;
            if (resident > gpu_peak_resident_) gpu_peak_resident_ = resident;
            return;
        }
        // fall through: pin failed for THIS leaf too — stream it.
    }
    {
        SolverConfig sub = make_sub_cfg(li, oop_w, ip_w);
        try {
            auto owned = std::make_unique<Solver>(sub, BackendType::GPU);
            owned->set_forced_iso(&iso_);
            owned->solve();
            read_subgame_values(owned.get(), li, want_br, want_nav);
        } catch (const std::exception& e) {
            // Transient GPU alloc failed (VRAM pressure spike from another
            // process mid-run). Solve this ONE leaf on CPU instead of letting
            // the whole decomposition degrade to the monolithic fallback.
            std::fprintf(stderr,
                         "[decompose] leaf %d GPU alloc failed (%s); "
                         "CPU fallback for this leaf\n", li, e.what());
            SolverConfig sub2 = make_sub_cfg(li, oop_w, ip_w);
            sub2.cpu_threads = 0;  // serial GPU loop: let this leaf use all cores
            auto owned = std::make_unique<Solver>(sub2, BackendType::CPU);
            owned->set_forced_iso(&iso_);
            owned->solve();
            read_subgame_values(owned.get(), li, want_br, want_nav);
        }
    }

    int resident = 0;
    for (const auto& s : gpu_pinned_) if (s) ++resident;
    ++resident;                            // + the transient streamed solver
    if (resident > gpu_peak_resident_) gpu_peak_resident_ = resident;
}

inline void TrunkDecomposition::solve_all_subgames(bool want_br, bool force_cpu,
                                                  bool want_nav) {
    const int L = static_cast<int>(leaves_.size());
    // Per-leaf progress line for the Tauri shell (engine.rs
    // parse_decompose_line turns these into UI progress events — keep the
    // two formats in sync). fprintf = one line-atomic write per completed
    // leaf, safe from inside the OMP region; the counter (not `li`) keeps
    // the reported count monotone under dynamic scheduling. run() sets
    // `sweep_` before every call, so want_br alone distinguishes the final
    // pass from a sweep here.
    std::atomic<int> progress_done{0};
    auto report_leaf = [&]() {
        int d = progress_done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (want_br) {
            std::fprintf(stderr, "[decompose] final pass leaf %d/%d\n", d, L);
        } else {
            std::fprintf(stderr, "[decompose] sweep %d/%d leaf %d/%d\n",
                         sweep_ + 1, opts_.outer_iterations, d, L);
        }
    };
    if (opts_.subgame_backend == BackendType::GPU && !force_cpu) {
        // GPU: a single turn subgame already saturates the device, so solve
        // them ONE AT A TIME (no host-level parallel-for). Nav is extracted
        // inside read_subgame_values right after each solve, before any slot is
        // evicted — so VRAM stays bounded.
        if (gpu_adaptive_) {
            // Adaptive pin-first-K. Measure free VRAM before leaf 0 (first
            // sweep only), then set K so K pinned + 1 transient fit without OOM.
            if (!gpu_cap_resolved_) gpu_free_start_ = gpu_free_bytes();
            for (int li = 0; li < L; ++li) {
                solve_one_subgame_gpu_pinned(li, want_br, want_nav);
                if (li == 0 && !gpu_cap_resolved_) {
                    gpu_cap_ = compute_adaptive_K(gpu_free_start_);
                    gpu_cap_resolved_ = true;
                }
                report_leaf();
            }
        } else {
            for (int li = 0; li < L; ++li) {
                solve_one_subgame_gpu(li, want_br, want_nav);
                report_leaf();
            }
        }
    } else {
        // CPU: subgames are independent → solve in parallel. Each is forced
        // single-threaded (sub.cpu_threads=1) so this doesn't oversubscribe;
        // the engine's own inner parallel regions run serially (nested OMP off).
        // leaf_nav_ slots are disjoint per li, so nav extraction is lock-free.
#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int li = 0; li < L; ++li) {
            solve_one_subgame(li, want_br, want_nav);
            report_leaf();
        }
    }
    subgame_solves_ += L;
}

// ============================================================================
// CFR / EV / BR passes over the trunk (chance leaves read cached values).
// ============================================================================

inline void TrunkDecomposition::cfr(
    uint32_t node, int traverser, int iteration,
    std::vector<float> roop, std::vector<float> rip,
    std::vector<float>& out)
{
    auto nt = static_cast<NodeType>(trunk_.node_types[node]);

    if (nt == NodeType::TERMINAL) { terminal_value(node, traverser, roop, rip, out); return; }

    if (nt == NodeType::CHANCE) {
        out.assign(nc_, 0.0f);
        uint8_t nch = trunk_.num_children[node];
        uint32_t off = trunk_.children_offset[node];
        uint32_t total_w = 0;
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = trunk_.children[off + k];
            uint32_t w = (child < trunk_.runout_weight.size()) ? trunk_.runout_weight[child] : 1;
            if (w == 0) w = 1;
            const auto& cv = (traverser == 0) ? inj_oop_[leaf_idx_[child]]
                                              : inj_ip_[leaf_idx_[child]];
            for (uint16_t c = 0; c < nc_; ++c) out[c] += static_cast<float>(w) * cv[c];
            total_w += w;
        }
        if (total_w > 0) { float inv = 1.0f / total_w; for (auto& v : out) v *= inv; }
        return;
    }

    int acting = trunk_.active_player[node];
    uint8_t na = trunk_.num_children[node];
    const auto& strat = cur_strat_[node];
    uint32_t off = trunk_.children_offset[node];

    if (acting == traverser) {
        float pos, neg, sw; compute_dcfr_factors(iteration, base_cfg_, pos, neg, sw);
        std::vector<float>& acting_reach = (acting == 0) ? roop : rip;
        if (dcfr_decay_and_add(base_cfg_)) {                 // POSTFLOP: s = s*sw + strat
            for (uint8_t a = 0; a < na; ++a)
                for (uint16_t c = 0; c < nc_; ++c)
                    strat_sum_[node][a * nc_ + c] =
                        strat_sum_[node][a * nc_ + c] * sw + strat[a * nc_ + c];
        } else {                                             // STANDARD: s += sw*reach*strat
            for (uint8_t a = 0; a < na; ++a)
                for (uint16_t c = 0; c < nc_; ++c)
                    strat_sum_[node][a * nc_ + c] +=
                        sw * acting_reach[c] * strat[a * nc_ + c];
        }
    }

    std::vector<std::vector<float>> avals(na);
    std::vector<float>& acting_reach = (acting == 0) ? roop : rip;
    for (uint8_t a = 0; a < na; ++a) {
        std::vector<float> cr_oop = roop, cr_ip = rip;
        std::vector<float>& car = (acting == 0) ? cr_oop : cr_ip;
        for (uint16_t c = 0; c < nc_; ++c) car[c] = acting_reach[c] * strat[a * nc_ + c];
        cfr(trunk_.children[off + a], traverser, iteration, cr_oop, cr_ip, avals[a]);
    }

    out.assign(nc_, 0.0f);
    if (acting == traverser) {
        for (uint8_t a = 0; a < na; ++a)
            for (uint16_t c = 0; c < nc_; ++c) out[c] += strat[a * nc_ + c] * avals[a][c];
        for (uint8_t a = 0; a < na; ++a)
            for (uint16_t c = 0; c < nc_; ++c)
                regrets_[node][a * nc_ + c] += avals[a][c] - out[c];
    } else {
        for (uint8_t a = 0; a < na; ++a)
            for (uint16_t c = 0; c < nc_; ++c) out[c] += avals[a][c];
    }
}

inline void TrunkDecomposition::regret_match() {
    for (uint32_t n = 0; n < trunk_.total_nodes; ++n) {
        if (!is_player(n)) continue;
        uint8_t na = trunk_.num_children[n];
        if (na == 0) continue;
        const float uniform = 1.0f / na;
        for (uint16_t c = 0; c < nc_; ++c) {
            float psum = 0.0f;
            for (uint8_t a = 0; a < na; ++a) psum += std::max(regrets_[n][a * nc_ + c], 0.0f);
            if (psum > 0.0f) {
                float inv = 1.0f / psum;
                for (uint8_t a = 0; a < na; ++a)
                    cur_strat_[n][a * nc_ + c] = std::max(regrets_[n][a * nc_ + c], 0.0f) * inv;
            } else {
                for (uint8_t a = 0; a < na; ++a) cur_strat_[n][a * nc_ + c] = uniform;
            }
        }
    }
}

inline void TrunkDecomposition::apply_discount(int iteration) {
    float pos, neg, sw; compute_dcfr_factors(iteration, base_cfg_, pos, neg, sw);
    for (uint32_t n = 0; n < trunk_.total_nodes; ++n) {
        if (!is_player(n) || trunk_.num_children[n] == 0) continue;
        for (auto& r : regrets_[n]) r *= (r > 0.0f) ? pos : neg;
    }
}

inline void TrunkDecomposition::finalize_strategy() {
    for (uint32_t n = 0; n < trunk_.total_nodes; ++n) {
        if (!is_player(n)) continue;
        uint8_t na = trunk_.num_children[n];
        if (na == 0) continue;
        for (uint16_t c = 0; c < nc_; ++c) {
            float tot = 0.0f;
            for (uint8_t a = 0; a < na; ++a) tot += strat_sum_[n][a * nc_ + c];
            if (tot > 1e-7f) {
                float inv = 1.0f / tot;
                for (uint8_t a = 0; a < na; ++a)
                    avg_strat_[n][a * nc_ + c] = strat_sum_[n][a * nc_ + c] * inv;
            } else {
                for (uint8_t a = 0; a < na; ++a) avg_strat_[n][a * nc_ + c] = 1.0f / na;
            }
        }
    }
}

inline void TrunkDecomposition::ev(
    uint32_t node, int perspective,
    std::vector<float> roop, std::vector<float> rip,
    std::vector<float>& out,
    std::map<uint32_t, std::vector<float>>* cap_vals,
    std::map<uint32_t, float>* cap_opp) const
{
    auto nt = static_cast<NodeType>(trunk_.node_types[node]);
    if (nt == NodeType::TERMINAL) { terminal_value(node, perspective, roop, rip, out); return; }
    if (nt == NodeType::CHANCE) {
        out.assign(nc_, 0.0f);
        uint8_t nch = trunk_.num_children[node];
        uint32_t off = trunk_.children_offset[node];
        uint32_t total_w = 0;
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = trunk_.children[off + k];
            uint32_t w = (child < trunk_.runout_weight.size()) ? trunk_.runout_weight[child] : 1;
            if (w == 0) w = 1;
            const auto& cv = (perspective == 0) ? inj_oop_[leaf_idx_[child]]
                                                : inj_ip_[leaf_idx_[child]];
            for (uint16_t c = 0; c < nc_; ++c) out[c] += static_cast<float>(w) * cv[c];
            total_w += w;
        }
        if (total_w > 0) { float inv = 1.0f / total_w; for (auto& v : out) v *= inv; }
        return;
    }
    int acting = trunk_.active_player[node];
    uint8_t na = trunk_.num_children[node];
    uint32_t off = trunk_.children_offset[node];
    const auto& strat = avg_strat_[node];
    out.assign(nc_, 0.0f);
    if (acting == perspective) {
        for (uint8_t a = 0; a < na; ++a) {
            std::vector<float> cv;
            ev(trunk_.children[off + a], perspective, roop, rip, cv, cap_vals, cap_opp);
            for (uint16_t c = 0; c < nc_; ++c) out[c] += strat[a * nc_ + c] * cv[c];
        }
        // Capture this node's value (acting player's perspective) + opponent
        // reach mass here, for the flop-node combo_evs (PIO normalization).
        if (cap_vals) (*cap_vals)[node] = out;
        if (cap_opp) {
            const std::vector<float>& opp = (acting == 0) ? rip : roop;
            float mass = 0.0f;
            for (uint16_t c = 0; c < nc_; ++c) mass += opp[c] * cw_[c];
            (*cap_opp)[node] = mass;
        }
    } else {
        std::vector<float>& ar = (acting == 0) ? roop : rip;
        for (uint8_t a = 0; a < na; ++a) {
            std::vector<float> cr_oop = roop, cr_ip = rip;
            std::vector<float>& car = (acting == 0) ? cr_oop : cr_ip;
            for (uint16_t c = 0; c < nc_; ++c) car[c] = ar[c] * strat[a * nc_ + c];
            std::vector<float> cv;
            ev(trunk_.children[off + a], perspective, cr_oop, cr_ip, cv, cap_vals, cap_opp);
            for (uint16_t c = 0; c < nc_; ++c) out[c] += cv[c];
        }
    }
}

inline void TrunkDecomposition::br(
    uint32_t node, int player,
    std::vector<float> roop, std::vector<float> rip,
    std::vector<float>& out) const
{
    auto nt = static_cast<NodeType>(trunk_.node_types[node]);
    if (nt == NodeType::TERMINAL) { terminal_value(node, player, roop, rip, out); return; }
    if (nt == NodeType::CHANCE) {
        out.assign(nc_, 0.0f);
        uint8_t nch = trunk_.num_children[node];
        uint32_t off = trunk_.children_offset[node];
        uint32_t total_w = 0;
        for (uint8_t k = 0; k < nch; ++k) {
            uint32_t child = trunk_.children[off + k];
            uint32_t w = (child < trunk_.runout_weight.size()) ? trunk_.runout_weight[child] : 1;
            if (w == 0) w = 1;
            const auto& cv = (player == 0) ? br_oop_[leaf_idx_[child]]
                                           : br_ip_[leaf_idx_[child]];
            for (uint16_t c = 0; c < nc_; ++c) out[c] += static_cast<float>(w) * cv[c];
            total_w += w;
        }
        if (total_w > 0) { float inv = 1.0f / total_w; for (auto& v : out) v *= inv; }
        return;
    }
    int acting = trunk_.active_player[node];
    uint8_t na = trunk_.num_children[node];
    uint32_t off = trunk_.children_offset[node];
    if (acting == player) {
        std::vector<std::vector<float>> cvs(na);
        for (uint8_t a = 0; a < na; ++a)
            br(trunk_.children[off + a], player, roop, rip, cvs[a]);
        out.assign(nc_, -1e30f);
        for (uint8_t a = 0; a < na; ++a)
            for (uint16_t c = 0; c < nc_; ++c) out[c] = std::max(out[c], cvs[a][c]);
    } else {
        const auto& strat = avg_strat_[node];
        std::vector<float>& ar = (acting == 0) ? roop : rip;
        out.assign(nc_, 0.0f);
        for (uint8_t a = 0; a < na; ++a) {
            std::vector<float> cr_oop = roop, cr_ip = rip;
            std::vector<float>& car = (acting == 0) ? cr_oop : cr_ip;
            for (uint16_t c = 0; c < nc_; ++c) car[c] = ar[c] * strat[a * nc_ + c];
            std::vector<float> cv;
            br(trunk_.children[off + a], player, cr_oop, cr_ip, cv);
            for (uint16_t c = 0; c < nc_; ++c) out[c] += cv[c];
        }
    }
}

// ============================================================================
// Stage 5: stitched UI navigation strategy tree.
//
// Mirrors Solver::build_strategy_tree's per-node fields over the trunk arrays
// (flop player nodes) and splices each turn subgame's already-extracted nav
// (leaf_nav_) under the trunk's turn chance — keyed EXACTLY like a monolithic
// enumerated-board tree so the sidecar JSON + frontend nav consume it unchanged.
// ============================================================================

inline std::vector<std::string>
TrunkDecomposition::trunk_action_labels(uint32_t n) const {
    std::vector<std::string> labels;
    if (n >= trunk_.total_nodes) return labels;
    uint32_t off = trunk_.children_offset[n];
    uint8_t nch = trunk_.num_children[n];
    float pot = trunk_.pots[n];
    for (uint8_t i = 0; i < nch; ++i) {
        auto at = static_cast<ActionType>(trunk_.child_action_types[off + i]);
        float amt = trunk_.child_action_amts[off + i];
        labels.push_back(Solver::action_to_label(at, amt, pot));
    }
    return labels;
}

inline std::vector<std::pair<std::string, float>>
TrunkDecomposition::trunk_global_strategy(uint32_t n) const {
    std::vector<std::pair<std::string, float>> out;
    if (n >= trunk_.total_nodes || avg_strat_[n].empty()) return out;
    uint8_t na = trunk_.num_children[n];
    if (na == 0) return out;
    auto labels = trunk_action_labels(n);
    int acting = trunk_.active_player[n];
    const auto& reach = (acting == 1) ? root_reach_ip_ : root_reach_oop_;
    std::vector<float> totals(na, 0.0f);
    float grand = 0.0f;
    for (uint16_t c = 0; c < nc_; ++c) {
        float rw = (c < reach.size()) ? reach[c] : 1.0f;
        float w = cw_[c] * rw;
        for (uint8_t a = 0; a < na; ++a) {
            float val = avg_strat_[n][a * nc_ + c] * w;
            totals[a] += val; grand += val;
        }
    }
    for (uint8_t a = 0; a < na; ++a) {
        float pct = (grand > 0.0f) ? (totals[a] / grand * 100.0f) : 0.0f;
        out.push_back({a < labels.size() ? labels[a] : std::to_string(a), pct});
    }
    return out;
}

inline std::vector<std::pair<std::string,
    std::vector<std::pair<std::string, float>>>>
TrunkDecomposition::trunk_combo_strategies(uint32_t n) const {
    std::vector<std::pair<std::string,
        std::vector<std::pair<std::string, float>>>> out;
    if (n >= trunk_.total_nodes || avg_strat_[n].empty()) return out;
    uint8_t na = trunk_.num_children[n];
    if (na == 0) return out;
    auto labels = trunk_action_labels(n);
    int acting = trunk_.active_player[n];
    const auto& reach = (acting == 1) ? root_reach_ip_ : root_reach_oop_;
    const auto& combo_table = get_combo_table();
    std::map<std::string, std::vector<float>> totals;
    std::map<std::string, float> weights;
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        uint16_t ci = iso_.original_to_canonical[i];
        if (ci == UINT16_MAX) continue;
        float rw = (ci < reach.size()) ? reach[ci] : 0.0f;
        if (rw <= 0.0f) continue;
        std::string label = combo_to_grid_label(combo_table[i]);
        if (label.size() > 3) continue;   // grid labels only (match Solver tree)
        auto& arr = totals[label];
        if (arr.empty()) arr.assign(na, 0.0f);
        for (uint8_t a = 0; a < na; ++a)
            arr[a] += avg_strat_[n][a * nc_ + ci] * rw;
        weights[label] += rw;
    }
    for (auto& [label, arr] : totals) {
        float w = weights[label];
        if (w <= 0.0f) continue;
        std::vector<std::pair<std::string, float>> e;
        e.reserve(na);
        for (uint8_t a = 0; a < na; ++a)
            e.push_back({a < labels.size() ? labels[a] : std::to_string(a), arr[a] / w});
        out.push_back({label, std::move(e)});
    }
    return out;
}

inline Solver::OpponentRangeResult
TrunkDecomposition::trunk_opponent_range(uint32_t n,
    const std::vector<float>& roop, const std::vector<float>& rip) const {
    Solver::OpponentRangeResult out{};
    if (n >= trunk_.total_nodes) return out;
    bool acting_is_ip = (trunk_.active_player[n] == 1);
    out.opponent = acting_is_ip ? "OOP" : "IP";
    const auto& reach = acting_is_ip ? roop : rip;   // opponent reach at node
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
    float maxw = 0.0f;
    for (auto& [l, w] : label_w) if (w > maxw) maxw = w;
    if (maxw <= 0.0f) return out;
    out.labels.reserve(label_w.size());
    for (auto& [l, w] : label_w) out.labels.push_back({l, w / maxw});
    std::sort(out.labels.begin(), out.labels.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

inline std::vector<std::pair<std::string, float>>
TrunkDecomposition::trunk_evs(uint32_t n) const {
    std::vector<std::pair<std::string, float>> out;
    if (n >= trunk_.total_nodes) return out;
    bool acting_is_ip = (trunk_.active_player[n] == 1);
    const auto& vals_map = acting_is_ip ? tnode_vals_ip_ : tnode_vals_oop_;
    auto it = vals_map.find(n);
    if (it == vals_map.end()) return out;
    const std::vector<float>& vals = it->second;
    const auto& opp_map = acting_is_ip ? tnode_opp_ip_ : tnode_opp_oop_;
    auto oit = opp_map.find(n);
    float opp_mass = (oit != opp_map.end()) ? oit->second : 0.0f;
    float norm = (opp_mass > 1e-6f) ? (1.0f / opp_mass) : 0.0f;
    const auto& reach = acting_is_ip ? root_reach_ip_ : root_reach_oop_;
    const auto& combo_table = get_combo_table();
    std::map<std::string, float> sum_w_val, sum_w;
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
    for (auto& [label, sw] : sum_w) {
        if (sw <= 0.0f) continue;
        const float v = sum_w_val[label] / sw;
        // Mirror Solver::evs_at: skip non-finite labels instead of emitting
        // inf/nan into the stitched nav tree.
        if (!std::isfinite(v)) continue;
        out.push_back({label, v});
    }
    return out;
}

inline void TrunkDecomposition::build_nav_tree(DecomposedResult& r) {
    // Capture flop-node EVs (averaged strategy). inj_* already hold the FINAL
    // averaged-strategy subgame values from the want_br final pass, so the
    // trunk EV traversal sees the converged equilibrium.
    {
        std::vector<float> tmp;
        ev(0, 0, root_reach_oop_, root_reach_ip_, tmp, &tnode_vals_oop_, &tnode_opp_oop_);
        ev(0, 1, root_reach_oop_, root_reach_ip_, tmp, &tnode_vals_ip_,  &tnode_opp_ip_);
    }

    static const char RANK_CH[] = "23456789TJQKA";
    static const char SUIT_CH[] = "cdhs";
    auto card_token = [&](uint8_t c) -> std::string {
        std::string s; s += RANK_CH[c / 4]; s += SUIT_CH[c % 4]; return s;
    };
    auto join = [](const std::string& a, const std::string& b) -> std::string {
        if (a.empty()) return b;
        if (b.empty()) return a;
        return a + "," + b;
    };

    auto& out = r.strategy_tree;
    const uint32_t cap = opts_.nav_max_nodes;   // 0 = unlimited
    bool truncated = false;

    // Merge one turn subgame's extracted nav under base_key. Prefix the turn
    // card into each entry's dealt_cards and stamp the turn runout options on
    // every entry (the monolithic walk propagates first-chance options to the
    // whole subtree). Moves entries out of leaf_nav_[li] then frees it.
    auto merge_leaf = [&](int li, const std::string& base_key, uint8_t turn_card,
                          const std::vector<Solver::RunoutOption>& turn_opts) {
        if (li < 0 || li >= static_cast<int>(leaf_nav_.size())) return;
        for (auto& kv : leaf_nav_[li]) {
            if (cap > 0 && out.size() >= cap) { truncated = true; break; }
            std::string key = join(base_key, kv.first);
            if (out.count(key)) continue;
            Solver::StrategyTreeEntry e = std::move(kv.second);
            std::vector<uint8_t> dc;
            dc.reserve(e.dealt_cards.size() + 1);
            dc.push_back(turn_card);
            for (uint8_t c : e.dealt_cards) dc.push_back(c);
            e.dealt_cards    = std::move(dc);
            e.runout_options = turn_opts;
            out[key] = std::move(e);
        }
        leaf_nav_[li].clear();   // free as we go (peak ≈ final stitched tree).
    };

    std::function<void(uint32_t, const std::string&, std::vector<float>,
                       std::vector<float>, int)> walk;
    walk = [&](uint32_t node, const std::string& path, std::vector<float> roop,
               std::vector<float> rip, int pdepth) {
        if (node >= trunk_.total_nodes) return;
        auto nt = static_cast<NodeType>(trunk_.node_types[node]);
        if (nt == NodeType::TERMINAL) return;

        if (nt == NodeType::CHANCE) {
            // Turn = first chance: enumerate canonical turn cards. lex-min (the
            // first real child) at the no-suffix key, others at "<path>#<card>".
            uint8_t nch = trunk_.num_children[node];
            uint32_t off = trunk_.children_offset[node];
            std::vector<Solver::RunoutOption> opts;
            for (uint8_t k = 0; k < nch; ++k) {
                uint32_t child = trunk_.children[off + k];
                uint8_t dcard = (child < trunk_.dealt_card.size())
                    ? trunk_.dealt_card[child] : 0xFFu;
                if (dcard == 0xFFu) continue;
                uint8_t w = (child < trunk_.runout_weight.size())
                    ? static_cast<uint8_t>(trunk_.runout_weight[child]) : 1;
                opts.push_back({dcard, w});
            }
            bool is_lex_min = true;
            for (uint8_t k = 0; k < nch; ++k) {
                uint32_t child = trunk_.children[off + k];
                uint8_t dcard = (child < trunk_.dealt_card.size())
                    ? trunk_.dealt_card[child] : 0xFFu;
                if (dcard == 0xFFu) continue;
                std::string base_key = is_lex_min ? path
                                                  : (path + "#" + card_token(dcard));
                merge_leaf(leaf_idx_[child], base_key, dcard, opts);
                is_lex_min = false;
                if (cap > 0 && out.size() >= cap) { truncated = true; break; }
            }
            return;
        }

        // Player node: emit entry, recurse with reach threaded by avg strategy.
        if (cap > 0 && out.size() >= cap) { truncated = true; return; }
        if (!out.count(path)) {
            Solver::StrategyTreeEntry e;
            e.acting           = (trunk_.active_player[node] == 1) ? "IP" : "OOP";
            e.action_labels    = trunk_action_labels(node);
            e.global_strategy  = trunk_global_strategy(node);
            e.combo_strategies = trunk_combo_strategies(node);
            auto opp = trunk_opponent_range(node, roop, rip);
            e.opponent_side  = opp.opponent;
            e.opponent_range = std::move(opp.labels);
            e.combo_evs      = trunk_evs(node);
            // dealt_cards / runout_options stay empty (no chance precedes flop).
            out[path] = std::move(e);
        }
        if (pdepth >= opts_.nav_max_player_depth) return;

        int acting = trunk_.active_player[node];
        uint8_t na = trunk_.num_children[node];
        uint32_t off = trunk_.children_offset[node];
        auto labels = trunk_action_labels(node);
        for (uint8_t a = 0; a < na; ++a) {
            std::vector<float> cr_oop = roop, cr_ip = rip;
            std::vector<float>& car = (acting == 0) ? cr_oop : cr_ip;
            for (uint16_t c = 0; c < nc_; ++c) car[c] *= avg_strat_[node][a * nc_ + c];
            std::string alabel = (a < labels.size()) ? labels[a] : std::to_string(a);
            std::string new_path = path.empty() ? alabel : (path + "," + alabel);
            walk(trunk_.children[off + a], new_path, cr_oop, cr_ip, pdepth + 1);
            if (cap > 0 && out.size() >= cap) { truncated = true; break; }
        }
    };

    walk(0, "", root_reach_oop_, root_reach_ip_, 0);
    r.strategy_tree_truncated = truncated;
    r.strategy_tree_nodes     = static_cast<uint32_t>(out.size());
}

// ============================================================================
// Driver
// ============================================================================

inline DecomposedResult TrunkDecomposition::run() {
    DecomposedResult r;
    build_trunk();
    init_reach();
    alloc_state();
    r.num_canonical = nc_;
    r.trunk_nodes   = trunk_.total_nodes;
    r.leaf_count    = static_cast<uint32_t>(leaves_.size());
    if (leaves_.empty()) return r;  // nothing to decompose (no turn chance).

    // K trunk CFR iterations per subgame refresh. K=1 reproduces the original
    // sequence exactly (regret_match → apply_discount → forward_reach →
    // solve_all_subgames → cfr, with trunk_iter == t), so the default is
    // byte-identical. K>1 re-runs only the cheap trunk half against the same
    // refreshed leaf value functions — see trunk_iterations_per_sweep.
    const int K = std::max(1, opts_.trunk_iterations_per_sweep);
    int trunk_iter = 0;
    for (int t = 0; t < opts_.outer_iterations; ++t) {
        for (int k = 0; k < K; ++k) {
            regret_match();
            apply_discount(trunk_iter);
            if (k == 0) {
                // Refresh the leaf value functions once per sweep, at the
                // strategy regret_match just produced (the expensive half).
                forward_reach(0, root_reach_oop_, root_reach_ip_, /*use_avg=*/false);
                sweep_ = t;                              // warm-start iteration offset.
                warm_leaves_ = 0;
                // Roadmap ④: per-sweep wall telemetry — the number the
                // pre-flight estimator predicts, so a long decomposed run
                // is diagnosable (and the estimator calibratable) from a
                // partial run's stderr.
                const auto sweep_t0 = std::chrono::steady_clock::now();
                solve_all_subgames(/*want_br=*/false);   // re-solve subgames (model B).
                const double sweep_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - sweep_t0).count();
                std::cerr << "[decompose] sweep " << (t + 1) << "/"
                          << opts_.outer_iterations << " solved "
                          << leaves_.size() << " subgames in "
                          << sweep_s << "s\n";
            }
            std::vector<float> out0, out1;
            cfr(0, 0, trunk_iter, root_reach_oop_, root_reach_ip_, out0);
            cfr(0, 1, trunk_iter, root_reach_oop_, root_reach_ip_, out1);
            ++trunk_iter;
        }
        r.outer_iterations_run = t + 1;
    }
    r.trunk_iterations_run = trunk_iter;

    finalize_strategy();

    // Final consistent subgame solve at the converged averaged strategy,
    // keeping root + BR values for the EV / exploitability traversals. When
    // cpu_final_pass is set (GPU mode), this last pass runs on the CPU
    // reference backend so the delivered strategies + reported exploitability
    // are CPU-quality (the outer iters above stay on the GPU).
    forward_reach(0, root_reach_oop_, root_reach_ip_, /*use_avg=*/true);
    // The final pass fixes the DELIVERED subgame strategies and the reported
    // exploitability, so warm-start matters most here: cold, this is a fresh
    // `inner_iterations` solve however many sweeps preceded it. Continue the
    // schedule past the last sweep. NOTE: cpu_final_pass routes this pass to
    // fresh CPU solvers, which cannot continue the GPU leaves' state — that
    // combination gets a cold final pass by construction.
    sweep_ = opts_.outer_iterations;
    warm_leaves_ = 0;
    {
        const auto final_t0 = std::chrono::steady_clock::now();
        solve_all_subgames(/*want_br=*/true, /*force_cpu=*/opts_.cpu_final_pass,
                           /*want_nav=*/opts_.build_nav);
        std::cerr << "[decompose] final pass solved " << leaves_.size()
                  << " subgames in "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - final_t0).count()
                  << "s\n";
    }
    r.warm_start_leaves = warm_leaves_.load();

    ev(0, 0, root_reach_oop_, root_reach_ip_, r.root_value_oop);

    // True full-game exploitability over trunk+subgames; normalization matches
    // Solver::compute_exploitability exactly so the numbers are comparable.
    std::vector<float> bro, bri;
    br(0, 0, root_reach_oop_, root_reach_ip_, bro);
    br(0, 1, root_reach_oop_, root_reach_ip_, bri);
    float br_oop_total = 0.0f, br_ip_total = 0.0f;
    float total_oop_w = 0.0f, total_ip_w = 0.0f;
    for (uint16_t c = 0; c < nc_; ++c) {
        br_oop_total += root_reach_oop_[c] * cw_[c] * bro[c];
        br_ip_total  += root_reach_ip_[c]  * cw_[c] * bri[c];
        total_oop_w  += root_reach_oop_[c] * cw_[c];
        total_ip_w   += root_reach_ip_[c]  * cw_[c];
    }
    float denom = total_oop_w * total_ip_w;
    float avg_oop = (denom > 1e-6f) ? (br_oop_total / denom) : 0.0f;
    float avg_ip  = (denom > 1e-6f) ? (br_ip_total  / denom) : 0.0f;
    float exploit_chips = (avg_oop + avg_ip) / 2.0f;
    r.exploitability_pct =
        std::max(0.0f, exploit_chips / std::max(base_cfg_.pot, 1.0f) * 100.0f);

    // Stage 5: stitch the UI navigation strategy tree (trunk flop betting +
    // every turn subgame spliced under the turn chance). Off by default.
    if (opts_.build_nav) build_nav_tree(r);

    r.subgame_solves = subgame_solves_;
    r.gpu_peak_resident = gpu_peak_resident_;
    r.gpu_cap_used = gpu_adaptive_ ? gpu_cap_ : 0;
    r.ok = true;
    return r;
}

inline DecomposedResult solve_decomposed(const SolverConfig& cfg,
                                         const DecompositionOptions& opts) {
    TrunkDecomposition d(cfg, opts);
    return d.run();
}

// ============================================================================
// Roadmap ④ (post-v1.9.0): Exact-mode productization — preset schedule guard
// + feasibility pre-flight estimator.
// ============================================================================

// ---- pow-4 schedule safety ------------------------------------------------
//
// The POSTFLOP DCFR schedule (compute_dcfr_factors) updates strategy_sum by
// decay-and-add with weight ((t−p4)/(t−p4+1))^3, p4 = nearest lower power of
// 4 — the accumulated average strategy is effectively WIPED each time the
// iteration index crosses 1, 4, 16, 64, 256, 1024, 4096. A run whose
// DELIVERED pass ends just past a wipe averages only a handful of
// iterations (measured 2026-07-15: inner=120 crossers regressed). Three
// windows matter for a (outer, inner, K) preset:
//   cold leaves  — re-solved fresh each sweep: window [0, inner)
//   warm leaves  — continue the DCFR index across sweeps; the delivered
//                  final pass is [outer·inner, (outer+1)·inner)
//   trunk        — accumulates outer·K continuous iterations from 0
inline constexpr unsigned decomp_pow4_floor(unsigned t) {
    unsigned p = 0;
    for (unsigned x = 1; x != 0 && x <= t; x *= 4) {
        p = x;
        if (x > (~0u) / 4) break;
    }
    return p;
}

/// True when the delivered window [start, end) keeps enough post-wipe
/// accumulation: the last wipe at or before `end` must leave at least
/// max(64, span/3) iterations (capped at the window span itself).
inline constexpr bool decomp_window_pow4_safe(unsigned start, unsigned end) {
    if (end <= start) return false;
    const unsigned span = end - start;
    const unsigned wipe = decomp_pow4_floor(end);
    const unsigned acc_from = (wipe > start) ? wipe : start;
    const unsigned acc = end - acc_from;
    unsigned need = span / 3u;
    if (need < 64u) need = 64u;
    if (need > span) need = span;
    return acc >= need;
}

inline constexpr bool decomp_preset_pow4_safe(int outer, int inner, int k) {
    return outer >= 1 && inner >= 1 && k >= 1
        && decomp_window_pow4_safe(0u, static_cast<unsigned>(inner))
        && decomp_window_pow4_safe(static_cast<unsigned>(outer * inner),
                                   static_cast<unsigned>((outer + 1) * inner))
        && decomp_window_pow4_safe(0u, static_cast<unsigned>(outer * k));
}

// These tuples MUST mirror DECOMPOSE_PRESETS in src/lib/poker.ts (quick /
// standard / deep). If you change one side, change both — this assert is the
// engine-side tripwire for a preset that lands on a strategy wipe.
static_assert(decomp_preset_pow4_safe(2, 150, 300),
              "quick decompose preset lands the delivered pass on a pow-4 wipe");
static_assert(decomp_preset_pow4_safe(2, 450, 300),
              "standard decompose preset lands the delivered pass on a pow-4 wipe");
static_assert(decomp_preset_pow4_safe(2, 900, 300),
              "deep decompose preset lands the delivered pass on a pow-4 wipe");

// ---- feasibility pre-flight ------------------------------------------------
//
// estimate_decomposition() prices a decomposed solve BEFORE committing to it:
// build the trunk only (set_truncate_at_chance — sub-second), count the
// (betting line × turn card) leaves, and price one representative subgame per
// betting line for the biggest lines. Emitted through --estimate-only so the
// UI can show "Exact ≈ N min, expected accuracy ~X" before the user commits.
//
// Accuracy contract matches estimate_solve_seconds(): distinguish "30
// seconds" from "30 minutes", not ±10%. Known omissions, all second-order:
// adaptive-pin re-upload savings (real spots pin few leaves, e.g. 11/805),
// trunk CFR (microseconds/iter), final-pass nav extraction.
struct DecompositionEstimate {
    bool     ok = false;
    uint32_t leaf_count = 0;     ///< distinct (betting line × turn card) subgames
    uint32_t line_count = 0;     ///< distinct betting lines feeding the chance street
    uint32_t trunk_nodes = 0;
    int      sweeps = 0;         ///< outer refreshes + 1 delivery pass
    double   per_sweep_seconds = 0.0;
    double   total_seconds = 0.0;
    float    spr = 0.0f;         ///< effective_stack / pot at the ROOT
    /// Honest quality banding, keyed on root SPR (2026-07-15 study):
    ///   SPR ≤ 3  → "high"       (fixture SPR 2: 0.09–0.30% exploit)
    ///   SPR ≤ 6  → "medium"     (AsKsQs SPR 5: 25–60%)
    ///   else     → "navigation" (m0_b6 SPR 17.7: 187–256% — browse real
    ///               runouts; don't sell equilibrium quality)
    std::string quality_tier;
    float    expected_exploit_lo_pct = 0.0f;
    float    expected_exploit_hi_pct = 0.0f;
    std::string backend_label;   ///< backend the pricing assumed
};

namespace decomp_estimate_detail {
// GPU subgame pricing is a FLAT per-leaf rate: streamed turn subgames are
// launch-bound (same regime the monolithic perf memos measured — kernel
// size barely matters next to launch overhead), so per-leaf cost is
// per-visit overhead + a per-iteration constant, independent of nc or
// subgame tree size. Fitted on FOUR measured runs (RTX 5090, 2026-07-15):
//   m0_b6  Ah9h4h 55/975 33/75 menus, 805 leaves, (5,150) → 73 min total
//   m0_b6  same spot,                             (2,450) → 66 min total
//   AsKd2c 100/500 default menus, 637 leaves, inner=50: sweep 437.5 s
//                                             + final want_br pass 757.8 s
// The rainbow run separates the sweep from the FINAL pass (best-response
// values ≈ +73%); with that multiplier all four points reproduce within
// ±4%. Slower GPUs scale roughly with launch latency, not FLOPs — we
// accept the single calibration (the banner already says ±2×). River-leaf
// subgames (turn-root boards) have fewer levels per iteration, so this
// over-estimates there — the safe side.
constexpr double kGpuSecondsPerLeafOverhead = 0.585;  ///< build + matchup + upload per visit
constexpr double kGpuSecondsPerSubgameIter  = 1.7e-3; ///< launch-bound per-iteration cost
/// The final delivery pass also computes best-response values — measured
/// 1.73× a plain sweep on the rainbow datum (nav OFF). Applied as
/// (outer + this) sweep-equivalents.
constexpr double kFinalPassSweepEquivalents = 1.73;
/// Nav-tree extraction on the final pass (opts.build_nav — always on for
/// app solves): measured on the desktop e2e run (AsKd2c Lite quick,
/// 245 leaves, wall 1406 s vs 798 s nav-off model ⇒ ~2.5 s/leaf at
/// nc=1176). Extraction walks emitted nodes × nc-long EV vectors, so scale
/// by nc/1176. This is the "§B nav-extraction parallelization" backlog
/// cost, priced honestly until it's optimized.
constexpr double kNavExtractSecondsPerLeafAtNc1176 = 2.5;
// CPU subgames ARE compute-bound, so the CPU path prices from the sampled
// sub-tree ops model (player_nodes × A × nc²) at the 1-thread rate, with a
// fitted shape correction (the sampled trees over-state real work: they
// are built under the subgame board's own enumeration while the real solve
// forces the parent iso). No direct CPU wall measurement yet — ETA-grade.
constexpr double kSubgameTreeOpsCorrection  = 0.48;
constexpr double kCpuMatchupSecondsPerCell  = 1.2e-7; ///< per nc² cell per distinct runout table
constexpr double kCpuFixedSecondsPerLeaf    = 0.07;   ///< Solver ctor + tree build + reach init
}  // namespace decomp_estimate_detail

inline DecompositionEstimate estimate_decomposition(const SolverConfig& cfg,
                                                    const DecompositionOptions& opts) {
    DecompositionEstimate e;
    // Needs a chance street below the root (flop → turn leaves, or turn →
    // river leaves). River roots have nothing to decompose.
    if (cfg.board_size < 3 || cfg.board_size >= 5) return e;

    // Subgames are FORCED into the parent board's canonical space (see the
    // file header: trunk and subgames share one iso, no reach bridge), so
    // their per-iteration ops and matchup cells scale with the PARENT nc² —
    // pricing them at the subgame board's own (finer) iso over-counted 4×
    // on the calibration spot.
    IsomorphismMapping iso = compute_isomorphism(cfg.board.data(), cfg.board_size);
    const uint64_t nc = iso.num_canonical;

    GameTreeBuilder builder(cfg);
    builder.set_truncate_at_chance(true);
    FlatGameTree trunk = builder.build();
    e.trunk_nodes = trunk.total_nodes;

    // Leaves grouped by the (pot, stack) they enter the subgame with — one
    // group per betting line; same-line turn cards share pot/stack and hence
    // subgame tree shape. Group count is small (dozens), linear scan is fine.
    struct Group {
        float pot = 0.0f, stack = 0.0f;
        uint32_t rep_leaf = 0;   ///< representative trunk node id
        uint32_t count = 0;
        double per_leaf_seconds = -1.0;  ///< filled for sampled groups
    };
    std::vector<Group> groups;
    for (uint32_t n = 0; n < trunk.total_nodes; ++n) {
        if (trunk.num_children[n] != 0) continue;
        if (n >= trunk.dealt_card.size() || trunk.dealt_card[n] == 0xFFu) continue;
        ++e.leaf_count;
        const float p = trunk.pots[n], s = trunk.stacks[n];
        bool found = false;
        for (auto& g : groups) {
            if (std::fabs(g.pot - p) < 1e-3f && std::fabs(g.stack - s) < 1e-3f) {
                ++g.count;
                found = true;
                break;
            }
        }
        if (!found) groups.push_back(Group{p, s, n, 1, -1.0});
    }
    if (e.leaf_count == 0) return e;
    e.line_count = static_cast<uint32_t>(groups.size());

    // Backend the subgames would run on. The GPU route needs an actual CUDA
    // device — mirror the solve route's silent CPU fallback when absent.
    const int cc = cuda_compute_capability();
    const bool gpu = (opts.subgame_backend == BackendType::GPU) && cc > 0;
    std::string label_lc;
    if (gpu) {
        label_lc = "cuda (subgame estimate)";
    } else {
        label_lc = (cfg.cpu_backend_kind == SolverConfig::CpuBackendKind::LEVELIZED)
                       ? "cpu-levelized" : "cpu-dcfr";
        label_lc += (cpu_simd::active_mode() == cpu_simd::SimdMode::Avx2)
                       ? "-avx2" : "-scalar";
    }
    // Each subgame solve is forced single-threaded (make_sub_cfg); the CPU
    // path parallelizes ACROSS leaves instead, so its per-leaf rate is the
    // 1-thread rate and the sweep total divides by the OMP team below.
    const double rate = estimated_throughput_ops_per_sec(label_lc, cc, 1u);
    e.backend_label = gpu ? "decomposed-gpu" : "decomposed-cpu";

    const int inner = std::max(1, opts.inner_iterations);
    if (gpu) {
        // Launch-bound flat rate — subgame size doesn't move the needle
        // (see constants above), so no sub-tree sampling needed.
        const double per_leaf =
            decomp_estimate_detail::kGpuSecondsPerLeafOverhead
            + inner * decomp_estimate_detail::kGpuSecondsPerSubgameIter;
        for (auto& g : groups) g.per_leaf_seconds = per_leaf;
    } else {
        // CPU: price one representative subgame per betting line for the
        // deepest remaining-SPR lines (they own the biggest subtrees and
        // dominate the sum); the rest inherit the nearest sampled SPR's
        // per-leaf price.
        std::sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) {
            const float sa = a.stack / std::max(a.pot, 1e-3f);
            const float sb = b.stack / std::max(b.pot, 1e-3f);
            return sa > sb;
        });
        constexpr size_t kMaxSampledLines = 6;
        for (size_t gi = 0; gi < groups.size() && gi < kMaxSampledLines; ++gi) {
            Group& g = groups[gi];
            SolverConfig sub = cfg;  // inherit menus / budgets / CPU knobs
            sub.board[cfg.board_size] = trunk.dealt_card[g.rep_leaf];
            sub.board_size = static_cast<uint8_t>(cfg.board_size + 1);
            sub.pot = g.pot;
            sub.effective_stack = g.stack;
            sub.oop_has_initiative = true;
            sub.max_iterations = inner;

            GameTreeBuilder sb(sub);
            sb.set_memory_policy(nc, sub.memory_budget,
                                 matchup_bytes_per_cell(sub, iso));
            FlatGameTree st = sb.build();

            uint64_t player_n = 0;
            // Distinct runout tables the subgame's precompute will fill
            // (dedup by dealt card — precompute keys tables per canonical
            // board) + 1 for the subgame's own board.
            bool seen[64] = {};
            uint64_t tables = 1;
            for (uint32_t n = 0; n < st.total_nodes; ++n) {
                auto t = static_cast<NodeType>(st.node_types[n]);
                if (t == NodeType::PLAYER_OOP || t == NodeType::PLAYER_IP) ++player_n;
                if (n < st.dealt_card.size()) {
                    const uint8_t dc = st.dealt_card[n];
                    if (dc != 0xFFu && dc < 64 && !seen[dc]) {
                        seen[dc] = true;
                        ++tables;
                    }
                }
            }

            const double ops =
                static_cast<double>(ops_per_solve_iteration(player_n, MAX_ACTIONS, nc))
                * decomp_estimate_detail::kSubgameTreeOpsCorrection;
            const double cfr_s = (rate > 0.0) ? ops * inner / rate : 0.0;
            const double cells = static_cast<double>(tables)
                               * static_cast<double>(nc)
                               * static_cast<double>(nc);
            g.per_leaf_seconds = decomp_estimate_detail::kCpuFixedSecondsPerLeaf
                               + cells * decomp_estimate_detail::kCpuMatchupSecondsPerCell
                               + cfr_s;
            if (std::getenv("DEEPSOLVER_DECOMP_EST_DEBUG")) {
                std::cerr << "[decomp-est] line pot=" << g.pot << " stack=" << g.stack
                          << " count=" << g.count << " player_n=" << player_n
                          << " nc=" << nc << " tables=" << tables
                          << " cfr_s=" << cfr_s
                          << " matchup_s=" << cells * decomp_estimate_detail::kCpuMatchupSecondsPerCell
                          << " leaf_s=" << g.per_leaf_seconds << "\n";
            }
        }
        // Unsampled lines inherit the nearest sampled price (by log-SPR).
        for (auto& g : groups) {
            if (g.per_leaf_seconds >= 0.0) continue;
            const double gspr =
                std::log(std::max(1e-3f, g.stack / std::max(g.pot, 1e-3f)));
            double best = 1e30, price = 0.0;
            for (size_t gi = 0; gi < groups.size() && gi < kMaxSampledLines; ++gi) {
                const Group& s = groups[gi];
                const double sspr =
                    std::log(std::max(1e-3f, s.stack / std::max(s.pot, 1e-3f)));
                const double d = std::fabs(gspr - sspr);
                if (d < best) { best = d; price = s.per_leaf_seconds; }
            }
            g.per_leaf_seconds = price;
        }
    }

    double sweep_total = 0.0;
    for (const auto& g : groups) sweep_total += g.count * g.per_leaf_seconds;
    if (!gpu) {
        // CPU path solves leaves omp-parallel (each single-threaded).
        unsigned threads = std::max(1u, std::thread::hardware_concurrency());
#ifdef _OPENMP
        threads = static_cast<unsigned>(std::max(1, omp_get_max_threads()));
#endif
        sweep_total /= static_cast<double>(
            std::min<uint32_t>(threads, e.leaf_count));
    }
    e.per_sweep_seconds = sweep_total;
    e.sweeps = std::max(1, opts.outer_iterations) + 1;
    // Total = outer plain sweeps + one heavier delivery pass (BR values;
    // multiplier measured on the rainbow datum, applied to both backends)
    // + nav-tree extraction when the caller will build the UI cache.
    e.total_seconds =
        (std::max(1, opts.outer_iterations)
         + decomp_estimate_detail::kFinalPassSweepEquivalents) * sweep_total;
    if (opts.build_nav) {
        e.total_seconds += static_cast<double>(e.leaf_count)
            * decomp_estimate_detail::kNavExtractSecondsPerLeafAtNc1176
            * (static_cast<double>(nc) / 1176.0);
    }

    e.spr = cfg.effective_stack / std::max(cfg.pot, 1e-3f);
    if (e.spr <= 3.0f) {
        e.quality_tier = "high";
        e.expected_exploit_lo_pct = 0.1f;
        e.expected_exploit_hi_pct = 1.0f;
    } else if (e.spr <= 6.0f) {
        e.quality_tier = "medium";
        e.expected_exploit_lo_pct = 10.0f;
        e.expected_exploit_hi_pct = 60.0f;
    } else {
        e.quality_tier = "navigation";
        e.expected_exploit_lo_pct = 100.0f;
        e.expected_exploit_hi_pct = 300.0f;
    }
    // The bands come from standard/deep-class runs (inner ≥ 450). A starved
    // quick pass lands above them on non-trivial SPR (measured: AsKd2c SPR 5
    // quick → 82%, (1,50) → 76%) — widen the ceiling so the promise stays
    // honest.
    if (opts.inner_iterations <= 150 && e.spr > 3.0f) {
        e.expected_exploit_hi_pct *= 1.5f;
    }

    e.ok = true;
    return e;
}

} // namespace deepsolver
