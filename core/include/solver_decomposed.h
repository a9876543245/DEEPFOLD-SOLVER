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
#include <functional>
#include <map>
#include <memory>
#include <string>
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
};

struct DecomposedResult {
    bool  ok = false;
    /// Per-canonical (flop iso) OOP counterfactual root value — same quantity
    /// and scaling as Solver::root_values(0), so directly comparable to a
    /// monolithic solve of the same spot.
    std::vector<float> root_value_oop;
    float exploitability_pct = 0.0f;   ///< True full-game exploitability.
    int   outer_iterations_run = 0;
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
        s->resolve();
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
        s->resolve_keep_board();   // same board resident → skip the matchup re-upload
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
    // Keep 15% headroom and reserve one `per` for the lone transient streamed
    // solver that is briefly alive while solving a non-pinned leaf.
    uint64_t usable = static_cast<uint64_t>(static_cast<double>(avail) * 0.85);
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
            slot = std::make_unique<Solver>(sub, BackendType::GPU);
            slot->set_forced_iso(&iso_);
            slot->solve();                 // first visit: full upload (board → device).
        } else {
            slot->set_ranges(oop_w, ip_w);
            slot->resolve_keep_board();     // later sweeps: reuse resident board.
        }
        read_subgame_values(slot.get(), li, want_br, want_nav);
    } else {
        SolverConfig sub = make_sub_cfg(li, oop_w, ip_w);
        auto owned = std::make_unique<Solver>(sub, BackendType::GPU);
        owned->set_forced_iso(&iso_);
        owned->solve();
        read_subgame_values(owned.get(), li, want_br, want_nav);
    }

    int resident = 0;
    for (const auto& s : gpu_pinned_) if (s) ++resident;
    if (li >= gpu_cap_) ++resident;        // + the transient streamed solver
    if (resident > gpu_peak_resident_) gpu_peak_resident_ = resident;
}

inline void TrunkDecomposition::solve_all_subgames(bool want_br, bool force_cpu,
                                                  bool want_nav) {
    const int L = static_cast<int>(leaves_.size());
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
            }
        } else {
            for (int li = 0; li < L; ++li) solve_one_subgame_gpu(li, want_br, want_nav);
        }
    } else {
        // CPU: subgames are independent → solve in parallel. Each is forced
        // single-threaded (sub.cpu_threads=1) so this doesn't oversubscribe;
        // the engine's own inner parallel regions run serially (nested OMP off).
        // leaf_nav_ slots are disjoint per li, so nav extraction is lock-free.
#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int li = 0; li < L; ++li) solve_one_subgame(li, want_br, want_nav);
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
        out.push_back({label, sum_w_val[label] / sw});
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

    for (int t = 0; t < opts_.outer_iterations; ++t) {
        regret_match();
        apply_discount(t);
        forward_reach(0, root_reach_oop_, root_reach_ip_, /*use_avg=*/false);
        solve_all_subgames(/*want_br=*/false);   // re-solve subgames (model B).
        std::vector<float> out0, out1;
        cfr(0, 0, t, root_reach_oop_, root_reach_ip_, out0);
        cfr(0, 1, t, root_reach_oop_, root_reach_ip_, out1);
        r.outer_iterations_run = t + 1;
    }

    finalize_strategy();

    // Final consistent subgame solve at the converged averaged strategy,
    // keeping root + BR values for the EV / exploitability traversals. When
    // cpu_final_pass is set (GPU mode), this last pass runs on the CPU
    // reference backend so the delivered strategies + reported exploitability
    // are CPU-quality (the outer iters above stay on the GPU).
    forward_reach(0, root_reach_oop_, root_reach_ip_, /*use_avg=*/true);
    solve_all_subgames(/*want_br=*/true, /*force_cpu=*/opts_.cpu_final_pass,
                       /*want_nav=*/opts_.build_nav);

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

} // namespace deepsolver
