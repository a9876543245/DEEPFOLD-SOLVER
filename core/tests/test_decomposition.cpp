/**
 * @file test_decomposition.cpp
 * @brief Stage-1 validation of streaming/subgame (chance-node) decomposition.
 *
 * Solves the SAME spot two ways and compares:
 *   - MONOLITHIC oracle: the existing engine enumerates turn+river in full
 *     (only possible on a board whose runouts iso-compress — here monotone
 *     AsKsQs — so it does NOT hit the collapse gate).
 *   - DECOMPOSED: solve_decomposed() solves a flop trunk coupled to per-turn
 *     subgames (model B re-solving).
 *
 * The decisive metric is full-game EXPLOITABILITY (root EV alone is
 * insufficient — a zero-sum game's value is shared by every equilibrium).
 * A convergence sweep over inner_iterations prints the exploitability gap
 * that decides B / B+gadget / A for later stages.
 *
 * Opt-in / experiment harness: it PASSES as long as the mechanism runs and
 * produces finite, sane numbers; the verdict is read from the printed table.
 */

#include "solver.h"
#include "solver_decomposed.h"
#include "card.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace deepsolver;

namespace {

SolverConfig make_fixture() {
    SolverConfig cfg;
    cfg.pot = 100.0f;
    cfg.effective_stack = 200.0f;          // SPR 2
    auto board = parse_board("AsKsQs");     // monotone → runouts iso-compress
    cfg.board_size = 3;
    for (int i = 0; i < 3; ++i) cfg.board[i] = board[i];

    cfg.bet_sizing.flop_sizes  = {0.5f};
    cfg.bet_sizing.turn_sizes  = {0.5f};
    cfg.bet_sizing.river_sizes = {0.5f};
    // All-in OFF keeps the trunk to fold terminals only (no flop all-in
    // showdown), isolating the coupling under test for Stage 1.
    cfg.bet_sizing.flop_allin  = false;
    cfg.bet_sizing.turn_allin  = false;
    cfg.bet_sizing.river_allin = false;
    cfg.raise_cap = 1;
    cfg.oop_has_initiative = true;

    cfg.ip_range_weights.fill(1.0f);
    cfg.oop_range_weights.fill(1.0f);
    cfg.has_custom_ranges = false;

    // Diagnostic toggle: the GPU-vs-CPU decomposition gap is the known POSTFLOP
    // sparse-backend-vs-reference strategy divergence at fixed iters (see memo
    // solver_state_vs_pio_2026-06). STANDARD is bit-exact across backends, so it
    // should collapse the gap (both converge slower, but together). Env-gated so
    // the default ctest stays on the POSTFLOP default.
    if (std::getenv("DEEPSOLVER_DECOMP_STANDARD"))
        cfg.dcfr_schedule = SolverConfig::DcfrSchedule::STANDARD;
    return cfg;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    auto& eval = get_evaluator();
    if (!eval.is_initialized()) eval.initialize();

    // ---- Stage 3.5 host-RAM sizing (manual; env-gated, returns early) -------
    // Measures the per-leaf host precompute (matchup tables) for a RAINBOW flop
    // so we can decide cache-all vs a host-side LRU. Builds the trunk to count
    // leaves AND distinct turn BOARDS (matchup depends only on board, so leaves
    // sharing a turn card can share one cached matchup), then ONE turn subgame
    // (forced flop iso, nc≈1176) for the real per-leaf bytes.
    // Run: DEEPSOLVER_DECOMP_MEASURE=1 test_decomposition
    if (std::getenv("DEEPSOLVER_DECOMP_MEASURE")) {
        SolverConfig cfg;
        cfg.pot = 100.0f; cfg.effective_stack = 60.0f;
        auto b = parse_board("AsKd2c"); cfg.board_size = 3;
        for (int i = 0; i < 3; ++i) cfg.board[i] = b[i];
        cfg.bet_sizing.flop_sizes = {0.75f};
        cfg.bet_sizing.turn_sizes = {0.75f};
        cfg.bet_sizing.river_sizes = {0.75f};
        cfg.bet_sizing.flop_allin = cfg.bet_sizing.turn_allin = cfg.bet_sizing.river_allin = false;
        cfg.raise_cap = 1; cfg.oop_has_initiative = true;
        cfg.ip_range_weights.fill(1.0f); cfg.oop_range_weights.fill(1.0f);

        IsomorphismMapping iso = compute_isomorphism(cfg.board.data(), cfg.board_size);
        GameTreeBuilder tb(cfg); tb.set_truncate_at_chance(true);
        FlatGameTree trunk = tb.build();
        int leaves = 0; uint8_t first_turn = 0xFF; bool seen[52] = {false}; int distinct_turns = 0;
        for (uint32_t n = 0; n < trunk.total_nodes; ++n) {
            if (trunk.num_children[n] == 0 && n < trunk.dealt_card.size() &&
                trunk.dealt_card[n] != 0xFFu) {
                uint8_t dc = trunk.dealt_card[n];
                if (first_turn == 0xFF) first_turn = dc;
                if (dc < 52 && !seen[dc]) { seen[dc] = true; ++distinct_turns; }
                ++leaves;
            }
        }

        SolverConfig sub = cfg;
        sub.board[3] = first_turn; sub.board_size = 4;
        sub.max_iterations = 120; sub.compute_combo_evs = false; sub.compute_exploitability = false;
        sub.has_custom_ranges = true;
        // GPU path solves leaves sequentially → each subgame's (omp) matchup
        // precompute already uses all cores; don't force single-thread here.
#ifdef DEEPSOLVER_USE_CUDA
        BackendType meas_backend = BackendType::GPU;
#else
        BackendType meas_backend = BackendType::CPU;
#endif
        Solver s(sub, meas_backend);
        s.set_forced_iso(&iso);
        SolverResult sr = s.solve();
        uint64_t per = s.matchup_host_bytes();

        printf("=== Stage 3.5 sizing + timing (rainbow AsKd2c, pot 100 stack 60) ===\n");
        printf("flop nc=%u  trunk leaves=%d  distinct turn boards=%d  "
               "per-leaf matchup host=%.1f MB\n",
               iso.num_canonical, leaves, distinct_turns, per / 1e6);
        printf("cache-by-LEAF  total = %.2f GB  (per-leaf x %d leaves)\n",
               (double)per * leaves / 1e9, leaves);
        printf("cache-by-BOARD total = %.2f GB  (per-board x %d distinct turns) "
               "<- matchup depends only on board\n",
               (double)per * distinct_turns / 1e9, distinct_turns);
        printf("--- per-leaf timing breakdown (one subgame, %s) ---\n", s.backend_name());
        printf("  tree_build      = %6.0f ms\n", sr.timing.tree_build_ms);
        printf("  isomorphism     = %6.0f ms\n", sr.timing.isomorphism_ms);
        printf("  precompute_match= %6.0f ms\n", sr.timing.precompute_matchups_ms);
        printf("  reach_init      = %6.0f ms\n", sr.timing.reach_init_ms);
        printf("  backend_prepare = %6.0f ms  (GPU: matchup upload + alloc)\n", sr.timing.backend_prepare_ms);
        printf("  iterations(%3d) = %6.0f ms\n", sub.max_iterations, sr.timing.iterations_ms);
        printf("  finalize        = %6.0f ms\n", sr.timing.finalize_ms);
        printf("  postsolve       = %6.0f ms\n", sr.timing.postsolve_ms);
        printf("  TOTAL           = %6.0f ms\n", sr.timing.total_ms);
        double reusable = sr.timing.tree_build_ms + sr.timing.isomorphism_ms +
                          sr.timing.precompute_matchups_ms;
        printf("=> board-only/reusable (tree+iso+precompute) = %.0f ms (%.0f%% of total); "
               "this is what a resident leaf SKIPS via resolve(). x %d leaves.\n",
               reusable, sr.timing.total_ms > 0 ? 100.0 * reusable / sr.timing.total_ms : 0.0,
               leaves);

#ifdef DEEPSOLVER_USE_CUDA
        // Stage 3.5 Increment-1 parity + timing: resolve_keep_board() (keeps the
        // device matchup, skips re-upload) must match resolve() (full re-upload)
        // within GPU run-to-run atomic noise. We first measure the noise floor
        // (resolve vs resolve), then compare resolve vs resolve_keep_board.
        auto maxdiff = [](const std::vector<float>& a, const std::vector<float>& b) {
            double m = 0.0; size_t n = std::min(a.size(), b.size());
            for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
            return m;
        };
        double r0 = now_ms(); s.resolve();            double t_resolve = now_ms() - r0;
        std::vector<float> a0 = s.root_values(0), a1 = s.root_values(1);
        s.resolve();
        std::vector<float> a0b = s.root_values(0), a1b = s.root_values(1);
        double noise = std::max(maxdiff(a0, a0b), maxdiff(a1, a1b));
        double k0 = now_ms(); s.resolve_keep_board();  double t_keep = now_ms() - k0;
        std::vector<float> b0 = s.root_values(0), b1 = s.root_values(1);
        double repd = std::max(maxdiff(a0, b0), maxdiff(a1, b1));
        printf("--- Increment-1: resolve()=%.0fms  resolve_keep_board()=%.0fms  "
               "(skips matchup upload)\n", t_resolve, t_keep);
        printf("    parity: noise floor(resolve vs resolve)=%.3e  "
               "resolve vs keep_board=%.3e  %s\n",
               noise, repd,
               (repd <= std::max(noise * 3.0, 1e-2)) ? "(OK: within noise)" : "(MISMATCH!)");
#endif
        return 0;
    }

#ifdef DEEPSOLVER_USE_CUDA
    // ---- Rainbow headline demo (manual; env-gated, returns early) ----------
    // The texture the whole project is about. On a RAINBOW flop, full turn+
    // river enumeration is ~76 GB (rainbow barely iso-compresses), so the
    // monolithic builder COLLAPSES the turn to one equity-blind child
    // (runout_approximated = true). Decomposition instead streams the REAL
    // per-turn-card subgames on the GPU under BOUNDED VRAM. There is no
    // monolithic oracle here (it can't be solved), so the demo proves the
    // headline structurally: real turn enumeration + bounded residency.
    // Run: DEEPSOLVER_RAINBOW_DEMO=1 [DEEPSOLVER_GPU_OUTER/_INNER/_CAP] exe
    if (std::getenv("DEEPSOLVER_RAINBOW_DEMO")) {
        SolverConfig cfg;
        cfg.pot = 100.0f;
        cfg.effective_stack = 150.0f;            // SPR 1.5 → tractable demo tree
        auto rb = parse_board("AsKd2c");         // rainbow → monolithic collapses
        cfg.board_size = 3;
        for (int i = 0; i < 3; ++i) cfg.board[i] = rb[i];
        cfg.bet_sizing.flop_sizes  = {0.75f};
        cfg.bet_sizing.turn_sizes  = {0.75f};
        cfg.bet_sizing.river_sizes = {0.75f};
        cfg.bet_sizing.flop_allin  = false;
        cfg.bet_sizing.turn_allin  = false;
        cfg.bet_sizing.river_allin = false;
        cfg.raise_cap = 1;
        cfg.oop_has_initiative = true;
        cfg.ip_range_weights.fill(1.0f);
        cfg.oop_range_weights.fill(1.0f);

        printf("=== Rainbow headline: AsKd2c, pot 100, stack 150 ===\n");

        // (1) Monolithic build — show the collapse (full enumeration is 76 GB).
        SolverConfig mc = cfg;
        mc.max_iterations = 1;
        mc.compute_exploitability = false;
        mc.compute_combo_evs = false;
        Solver mono(mc, BackendType::CPU);
        SolverResult mr = mono.solve();
        printf("monolithic build : nodes=%u  runout_approximated=%s\n",
               mr.timing.tree_nodes,
               mr.runout_approximated ? "TRUE (turn collapsed, equity-blind)"
                                      : "false (small enough to enumerate)");

        // (2) Decomposed GPU streaming — real subgames, bounded VRAM.
        int g_outer = 5, g_inner = 50, g_cap = 4;
        if (const char* e = std::getenv("DEEPSOLVER_GPU_OUTER")) g_outer = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_INNER")) g_inner = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_CAP"))   g_cap   = std::atoi(e);
        DecompositionOptions o;
        o.outer_iterations = g_outer;
        o.inner_iterations = g_inner;
        o.subgame_backend  = BackendType::GPU;
        o.gpu_resident_cap = g_cap;
        double rt0 = now_ms();
        DecomposedResult dr = solve_decomposed(cfg, o);
        double rms = now_ms() - rt0;
        if (!dr.ok) { printf("FAIL: rainbow decomposed run failed (ok=false).\n"); return 1; }
        printf("decomposed GPU   : real turn leaves=%u (vs 1 collapsed)  nc=%u  "
               "peak_resident=%d/cap=%d  exploit=%.4f%%  %.1fs\n",
               dr.leaf_count, dr.num_canonical, dr.gpu_peak_resident, g_cap,
               dr.exploitability_pct, rms / 1000.0);
        if (dr.gpu_peak_resident > g_cap) {
            printf("FAIL: peak residency %d exceeds cap %d.\n",
                   dr.gpu_peak_resident, g_cap); return 1;
        }
        printf("PASS — rainbow solved with full turn/river enumeration under "
               "BOUNDED VRAM (%d resident subgames) where monolithic is ~76 GB.\n",
               dr.gpu_peak_resident);
        return 0;
    }
#endif

    // ---- Monolithic oracle -------------------------------------------------
    SolverConfig mono_cfg = make_fixture();
    mono_cfg.max_iterations = 2000;
    mono_cfg.target_exploitability = 0.003f;
    mono_cfg.exploitability_check_interval = 50;
    mono_cfg.compute_exploitability = true;
    mono_cfg.compute_combo_evs = false;

    printf("=== Decomposition validation: AsKsQs, pot 100, stack 200 ===\n");
    double t0 = now_ms();
    Solver mono(mono_cfg, BackendType::CPU);
    SolverResult mres = mono.solve();
    double mono_ms = now_ms() - t0;
    std::vector<float> mono_ev = mono.root_values(0);

    printf("monolithic: nodes=%u  runout_approximated=%s  exploit=%.4f%%  "
           "iters=%d  (%.0f ms)\n",
           mres.timing.tree_nodes,
           mres.runout_approximated ? "TRUE(!!)" : "false",
           mres.exploitability_pct, mres.iterations_run, mono_ms);

    if (mres.runout_approximated) {
        printf("FAIL: oracle collapsed — not a valid apples-to-apples fixture.\n");
        return 1;
    }
    if (mono_ev.empty()) { printf("FAIL: monolithic root_values empty.\n"); return 1; }

    // ---- Decomposed convergence sweep --------------------------------------
    // Default = the decisive single point validated 2026-06-26: at outer=30 /
    // inner=150 the decomposed strategy reaches ~0.09% exploitability — at or
    // below the monolithic oracle. Pass extra args to sweep, e.g.
    //   test_decomposition 60 80 250 600
    int outer = (argc > 1) ? std::atoi(argv[1]) : 30;
    std::vector<int> inner_sweep;
    if (argc > 2) { for (int i = 2; i < argc; ++i) inner_sweep.push_back(std::atoi(argv[i])); }
    else inner_sweep = {150};

    printf("\n inner | outer |  exploit%%  | gap vs mono | rootEV MAE | rootEV max | "
           "subsolves |   time\n");
    printf("-------+-------+------------+-------------+------------+------------+"
           "-----------+--------\n");

    float worst_exploit = 0.0f;
    float best_exploit  = 1e9f;
    bool any_ok = false;

    for (int inner : inner_sweep) {
        DecompositionOptions opts;
        opts.outer_iterations = outer;
        opts.inner_iterations = inner;
        opts.cache_subgames = true;   // validation-harness speedup (resident)

        double d0 = now_ms();
        DecomposedResult dr = solve_decomposed(make_fixture(), opts);
        double dms = now_ms() - d0;

        if (!dr.ok) { printf("  %4d |  %4d | DECOMP FAILED (ok=false)\n", inner, outer); continue; }
        any_ok = true;

        // Per-combo OOP root-value error, weighted by OOP reach (in-range only).
        double num = 0.0, den = 0.0, maxd = 0.0;
        const auto& iso_ev = dr.root_value_oop;
        for (size_t c = 0; c < mono_ev.size() && c < iso_ev.size(); ++c) {
            double d = std::fabs((double)iso_ev[c] - (double)mono_ev[c]);
            num += d; den += 1.0;
            if (d > maxd) maxd = d;
        }
        double mae = (den > 0) ? num / den : 0.0;
        float gap = dr.exploitability_pct - mres.exploitability_pct;
        worst_exploit = std::max(worst_exploit, dr.exploitability_pct);
        best_exploit  = std::min(best_exploit, dr.exploitability_pct);

        printf("  %4d |  %4d |  %8.4f  |  %+9.4f  | %10.4f | %10.4f | %9d | %6.1fs\n",
               inner, dr.outer_iterations_run, dr.exploitability_pct, gap,
               mae, maxd, dr.subgame_solves, dms / 1000.0);
        fflush(stdout);
    }

    printf("\n");
    if (!any_ok) { printf("FAIL: no decomposed run succeeded.\n"); return 1; }
    if (!std::isfinite(best_exploit)) { printf("FAIL: non-finite exploitability.\n"); return 1; }

    // Convergence guard. Validated 2026-06-26: model B (re-solving, NO gadget)
    // converges to the monolithic equilibrium quality — the best sweep point
    // matched/beat the oracle. The decomposed best exploitability must stay
    // within a small margin of the monolithic oracle; a regression in the
    // trunk↔subgame coupling would blow this up. Margin is generous so low-arg
    // (under-converged) manual sweeps don't flake — at the default (30/150) the
    // real value is ~0.09% vs an oracle ~0.11%, an enormous margin.
    const float kMargin = 0.5f;  // percentage points
    // Diagnostic modes (e.g. STANDARD schedule) are deliberately under-converged
    // at the fixed iter count, so the convergence assertion is informational then.
    const bool diag_mode = std::getenv("DEEPSOLVER_DECOMP_STANDARD") != nullptr;
    if (!diag_mode && best_exploit > mres.exploitability_pct + kMargin) {
        printf("FAIL: best decomposed exploit %.4f%% exceeds monolithic %.4f%% "
               "+ %.2f%% margin — coupling regressed.\n",
               best_exploit, mres.exploitability_pct, kMargin);
        return 1;
    }

    printf("PASS — model B validated: decomposed best exploit %.4f%% vs monolithic "
           "%.4f%% (re-solving converges to GTO quality, no gadget needed).\n",
           best_exploit, mres.exploitability_pct);

    // ======================================================================
    // Roadmap ③ — subgame WARM-START arm.
    //
    // Default (cold) restarts every subgame from zero regrets each outer sweep,
    // so a subgame is never deeper than inner_iterations however many sweeps
    // run. This fixture's tiny turn subgames are already solved by inner=150,
    // which is exactly why it converges here and why a REAL-size spot at the
    // same 30x150 does not (measured: 27.9%). Warm-start lets a subgame
    // accumulate outer x inner depth instead.
    //
    // What's asserted here is CORRECTNESS, not speed: warm-start carries
    // regrets accumulated under the previous sweep's reach, so the risk is that
    // it converges to something OTHER than the equilibrium. With a monolithic
    // oracle in hand we can check the real thing — warm must land at oracle
    // quality too. Run at a deliberately SHORT inner (where cold is starved) so
    // the arms are actually distinguishable.
    {
        const int kWarmOuter = 30;
        const int kWarmInner = 20;   // starved on purpose: cold can't converge.

        auto run_arm = [&](bool warm) {
            DecompositionOptions o;
            o.outer_iterations = kWarmOuter;
            o.inner_iterations = kWarmInner;
            o.cache_subgames   = true;
            o.warm_start_subgames = warm;
            return solve_decomposed(make_fixture(), o);
        };

        DecomposedResult cold = run_arm(false);
        DecomposedResult warm = run_arm(true);
        if (!cold.ok || !warm.ok) {
            printf("FAIL: warm-start arm did not run (ok=false).\n");
            return 1;
        }
        printf("\n=== Roadmap ③ warm-start (outer=%d, inner=%d — starved inner) ===\n",
               kWarmOuter, kWarmInner);
        printf("  cold: exploit=%.4f%%  warm_leaves=%d\n",
               cold.exploitability_pct, cold.warm_start_leaves);
        printf("  warm: exploit=%.4f%%  warm_leaves=%d/%u\n",
               warm.exploitability_pct, warm.warm_start_leaves, warm.leaf_count);

        if (!std::isfinite(warm.exploitability_pct)) {
            printf("FAIL: warm-start produced non-finite exploitability.\n");
            return 1;
        }
        // Wiring guard: every leaf is cached here, so every leaf must warm-start.
        // A silent fallback to the cold path would make the arms identical and
        // the comparison meaningless.
        if (warm.warm_start_leaves != static_cast<int>(warm.leaf_count)) {
            printf("FAIL: only %d/%u leaves warm-started — warm-start not wired "
                   "through the cached path.\n",
                   warm.warm_start_leaves, warm.leaf_count);
            return 1;
        }
        if (cold.warm_start_leaves != 0) {
            printf("FAIL: cold arm reported %d warm leaves.\n", cold.warm_start_leaves);
            return 1;
        }
        // The real assertion: warm-start must not converge somewhere OTHER than
        // the equilibrium. Same oracle margin the cold sweep is held to.
        if (warm.exploitability_pct > mres.exploitability_pct + kMargin) {
            printf("FAIL: warm-start exploit %.4f%% exceeds monolithic %.4f%% + "
                   "%.2f%% margin — stale regrets are pulling it off-equilibrium.\n",
                   warm.exploitability_pct, mres.exploitability_pct, kMargin);
            return 1;
        }
        printf("PASS — warm-start reaches oracle quality (%.4f%% vs monolithic "
               "%.4f%%) at an inner count where cold is starved (%.4f%%).\n",
               warm.exploitability_pct, mres.exploitability_pct,
               cold.exploitability_pct);
    }

    // ======================================================================
    // Roadmap ③-A — trunk_iterations_per_sweep.
    //
    // `outer_iterations` used to mean BOTH "trunk CFR iterations" and "subgame
    // re-solves", 1:1 — so outer=8 gave the trunk 8 CFR iterations, where a
    // monolithic solve of the same flop needs thousands. But at a chance node
    // the trunk CFR reads a CACHED leaf value vector and never recurses into a
    // subgame, so trunk iterations cost ~nothing next to a subgame sweep.
    //
    // Guards the two claims the ③ conclusion rests on:
    //   1. K=1 is EXACTLY the old behaviour (trunk_iterations_run == outer).
    //   2. K>1 is ~free. If a refactor ever makes a trunk iteration expensive
    //      (e.g. by recursing into subgames instead of reading inj_), this fails.
    //
    // It deliberately does NOT assert that K improves exploitability, because
    // that is NOT universally true and the direction depends on which side is
    // starved. K trades trunk convergence against leaf-value staleness:
    //   - This fixture (SPR 2, single 0.5 sizing, raise_cap 1) has an almost
    //     converged trunk at K=1 (~0.96%), so extra trunk CFR just over-fits
    //     the trunk to a value function refreshed only 8 times: K=200 measured
    //     ~1.70% — WORSE.
    //   - A real, badly-under-converged trunk goes the other way: AsKsQs via
    //     the CLI (SPR 5, 0.33+0.75 menus, all-in on) at the same 8/60 measured
    //     60.06% -> 25.72% at identical wall clock.
    // Asserting the direction here would encode the wrong regime as the rule.
    {
        const int kOuter = 8, kInner = 60, kK = 200;

        auto run_k = [&](int k) {
            DecompositionOptions o;
            o.outer_iterations = kOuter;
            o.inner_iterations = kInner;
            o.cache_subgames   = true;
            o.trunk_iterations_per_sweep = k;
            double t = now_ms();
            DecomposedResult r = solve_decomposed(make_fixture(), o);
            return std::make_pair(r, now_ms() - t);
        };

        auto [k1, k1_ms]  = run_k(1);
        auto [kN, kN_ms]  = run_k(kK);
        if (!k1.ok || !kN.ok) { printf("FAIL: trunk-iters arm did not run.\n"); return 1; }

        printf("\n=== Roadmap ③-A trunk iterations (outer=%d, inner=%d) ===\n",
               kOuter, kInner);
        printf("  K=%3d: trunk_iters=%4d  exploit=%.4f%%  %.1fs\n",
               1, k1.trunk_iterations_run, k1.exploitability_pct, k1_ms / 1000.0);
        printf("  K=%3d: trunk_iters=%4d  exploit=%.4f%%  %.1fs\n",
               kK, kN.trunk_iterations_run, kN.exploitability_pct, kN_ms / 1000.0);

        if (k1.trunk_iterations_run != kOuter) {
            printf("FAIL: K=1 ran %d trunk iterations, expected %d — the default "
                   "is no longer the historical 1:1 coupling.\n",
                   k1.trunk_iterations_run, kOuter);
            return 1;
        }
        if (kN.trunk_iterations_run != kOuter * kK) {
            printf("FAIL: K=%d ran %d trunk iterations, expected %d.\n",
                   kK, kN.trunk_iterations_run, kOuter * kK);
            return 1;
        }
        if (!std::isfinite(kN.exploitability_pct)) {
            printf("FAIL: K=%d produced non-finite exploitability.\n", kK);
            return 1;
        }
        if (kN.subgame_solves != k1.subgame_solves) {
            printf("FAIL: K=%d ran %d subgame solves vs %d for K=1 — the extra "
                   "trunk iterations must NOT re-solve subgames.\n",
                   kK, kN.subgame_solves, k1.subgame_solves);
            return 1;
        }
        // The economic claim: ~free. Generous bound — a wall-clock assertion on
        // a shared machine must fail only on a real regression (a trunk
        // iteration becoming O(subgame)), not on noise.
        if (kN_ms > k1_ms * 2.0 + 5000.0) {
            printf("FAIL: K=%d took %.1fs vs %.1fs for K=1 — trunk iterations "
                   "are supposed to be nearly free (they read cached leaf "
                   "values); something is re-solving subgames per trunk iter.\n",
                   kK, kN_ms / 1000.0, k1_ms / 1000.0);
            return 1;
        }
        printf("PASS — %dx trunk CFR costs %.1fs vs %.1fs and the same %d "
               "subgame solves. Exploitability %.4f%% -> %.4f%% is NOT asserted: "
               "this fixture's trunk is already converged at K=1, so K over-fits "
               "it to stale leaf values (see comment).\n",
               kK, k1_ms / 1000.0, kN_ms / 1000.0, kN.subgame_solves,
               k1.exploitability_pct, kN.exploitability_pct);
    }

    // ======================================================================
    // Stage 5 — stitched UI navigation strategy tree (build_nav). Validates
    // that decomposition emits a navigable tree keyed EXACTLY like a monolithic
    // enumerated board: a flop root at the empty key, real turn cards enumerated
    // (dealt_cards + a multi-option runout picker + "#<card>" keys for the
    // non-lex-min turns), the river auto-skipped (turn+river depth in
    // dealt_cards), and EVs present. Structure only — a few iters suffice
    // (convergence was validated above); this is the Stage-5 engine milestone.
    // ======================================================================
    printf("\n=== Stage 5: stitched navigation strategy tree ===\n");
    {
        DecompositionOptions o;
        o.outer_iterations = 4;
        o.inner_iterations = 20;
        o.cache_subgames   = true;
        o.build_nav        = true;
        DecomposedResult dr = solve_decomposed(make_fixture(), o);
        if (!dr.ok) { printf("FAIL: nav decomposed run failed (ok=false).\n"); return 1; }
        const auto& tree = dr.strategy_tree;
        printf("stitched tree: %zu nodes  (leaves=%u turn cards)  truncated=%s\n",
               tree.size(), dr.leaf_count, dr.strategy_tree_truncated ? "true" : "false");
        if (tree.empty()) { printf("FAIL: stitched strategy_tree is empty.\n"); return 1; }

        // (1) Flop root at the no-suffix empty key, fully populated.
        auto root = tree.find("");
        if (root == tree.end()) { printf("FAIL: no root entry (key \"\").\n"); return 1; }
        if (root->second.acting != "OOP") {
            printf("FAIL: root acting=%s, expected OOP.\n", root->second.acting.c_str()); return 1; }
        if (root->second.action_labels.empty() || root->second.global_strategy.empty()) {
            printf("FAIL: root entry missing action_labels/global_strategy.\n"); return 1; }
        if (root->second.combo_evs.empty()) {
            printf("FAIL: root entry missing combo_evs (flop EV capture broken).\n"); return 1; }
        if (!root->second.dealt_cards.empty() || !root->second.runout_options.empty()) {
            printf("FAIL: flop root must have empty dealt_cards/runout_options.\n"); return 1; }

        // (2) Turn-enumeration markers across the whole tree.
        int n_turn = 0, n_picker = 0, n_hash = 0, n_turn_river = 0, n_sub_ev = 0;
        for (const auto& kv : tree) {
            const auto& e = kv.second;
            if (e.dealt_cards.size() >= 1) ++n_turn;
            if (e.runout_options.size() > 1) ++n_picker;
            if (kv.first.find('#') != std::string::npos) ++n_hash;
            if (e.dealt_cards.size() >= 2) ++n_turn_river;
            if (e.dealt_cards.size() >= 1 && !e.combo_evs.empty()) ++n_sub_ev;
        }
        printf("  turn nodes=%d  runout-picker nodes=%d  '#'-keyed (non-lex-min turn)=%d  "
               "turn+river depth=%d  subgame-nodes-with-EV=%d\n",
               n_turn, n_picker, n_hash, n_turn_river, n_sub_ev);

        if (dr.leaf_count <= 1) { printf("FAIL: only %u turn leaf — not a real enumeration.\n", dr.leaf_count); return 1; }
        if (n_turn == 0)        { printf("FAIL: no entry carries a dealt turn card.\n"); return 1; }
        if (n_picker == 0)      { printf("FAIL: no entry exposes a multi-option runout picker.\n"); return 1; }
        if (n_hash == 0)        { printf("FAIL: no '#'-keyed non-lex-min turn subtree (all turns must splice).\n"); return 1; }
        if (n_turn_river == 0)  { printf("FAIL: no turn+river-depth entry (river auto-skip missing).\n"); return 1; }
        if (n_sub_ev == 0)      { printf("FAIL: no subgame node carries combo_evs.\n"); return 1; }

        int shown = 0;
        for (const auto& kv : tree) {
            if (!kv.second.dealt_cards.empty()) {
                printf("  e.g. key=\"%s\"  acting=%s  dealt=%zu  runouts=%zu\n",
                       kv.first.c_str(), kv.second.acting.c_str(),
                       kv.second.dealt_cards.size(), kv.second.runout_options.size());
                if (++shown >= 3) break;
            }
        }
        printf("PASS — stitched nav tree: flop trunk + %u turn subgames spliced "
               "(lex-min no-suffix, others '#<card>'), river auto-skipped, EVs present.\n",
               dr.leaf_count);
    }

#ifdef DEEPSOLVER_USE_CUDA
    // ======================================================================
    // Stage 3 — GPU subgame streaming. Same spot, subgames solved on the
    // GpuBackend (each subgame is a turn solve). Two claims:
    //   (A) PARITY  — all subgames resident (cap 0) must reach the SAME
    //                 exploitability as the monolithic oracle ⇒ the GPU
    //                 trunk↔subgame coupling is numerically correct.
    //   (B) BOUNDED — cap = N keeps at most N GPU solvers alive at once, so
    //                 peak VRAM is bounded INDEPENDENT of the turn-card count,
    //                 AND the answer is unchanged (cap is a memory knob, not a
    //                 math knob — GpuBackend::prepare() resets device regrets
    //                 each solve, so resolve()/rebuild are both from-scratch).
    // This is the production rainbow-flop path: the monolithic rainbow solve
    // is ~76 GB; a couple of resident turn subgames are a few GB.
    // Env overrides for manual experiments: DEEPSOLVER_GPU_OUTER / _INNER / _CAP.
    // ======================================================================
    printf("\n=== Stage 3: GPU subgame streaming ===\n");
    {
        int g_outer = outer;
        int g_inner = inner_sweep.back();
        int g_cap   = 2;   // bounded-run residency cap
        if (const char* e = std::getenv("DEEPSOLVER_GPU_OUTER")) g_outer = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_INNER")) g_inner = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_CAP"))   g_cap   = std::atoi(e);

        // (A) all-resident parity run.
        DecompositionOptions a;
        a.outer_iterations = g_outer;
        a.inner_iterations = g_inner;
        a.subgame_backend  = BackendType::GPU;
        a.gpu_resident_cap = 0;                 // keep every subgame resident
        double a0 = now_ms();
        DecomposedResult ar = solve_decomposed(make_fixture(), a);
        double ams = now_ms() - a0;
        if (!ar.ok) { printf("FAIL: GPU all-resident run failed (ok=false).\n"); return 1; }
        printf("GPU all-resident : leaves=%u  peak_resident=%d  exploit=%.4f%%  "
               "(mono %.4f%%)  %.1fs\n",
               ar.leaf_count, ar.gpu_peak_resident, ar.exploitability_pct,
               mres.exploitability_pct, ams / 1000.0);

        // (B) bounded streaming run.
        DecompositionOptions b;
        b.outer_iterations = g_outer;
        b.inner_iterations = g_inner;
        b.subgame_backend  = BackendType::GPU;
        b.gpu_resident_cap = g_cap;             // at most g_cap resident
        double b0 = now_ms();
        DecomposedResult brun = solve_decomposed(make_fixture(), b);
        double bms = now_ms() - b0;
        if (!brun.ok) { printf("FAIL: GPU bounded run failed (ok=false).\n"); return 1; }
        printf("GPU bounded cap=%d: leaves=%u  peak_resident=%d  exploit=%.4f%%  %.1fs\n",
               g_cap, brun.leaf_count, brun.gpu_peak_resident, brun.exploitability_pct,
               bms / 1000.0);

        if (!std::isfinite(ar.exploitability_pct) || !std::isfinite(brun.exploitability_pct)) {
            printf("FAIL: non-finite GPU exploitability.\n"); return 1;
        }
        // Parity: GPU must match the monolithic oracle within the same margin.
        if (!diag_mode && ar.exploitability_pct > mres.exploitability_pct + kMargin) {
            printf("FAIL: GPU exploit %.4f%% exceeds monolithic %.4f%% + %.2f%% "
                   "margin — GPU coupling regressed.\n",
                   ar.exploitability_pct, mres.exploitability_pct, kMargin);
            return 1;
        }
        // Bounded: peak residency must respect the cap.
        if (brun.gpu_peak_resident > g_cap) {
            printf("FAIL: GPU peak residency %d exceeds bound %d — NOT bounded.\n",
                   brun.gpu_peak_resident, g_cap);
            return 1;
        }
        // cap is a memory knob, not a math knob: bounded ≈ all-resident.
        float capgap = std::fabs(brun.exploitability_pct - ar.exploitability_pct);
        if (capgap > 0.05f) {
            printf("FAIL: bounded vs all-resident exploit differ by %.4f%% — the "
                   "resident cap changed the answer (bug).\n", capgap);
            return 1;
        }
        printf("PASS — GPU streaming validated: exploit %.4f%% matches oracle "
               "%.4f%%; bounded to %d resident (all-resident peak %d), Δ=%.4f%%.\n",
               brun.exploitability_pct, mres.exploitability_pct,
               brun.gpu_peak_resident, ar.gpu_peak_resident, capgap);

        // (C) Hybrid quality fix: GPU outer iters (fast value generation) + a
        // CPU final pass (CPU-quality delivered strategies). Should recover the
        // ~0.13pp GPU-backend offset and land near the all-CPU decomposition.
        DecompositionOptions c;
        c.outer_iterations = g_outer;
        c.inner_iterations = g_inner;
        c.subgame_backend  = BackendType::GPU;
        c.gpu_resident_cap = 0;
        c.cpu_final_pass   = true;
        double c0 = now_ms();
        DecomposedResult cr = solve_decomposed(make_fixture(), c);
        double cms = now_ms() - c0;
        if (!cr.ok) { printf("FAIL: GPU+CPU-final run failed (ok=false).\n"); return 1; }
        printf("GPU+CPU-final    : exploit=%.4f%%  (CPU-decomp %.4f%%, GPU-only %.4f%%)  %.1fs\n",
               cr.exploitability_pct, best_exploit, ar.exploitability_pct, cms / 1000.0);
        if (!diag_mode) {
            if (cr.exploitability_pct <= best_exploit + 0.10f)
                printf("PASS — hybrid recovers CPU quality: %.4f%% ≈ CPU-decomp %.4f%% "
                       "(GPU-only was %.4f%%).\n",
                       cr.exploitability_pct, best_exploit, ar.exploitability_pct);
            else
                printf("NOTE: hybrid %.4f%% above CPU-decomp %.4f%%+0.10%% — "
                       "the offset is not only in the final pass.\n",
                       cr.exploitability_pct, best_exploit);
        }

        // (D) Stage 3.5 ADAPTIVE pin-first-K (gpu_resident_cap = -1): K is
        // measured from free VRAM. Must match the all-resident answer (proves
        // the pinned resolve_keep_board reuse is correct) and report the chosen
        // K (proves it adapts to the GPU without OOM).
        DecompositionOptions d;
        d.outer_iterations = g_outer;
        d.inner_iterations = g_inner;
        d.subgame_backend  = BackendType::GPU;
        d.gpu_resident_cap = -1;            // adaptive
        double d0 = now_ms();
        DecomposedResult drr = solve_decomposed(make_fixture(), d);
        double dms = now_ms() - d0;
        if (!drr.ok) { printf("FAIL: GPU adaptive run failed (ok=false).\n"); return 1; }
        printf("GPU adaptive     : K(pinned)=%d/%u  peak_resident=%d  exploit=%.4f%%  %.1fs\n",
               drr.gpu_cap_used, drr.leaf_count, drr.gpu_peak_resident,
               drr.exploitability_pct, dms / 1000.0);
        if (!std::isfinite(drr.exploitability_pct)) {
            printf("FAIL: non-finite adaptive exploit.\n"); return 1;
        }
        if (!diag_mode && drr.exploitability_pct > mres.exploitability_pct + kMargin) {
            printf("FAIL: adaptive exploit %.4f%% exceeds monolithic %.4f%% + %.2f%% — "
                   "pinned reuse regressed the answer.\n",
                   drr.exploitability_pct, mres.exploitability_pct, kMargin);
            return 1;
        }
        float adgap = std::fabs(drr.exploitability_pct - ar.exploitability_pct);
        if (adgap > 0.05f) {
            printf("FAIL: adaptive vs all-resident exploit differ by %.4f%% — "
                   "pinned reuse changed the answer (bug).\n", adgap);
            return 1;
        }
        printf("PASS — GPU adaptive: pinned K=%d turn subgames chosen from measured "
               "VRAM, exploit %.4f%% matches all-resident %.4f%% (Δ=%.4f%%), no OOM.\n",
               drr.gpu_cap_used, drr.exploitability_pct, ar.exploitability_pct, adgap);
    }

    // ======================================================================
    // Stage 5 (rainbow headline) — the texture the project is about. On a
    // RAINBOW flop the monolithic build COLLAPSES the turn to one equity-blind
    // child (runout_approximated = true; full enumeration is ~76 GB). The
    // decomposition route with build_nav instead enumerates the REAL turn cards
    // on the GPU under BOUNDED VRAM and stitches a navigable strategy tree —
    // exactly what the sidecar's --decompose-runouts auto path delivers. Small
    // iters: this asserts STRUCTURE + bounded residency, not convergence.
    // ======================================================================
    printf("\n=== Stage 5 (rainbow): stitched nav under bounded VRAM ===\n");
    {
        SolverConfig rb;
        rb.pot = 100.0f;
        rb.effective_stack = 60.0f;                 // low SPR → tractable tree
        auto b = parse_board("AsKd2c");             // rainbow → monolithic collapses
        rb.board_size = 3;
        for (int i = 0; i < 3; ++i) rb.board[i] = b[i];
        rb.bet_sizing.flop_sizes  = {0.75f};
        rb.bet_sizing.turn_sizes  = {0.75f};
        rb.bet_sizing.river_sizes = {0.75f};
        rb.bet_sizing.flop_allin = rb.bet_sizing.turn_allin = rb.bet_sizing.river_allin = false;
        rb.raise_cap = 1;
        rb.oop_has_initiative = true;
        rb.ip_range_weights.fill(1.0f);
        rb.oop_range_weights.fill(1.0f);

        // (1) Monolithic build collapses (the bug this feature fixes).
        SolverConfig mc = rb;
        mc.max_iterations = 1;
        mc.compute_exploitability = false;
        mc.compute_combo_evs = false;
        Solver rbmono(mc, BackendType::CPU);
        SolverResult rbmr = rbmono.solve();
        printf("rainbow monolithic: nodes=%u  runout_approximated=%s\n",
               rbmr.timing.tree_nodes,
               rbmr.runout_approximated ? "TRUE (collapsed, equity-blind)" : "false");
        if (!rbmr.runout_approximated) {
            printf("FAIL: rainbow did not collapse — fixture no longer exercises the gate.\n");
            return 1;
        }

        // (2) Decomposed GPU streaming + build_nav: real turns, bounded VRAM.
        DecompositionOptions o;
        o.outer_iterations = 2;
        o.inner_iterations = 10;
        o.subgame_backend  = BackendType::GPU;
        o.gpu_resident_cap = 4;
        o.build_nav        = true;
        o.nav_max_player_depth = 8;
        if (const char* e = std::getenv("DEEPSOLVER_GPU_OUTER")) o.outer_iterations = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_INNER")) o.inner_iterations = std::atoi(e);
        if (const char* e = std::getenv("DEEPSOLVER_GPU_CAP"))   o.gpu_resident_cap = std::atoi(e);
        double r0 = now_ms();
        DecomposedResult dr = solve_decomposed(rb, o);
        double rms = now_ms() - r0;
        if (!dr.ok) { printf("FAIL: rainbow decomposed run failed (ok=false).\n"); return 1; }

        int n_hash = 0, n_turn = 0, n_picker = 0;
        for (const auto& kv : dr.strategy_tree) {
            if (kv.first.find('#') != std::string::npos) ++n_hash;
            if (kv.second.dealt_cards.size() >= 1) ++n_turn;
            if (kv.second.runout_options.size() > 1) ++n_picker;
        }
        printf("decomposed GPU+nav: real turn leaves=%u (vs 1 collapsed)  nc=%u  "
               "peak_resident=%d/cap=%d  tree=%zu nodes  '#'turns=%d  pickers=%d  %.1fs\n",
               dr.leaf_count, dr.num_canonical, dr.gpu_peak_resident, o.gpu_resident_cap,
               dr.strategy_tree.size(), n_hash, n_picker, rms / 1000.0);

        if (dr.leaf_count <= 1)            { printf("FAIL: only %u turn leaf — no real enumeration.\n", dr.leaf_count); return 1; }
        if (dr.strategy_tree.empty())      { printf("FAIL: rainbow stitched tree empty.\n"); return 1; }
        if (n_turn == 0)                   { printf("FAIL: no turn card in the stitched tree.\n"); return 1; }
        if (n_hash == 0)                   { printf("FAIL: no '#'-keyed turn subtree (turns not all spliced).\n"); return 1; }
        if (n_picker == 0)                 { printf("FAIL: no multi-option runout picker.\n"); return 1; }
        if (dr.gpu_peak_resident > o.gpu_resident_cap) {
            printf("FAIL: peak residency %d exceeds cap %d — NOT bounded.\n",
                   dr.gpu_peak_resident, o.gpu_resident_cap); return 1;
        }
        printf("PASS — rainbow: monolithic collapses (equity-blind), decomposition "
               "enumerates %u real turn cards into a navigable tree under BOUNDED "
               "VRAM (%d resident) where monolithic is ~76 GB.\n",
               dr.leaf_count, dr.gpu_peak_resident);
    }
#endif
    return 0;
}
