import { useState, useCallback, useRef } from 'react';
import type { SolverRequest, SolverResponse, ComboStrategy, EstimateResponse, EngineAction } from '../lib/poker';
import {
  GRID_LABELS,
  SUITS,
  parseBoardCards,
  getHandStrength,
  expandComboLabel,
  RANK_VALUES,
} from '../lib/poker';
import { parseRange } from '../lib/ranges';
import { isTauri } from '../lib/tauriEnv';

// ============================================================================
// Progress state
// ============================================================================

export interface SolverProgress {
  iteration: number;
  total: number;
  elapsed: number;
  phase: string;
  pct: number; // 0-100
  /** Present while an Exact (runout decomposition) run is in its decompose
   *  phase. iteration/total then carry subgame counts for the current pass,
   *  not CFR iterations — StrategyPanel swaps the tile label accordingly. */
  decompose?: DecomposeProgressInfo;
}

/** Mirrors the Rust `DecomposeProgress` event payload (camelCased). */
export interface DecomposeProgressInfo {
  phase: 'sweep' | 'final' | 'finalize';
  sweep: number;      // 1-based; 0 outside the sweep phase
  sweepTotal: number;
  leaf: number;
  leafTotal: number;
}

// ============================================================================
// Contextual strategy generation
// ============================================================================

function contextualStrategy(
  strength: number,
  _idx: number,
  facingAction: 'none' | 'bet' | 'raise',
): ComboStrategy {
  const noise = (Math.sin(_idx * 137.5) * 0.08);

  if (facingAction === 'none') {
    if (strength > 0.85) {
      return { 'Check': 0.15 + noise * 0.5, 'Bet 33%': 0.20, 'Bet 75%': 0.50 - noise * 0.3, 'All-in': 0.15 };
    } else if (strength > 0.65) {
      return { 'Check': 0.10, 'Bet 33%': 0.55 + noise, 'Bet 75%': 0.30 - noise, 'All-in': 0.05 };
    } else if (strength > 0.45) {
      return { 'Check': 0.55 + noise, 'Bet 33%': 0.35 - noise, 'Bet 75%': 0.08, 'All-in': 0.02 };
    } else if (strength > 0.25) {
      return { 'Check': 0.72 + noise, 'Bet 33%': 0.18 - noise * 0.5, 'Bet 75%': 0.08, 'All-in': 0.02 };
    } else {
      return { 'Check': 0.65, 'Bet 33%': 0.15 + noise, 'Bet 75%': 0.12 - noise, 'All-in': 0.08 };
    }
  } else if (facingAction === 'bet') {
    if (strength > 0.85) {
      return { 'Fold': 0.0, 'Call': 0.30 + noise, 'Raise 3x': 0.55 - noise, 'All-in': 0.15 };
    } else if (strength > 0.65) {
      return { 'Fold': 0.05, 'Call': 0.70 + noise, 'Raise 3x': 0.20 - noise, 'All-in': 0.05 };
    } else if (strength > 0.45) {
      return { 'Fold': 0.25 - noise * 0.5, 'Call': 0.60 + noise, 'Raise 3x': 0.12, 'All-in': 0.03 };
    } else if (strength > 0.25) {
      return { 'Fold': 0.55 + noise, 'Call': 0.30 - noise, 'Raise 3x': 0.10, 'All-in': 0.05 };
    } else {
      return { 'Fold': 0.75, 'Call': 0.08, 'Raise 3x': 0.07 + noise, 'All-in': 0.10 - noise };
    }
  } else {
    if (strength > 0.85) {
      return { 'Fold': 0.0, 'Call': 0.45 + noise, 'All-in': 0.55 - noise };
    } else if (strength > 0.65) {
      return { 'Fold': 0.15, 'Call': 0.70 + noise, 'All-in': 0.15 - noise };
    } else if (strength > 0.45) {
      return { 'Fold': 0.50, 'Call': 0.45 + noise, 'All-in': 0.05 };
    } else {
      return { 'Fold': 0.85 + noise * 0.3, 'Call': 0.10 - noise * 0.2, 'All-in': 0.05 };
    }
  }
}

// ============================================================================
// Per-combo strategy generation (grid labels + specific combos)
// ============================================================================

/** What the last action of an engine history string faces. */
function facingOf(history?: string): 'none' | 'bet' | 'raise' {
  const last = history ? history.split(',').pop() ?? '' : '';
  if (last.startsWith('Bet')) return 'bet';
  if (last.startsWith('Raise') || last.startsWith('All-in')) return 'raise';
  return 'none';
}

function generateComboStrategies(
  board: string,
  heroRangeStr?: string,
  history?: string,
): Record<string, ComboStrategy> {
  const boardCards = parseBoardCards(board);
  const boardRanks = boardCards.map(c => c[0]).sort((a, b) =>
    (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0)
  );

  const street = board.length >= 10 ? 'river' : board.length >= 8 ? 'turn' : 'flop';

  const facingAction = facingOf(history);

  const heroRange = heroRangeStr ? parseRange(heroRangeStr) : null;
  const strategies: Record<string, ComboStrategy> = {};
  const flat = GRID_LABELS.flat();

  const depth = history ? history.split(',').length : 0;
  const streetMod = street === 'river' ? 0.12 : street === 'turn' ? 0.06 : 0;

  // Build set of board suits for board-interaction heuristic
  const boardSuits = new Set(boardCards.map(c => c[1]));

  for (let idx = 0; idx < flat.length; idx++) {
    const label = flat[idx];
    const rangeFreq = heroRange ? (heroRange[label] ?? 0) : 1.0;
    const depthThreshold = 0.001 + depth * 0.08 + streetMod;

    if (rangeFreq <= depthThreshold) {
      strategies[label] = { 'Not in range': 1.0 };
      continue;
    }

    const strength = getHandStrength(label, boardRanks);
    const baseStrategy = contextualStrategy(strength, idx, facingAction);
    strategies[label] = baseStrategy;

    // Also generate per-specific-combo strategies with suit-based variation
    const specifics = expandComboLabel(label, boardCards);
    for (let si = 0; si < specifics.length; si++) {
      const sc = specifics[si];
      if (sc.isDead) {
        strategies[sc.combo] = { 'Not in range': 1.0 };
        continue;
      }

      // Vary strategy slightly based on suit interaction with board
      const hasFlushDraw = sc.suit1 === sc.suit2 && boardSuits.has(sc.suit1);
      const hasBackdoor = sc.suit1 !== sc.suit2 && (boardSuits.has(sc.suit1) || boardSuits.has(sc.suit2));
      const suitBonus = hasFlushDraw ? 0.08 : hasBackdoor ? 0.03 : 0;

      // Apply per-suit noise for unique variation
      const suitIdx = SUITS.indexOf(sc.suit1) * 4 + SUITS.indexOf(sc.suit2);
      const suitNoise = Math.sin((idx * 13 + suitIdx) * 73.7) * 0.06;

      const adjustedStrength = Math.min(1, Math.max(0, strength + suitBonus + suitNoise));
      strategies[sc.combo] = contextualStrategy(adjustedStrength, idx * 16 + suitIdx, facingAction);
    }
  }

  return strategies;
}

function computeGlobalStrategy(combos: Record<string, ComboStrategy>): Record<string, string> {
  const counts: Record<string, number> = {};
  let inRangeCount = 0;

  // Only use grid-label-level strategies (not specific combos) for global
  const flat = new Set(GRID_LABELS.flat());

  for (const [key, strategy] of Object.entries(combos)) {
    if (!flat.has(key)) continue; // skip specific combos
    if (strategy['Not in range']) continue;
    inRangeCount++;
    for (const [action, freq] of Object.entries(strategy)) {
      counts[action] = (counts[action] || 0) + freq;
    }
  }

  if (inRangeCount === 0) return {};

  const result: Record<string, string> = {};
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  for (const [action, sum] of sorted) {
    const pct = (sum / inRangeCount) * 100;
    if (pct >= 0.5) {
      result[action] = `${pct.toFixed(1)}%`;
    }
  }

  return result;
}

// ============================================================================
// Mock solver with progress simulation
// ============================================================================

async function mockSolve(
  request: SolverRequest,
  onProgress?: (p: SolverProgress) => void,
): Promise<SolverResponse> {
  const total = request.iterations ?? 300;
  const totalTime = request.history ? 400 : 1000;
  const startMs = Date.now();

  // Simulate progress ticks (use setTimeout for compatibility with background tabs)
  await new Promise<void>(resolve => {
    const tick = () => {
      const elapsed = Date.now() - startMs;
      const iter = Math.min(total, Math.floor((elapsed / totalTime) * total));
      const pct = (iter / total) * 100;

      let phase = 'Building tree...';
      if (pct > 10) phase = `Solving iteration ${iter}/${total}...`;
      if (pct > 90) phase = 'Finalizing...';

      onProgress?.({ iteration: iter, total, elapsed, phase, pct: Math.min(99, pct) });

      if (elapsed < totalTime) {
        setTimeout(tick, 50);
      } else {
        onProgress?.({ iteration: total, total, elapsed, phase: 'Done', pct: 100 });
        resolve();
      }
    };
    setTimeout(tick, 50);
  });

  const comboStrategies = generateComboStrategies(
    request.board,
    request.hero_range,
    request.history,
  );
  const globalStrategy = computeGlobalStrategy(comboStrategies);

  // Browser preview: a root node whose buttons mirror the heuristic labels.
  // Navigating further needs the engine (desktop app).
  const actions: EngineAction[] = Object.keys(globalStrategy).map(label => ({
    label,
    type: label.startsWith('Check') ? 'check' : label.startsWith('All-in') ? 'allin'
      : label.startsWith('Fold') ? 'fold' : label.startsWith('Call') ? 'call'
      : label.startsWith('Raise') ? 'raise' : 'bet',
    amount: 0, raise_to: 0, next: 'player',
  }));
  const baseResponse: SolverResponse = {
    status: 'success',
    iterations_run: total,
    exploitability_pct: 0.32,
    global_strategy: globalStrategy,
    combo_strategies: comboStrategies,
    acting_player: 'OOP',
    node: {
      kind: 'player', street: request.board.length >= 10 ? 2 : request.board.length >= 8 ? 1 : 0,
      pot: request.pot_size, stack_oop: request.effective_stack, stack_ip: request.effective_stack,
      to_call: 0, board: [], actions, runouts: [],
    },
  };

  if (request.target_combo) {
    const combo = request.target_combo;
    let strategy = comboStrategies[combo];
    const wasOffRange = !strategy || !!strategy['Not in range'];

    // If off-range, force-generate a real strategy for this combo
    // (simulates re-solving with it added to the range)
    if (wasOffRange) {
      const boardCards = parseBoardCards(request.board);
      const boardRanks = boardCards.map(c => c[0]).sort((a, b) =>
        (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0)
      );
      const strength = getHandStrength(combo, boardRanks);

      strategy = contextualStrategy(strength, combo.charCodeAt(0) * 17, facingOf(request.history));
      // Store it back so combo variants can find it too
      comboStrategies[combo] = strategy;
    }

    if (strategy) {
      const bestAction = Object.entries(strategy)
        .filter(([k]) => k !== 'Not in range')
        .reduce(
          (best, [action, freq]) => freq > best[1] ? [action, freq] as [string, number] : best,
          ['', 0] as [string, number]
        )[0];

      const boardCards = parseBoardCards(request.board);
      const boardRanks = boardCards.map(c => c[0]).sort((a, b) =>
        (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0)
      );
      const strength = getHandStrength(combo, boardRanks);
      const ev = (strength - 0.5) * request.pot_size * 0.8;

      baseResponse.target_combo_analysis = {
        combo,
        best_action: bestAction,
        ev: parseFloat(ev.toFixed(2)),
        strategy_mix: Object.fromEntries(
          Object.entries(strategy)
            .filter(([k, f]) => k !== 'Not in range' && f > 0.01)
            .map(([a, f]) => [a, `${(f * 100).toFixed(1)}%`])
        ),
      };
    }
  }

  return baseResponse;
}

// ============================================================================
// useSolver hook
// ============================================================================

// v1.8.3: GPU OOM detection for the engine's structured error string.
//   "GpuBackend: solver state would require N MB but only M MB free on device"
// is what the engine produces when a wide-range × monotone × 2-sizing tree
// overflows the user's GPU memory. We retry once with single-sizing flop
// to drop tree size by ~50% and fit common 24-32 GB GPUs.
function isGpuOom(msg: string): boolean {
  return /GpuBackend:.*solver state would require.*out of memory/i.test(msg)
      || /needs.*MB.*free.*MB.*out of memory/i.test(msg);
}

/** The error a Stop click ends a solve with (Rust engine::CANCELLED_MESSAGE). */
const CANCELLED_MESSAGE = 'Solve cancelled';

/**
 * Runs solves: progress, the pre-solve estimate, errors. `solve()` resolves
 * to the engine's response, or null when the solve failed (`error` says why)
 * or was stopped. What is shown is the caller's business (useTreeNav).
 */
export function useSolver() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [progress, setProgress] = useState<SolverProgress | null>(null);
  // v1.2.2: pre-solve ETA + memory preview from `--estimate-only`. Set by
  // `solve()` before the actual subprocess fires (sub-second cost) so the
  // UI can show "Estimated 12 minutes on CPU" before the user commits.
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null);
  const timerRef = useRef<number | null>(null);
  // One solve at a time: a second call while one runs is refused instead of
  // racing it (both used to write the shared result).
  const runningRef = useRef(false);

  const solve = useCallback(async (request: SolverRequest): Promise<SolverResponse | null> => {
    if (runningRef.current) return null;
    runningRef.current = true;
    setLoading(true);
    setError(null);
    setEstimate(null);  // clear stale estimate from previous solve
    setProgress({ iteration: 0, total: request.iterations ?? 300, elapsed: 0, phase: 'Starting...', pct: 0 });
    const start = Date.now();

    // v1.2.2: fire the pre-solve estimate in parallel with the loading
    // state. Sub-second on most spots, ~1s on monotone iso boards. The
    // banner appears as soon as the estimate lands, well before iterations
    // start showing progress. If the estimate call itself fails (engine
    // missing, broken request), we silently skip — better to start the
    // real solve than block on an estimator hiccup.
    if (isTauri()) {
      (async () => {
        try {
          const { invoke } = await import('@tauri-apps/api/core');
          const est = await invoke<EstimateResponse>('estimate_solve', { request });
          setEstimate(est);
        } catch (e) {
          // Don't surface — the actual solve provides the same data on
          // completion. This only suppresses the pre-solve banner.
          console.warn('estimate_solve failed (non-fatal):', e);
        }
      })();
    }

    // v1.4.1: in Tauri mode, listen for real `engine-progress` events the
    // Rust shell emits while parsing engine stderr line-by-line. Each event
    // carries the actual iteration number, exploitability, and elapsed ms
    // from the C++ solver. Replaces the previous setInterval-based fake
    // progress that estimated ~70ms/iter and capped at 95% — that broke
    // entirely on slow CPU spots (5+ s/iter) where the bar would freeze at
    // 95% / iter 285 within 20 seconds and stay there for the rest of the
    // run. We keep a small JS-side ticker for "elapsed" so the UI doesn't
    // feel frozen between engine ticks, but the iter count is real.
    let progressUnlisten: (() => void) | null = null;
    let decomposeUnlisten: (() => void) | null = null;
    let lastIter = 0;
    let lastExploit = 0;
    // Exact mode: once the engine's decompose phase emits per-subgame
    // progress, it owns the bar. The monolithic pre-solve's [Iter] stream has
    // ended by then (decompose runs strictly after it), so the bar visibly
    // resets from ~99% into "Exact: sweep 1/N" — that phase boundary is real,
    // and the SolveEtaBanner above already told the user the Exact ETA.
    let decompProgress: SolverProgress | null = null;
    if (isTauri()) {
      const { listen } = await import('@tauri-apps/api/event');
      progressUnlisten = await listen<{ iteration: number; exploitability_pct: number; elapsed_ms: number }>(
        'engine-progress',
        (e) => {
          if (decompProgress) return; // decompose phase owns progress now
          lastIter = e.payload.iteration;
          lastExploit = e.payload.exploitability_pct;
          const total = request.iterations ?? 300;
          const pct = Math.min(99, (lastIter / total) * 100);
          const phase = `Solving iteration ${lastIter}/${total}...`;
          setProgress({
            iteration: lastIter,
            total,
            elapsed: e.payload.elapsed_ms,
            phase,
            pct,
          });
        },
      );
      decomposeUnlisten = await listen<{
        phase: 'sweep' | 'final' | 'finalize';
        sweep: number; sweep_total: number; leaf: number; leaf_total: number;
      }>(
        'engine-progress-decompose',
        (e) => {
          const p = e.payload;
          const info: DecomposeProgressInfo = {
            phase: p.phase, sweep: p.sweep, sweepTotal: p.sweep_total,
            leaf: p.leaf, leafTotal: p.leaf_total,
          };
          // Work model: each sweep = L leaf-units; the final consistent-BR
          // pass ≈ 3× a sweep leaf (1.73× want_br solve + nav extraction —
          // see the calibrated ETA constants in solver_decomposed.h).
          const FINAL_W = 3;
          const L = Math.max(1, p.leaf_total);
          const sweeps = Math.max(1, p.sweep_total);
          const totalWork = sweeps * L + FINAL_W * L;
          let pct: number;
          let phase: string;
          let leafShown = p.leaf;
          let leafTotalShown = p.leaf_total;
          if (p.phase === 'sweep') {
            const done = (Math.max(1, p.sweep) - 1) * L + p.leaf;
            pct = Math.min(99, (done / totalWork) * 100);
            phase = `Exact: sweep ${p.sweep}/${p.sweep_total} · subgame ${p.leaf}/${p.leaf_total}`;
          } else if (p.phase === 'final') {
            const done = sweeps * L + FINAL_W * p.leaf;
            pct = Math.min(99, (done / totalWork) * 100);
            phase = `Exact: final pass · subgame ${p.leaf}/${p.leaf_total}`;
          } else {
            // finalize: subgames done; EV/BR traversals + nav stitch + JSON
            // remain. Keep the completed final-pass counts on the tile.
            pct = 99;
            phase = 'Exact: finalizing...';
            leafShown = decompProgress?.decompose?.leafTotal ?? 0;
            leafTotalShown = decompProgress?.decompose?.leafTotal ?? 0;
          }
          decompProgress = {
            iteration: leafShown,
            total: leafTotalShown,
            elapsed: Date.now() - start,
            phase,
            pct,
            decompose: info,
          };
          setProgress(decompProgress);
        },
      );
    }

    // Lightweight elapsed-only ticker so the UI doesn't appear frozen
    // between engine progress events on slow per-iter spots. Doesn't fake
    // the iter count anymore — that comes from `engine-progress`.
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      const now = Date.now();
      const total = request.iterations ?? 300;
      const elapsedMs = now - start;
      if (decompProgress) {
        // Decompose phase owns the bar — only refresh the elapsed clock.
        setProgress({ ...decompProgress, elapsed: elapsedMs });
        return;
      }
      const pct = Math.min(99, (lastIter / total) * 100);
      const phase = lastIter > 0
        ? `Solving iteration ${lastIter}/${total}...`
        : 'Building tree...';
      setProgress({ iteration: lastIter, total, elapsed: elapsedMs, phase, pct });
      void lastExploit;
    }, 250);

    try {
      let response: SolverResponse;

      if (isTauri()) {
        const { invoke } = await import('@tauri-apps/api/core');
        // A GPU failure (out of memory included) is retried on the CPU by
        // the Rust side (engine::run_solver), which says so in
        // resources.fallback_reason.
        response = await invoke<SolverResponse>('solve', { request });
      } else {
        // Clear the generic timer — mock solver handles its own progress
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        response = await mockSolve(request, setProgress);
      }

      setElapsed(Date.now() - start);
      setProgress({ iteration: response.iterations_run, total: request.iterations ?? 300, elapsed: Date.now() - start, phase: 'Done', pct: 100 });
      return response;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === CANCELLED_MESSAGE) {
        // Stop: neutral — the previous view stays.
      } else if (isGpuOom(msg)) {
        // User-actionable hint instead of the raw engine error string.
        setError('GPU memory exceeded on this spot\'s tree. Try a smaller bet-sizing menu (e.g. the "Lite" preset: one 50% size per street).');
      } else {
        setError(msg);
      }
      setProgress(null);
      return null;
    } finally {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      if (progressUnlisten) { progressUnlisten(); progressUnlisten = null; }
      if (decomposeUnlisten) { decomposeUnlisten(); decomposeUnlisten = null; }
      setLoading(false);
      runningRef.current = false;
    }
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setElapsed(0);
    setProgress(null);
    setEstimate(null);
  }, []);

  return { loading, error, setError, elapsed, progress, estimate, solve, reset };
}
