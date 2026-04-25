import { useState, useCallback, useRef } from 'react';
import type { SolverRequest, SolverResponse, ComboStrategy } from '../lib/poker';
import {
  GRID_LABELS,
  SUITS,
  parseBoardCards,
  getHandStrength,
  expandComboLabel,
  RANK_VALUES,
} from '../lib/poker';
import { parseRange } from '../lib/ranges';
import type { ActionStep } from '../lib/gameTree';
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

function generateComboStrategies(
  board: string,
  heroRangeStr?: string,
  actionPath?: ActionStep[],
): Record<string, ComboStrategy> {
  const boardCards = parseBoardCards(board);
  const boardRanks = boardCards.map(c => c[0]).sort((a, b) =>
    (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0)
  );

  const street = board.length >= 10 ? 'river' : board.length >= 8 ? 'turn' : 'flop';

  let facingAction: 'none' | 'bet' | 'raise' = 'none';
  if (actionPath && actionPath.length > 0) {
    const last = actionPath[actionPath.length - 1];
    if (last.player !== 'Deal') {
      if (last.action.type === 'bet') facingAction = 'bet';
      else if (last.action.type === 'raise' || last.action.type === 'allin') facingAction = 'raise';
    }
  }

  const heroRange = heroRangeStr ? parseRange(heroRangeStr) : null;
  const strategies: Record<string, ComboStrategy> = {};
  const flat = GRID_LABELS.flat();

  const depth = actionPath ? actionPath.filter(s => s.player !== 'Deal').length : 0;
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
  const hasActions = request.action_path && request.action_path.length > 0;
  const totalTime = hasActions ? 400 : 1000;
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
    request.action_path,
  );
  const globalStrategy = computeGlobalStrategy(comboStrategies);

  const baseResponse: SolverResponse = {
    status: 'success',
    iterations_run: total,
    exploitability_pct: 0.32,
    global_strategy: globalStrategy,
    combo_strategies: comboStrategies,
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

      let facingAction: 'none' | 'bet' | 'raise' = 'none';
      if (request.action_path && request.action_path.length > 0) {
        const last = request.action_path[request.action_path.length - 1];
        if (last.player !== 'Deal') {
          if (last.action.type === 'bet') facingAction = 'bet';
          else if (last.action.type === 'raise' || last.action.type === 'allin') facingAction = 'raise';
        }
      }

      strategy = contextualStrategy(strength, combo.charCodeAt(0) * 17, facingAction);
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

export function useSolver() {
  const [result, setResultRaw] = useState<SolverResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [progress, setProgress] = useState<SolverProgress | null>(null);
  const timerRef = useRef<number | null>(null);
  // Always-fresh mirror of `result` so `navigate()` can decide hit/miss
  // synchronously without a stale closure (state updates may be batched and
  // the updater function may run after navigate returns).
  const resultRef = useRef<SolverResponse | null>(null);

  // Wrap setResult so external callers (App.tsx) keep the ref in sync.
  // Supports both direct value and functional updater forms.
  const setResult = useCallback(
    (next: SolverResponse | null
       | ((prev: SolverResponse | null) => SolverResponse | null)) => {
      if (typeof next === 'function') {
        setResultRaw(prev => {
          const computed = (next as (p: SolverResponse | null) => SolverResponse | null)(prev);
          resultRef.current = computed;
          return computed;
        });
      } else {
        resultRef.current = next;
        setResultRaw(next);
      }
    },
    [],
  );

  const solve = useCallback(async (request: SolverRequest) => {
    setLoading(true);
    setError(null);
    setResult(null);  // ref kept in sync by wrapped setter
    setProgress({ iteration: 0, total: request.iterations ?? 300, elapsed: 0, phase: 'Starting...', pct: 0 });
    const start = Date.now();

    // Start elapsed timer for Tauri mode
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      const now = Date.now();
      const total = request.iterations ?? 300;
      const elapsedMs = now - start;
      // Estimate ~70ms per iteration for Tauri, smooth progress
      const estimatedTotal = total * 70;
      const pct = Math.min(95, (elapsedMs / estimatedTotal) * 100);
      const estIter = Math.min(total - 1, Math.floor((pct / 100) * total));
      let phase = 'Building tree...';
      if (pct > 5) phase = `Solving iteration ${estIter}/${total}...`;
      if (pct > 90) phase = 'Finalizing...';
      setProgress({ iteration: estIter, total, elapsed: elapsedMs, phase, pct });
    }, 100);

    try {
      let response: SolverResponse;

      if (isTauri()) {
        const { invoke } = await import('@tauri-apps/api/core');

        if (request.action_path && request.action_path.length > 0) {
          request.history = request.action_path
            .filter(step => step.player !== 'Deal')
            .map(step => step.action.label)
            .join(',');
        }

        if (request.node_locks && typeof request.node_locks !== 'string') {
          request.node_locks = JSON.stringify(request.node_locks);
        }

        response = await invoke<SolverResponse>('solve', { request });
      } else {
        // Clear the generic timer — mock solver handles its own progress
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        response = await mockSolve(request, setProgress);
      }

      setResult(response);  // wrapped setter syncs the ref
      setElapsed(Date.now() - start);
      setProgress({ iteration: response.iterations_run, total: request.iterations ?? 300, elapsed: Date.now() - start, phase: 'Done', pct: 100 });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setProgress(null);
    } finally {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);  // wrapped setter syncs the ref
    setError(null);
    setElapsed(0);
    setProgress(null);
  }, [setResult]);

  /**
   * Route A navigation: try to satisfy the request from `result.strategy_tree`
   * cache instead of invoking the engine. Returns true on a cache hit (UI was
   * updated synchronously). On a miss, returns false — the caller should
   * fall back to `solve(...)`.
   *
   * `history` is the comma-separated PLAYER-action path the engine indexed
   * the cache by. Same string the engine receives via `--history`.
   */
  const navigate = useCallback((history: string): boolean => {
    // Read latest via ref so two clicks in one React tick both see the
    // post-first-click state (avoids stale-closure bug). Use functional
    // setState write so React's batching is respected.
    const current = resultRef.current;
    if (!current?.strategy_tree) return false;
    const entry = current.strategy_tree[history];
    if (!entry) return false;

    const next: SolverResponse = {
      ...current,
      global_strategy: entry.global_strategy,
      combo_strategies: entry.combo_strategies,
      acting_player: entry.acting as 'OOP' | 'IP',
      opponent_side: entry.opponent_side as 'OOP' | 'IP',
      opponent_range: entry.opponent_range,
      // Path B: surface the runout context so the picker UI can render.
      dealt_cards: entry.dealt_cards,
      runout_options: entry.runout_options,
      // EV per grid label for the EV/equity disclosure UI.
      combo_evs: entry.combo_evs,
    };
    setResult(next);  // wrapped setter syncs the ref
    return true;
  }, [setResult]);

  return { result, setResult, loading, error, elapsed, progress, solve, reset, navigate };
}
