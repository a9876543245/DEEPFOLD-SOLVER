/**
 * useGtoAutoRange — auto-load a GTO preflop chart matching the current
 * position matchup, and apply its range as the IP / OOP default.
 *
 * Mapping logic: each PositionMatchup has positions like 'BTN', 'BB', etc.
 * We look up matching charts in the bundled `gto_output/` library and pick
 * the first hit. The mapping is necessarily heuristic — chart filenames
 * carry spreadsheet cell refs, not full semantic metadata, so we trust the
 * user to refine via the GTO Chart Browser if needed.
 *
 * Behavior:
 *   - On mount: fetch full scenario list once (cached for the session).
 *   - On matchup change: find best match for IP + OOP, apply ranges, set
 *     a status object the UI can render to disclose what was auto-loaded.
 *   - If no match found: fall back to the matchup's hardcoded `ipRange` /
 *     `oopRange` (caller's setCustomIpRange/OopRange is left as null).
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { isTauri } from '../lib/tauriEnv';
import { parseRange } from '../lib/ranges';
import type { PositionMatchup, Position } from '../lib/ranges';
import type { GameContext } from '../lib/poker';
import type { GtoScenario, GtoChart } from '../components/GtoChartBrowser';

export interface AppliedGtoRange {
  side: 'IP' | 'OOP';
  position: Position;
  scenarioId: string;
  description: string;
}

/** Map our MATCHUPS positions to chart hero_position labels. They differ
 *  for one role: our "MP" is the same seat as the dataset's "HJ" in
 *  6-handed games. */
function chartHeroFor(pos: Position): string {
  return pos === 'MP' ? 'HJ' : pos;
}

/** Find the best chart in `scenarios` matching (game, scenario, position,
 *  effective_bb). Effective stack is honored when set; otherwise charts
 *  without a pinned stack are preferred (= the "generic" entry). */
function findBestChart(
  scenarios: GtoScenario[],
  game: string,
  scenarioType: string,
  position: string,
  effectiveBB: number | null,
): GtoScenario | null {
  const base = scenarios.filter(s =>
    s.game_type === game &&
    s.scenario_type === scenarioType &&
    s.hero_position === position
  );
  if (base.length === 0) return null;

  // Stack filter: exact match wins; if user picked a stack but no chart
  // pins it, fall back to the closest pinned value (smallest |Δ|).
  if (effectiveBB != null) {
    const exact = base.filter(s => s.effective_bb === effectiveBB);
    if (exact.length) return exact[0];
    const pinned = base.filter(s => s.effective_bb != null);
    if (pinned.length) {
      pinned.sort((a, b) =>
        Math.abs((a.effective_bb ?? 0) - effectiveBB) -
        Math.abs((b.effective_bb ?? 0) - effectiveBB),
      );
      return pinned[0];
    }
    // None of them pin a stack — fall through to the generic entry below.
  }

  // No stack preference (or no pinned chart matched): prefer stackless,
  // else any.
  const stackless = base.filter(s => s.effective_bb == null);
  return (stackless[0] ?? base[0]);
}

export function useGtoAutoRange(
  matchup: PositionMatchup | null,
  gameContext: GameContext,
  setCustomIpRange: (r: string | null) => void,
  setCustomOopRange: (r: string | null) => void,
) {
  const [scenarios, setScenarios] = useState<GtoScenario[] | null>(null);
  const [applied, setApplied] = useState<AppliedGtoRange[]>([]);
  const lastKey = useRef<string | null>(null);

  // One-shot scenario list fetch on mount (Tauri only).
  useEffect(() => {
    if (!isTauri()) return;
    (async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const list = await invoke<GtoScenario[]>('list_gto_scenarios');
        setScenarios(list);
      } catch {
        // Silent — auto-apply just becomes a no-op if list fails.
      }
    })();
  }, []);

  // On matchup OR game-context change, try to auto-apply matching GTO ranges.
  useEffect(() => {
    if (!matchup || !scenarios || !isTauri()) {
      setApplied([]);
      return;
    }
    // Cache key includes BOTH matchup AND game context so flipping the
    // game type (e.g. Cash → MTT) re-runs the lookup even if positions
    // didn't change.
    const key = `${matchup.label}|${matchup.potType}|${matchup.ip}|${matchup.oop}` +
                `||${gameContext.gameType}|${gameContext.scenarioType}|${gameContext.effectiveBB ?? 'any'}`;
    if (lastKey.current === key) return;
    lastKey.current = key;

    const ipChart  = findBestChart(
      scenarios, gameContext.gameType, gameContext.scenarioType,
      chartHeroFor(matchup.ip), gameContext.effectiveBB,
    );
    const oopChart = findBestChart(
      scenarios, gameContext.gameType, gameContext.scenarioType,
      chartHeroFor(matchup.oop), gameContext.effectiveBB,
    );

    const newApplied: AppliedGtoRange[] = [];

    (async () => {
      const { invoke } = await import('@tauri-apps/api/core');
      // Fetch + apply IP chart.
      if (ipChart) {
        try {
          const chart = await invoke<GtoChart>('load_gto_chart', { id: ipChart.id });
          const rangeStr = chartToInRange(chart);
          if (rangeStr) {
            setCustomIpRange(rangeStr);
            newApplied.push({
              side: 'IP', position: matchup.ip,
              scenarioId: chart.id, description: chart.context.description,
            });
          }
        } catch {/* skip */}
      } else {
        // No GTO match — keep matchup's hardcoded default by setting null.
        setCustomIpRange(null);
      }

      if (oopChart) {
        try {
          const chart = await invoke<GtoChart>('load_gto_chart', { id: oopChart.id });
          const rangeStr = chartToInRange(chart);
          if (rangeStr) {
            setCustomOopRange(rangeStr);
            newApplied.push({
              side: 'OOP', position: matchup.oop,
              scenarioId: chart.id, description: chart.context.description,
            });
          }
        } catch {/* skip */}
      } else {
        setCustomOopRange(null);
      }

      setApplied(newApplied);
    })();
  }, [matchup, gameContext, scenarios, setCustomIpRange, setCustomOopRange]);

  // Manual override: clear the auto-applied disclosure when user edits a
  // range. Caller passes their custom range setter directly; this is just
  // the disclosure-clearing hook.
  const clearApplied = useCallback((side: 'IP' | 'OOP') => {
    setApplied(prev => prev.filter(a => a.side !== side));
  }, []);

  return { applied, clearApplied };
}

/** Combine all non-fold actions in a chart into a single in-range string.
 *  Mirrors the helper in GtoChartBrowser but available standalone here. */
function chartToInRange(chart: GtoChart): string {
  const merged: Record<string, number> = {};
  for (const action of chart.actions) {
    if (/fold/i.test(action)) continue;
    const parsed = parseRange(chart.ranges[action] ?? '');
    for (const [combo, f] of Object.entries(parsed)) {
      merged[combo] = (merged[combo] ?? 0) + f;
    }
  }
  return Object.entries(merged)
    .filter(([, f]) => f > 0.001)
    .map(([combo, f]) => `${combo}:${Math.min(1, f).toFixed(3)}`)
    .join(',');
}
