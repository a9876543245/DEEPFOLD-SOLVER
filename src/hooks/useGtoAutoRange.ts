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

/** Pick the GTO scenario_type bucket matching a matchup. Cash 6max default. */
function scenarioTypeFor(_matchup: PositionMatchup): string {
  // For now: assume Cash 6max 100bb. Extend when MATCHUPS gains more types.
  return '6max_100bb';
}

/** Find the best chart in `scenarios` matching (game, scenario, position).
 *  Tie-break: prefer charts with no `effective_bb` (= "default stack"). */
function findBestChart(
  scenarios: GtoScenario[],
  game: string,
  scenarioType: string,
  position: string,
): GtoScenario | null {
  const matches = scenarios.filter(s =>
    s.game_type === game &&
    s.scenario_type === scenarioType &&
    s.hero_position === position
  );
  if (matches.length === 0) return null;
  // Prefer charts without a pinned stack (most generic).
  const stackless = matches.filter(s => s.effective_bb == null);
  return (stackless[0] ?? matches[0]);
}

export function useGtoAutoRange(
  matchup: PositionMatchup | null,
  setCustomIpRange: (r: string | null) => void,
  setCustomOopRange: (r: string | null) => void,
) {
  const [scenarios, setScenarios] = useState<GtoScenario[] | null>(null);
  const [applied, setApplied] = useState<AppliedGtoRange[]>([]);
  const lastMatchupKey = useRef<string | null>(null);

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

  // On matchup change, try to auto-apply matching GTO ranges.
  useEffect(() => {
    if (!matchup || !scenarios || !isTauri()) {
      setApplied([]);
      return;
    }
    const key = `${matchup.label}|${matchup.potType}|${matchup.ip}|${matchup.oop}`;
    if (lastMatchupKey.current === key) return;
    lastMatchupKey.current = key;

    const game = 'Cash';
    const scenarioType = scenarioTypeFor(matchup);
    const ipChart  = findBestChart(scenarios, game, scenarioType, chartHeroFor(matchup.ip));
    const oopChart = findBestChart(scenarios, game, scenarioType, chartHeroFor(matchup.oop));

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
  }, [matchup, scenarios, setCustomIpRange, setCustomOopRange]);

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
