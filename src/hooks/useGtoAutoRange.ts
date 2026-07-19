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
import { parseRange, preflopRoles } from '../lib/ranges';
import type { PositionMatchup, Position } from '../lib/ranges';
import type { GameContext } from '../lib/poker';
import type { GtoScenario, GtoChart } from '../components/GtoChartBrowser';

export interface AppliedGtoRange {
  side: 'IP' | 'OOP';
  position: Position;
  scenarioId: string;
  description: string;
}

/** Extract the bundled-folder portion of a chart id ("cash/6max_100bb" /
 *  "mtt/vs_open_3b"). The chart library data update made `scenario_type`
 *  semantic ("RFI", "vs_Open", "vs_3B", ...) and no longer matches the
 *  folder name, so all bucket filtering uses the path prefix instead. */
function chartFolder(s: GtoScenario): string {
  const parts = s.id.split('/');
  return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : s.id;
}

/** For a (potType, role) combination, return scenario_type names ranked
 *  by how well they fit. The first match found in the bundled charts wins.
 *  Multiple candidates are listed because not every (folder × position)
 *  has every scenario type.
 *
 *  Keyed on preflop role, NOT on ip/oop: the opener is OOP in half the
 *  matchups, and asking for the wrong family silently mis-loads a range
 *  (e.g. BTN's *open* in a pot where BTN only called). */
function scenarioCandidates(
  potType: 'SRP' | '3BET',
  role: 'opener' | 'responder',
): string[] {
  // SRP context (single raise + call):
  //   opener   → their range is the open (RFI)
  //   responder → the caller → "vs_Open" facing the open
  // 3BET context (raise + 3-bet + call):
  //   opener   → the original raiser facing a 3-bet → "vs_3B"
  //   responder → the 3-bettor → "vs_Open" with a 3-bet response
  if (potType === 'SRP') {
    return role === 'opener'
      ? ['RFI', 'SB_vs_BB']                           // the raiser
      : ['vs_Open', 'vs_RFI'];                        // the caller
  }
  // 3BET
  return role === 'opener'
    ? ['vs_3B', 'vs_4B', 'vs_4B_allin']               // raiser facing the 3-bet
    : ['vs_Open', 'vs_RFI'];                          // the 3-bettor
}

/** Find the best chart in `scenarios` matching (folder, position, scenario,
 *  effective_bb). Tries the scenario candidates in order; first folder+position
 *  hit with the best stack match wins. */
function findBestChart(
  scenarios: GtoScenario[],
  folderPath: string,                  // e.g. "cash/6max_100bb"
  position: string,                    // hero_position to match
  scenarioRanking: string[],
  effectiveBB: number | null,
): GtoScenario | null {
  const inFolder = scenarios.filter(s =>
    chartFolder(s) === folderPath &&
    s.hero_position === position
  );
  if (inFolder.length === 0) return null;

  const stackPick = (pool: GtoScenario[]): GtoScenario | null => {
    if (pool.length === 0) return null;
    if (effectiveBB != null) {
      const exact = pool.filter(s => s.effective_bb === effectiveBB);
      if (exact.length) return exact[0];
      const pinned = pool.filter(s => s.effective_bb != null);
      if (pinned.length) {
        pinned.sort((a, b) =>
          Math.abs((a.effective_bb ?? 0) - effectiveBB) -
          Math.abs((b.effective_bb ?? 0) - effectiveBB),
        );
        return pinned[0];
      }
    }
    const stackless = pool.filter(s => s.effective_bb == null);
    return (stackless[0] ?? pool[0]);
  };

  // Try each scenario candidate in order; pick from the first that has
  // any chart matching folder+position.
  for (const sc of scenarioRanking) {
    const filtered = inFolder.filter(s => s.scenario_type === sc);
    const hit = stackPick(filtered);
    if (hit) return hit;
  }
  // Last resort: any chart for folder+position regardless of scenario.
  return stackPick(inFolder);
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

    // Folder path = "<game>/<scenarioType>" lowercased game (matches the
    // bundled chart id format like "cash/6max_100bb").
    const folderPath = `${gameContext.gameType.toLowerCase()}/${gameContext.scenarioType}`;
    const potType = matchup.potType;  // "SRP" | "3BET"

    const { opener } = preflopRoles(matchup);
    const roleOf = (p: Position): 'opener' | 'responder' =>
      p === opener ? 'opener' : 'responder';

    const ipChart  = findBestChart(
      scenarios, folderPath, matchup.ip,
      scenarioCandidates(potType, roleOf(matchup.ip)), gameContext.effectiveBB,
    );
    const oopChart = findBestChart(
      scenarios, folderPath, matchup.oop,
      scenarioCandidates(potType, roleOf(matchup.oop)), gameContext.effectiveBB,
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
