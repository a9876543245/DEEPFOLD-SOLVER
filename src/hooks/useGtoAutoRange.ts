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

/** The heads-up preflop line of a chart, e.g. ["BTN", "2.5bb", "BB"]. */
function lineTokens(s: GtoScenario): string[] {
  return (s.preflop_line ?? '').trim().split(/\s+/).filter(Boolean);
}

/** Find the chart in `scenarios` for (folder, scenario type, hero) whose
 *  preflop line passes `lineOk`, with the closest stack depth.
 *
 *  2026-10-07 audit: the line decides. Matching on the hero's seat alone
 *  picked the first chart of the folder — BB's defence vs an UTG open for a
 *  BTN-vs-BB pot, or a multiway line ("CO 2.5bb BTN Call BB"). */
function findChart(
  scenarios: GtoScenario[],
  folderPath: string,                  // e.g. "cash/6max_100bb"
  scenarioType: string,
  hero: string,
  lineOk: (tokens: string[]) => boolean,
  effectiveBB: number | null,
): GtoScenario | null {
  const pool = scenarios.filter(s =>
    chartFolder(s) === folderPath &&
    s.scenario_type === scenarioType &&
    s.hero_position === hero &&
    lineOk(lineTokens(s)),
  );
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
  return pool.find(s => s.effective_bb == null) ?? pool[0];
}

/** Per-label weights of the chart's actions that pass `pick` (summed). The
 *  charts hold each hand's frequencies AT this decision (they sum to 1 over
 *  the actions for every hand that gets there). */
function actionWeights(chart: GtoChart, pick: (action: string) => boolean): Record<string, number> {
  const out: Record<string, number> = {};
  for (const action of chart.actions) {
    if (!pick(action)) continue;
    for (const [label, f] of Object.entries(parseRange(chart.ranges[action] ?? ''))) {
      out[label] = (out[label] ?? 0) + f;
    }
  }
  return out;
}

const isRaise = (a: string) => /^raise/i.test(a);      // opens, 3-bets (not all-ins)
const isCall = (a: string) => /^call$/i.test(a);

function toRangeString(weights: Record<string, number>): string {
  return Object.entries(weights)
    .filter(([, f]) => f > 0.001)
    .map(([label, f]) => `${label}:${Math.min(1, f).toFixed(3)}`)
    .join(',');
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

    const { opener, responder } = preflopRoles(matchup);
    const bb = gameContext.effectiveBB;
    // Heads-up lines: "<opener>" / "<opener> <open> <responder>" /
    // "<opener> <open> <responder> <3-bet> <opener>".
    const rfi = findChart(scenarios, folderPath, 'RFI', opener,
      t => t.length === 1 && t[0] === opener, bb);
    const vsOpen = findChart(scenarios, folderPath, 'vs_Open', responder,
      t => t.length === 3 && t[0] === opener && t[2] === responder, bb);
    const vs3b = potType === '3BET'
      ? findChart(scenarios, folderPath, 'vs_3B', opener,
          t => t.length === 5 && t[0] === opener && t[2] === responder && t[4] === opener, bb)
      : null;

    (async () => {
      const { invoke } = await import('@tauri-apps/api/core');
      const load = async (sc: GtoScenario | null) => {
        if (!sc) return null;
        try {
          return await invoke<GtoChart>('load_gto_chart', { id: sc.id });
        } catch {
          return null;
        }
      };
      const [rfiChart, vsOpenChart, vs3bChart] = await Promise.all([load(rfi), load(vsOpen), load(vs3b)]);

      // Each player's flop range: their own action frequencies along the
      // preflop line, multiplied. SRP: the open; the call of it. 3BP: the
      // 3-bet; the open times the call of the 3-bet. (All non-fold actions
      // used to be merged: AA and the other 3-bets sat in BB's flatting
      // range of a single-raised pot.)
      let openerRange: string | null = null;
      let responderRange: string | null = null;
      let openerChart: GtoChart | null = null;
      let responderChart: GtoChart | null = null;
      if (potType === 'SRP') {
        if (rfiChart) { openerRange = toRangeString(actionWeights(rfiChart, isRaise)); openerChart = rfiChart; }
        if (vsOpenChart) { responderRange = toRangeString(actionWeights(vsOpenChart, isCall)); responderChart = vsOpenChart; }
      } else {
        if (vsOpenChart) { responderRange = toRangeString(actionWeights(vsOpenChart, isRaise)); responderChart = vsOpenChart; }
        if (rfiChart && vs3bChart) {
          const open = actionWeights(rfiChart, isRaise);
          const call = actionWeights(vs3bChart, isCall);
          const joint: Record<string, number> = {};
          for (const [label, f] of Object.entries(call)) joint[label] = f * (open[label] ?? 0);
          openerRange = toRangeString(joint);
          openerChart = vs3bChart;
        }
      }

      const newApplied: AppliedGtoRange[] = [];
      const apply = (pos: Position, range: string | null, chart: GtoChart | null) => {
        const side: 'IP' | 'OOP' = pos === matchup.ip ? 'IP' : 'OOP';
        const set = side === 'IP' ? setCustomIpRange : setCustomOopRange;
        if (range && chart) {
          set(range);
          newApplied.push({ side, position: pos, scenarioId: chart.id, description: chart.context.description });
        } else {
          // No chart for this line — keep the matchup's hardcoded default.
          set(null);
        }
      };
      apply(opener, openerRange, openerChart);
      apply(responder, responderRange, responderChart);
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
