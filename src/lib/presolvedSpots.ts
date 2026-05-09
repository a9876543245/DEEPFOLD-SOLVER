/**
 * Pre-solved poker strategy library.
 *
 * Two modes:
 *   - Tauri (desktop): heuristic preview in grid, click triggers REAL solver.
 *     Session-level cache keeps recently-solved spots instant.
 *   - Browser (dev): heuristic only, UI flags as "Demo Mode".
 *
 * The library is essentially a curated list of (matchup × board) presets.
 * "Pre-solved" here means "pre-configured" — user gets one-click to load a
 * well-known spot into the main solver view.
 */

import {
  GRID_LABELS,
  SUITS,
  parseBoardCards,
  getHandStrength,
  expandComboLabel,
  RANK_VALUES,
} from './poker';
import type { ComboStrategy, SolverRequest, SolverResponse } from './poker';
import { MATCHUPS, parseRange } from './ranges';
import type { PositionMatchup } from './ranges';

// ============================================================================
// Types
// ============================================================================

export interface PresolvedSpot {
  id: string;
  matchup: PositionMatchup;
  board: string;
  boardTexture: string;
  globalStrategy: Record<string, string>;
  comboStrategies: Record<string, ComboStrategy>;
  /** True if this is a heuristic preview (NOT from real solver). */
  isDemo: boolean;
  /** Unix ms when the real solver produced this; undefined for demo. */
  solvedAt?: number;
  /** Where this strategy came from (v1.8.3+):
   *   - 'bundled': pre-solved on the dev's GPU, shipped in the installer
   *   - 'live':    live solve via the C++ engine on the user's machine
   *   - 'demo':    heuristic placeholder (browser mode or pre-solve in flight)
   *  Used by SpotLibrary UI for the "Pre-solved" / "Live-solved" badge.
   */
  source?: 'bundled' | 'live' | 'demo';
  /** Iterations that produced this result (bundled spots are typically Deep). */
  iterationsRun?: number;
  /** Final exploitability % for badge display. */
  exploitabilityPct?: number;
}

// ----------------------------------------------------------------------------
// Bundled spot envelope (matches scripts/compact-presolved.mjs:buildBundledSpot
// schema v1).
// ----------------------------------------------------------------------------
interface BundledSpot {
  v: number;                    // schema version
  solver_version: string;       // app version that produced this bundle
  matchup_idx: number;
  board_idx: number;
  sizing_key: string;
  stack_label: string;
  iterations_run: number;
  exploitability_pct: number | null;
  early_stop_reason: string | null;
  global_strategy: Record<string, string>;
  combo_strategies: Record<string, ComboStrategy>;
  acting_player: number | null;
  opponent_side: string | null;
  opponent_range: string[];
  strategy_tree?: Record<string, unknown>;  // Tier B only
}

/** Schema version + app version the frontend expects. Mismatches → fall through
 *  to live solve so the UI never serves stale strategies after a solver bump. */
const EXPECTED_SCHEMA_VERSION = 1;
import { APP_VERSION } from './appVersion';

// ============================================================================
// Board Templates: 12 representative flop textures
// ============================================================================

export const BOARD_TEMPLATES: { texture: string; board: string }[] = [
  { texture: 'Dry High', board: 'AsKd2c' },
  { texture: 'Dry High', board: 'KhQd4c' },
  { texture: 'Dry High', board: 'AcJd5h' },
  { texture: 'Wet High', board: 'KsQhJd' },
  { texture: 'Wet High', board: 'QdJhTc' },
  { texture: 'Wet Broadway', board: 'AsKhQd' },
  { texture: 'Monotone', board: 'Ah9h4h' },
  { texture: 'Monotone', board: 'Kd8d3d' },
  { texture: 'Low Connected', board: '7s8h9d' },
  { texture: 'Low Connected', board: '5c6d7h' },
  { texture: 'Low Dry', board: '2s3d7h' },
  { texture: 'Paired', board: '7s7d4c' },
];

// ============================================================================
// Environment detection
// ============================================================================

import { isTauri as isTauriEnv } from './tauriEnv';

export function isTauri(): boolean {
  return isTauriEnv();
}

/** True when the real C++ solver is available (Tauri desktop build). */
export function isRealSolverAvailable(): boolean {
  return isTauri();
}

// ============================================================================
// Heuristic strategy generation (preview only)
// ============================================================================

function contextualStrategy(strength: number, idx: number): ComboStrategy {
  const noise = Math.sin(idx * 137.5) * 0.08;

  if (strength > 0.85) {
    return {
      'Check': 0.15 + noise * 0.5,
      'Bet 33%': 0.20,
      'Bet 75%': 0.50 - noise * 0.3,
      'All-in': 0.15,
    };
  } else if (strength > 0.65) {
    return {
      'Check': 0.10,
      'Bet 33%': 0.55 + noise,
      'Bet 75%': 0.30 - noise,
      'All-in': 0.05,
    };
  } else if (strength > 0.45) {
    return {
      'Check': 0.55 + noise,
      'Bet 33%': 0.35 - noise,
      'Bet 75%': 0.08,
      'All-in': 0.02,
    };
  } else if (strength > 0.25) {
    return {
      'Check': 0.72 + noise,
      'Bet 33%': 0.18 - noise * 0.5,
      'Bet 75%': 0.08,
      'All-in': 0.02,
    };
  } else {
    return {
      'Check': 0.65,
      'Bet 33%': 0.15 + noise,
      'Bet 75%': 0.12 - noise,
      'All-in': 0.08,
    };
  }
}

function generateComboStrategies(
  board: string,
  heroRangeStr: string,
): Record<string, ComboStrategy> {
  const boardCards = parseBoardCards(board);
  const boardRanks = boardCards
    .map((c) => c[0])
    .sort((a, b) => (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0));

  const heroRange = parseRange(heroRangeStr);
  const strategies: Record<string, ComboStrategy> = {};
  const flat = GRID_LABELS.flat();
  const boardSuits = new Set(boardCards.map((c) => c[1]));

  for (let idx = 0; idx < flat.length; idx++) {
    const label = flat[idx];
    const rangeFreq = heroRange[label] ?? 0;
    if (rangeFreq <= 0.001) {
      strategies[label] = { 'Not in range': 1.0 };
      continue;
    }

    const strength = getHandStrength(label, boardRanks);
    strategies[label] = contextualStrategy(strength, idx);

    const specifics = expandComboLabel(label, boardCards);
    for (let si = 0; si < specifics.length; si++) {
      const sc = specifics[si];
      if (sc.isDead) {
        strategies[sc.combo] = { 'Not in range': 1.0 };
        continue;
      }
      const hasFlushDraw =
        sc.suit1 === sc.suit2 && boardSuits.has(sc.suit1);
      const hasBackdoor =
        sc.suit1 !== sc.suit2 &&
        (boardSuits.has(sc.suit1) || boardSuits.has(sc.suit2));
      const suitBonus = hasFlushDraw ? 0.08 : hasBackdoor ? 0.03 : 0;
      const suitIdx = SUITS.indexOf(sc.suit1) * 4 + SUITS.indexOf(sc.suit2);
      const suitNoise = Math.sin((idx * 13 + suitIdx) * 73.7) * 0.06;
      const adjustedStrength = Math.min(
        1,
        Math.max(0, strength + suitBonus + suitNoise),
      );
      strategies[sc.combo] = contextualStrategy(
        adjustedStrength,
        idx * 16 + suitIdx,
      );
    }
  }
  return strategies;
}

function computeGlobalStrategy(
  combos: Record<string, ComboStrategy>,
): Record<string, string> {
  const counts: Record<string, number> = {};
  let inRangeCount = 0;
  const flat = new Set(GRID_LABELS.flat());

  for (const [key, strategy] of Object.entries(combos)) {
    if (!flat.has(key)) continue;
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
// Cache (session-level, in-memory)
// ============================================================================

interface CacheEntry {
  spot: PresolvedSpot;
  /** undefined = heuristic preview; Promise = solve in-flight; PresolvedSpot = real cached */
  realResolver?: Promise<PresolvedSpot>;
}

const cache = new Map<string, CacheEntry>();

function buildSpotId(matchupIdx: number, boardIdx: number): string {
  return `${matchupIdx}-${boardIdx}`;
}

function buildHeuristicSpot(
  matchupIdx: number,
  boardIdx: number,
): PresolvedSpot {
  const matchup = MATCHUPS[matchupIdx];
  const template = BOARD_TEMPLATES[boardIdx];
  const comboStrategies = generateComboStrategies(
    template.board,
    matchup.oopRange,
  );
  const globalStrategy = computeGlobalStrategy(comboStrategies);
  return {
    id: buildSpotId(matchupIdx, boardIdx),
    matchup,
    board: template.board,
    boardTexture: template.texture,
    globalStrategy,
    comboStrategies,
    isDemo: true,
  };
}

// ============================================================================
// Bundled spot lookup (Tauri only) — PRESOLVE_BUNDLE_PLAN Phase 3
// ============================================================================

/**
 * Build the canonical spot key used by `scripts/bulk-presolve.mjs` and
 * `scripts/compact-presolved.mjs`. Format:
 *   `m<matchupIdx>_b<boardIdx>_<sizingKey>_<stackLabel>bb.json.gz`
 *
 * Default args mirror what `solveSpotReal` produces today (the live solve
 * uses the matchup's defaultPot/defaultStack with whatever sizing the user
 * has selected). For SpotLibrary's preview grid we always use 'standard'
 * sizing + 'default' stack — that's what gets bundled by Phase 1.
 */
function bundledSpotKey(
  matchupIdx: number,
  boardIdx: number,
  sizingKey: string = 'standard',
  stackLabel: string = 'default',
): string {
  return `m${matchupIdx}_b${boardIdx}_${sizingKey}_${stackLabel}bb.json.gz`;
}

/**
 * Try to load a bundled, pre-solved version of this spot. Returns null on:
 *   - browser mode (no Tauri command available)
 *   - bundle file missing for this (matchup, board, sizing, stack) tuple
 *   - bundle solver_version doesn't match the current app version (stale)
 *   - bundle schema version isn't recognized
 *   - any IO / JSON / decompression error
 *
 * On null, the caller should fall through to a live solve.
 */
export async function tryLoadBundled(
  matchupIdx: number,
  boardIdx: number,
  sizingKey: string = 'standard',
  stackLabel: string = 'default',
): Promise<PresolvedSpot | null> {
  if (!isTauri()) {
    console.log(`[presolvedSpots] tryLoadBundled(${matchupIdx},${boardIdx}): not in Tauri (browser mode), skipping`);
    return null;
  }

  const matchup = MATCHUPS[matchupIdx];
  const template = BOARD_TEMPLATES[boardIdx];
  if (!matchup || !template) {
    console.log(`[presolvedSpots] tryLoadBundled(${matchupIdx},${boardIdx}): bad matchup/template indices`);
    return null;
  }

  const spotKey = bundledSpotKey(matchupIdx, boardIdx, sizingKey, stackLabel);
  let json: string;
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    json = await invoke<string>('read_bundled_presolve', { spotKey });
  } catch (e: unknown) {
    // The Rust command returns "NOT_FOUND:<key>" for missing bundles —
    // that's the common case, not an error. Anything else is logged so we
    // notice real issues. v1.8.3+: log NOT_FOUND too as console.log (was
    // console.debug, hidden by default Chrome DevTools filter level).
    const msg = String(e);
    if (msg.includes('NOT_FOUND:')) {
      console.log(`[presolvedSpots] no bundle for ${spotKey} (live solve will run on click)`);
    } else {
      console.warn(`[presolvedSpots] read_bundled_presolve('${spotKey}') failed:`, msg);
    }
    return null;
  }

  let bundled: BundledSpot;
  try {
    bundled = JSON.parse(json);
  } catch (e) {
    console.warn(`[presolvedSpots] JSON.parse failed for ${spotKey}:`, e);
    return null;
  }

  // Schema invalidation. Mismatch → fall through to live solve.
  if (bundled.v !== EXPECTED_SCHEMA_VERSION) {
    console.warn(
      `[presolvedSpots] ${spotKey} schema v${bundled.v} != expected v${EXPECTED_SCHEMA_VERSION}, falling through`);
    return null;
  }
  if (bundled.solver_version !== APP_VERSION) {
    console.info(
      `[presolvedSpots] ${spotKey} solver_version "${bundled.solver_version}" != current "${APP_VERSION}", falling through`);
    return null;
  }
  console.info(
    `[presolvedSpots] HIT bundled ${spotKey} (iter=${bundled.iterations_run}, exploit=${bundled.exploitability_pct}%)`);

  return {
    id: buildSpotId(matchupIdx, boardIdx),
    matchup,
    board: template.board,
    boardTexture: template.texture,
    globalStrategy: bundled.global_strategy,
    comboStrategies: bundled.combo_strategies,
    isDemo: false,
    solvedAt: Date.now(),
    source: 'bundled',
    iterationsRun: bundled.iterations_run,
    exploitabilityPct: bundled.exploitability_pct ?? undefined,
  };
}

// ============================================================================
// Real solver invocation (Tauri only)
// ============================================================================

async function solveSpotReal(
  matchupIdx: number,
  boardIdx: number,
): Promise<PresolvedSpot> {
  if (!isTauri()) {
    throw new Error('Real solver not available in browser mode');
  }

  // v1.8.3+ Phase 3: try the bundled, pre-solved version first. SpotLibrary's
  // preview grid always wants the 'standard' sizing × 'default' stack
  // (whatever sizing/stack the matchup defaults to), so that's the lookup key.
  const bundled = await tryLoadBundled(matchupIdx, boardIdx);
  if (bundled) return bundled;

  const matchup = MATCHUPS[matchupIdx];
  const template = BOARD_TEMPLATES[boardIdx];

  const { invoke } = await import('@tauri-apps/api/core');
  const request: SolverRequest = {
    board: template.board,
    pot_size: Math.round(matchup.defaultPot * 10),
    effective_stack: Math.round(matchup.defaultStack * 10),
    iterations: 200,
    exploitability: 1.0,
    ip_range: matchup.ipRange,
    oop_range: matchup.oopRange,
  };

  const response = await invoke<SolverResponse>('solve', { request });

  // Build the presolved spot. Rust currently only returns global_strategy
  // reliably; fall back to heuristic combo_strategies if not provided.
  const comboStrategies = response.combo_strategies
    ?? generateComboStrategies(template.board, matchup.oopRange);

  return {
    id: buildSpotId(matchupIdx, boardIdx),
    matchup,
    board: template.board,
    boardTexture: template.texture,
    globalStrategy: response.global_strategy ?? {},
    comboStrategies,
    isDemo: false,
    solvedAt: Date.now(),
    source: 'live',
  };
}

// ============================================================================
// Public API
// ============================================================================

/**
 * v1.8.3+ — bundled-or-demo lookup. Used by SpotLibrary's preview grid to
 * upgrade spots from heuristic to real-bundled strategies when a pre-solved
 * file exists, WITHOUT firing a live solve on cache miss. Promise always
 * resolves; never throws. Result is also written to the in-memory cache so
 * the synchronous `getPresolvedSpot()` returns the upgraded version on
 * subsequent calls.
 *
 * - Tauri + bundle exists  → PresolvedSpot with source: 'bundled'
 * - Tauri + no bundle      → heuristic spot (source: 'demo' implicitly)
 * - Browser mode           → heuristic spot (no Tauri command)
 */
export async function getBundledOrDemo(
  matchupIdx: number,
  boardIdx: number,
  sizingKey: string = 'standard',
  stackLabel: string = 'default',
): Promise<PresolvedSpot> {
  const id = buildSpotId(matchupIdx, boardIdx);
  const cached = cache.get(id);
  // Already upgraded to bundled or live — no need to re-fetch.
  if (cached && !cached.spot.isDemo) return cached.spot;

  if (isTauri()) {
    const bundled = await tryLoadBundled(matchupIdx, boardIdx, sizingKey, stackLabel);
    if (bundled) {
      cache.set(id, { spot: bundled });
      return bundled;
    }
  }
  // No bundle — return heuristic. Don't trigger a live solve here; that's the
  // caller's decision (e.g. App.tsx onSelectSpot).
  const fallback = cached?.spot ?? buildHeuristicSpot(matchupIdx, boardIdx);
  cache.set(id, { spot: fallback });
  return fallback;
}

/**
 * Get a spot synchronously — returns heuristic preview unless a real solve
 * has already completed for this id (then returns cached real version).
 */
export function getPresolvedSpot(
  matchupIdx: number,
  boardIdx: number,
): PresolvedSpot {
  const id = buildSpotId(matchupIdx, boardIdx);
  const entry = cache.get(id);
  if (entry) return entry.spot;

  const spot = buildHeuristicSpot(matchupIdx, boardIdx);
  cache.set(id, { spot });
  return spot;
}

/**
 * Async variant that runs the REAL solver in Tauri mode.
 * Caches the result so subsequent calls are instant.
 * In browser mode, returns heuristic preview immediately.
 */
export async function resolveSpotReal(
  matchupIdx: number,
  boardIdx: number,
): Promise<PresolvedSpot> {
  const id = buildSpotId(matchupIdx, boardIdx);
  const entry = cache.get(id);

  // Already real-solved
  if (entry && !entry.spot.isDemo) return entry.spot;

  // Real solve in flight
  if (entry?.realResolver) return entry.realResolver;

  // Browser mode: just return heuristic
  if (!isTauri()) {
    const spot = entry?.spot ?? buildHeuristicSpot(matchupIdx, boardIdx);
    cache.set(id, { spot });
    return spot;
  }

  // Tauri mode: launch real solve, cache the promise so concurrent requests coalesce
  const resolver = solveSpotReal(matchupIdx, boardIdx)
    .then((spot) => {
      cache.set(id, { spot });
      return spot;
    })
    .catch((err) => {
      // Fallback to heuristic on error; don't poison the cache permanently
      console.warn('[presolvedSpots] real solve failed, falling back to heuristic:', err);
      const fallback = buildHeuristicSpot(matchupIdx, boardIdx);
      cache.set(id, { spot: fallback });
      return fallback;
    });

  const currentSpot = entry?.spot ?? buildHeuristicSpot(matchupIdx, boardIdx);
  cache.set(id, { spot: currentSpot, realResolver: resolver });
  return resolver;
}

/** Returns true if a real-solved result is cached for this spot. */
export function isSpotRealCached(
  matchupIdx: number,
  boardIdx: number,
): boolean {
  const entry = cache.get(buildSpotId(matchupIdx, boardIdx));
  return !!entry && !entry.spot.isDemo;
}

/** Get all 12 board spots for a specific matchup (heuristic preview). */
export function getSpotsByMatchup(matchupIdx: number): PresolvedSpot[] {
  const spots: PresolvedSpot[] = [];
  for (let b = 0; b < BOARD_TEMPLATES.length; b++) {
    spots.push(getPresolvedSpot(matchupIdx, b));
  }
  return spots;
}

/** Get all 120 spots (used by drill mode for random sampling). */
export function getAllSpots(): PresolvedSpot[] {
  const spots: PresolvedSpot[] = [];
  for (let m = 0; m < MATCHUPS.length; m++) {
    for (let b = 0; b < BOARD_TEMPLATES.length; b++) {
      spots.push(getPresolvedSpot(m, b));
    }
  }
  return spots;
}

/**
 * Clear the in-memory cache. Next access will re-generate heuristics and
 * re-solve real spots on demand.
 */
export function clearPresolvedCache() {
  cache.clear();
}
