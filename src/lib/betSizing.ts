/**
 * Bet sizing menus (2026-10-06): one Pio-style spec — per player and street
 * a bet menu, a raise menu, OOP's donk menu and an all-in switch, plus a raise
 * cap and an all-in threshold — sent to the engine as `--bet-sizing` JSON.
 *
 * The presets are specs too. They reproduce the engine's legacy trees (one
 * pot-fraction list per street for bets AND raises of both players, OOP donks
 * the later streets with the same list), so preset solves are unchanged; and
 * because every solve carries the full spec, a later-street re-solve gets
 * exactly the menus the original tree had on that street.
 *
 * Size syntax (engine parse_size_token): "N%" = N percent of the pot (for a
 * raise: the raise on top of the call, as a share of the pot after calling);
 * "Nx" = raise TO N times the bet faced (raise menus only).
 */

export type StreetKey = 'flop' | 'turn' | 'river';
export const STREETS: StreetKey[] = ['flop', 'turn', 'river'];

export interface StreetMenu {
  bet: string[];
  raise: string[];
  /** OOP only: leading into the previous street's aggressor. Empty = check only. */
  donk: string[];
  allin: boolean;
}

export interface BetSizingSpec {
  oop: Record<StreetKey, StreetMenu>;
  ip: Record<StreetKey, StreetMenu>;
  /** Max raises per street. */
  raise_cap: number;
  /** A bet/raise leaving less than this % of the pot behind once called is an all-in. */
  allin_threshold: number;
}

export type SizingPresetKey = 'lite' | 'standard' | 'polar' | 'small_ball';

/** Pot percentages per street (engine legacy lists). */
export const SIZING_PRESETS: Record<SizingPresetKey, Record<StreetKey, number[]>> = {
  // Single 50% size per street: the smallest sensible menu, ~30-40% faster
  // than Standard with a small EV gap.
  lite:       { flop: [50],      turn: [50],      river: [50] },
  standard:   { flop: [33, 75],  turn: [33, 75],  river: [33, 75] },
  polar:      { flop: [75, 150], turn: [75, 150], river: [75, 150] },
  small_ball: { flop: [25, 33],  turn: [25, 33],  river: [33, 50] },
};

/** Engine limits (types.h kMaxSizesPerMenu, parse_size_token ranges). */
export const MAX_SIZES_PER_MENU = 6;

export function presetSpec(key: SizingPresetKey): BetSizingSpec {
  const p = SIZING_PRESETS[key];
  const menu = (st: StreetKey, donk: boolean): StreetMenu => {
    const sizes = p[st].map(v => `${v}%`);
    return { bet: sizes, raise: [...sizes], donk: donk ? [...sizes] : [], allin: true };
  };
  return {
    // Legacy derivation: OOP leads the root (flop) street with initiative,
    // so its flop donk menu never applies; later streets donk with the list.
    oop: { flop: menu('flop', false), turn: menu('turn', true), river: menu('river', true) },
    ip: { flop: menu('flop', false), turn: menu('turn', false), river: menu('river', false) },
    raise_cap: 3,
    allin_threshold: 12,
  };
}

export function cloneSpec(spec: BetSizingSpec): BetSizingSpec {
  return JSON.parse(JSON.stringify(spec));
}

/** Parse "33 75%, 2.5x" into ["33%", "75%", "2.5x"], or an error message. */
export function parseSizeList(text: string, raise: boolean): { sizes: string[]; error?: string } {
  const sizes: string[] = [];
  for (const raw of text.split(/[\s,;/]+/)) {
    const tok = raw.trim().toLowerCase();
    if (!tok) continue;
    const mult = tok.endsWith('x');
    const num = Number(mult || tok.endsWith('%') ? tok.slice(0, -1) : tok);
    if (!Number.isFinite(num)) return { sizes, error: `"${raw}" is not a size` };
    if (mult) {
      if (!raise) return { sizes, error: `"${raw}": an "x" size only applies to raises` };
      if (!(num > 1 && num <= 100)) return { sizes, error: `"${raw}": a raise multiple must be in (1, 100]` };
      sizes.push(`${num}x`);
    } else {
      if (!(num > 0 && num <= 1000)) return { sizes, error: `"${raw}": a pot percentage must be in (0, 1000]` };
      sizes.push(`${num}%`);
    }
  }
  if (sizes.length > MAX_SIZES_PER_MENU) {
    return { sizes, error: `at most ${MAX_SIZES_PER_MENU} sizes per menu` };
  }
  return { sizes };
}

export function formatSizeList(sizes: string[]): string {
  return sizes.join(' ');
}

/** First problem in the spec, or null. */
export function validateSpec(spec: BetSizingSpec): string | null {
  for (const side of ['oop', 'ip'] as const) {
    for (const st of STREETS) {
      const m = spec[side][st];
      for (const [name, list, raise] of [['bet', m.bet, false], ['raise', m.raise, true], ['donk', m.donk, false]] as const) {
        const { error } = parseSizeList(list.join(' '), raise);
        if (error) return `${side.toUpperCase()} ${st} ${name}: ${error}`;
      }
      if (side === 'ip' && m.donk.length > 0) return `IP ${st}: only OOP has donk bets`;
    }
  }
  if (!Number.isInteger(spec.raise_cap) || spec.raise_cap < 0 || spec.raise_cap > 10) {
    return 'Raise cap must be a whole number from 0 to 10';
  }
  if (!(spec.allin_threshold >= 0 && spec.allin_threshold <= 100)) {
    return 'All-in threshold must be a percentage from 0 to 100';
  }
  return null;
}

/** The engine's `--bet-sizing` JSON. */
export function specToEngineJson(spec: BetSizingSpec): string {
  return JSON.stringify(spec);
}

/** Stable key for "same menus?" comparisons (cache invalidation). */
export function specKey(spec: BetSizingSpec): string {
  return specToEngineJson(spec);
}

/** Short one-line summary, e.g. "Flop 33 75 · Turn 33 75 · River 33 75". */
export function describeSpec(spec: BetSizingSpec): string {
  const label = (st: StreetKey) => {
    const a = spec.oop[st].bet.map(s => s.replace('%', '')).join(' ');
    const b = spec.ip[st].bet.map(s => s.replace('%', '')).join(' ');
    return a === b ? (a || '—') : `${a || '—'} / ${b || '—'}`;
  };
  return `Flop ${label('flop')} · Turn ${label('turn')} · River ${label('river')}`;
}

/** Which preset the spec equals, if any. */
export function matchPreset(spec: BetSizingSpec): SizingPresetKey | null {
  const key = specKey(spec);
  for (const k of Object.keys(SIZING_PRESETS) as SizingPresetKey[]) {
    if (specKey(presetSpec(k)) === key) return k;
  }
  return null;
}
