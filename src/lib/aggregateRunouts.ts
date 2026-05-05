/**
 * Aggregated Report Lite — Day 1
 *
 * Pure functions for extracting per-turn-card aggregations from a flop
 * solve's `strategy_tree`. Lets the UI render a 13×4 grid of "what
 * happens on each turn card" without re-invoking the solver.
 *
 * See AGGREGATED_REPORT_DESIGN.md for the data model + caveats. The TL;DR:
 *
 * - Lex-min canonical turn shares the no-suffix path (e.g. `Check,Bet_33,Call`),
 *   other canonical reps get `#<card>` suffix keys.
 * - Rainbow flops fall back to single-child chance — `aggregateTurns` returns
 *   an empty array and the modal shows "no runout data" state.
 * - Per-turn EV is approximated as `mean(combo_evs.values())`. Range-weighted
 *   EV needs the acting player's reach which the engine doesn't currently
 *   expose; tracked as a Day 2+ improvement.
 */

import type { StrategyTreeEntry } from './poker';

/** One turn card's aggregated info. Keyed for the 13×4 grid render. */
export interface TurnAggregate {
  /** Card token e.g. "2s", "Th". */
  card: string;
  /** Rank index 0..12 (2 = 0, A = 12). */
  rankIdx: number;
  /** Suit char ('c' | 'd' | 'h' | 's'). */
  suit: string;
  /** Who's acting on this turn ("OOP" | "IP"). */
  acting: string;
  /** Action label → "%"-formatted frequency, e.g. {"Check":"40.0%","Bet_75":"60.0%"}. */
  strategy: Record<string, string>;
  /** Mean of `combo_evs` values, in chips. NaN-safe (returns 0 if empty). */
  meanEv: number;
  /** Iso orbit size — how many real cards this canonical rep stands in for. */
  weight: number;
}

const RANK_CHARS = '23456789TJQKA';

/** Returns rank index 0..12 for "2","3",..."A". -1 if invalid. */
function rankIndex(rankCh: string): number {
  return RANK_CHARS.indexOf(rankCh);
}

/** Parse "2s" into {rank:0, suit:'s'}. Returns nulls if invalid. */
function parseCard(token: string): { rankIdx: number; suit: string } | null {
  if (!token || token.length < 2) return null;
  const r = rankIndex(token[0]);
  const s = token[1].toLowerCase();
  if (r < 0 || !'cdhs'.includes(s)) return null;
  return { rankIdx: r, suit: s };
}

/** Look up the orbit weight for a given turn card from runout_options.
 *  runout_options uses the same card token format ("2s", "Th", etc.). */
function lookupWeight(
  options: Array<{ card: string; weight: number }> | undefined,
  card: string,
): number | null {
  if (!options) return null;
  const hit = options.find((o) => o.card === card);
  return hit ? hit.weight : null;
}

/** Mean of EV values. Returns 0 for empty input (so display shows 0 rather
 *  than NaN). Caller can detect "no EVs available" by checking the source
 *  combo_evs object size before calling. */
function meanEv(combo_evs: Record<string, number> | undefined): number {
  if (!combo_evs) return 0;
  const vs = Object.values(combo_evs);
  if (!vs.length) return 0;
  let s = 0;
  for (const v of vs) s += v;
  return s / vs.length;
}

/**
 * Walk a strategy_tree to collect every IMMEDIATE post-chance entry under
 * `flopHistory`. Returns one aggregate per distinct turn card.
 *
 * Path matching:
 *   - `key === flopHistory`             → lex-min canonical (no suffix)
 *   - `key === flopHistory + '#' + cardTok`
 *                                       → other canonical rep
 *   - any deeper key (extra "," or "#") is a descendant — skip.
 *
 * Sort: by suit (c, d, h, s), then by rank ascending.
 */
export function aggregateTurns(
  tree: Record<string, StrategyTreeEntry> | undefined,
  flopHistory: string,
): TurnAggregate[] {
  if (!tree) return [];
  const out: TurnAggregate[] = [];

  for (const [key, entry] of Object.entries(tree)) {
    if (!entry || !entry.dealt_cards || entry.dealt_cards.length === 0) continue;

    const isLexMin = key === flopHistory;
    let isHashSuffix = false;
    if (!isLexMin && key.startsWith(flopHistory + '#')) {
      const tail = key.slice(flopHistory.length + 1);
      // Tail must be exactly one card token, no further commas/hashes
      isHashSuffix = !/[,#]/.test(tail) && tail.length >= 2;
    }
    if (!isLexMin && !isHashSuffix) continue;

    const turnCard = entry.dealt_cards[entry.dealt_cards.length - 1];
    const parsed = parseCard(turnCard);
    if (!parsed) continue;

    out.push({
      card: turnCard,
      rankIdx: parsed.rankIdx,
      suit: parsed.suit,
      acting: entry.acting,
      strategy: entry.global_strategy ?? {},
      meanEv: meanEv(entry.combo_evs),
      weight: lookupWeight(entry.runout_options, turnCard) ?? 1,
    });
  }

  out.sort((a, b) => {
    const sd = 'cdhs'.indexOf(a.suit) - 'cdhs'.indexOf(b.suit);
    if (sd !== 0) return sd;
    return a.rankIdx - b.rankIdx;
  });
  return out;
}

/** {label, freqPct} for the highest-frequency action. Returns null if
 *  strategy is empty or unparseable. Used to colour-code the grid cells. */
export function dominantAction(
  strategy: Record<string, string> | undefined,
): { label: string; freq: number } | null {
  if (!strategy) return null;
  let bestLabel: string | null = null;
  let bestFreq = -1;
  for (const [label, val] of Object.entries(strategy)) {
    const f = parseFloat(val);
    if (Number.isFinite(f) && f > bestFreq) {
      bestFreq = f;
      bestLabel = label;
    }
  }
  if (!bestLabel) return null;
  return { label: bestLabel, freq: bestFreq };
}

/** Map an action label to a hex colour. Stable across cells so the grid
 *  reads as "what colour dominates each turn". Conservative palette so
 *  it works on light/dark themes. */
export function actionColor(label: string): string {
  if (label === 'Check' || label === 'Call') return '#6b7280';     // grey
  if (label === 'Fold') return '#374151';                          // darker grey
  if (label.startsWith('Bet_')) {
    const sz = parseInt(label.slice(4), 10);
    if (!Number.isFinite(sz)) return '#10b981';
    if (sz <= 33) return '#10b981';                                // small bet — green
    if (sz <= 75) return '#f59e0b';                                // medium — orange
    return '#ef4444';                                              // large — red
  }
  if (label.startsWith('Raise')) return '#8b5cf6';                 // purple
  if (label === 'All-in') return '#dc2626';                        // hard red
  return '#3b82f6';                                                // unknown — blue
}

// ─────────────────────────────────────────────────────────────────────
// Day 2 — Card Class Buckets
// ─────────────────────────────────────────────────────────────────────

/** Texture bucket a turn card falls into vs the flop. Mutually exclusive;
 *  classifier picks the first hit in priority order (pair > flush > straight
 *  > overcard > brick). The straight rule is a coarse approximation — it
 *  only checks whether the turn rank sits inside the flop's [min-1, max+1]
 *  span, so e.g. a T turn on AKQ (which actually completes a straight when
 *  combined with J) is currently classified as Brick. Good enough for a
 *  first-pass texture report; refine in Day 3 if users ask. */
export type CardClass = 'pair' | 'flush' | 'straight' | 'overcard' | 'brick';

/** Stable ordering for table rows + colour map. */
export const CARD_CLASS_ORDER: CardClass[] = [
  'pair', 'flush', 'straight', 'overcard', 'brick',
];

export const CARD_CLASS_LABELS: Record<CardClass, string> = {
  pair: 'Pair',
  flush: 'Flush completion',
  straight: 'Straight completion',
  overcard: 'Overcard',
  brick: 'Brick',
};

export const CARD_CLASS_COLORS: Record<CardClass, string> = {
  pair: '#f59e0b',      // amber
  flush: '#3b82f6',     // blue
  straight: '#8b5cf6',  // purple
  overcard: '#10b981',  // green
  brick: '#6b7280',     // grey
};

/** Classify a turn card relative to the flop. See `CardClass` doc for the
 *  priority order + the straight approximation caveat. Returns 'brick' if
 *  the card or flop is unparseable (defensive default). */
export function classifyTurn(turnCard: string, flopCards: string[]): CardClass {
  const t = parseCard(turnCard);
  if (!t) return 'brick';

  const flopParsed: Array<{ rankIdx: number; suit: string }> = [];
  for (const c of flopCards) {
    const p = parseCard(c);
    if (p) flopParsed.push(p);
  }
  if (flopParsed.length === 0) return 'brick';

  const flopRanks = flopParsed.map((c) => c.rankIdx);
  const flopSuits = flopParsed.map((c) => c.suit);

  // Pair: turn rank matches any flop rank
  if (flopRanks.includes(t.rankIdx)) return 'pair';

  // Flush completion: turn suit has 2+ cards on flop
  const sameSuitCount = flopSuits.filter((s) => s === t.suit).length;
  if (sameSuitCount >= 2) return 'flush';

  // Straight (coarse): turn rank within [min-1, max+1] of flop ranks
  const minRank = Math.min(...flopRanks);
  const maxRank = Math.max(...flopRanks);
  if (t.rankIdx >= minRank - 1 && t.rankIdx <= maxRank + 1) return 'straight';

  // Overcard: turn rank strictly above all flop ranks
  if (t.rankIdx > maxRank) return 'overcard';

  return 'brick';
}

/** Per-bucket roll-up of turn aggregates. `strategy` percentages and `meanEv`
 *  are weighted by `iso_weight` — i.e. a turn with weight 3 contributes 3×
 *  its strategy/EV to the bucket sums. `cards` lists the canonical reps that
 *  fell into the bucket so callers can render a tooltip / drill-down. */
export interface BucketAggregate {
  cardClass: CardClass;
  /** Display label (e.g. "Flush completion"). */
  label: string;
  /** Distinct turn aggregates in this bucket. */
  count: number;
  /** Sum of iso weights — useful as the "real turn count" denominator. */
  totalWeight: number;
  /** Action label → "%"-formatted weighted-average frequency, e.g. "34.6%". */
  strategy: Record<string, string>;
  /** Iso-weight-weighted mean EV in chips. */
  meanEv: number;
  /** Card tokens contributing to this bucket. */
  cards: string[];
}

export function bucketAggregates(
  turns: TurnAggregate[],
  flopCards: string[],
): BucketAggregate[] {
  type Acc = {
    cards: string[];
    totalWeight: number;
    strategySum: Record<string, number>;
    evWeightedSum: number;
  };
  const buckets: Record<CardClass, Acc> = {
    pair: { cards: [], totalWeight: 0, strategySum: {}, evWeightedSum: 0 },
    flush: { cards: [], totalWeight: 0, strategySum: {}, evWeightedSum: 0 },
    straight: { cards: [], totalWeight: 0, strategySum: {}, evWeightedSum: 0 },
    overcard: { cards: [], totalWeight: 0, strategySum: {}, evWeightedSum: 0 },
    brick: { cards: [], totalWeight: 0, strategySum: {}, evWeightedSum: 0 },
  };

  for (const turn of turns) {
    const cls = classifyTurn(turn.card, flopCards);
    const b = buckets[cls];
    b.cards.push(turn.card);
    b.totalWeight += turn.weight;
    b.evWeightedSum += turn.meanEv * turn.weight;
    for (const [label, freqStr] of Object.entries(turn.strategy)) {
      const f = parseFloat(freqStr);
      if (!Number.isFinite(f)) continue;
      b.strategySum[label] = (b.strategySum[label] ?? 0) + f * turn.weight;
    }
  }

  const out: BucketAggregate[] = [];
  for (const cls of CARD_CLASS_ORDER) {
    const b = buckets[cls];
    if (b.cards.length === 0) continue;
    const strategy: Record<string, string> = {};
    if (b.totalWeight > 0) {
      for (const [label, sum] of Object.entries(b.strategySum)) {
        strategy[label] = (sum / b.totalWeight).toFixed(1) + '%';
      }
    }
    out.push({
      cardClass: cls,
      label: CARD_CLASS_LABELS[cls],
      count: b.cards.length,
      totalWeight: b.totalWeight,
      strategy,
      meanEv: b.totalWeight > 0 ? b.evWeightedSum / b.totalWeight : 0,
      cards: b.cards,
    });
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Day 2 — Sort modes
// ─────────────────────────────────────────────────────────────────────

export type SortMode = 'card' | 'bestEv' | 'worstEv' | 'aggressive';

/** Returns total frequency of "aggressive" actions (Bet_*, Raise*, All-in)
 *  for ranking. Folds/Checks/Calls don't count. Used by `sortAggregates`. */
function aggressiveScore(strategy: Record<string, string>): number {
  let sum = 0;
  for (const [label, freqStr] of Object.entries(strategy)) {
    const f = parseFloat(freqStr);
    if (!Number.isFinite(f)) continue;
    if (label.startsWith('Bet_') || label.startsWith('Raise') || label === 'All-in') {
      sum += f;
    }
  }
  return sum;
}

/** Re-orders aggregates without mutating the input. `card` mode preserves
 *  the natural suit-then-rank order produced by `aggregateTurns`. */
export function sortAggregates(turns: TurnAggregate[], mode: SortMode): TurnAggregate[] {
  if (mode === 'card') return turns.slice();
  const arr = turns.slice();
  if (mode === 'bestEv') {
    arr.sort((a, b) => b.meanEv - a.meanEv);
  } else if (mode === 'worstEv') {
    arr.sort((a, b) => a.meanEv - b.meanEv);
  } else if (mode === 'aggressive') {
    arr.sort((a, b) => aggressiveScore(b.strategy) - aggressiveScore(a.strategy));
  }
  return arr;
}

// ─────────────────────────────────────────────────────────────────────
// Day 2 — CSV export
// ─────────────────────────────────────────────────────────────────────

/** Build a CSV string from turn aggregates. Columns: card, acting, every
 *  observed action's `_pct` (sorted alphabetically for stable output),
 *  mean_ev, iso_weight. Frequencies stored as plain numbers (no '%' suffix)
 *  so spreadsheet tools can do math on them. */
export function turnsToCSV(turns: TurnAggregate[]): string {
  const labelSet = new Set<string>();
  for (const t of turns) {
    for (const label of Object.keys(t.strategy)) labelSet.add(label);
  }
  const labels = Array.from(labelSet).sort();

  const header = ['card', 'acting', ...labels.map((l) => `${l}_pct`), 'mean_ev', 'iso_weight'];
  const rows = turns.map((t) => {
    const cells = [
      t.card,
      t.acting,
      ...labels.map((l) => {
        const raw = t.strategy[l];
        if (raw === undefined) return '';
        const f = parseFloat(raw);
        return Number.isFinite(f) ? f.toFixed(1) : '';
      }),
      t.meanEv.toFixed(2),
      String(t.weight),
    ];
    return cells.join(',');
  });
  return [header.join(','), ...rows].join('\n');
}
