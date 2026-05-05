/**
 * 1326 Combo Drill — Day 3
 *
 * Pure functions for per-specific-combo blocker analysis.
 *
 * Why this exists: the engine emits per-class (169-label) strategy and EV,
 * not per-specific-combo. So when the user has e.g. AKs in their range
 * with a mixed strategy, they can't tell from the class data alone which
 * of the 4 specific AKs combos (AsKs/AhKh/AdKd/AcKc) is the best
 * candidate for the aggressive line.
 *
 * The classic poker tie-breaker is **blockers**: a hero combo that
 * removes more cards from the opponent's value range is a better
 * candidate to bluff/raise with, and conversely a worse candidate to
 * call/bluff-catch with.
 *
 * `blockerImpact(heroCombo, opponentRange, board)` answers:
 *   "How much of opp's range does this specific hero combo block?"
 *
 * It's a board-aware count: opp combos that share a card with the board
 * are excluded from both totals, then opp combos that share a card with
 * hero are counted as blocked. Result is a weighted percentage plus the
 * top-5 most-blocked opponent classes for context.
 *
 * Caveat: per-combo strategy/EV is still class-shared. Day 3.5+ would add
 * per-combo emission from the engine; until then the drill panel labels
 * those values as "shared with class" so users don't misread them.
 */

import { expandComboLabel, type SpecificCombo } from './poker';

/** Result of `blockerImpact`. */
export interface BlockerInfo {
  /** Total live opp combos in range (weighted by opp_range freq, board-removed). */
  totalOppCombos: number;
  /** Live opp combos blocked by hero (weighted). */
  blockedCombos: number;
  /** % of opp range blocked, 0..100. NaN-safe (returns 0 when total is 0). */
  blockedPct: number;
  /** Top opponent class labels most blocked by this hero combo. Useful for
   *  showing "blocks AsKs/AsQs" type tooltips. Sorted by contribution desc. */
  topBlocked: Array<{
    label: string;
    /** Raw count of opp specific combos in this label hero blocks (1..N). */
    blockedCount: number;
    /** opp_range weight for the label. */
    weight: number;
    /** blockedCount × weight — used for sorting. */
    contribution: number;
  }>;
}

/** Per-combo drill result. */
export interface DrilledCombo {
  combo: SpecificCombo;
  blocker: BlockerInfo;
}

/** Empty blocker — used when input is missing or hero combo is dead. */
const EMPTY_BLOCKER: BlockerInfo = {
  totalOppCombos: 0, blockedCombos: 0, blockedPct: 0, topBlocked: [],
};

/**
 * Compute how much of `opponentRange` is blocked by `heroCombo`.
 *
 * Algorithm:
 *   For each label in opp range with weight > 0:
 *     1. Expand label → 4/6/12 specific opp combos (already board-removed).
 *     2. Of the live ones, count those sharing a card with heroCombo.
 *     3. Add `liveCount × weight` to total, `blockedCount × weight` to blocked.
 *   Top-blocked = labels with at least one block, sorted by contribution.
 */
export function blockerImpact(
  heroCombo: SpecificCombo,
  opponentRange: Record<string, number> | undefined,
  boardCards: string[],
): BlockerInfo {
  if (!opponentRange) return EMPTY_BLOCKER;
  if (heroCombo.isDead) return EMPTY_BLOCKER;

  const heroCards = new Set([
    heroCombo.rank1 + heroCombo.suit1,
    heroCombo.rank2 + heroCombo.suit2,
  ]);

  let totalCombos = 0;
  let blockedCombos = 0;
  const perLabel: BlockerInfo['topBlocked'] = [];

  for (const [label, rawWeight] of Object.entries(opponentRange)) {
    const weight = typeof rawWeight === 'number' ? rawWeight : parseFloat(String(rawWeight));
    if (!Number.isFinite(weight) || weight <= 0) continue;

    const oppCombos = expandComboLabel(label, boardCards);
    let labelLive = 0;
    let labelBlocked = 0;
    for (const opp of oppCombos) {
      if (opp.isDead) continue;
      labelLive += 1;
      const oppC1 = opp.rank1 + opp.suit1;
      const oppC2 = opp.rank2 + opp.suit2;
      if (heroCards.has(oppC1) || heroCards.has(oppC2)) {
        labelBlocked += 1;
      }
    }
    if (labelLive > 0) {
      totalCombos += labelLive * weight;
      blockedCombos += labelBlocked * weight;
      if (labelBlocked > 0) {
        perLabel.push({
          label,
          blockedCount: labelBlocked,
          weight,
          contribution: labelBlocked * weight,
        });
      }
    }
  }

  perLabel.sort((a, b) => b.contribution - a.contribution);

  return {
    totalOppCombos: totalCombos,
    blockedCombos,
    blockedPct: totalCombos > 0 ? (blockedCombos / totalCombos) * 100 : 0,
    topBlocked: perLabel.slice(0, 5),
  };
}

/** Drill a 169-class label into per-specific-combo blocker results.
 *  No sorting — caller decides (e.g. live-first then by blocker desc). */
export function analyzeClass(
  label: string,
  opponentRange: Record<string, number> | undefined,
  boardCards: string[],
): DrilledCombo[] {
  const combos = expandComboLabel(label, boardCards);
  return combos.map((combo) => ({
    combo,
    blocker: blockerImpact(combo, opponentRange, boardCards),
  }));
}

/** Sort drilled combos: live first, then by blocker % desc. Returns a new
 *  array — caller's input is untouched. */
export function sortDrilledByBlocker(combos: DrilledCombo[]): DrilledCombo[] {
  const out = combos.slice();
  out.sort((a, b) => {
    if (a.combo.isDead !== b.combo.isDead) return a.combo.isDead ? 1 : -1;
    return b.blocker.blockedPct - a.blocker.blockedPct;
  });
  return out;
}

/** Returns labels with non-zero opp-range weight, sorted by weight desc.
 *  Used by the drill panel's class picker to filter "in-range" classes. */
export function rangedLabels(
  range: Record<string, number> | undefined,
): Array<{ label: string; weight: number }> {
  if (!range) return [];
  const out: Array<{ label: string; weight: number }> = [];
  for (const [label, rawWeight] of Object.entries(range)) {
    const w = typeof rawWeight === 'number' ? rawWeight : parseFloat(String(rawWeight));
    if (Number.isFinite(w) && w > 0) out.push({ label, weight: w });
  }
  out.sort((a, b) => b.weight - a.weight);
  return out;
}
