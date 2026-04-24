// Poker constants and utilities for the frontend
export const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const;
export const SUITS = ['s', 'h', 'd', 'c'] as const;

export type Rank = typeof RANKS[number];
export type Suit = typeof SUITS[number];

export const SUIT_SYMBOLS: Record<Suit, string> = {
  s: '♠', h: '♥', d: '♦', c: '♣'
};

export const SUIT_COLORS: Record<Suit, string> = {
  s: 'var(--color-suit-spade)',
  h: 'var(--color-suit-heart)',
  d: 'var(--color-suit-diamond)',
  c: 'var(--color-suit-club)',
};

// 13x13 grid layout: rows = first rank, cols = second rank
// Upper-right triangle = suited, lower-left = offsuit, diagonal = pairs
export const GRID_LABELS: string[][] = RANKS.map((r1, i) =>
  RANKS.map((r2, j) => {
    if (i === j) return `${r1}${r2}`;
    if (i < j) return `${r1}${r2}s`;
    return `${r2}${r1}o`;
  })
);

// Rank numeric values for hand strength calculations
export const RANK_VALUES: Record<string, number> = {
  'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
  '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2,
};

// Strategy action colors (GTO Wizard-like palette)
export const ACTION_COLORS: Record<string, string> = {
  'Not in range': '#1C1C1E',
  'Fold': '#636366',
  'Check': '#30D158',
  'Call': '#30D158',
  'Bet_33': '#FF453A',
  'Bet_75': '#FF6B35',
  'Bet': '#FF453A',
  'Raise': '#BF5AF2',
  'All-in': '#FF9F0A',
};

export function getActionColor(label: string): string {
  // Exact match first
  if (ACTION_COLORS[label]) return ACTION_COLORS[label];
  // Prefix match
  for (const [key, color] of Object.entries(ACTION_COLORS)) {
    if (label.startsWith(key)) return color;
  }
  return '#8E8E93';
}

// ============================================================================
// Per-combo strategy type
// ============================================================================

/** Strategy frequencies for a single combo, keyed by action name */
export type ComboStrategy = Record<string, number>;

export interface NodeLock {
  history: string;
  combo: string;
  strategy: number[];
}

// Solver types (mirrors Rust types)
export interface SolverRequest {
  board: string;
  pot_size: number;
  effective_stack: number;
  history?: string;
  target_combo?: string;
  iterations?: number;
  exploitability?: number;
  /** IP player's preflop range (TexasSolver format) */
  ip_range?: string;
  /** OOP player's preflop range (TexasSolver format) */
  oop_range?: string;
  /** JSON encoded string of NodeLock[] */
  node_locks?: string;
  /** Hero's range — used to filter the range grid display */
  hero_range?: string;
  /** Action path for game tree navigation */
  action_path?: import('../lib/gameTree').ActionStep[];
  /** Execution backend: 'auto' | 'cpu' | 'gpu'. Defaults to 'auto'. */
  backend?: string;
  /** OOP has initiative at the root flop node (can bet). Defaults to true.
   *  Set to false for SRP scenarios where OOP should only check to the
   *  preflop raiser. */
  oop_has_initiative?: boolean;
  /** Allow OOP donk bets even without initiative. Defaults to false. */
  allow_donk_bet?: boolean;
  /** Bet sizes per street as pot fractions (e.g. [0.33, 0.75]).
   *  The backend uses these for both bets AND raises, so the sizing
   *  choice also shapes the raise tree. Must be passed through when the
   *  UI uses non-default sizing (Small Ball, Polar, etc.) or the solver
   *  tree will not contain the actions the UI offers. */
  flop_sizes?: number[];
  turn_sizes?: number[];
  river_sizes?: number[];
}

export interface ComboAnalysis {
  combo: string;
  best_action: string;
  ev: number;
  strategy_mix: Record<string, string>;
}

export interface SolverResponse {
  status: string;
  iterations_run: number;
  exploitability_pct: number;
  global_strategy: Record<string, string>;
  /** Per-combo strategy frequencies: { "AA": { "Check": 0.1, "Bet_33": 0.6, ... }, ... } */
  combo_strategies?: Record<string, ComboStrategy>;
  target_combo_analysis?: ComboAnalysis;
  /** Active backend name (e.g. "CPU-DCFR", "CUDA (RTX 5090, ...)") */
  backend?: string;
  /** Player acting at the current node. */
  acting_player?: 'OOP' | 'IP';
  /** Opposing side for the current node (non-acting player). */
  opponent_side?: 'OOP' | 'IP';
  /** Opponent's reach-weighted range at this node, keyed by grid label.
   *  Values in [0, 1] normalized so the heaviest label at this node = 1.0.
   *  Used by the "view opponent" toggle in RangeGrid to render a heatmap
   *  of "what hands does the opponent have here". */
  opponent_range?: Record<string, number>;
  /** Route A navigation cache: history-path -> per-node strategy bundle.
   *  Lets the UI navigate within an already-solved tree without re-invoking
   *  the engine. Path format matches the comma-separated player-action
   *  history sent to the engine ("" = root, "Check,Bet_75" = OOP checks
   *  then IP bets 75% pot). Populated on every solve. */
  strategy_tree?: Record<string, StrategyTreeEntry>;
  /** Path B: cumulative dealt cards from root via this path's chance jumps,
   *  for the CURRENTLY-DISPLAYED node. Empty pre-chance. Mirrors the field
   *  in the cache entry — `useSolver.navigate()` copies it to the top level
   *  so the UI can read it without descending into strategy_tree. */
  dealt_cards?: string[];
  /** Path B: canonical runout reps available at the immediate prior chance,
   *  for the CURRENTLY-DISPLAYED node. Mirrors strategy_tree entry field. */
  runout_options?: RunoutOption[];
}

/** One entry in `strategy_tree`. Mirrors the per-node fields the UI reads
 *  from SolverResponse. Lets `useSolver.navigate(history)` synthesize a
 *  fresh response from cache without an engine round-trip. */
export interface StrategyTreeEntry {
  acting: 'OOP' | 'IP';
  action_labels: string[];
  global_strategy: Record<string, string>;
  combo_strategies: Record<string, ComboStrategy>;
  opponent_side: 'OOP' | 'IP';
  opponent_range: Record<string, number>;
  /** Path B: cumulative dealt cards from root via this path's chance jumps.
   *  Empty for pre-chance nodes. Used by the UI to disclose the actual
   *  runout the cached strategies represent (e.g. "2c"). */
  dealt_cards?: string[];
  /** Path B: canonical runout reps at the immediate prior chance step.
   *  Empty when no chance preceded this node. Powers the runout-picker UI.
   *  The currently-active runout is the last entry of `dealt_cards`. */
  runout_options?: RunoutOption[];
}

/** One canonical runout option for the Path B runout-picker UI. */
export interface RunoutOption {
  card: string;     // "2c", "As", etc.
  weight: number;   // orbit size
}

/// GPU detection info — returned by Tauri command `get_gpu_info`.
export interface GpuInfo {
  has_cuda_gpu: boolean;
  gpu_description: string;
  gpu_backend_functional: boolean;
}

// ============================================================================
// Range Templates — one-click presets
// ============================================================================

export interface RangeTemplate {
  name: string;
  description: string;
  weights: Record<string, number>;
}

function buildTemplate(combos: string[], weight = 1.0): Record<string, number> {
  const w: Record<string, number> = {};
  for (const c of combos) w[c] = weight;
  return w;
}

const PAIRS = ['AA','KK','QQ','JJ','TT','99','88','77','66','55','44','33','22'];
const SUITED_BW = ['AKs','AQs','AJs','ATs','KQs','KJs','KTs','QJs','QTs','JTs'];
const SUITED_CONN = ['T9s','98s','87s','76s','65s','54s'];
const OFFSUIT_BW = ['AKo','AQo','AJo','ATo','KQo','KJo','QJo'];

export const RANGE_TEMPLATES: RangeTemplate[] = [
  {
    name: 'Top 15%',
    description: 'Tight — premium hands',
    weights: buildTemplate([
      ...PAIRS.slice(0, 6), // AA-99
      ...SUITED_BW.slice(0, 6), // AKs-KTs
      'AKo', 'AQo',
    ]),
  },
  {
    name: 'Top 30%',
    description: 'Standard open range',
    weights: buildTemplate([
      ...PAIRS, // All pairs
      ...SUITED_BW,
      ...SUITED_CONN.slice(0, 4), // T9s-76s
      ...OFFSUIT_BW,
      'A9s','A8s','A7s','A6s','A5s','A4s','A3s','A2s',
      'K9s','Q9s','J9s',
    ]),
  },
  {
    name: 'Top 50%',
    description: 'Loose — wide range',
    weights: buildTemplate([
      ...PAIRS,
      ...SUITED_BW,
      ...SUITED_CONN,
      ...OFFSUIT_BW,
      'A9s','A8s','A7s','A6s','A5s','A4s','A3s','A2s',
      'K9s','K8s','K7s','K6s','K5s','K4s','K3s','K2s',
      'Q9s','Q8s','J9s','J8s','T9s','T8s','97s','96s','86s','75s','64s','53s',
      'KTo','QTo','JTo','T9o','98o',
      'A9o','A8o','A7o','A6o','A5o',
    ]),
  },
  {
    name: 'All Pairs',
    description: '22-AA (6%)',
    weights: buildTemplate(PAIRS),
  },
  {
    name: 'Suited Broadway',
    description: 'AKs-JTs (5%)',
    weights: buildTemplate(SUITED_BW),
  },
  {
    name: 'Suited Connectors',
    description: 'T9s-54s (3%)',
    weights: buildTemplate(SUITED_CONN),
  },
];

// ============================================================================
// Combo Expansion: grid label -> specific combos
// ============================================================================

export interface SpecificCombo {
  combo: string;        // e.g., "AhKh"
  rank1: string;
  suit1: Suit;
  rank2: string;
  suit2: Suit;
  isDead: boolean;
}

/**
 * Expand a grid label into all specific combos, marking dead ones.
 * "AKs" -> 4 suited combos, "AKo" -> 12 offsuit combos, "AA" -> 6 pair combos
 */
export function expandComboLabel(label: string, boardCards: string[]): SpecificCombo[] {
  const deadSet = new Set(boardCards);
  const isPair = label.length === 2 && label[0] === label[1];
  const isSuited = label.endsWith('s');
  const r1 = label[0];
  const r2 = isPair ? label[1] : label[1];

  const combos: SpecificCombo[] = [];

  if (isPair) {
    // Pairs: C(4,2) = 6 combos
    for (let i = 0; i < SUITS.length; i++) {
      for (let j = i + 1; j < SUITS.length; j++) {
        const c1 = r1 + SUITS[i];
        const c2 = r2 + SUITS[j];
        combos.push({
          combo: c1 + c2,
          rank1: r1, suit1: SUITS[i],
          rank2: r2, suit2: SUITS[j],
          isDead: deadSet.has(c1) || deadSet.has(c2),
        });
      }
    }
  } else if (isSuited) {
    // Suited: 4 combos (same suit)
    for (const s of SUITS) {
      const c1 = r1 + s;
      const c2 = r2 + s;
      combos.push({
        combo: c1 + c2,
        rank1: r1, suit1: s,
        rank2: r2, suit2: s,
        isDead: deadSet.has(c1) || deadSet.has(c2),
      });
    }
  } else {
    // Offsuit: 12 combos (different suits)
    for (const s1 of SUITS) {
      for (const s2 of SUITS) {
        if (s1 === s2) continue;
        const c1 = r1 + s1;
        const c2 = r2 + s2;
        combos.push({
          combo: c1 + c2,
          rank1: r1, suit1: s1,
          rank2: r2, suit2: s2,
          isDead: deadSet.has(c1) || deadSet.has(c2),
        });
      }
    }
  }

  return combos;
}

// ============================================================================
// Hand strength heuristic (for mock solver)
// ============================================================================

/**
 * Parse board cards from string (e.g., "AsKd7c" → ["As", "Kd", "7c"])
 */
export function parseBoardCards(board: string): string[] {
  const cards: string[] = [];
  for (let i = 0; i < board.length; i += 2) {
    cards.push(board.substring(i, i + 2));
  }
  return cards;
}

/**
 * Get a rough hand category from a combo label.
 * Returns a value from 0 (trash) to 1 (nuts) for strategy generation.
 */
export function getHandStrength(label: string, boardRanks: string[]): number {
  const isPair = !label.endsWith('s') && !label.endsWith('o') && label[0] === label[1];
  const isSuited = label.endsWith('s');
  const r1 = label[0];
  const r2 = isPair ? label[1] : label[1];
  const v1 = RANK_VALUES[r1] || 0;
  const v2 = RANK_VALUES[r2] || 0;
  const highCard = Math.max(v1, v2);
  const lowCard = Math.min(v1, v2);
  const gap = highCard - lowCard;

  // Check for board interaction
  const hitsBoard = boardRanks.includes(r1) || boardRanks.includes(r2);
  const hitsTop = boardRanks.length > 0 && (r1 === boardRanks[0] || r2 === boardRanks[0]);
  const hasOverpair = isPair && v1 > (RANK_VALUES[boardRanks[0]] || 0);
  const hasSet = isPair && boardRanks.includes(r1);

  let strength = 0;

  if (hasSet) {
    // Set — very strong
    strength = 0.92 + (v1 / 14) * 0.08;
  } else if (hasOverpair) {
    // Overpair
    strength = 0.78 + (v1 / 14) * 0.12;
  } else if (isPair && hitsBoard) {
    // Set on board
    strength = 0.90 + (v1 / 14) * 0.10;
  } else if (isPair) {
    // Underpair / pocket pair
    strength = 0.35 + (v1 / 14) * 0.35;
  } else if (hitsTop && highCard >= 12) {
    // Top pair with good kicker
    strength = 0.65 + (lowCard / 14) * 0.2;
  } else if (hitsTop) {
    // Top pair weak kicker
    strength = 0.50 + (lowCard / 14) * 0.15;
  } else if (hitsBoard && highCard >= 12) {
    // Middle/bottom pair good kicker
    strength = 0.40 + (lowCard / 14) * 0.15;
  } else if (hitsBoard) {
    // Middle/bottom pair
    strength = 0.25 + (lowCard / 14) * 0.15;
  } else if (highCard >= 14 && lowCard >= 13) {
    // AK type — unpaired overcards
    strength = 0.55 + (isSuited ? 0.08 : 0);
  } else if (highCard >= 14) {
    // Ace-high (no pair)
    strength = 0.30 + (lowCard / 14) * 0.15 + (isSuited ? 0.06 : 0);
  } else if (gap <= 2 && highCard >= 8) {
    // Connectors / one-gappers
    strength = 0.20 + (highCard / 14) * 0.15 + (isSuited ? 0.10 : 0);
  } else if (isSuited && highCard >= 10) {
    // Suited broadway
    strength = 0.25 + (highCard / 14) * 0.1;
  } else {
    // Trash
    strength = (highCard + lowCard) / 28 * 0.20 + (isSuited ? 0.05 : 0);
  }

  return Math.max(0, Math.min(1, strength));
}

/**
 * Generate a realistic strategy mix for a given hand strength.
 * Strong hands bet more; weak hands fold/check more; medium hands mix.
 */
export function generateStrategyFromStrength(
  strength: number,
  seed: number = 0,
): ComboStrategy {
  // Add deterministic "randomness" from seed for variety
  const noise = (Math.sin(seed * 127.1 + 311.7) * 43758.5453) % 1;
  const n = Math.abs(noise) * 0.15; // up to ±15% noise

  if (strength > 0.85) {
    // Monsters: mostly bet big or check-raise (sometimes trap)
    const trap = 0.10 + n * 0.5;
    const betBig = 0.35 - n * 0.3;
    const allIn = 0.25 + n * 0.4;
    const bet33 = 1 - trap - betBig - allIn;
    return { 'Check': trap, 'Bet_33': Math.max(0.05, bet33), 'Bet_75': betBig, 'All-in': allIn };
  } else if (strength > 0.65) {
    // Strong: bet for value
    const check = 0.15 + n;
    const bet33 = 0.45 - n * 0.5;
    const bet75 = 0.30 + n * 0.3;
    const allIn = 0.10 - n * 0.3;
    return { 'Check': check, 'Bet_33': bet33, 'Bet_75': bet75, 'All-in': Math.max(0, allIn) };
  } else if (strength > 0.45) {
    // Medium: thin value / protection bets
    const check = 0.40 + n;
    const bet33 = 0.35 - n * 0.5;
    const bet75 = 0.15 - n * 0.3;
    const allIn = 0.05;
    const fold = 0.05 + n * 0.3;
    return { 'Fold': fold, 'Check': check, 'Bet_33': bet33, 'Bet_75': Math.max(0, bet75), 'All-in': allIn };
  } else if (strength > 0.25) {
    // Weak: mostly check/fold, some bluffs
    const fold = 0.25 + n;
    const check = 0.45 - n * 0.3;
    const bet33 = 0.20 - n * 0.5;
    const bet75 = 0.05 + n * 0.3;
    const allIn = 0.05;
    return { 'Fold': fold, 'Check': check, 'Bet_33': Math.max(0, bet33), 'Bet_75': bet75, 'All-in': allIn };
  } else {
    // Trash: fold or check
    const fold = 0.55 + n;
    const check = 0.35 - n * 0.5;
    const bet33 = 0.07 + n * 0.2; // occasional bluff
    const bet75 = 0.03;
    return { 'Fold': fold, 'Check': Math.max(0.1, check), 'Bet_33': bet33, 'Bet_75': bet75 };
  }
}
