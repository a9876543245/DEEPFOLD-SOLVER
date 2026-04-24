/**
 * Drill scenario generation and scoring engine.
 *
 * Generates random GTO quiz scenarios from preflop ranges + random boards,
 * scores user answers against the correct mixed strategy, and tracks session
 * results.
 */

import {
  GRID_LABELS,
  parseBoardCards,
  getHandStrength,
  RANK_VALUES,
  SUITS,
  RANKS,
} from './poker';
import type { ComboStrategy } from './poker';
import { MATCHUPS, parseRange } from './ranges';

// ============================================================================
// Types
// ============================================================================

export interface DrillScenario {
  id: number;
  board: string;
  matchupLabel: string;
  heroPosition: string; // 'IP' or 'OOP'
  heroCombo: string; // Grid label like "AKs"
  correctStrategy: ComboStrategy;
  availableActions: string[];
  boardTexture: string;
}

export interface DrillResult {
  score: number; // 0-100
  grade: 'optimal' | 'good' | 'acceptable' | 'mistake';
  correctAction: string;
  correctFreq: number;
  userFreq: number; // frequency of the user's chosen action
}

export interface DrillSession {
  scenarios: DrillScenario[];
  results: (DrillResult | null)[];
  currentIndex: number;
  totalScore: number;
}

// ============================================================================
// Board generation
// ============================================================================

/** Generate a random 3-card flop from the 52-card deck. */
export function generateRandomBoard(): string {
  const deck: string[] = [];
  for (const r of RANKS) {
    for (const s of SUITS) {
      deck.push(`${r}${s}`);
    }
  }

  // Fisher-Yates partial shuffle — pick 3
  for (let i = deck.length - 1; i > deck.length - 4; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }

  return deck[deck.length - 1] + deck[deck.length - 2] + deck[deck.length - 3];
}

// ============================================================================
// Board texture classification
// ============================================================================

/** Classify the texture of a 3-card board. */
export function classifyBoardTexture(board: string): string {
  const cards = parseBoardCards(board);
  const ranks = cards.map((c) => RANK_VALUES[c[0]] || 0).sort((a, b) => b - a);
  const suits = cards.map((c) => c[1]);

  const tags: string[] = [];

  // Suit analysis
  const suitCounts: Record<string, number> = {};
  for (const s of suits) {
    suitCounts[s] = (suitCounts[s] || 0) + 1;
  }
  const maxSuitCount = Math.max(...Object.values(suitCounts));
  if (maxSuitCount === 3) {
    tags.push('Monotone');
  }

  // Paired board
  const rankSet = new Set(ranks);
  if (rankSet.size < ranks.length) {
    tags.push('Paired');
  }

  // Connectedness: check if any two cards are within 2 gaps
  const connected =
    Math.abs(ranks[0] - ranks[1]) <= 2 ||
    Math.abs(ranks[1] - ranks[2]) <= 2 ||
    Math.abs(ranks[0] - ranks[2]) <= 2;

  if (maxSuitCount === 2 && connected) {
    tags.push('Wet');
  }

  // High / Low
  if (ranks[0] >= RANK_VALUES['Q']) {
    tags.push('High');
  } else if (ranks[0] <= RANK_VALUES['9']) {
    tags.push('Low');
  }

  if (tags.length === 0) {
    return 'Dry';
  }

  return tags.join(' / ');
}

// ============================================================================
// Contextual strategy generation (simplified copy from useSolver.ts)
// ============================================================================

/**
 * Produce a mixed strategy for a given hand strength.
 * This is a simplified version of the contextualStrategy function in useSolver.ts.
 * Only handles the 'none' facing-action case (initial action on flop).
 */
function contextualStrategy(strength: number, idx: number): ComboStrategy {
  const noise = Math.sin(idx * 137.5) * 0.08;

  if (strength > 0.85) {
    return {
      Check: 0.15 + noise * 0.5,
      'Bet 33%': 0.2,
      'Bet 75%': 0.5 - noise * 0.3,
      'All-in': 0.15,
    };
  } else if (strength > 0.65) {
    return {
      Check: 0.1,
      'Bet 33%': 0.55 + noise,
      'Bet 75%': 0.3 - noise,
      'All-in': 0.05,
    };
  } else if (strength > 0.45) {
    return {
      Check: 0.55 + noise,
      'Bet 33%': 0.35 - noise,
      'Bet 75%': 0.08,
      'All-in': 0.02,
    };
  } else if (strength > 0.25) {
    return {
      Check: 0.72 + noise,
      'Bet 33%': 0.18 - noise * 0.5,
      'Bet 75%': 0.08,
      'All-in': 0.02,
    };
  } else {
    return {
      Check: 0.65,
      'Bet 33%': 0.15 + noise,
      'Bet 75%': 0.12 - noise,
      'All-in': 0.08,
    };
  }
}

// ============================================================================
// Scenario generation
// ============================================================================

/** Generate a single drill scenario. */
export function generateDrillScenario(id: number): DrillScenario {
  // 1. Random matchup
  const matchup = MATCHUPS[Math.floor(Math.random() * MATCHUPS.length)];

  // 2. Random board
  const board = generateRandomBoard();
  const boardTexture = classifyBoardTexture(board);

  // 3. Pick hero side (IP or OOP)
  const isHeroIP = Math.random() < 0.5;
  const heroPosition = isHeroIP ? 'IP' : 'OOP';
  const heroRangeStr = isHeroIP ? matchup.ipRange : matchup.oopRange;

  // 4. Parse range and pick a random combo weighted by frequency
  const parsedRange = parseRange(heroRangeStr);
  const entries = Object.entries(parsedRange).filter(([, freq]) => freq > 0.05);

  if (entries.length === 0) {
    // Fallback: use first matchup entry
    entries.push(['AKs', 1.0]);
  }

  // Weighted random selection
  const totalWeight = entries.reduce((sum, [, freq]) => sum + freq, 0);
  let roll = Math.random() * totalWeight;
  let heroCombo = entries[0][0];
  for (const [combo, freq] of entries) {
    roll -= freq;
    if (roll <= 0) {
      heroCombo = combo;
      break;
    }
  }

  // 5. Generate strategy for the chosen combo
  const boardCards = parseBoardCards(board);
  const boardRanks = boardCards
    .map((c) => c[0])
    .sort((a, b) => (RANK_VALUES[b] || 0) - (RANK_VALUES[a] || 0));
  const strength = getHandStrength(heroCombo, boardRanks);
  const strategy = contextualStrategy(strength, id * 31 + heroCombo.charCodeAt(0));

  // Normalize strategy to sum to 1
  const total = Object.values(strategy).reduce((s, v) => s + v, 0);
  const normalized: ComboStrategy = {};
  for (const [action, freq] of Object.entries(strategy)) {
    normalized[action] = freq / total;
  }

  // 6. Extract available actions (freq > 1%)
  const availableActions = Object.entries(normalized)
    .filter(([, freq]) => freq > 0.01)
    .map(([action]) => action);

  return {
    id,
    board,
    matchupLabel: `${matchup.oop} vs ${matchup.ip} (${matchup.potType})`,
    heroPosition,
    heroCombo,
    correctStrategy: normalized,
    availableActions,
    boardTexture,
  };
}

// ============================================================================
// Scoring
// ============================================================================

/** Score a user's answer against the correct strategy. */
export function scoreDrillAnswer(
  userAction: string,
  correctStrategy: ComboStrategy,
): DrillResult {
  // Find highest-frequency action
  let correctAction = '';
  let correctFreq = 0;
  for (const [action, freq] of Object.entries(correctStrategy)) {
    if (freq > correctFreq) {
      correctFreq = freq;
      correctAction = action;
    }
  }

  // Find user's action frequency
  const userFreq = correctStrategy[userAction] ?? 0;

  // Determine grade and score
  let grade: DrillResult['grade'];
  let score: number;

  if (userAction === correctAction) {
    grade = 'optimal';
    score = 100;
  } else if (userFreq > 0.2) {
    grade = 'good';
    score = 70;
  } else if (userFreq > 0.05) {
    grade = 'acceptable';
    score = 40;
  } else {
    grade = 'mistake';
    score = 10;
  }

  return {
    score,
    grade,
    correctAction,
    correctFreq,
    userFreq,
  };
}

// ============================================================================
// Session management
// ============================================================================

/** Create a new drill session with the given number of scenarios. */
export function createDrillSession(count: number = 10): DrillSession {
  const scenarios: DrillScenario[] = [];
  for (let i = 0; i < count; i++) {
    scenarios.push(generateDrillScenario(i));
  }

  return {
    scenarios,
    results: new Array(count).fill(null),
    currentIndex: 0,
    totalScore: 0,
  };
}

/** Get aggregated summary of a completed drill session. */
export function getSessionSummary(session: DrillSession): {
  totalScore: number;
  maxScore: number;
  optimalCount: number;
  goodCount: number;
  mistakeCount: number;
} {
  let totalScore = 0;
  let optimalCount = 0;
  let goodCount = 0;
  let mistakeCount = 0;
  const maxScore = session.scenarios.length * 100;

  for (const result of session.results) {
    if (!result) continue;
    totalScore += result.score;
    switch (result.grade) {
      case 'optimal':
        optimalCount++;
        break;
      case 'good':
        goodCount++;
        break;
      case 'acceptable':
        // counted implicitly (not optimal, not good, not mistake)
        break;
      case 'mistake':
        mistakeCount++;
        break;
    }
  }

  return { totalScore, maxScore, optimalCount, goodCount, mistakeCount };
}
