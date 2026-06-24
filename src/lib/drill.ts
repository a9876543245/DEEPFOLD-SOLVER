/**
 * Drill scenario generation and scoring engine.
 *
 * Generates GTO quiz scenarios from REAL pre-solved spots (bundled solutions in
 * the Tauri app; a clearly-flagged heuristic demo only when no bundle exists or
 * in browser mode), and scores the user's chosen action against the solved
 * strategy's frequency for that action.
 *
 * Scoring is GTO-frequency match, not EV-loss: the engine emits per-combo node
 * EV but not per-action EV, so true EV-loss isn't computable from the current
 * output. Each scenario carries `isDemo`/`source` so the UI can label honestly.
 */

import type { ComboStrategy } from './poker';
import { MATCHUPS } from './ranges';
import { getBundledOrDemo, BOARD_TEMPLATES } from './presolvedSpots';

// ============================================================================
// Types
// ============================================================================

export interface DrillScenario {
  id: number;
  board: string;
  matchupLabel: string;
  heroPosition: string; // Always 'OOP' — pre-solved root is the OOP flop decision
  heroCombo: string; // Grid label like "AKs"
  correctStrategy: ComboStrategy;
  availableActions: string[];
  boardTexture: string;
  /** True if the strategy is the heuristic demo, not a real solve. */
  isDemo: boolean;
  /** Provenance of the strategy: 'bundled' | 'live' | 'demo'. */
  source?: 'bundled' | 'live' | 'demo';
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
// Scenario generation
// ============================================================================

/**
 * Generate a single drill scenario from a real pre-solved spot.
 *
 * The pre-solved bundle covers the OOP flop root decision, so the hero is OOP
 * and the correct strategy is the solver's actual mixed strategy for the picked
 * combo. Falls back to the heuristic demo spot (flagged via `isDemo`) only when
 * no bundle is available (browser mode / missing bundle).
 */
export async function generateDrillScenario(id: number): Promise<DrillScenario> {
  const matchupIdx = Math.floor(Math.random() * MATCHUPS.length);
  const boardIdx = Math.floor(Math.random() * BOARD_TEMPLATES.length);
  const matchup = MATCHUPS[matchupIdx];

  const spot = await getBundledOrDemo(matchupIdx, boardIdx);

  // Only quiz combos that actually have a solved strategy at this node.
  const comboLabels = Object.keys(spot.comboStrategies).filter((label) => {
    const s = spot.comboStrategies[label];
    return s && Object.values(s).some((f) => f > 0.005);
  });
  const heroCombo =
    comboLabels.length > 0
      ? comboLabels[Math.floor(Math.random() * comboLabels.length)]
      : 'AKs';

  const correctStrategy = spot.comboStrategies[heroCombo] ?? {};

  // Offer the node's full action menu (from the aggregate strategy), not just
  // the actions this combo happens to take — otherwise the menu leaks the
  // answer. Fall back to the combo's own actions if the aggregate is empty.
  const menu = Object.keys(spot.globalStrategy);
  const availableActions =
    menu.length > 0
      ? menu
      : Object.entries(correctStrategy)
          .filter(([, freq]) => freq > 0.01)
          .map(([action]) => action);

  return {
    id,
    board: spot.board,
    matchupLabel: `${matchup.oop} vs ${matchup.ip} (${matchup.potType})`,
    heroPosition: 'OOP',
    heroCombo,
    correctStrategy,
    availableActions,
    boardTexture: spot.boardTexture,
    isDemo: spot.isDemo,
    source: spot.source,
  };
}

// ============================================================================
// Scoring
// ============================================================================

/** Score a user's answer against the solved strategy's action frequencies. */
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
export async function createDrillSession(count: number = 10): Promise<DrillSession> {
  const scenarios = await Promise.all(
    Array.from({ length: count }, (_, i) => generateDrillScenario(i)),
  );

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
