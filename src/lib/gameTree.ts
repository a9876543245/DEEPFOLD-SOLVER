/**
 * Game Tree model for postflop decision navigation.
 *
 * The tree alternates between OOP (out-of-position) and IP (in-position) players.
 * OOP always acts first on each street.
 *
 * Terminal conditions:
 *   - Fold → showdown avoided, folder loses current pot equity
 *   - Call after bet/raise on river → showdown
 *   - Check-check on river → showdown
 *   - All-in + call → run out remaining streets
 */

export type Player = 'OOP' | 'IP';
export type Street = 'flop' | 'turn' | 'river';
export type ActionType = 'check' | 'bet' | 'raise' | 'call' | 'fold' | 'allin';

export interface GameAction {
  /** Display label, e.g., "Bet 33%", "Check", "Raise 3x" */
  label: string;
  /** Action type classification */
  type: ActionType;
  /** Bet/raise amount in chips (undefined for check/fold) */
  amount?: number;
  /** Size as percentage of pot (for display) */
  potPct?: number;
}

export interface ActionStep {
  action: GameAction;
  player: Player | 'Deal';
  /** Pot size AFTER this action */
  potAfter: number;
  /** Effective stack AFTER this action */
  stackAfter: number;
}

export interface GameTreeNode {
  /** Unique node ID (path-based, e.g., "root", "root.bet33", "root.bet33.raise") */
  id: string;
  /** Player who acts at this node */
  activePlayer: Player;
  /** Current street */
  street: Street;
  /** Pot size at this node */
  pot: number;
  /** Remaining effective stack */
  effectiveStack: number;
  /** Available actions from this node */
  actions: GameAction[];
  /** Whether this is a terminal node (fold, showdown) */
  isTerminal: boolean;
  /** Action path leading to this node */
  path: ActionStep[];
  /** Whether this node is awaiting a deal card (Turn or River) before actions can proceed */
  awaitingDeal?: 'turn' | 'river';
}

// ============================================================================
// Bet sizing configurations
// ============================================================================

export interface BetSizing {
  label: string;
  /** Sizes as fraction of pot (0.33 = 33%) */
  flopBetSizes: number[];
  flopRaiseSizes: number[];  // multiplier of the bet (e.g., 3 = 3x)
  turnBetSizes: number[];
  turnRaiseSizes: number[];
  riverBetSizes: number[];
  riverRaiseSizes: number[];
}

export const BET_SIZINGS: Record<string, BetSizing> = {
  standard: {
    label: 'Standard',
    flopBetSizes: [0.33, 0.75],
    flopRaiseSizes: [3],
    turnBetSizes: [0.33, 0.75],
    turnRaiseSizes: [3],
    riverBetSizes: [0.33, 0.75],
    riverRaiseSizes: [3],
  },
  polar: {
    label: 'Polar',
    flopBetSizes: [0.75, 1.5],
    flopRaiseSizes: [3],
    turnBetSizes: [0.75, 1.5],
    turnRaiseSizes: [2.5],
    riverBetSizes: [0.75, 1.5],
    riverRaiseSizes: [2.5],
  },
  small_ball: {
    label: 'Small Ball',
    flopBetSizes: [0.25, 0.33],
    flopRaiseSizes: [3],
    turnBetSizes: [0.25, 0.33],
    turnRaiseSizes: [3],
    riverBetSizes: [0.33, 0.5],
    riverRaiseSizes: [2.5],
  },
};

// ============================================================================
// Tree construction helpers
// ============================================================================

/**
 * Get bet sizes for the current street.
 */
function getBetSizes(street: Street, sizing: BetSizing): number[] {
  switch (street) {
    case 'flop': return sizing.flopBetSizes;
    case 'turn': return sizing.turnBetSizes;
    case 'river': return sizing.riverBetSizes;
  }
}

function getRaiseSizes(street: Street, sizing: BetSizing): number[] {
  switch (street) {
    case 'flop': return sizing.flopRaiseSizes;
    case 'turn': return sizing.turnRaiseSizes;
    case 'river': return sizing.riverRaiseSizes;
  }
}

/**
 * Generate available actions at a game tree node.
 *
 * @param pot - Current pot size
 * @param stack - Remaining effective stack
 * @param street - Current street
 * @param facingBet - The bet amount we're facing (0 if no bet yet)
 * @param sizing - Bet sizing configuration
 */
// SPR threshold below which the backend's game-tree builder replaces any
// bet / raise with pure all-in (see core/include/game_tree_builder.h
// `should_force_allin`). Must stay in sync or the UI will offer actions
// that don't exist in the solved tree, leading to broken navigation.
const ALLIN_SPR_THRESHOLD = 0.12;

/** Mirrors backend should_force_allin: if committing this bet leaves SPR
 *  below threshold after opponent calls, the backend collapses the sizing
 *  into an all-in, so we must not offer the sized bet to the user. */
function forcesAllIn(pot: number, stack: number, bet: number): boolean {
  const remaining = stack - bet;
  if (remaining <= 0) return true;
  const newPot = pot + bet * 2;
  if (newPot <= 0) return true;
  return remaining / newPot < ALLIN_SPR_THRESHOLD;
}

export function getAvailableActions(
  pot: number,
  stack: number,
  street: Street,
  facingBet: number,
  sizing: BetSizing = BET_SIZINGS.standard,
): GameAction[] {
  const actions: GameAction[] = [];
  let allInForced = false;

  if (facingBet <= 0) {
    // No bet to face — can check or bet
    actions.push({ label: 'Check', type: 'check' });

    // Bet sizes — skip any size that the backend's SPR pruning would have
    // collapsed into an all-in. The first forced-all-in size also short-
    // circuits larger ones (matches the backend's `break` after force).
    for (const pct of getBetSizes(street, sizing)) {
      const amount = Math.round(pot * pct);
      if (amount <= 0 || amount >= stack) continue;
      if (forcesAllIn(pot, stack, amount)) {
        allInForced = true;
        break;
      }
      const pctLabel = Math.round(pct * 100);
      actions.push({
        label: `Bet ${pctLabel}%`,
        type: 'bet',
        amount,
        potPct: pct,
      });
    }

    // All-in: always offered when any reasonable stack left. We keep the
    // prior "stack <= pot * 3" heuristic so we don't spam tiny jams in
    // very deep-stacked preflop spots — but if SPR pruning forced an
    // all-in, we always include it regardless.
    if (stack > 0 && (allInForced || stack <= pot * 3)) {
      actions.push({ label: 'All-in', type: 'allin', amount: stack });
    }
  } else {
    // Facing a bet — Fold / Call / Raise / All-in
    actions.push({ label: 'Fold', type: 'fold' });

    if (facingBet <= stack) {
      actions.push({
        label: 'Call',
        type: 'call',
        amount: facingBet,
      });
    }

    // Raise sizes — use SAME fractions as bet sizes to match backend:
    //   raise_to = facingBet + (pot + facingBet) * frac
    // This is the convention in game_tree_builder.h. Previously we used
    // "multiplier of facing bet" (e.g. 3x) which produced amounts the
    // backend tree didn't contain, forcing fuzzy substitution. Aligning
    // on pot-fraction makes history navigation exact.
    const potWithBet = pot + facingBet;
    for (const frac of getBetSizes(street, sizing)) {
      const raiseAmount = Math.round(facingBet + potWithBet * frac);
      if (raiseAmount <= facingBet || raiseAmount >= stack) continue;
      if (forcesAllIn(pot + facingBet, stack, raiseAmount)) {
        allInForced = true;
        break;
      }
      const pctLabel = Math.round(frac * 100);
      actions.push({
        label: `Raise ${pctLabel}%`,
        type: 'raise',
        amount: raiseAmount,
        potPct: frac,
      });
    }

    // All-in raise — always offered if stack allows and it wasn't already
    // added. Forced-all-in from pruning above takes precedence.
    if (stack > facingBet && stack > 0) {
      const isAlreadyListed = actions.some(a => a.type === 'allin');
      if (!isAlreadyListed) {
        actions.push({ label: 'All-in', type: 'allin', amount: stack });
      }
    }
  }

  return actions;
}

/**
 * Create the root node of the game tree after solving.
 */
export function createRootNode(
  pot: number,
  effectiveStack: number,
  street: Street = 'flop',
  sizingKey: string = 'standard',
): GameTreeNode {
  const sizing = BET_SIZINGS[sizingKey] || BET_SIZINGS.standard;

  return {
    id: 'root',
    activePlayer: 'OOP',
    street,
    pot,
    effectiveStack,
    actions: getAvailableActions(pot, effectiveStack, street, 0, sizing),
    isTerminal: false,
    path: [],
  };
}

/**
 * Navigate to a child node by taking an action.
 *
 * Returns the next GameTreeNode after the action is applied.
 */
export function takeAction(
  currentNode: GameTreeNode,
  action: GameAction,
  sizingKey: string = 'standard',
): GameTreeNode {
  const sizing = BET_SIZINGS[sizingKey] || BET_SIZINGS.standard;
  const { pot, effectiveStack, activePlayer, street, path } = currentNode;

  let newPot = pot;
  let newStack = effectiveStack;
  let newPlayer: Player = activePlayer === 'OOP' ? 'IP' : 'OOP';
  let newStreet = street;
  let isTerminal = false;
  let facingBet = 0;

  switch (action.type) {
    case 'check': {
      // Check — if this is the second check (IP checks back), advance street
      const lastStep = path[path.length - 1];
      const isCheckBack = lastStep && lastStep.action.type === 'check';

      if (isCheckBack) {
        // Check-check: advance to next street
        if (street === 'river') {
          isTerminal = true; // Showdown
        } else {
          newStreet = street === 'flop' ? 'turn' : 'river';
          newPlayer = 'OOP'; // OOP acts first on new street
        }
      }
      break;
    }

    case 'bet': {
      const betAmount = action.amount || 0;
      newPot = pot + betAmount;
      newStack = effectiveStack - betAmount;
      facingBet = betAmount;
      break;
    }

    case 'raise': {
      const raiseAmount = action.amount || 0;
      // Facing a bet: find the previous bet amount
      const prevBetStep = [...path].reverse().find(s => s.action.type === 'bet' || s.action.type === 'raise');
      const prevBet = prevBetStep?.action.amount || 0;
      newPot = pot + raiseAmount;
      newStack = effectiveStack - raiseAmount;
      facingBet = raiseAmount - prevBet; // the additional amount opponent needs to call
      break;
    }

    case 'call': {
      const callAmount = action.amount || 0;
      newPot = pot + callAmount;
      newStack = effectiveStack - callAmount;

      // Call closes the action — advance street or showdown
      if (street === 'river') {
        isTerminal = true;
      } else {
        newStreet = street === 'flop' ? 'turn' : 'river';
        newPlayer = 'OOP';
      }
      facingBet = 0;
      break;
    }

    case 'fold': {
      isTerminal = true;
      break;
    }

    case 'allin': {
      const allinAmount = action.amount || effectiveStack;
      newPot = pot + allinAmount;
      newStack = 0;
      facingBet = allinAmount;

      // If this is a call-all-in (opponent was already all-in)
      const prevStep = path[path.length - 1];
      if (prevStep && (prevStep.action.type === 'allin' || prevStep.action.type === 'bet' || prevStep.action.type === 'raise')) {
        isTerminal = true; // All-in called = run out board
      }
      break;
    }
  }

  const newStep: ActionStep = {
    action,
    player: activePlayer,
    potAfter: newPot,
    stackAfter: newStack,
  };

  const newPath = [...path, newStep];
  const nodeId = newPath.map(s => s.action.label.replace(/\s+/g, '_')).join('.');

  // If advancing street, insert a "Deal" step
  if (newStreet !== street && !isTerminal) {
    const dealStep: ActionStep = {
      action: { label: newStreet === 'turn' ? 'Turn' : 'River', type: 'check' },
      player: 'Deal',
      potAfter: newPot,
      stackAfter: newStack,
    };
    newPath.push(dealStep);
  }

  // Check if we're advancing to a new street (need a card dealt)
  const advancedStreet = newStreet !== street && !isTerminal;
  const awaitingDeal = advancedStreet
    ? (newStreet === 'turn' ? 'turn' as const : 'river' as const)
    : undefined;

  return {
    id: nodeId || 'child',
    activePlayer: newPlayer,
    street: newStreet,
    pot: newPot,
    effectiveStack: newStack,
    actions: (isTerminal || awaitingDeal) ? [] : getAvailableActions(newPot, newStack, newStreet, facingBet, sizing),
    isTerminal,
    path: newPath,
    awaitingDeal,
  };
}

/**
 * Get a human-readable label for the node's context.
 */
export function getNodeContextLabel(node: GameTreeNode): string {
  if (node.isTerminal) {
    const lastAction = node.path[node.path.length - 1];
    if (lastAction?.action.type === 'fold') {
      return `${lastAction.player} folds — ${lastAction.player === 'OOP' ? 'IP' : 'OOP'} wins`;
    }
    return 'Showdown';
  }
  if (node.awaitingDeal) {
    return `Deal ${node.awaitingDeal === 'turn' ? 'Turn' : 'River'} card`;
  }
  return `${node.activePlayer} to act`;
}

/**
 * Transition a node from "awaiting deal" to active play.
 * Called after the user selects a Turn or River card.
 * The node gets populated with available actions for OOP to act.
 */
export function dealCard(
  node: GameTreeNode,
  sizingKey: string = 'standard',
): GameTreeNode {
  if (!node.awaitingDeal) return node;

  const sizing = BET_SIZINGS[sizingKey] || BET_SIZINGS.standard;

  return {
    ...node,
    awaitingDeal: undefined,
    actions: getAvailableActions(node.pot, node.effectiveStack, node.street, 0, sizing),
  };
}
