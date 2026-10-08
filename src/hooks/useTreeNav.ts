/**
 * Engine-driven tree navigation (2026-10-06 audit).
 *
 * The UI used to keep its own copy of the betting tree (lib/gameTree.ts):
 * one shared stack for both players, its own raise formula, a "Deal" step
 * that counted as a check, labels the engine did not use — so it showed nodes
 * other than the ones solved, and every street change re-solved from the
 * preflop ranges while replaying the flop actions onto the new board.
 *
 * Now every node comes from the engine: the solve's strategy-tree cache
 * first, then the solve kept in memory (`query_node`). A street whose runouts
 * the solve collapsed (turn/river approximated) is re-solved from the exact
 * ranges at its chance node (`query_ranges`) — a new segment of the line.
 */
import { useCallback, useRef, useState } from 'react';
import type {
  ChanceRanges, EngineAction, NodeView, RunoutOption, SolverRequest, SolverResponse,
} from '../lib/poker';
import { isTauri } from '../lib/tauriEnv';

export interface LineStep {
  /** Engine label, e.g. "Bet_75". */
  label: string;
  actor: 'OOP' | 'IP';
  action: EngineAction;
  /** The card dealt after this action ended the street. */
  card?: string;
  /** That card is the chance node's first runout (cache keys omit it). */
  cardIsFirst?: boolean;
}

/** One solve along the line: the first one, then later-street re-solves. */
export interface Segment {
  response: SolverResponse;
  request: SolverRequest;
  /** Line index of this solve's root (0 for the first solve). */
  start: number;
}

export interface AwaitingCard {
  street: 'turn' | 'river';
  /** Canonical runout cards; empty when the solve collapsed this chance. */
  runouts: RunoutOption[];
  /** No card was solved here: dealing one re-solves the street. */
  collapsed: boolean;
  /** A presolve pack ends here: the re-solve starts from its stored ranges. */
  packed?: boolean;
}

type Solve = (request: SolverRequest) => Promise<SolverResponse | null>;

function stepToken(s: LineStep, forCache: boolean): string {
  if (!s.card || (forCache && s.cardIsFirst)) return s.label;
  return `${s.label}#${s.card}`;
}

/** Engine history of `steps`. The cache form omits a chance's first card
 *  (the strategy tree keys the lex-min runout without a suffix). */
export function historyOf(steps: LineStep[], forCache = false): string {
  return steps.map(s => stepToken(s, forCache)).join(',');
}

/** The view of a solve's root. */
export function rootView(resp: SolverResponse): NodeView {
  const e = resp.strategy_tree?.[''];
  if (e && e.kind) return e as unknown as NodeView;
  const n = resp.node;
  return {
    kind: n?.kind ?? 'player',
    street: n?.street ?? 0,
    pot: n?.pot ?? 0,
    stack_oop: n?.stack_oop ?? 0,
    stack_ip: n?.stack_ip ?? 0,
    to_call: n?.to_call ?? 0,
    board: n?.board ?? [],
    actions: n?.actions ?? [],
    runouts: n?.runouts ?? [],
    acting: resp.acting_player ?? 'OOP',
    global_strategy: resp.global_strategy,
    combo_strategies: resp.combo_strategies,
    opponent_side: resp.opponent_side,
    opponent_range: resp.opponent_range,
    combo_evs: resp.combo_evs,
  };
}

async function queryNode(sessionId: number, history: string): Promise<NodeView> {
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke<NodeView>('query_node', { sessionId, history });
}

function hasSession(seg: Segment): boolean {
  return seg.response.session_id != null && isTauri();
}

/** The node after `steps` (relative to the segment root). */
async function loadNode(seg: Segment, steps: LineStep[]): Promise<NodeView | null> {
  if (steps.length === 0) return rootView(seg.response);
  const last = steps[steps.length - 1];
  if (last.action.next === 'terminal') {
    const before = await loadNode(seg, steps.slice(0, -1));
    return before ? terminalView(before, last) : null;
  }
  const entry = seg.response.strategy_tree?.[historyOf(steps, true)];
  if (entry && entry.kind) return entry as unknown as NodeView;
  if (hasSession(seg)) return queryNode(seg.response.session_id!, historyOf(steps));
  return null;
}

/** A terminal, from the node its action was taken at (no query needed). */
function terminalView(before: NodeView, last: LineStep): NodeView {
  const fold = last.action.type === 'fold';
  return {
    kind: 'terminal', street: before.street, board: before.board,
    pot: before.pot + (fold ? 0 : last.action.amount),
    stack_oop: 0, stack_ip: 0, to_call: 0, actions: [], runouts: [],
    terminal: fold ? (last.actor === 'OOP' ? 'fold_oop' : 'fold_ip') : 'showdown',
  };
}

/** The chance a street-ending last step leads to, and the node it was
 *  taken at (which says the street). */
async function loadChanceAfter(seg: Segment, steps: LineStep[]) {
  const before = await loadNode(seg, steps.slice(0, -1));
  if (!before) return null;
  const a = await loadChance(seg, steps, before.street === 0 ? 'turn' : 'river');
  return a ? { before, awaiting: a } : null;
}

/** The chance node a street-ending last step leads to. */
async function loadChance(seg: Segment, steps: LineStep[], street: 'turn' | 'river'): Promise<AwaitingCard | null> {
  // The cache key without a card is the FIRST runout's node; its
  // runout_options are this chance's cards (none when it was collapsed).
  const entry = seg.response.strategy_tree?.[historyOf(steps, true)];
  if (entry && entry.kind) {
    const runouts = entry.runout_options ?? [];
    return { street, runouts, collapsed: runouts.length === 0 };
  }
  if (hasSession(seg)) {
    const reply = await queryNode(seg.response.session_id!, historyOf(steps));
    if (reply.kind !== 'chance') return null;
    return { street, runouts: reply.runouts, collapsed: reply.runouts.length === 0 };
  }
  if (seg.response.chance_ranges?.[historyOf(steps)]) {
    return { street, runouts: [], collapsed: true, packed: true };
  }
  return null;
}

async function closeSessions(segs: Segment[]) {
  if (!isTauri()) return;
  const { invoke } = await import('@tauri-apps/api/core');
  for (const s of segs) {
    if (s.response.session_id != null) {
      invoke('close_session', { sessionId: s.response.session_id }).catch(() => {});
    }
  }
}

const NOT_AVAILABLE =
  'This node is not in the solved tree that is still available — solve again to explore it.';

export function useTreeNav(solve: Solve) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [line, setLine] = useState<LineStep[]>([]);
  const [view, setView] = useState<NodeView | null>(null);
  const [awaiting, setAwaiting] = useState<AwaitingCard | null>(null);
  const [navError, setNavError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Latest state for the async handlers (two quick clicks must not both
  // start from the pre-click line).
  const stateRef = useRef({ segments, line, view, awaiting });
  stateRef.current = { segments, line, view, awaiting };

  const commit = useCallback((segs: Segment[], steps: LineStep[], v: NodeView | null, a: AwaitingCard | null) => {
    stateRef.current = { segments: segs, line: steps, view: v, awaiting: a };
    setSegments(segs);
    setLine(steps);
    setView(v);
    setAwaiting(a);
    setNavError(null);
  }, []);

  const run = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setNavError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  /** A fresh solve becomes the whole line. */
  const start = useCallback((response: SolverResponse, request: SolverRequest) => {
    closeSessions(stateRef.current.segments.filter(s => s.response !== response));
    commit([{ response, request, start: 0 }], [], rootView(response), null);
  }, [commit]);

  const clear = useCallback(() => {
    closeSessions(stateRef.current.segments);
    commit([], [], null, null);
  }, [commit]);

  /** Free the in-memory solves but keep the line navigable from the
   *  strategy-tree cache: a new root solve needs that RAM (a session holds
   *  its whole solved tree). */
  const releaseSessions = useCallback(() => {
    const segs = stateRef.current.segments;
    if (!segs.some(s => s.response.session_id != null)) return;
    closeSessions(segs);
    const stripped = segs.map(s => ({ ...s, response: { ...s.response, session_id: undefined } }));
    stateRef.current = { ...stateRef.current, segments: stripped };
    setSegments(stripped);
  }, []);

  const act = useCallback((action: EngineAction) => run(async () => {
    const { segments: segs, line: steps, view: v } = stateRef.current;
    if (!v || v.kind !== 'player' || segs.length === 0) return;
    const seg = segs[segs.length - 1];
    const next = [...steps, { label: action.label, actor: v.acting ?? 'OOP', action }];
    const rel = next.slice(seg.start);
    if (action.next === 'chance') {
      const a = await loadChance(seg, rel, v.street === 0 ? 'turn' : 'river');
      if (!a) throw new Error(NOT_AVAILABLE);
      commit(segs, next, v, a);
      return;
    }
    const node = await loadNode(seg, rel);
    if (!node) throw new Error(NOT_AVAILABLE);
    commit(segs, next, node, null);
  }), [run, commit]);

  /** Deal the awaited card. A collapsed chance re-solves the new street from
   *  the exact ranges there, with the same menus and solve settings. */
  const deal = useCallback((card: string) => run(async () => {
    const { segments: segs, line: steps, awaiting: a, view: v } = stateRef.current;
    if (!a || steps.length === 0 || segs.length === 0) return;
    const seg = segs[segs.length - 1];
    const lastIdx = steps.length - 1;
    const dealt: LineStep = {
      ...steps[lastIdx], card, cardIsFirst: a.runouts.length > 0 && a.runouts[0].card === card,
    };
    const next = [...steps.slice(0, lastIdx), dealt];
    if (!a.collapsed) {
      const node = await loadNode(seg, next.slice(seg.start));
      if (!node) throw new Error(NOT_AVAILABLE);
      commit(segs, next, node, null);
      return;
    }
    const history = historyOf(steps.slice(seg.start));
    const packed = seg.response.chance_ranges?.[history];
    if (!packed && !hasSession(seg)) {
      throw new Error('Re-solving the next street needs the solve in memory — solve the spot again.');
    }
    const { invoke } = await import('@tauri-apps/api/core');
    const ranges = packed ?? await invoke<ChanceRanges>(
      'query_ranges', { sessionId: seg.response.session_id, history });
    if (!ranges.oop || !ranges.ip) {
      throw new Error('One player never reaches this point of the line — there is nothing to re-solve.');
    }
    const dealtBefore = steps.slice(seg.start).filter(s => s.card).map(s => s.card).join('');
    const request: SolverRequest = {
      ...seg.request,
      board: seg.request.board + dealtBefore + card,
      pot_size: ranges.pot,
      effective_stack: ranges.stack,
      oop_range: ranges.oop,
      ip_range: ranges.ip,
      oop_has_initiative: ranges.oop_has_initiative,
      node_locks: undefined,
      history: undefined,
      target_combo: undefined,
      target_player: undefined,
      serve: true,
    };
    const resp = await solve(request);
    if (!resp) return;   // cancelled or failed: still waiting for the card
    const cur = stateRef.current;
    if (cur.line !== steps || cur.view !== v) return;   // navigated meanwhile
    commit([...segs, { response: resp, request, start: next.length }], next, rootView(resp), null);
  }), [run, commit, solve]);

  /** Go to the node after step `index` (with its card); -1 = the root. */
  const back = useCallback((index: number) => run(async () => {
    const { segments: segs, line: steps } = stateRef.current;
    if (segs.length === 0) return;
    const kept = steps.slice(0, index + 1);
    const keep = segs.filter(s => s.start <= kept.length);
    closeSessions(segs.filter(s => s.start > kept.length));
    const seg = keep[keep.length - 1];
    const last = kept[kept.length - 1];
    if (last && last.action.next === 'chance' && !last.card) {
      const c = await loadChanceAfter(seg, kept.slice(seg.start));
      if (!c) throw new Error(NOT_AVAILABLE);
      commit(keep, kept, c.before, c.awaiting);
      return;
    }
    const node = await loadNode(seg, kept.slice(seg.start));
    if (!node) throw new Error(NOT_AVAILABLE);
    commit(keep, kept, node, null);
  }), [run, commit]);

  /** Back to the deal of step `index`'s card (pick another card). */
  const redeal = useCallback((index: number) => run(async () => {
    const { segments: segs, line: steps } = stateRef.current;
    const step = steps[index];
    if (!step || !step.card) return;
    const kept = [...steps.slice(0, index), { ...step, card: undefined, cardIsFirst: undefined }];
    const keep = segs.filter(s => s.start <= index);
    closeSessions(segs.filter(s => s.start > index));
    const seg = keep[keep.length - 1];
    const c = await loadChanceAfter(seg, kept.slice(seg.start));
    if (!c) throw new Error(NOT_AVAILABLE);
    commit(keep, kept, c.before, c.awaiting);
  }), [run, commit]);

  /** Re-solve the current segment with changed settings (node locks) and
   *  return to the same node. Later segments cannot exist: locks are set on
   *  the node shown, which is in the last segment. */
  const resolveSegment = useCallback((patch: Partial<SolverRequest>) => run(async () => {
    const { segments: segs, line: steps, awaiting: a } = stateRef.current;
    if (segs.length === 0) return;
    const seg = segs[segs.length - 1];
    const request: SolverRequest = { ...seg.request, ...patch, serve: true };
    const resp = await solve(request);
    if (!resp) return;
    const fresh: Segment = { response: resp, request, start: seg.start };
    closeSessions([seg]);
    const rel = steps.slice(seg.start);
    if (a) {
      const c = await loadChanceAfter(fresh, rel);
      if (!c) throw new Error(NOT_AVAILABLE);
      commit([...segs.slice(0, -1), fresh], steps, c.before, c.awaiting);
      return;
    }
    const node = await loadNode(fresh, rel);
    if (!node) throw new Error(NOT_AVAILABLE);
    commit([...segs.slice(0, -1), fresh], steps, node, null);
  }), [run, commit, solve]);

  const segment = segments.length > 0 ? segments[segments.length - 1] : null;
  /** Session-form history of the shown node within its segment. */
  const segmentHistory = segment ? historyOf(line.slice(segment.start)) : '';
  /** Cache-form of the same (strategy-tree keys). */
  const segmentCacheKey = segment ? historyOf(line.slice(segment.start), true) : '';
  /** Root board + every card dealt so far, in deal order. */
  const boardCards: string[] = [];
  if (segments.length > 0) {
    const b = segments[0].request.board;
    for (let i = 0; i + 1 < b.length; i += 2) boardCards.push(b.slice(i, i + 2));
    for (const s of line) if (s.card) boardCards.push(s.card);
  }

  return {
    segments, segment, line, view, awaiting, navError, busy,
    segmentHistory, segmentCacheKey, boardCards,
    start, clear, releaseSessions, act, deal, back, redeal, resolveSegment,
    clearError: () => setNavError(null),
  };
}
