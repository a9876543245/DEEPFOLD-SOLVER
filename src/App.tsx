import { useState, useCallback, useMemo, useEffect } from 'react';
import { BoardSelector } from './components/BoardSelector';
import { PositionSelector } from './components/PositionSelector';
import { SolverControls } from './components/SolverControls';
import { SolveEtaBanner } from './components/SolveEtaBanner';
import { RangeGrid } from './components/RangeGrid';
import type { GridDisplayMode } from './components/RangeGrid';
import { StrategyPanel } from './components/StrategyPanel';
import { ActionNavigator } from './components/ActionNavigator';
import { ActionBar, prettyAction } from './components/ActionBar';
import { TurnRiverCardSelector } from './components/TurnRiverCardSelector';
import { BetSizingEditor } from './components/BetSizingEditor';
import { useSolver } from './hooks/useSolver';
import { useTreeNav } from './hooks/useTreeNav';
import type {
  SolverRequest, SolverResponse, NodeLock, ComboAnalysis, GameContext, MemoryProfile,
  SolveMode, EstimateResponse, EngineAction,
} from './lib/poker';
import { SOLVE_MODE_PRESETS, DECOMPOSE_PRESETS } from './lib/poker';
import type { Position, PositionMatchup } from './lib/ranges';
import { derivePotStack } from './lib/ranges';
import { presetSpec, specToEngineJson, type BetSizingSpec } from './lib/betSizing';
import { RangeEditorModal } from './components/RangeEditorModal';
import { NodeLockEditor } from './components/NodeLockEditor';
import { GuideModal } from './components/GuideModal';
import { AuthGate } from './components/AuthGate';
import { SpotLibrary } from './components/SpotLibrary';
import { BackendIndicator } from './components/BackendIndicator';
import { UpdateBanner } from './components/UpdateBanner';
import { isRealSolverAvailable } from './lib/presolvedSpots';
import { DrillMode } from './components/DrillMode';
import { GtoChartBrowser } from './components/GtoChartBrowser';
import { GameContextSelector } from './components/GameContextSelector';

/**
 * Feature flag — temporarily disables the entire GTO Chart Library surface
 * (header button, browser modal, GameContextSelector, auto-load hook,
 * "GTO ranges loaded" disclosure banner).
 *
 * Why disabled: two independent poker pros audited the bundled
 * `gto_output/` chart data and found extraction errors in ~50-75% of
 * sampled charts (premium hands folding, trash hands raising, missing
 * actions, scenario/action mismatches). The OCR/extraction pipeline that
 * generates the JSON from the source spreadsheet needs to be fixed first.
 *
 * Set back to `true` once the chart data passes the 5-rule sanity check
 * documented in STATUS-AND-ROADMAP.md.
 */
// Re-enabled: gto_output is now generated from imported text ranges.
const GTO_CHART_LIBRARY_ENABLED = true;
import { useGtoAutoRange } from './hooks/useGtoAutoRange';
import { HelpCircle, BookOpen, Crosshair } from 'lucide-react';
import { useT, useLanguage, LANGUAGES } from './lib/i18n';
import type { AuthUser } from './lib/auth';

const SUIT_MAP: Record<string, { symbol: string; color: string }> = {
  s: { symbol: '♠', color: '#E8E8E8' },
  h: { symbol: '♥', color: '#FF453A' },
  d: { symbol: '♦', color: '#0A84FF' },
  c: { symbol: '♣', color: '#30D158' },
};

function App() {
  const t = useT();
  const { lang, setLang } = useLanguage();
  const [, setAuthUser] = useState<AuthUser | null>(null);
  // Core state
  // The board the solve starts on (3-5 cards, e.g. "AsKd7c"). Later cards are
  // dealt along the line (useTreeNav), never edited into this string.
  const [rootBoard, setRootBoard] = useState('');
  const [pot, setPot] = useState(100);
  const [stack, setStack] = useState(500);
  const [iterations, setIterations] = useState(300);
  const [hoveredCombo, setHoveredCombo] = useState<string | null>(null);
  const [selectedMatchup, setSelectedMatchup] = useState<PositionMatchup | null>(null);
  const [heroPosition, setHeroPosition] = useState<Position | null>(null);

  // Advanced solver state
  const [customIpRange, setCustomIpRange] = useState<string | null>(null);
  const [customOopRange, setCustomOopRange] = useState<string | null>(null);
  // Locks of the next root solve. Locks set on a later-street re-solve live
  // in that solve's request (useTreeNav segment).
  const [nodeLocks, setNodeLocks] = useState<NodeLock[]>([]);
  const [gtoBrowserOpen, setGtoBrowserOpen] = useState(false);

  // Game context — drives which preflop chart bucket auto-loads as default
  // ranges. Default = Cash 6max 100bb (matches the legacy MATCHUPS dataset).
  const [gameContext, setGameContext] = useState<GameContext>({
    gameType: 'Cash',
    scenarioType: '6max_100bb_2_5x_500rake',
    effectiveBB: 100,
  });

  // Auto-load GTO preflop ranges matching (game context × matchup).
  // Disabled while GTO_CHART_LIBRARY_ENABLED is false — pass null matchup
  // so the hook short-circuits without applying ranges or populating the
  // disclosure banner. Hardcoded MATCHUPS defaults are used instead.
  const { applied: appliedGtoRanges } = useGtoAutoRange(
    GTO_CHART_LIBRARY_ENABLED ? selectedMatchup : null,
    gameContext, setCustomIpRange, setCustomOopRange);

  // When the user changes the effective stack via GameContextSelector,
  // re-derive pot+stack from the *currently selected matchup* (not from a
  // hard-coded 5.5bb/22bb table). This preserves position-pair-specific
  // pot values — BTN vs BB 3BP and SB vs BB 3BP have different pot sizes
  // and the matchup data already knows that.
  //
  // No matchup selected → no auto-derivation; leave the manual defaults
  // (pot=100, stack=500) alone so the user can still solve a free spot
  // without picking a matchup first.
  useEffect(() => {
    if (gameContext.effectiveBB == null || !selectedMatchup) return;
    const { potChips, stackChips } = derivePotStack(selectedMatchup, gameContext.effectiveBB);
    setPot(potChips);
    setStack(stackChips);
  }, [gameContext.effectiveBB, selectedMatchup]);

  // Off-range analyses already solved, keyed by solve + node + hand: a
  // re-click is instant.
  const [offRangeCache, setOffRangeCache] = useState<Record<string, ComboAnalysis>>({});
  // The clicked hand's analysis at the node shown (cleared on navigation).
  const [targetAnalysis, setTargetAnalysis] = useState<ComboAnalysis | null>(null);

  // Grid display mode
  const [gridMode, setGridMode] = useState<GridDisplayMode>('mix');
  const [heatmapAction, setHeatmapAction] = useState<string>('');
  const [gridViewSide, setGridViewSide] = useState<'acting' | 'opponent'>('acting');

  // 2026-10-06: bet sizing menus (a preset or a custom Pio-style spec). The
  // engine builds the tree from it and the UI shows the engine's own nodes,
  // so the two can no longer disagree.
  const [sizingSpec, setSizingSpec] = useState<BetSizingSpec>(() => presetSpec('standard'));
  const [sizingEditorOpen, setSizingEditorOpen] = useState(false);
  // Suit isomorphism: 'exact' (Pio-style, default) or 'fast'.
  const [isoMode, setIsoMode] = useState<'exact' | 'fast'>('exact');

  // Memory profile preset for the next solve (Polish #1). Default 'balanced'
  // matches the engine's `--memory-profile` default and the Rust resolver in
  // src-tauri/src/types.rs::ResolvedMemoryBudget::from_profile.
  const [memoryProfile, setMemoryProfile] = useState<MemoryProfile>('balanced');

  // v1.3.0: solve mode preset (Quick/Standard/Deep). Bundles iter cap +
  // time budget. Threaded into buildRequest so the engine respects the
  // budget. Default 'standard' = 300 iter / 5 min budget, the right
  // balance for most spots.
  const [solveMode, setSolveMode] = useState<SolveMode>('standard');

  // Stage 5: runout decomposition toggle. 'off' = legacy (turn/river equity
  // approximated on rainbow boards too large to enumerate). 'auto' = solve
  // real turn/river runouts via flop-trunk + per-turn-card subgame
  // decomposition (slower, no approximation). Orthogonal to solveMode.
  const [decomposeRunouts, setDecomposeRunouts] = useState<'off' | 'auto'>('off');
  // Roadmap ④: Exact feasibility pre-flight. When Exact is selected, ask the
  // engine (--estimate-only) what a decomposed solve of the CURRENT config
  // costs — leaves, ETA, SPR-keyed expected-accuracy band — so the toggle
  // hint prices the commitment BEFORE the user hits Solve.
  const [exactPreflight, setExactPreflight] = useState<EstimateResponse | null>(null);

  // Modals
  const [showGuide, setShowGuide] = useState(false);
  const [showSpotLibrary, setShowSpotLibrary] = useState(false);
  const [showDrill, setShowDrill] = useState(false);
  const [editingRange, setEditingRange] = useState<'IP' | 'OOP' | null>(null);
  const [editingNodeLock, setEditingNodeLock] = useState<{ combo: string; actions: string[]; initialStrategy?: Record<string, number> } | null>(null);

  const { loading, error, setError, elapsed, progress, estimate, solve, reset } = useSolver();
  const nav = useTreeNav(solve);
  const { view, segment, line, awaiting, busy: navBusy } = nav;
  const { act, deal, back, redeal, start, clear, releaseSessions, resolveSegment } = nav;

  // A new node: the clicked hand's analysis belonged to the old one.
  useEffect(() => { setTargetAnalysis(null); }, [view, awaiting]);

  // Determine hero's range
  const getHeroRange = useCallback(() => {
    if (!selectedMatchup || !heroPosition) return undefined;
    const isIPHero = selectedMatchup.ip === heroPosition;
    if (isIPHero) return customIpRange ?? selectedMatchup.ipRange;
    return customOopRange ?? selectedMatchup.oopRange;
  }, [selectedMatchup, heroPosition, customIpRange, customOopRange]);

  // Auto-set pot/stack from matchup defaults, scaled to the current
  // effective stack depth (falls back to 100BB if the GameContext has no
  // stack pinned — same as the matchup's encoded default).
  const handleMatchupChange = useCallback((matchup: PositionMatchup, heroPos: Position) => {
    setSelectedMatchup(matchup);
    setHeroPosition(heroPos);
    const { potChips, stackChips } = derivePotStack(matchup, gameContext.effectiveBB ?? 100);
    setPot(potChips);
    setStack(stackChips);
    // Reset tree on matchup change
    setCustomIpRange(null);
    setCustomOopRange(null);
    setNodeLocks([]);
    clear();
    reset();
  }, [reset, clear, gameContext.effectiveBB]);

  /** The request of a root solve with the current settings. */
  const buildRequest = useCallback((over?: {
    board?: string; pot?: number; stack?: number; ipRange?: string; oopRange?: string;
    locks?: NodeLock[];
  }): SolverRequest => {
    const locks = over?.locks ?? nodeLocks;
    return {
      board: over?.board ?? rootBoard,
      pot_size: over?.pot ?? pot,
      effective_stack: over?.stack ?? stack,
      iterations,
      // 2026-10-06 audit: the solve mode's target (was a constant 0.5).
      exploitability: SOLVE_MODE_PRESETS[solveMode].exploitability,
      ip_range: over?.ipRange ?? customIpRange ?? selectedMatchup?.ipRange,
      oop_range: over?.oopRange ?? customOopRange ?? selectedMatchup?.oopRange,
      node_locks: locks.length > 0 ? JSON.stringify(locks) : undefined,
      hero_range: getHeroRange(),
      bet_sizing: specToEngineJson(sizingSpec),
      iso: isoMode,
      serve: true,
      memory_profile: memoryProfile,
      // v1.3.0: solve mode picks iter cap + time budget. The user-set
      // `iterations` field above acts as a manual override that wins when
      // the user types a custom value into Advanced settings; otherwise
      // we use the mode preset.
      time_budget_seconds: SOLVE_MODE_PRESETS[solveMode].time_budget_seconds,
      // Stage 5: runout decomposition. 'off' omits the flag (sidecar default);
      // 'auto' solves real turn/river runouts on rainbow boards.
      decompose_runouts: decomposeRunouts,
      // Roadmap ④: Exact iteration presets keyed on solveMode (subgame
      // DEPTH dominates quality per the 2026-07-15 study — presets scale
      // `inner`, keep sweeps minimal).
      ...(decomposeRunouts === 'auto' ? {
        decompose_outer: DECOMPOSE_PRESETS[solveMode].outer,
        decompose_inner: DECOMPOSE_PRESETS[solveMode].inner,
        decompose_trunk_iters: DECOMPOSE_PRESETS[solveMode].trunk_iters,
        decompose_warm_start: DECOMPOSE_PRESETS[solveMode].warm_start,
      } : {}),
      // v1.7.0: GUI defaults to the levelized CPU backend (4-5x faster than
      // reference on a typical 8-thread laptop CPU). cpu_simd='auto' lets
      // CPUID pick AVX2 vs scalar at startup, cpu_threads=0 means "use
      // every available core".
      cpu_backend: 'levelized',
      cpu_simd: 'auto',
      cpu_threads: 0,
    };
  }, [rootBoard, pot, stack, iterations, selectedMatchup, getHeroRange, customIpRange, customOopRange, nodeLocks, sizingSpec, isoMode, memoryProfile, solveMode, decomposeRunouts]);

  // Roadmap ④: debounced pre-commit estimate for Exact mode. Fires when the
  // Exact pill is on and any solve-shaping config changes; skipped outside
  // Tauri (browser preview can't invoke the sidecar) and while a solve runs.
  useEffect(() => {
    if (decomposeRunouts !== 'auto' || loading) return;
    if (rootBoard.length < 6) { setExactPreflight(null); return; }
    if (!(window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__) return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const request = buildRequest();
        const est = await invoke<EstimateResponse>('estimate_solve', { request });
        if (!cancelled) setExactPreflight(est);
      } catch (e) {
        // Non-fatal: mark as failed (non-null, no decompose block) so the
        // hint shows the static SPR copy instead of "Pricing…" forever.
        console.warn('exact preflight estimate failed (non-fatal):', e);
        if (!cancelled) {
          setExactPreflight({ status: 'error', resources: {} } as EstimateResponse);
        }
      }
    }, 600);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [decomposeRunouts, loading, rootBoard, buildRequest]);

  /** Solve `request` and make it the line shown. */
  const solveRoot = useCallback(async (request: SolverRequest) => {
    releaseSessions();
    const resp = await solve(request);
    if (resp) {
      setOffRangeCache({});
      start(resp, request);
    }
  }, [solve, start, releaseSessions]);

  // Initial solve (root node)
  const handleSolve = useCallback(() => {
    if (rootBoard.length < 6 || loading) return;
    solveRoot(buildRequest());
  }, [rootBoard, loading, solveRoot, buildRequest]);

  const handleAction = useCallback((action: EngineAction) => {
    if (loading || navBusy) return;
    act(action);
  }, [loading, navBusy, act]);

  const handleDealCard = useCallback((card: string) => {
    if (loading || navBusy) return;
    deal(card);
  }, [loading, navBusy, deal]);

  /** Breadcrumbs: every action, plus every dealt card as its own crumb. */
  const crumbs = useMemo(() => {
    const out: { label: string; player: 'OOP' | 'IP' | 'Deal'; step: number; card: boolean }[] = [];
    line.forEach((s, i) => {
      out.push({ label: prettyAction(s.label), player: s.actor, step: i, card: false });
      if (s.card) out.push({ label: s.card, player: 'Deal', step: i, card: true });
    });
    return out;
  }, [line]);

  const handleNavigate = useCallback((index: number) => {
    if (loading || navBusy) return;
    if (index < 0) { back(-1); return; }
    const c = crumbs[index];
    if (!c) return;
    // An action that ended a street: back to its deal (pick another card).
    if (!c.card && line[c.step]?.card) redeal(c.step);
    else back(c.step);
  }, [loading, navBusy, back, redeal, crumbs, line]);

  // Minimal keyboard study layer: ←/Backspace steps back one node along the
  // line, 1-9 takes the Nth action at the current node (walk the line without
  // the mouse), S = strategy-mix grid, E = EV grid. Ignored while typing in a
  // field, while any modal owns the screen, or while a solve is in flight.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
      if (showGuide || showSpotLibrary || showDrill || gtoBrowserOpen || editingRange || editingNodeLock || sizingEditorOpen) return;
      if (loading || navBusy) return;

      // 1-9 → take the Nth available action at the current node.
      if (e.key >= '1' && e.key <= '9') {
        const idx = e.key.charCodeAt(0) - '1'.charCodeAt(0);
        if (!awaiting && view?.kind === 'player' && idx < view.actions.length) {
          e.preventDefault();
          act(view.actions[idx]);
        }
        return;
      }

      switch (e.key) {
        case 'ArrowLeft':
        case 'Backspace':
          if (line.length > 0) {
            e.preventDefault();
            const last = line.length - 1;
            if (line[last].card) redeal(last);   // back to that deal
            else back(last - 1);
          }
          break;
        case 's': case 'S':
          setGridMode('mix');
          break;
        case 'e': case 'E':
          setGridMode('ev');
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [view, awaiting, line, act, back, redeal, loading, navBusy, showGuide, showSpotLibrary, showDrill, gtoBrowserOpen, editingRange, editingNodeLock, sizingEditorOpen]);

  // Click on a combo cell. In range at this node: its strategy, instantly.
  // Out of range: a solve with the hand added to the acting player's range
  // (tiny weight) answers what it would do here.
  const handleCellClick = useCallback(async (label: string) => {
    if (!view || view.kind !== 'player' || awaiting || !segment || loading) return;
    const strategy = view.combo_strategies?.[label];
    if (strategy && !strategy['Not in range']) {
      const analysis: ComboAnalysis = {
        combo: label, best_action: '', ev: view.combo_evs?.[label] ?? 0, strategy_mix: {},
      };
      let bestFreq = 0;
      for (const [action, freq] of Object.entries(strategy)) {
        if (action === 'Not in range') continue;
        if (freq > 0.01) analysis.strategy_mix[action] = `${(freq * 100).toFixed(1)}%`;
        if (freq > bestFreq) { bestFreq = freq; analysis.best_action = action; }
      }
      setTargetAnalysis(analysis);
      return;
    }
    const key = `${segment.response.session_id ?? segment.request.board}|${nav.segmentHistory}|${label}`;
    const cached = offRangeCache[key];
    if (cached) { setTargetAnalysis(cached); return; }
    const request: SolverRequest = {
      ...segment.request,
      history: nav.segmentHistory || undefined,
      target_combo: label,
      target_player: view.acting === 'IP' ? 'ip' : 'oop',
      serve: false,
      strategy_tree_max_nodes: 1,
    };
    const resp = await solve(request);
    const a = resp?.target_combo_analysis;
    if (a) {
      setOffRangeCache(prev => ({ ...prev, [key]: a }));
      setTargetAnalysis(a);
    }
  }, [view, awaiting, segment, loading, nav.segmentHistory, offRangeCache, solve]);

  // What the grid and the strategy panel show: the solve's response with the
  // current node's fields.
  const displayResult = useMemo<SolverResponse | null>(() => {
    if (!segment || !view) return null;
    return {
      ...segment.response,
      global_strategy: view.global_strategy ?? {},
      combo_strategies: view.combo_strategies,
      acting_player: view.acting,
      opponent_side: view.opponent_side,
      opponent_range: view.opponent_range,
      combo_evs: view.combo_evs,
      target_combo_analysis: targetAnalysis ?? undefined,
    };
  }, [segment, view, targetAnalysis]);

  // Locks of the solve shown (root: the nodeLocks state; later streets:
  // their own request).
  const segmentLocks = useMemo<NodeLock[]>(() => {
    if (!segment?.request.node_locks) return [];
    try { return JSON.parse(segment.request.node_locks) as NodeLock[]; } catch { return []; }
  }, [segment]);

  // Position labels
  const heroIsIP = selectedMatchup && heroPosition ? selectedMatchup.ip === heroPosition : null;
  const villainPosition = selectedMatchup && heroPosition
    ? (selectedMatchup.ip === heroPosition ? selectedMatchup.oop : selectedMatchup.ip)
    : null;

  const boardString = nav.boardCards.join('');
  const hasSolved = !!segment;

  return (
    <AuthGate onAuth={setAuthUser}>
    <div className="app-layout">
      {/* Header */}
      <header className="header-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16, fontWeight: 800,
          }}>D</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.01em' }}>{t('app.title')}</div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: -2 }}>
              {t('app.subtitle')}
            </div>
          </div>
        </div>

        {/* Breadcrumb Navigator */}
        <ActionNavigator history={crumbs} onNavigate={handleNavigate} />

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <BackendIndicator />
          {selectedMatchup && heroPosition && villainPosition && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 10px', background: 'var(--color-glass)',
              borderRadius: 'var(--radius-full)', fontSize: 12, fontWeight: 600,
            }}>
              <span style={{ color: heroIsIP ? '#69DB7C' : '#B197FC' }}>{heroPosition}</span>
              <span style={{ color: 'var(--color-text-tertiary)', fontSize: 10 }}>vs</span>
              <span style={{ color: heroIsIP ? '#B197FC' : '#69DB7C' }}>{villainPosition}</span>
              {selectedMatchup.potType === '3BET' && (
                <span style={{
                  fontSize: 8, fontWeight: 700, color: 'var(--color-purple)',
                  background: 'rgba(191,90,242,0.2)', padding: '1px 4px', borderRadius: 3,
                }}>3BP</span>
              )}
              <span style={{
                fontSize: 9, fontWeight: 700,
                color: heroIsIP ? 'var(--color-green)' : 'var(--color-orange)',
                marginLeft: 2,
              }}>{heroIsIP ? 'IP' : 'OOP'}</span>
            </div>
          )}
          {loading && progress && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
              <div style={{
                width: 80, height: 4, background: 'var(--color-bg-tertiary)',
                borderRadius: 'var(--radius-full)', overflow: 'hidden',
              }}>
                <div className="progress-bar-fill" style={{
                  height: '100%', width: `${progress.pct}%`,
                  borderRadius: 'var(--radius-full)',
                  transition: 'width 150ms ease-out',
                }} />
              </div>
              <span style={{ color: 'var(--color-text-secondary)', fontSize: 11 }}>
                {Math.round(progress.pct)}%
              </span>
            </div>
          )}
          {/* Spot Library button */}
          <button onClick={() => setShowSpotLibrary(true)} title={t('spots.title')}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
              borderRadius: 'var(--radius-full)', border: 'none', cursor: 'pointer',
              background: 'var(--color-glass)', color: 'var(--color-text-secondary)',
              fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
            }}
            onMouseOver={e => { e.currentTarget.style.background = 'var(--color-glass-hover)'; e.currentTarget.style.color = '#fff'; }}
            onMouseOut={e => { e.currentTarget.style.background = 'var(--color-glass)'; e.currentTarget.style.color = 'var(--color-text-secondary)'; }}
          >
            <BookOpen size={13} /> {t('spots.title')}
          </button>

          {/* Drill button */}
          <button onClick={() => setShowDrill(true)} title={t('drill.start')}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
              borderRadius: 'var(--radius-full)', border: 'none', cursor: 'pointer',
              background: 'linear-gradient(135deg, rgba(48,209,88,0.15), rgba(10,132,255,0.15))',
              color: 'var(--color-green)', fontSize: 11, fontWeight: 700,
              fontFamily: 'inherit', transition: 'all 150ms ease',
            }}
            onMouseOver={e => { e.currentTarget.style.background = 'var(--color-green)'; e.currentTarget.style.color = '#000'; }}
            onMouseOut={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(48,209,88,0.15), rgba(10,132,255,0.15))'; e.currentTarget.style.color = 'var(--color-green)'; }}
          >
            <Crosshair size={13} /> {t('drill.start')}
          </button>

          {/* GTO Chart Library button — disabled while chart data fails audit */}
          {GTO_CHART_LIBRARY_ENABLED && (
            <button onClick={() => setGtoBrowserOpen(true)} title="GTO Preflop Chart Library"
              style={{
                display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
                borderRadius: 'var(--radius-full)', border: 'none', cursor: 'pointer',
                background: 'var(--color-glass)', color: 'var(--color-text-secondary)',
                fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
              }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--color-glass-hover)'; e.currentTarget.style.color = '#fff'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'var(--color-glass)'; e.currentTarget.style.color = 'var(--color-text-secondary)'; }}
            >
              GTO Charts
            </button>
          )}

          {/* Language switcher */}
          <div style={{
            display: 'flex', background: 'var(--color-glass)',
            borderRadius: 'var(--radius-full)', padding: 2, gap: 1,
          }}>
            {LANGUAGES.map(l => (
              <button key={l.code} onClick={() => setLang(l.code)}
                title={l.label}
                style={{
                  width: 26, height: 22, borderRadius: 'var(--radius-full)',
                  border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                  fontSize: 10, fontWeight: 700,
                  background: lang === l.code ? 'var(--color-accent)' : 'transparent',
                  color: lang === l.code ? '#fff' : 'var(--color-text-tertiary)',
                  transition: 'all 150ms ease',
                }}
              >{l.flag}</button>
            ))}
          </div>
          <button onClick={() => setShowGuide(true)} title={t('app.help')}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: 28, height: 28, borderRadius: '50%', border: 'none',
              background: 'var(--color-glass)', cursor: 'pointer',
              color: 'var(--color-text-secondary)', transition: 'all 150ms ease',
            }}
            onMouseOver={e => { e.currentTarget.style.background = 'var(--color-glass-hover)'; e.currentTarget.style.color = '#fff'; }}
            onMouseOut={e => { e.currentTarget.style.background = 'var(--color-glass)'; e.currentTarget.style.color = 'var(--color-text-secondary)'; }}
          >
            <HelpCircle size={15} />
          </button>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: loading ? 'var(--color-orange)' : 'var(--color-green)',
          }} />
        </div>
      </header>

      {/* Update banner (no-op in browser mode) */}
      <UpdateBanner />

      {/* Auto-loaded GTO range disclosure. Shows after a matchup change
       *  when the bundled chart library found a match for IP/OOP. Click
       *  opens the GTO browser so user can pick a different chart.
       *  gridColumn: 1 / -1 forces full-width like the header — without it
       *  the banner eats a grid cell and pushes sidebar/main/right out of
       *  alignment (v1.0.7 layout bug). */}
      {appliedGtoRanges.length > 0 && (
        <div style={{
          gridColumn: '1 / -1',
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 12px', background: 'rgba(124, 58, 237, 0.08)',
          borderBottom: '1px solid rgba(124, 58, 237, 0.2)',
          fontSize: 12,
        }}>
          <span style={{ color: 'var(--color-text-secondary)' }}>GTO ranges loaded:</span>
          {appliedGtoRanges.map(r => (
            <button key={r.scenarioId} onClick={() => setGtoBrowserOpen(true)}
              title={`Click to browse GTO library and pick a different ${r.side} chart`}
              style={{
                padding: '2px 8px', background: 'rgba(124, 58, 237, 0.15)',
                border: '1px solid rgba(124, 58, 237, 0.3)', borderRadius: 4,
                color: '#c4b5fd', cursor: 'pointer', fontFamily: 'inherit', fontSize: 11,
              }}
            >
              {r.side} {r.position}: {r.scenarioId.split('/').pop()}
            </button>
          ))}
        </div>
      )}


      {/* Left Sidebar */}
      <aside className="sidebar-left">
        {/* Game / Stack context — disabled while GTO chart library is
         *  off (GTO_CHART_LIBRARY_ENABLED flag). Without the chart lookup
         *  the selector has no effect on default ranges, so hiding the
         *  whole panel is cleaner than greying it out. */}
        {GTO_CHART_LIBRARY_ENABLED && (
          <GameContextSelector value={gameContext} onChange={setGameContext} />
        )}

        <PositionSelector
          selectedMatchup={selectedMatchup}
          onMatchupChange={handleMatchupChange}
          onReset={() => {
            setSelectedMatchup(null);
            setHeroPosition(null);
            setCustomIpRange(null);
            setCustomOopRange(null);
          }}
          onEditRange={(isIP) => setEditingRange(isIP ? 'IP' : 'OOP')}
        />
        <BoardSelector board={rootBoard} onBoardChange={setRootBoard} />

        <SolverControls
          pot={pot} stack={stack} iterations={iterations}
          onPotChange={setPot} onStackChange={setStack} onIterationsChange={setIterations}
          onSolve={handleSolve} loading={loading}
          sizingSpec={sizingSpec} onSizingSpecChange={setSizingSpec}
          onEditSizing={() => setSizingEditorOpen(true)}
          isoMode={isoMode} onIsoModeChange={setIsoMode}
          memoryProfile={memoryProfile} onMemoryProfileChange={setMemoryProfile}
          expectedEffectiveBB={gameContext.effectiveBB}
          solveMode={solveMode}
          onSolveModeChange={(m) => {
            // v1.3.0: changing mode resets iter cap to the preset, but the
            // user can still override via Advanced.maxIter afterwards.
            setSolveMode(m);
            setIterations(SOLVE_MODE_PRESETS[m].iterations);
          }}
          decomposeRunouts={decomposeRunouts}
          onDecomposeRunoutsChange={setDecomposeRunouts}
          exactPreflight={exactPreflight}
          onStop={async () => {
            // Pure abort — kills the engine subprocess. The previous view
            // stays; time-budget is the path for "stop with what we have"
            // (auto-fires when the budget hits).
            try {
              if ((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__) {
                const { invoke } = await import('@tauri-apps/api/core');
                await invoke('cancel_solve');
              }
            } catch (e) {
              console.warn('cancel_solve failed (non-fatal):', e);
            }
          }}
        />
        {/* v1.2.2 → v1.3.0: pre-solve ETA banner. Now budget-aware — when
            a time budget is set (always, via solve mode), the headline is
            min(estimate, budget) so we don't lie about the wait. */}
        <SolveEtaBanner
          estimate={estimate}
          loading={loading}
          timeBudgetSeconds={SOLVE_MODE_PRESETS[solveMode].time_budget_seconds}
        />
        {(error || nav.navError) && (
          <div className="glass-panel animate-fade-in" style={{
            padding: 12, borderColor: 'var(--color-red)', background: 'var(--color-red-dim)',
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-red)', marginBottom: 4 }}>{t('error')}</div>
            <div style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>{error ?? nav.navError}</div>
          </div>
        )}
        {nodeLocks.length > 0 && (
          <div className="glass-panel" style={{ padding: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
              {t('locks.active', { n: nodeLocks.length })}
            </span>
            <button className="btn-secondary" style={{ fontSize: 11, padding: '3px 10px' }}
              onClick={() => setNodeLocks([])}>{t('locks.clear')}</button>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main className="main-content">

        {/* Street Status Bar — the board as dealt along the line */}
        {hasSolved && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 0,
            padding: '10px 16px',
            background: 'var(--color-glass)',
            backdropFilter: 'blur(20px)',
            borderRadius: 12,
            border: '1px solid var(--color-glass-border)',
            marginBottom: 14, maxWidth: 720, width: '100%',
          }}>
            {(() => {
              const cards = nav.boardCards;
              // Street of the node shown (a pending deal belongs to the next).
              const currentStreet = awaiting
                ? (awaiting.street === 'turn' ? 1 : 2)
                : Math.max(0, cards.length - 3);
              // Actions per street along the line.
              const perStreet: string[][] = [[], [], []];
              let st = Math.max(0, (segment ? nav.segments[0].request.board.length / 2 : 3) - 3);
              for (const s of line) {
                perStreet[Math.min(2, st)].push(`${s.actor} ${prettyAction(s.label)}`);
                if (s.card) st += 1;
              }

              const renderCard = (card: string, idx: number) => {
                const rank = card[0];
                const suit = SUIT_MAP[card[1]];
                return (
                  <span key={idx} style={{
                    display: 'inline-flex', alignItems: 'center', gap: 1,
                    padding: '3px 6px', borderRadius: 5,
                    background: 'rgba(255,255,255,0.06)',
                    fontSize: 13, fontWeight: 800,
                    fontFamily: 'var(--font-mono)',
                    color: suit?.color || '#fff',
                    border: '1px solid rgba(255,255,255,0.08)',
                  }}>
                    {rank}<span style={{ fontSize: 11 }}>{suit?.symbol}</span>
                  </span>
                );
              };

              const renderPlaceholder = () => (
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 30, height: 26, borderRadius: 5,
                  border: '2px dashed rgba(255,255,255,0.15)',
                  fontSize: 11, color: 'var(--color-text-tertiary)',
                }}>?</span>
              );

              const streetActionSummary = (actions: string[]) => {
                if (actions.length === 0) return null;
                // Show last 2 actions max
                const shown = actions.slice(-2);
                return (
                  <div style={{ fontSize: 9, color: 'var(--color-text-tertiary)', marginTop: 3, lineHeight: 1.3 }}>
                    {shown.map((a, i) => (
                      <span key={i}>
                        {i > 0 && ' → '}{a}
                      </span>
                    ))}
                    {actions.length > 2 && <span> (+{actions.length - 2})</span>}
                  </div>
                );
              };

              const section = (idx: 0 | 1 | 2, name: string, color: string, tint: string) => {
                const isCurrent = currentStreet === idx;
                const streetCards = idx === 0 ? cards.slice(0, 3) : cards.slice(idx + 2, idx + 3);
                const reached = streetCards.length > 0 || isCurrent;
                return (
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    padding: '4px 8px', borderRadius: 8,
                    background: isCurrent ? `${tint}14` : 'transparent',
                    border: isCurrent ? `1px solid ${tint}33` : '1px solid transparent',
                    opacity: reached ? 1 : 0.3,
                    transition: 'all 200ms ease',
                  }}>
                    <div style={{
                      fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '0.8px', marginBottom: 4,
                      color: isCurrent ? color : 'var(--color-text-tertiary)',
                    }}>{name}</div>
                    <div style={{ display: 'flex', gap: 3 }}>
                      {streetCards.length > 0 ? streetCards.map((c, i) => renderCard(c, i)) : renderPlaceholder()}
                    </div>
                    {streetActionSummary(perStreet[idx])}
                  </div>
                );
              };

              const arrow = (on: boolean) => (
                <div style={{
                  width: 24, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: on ? 'var(--color-text-tertiary)' : 'rgba(255,255,255,0.1)',
                  fontSize: 14,
                }}>→</div>
              );

              return (
                <>
                  {section(0, 'Flop', '#30D158', '#30D158')}
                  {arrow(cards.length > 3)}
                  {section(1, 'Turn', '#FF9F0A', '#FF9F0A')}
                  {arrow(cards.length > 4)}
                  {section(2, 'River', '#FF453A', '#FF453A')}
                </>
              );
            })()}
          </div>
        )}

        <RangeGrid
          result={loading ? null : displayResult}
          displayMode={gridMode}
          heatmapAction={heatmapAction}
          onDisplayModeChange={(mode, action) => {
            setGridMode(mode);
            if (action) setHeatmapAction(action);
          }}
          viewSide={gridViewSide}
          onViewSideChange={setGridViewSide}
          onCellHover={setHoveredCombo}
          onCellClick={handleCellClick}
        />

        {/* Card selector — the line reached a deal */}
        {hasSolved && awaiting && !loading && (
          <div style={{ marginTop: 12, maxWidth: 720, width: '100%' }}>
            <TurnRiverCardSelector
              street={awaiting.street}
              currentBoard={boardString}
              onCardSelect={handleDealCard}
              hint={awaiting.collapsed ? t('deal.resolveHint')
                : (segment?.response.session_id == null ? t('deal.cachedOnly') : undefined)}
              allowed={!awaiting.collapsed && segment?.response.session_id == null
                ? new Set(awaiting.runouts.map(r => r.card)) : undefined}
            />
          </div>
        )}

        {/* Action Bar — the node's actions as solved */}
        {hasSolved && view && !awaiting && !loading && (
          <div style={{ marginTop: 12, maxWidth: 720, width: '100%' }}>
            <ActionBar view={view} onAction={handleAction} loading={loading || navBusy} />
            {view.kind === 'player' && view.actions.length === 0 && (
              <button className="btn-primary" style={{ marginTop: 8, width: '100%' }} onClick={handleSolve}>
                {t('solve')}
              </button>
            )}
          </div>
        )}

        {!displayResult && !loading && (
          <div style={{
            textAlign: 'center', color: 'var(--color-text-tertiary)',
            fontSize: 13, maxWidth: 300,
          }}>
            {t('app.placeholder')}
          </div>
        )}
      </main>

      {/* Right Sidebar */}
      <aside className="sidebar-right">
        <StrategyPanel
          result={displayResult}
          hoveredCombo={hoveredCombo}
          elapsed={elapsed}
          loading={loading}
          progress={progress}
          board={boardString}
          currentHistory={nav.segmentCacheKey}
          onLockNode={() => {
            const targetCombo = targetAnalysis?.combo;
            if (!view || view.kind !== 'player' || !targetCombo) return;
            const history = nav.segmentHistory;
            // The node's actions in engine order: a lock gives one
            // frequency per action, in this order.
            const actions = view.actions.map(a => a.label);
            const existing = segmentLocks.find(l => l.combo === targetCombo && l.history === history);
            let initialStrategy: Record<string, number> | undefined;
            if (existing) {
              initialStrategy = {};
              actions.forEach((a, i) => { initialStrategy![a] = existing.strategy[i] ?? 0; });
            }
            setEditingNodeLock({ combo: targetCombo, actions, initialStrategy });
          }}
        />
      </aside>

      {/* Modals */}
      {editingRange && (
        <RangeEditorModal
          title={`Edit ${editingRange} Range`}
          initialRangeStr={editingRange === 'IP' ? (customIpRange ?? selectedMatchup?.ipRange) : (customOopRange ?? selectedMatchup?.oopRange)}
          onSave={(str: string) => {
            if (editingRange === 'IP') setCustomIpRange(str);
            else setCustomOopRange(str);
            setEditingRange(null);
          }}
          onClose={() => setEditingRange(null)}
        />
      )}

      {editingNodeLock && segment && (
        <NodeLockEditor
          combo={editingNodeLock.combo}
          actions={editingNodeLock.actions}
          initialStrategy={editingNodeLock.initialStrategy}
          onSave={(str: Record<string, number>) => {
            const history = nav.segmentHistory;
            const lock: NodeLock = {
              history,
              combo: editingNodeLock.combo,
              strategy: editingNodeLock.actions.map(a => str[a] ?? 0),
            };
            const locks = [
              ...segmentLocks.filter(l => !(l.combo === lock.combo && l.history === history)),
              lock,
            ];
            setEditingNodeLock(null);
            // The first solve's locks also carry into the next Solve.
            if (nav.segments.length === 1) setNodeLocks(locks);
            // Re-solve this street with the lock and come back to the node.
            resolveSegment({ node_locks: JSON.stringify(locks) });
          }}
          onClose={() => setEditingNodeLock(null)}
        />
      )}

      {sizingEditorOpen && (
        <BetSizingEditor
          spec={sizingSpec}
          onSave={(spec) => { setSizingSpec(spec); setSizingEditorOpen(false); }}
          onClose={() => setSizingEditorOpen(false)}
        />
      )}

      {showGuide && <GuideModal onClose={() => setShowGuide(false)} />}

      {/* GTO chart library — preflop chart browser + range applicator.
       *  Disabled while the bundled chart data fails poker correctness
       *  audit (see GTO_CHART_LIBRARY_ENABLED flag at top of file). */}
      {GTO_CHART_LIBRARY_ENABLED && (
        <GtoChartBrowser
          open={gtoBrowserOpen}
          onClose={() => setGtoBrowserOpen(false)}
          onApply={(rangeStr, side) => {
            if (side === 'IP') setCustomIpRange(rangeStr);
            else setCustomOopRange(rangeStr);
          }}
        />
      )}

      {showSpotLibrary && (
        <SpotLibrary
          onSelectSpot={(spot) => {
            // Configure state from spot
            const spotPot = Math.round(spot.matchup.defaultPot * 10);
            const spotStack = Math.round(spot.matchup.defaultStack * 10);
            setRootBoard(spot.board);
            setSelectedMatchup(spot.matchup);
            setHeroPosition(spot.matchup.oop as Position);
            setPot(spotPot);
            setStack(spotStack);
            setCustomIpRange(null);
            setCustomOopRange(null);
            setNodeLocks([]);
            setShowSpotLibrary(false);
            // The spot's own settings (state updates land after this call).
            const request = buildRequest({
              board: spot.board, pot: spotPot, stack: spotStack,
              ipRange: spot.matchup.ipRange, oopRange: spot.matchup.oopRange, locks: [],
            });

            // v1.8.3+ Phase 3: a bundled, pre-solved spot shows its root
            // strategy at once; Solve explores the tree from there.
            if (spot.source === 'bundled' || !isRealSolverAvailable()) {
              setError(null);
              start({
                status: 'success',
                iterations_run: spot.iterationsRun ?? 0,
                exploitability_pct: spot.exploitabilityPct ?? 0,
                global_strategy: spot.globalStrategy,
                combo_strategies: spot.comboStrategies,
                acting_player: 'OOP',
                node: {
                  kind: 'player', street: 0, pot: spotPot, stack_oop: spotStack, stack_ip: spotStack,
                  to_call: 0, board: [], actions: [], runouts: [],
                },
              }, request);
              return;
            }
            // Tauri mode: a REAL solve with this spot's config and the
            // current sizing menus.
            solveRoot(request);
          }}
          onClose={() => setShowSpotLibrary(false)}
        />
      )}
      {showDrill && <DrillMode onClose={() => setShowDrill(false)} />}
    </div>
    </AuthGate>
  );
}

export default App;
