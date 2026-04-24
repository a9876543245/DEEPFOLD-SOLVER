import React, { useState, useCallback, useMemo } from 'react';
import { BoardSelector } from './components/BoardSelector';
import { PositionSelector } from './components/PositionSelector';
import { SolverControls } from './components/SolverControls';
import { RangeGrid } from './components/RangeGrid';
import type { GridDisplayMode } from './components/RangeGrid';
import { StrategyPanel } from './components/StrategyPanel';
import { ActionNavigator } from './components/ActionNavigator';
import { ActionBar } from './components/ActionBar';
import { TurnRiverCardSelector } from './components/TurnRiverCardSelector';
import { useSolver } from './hooks/useSolver';
import type { SolverRequest, NodeLock, ComboAnalysis } from './lib/poker';
import { getHandStrength, RANK_VALUES } from './lib/poker';
import type { Position, PositionMatchup } from './lib/ranges';
import { createRootNode, takeAction, dealCard, BET_SIZINGS } from './lib/gameTree';
import type { GameTreeNode, GameAction, ActionStep } from './lib/gameTree';
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
import { RunoutPicker } from './components/RunoutPicker';
import { HelpCircle, BookOpen, Crosshair } from 'lucide-react';
import { useT, useLanguage, LANGUAGES } from './lib/i18n';
import type { AuthUser } from './lib/auth';
import { clearSession } from './lib/auth';

function App() {
  const t = useT();
  const { lang, setLang } = useLanguage();
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  // Core state
  const [flopBoard, setFlopBoard] = useState('');  // Flop only (6 chars, e.g. "AsKd7c")
  const [turnCard, setTurnCard] = useState('');     // Turn card (2 chars, e.g. "2h")
  const [riverCard, setRiverCard] = useState('');   // River card (2 chars, e.g. "Js")
  const [pot, setPot] = useState(100);
  const [stack, setStack] = useState(500);
  const [iterations, setIterations] = useState(300);
  const [hoveredCombo, setHoveredCombo] = useState<string | null>(null);
  const [selectedMatchup, setSelectedMatchup] = useState<PositionMatchup | null>(null);
  const [heroPosition, setHeroPosition] = useState<Position | null>(null);

  // Game tree state
  const [currentNode, setCurrentNode] = useState<GameTreeNode | null>(null);
  const [hasSolved, setHasSolved] = useState(false);

  // Advanced solver state
  const [customIpRange, setCustomIpRange] = useState<string | null>(null);
  const [customOopRange, setCustomOopRange] = useState<string | null>(null);
  const [nodeLocks, setNodeLocks] = useState<NodeLock[]>([]);
  const [gtoBrowserOpen, setGtoBrowserOpen] = useState(false);
  // Path B runout choice. null = lex-min default. When user clicks a card
  // in the RunoutPicker, this is set to that card and the next nav uses
  // the "#XX" suffix in the cache lookup.
  const [selectedRunout, setSelectedRunout] = useState<string | null>(null);

  // Grid display mode
  const [gridMode, setGridMode] = useState<GridDisplayMode>('mix');
  const [heatmapAction, setHeatmapAction] = useState<string>('');
  const [gridViewSide, setGridViewSide] = useState<'acting' | 'opponent'>('acting');

  // Bet sizing preset — drives both UI action list (gameTree.ts) AND the
  // solver tree (passed via SolverRequest.flop_sizes/turn_sizes/river_sizes).
  // These MUST stay consistent or history navigation silently falls through
  // to the nearest backend node and shows misleading strategy.
  const [sizingKey, setSizingKey] = useState<'standard' | 'polar' | 'small_ball'>('standard');

  // Modals
  const [showGuide, setShowGuide] = useState(false);
  const [showSpotLibrary, setShowSpotLibrary] = useState(false);
  const [showDrill, setShowDrill] = useState(false);
  const [editingRange, setEditingRange] = useState<'IP' | 'OOP' | null>(null);
  const [editingNodeLock, setEditingNodeLock] = useState<{ combo: string; actions: string[]; initialStrategy?: Record<string, number> } | null>(null);

  // Compute full board from flop + turn + river
  const fullBoard = useMemo(() => {
    let b = flopBoard;
    if (turnCard) b += turnCard;
    if (riverCard) b += riverCard;
    return b;
  }, [flopBoard, turnCard, riverCard]);

  const { result, setResult, loading, error, elapsed, progress, solve, reset, navigate } = useSolver();

  // Helper: convert an ActionStep[] path into the engine's history string.
  // Skips Deal steps. If `selectedRunout` is set, attaches "#<card>" to the
  // action immediately before the FIRST Deal — Path B uses this to switch
  // to a non-lex-min canonical runout in the cached strategy tree.
  const pathToHistory = useCallback((steps: ActionStep[]): string => {
    const out: string[] = [];
    let firstDealAttached = false;
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      if (s.player === 'Deal') continue;
      let label = s.action.label;
      if (!firstDealAttached && selectedRunout) {
        // Attach to the action immediately before the first Deal step.
        const next = steps[i + 1];
        if (next?.player === 'Deal') {
          label += '#' + selectedRunout;
          firstDealAttached = true;
        }
      }
      out.push(label);
    }
    return out.join(',');
  }, [selectedRunout]);

  // Determine hero's range
  const getHeroRange = useCallback(() => {
    if (!selectedMatchup || !heroPosition) return undefined;
    const isIPHero = selectedMatchup.ip === heroPosition;
    if (isIPHero) return customIpRange ?? selectedMatchup.ipRange;
    return customOopRange ?? selectedMatchup.oopRange;
  }, [selectedMatchup, heroPosition, customIpRange, customOopRange]);

  // Auto-set pot/stack from matchup defaults
  const handleMatchupChange = useCallback((matchup: PositionMatchup, heroPos: Position) => {
    setSelectedMatchup(matchup);
    setHeroPosition(heroPos);
    // Apply default pot sizing from matchup (convert from bb to chips)
    setPot(Math.round(matchup.defaultPot * 10));
    setStack(Math.round(matchup.defaultStack * 10));
    // Reset tree on matchup change
    setCurrentNode(null);
    setHasSolved(false);
    setTurnCard('');
    setRiverCard('');
    setCustomIpRange(null);
    setCustomOopRange(null);
    setNodeLocks([]);
    reset();
  }, [reset]);

  // Build solver request for given node context
  const buildRequest = useCallback((boardStr: string, actionPath?: ActionStep[], nodePot?: number, nodeStack?: number): SolverRequest => {
    // Resolve bet-size arrays from the active preset. These MUST be passed
    // to the backend — without them it silently uses its own {0.33, 0.75}
    // default and the tree diverges from what the UI shows.
    const sz = BET_SIZINGS[sizingKey] ?? BET_SIZINGS.standard;
    return {
      board: boardStr,
      pot_size: nodePot ?? pot,
      effective_stack: nodeStack ?? stack,
      iterations,
      exploitability: 0.5,
      ip_range: customIpRange ?? selectedMatchup?.ipRange,
      oop_range: customOopRange ?? selectedMatchup?.oopRange,
      node_locks: nodeLocks.length > 0 ? JSON.stringify(nodeLocks) : undefined,
      hero_range: getHeroRange(),
      action_path: actionPath,
      flop_sizes: sz.flopBetSizes,
      turn_sizes: sz.turnBetSizes,
      river_sizes: sz.riverBetSizes,
    };
  }, [pot, stack, iterations, selectedMatchup, getHeroRange, customIpRange, customOopRange, nodeLocks, sizingKey]);

  // Initial solve (root node)
  const handleSolve = useCallback(() => {
    if (flopBoard.length < 6) return;

    // Determine starting street from board card count
    const numCards = flopBoard.length / 2;
    let startStreet: 'flop' | 'turn' | 'river';
    let solveBoard: string;

    if (numCards >= 5) {
      startStreet = 'river';
      solveBoard = flopBoard;
      // Split: first 6 chars = flop, next 2 = turn, next 2 = river
      // Update turnCard/riverCard, then trim flopBoard to flop-only
      const tc = flopBoard.substring(6, 8);
      const rc = flopBoard.substring(8, 10);
      setFlopBoard(flopBoard.substring(0, 6));
      setTurnCard(tc);
      setRiverCard(rc);
    } else if (numCards >= 4) {
      startStreet = 'turn';
      solveBoard = flopBoard;
      const tc = flopBoard.substring(6, 8);
      setFlopBoard(flopBoard.substring(0, 6));
      setTurnCard(tc);
      setRiverCard('');
    } else {
      startStreet = 'flop';
      solveBoard = flopBoard;
      setTurnCard('');
      setRiverCard('');
    }

    const rootNode = createRootNode(pot, stack, startStreet, sizingKey);
    setCurrentNode(rootNode);
    setHasSolved(true);
    solve(buildRequest(solveBoard, [], pot, stack));
  }, [flopBoard, pot, stack, solve, buildRequest, sizingKey]);

  // Navigate game tree by taking an action
  const handleAction = useCallback((action: GameAction) => {
    if (!currentNode || loading) return;
    const nextNode = takeAction(currentNode, action, sizingKey);
    setCurrentNode(nextNode);

    if (nextNode.awaitingDeal) {
      // Street advanced — need a card dealt.
      // Clear the card for the new street so the selector appears.
      if (nextNode.awaitingDeal === 'turn') {
        setTurnCard('');
        setRiverCard('');
      } else {
        setRiverCard('');
      }
      // Don't re-solve yet; wait for card selection
      return;
    }

    if (!nextNode.isTerminal) {
      // Route A: try cache first (instant). Falls back to re-solve only if
      // the new node isn't in the strategy_tree (e.g. depth beyond cache
      // horizon, or a re-solve was needed earlier and discarded the cache).
      const history = pathToHistory(nextNode.path);
      if (navigate(history)) return;

      // Cache miss → real solve.
      solve(buildRequest(fullBoard, nextNode.path, nextNode.pot, nextNode.effectiveStack));
    }
  }, [currentNode, loading, solve, navigate, pathToHistory, buildRequest, fullBoard, sizingKey]);

  // Handle Turn/River card dealing
  const handleDealCard = useCallback((card: string) => {
    if (!currentNode?.awaitingDeal) return;

    let newBoard: string;
    if (currentNode.awaitingDeal === 'turn') {
      setTurnCard(card);
      newBoard = flopBoard + card;
    } else {
      setRiverCard(card);
      newBoard = flopBoard + turnCard + card;
    }

    // Transition node from awaiting → active
    const activeNode = dealCard(currentNode);
    setCurrentNode(activeNode);

    // Re-solve with updated board
    solve(buildRequest(newBoard, activeNode.path, activeNode.pot, activeNode.effectiveStack));
  }, [currentNode, flopBoard, turnCard, solve, buildRequest]);

  // Navigate breadcrumbs (time travel)
  const handleNavigate = useCallback((index: number) => {
    // Guard against breadcrumb clicks while a solve is in flight. Without
    // this, the click can race the in-flight invoke — the new solve clears
    // result to null, then the old invoke resolves and overwrites it with
    // strategies from the previous spot, leaving the UI in a stale state.
    if (loading) return;

    if (index === -1) {
      // Back to root — reset to flop
      const rootNode = createRootNode(pot, stack, 'flop', sizingKey);
      setCurrentNode(rootNode);
      setTurnCard('');
      setRiverCard('');
      solve(buildRequest(flopBoard, [], pot, stack));
    } else if (currentNode) {
      // Rebuild tree to this point
      let node = createRootNode(pot, stack, 'flop', sizingKey);
      const targetPath = currentNode.path.slice(0, index + 1);
      for (const step of targetPath) {
        if (step.player !== 'Deal') {
          node = takeAction(node, step.action, sizingKey);
        }
      }

      // Determine which board to use based on the street
      let navBoard = flopBoard;
      if (node.street === 'turn' && turnCard) navBoard = flopBoard + turnCard;
      else if (node.street === 'river' && turnCard && riverCard) navBoard = flopBoard + turnCard + riverCard;

      // Clear cards for streets we're before
      if (node.street === 'flop') {
        setTurnCard('');
        setRiverCard('');
      } else if (node.street === 'turn') {
        setRiverCard('');
      }

      setCurrentNode(node);

      if (node.awaitingDeal) {
        // Navigated to a deal point — show card selector
        return;
      }

      // Route A: try cache first. Same-street time travel (most common
      // breadcrumb click) hits the cache; cross-street nav (rare here, but
      // possible after a deal) falls through to a real solve.
      const history = pathToHistory(node.path);
      if (navigate(history)) return;

      solve(buildRequest(navBoard, node.path, node.pot, node.effectiveStack));
    }
  }, [pot, stack, flopBoard, turnCard, riverCard, currentNode, loading, solve, navigate, pathToHistory, buildRequest, sizingKey]);

  // Click on a combo cell
  // In-range: instant local lookup from existing solve result
  // Off-range: re-solve with that combo forced into the range (real GTO calculation)
  const handleCellClick = useCallback((label: string) => {
    if (!result?.combo_strategies || flopBoard.length < 6) return;

    const strategy = result.combo_strategies[label];
    const isNotInRange = !strategy || !!strategy['Not in range'];

    if (!isNotInRange && strategy) {
      // In-range — instant local lookup, no re-solve needed
      const analysis: ComboAnalysis = { combo: label, best_action: '', ev: 0, strategy_mix: {} };
      let bestFreq = 0;
      for (const [action, freq] of Object.entries(strategy)) {
        if (action === 'Not in range') continue;
        if (freq > 0.01) analysis.strategy_mix[action] = `${(freq * 100).toFixed(1)}%`;
        if (freq > bestFreq) { bestFreq = freq; analysis.best_action = action; }
      }
      setResult(prev => prev ? { ...prev, target_combo_analysis: analysis } : prev);
    } else {
      // Off-range — real re-solve with target_combo, forcing it into the range
      const request = buildRequest(fullBoard, currentNode?.path, currentNode?.pot, currentNode?.effectiveStack);
      request.target_combo = label;
      solve(request);
    }
  }, [result, flopBoard, fullBoard, currentNode, solve, buildRequest]);

  // Breadcrumb history for ActionNavigator
  const breadcrumbHistory = useMemo(() => {
    if (!currentNode) return [];
    return currentNode.path.map(step => ({
      label: step.action.label,
      player: step.player as 'OOP' | 'IP' | 'Deal',
    }));
  }, [currentNode]);

  // Position labels
  const heroIsIP = selectedMatchup && heroPosition ? selectedMatchup.ip === heroPosition : null;
  const villainPosition = selectedMatchup && heroPosition
    ? (selectedMatchup.ip === heroPosition ? selectedMatchup.oop : selectedMatchup.ip)
    : null;

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
        <ActionNavigator history={breadcrumbHistory} onNavigate={handleNavigate} />

        {/* Path B runout picker — shows when current cache entry has runout
         *  options (i.e. iso enumeration was engaged at the prior chance).
         *  Picking a different card swaps the cache lookup and re-renders
         *  strategies for that runout. */}
        {result?.runout_options && result.runout_options.length > 1 && (
          <RunoutPicker
            options={result.runout_options}
            active={result.dealt_cards?.[result.dealt_cards.length - 1] ?? null}
            onPick={(card) => {
              setSelectedRunout(card);
              // Trigger re-navigation with new runout choice. handleNavigate
              // would expect a breadcrumb index, so call navigate() directly
              // via the same path-rebuild used elsewhere.
              if (currentNode) {
                // Rebuild the engine history with the new runout token, then
                // do a cache lookup. Cache miss falls through to re-solve.
                // We mimic the post-action nav flow.
                const tempHistory = (() => {
                  const out: string[] = [];
                  let attached = false;
                  for (let i = 0; i < currentNode.path.length; i++) {
                    const s = currentNode.path[i];
                    if (s.player === 'Deal') continue;
                    let label = s.action.label;
                    if (!attached) {
                      const next = currentNode.path[i + 1];
                      if (next?.player === 'Deal') {
                        label += '#' + card;
                        attached = true;
                      }
                    }
                    out.push(label);
                  }
                  return out.join(',');
                })();
                if (!navigate(tempHistory)) {
                  solve(buildRequest(fullBoard, currentNode.path, currentNode.pot, currentNode.effectiveStack));
                }
              }
            }}
          />
        )}

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

          {/* GTO Chart Library button */}
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

      {/* Left Sidebar */}
      <aside className="sidebar-left">
        <PositionSelector 
          selectedMatchup={selectedMatchup} 
          onMatchupChange={handleMatchupChange} 
          onReset={() => { 
            setSelectedMatchup(null); 
            setHeroPosition(null); 
            setTurnCard(''); 
            setRiverCard(''); 
            setCustomIpRange(null);
            setCustomOopRange(null);
          }} 
          onEditRange={(isIP) => setEditingRange(isIP ? 'IP' : 'OOP')}
        />
        <BoardSelector board={flopBoard} onBoardChange={(b) => { setFlopBoard(b); setTurnCard(''); setRiverCard(''); }} />

        <SolverControls
          pot={pot} stack={stack} iterations={iterations}
          onPotChange={setPot} onStackChange={setStack} onIterationsChange={setIterations}
          onSolve={handleSolve} loading={loading}
          sizingKey={sizingKey} onSizingChange={setSizingKey}
        />
        {error && (
          <div className="glass-panel animate-fade-in" style={{
            padding: 12, borderColor: 'var(--color-red)', background: 'var(--color-red-dim)',
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-red)', marginBottom: 4 }}>{t('error')}</div>
            <div style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>{error}</div>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main className="main-content">

        {/* Street Status Bar — shows board progression */}
        {hasSolved && currentNode && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 0,
            padding: '10px 16px',
            background: 'var(--color-glass)',
            backdropFilter: 'blur(20px)',
            borderRadius: 12,
            border: '1px solid var(--color-glass-border)',
            marginBottom: 14, maxWidth: 720, width: '100%',
          }}>
            {/* Flop */}
            {(() => {
              const SUIT_MAP: Record<string, { symbol: string; color: string }> = {
                s: { symbol: '♠', color: '#E8E8E8' },
                h: { symbol: '♥', color: '#FF453A' },
                d: { symbol: '♦', color: '#0A84FF' },
                c: { symbol: '♣', color: '#30D158' },
              };
              const isCurrentFlop = currentNode.street === 'flop';
              const isCurrentTurn = currentNode.street === 'turn';
              const isCurrentRiver = currentNode.street === 'river';

              // Parse flop cards
              const flopCards: string[] = [];
              for (let i = 0; i < flopBoard.length; i += 2) {
                flopCards.push(flopBoard.substring(i, i + 2));
              }


              // Get actions per street from path

              const getStreetActions = (street: 'flop' | 'turn' | 'river') => {
                const actions: string[] = [];
                let inStreet = street === 'flop';
                for (const step of currentNode.path) {
                  if (step.player === 'Deal') {
                    if (step.action.label === 'Turn' && street === 'turn') inStreet = true;
                    else if (step.action.label === 'River' && street === 'river') inStreet = true;
                    else if (inStreet) break;
                    continue;
                  }
                  if (inStreet) {
                    actions.push(`${step.player} ${step.action.label}`);
                  }
                }
                return actions;
              };

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

              return (
                <>
                  {/* Flop section */}
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    padding: '4px 8px', borderRadius: 8,
                    background: isCurrentFlop ? 'rgba(48,209,88,0.08)' : 'transparent',
                    border: isCurrentFlop ? '1px solid rgba(48,209,88,0.2)' : '1px solid transparent',
                    transition: 'all 200ms ease',
                  }}>
                    <div style={{
                      fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '0.8px', marginBottom: 4,
                      color: isCurrentFlop ? '#30D158' : 'var(--color-text-tertiary)',
                    }}>Flop</div>
                    <div style={{ display: 'flex', gap: 3 }}>
                      {flopCards.map((c, i) => renderCard(c, i))}
                    </div>
                    {streetActionSummary(getStreetActions('flop'))}
                  </div>

                  {/* Separator */}
                  <div style={{
                    width: 24, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: turnCard ? 'var(--color-text-tertiary)' : 'rgba(255,255,255,0.1)',
                    fontSize: 14,
                  }}>→</div>

                  {/* Turn section */}
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    padding: '4px 8px', borderRadius: 8,
                    background: isCurrentTurn ? 'rgba(255,159,10,0.08)' : 'transparent',
                    border: isCurrentTurn ? '1px solid rgba(255,159,10,0.2)' : '1px solid transparent',
                    opacity: turnCard || isCurrentTurn || currentNode.awaitingDeal === 'turn' ? 1 : 0.3,
                    transition: 'all 200ms ease',
                  }}>
                    <div style={{
                      fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '0.8px', marginBottom: 4,
                      color: isCurrentTurn ? '#FF9F0A' : 'var(--color-text-tertiary)',
                    }}>Turn</div>
                    <div style={{ display: 'flex', gap: 3 }}>
                      {turnCard ? renderCard(turnCard, 0) : renderPlaceholder()}
                    </div>
                    {turnCard && streetActionSummary(getStreetActions('turn'))}
                  </div>

                  {/* Separator */}
                  <div style={{
                    width: 24, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: riverCard ? 'var(--color-text-tertiary)' : 'rgba(255,255,255,0.1)',
                    fontSize: 14,
                  }}>→</div>

                  {/* River section */}
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    padding: '4px 8px', borderRadius: 8,
                    background: isCurrentRiver ? 'rgba(255,69,58,0.08)' : 'transparent',
                    border: isCurrentRiver ? '1px solid rgba(255,69,58,0.2)' : '1px solid transparent',
                    opacity: riverCard || isCurrentRiver || currentNode.awaitingDeal === 'river' ? 1 : 0.3,
                    transition: 'all 200ms ease',
                  }}>
                    <div style={{
                      fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '0.8px', marginBottom: 4,
                      color: isCurrentRiver ? '#FF453A' : 'var(--color-text-tertiary)',
                    }}>River</div>
                    <div style={{ display: 'flex', gap: 3 }}>
                      {riverCard ? renderCard(riverCard, 0) : renderPlaceholder()}
                    </div>
                    {riverCard && streetActionSummary(getStreetActions('river'))}
                  </div>
                </>
              );
            })()}
          </div>
        )}

        <RangeGrid
          result={result}
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

        {/* Turn/River Card Selector — shown when awaiting deal */}
        {hasSolved && currentNode?.awaitingDeal && (
          <div style={{ marginTop: 12, maxWidth: 720, width: '100%' }}>
            <TurnRiverCardSelector
              street={currentNode.awaitingDeal}
              currentBoard={currentNode.awaitingDeal === 'turn' ? flopBoard : flopBoard + turnCard}
              onCardSelect={handleDealCard}
            />
          </div>
        )}

        {/* Action Bar — shown after solving (not during deal) */}
        {hasSolved && currentNode && !currentNode.awaitingDeal && (result || currentNode.isTerminal) && (
          <div style={{ marginTop: 12, maxWidth: 720, width: '100%' }}>
            <ActionBar node={currentNode} onAction={handleAction} loading={loading} />
          </div>
        )}

        {!result && !loading && (
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
          result={result}
          hoveredCombo={hoveredCombo}
          elapsed={elapsed}
          loading={loading}
          progress={progress}
          board={fullBoard}
          heroRange={getHeroRange()}
          onLockNode={() => {
            const targetCombo = result?.target_combo_analysis?.combo;
            if (!result || !targetCombo || !currentNode) return;
            const currentHistory = currentNode.path
                .filter(step => step.player !== 'Deal')
                .map(step => step.action.label)
                .join(',');
            // If already locked, edit the existing lock
            const existingLock = nodeLocks.find(l => l.combo === targetCombo && l.history === currentHistory);
            
            // Build actions list
            const comboStrategy = result.combo_strategies?.[targetCombo];
            let availableActions = comboStrategy 
              ? Object.keys(comboStrategy).filter(k => k !== 'Not in range')
              : [];
              
            // If combo is not in range or has no actions, fallback to the global action tree so user can still force hypothetical locks
            if (availableActions.length === 0) {
              availableActions = Object.keys(result.global_strategy);
            }
            
            console.log("[DEBUG] Activating NodeLockEditor for", targetCombo, availableActions);
            
            // Reconstruct initial strategy if it existed
            let initialStrategy: Record<string, number> | undefined = undefined;
            if (existingLock) {
              initialStrategy = {};
              availableActions.forEach((a, i) => {
                initialStrategy![a] = existingLock.strategy[i] ?? 0;
              });
            }

            setEditingNodeLock({ combo: targetCombo, actions: availableActions, initialStrategy });
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

      {editingNodeLock && currentNode && (
        <NodeLockEditor
          combo={editingNodeLock.combo}
          actions={editingNodeLock.actions}
          initialStrategy={editingNodeLock.initialStrategy}
          onSave={(str: Record<string, number>) => {
            const currentHistory = currentNode.path
                .filter(step => step.player !== 'Deal')
                .map(step => step.action.label)
                .join(',');

            // ordered by actions
            const strategyArr = editingNodeLock.actions.map(a => str[a] ?? 0);
            
            setNodeLocks(prev => {
              const cleaned = prev.filter(l => !(l.combo === editingNodeLock.combo && l.history === currentHistory));
              return [...cleaned, { history: currentHistory, combo: editingNodeLock.combo, strategy: strategyArr }];
            });
            setEditingNodeLock(null);
            
            // Trigger auto-resolve when locking to immediately see effect
            if (flopBoard.length >= 6) {
              // Wait for state to settle then re-solve
              setTimeout(() => {
                solve(buildRequest(currentNode.street === 'flop' ? flopBoard : currentNode.street === 'turn' ? flopBoard + turnCard : flopBoard + turnCard + riverCard, currentNode.path, currentNode.pot, currentNode.effectiveStack));
              }, 0);
            }
          }}
          onClose={() => setEditingNodeLock(null)}
        />
      )}

      {showGuide && <GuideModal onClose={() => setShowGuide(false)} />}

      {/* GTO chart library — preflop chart browser + range applicator */}
      <GtoChartBrowser
        open={gtoBrowserOpen}
        onClose={() => setGtoBrowserOpen(false)}
        onApply={(rangeStr, side) => {
          if (side === 'IP') setCustomIpRange(rangeStr);
          else setCustomOopRange(rangeStr);
        }}
      />

      {showSpotLibrary && (
        <SpotLibrary
          onSelectSpot={(spot) => {
            // Configure state from spot
            const spotPot = Math.round(spot.matchup.defaultPot * 10);
            const spotStack = Math.round(spot.matchup.defaultStack * 10);
            setFlopBoard(spot.board);
            setTurnCard('');
            setRiverCard('');
            setHasSolved(true);
            setSelectedMatchup(spot.matchup);
            setHeroPosition(spot.matchup.oop as any);
            setPot(spotPot);
            setStack(spotStack);
            setCustomIpRange(null);
            setCustomOopRange(null);
            setNodeLocks([]);
            const rootNode = createRootNode(spotPot, spotStack, 'flop');
            setCurrentNode(rootNode);
            setShowSpotLibrary(false);

            if (isRealSolverAvailable()) {
              // Tauri mode: trigger a REAL solve with this spot's config.
              // User sees the normal solving progress indicator.
              solve({
                board: spot.board,
                pot_size: spotPot,
                effective_stack: spotStack,
                iterations,
                exploitability: 0.5,
                ip_range: spot.matchup.ipRange,
                oop_range: spot.matchup.oopRange,
              });
            } else {
              // Browser mode: fall back to heuristic preview data (marked isDemo).
              setResult({
                status: 'success',
                iterations_run: 200,
                exploitability_pct: 0.32,
                global_strategy: spot.globalStrategy,
                combo_strategies: spot.comboStrategies,
              });
            }
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
