import React, { useState, useMemo } from 'react';
import { GRID_LABELS, getActionColor } from '../lib/poker';
import type { SolverResponse, ComboStrategy } from '../lib/poker';
import { useT } from '../lib/i18n';

/**
 * Grid display modes — what to colour each 169-cell by.
 *
 * - `mix`             : action-mix gradient (default; same as competing solvers)
 * - `action-heatmap`  : single-action intensity (e.g., "show only Bet 75% freq")
 * - `ev`              : per-class EV heatmap, red→grey→green normalized to the
 *                       in-range EV span. Surfaces "which combos are profit
 *                       centres vs which are losing" at a glance.
 * - `aggression`      : sum of Bet/Raise/All-in frequencies, cool→hot.
 *                       Answers "how often does this class take an
 *                       aggressive line vs passive (Check/Call)?"
 */
export type GridDisplayMode = 'mix' | 'action-heatmap' | 'ev' | 'aggression';
export type GridViewSide = 'acting' | 'opponent';

/** Actions counted as "aggressive" for the aggression heatmap. */
function isAggressiveAction(label: string): boolean {
  return label.startsWith('Bet_') ||
         label.startsWith('Raise') ||
         label === 'All-in' ||
         label === 'Bet';
}

/** Sum of aggressive-action frequencies in a combo strategy. Returns 0..1. */
function aggressionScore(strategy: ComboStrategy): number {
  let s = 0;
  for (const [action, freq] of Object.entries(strategy)) {
    if (action === 'Not in range') continue;
    if (isAggressiveAction(action)) s += freq;
  }
  return Math.max(0, Math.min(1, s));
}

/**
 * Map a normalized value in [0, 1] to a diverging red→grey→green color.
 * 0 = saturated red, 0.5 = neutral grey, 1 = saturated green.
 * Used for the EV heatmap so the user can spot loss/profit centers fast.
 */
function divergingColor(t: number): string {
  const tt = Math.max(0, Math.min(1, t));
  if (tt < 0.5) {
    // red → grey
    const k = tt * 2;                          // 0..1
    const r = Math.round(220 - 100 * k);       // 220 → 120
    const g = Math.round(38 + 82 * k);         // 38 → 120
    const b = Math.round(38 + 82 * k);         // 38 → 120
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // grey → green
    const k = (tt - 0.5) * 2;                  // 0..1
    const r = Math.round(120 - 104 * k);       // 120 → 16
    const g = Math.round(120 + 65 * k);        // 120 → 185
    const b = Math.round(120 - 39 * k);        // 120 → 81
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/**
 * Map an aggression score [0, 1] to a cool→hot color gradient
 * (grey → orange → red). Distinct from the EV diverging palette so the
 * two modes don't look the same at a glance.
 */
function aggressionColor(score: number): string {
  if (score < 0.005) return 'var(--color-bg-tertiary)';
  const t = Math.max(0, Math.min(1, score));
  const r = Math.round(120 + 135 * t);         // 120 → 255
  const g = Math.round(120 - 90 * t);          // 120 → 30
  const b = Math.round(120 - 110 * t);         // 120 → 10
  return `rgb(${r}, ${g}, ${b})`;
}

/** Format an EV value for cell display. Signed, 1 decimal for small values
 *  to avoid "+0" hiding small wins, integer for big ones. */
function formatEv(ev: number): string {
  if (!Number.isFinite(ev)) return '—';
  const abs = Math.abs(ev);
  const sign = ev >= 0 ? '+' : '';
  if (abs >= 100) return `${sign}${ev.toFixed(0)}`;
  if (abs >= 10) return `${sign}${ev.toFixed(0)}`;
  return `${sign}${ev.toFixed(1)}`;
}

interface Props {
  result: SolverResponse | null;
  displayMode?: GridDisplayMode;
  heatmapAction?: string;
  availableActions?: string[];
  onDisplayModeChange?: (mode: GridDisplayMode, action?: string) => void;
  onCellHover?: (label: string | null) => void;
  onCellClick?: (label: string) => void;
  /** Whose range/strategy to display. Defaults to "acting". */
  viewSide?: GridViewSide;
  onViewSideChange?: (side: GridViewSide) => void;
}

/** Parse a combo strategy into sorted action entries */
function parseStrategy(strategy: ComboStrategy) {
  return Object.entries(strategy)
    .filter(([, freq]) => freq > 0.005)
    .sort((a, b) => b[1] - a[1])
    .map(([action, frequency]) => ({
      action,
      frequency,
      color: getActionColor(action),
    }));
}

/** Generate a CSS gradient from sorted action entries */
function strategyToGradient(entries: { action: string; frequency: number; color: string }[]): string {
  if (!entries.length) return 'var(--color-bg-tertiary)';
  const total = entries.reduce((s, e) => s + e.frequency, 0);
  const stops: string[] = [];
  let cumulative = 0;
  for (const entry of entries) {
    const norm = entry.frequency / total;
    const start = cumulative * 100;
    cumulative += norm;
    const end = cumulative * 100;
    stops.push(`${entry.color} ${start.toFixed(1)}% ${end.toFixed(1)}%`);
  }
  return `linear-gradient(135deg, ${stops.join(', ')})`;
}

/** Generate heatmap color for a single action frequency */
function heatmapColor(freq: number, actionColor: string): string {
  if (freq < 0.005) return 'var(--color-bg-tertiary)';
  // Interpolate from dark to full action color based on frequency
  const alpha = Math.min(1, freq * 1.2); // slightly boost visibility
  // Convert hex to rgba
  const r = parseInt(actionColor.slice(1, 3), 16);
  const g = parseInt(actionColor.slice(3, 5), 16);
  const b = parseInt(actionColor.slice(5, 7), 16);
  if (isNaN(r)) return actionColor; // fallback
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * 13x13 Dynamic Strategy Grid with view mode switching.
 * - "mix": Multi-action gradient per cell (default)
 * - "action-heatmap": Single-action intensity map
 */
export function RangeGrid({
  result, displayMode = 'mix', heatmapAction, availableActions = [],
  onDisplayModeChange, onCellHover, onCellClick,
  viewSide = 'acting', onViewSideChange,
}: Props) {
  const t = useT();
  const [hoveredCell, setHoveredCell] = useState<string | null>(null);
  const [showActionPicker, setShowActionPicker] = useState(false);

  // Derive available actions from result if not provided
  const actions = useMemo(() => {
    if (availableActions.length > 0) return availableActions;
    if (!result?.global_strategy) return [];
    return Object.keys(result.global_strategy);
  }, [result, availableActions]);

  // Pre-compute per-cell data
  const cellData = useMemo(() => {
    const data: Record<string, {
      background: string;
      entries: { action: string; frequency: number; color: string }[];
      /** For action-heatmap and aggression: the underlying 0..1 frequency. */
      heatFreq?: number;
      /** For ev mode: the raw EV value to render in the cell label. */
      ev?: number;
      /** For aggression mode: aggression score 0..1. */
      aggScore?: number;
    }> = {};

    const flat = GRID_LABELS.flat();
    const isHeatmap = displayMode === 'action-heatmap' && heatmapAction;
    const actionColor = isHeatmap ? getActionColor(heatmapAction) : '';

    // EV mode needs a min/max sweep to normalize colors against the in-range
    // EV span. We do this once up front so each cell can look up its
    // normalized t in O(1). Cells with combo_strategies but missing combo_evs
    // fall back to their strategy gradient (better than a misleading colour).
    let evMin = Infinity;
    let evMax = -Infinity;
    if (displayMode === 'ev' && result?.combo_evs && result?.combo_strategies) {
      for (const label of flat) {
        const cs = result.combo_strategies[label];
        if (!cs || cs['Not in range']) continue;
        const ev = result.combo_evs[label];
        if (typeof ev !== 'number' || !Number.isFinite(ev)) continue;
        if (ev < evMin) evMin = ev;
        if (ev > evMax) evMax = ev;
      }
    }
    const evSpan = evMax - evMin;

    // Opponent-view mode: render a heatmap of the opponent's reach-weighted
    // range at this node. Values are already normalized to [0, 1] by the
    // backend (heaviest label = 1.0). Each cell's intensity = that label's
    // weight. This is a separate rendering path from strategy gradients.
    if (viewSide === 'opponent' && result?.opponent_range) {
      const oppRange = result.opponent_range;
      const rangeColor = '#60a5fa';  // neutral blue to distinguish from action colors
      for (const label of flat) {
        const w = oppRange[label] ?? 0;
        if (w <= 0.005) {
          data[label] = { background: 'var(--color-bg-tertiary)', entries: [] };
          continue;
        }
        data[label] = {
          background: heatmapColor(w, rangeColor),
          entries: [{ action: 'range', frequency: w, color: rangeColor }],
          heatFreq: w,
        };
      }
      return data;
    }

    for (const label of flat) {
      const comboStrategy = result?.combo_strategies?.[label];

      if (comboStrategy) {
        // Treat "Not in range" as a sentinel value — the backend doesn't emit
        // combo_strategies entries for combos with reach 0, but the mock does
        // (for offline/dev mode). Either way, don't paint these with
        // strategy gradients — they'd just be misleading noise in a cell the
        // user deliberately excluded.
        const isOutOfRange = !!comboStrategy['Not in range'];
        if (isOutOfRange) {
          data[label] = { background: 'var(--color-bg-tertiary)', entries: [] };
          continue;
        }

        const entries = parseStrategy(comboStrategy);

        if (isHeatmap) {
          const freq = comboStrategy[heatmapAction!] ?? 0;
          data[label] = {
            background: heatmapColor(freq, actionColor),
            entries,
            heatFreq: freq,
          };
        } else if (displayMode === 'ev') {
          const ev = result?.combo_evs?.[label];
          if (typeof ev === 'number' && Number.isFinite(ev) && evSpan > 0.01) {
            const t = (ev - evMin) / evSpan;
            data[label] = {
              background: divergingColor(t),
              entries,
              ev,
            };
          } else {
            // No EV data, or all in-range EVs identical — fall back to mix
            // gradient so the cell isn't an opaque grey lie.
            data[label] = {
              background: strategyToGradient(entries),
              entries,
              ev: typeof ev === 'number' ? ev : undefined,
            };
          }
        } else if (displayMode === 'aggression') {
          const score = aggressionScore(comboStrategy);
          data[label] = {
            background: aggressionColor(score),
            entries,
            aggScore: score,
          };
        } else {
          data[label] = {
            background: strategyToGradient(entries),
            entries,
          };
        }
      } else {
        // No per-combo data → out-of-range cell. Render as a muted/grey tile
        // so it's visually distinct from the solved cells. Do NOT fall back
        // to global_strategy here: that painted every out-of-range cell with
        // the average mix and made the grid look like the user's range was
        // 100% when it wasn't.
        data[label] = { background: 'var(--color-bg-tertiary)', entries: [] };
      }
    }
    return data;
  }, [result, displayMode, heatmapAction, viewSide]);

  const handleMouseEnter = (label: string) => {
    setHoveredCell(label);
    onCellHover?.(label);
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
    onCellHover?.(null);
  };

  const actingPlayer = result?.acting_player;
  const opponentSide = result?.opponent_side;
  const opponentRangeAvailable = !!(result?.opponent_range &&
    Object.keys(result.opponent_range).length > 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 700, width: '100%' }}>
      {/* Header: current view + side toggle */}
      {result && actingPlayer && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '8px 12px', background: 'var(--color-glass)',
          borderRadius: 8, border: '1px solid var(--color-glass-border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{
              fontSize: 10, fontWeight: 700, letterSpacing: '0.5px',
              textTransform: 'uppercase', color: 'var(--color-text-tertiary)',
            }}>
              {viewSide === 'opponent' ? t('grid.opponentLabel') : t('grid.actingLabel')}
            </span>
            <span style={{
              fontSize: 14, fontWeight: 700,
              color: viewSide === 'opponent' ? '#60a5fa' : 'var(--color-accent)',
            }}>
              {viewSide === 'opponent' ? (opponentSide ?? '—') : actingPlayer}
            </span>
          </div>

          {/* Acting / Opponent side toggle */}
          {onViewSideChange && opponentRangeAvailable && (
            <div style={{ display: 'flex', gap: 4 }}>
              <button onClick={() => onViewSideChange('acting')}
                style={{
                  padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
                  background: viewSide === 'acting' ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
                  color: viewSide === 'acting' ? '#fff' : 'var(--color-text-secondary)',
                  fontSize: 11, fontWeight: 600, fontFamily: 'inherit',
                  transition: 'all 150ms ease',
                }}>
                {t('grid.viewActing')} ({actingPlayer})
              </button>
              <button onClick={() => onViewSideChange('opponent')}
                style={{
                  padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
                  background: viewSide === 'opponent' ? '#60a5fa' : 'var(--color-bg-tertiary)',
                  color: viewSide === 'opponent' ? '#fff' : 'var(--color-text-secondary)',
                  fontSize: 11, fontWeight: 600, fontFamily: 'inherit',
                  transition: 'all 150ms ease',
                }}>
                {t('grid.viewOpponent')} ({opponentSide ?? '—'})
              </button>
            </div>
          )}
        </div>
      )}

      {/* View Mode Switcher */}
      {result && onDisplayModeChange && viewSide === 'acting' && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 10px', background: 'var(--color-glass)',
          borderRadius: 8, border: '1px solid var(--color-glass-border)',
        }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginRight: 4 }}>
            {t('grid.view')}
          </span>
          {/* Strategy Mix button */}
          <button onClick={() => { onDisplayModeChange('mix'); setShowActionPicker(false); }}
            style={{
              padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: displayMode === 'mix' ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
              color: displayMode === 'mix' ? '#fff' : 'var(--color-text-secondary)',
              fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
            }}>
            {t('grid.strategyMix')}
          </button>

          {/* EV button (Day 4) */}
          <button onClick={() => { onDisplayModeChange('ev'); setShowActionPicker(false); }}
            disabled={!result?.combo_evs || Object.keys(result.combo_evs).length === 0}
            title="Per-class EV heatmap (red = losing, green = winning)"
            style={{
              padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: displayMode === 'ev' ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
              color: displayMode === 'ev' ? '#fff' : 'var(--color-text-secondary)',
              fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
              opacity: !result?.combo_evs ? 0.5 : 1,
            }}>
            {t('grid.ev')}
          </button>

          {/* Aggression button (Day 4) */}
          <button onClick={() => { onDisplayModeChange('aggression'); setShowActionPicker(false); }}
            title="Sum of Bet/Raise/All-in frequencies per class (cool = passive, hot = aggressive)"
            style={{
              padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: displayMode === 'aggression' ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
              color: displayMode === 'aggression' ? '#fff' : 'var(--color-text-secondary)',
              fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
            }}>
            {t('grid.aggression')}
          </button>

          {/* Action Heatmap button */}
          <div style={{ position: 'relative' }}>
            <button onClick={() => {
              if (displayMode === 'action-heatmap') {
                setShowActionPicker(!showActionPicker);
              } else {
                onDisplayModeChange('action-heatmap', actions[0]);
                setShowActionPicker(true);
              }
            }}
              style={{
                padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
                background: displayMode === 'action-heatmap' ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
                color: displayMode === 'action-heatmap' ? '#fff' : 'var(--color-text-secondary)',
                fontSize: 11, fontWeight: 600, fontFamily: 'inherit', transition: 'all 150ms ease',
              }}>
              {t('grid.heatmap')}{heatmapAction && displayMode === 'action-heatmap' ? `: ${heatmapAction}` : ''}
            </button>

            {/* Action picker dropdown */}
            {showActionPicker && displayMode === 'action-heatmap' && (
              <div style={{
                position: 'absolute', top: '100%', left: 0, marginTop: 4,
                background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)',
                borderRadius: 8, padding: 4, zIndex: 50,
                boxShadow: '0 8px 24px rgba(0,0,0,0.4)', minWidth: 140,
              }}>
                {actions.map(action => (
                  <button key={action} onClick={() => {
                    onDisplayModeChange('action-heatmap', action);
                    setShowActionPicker(false);
                  }}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                      padding: '6px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
                      background: heatmapAction === action ? 'rgba(255,255,255,0.08)' : 'transparent',
                      color: '#fff', fontSize: 12, fontWeight: 500,
                      fontFamily: 'inherit', textAlign: 'left',
                    }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: getActionColor(action) }} />
                    {action}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Grid */}
      <div className="range-grid animate-fade-in">
        {GRID_LABELS.flat().map((label, idx) => {
          const cell = cellData[label];
          // Pick what the cell label shows based on the active mode.
          let labelText: string = label;
          if (displayMode === 'action-heatmap' && cell?.heatFreq !== undefined) {
            labelText = `${Math.round(cell.heatFreq * 100)}`;
          } else if (displayMode === 'ev' && cell?.ev !== undefined) {
            labelText = formatEv(cell.ev);
          } else if (displayMode === 'aggression' && cell?.aggScore !== undefined) {
            labelText = `${Math.round(cell.aggScore * 100)}`;
          }
          return (
            <div
              key={idx}
              className="range-cell"
              style={{ background: cell?.background || 'var(--color-bg-tertiary)' }}
              onMouseEnter={() => handleMouseEnter(label)}
              onMouseLeave={handleMouseLeave}
              onClick={() => onCellClick?.(label)}
            >
              <span className="label">{labelText}</span>
              {hoveredCell === label && cell?.entries.length > 0 && (
                <div className="tooltip">
                  <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13 }}>
                    {label}
                    {cell.ev !== undefined && (
                      <span style={{
                        marginLeft: 8, fontSize: 11, fontWeight: 500,
                        color: cell.ev >= 0 ? '#10b981' : '#ef4444',
                      }}>
                        EV {formatEv(cell.ev)}
                      </span>
                    )}
                    {cell.aggScore !== undefined && (
                      <span style={{
                        marginLeft: 8, fontSize: 11, fontWeight: 500,
                        color: 'var(--color-text-tertiary)',
                      }}>
                        AGG {Math.round(cell.aggScore * 100)}%
                      </span>
                    )}
                  </div>
                  {cell.entries.map((e) => (
                    <div key={e.action} style={{
                      display: 'flex', gap: 8, alignItems: 'center',
                      justifyContent: 'space-between', minWidth: 120,
                    }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: e.color, flexShrink: 0 }} />
                        <span style={{ fontSize: 11 }}>{e.action}</span>
                      </div>
                      <span className="text-mono" style={{ fontSize: 11, fontWeight: 600 }}>
                        {(e.frequency * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
