import React, { useState, useMemo } from 'react';
import { GRID_LABELS, getActionColor } from '../lib/poker';
import type { SolverResponse, ComboStrategy } from '../lib/poker';
import { useT } from '../lib/i18n';

export type GridDisplayMode = 'mix' | 'action-heatmap';
export type GridViewSide = 'acting' | 'opponent';

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
      heatFreq?: number;
    }> = {};

    const flat = GRID_LABELS.flat();
    const isHeatmap = displayMode === 'action-heatmap' && heatmapAction;
    const actionColor = isHeatmap ? getActionColor(heatmapAction) : '';

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
          const isHeatmap = displayMode === 'action-heatmap';
          return (
            <div
              key={idx}
              className="range-cell"
              style={{ background: cell?.background || 'var(--color-bg-tertiary)' }}
              onMouseEnter={() => handleMouseEnter(label)}
              onMouseLeave={handleMouseLeave}
              onClick={() => onCellClick?.(label)}
            >
              <span className="label">
                {isHeatmap && cell?.heatFreq !== undefined
                  ? `${Math.round(cell.heatFreq * 100)}`
                  : label}
              </span>
              {hoveredCell === label && cell?.entries.length > 0 && (
                <div className="tooltip">
                  <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13 }}>{label}</div>
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
