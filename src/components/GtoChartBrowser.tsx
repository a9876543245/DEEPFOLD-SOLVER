/**
 * GtoChartBrowser — preflop chart library viewer.
 *
 * Loads scenarios from the bundled `gto_output/` dataset (~2550 charts) via
 * Tauri commands `list_gto_scenarios` + `load_gto_chart`. User can:
 *   - Filter by game type / scenario / position
 *   - Preview a chart's action mix on a 13x13 grid
 *   - Apply the chart's combined "in-range" set as the IP or OOP range in
 *     the main solver (Workstream A: default preflop ranges)
 */
import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { GRID_LABELS } from '../lib/poker';
import { parseRange } from '../lib/ranges';
import { isTauri } from '../lib/tauriEnv';

// ============================================================================
// Types (mirror src-tauri/src/gto_charts.rs)
// ============================================================================

export interface GtoScenario {
  id: string;
  game_type: string;          // "Cash" | "MTT"
  scenario_type: string;       // "6max_100bb" | "vs_open_3b" | etc.
  hero_position: string;       // "BB" | "BTN" | etc.
  effective_bb: number | null;
  description: string;
}

export interface GtoChart {
  id: string;
  context: GtoScenario;
  actions: string[];
  ranges: Record<string, string>;  // action -> UPI range string
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** Apply the chart's "in-range" set (everything that's not 100% Fold) as
   *  the IP or OOP range in the main solver. */
  onApply: (rangeStr: string, side: 'IP' | 'OOP') => void;
}

// ============================================================================
// Chart -> per-cell action mix
// ============================================================================

interface CellMix {
  total: number;                          // sum of non-fold frequencies (0..1)
  byAction: Record<string, number>;       // action -> frequency
}

function computeChartMix(chart: GtoChart): Record<string, CellMix> {
  const out: Record<string, CellMix> = {};
  const parsedByAction: Record<string, Record<string, number>> = {};
  for (const action of chart.actions) {
    parsedByAction[action] = parseRange(chart.ranges[action] ?? '');
  }
  for (const row of GRID_LABELS) {
    for (const label of row) {
      const mix: Record<string, number> = {};
      let nonFold = 0;
      for (const action of chart.actions) {
        const f = parsedByAction[action][label] ?? 0;
        if (f > 0) {
          mix[action] = f;
          if (!/fold/i.test(action)) nonFold += f;
        }
      }
      out[label] = { total: nonFold, byAction: mix };
    }
  }
  return out;
}

/** Combine a chart's non-fold actions into a single PioSolver-style range
 *  string for use as an IP/OOP default range. Frequencies sum across all
 *  non-fold actions for each combo. */
function chartToInRange(chart: GtoChart): string {
  const parsedByAction: Record<string, Record<string, number>> = {};
  for (const action of chart.actions) {
    if (/fold/i.test(action)) continue;
    parsedByAction[action] = parseRange(chart.ranges[action] ?? '');
  }
  const merged: Record<string, number> = {};
  for (const action in parsedByAction) {
    for (const [combo, f] of Object.entries(parsedByAction[action])) {
      merged[combo] = (merged[combo] ?? 0) + f;
    }
  }
  return Object.entries(merged)
    .filter(([, f]) => f > 0.001)
    .map(([combo, f]) => `${combo}:${Math.min(1, f).toFixed(3)}`)
    .join(',');
}

// ============================================================================
// Action color map
// ============================================================================

function actionColor(action: string): string {
  const a = action.toLowerCase();
  if (a.includes('fold')) return '#9ca3af';            // neutral gray
  if (a.includes('all-in') || a.includes('allin')) return '#7c2d12'; // dark red
  if (a.includes('raise') || a.includes('bet')) return '#dc2626';     // red
  if (a.includes('call') || a.includes('check')) return '#16a34a';    // green
  return '#6366f1';  // fallback indigo
}

// ============================================================================
// Component
// ============================================================================

export function GtoChartBrowser({ open, onClose, onApply }: Props) {
  const [scenarios, setScenarios] = useState<GtoScenario[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState({ game: 'ALL', scenario: 'ALL', position: 'ALL' });
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [chart, setChart] = useState<GtoChart | null>(null);
  const [chartLoading, setChartLoading] = useState(false);

  // Load scenario list once when the modal first opens.
  useEffect(() => {
    if (!open || scenarios.length > 0) return;
    if (!isTauri()) {
      setError('GTO chart library is only available in the desktop app.');
      return;
    }
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const list = await invoke<GtoScenario[]>('list_gto_scenarios');
        setScenarios(list);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [open, scenarios.length]);

  // Load the selected chart on demand.
  useEffect(() => {
    if (!selectedId) { setChart(null); return; }
    if (!isTauri()) return;
    setChartLoading(true);
    (async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const c = await invoke<GtoChart>('load_gto_chart', { id: selectedId });
        setChart(c);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setChartLoading(false);
      }
    })();
  }, [selectedId]);

  // Build filter dropdown options from the scenario list.
  const games     = useMemo(() => Array.from(new Set(scenarios.map(s => s.game_type))).sort(), [scenarios]);
  const scenTypes = useMemo(() => Array.from(new Set(scenarios.map(s => s.scenario_type))).sort(), [scenarios]);
  const positions = useMemo(() => Array.from(new Set(scenarios.map(s => s.hero_position))).sort(), [scenarios]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return scenarios.filter(s => {
      if (filter.game     !== 'ALL' && s.game_type     !== filter.game)     return false;
      if (filter.scenario !== 'ALL' && s.scenario_type !== filter.scenario) return false;
      if (filter.position !== 'ALL' && s.hero_position !== filter.position) return false;
      if (q && !`${s.id} ${s.description}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [scenarios, filter, search]);

  const cellMix = useMemo(() => chart ? computeChartMix(chart) : null, [chart]);

  const apply = useCallback((side: 'IP' | 'OOP') => {
    if (!chart) return;
    const rangeStr = chartToInRange(chart);
    onApply(rangeStr, side);
    onClose();
  }, [chart, onApply, onClose]);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '90vw', maxWidth: 1200, height: '85vh',
          background: '#1a1a1a', color: '#e5e5e5',
          borderRadius: 8, padding: 16, display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
          <h2 style={{ margin: 0, flex: 1 }}>GTO Preflop Chart Library</h2>
          <button onClick={onClose} style={{ padding: '4px 12px', fontSize: 14 }}>Close</button>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search…"
            style={{ flex: 1, minWidth: 200, padding: 6 }}
          />
          <select value={filter.game} onChange={(e) => setFilter(f => ({ ...f, game: e.target.value }))}>
            <option value="ALL">All games ({games.length})</option>
            {games.map(g => <option key={g} value={g}>{g}</option>)}
          </select>
          <select value={filter.scenario} onChange={(e) => setFilter(f => ({ ...f, scenario: e.target.value }))}>
            <option value="ALL">All scenarios ({scenTypes.length})</option>
            {scenTypes.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={filter.position} onChange={(e) => setFilter(f => ({ ...f, position: e.target.value }))}>
            <option value="ALL">All positions</option>
            {positions.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        {/* Body: list + preview */}
        <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
          {/* Scenario list */}
          <div style={{ width: 360, overflowY: 'auto', background: '#0e0e0e', borderRadius: 4 }}>
            {loading && <div style={{ padding: 12 }}>Loading scenarios…</div>}
            {error && <div style={{ padding: 12, color: '#f87171' }}>Error: {error}</div>}
            {!loading && !error && filtered.length === 0 && (
              <div style={{ padding: 12, color: '#6b7280' }}>No matches</div>
            )}
            {filtered.map(s => (
              <div
                key={s.id}
                onClick={() => setSelectedId(s.id)}
                style={{
                  padding: '6px 10px', cursor: 'pointer',
                  borderBottom: '1px solid #222',
                  background: selectedId === s.id ? '#2563eb33' : 'transparent',
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 500 }}>
                  {s.hero_position}{s.effective_bb !== null ? ` ${s.effective_bb}bb` : ''} — {s.scenario_type}
                </div>
                <div style={{ fontSize: 11, color: '#9ca3af' }}>{s.id}</div>
              </div>
            ))}
            <div style={{ padding: 8, fontSize: 11, color: '#6b7280', borderTop: '1px solid #222' }}>
              Showing {filtered.length} of {scenarios.length} scenarios
            </div>
          </div>

          {/* Preview pane */}
          <div style={{ flex: 1, overflowY: 'auto', background: '#0e0e0e', borderRadius: 4, padding: 12 }}>
            {!selectedId && (
              <div style={{ color: '#6b7280' }}>Pick a scenario from the left to preview.</div>
            )}
            {chartLoading && <div>Loading chart…</div>}
            {chart && cellMix && (
              <div>
                <div style={{ marginBottom: 8 }}>
                  <strong>{chart.context.description}</strong>
                  <div style={{ fontSize: 12, color: '#9ca3af' }}>
                    Hero: {chart.context.hero_position}
                    {chart.context.effective_bb !== null ? ` · ${chart.context.effective_bb}bb` : ''}
                    · {chart.actions.length} actions
                  </div>
                </div>

                {/* Action legend */}
                <div style={{ display: 'flex', gap: 12, marginBottom: 8, flexWrap: 'wrap' }}>
                  {chart.actions.map(a => (
                    <div key={a} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
                      <span style={{ width: 12, height: 12, background: actionColor(a), borderRadius: 2 }} />
                      {a}
                    </div>
                  ))}
                </div>

                {/* 13x13 grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(13, 1fr)', gap: 1 }}>
                  {GRID_LABELS.map((row, ri) => row.map((label, ci) => {
                    const mix = cellMix[label];
                    const isPair = ri === ci;
                    const total = mix.total;
                    return (
                      <div
                        key={`${ri}-${ci}`}
                        title={`${label}: ${chart.actions.map(a => `${a} ${((mix.byAction[a] ?? 0) * 100).toFixed(0)}%`).join(', ')}`}
                        style={{
                          aspectRatio: '1', position: 'relative',
                          background: total < 0.001 ? '#2a2a2a' : '#1a1a1a',
                          fontSize: 9, color: '#e5e5e5',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          border: isPair ? '1px solid #444' : '1px solid #222',
                          overflow: 'hidden',
                        }}
                      >
                        {/* Action strips */}
                        {chart.actions.map((a, ai) => {
                          const f = mix.byAction[a] ?? 0;
                          if (f < 0.001) return null;
                          const startPct = chart.actions.slice(0, ai).reduce((acc, p) => acc + (mix.byAction[p] ?? 0), 0) * 100;
                          const widthPct = f * 100;
                          return (
                            <div key={a} style={{
                              position: 'absolute', top: 0, bottom: 0,
                              left: `${startPct}%`, width: `${widthPct}%`,
                              background: actionColor(a),
                              opacity: 0.85,
                            }} />
                          );
                        })}
                        <span style={{ position: 'relative', zIndex: 1, textShadow: '0 1px 2px black' }}>{label}</span>
                      </div>
                    );
                  }))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            disabled={!chart}
            onClick={() => apply('OOP')}
            style={{ padding: '6px 14px', background: '#7c3aed', color: 'white', border: 'none', borderRadius: 4, cursor: chart ? 'pointer' : 'not-allowed', opacity: chart ? 1 : 0.5 }}
          >
            Apply as OOP range
          </button>
          <button
            disabled={!chart}
            onClick={() => apply('IP')}
            style={{ padding: '6px 14px', background: '#2563eb', color: 'white', border: 'none', borderRadius: 4, cursor: chart ? 'pointer' : 'not-allowed', opacity: chart ? 1 : 0.5 }}
          >
            Apply as IP range
          </button>
        </div>
      </div>
    </div>
  );
}
