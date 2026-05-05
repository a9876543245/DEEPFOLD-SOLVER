/**
 * Aggregated Report Lite — modal.
 *
 * Day 1: 13×4 turn-card grid for the currently-displayed flop spot, hover
 * reveals full strategy + mean EV.
 *
 * Day 2 additions:
 *   - View toggle: "By Card" (per-turn grid) vs "By Class" (texture buckets:
 *     Pair / Flush completion / Straight completion / Overcard / Brick).
 *   - Sort modes for By Card view: Card order / Best EV / Worst EV /
 *     Most aggressive. Non-card-order sorts switch to a flat 1D wrap so
 *     the suit×rank layout doesn't lie about position.
 *   - CSV export — downloads a sub-100KB blob via `<a download>`.
 *
 * See AGGREGATED_REPORT_DESIGN.md for the data flow.
 */

import { useMemo, useState } from 'react';
import { X, Download, Grid3x3, Layers, ArrowUpDown, TrendingUp, TrendingDown, Zap } from 'lucide-react';
import {
  aggregateTurns,
  bucketAggregates,
  sortAggregates,
  turnsToCSV,
  dominantAction,
  actionColor,
  CARD_CLASS_COLORS,
  type TurnAggregate,
  type BucketAggregate,
  type SortMode,
} from '../lib/aggregateRunouts';
import type { StrategyTreeEntry } from '../lib/poker';

interface Props {
  /** Strategy tree from the most recent solve. */
  strategyTree: Record<string, StrategyTreeEntry> | undefined;
  /** Flop history (action sequence ending in the chance, e.g. "Check,Bet_33,Call"). */
  flopHistory: string;
  /** Cards on the flop board (3 cards, e.g. ["As","Ks","Qs"]). Used to grey them out. */
  flopCards: string[];
  onClose: () => void;
}

type ViewMode = 'card' | 'class';

const RANK_CHARS = '23456789TJQKA';
const SUIT_CHARS = ['c', 'd', 'h', 's'] as const;
const SUIT_LABELS: Record<string, string> = { c: '♣', d: '♦', h: '♥', s: '♠' };
const SUIT_COLORS: Record<string, string> = {
  c: '#9ca3af', d: '#60a5fa', h: '#f87171', s: '#e5e7eb',
};

/** Group aggregates into a (suit → rank → TurnAggregate) lookup so the grid
 *  can render in fixed positions. */
function indexAggregates(aggs: TurnAggregate[]): Record<string, Record<number, TurnAggregate>> {
  const idx: Record<string, Record<number, TurnAggregate>> = { c: {}, d: {}, h: {}, s: {} };
  for (const a of aggs) {
    if (!idx[a.suit]) idx[a.suit] = {};
    idx[a.suit][a.rankIdx] = a;
  }
  return idx;
}

/** Trigger a CSV download in-browser. No Tauri filesystem needed since
 *  output is sub-100KB. */
function downloadCSV(filename: string, csv: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Defer revoke so Safari can finish the download
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Build a stable filename for the CSV, e.g. `runout_AsKsQs_Check_Bet_33_Call.csv` */
function csvFilename(flopCards: string[], flopHistory: string): string {
  const board = flopCards.join('') || 'flop';
  const hist = (flopHistory || 'root').replace(/[,#]/g, '_');
  return `runout_${board}_${hist}.csv`;
}

export function RunoutReportModal({ strategyTree, flopHistory, flopCards, onClose }: Props) {
  const aggregates = useMemo(
    () => aggregateTurns(strategyTree, flopHistory),
    [strategyTree, flopHistory],
  );
  const flopSet = useMemo(() => new Set(flopCards.map((c) => c.toLowerCase())), [flopCards]);
  const [hover, setHover] = useState<TurnAggregate | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('card');
  const [sortMode, setSortMode] = useState<SortMode>('card');

  const sorted = useMemo(
    () => sortAggregates(aggregates, sortMode),
    [aggregates, sortMode],
  );
  const indexed = useMemo(() => indexAggregates(sorted), [sorted]);
  const buckets = useMemo(
    () => bucketAggregates(aggregates, flopCards),
    [aggregates, flopCards],
  );

  const hasData = aggregates.length > 0;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        className="animate-slide-up"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 760, maxHeight: '90vh',
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
          borderRadius: 16, display: 'flex', flexDirection: 'column',
          boxShadow: '0 32px 80px rgba(0,0,0,0.5)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '18px 24px', borderBottom: '1px solid var(--color-border)',
            background: 'var(--color-bg-primary)', flexShrink: 0,
          }}
        >
          <div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>Runout Report</div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
              {hasData
                ? `${aggregates.length} canonical turn cards · ${flopHistory || 'root'}`
                : 'No runout data — board did not enumerate turns'}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {hasData && (
              <button
                onClick={() => downloadCSV(csvFilename(flopCards, flopHistory), turnsToCSV(aggregates))}
                title="Download all turn aggregates as CSV"
                style={{
                  padding: '8px 12px', border: '1px solid var(--color-border)',
                  background: 'var(--color-glass)', borderRadius: 8, cursor: 'pointer',
                  color: 'var(--color-text-secondary)',
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontSize: 12, fontWeight: 600,
                }}
              >
                <Download size={14} /> CSV
              </button>
            )}
            <button
              onClick={onClose}
              aria-label="Close"
              style={{
                padding: 8, border: 'none', background: 'var(--color-glass)',
                borderRadius: 8, cursor: 'pointer', color: 'var(--color-text-secondary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Toolbar */}
        {hasData && (
          <Toolbar
            viewMode={viewMode}
            sortMode={sortMode}
            bucketCount={buckets.length}
            onViewMode={setViewMode}
            onSortMode={setSortMode}
          />
        )}

        {/* Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          {!hasData && <NoDataState />}
          {hasData && viewMode === 'card' && sortMode === 'card' && (
            <Grid indexed={indexed} flopSet={flopSet} onHover={setHover} />
          )}
          {hasData && viewMode === 'card' && sortMode !== 'card' && (
            <FlatGrid sorted={sorted} onHover={setHover} />
          )}
          {hasData && viewMode === 'class' && (
            <BucketTable buckets={buckets} totalTurns={aggregates.length} />
          )}
          {hover && viewMode === 'card' && <DetailPane agg={hover} />}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Toolbar
// ─────────────────────────────────────────────────────────────────────

interface ToolbarProps {
  viewMode: ViewMode;
  sortMode: SortMode;
  bucketCount: number;
  onViewMode: (m: ViewMode) => void;
  onSortMode: (m: SortMode) => void;
}

function Toolbar({ viewMode, sortMode, bucketCount, onViewMode, onSortMode }: ToolbarProps) {
  return (
    <div
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 12, padding: '10px 24px',
        borderBottom: '1px solid var(--color-border)',
        background: 'var(--color-bg-primary)',
        flexShrink: 0, flexWrap: 'wrap',
      }}
    >
      {/* View toggle */}
      <div style={{ display: 'flex', gap: 4 }}>
        <ToggleBtn
          active={viewMode === 'card'}
          onClick={() => onViewMode('card')}
          icon={<Grid3x3 size={12} />}
          label="By Card"
        />
        <ToggleBtn
          active={viewMode === 'class'}
          onClick={() => onViewMode('class')}
          icon={<Layers size={12} />}
          label={`By Class${bucketCount > 0 ? ` (${bucketCount})` : ''}`}
        />
      </div>

      {/* Sort modes (only meaningful in By Card view) */}
      {viewMode === 'card' && (
        <div style={{ display: 'flex', gap: 4 }}>
          <ToggleBtn
            active={sortMode === 'card'}
            onClick={() => onSortMode('card')}
            icon={<ArrowUpDown size={12} />}
            label="Card"
          />
          <ToggleBtn
            active={sortMode === 'bestEv'}
            onClick={() => onSortMode('bestEv')}
            icon={<TrendingUp size={12} />}
            label="Best EV"
          />
          <ToggleBtn
            active={sortMode === 'worstEv'}
            onClick={() => onSortMode('worstEv')}
            icon={<TrendingDown size={12} />}
            label="Worst EV"
          />
          <ToggleBtn
            active={sortMode === 'aggressive'}
            onClick={() => onSortMode('aggressive')}
            icon={<Zap size={12} />}
            label="Aggressive"
          />
        </div>
      )}
    </div>
  );
}

interface ToggleBtnProps {
  active: boolean;
  onClick: () => void;
  icon?: React.ReactNode;
  label: string;
}

function ToggleBtn({ active, onClick, icon, label }: ToggleBtnProps) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '6px 10px',
        border: '1px solid ' + (active ? 'var(--color-accent, #3b82f6)' : 'var(--color-border)'),
        background: active ? 'var(--color-accent, #3b82f6)' : 'var(--color-glass)',
        color: active ? '#fff' : 'var(--color-text-secondary)',
        borderRadius: 6, cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: 5,
        fontSize: 11, fontWeight: 600,
      }}
    >
      {icon} {label}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────
// No data state
// ─────────────────────────────────────────────────────────────────────

function NoDataState() {
  return (
    <div
      style={{
        padding: '40px 20px', textAlign: 'center',
        color: 'var(--color-text-secondary)', fontSize: 13, lineHeight: 1.6,
      }}
    >
      <div style={{ fontSize: 48, opacity: 0.3, marginBottom: 12 }}>?</div>
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
        This board didn&apos;t enumerate turn runouts.
      </div>
      <div style={{ maxWidth: 480, margin: '0 auto', opacity: 0.85 }}>
        Rainbow flops fall back to a single chance child to keep memory bounded
        — there&apos;s no per-turn data to aggregate. Try a monotone, two-tone,
        or paired board, or solve a turn spot directly.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// 13×4 grid (card-order view)
// ─────────────────────────────────────────────────────────────────────

interface GridProps {
  indexed: Record<string, Record<number, TurnAggregate>>;
  flopSet: Set<string>;
  onHover: (agg: TurnAggregate | null) => void;
}

function Grid({ indexed, flopSet, onHover }: GridProps) {
  return (
    <div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '24px repeat(13, 1fr)',
          gap: 4, marginBottom: 4,
        }}
      >
        <div />
        {RANK_CHARS.split('').map((r) => (
          <div
            key={r}
            style={{
              fontSize: 11, fontWeight: 700, textAlign: 'center',
              color: 'var(--color-text-tertiary)',
            }}
          >
            {r}
          </div>
        ))}
      </div>
      {SUIT_CHARS.map((s) => (
        <div
          key={s}
          style={{
            display: 'grid',
            gridTemplateColumns: '24px repeat(13, 1fr)',
            gap: 4, marginBottom: 4,
          }}
        >
          <div
            style={{
              fontSize: 14, fontWeight: 700,
              color: SUIT_COLORS[s], textAlign: 'center',
              alignSelf: 'center',
            }}
          >
            {SUIT_LABELS[s]}
          </div>
          {RANK_CHARS.split('').map((r, ri) => (
            <Cell
              key={r + s}
              card={r.toLowerCase() + s}
              agg={indexed[s]?.[ri]}
              isFlopCard={flopSet.has((r + s).toLowerCase())}
              onHover={onHover}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Flat 1D grid (sorted view)
// ─────────────────────────────────────────────────────────────────────

function FlatGrid({ sorted, onHover }: { sorted: TurnAggregate[]; onHover: (agg: TurnAggregate | null) => void }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(56px, 1fr))',
        gap: 6,
      }}
    >
      {sorted.map((agg) => (
        <FlatCell key={agg.card} agg={agg} onHover={onHover} />
      ))}
    </div>
  );
}

function FlatCell({ agg, onHover }: { agg: TurnAggregate; onHover: (agg: TurnAggregate | null) => void }) {
  const dom = dominantAction(agg.strategy);
  const bg = dom ? actionColor(dom.label) : 'var(--color-bg-tertiary)';
  const freqLabel = dom ? `${dom.freq.toFixed(0)}%` : '—';
  return (
    <div
      onMouseEnter={() => onHover(agg)}
      onMouseLeave={() => onHover(null)}
      style={{
        aspectRatio: '1', borderRadius: 6,
        background: bg, color: '#fff',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        fontSize: 10, fontWeight: 700,
        cursor: 'help',
        boxShadow: '0 1px 2px rgba(0,0,0,0.35) inset',
        position: 'relative',
      }}
    >
      <div style={{ fontSize: 11, color: SUIT_COLORS[agg.suit] }}>
        {agg.card.toUpperCase().slice(0, 1)}{SUIT_LABELS[agg.suit]}
      </div>
      <div style={{ fontSize: 9, opacity: 0.9 }}>{freqLabel}</div>
      <div style={{ fontSize: 8, opacity: 0.8 }}>
        EV {agg.meanEv.toFixed(1)}
      </div>
    </div>
  );
}

interface CellProps {
  card: string;
  agg: TurnAggregate | undefined;
  isFlopCard: boolean;
  onHover: (agg: TurnAggregate | null) => void;
}

function Cell({ card, agg, isFlopCard, onHover }: CellProps) {
  if (isFlopCard) {
    return (
      <div
        title={`${card} is on the flop`}
        style={{
          aspectRatio: '1', borderRadius: 6,
          background: 'rgba(0,0,0,0.25)',
          border: '1px dashed var(--color-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 10, color: 'var(--color-text-tertiary)',
          opacity: 0.4,
        }}
      >
        flop
      </div>
    );
  }
  if (!agg) {
    return (
      <div
        title={`${card} not in canonical reps`}
        style={{
          aspectRatio: '1', borderRadius: 6,
          background: 'var(--color-bg-tertiary)',
          opacity: 0.35, display: 'flex', alignItems: 'center',
          justifyContent: 'center',
          fontSize: 10, color: 'var(--color-text-tertiary)',
        }}
      >
        —
      </div>
    );
  }
  const dom = dominantAction(agg.strategy);
  const bg = dom ? actionColor(dom.label) : 'var(--color-bg-tertiary)';
  const freqLabel = dom ? `${dom.freq.toFixed(0)}%` : '—';
  return (
    <div
      onMouseEnter={() => onHover(agg)}
      onMouseLeave={() => onHover(null)}
      style={{
        aspectRatio: '1', borderRadius: 6,
        background: bg, color: '#fff',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        fontSize: 10, fontWeight: 700,
        cursor: 'help',
        boxShadow: '0 1px 2px rgba(0,0,0,0.35) inset',
      }}
    >
      <div style={{ fontSize: 11 }}>{freqLabel}</div>
      {agg.weight > 1 && (
        <div style={{ fontSize: 8, opacity: 0.85 }}>
          ×{agg.weight}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Detail pane (hover)
// ─────────────────────────────────────────────────────────────────────

function DetailPane({ agg }: { agg: TurnAggregate }) {
  const entries = Object.entries(agg.strategy);
  return (
    <div
      style={{
        marginTop: 16, padding: 14,
        background: 'var(--color-glass)', borderRadius: 10,
        border: '1px solid var(--color-glass-border)',
        fontSize: 12,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 700 }}>
          {agg.card.toUpperCase()}
        </span>
        <span style={{ color: 'var(--color-text-tertiary)' }}>
          {agg.acting} acts · iso class ×{agg.weight}
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 8 }}>
        {entries.map(([label, freq]) => (
          <span
            key={label}
            style={{
              padding: '4px 10px', borderRadius: 999,
              background: actionColor(label), color: '#fff',
              fontSize: 11, fontWeight: 600,
            }}
          >
            {label} {freq}
          </span>
        ))}
      </div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: 11 }}>
        Mean EV: {agg.meanEv.toFixed(1)} chips
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// By-Class bucket table
// ─────────────────────────────────────────────────────────────────────

function BucketTable({ buckets, totalTurns }: { buckets: BucketAggregate[]; totalTurns: number }) {
  if (buckets.length === 0) {
    return (
      <div
        style={{
          padding: '20px', textAlign: 'center',
          color: 'var(--color-text-tertiary)', fontSize: 12,
        }}
      >
        No buckets to display.
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div
        style={{
          fontSize: 11, color: 'var(--color-text-tertiary)',
          marginBottom: 4,
        }}
      >
        Texture buckets ({buckets.length} of 5 populated · {totalTurns} canonical turns total).
        Strategy & EV are weighted by iso orbit size.
      </div>
      {buckets.map((b) => (
        <BucketRow key={b.cardClass} bucket={b} />
      ))}
    </div>
  );
}

function BucketRow({ bucket }: { bucket: BucketAggregate }) {
  const color = CARD_CLASS_COLORS[bucket.cardClass];
  const entries = Object.entries(bucket.strategy)
    .map(([label, pct]) => ({ label, pct, num: parseFloat(pct) }))
    .filter((e) => Number.isFinite(e.num) && e.num > 0.5)
    .sort((a, b) => b.num - a.num);

  return (
    <div
      style={{
        padding: 12,
        background: 'var(--color-glass)',
        border: '1px solid var(--color-glass-border)',
        borderRadius: 10,
        borderLeft: `4px solid ${color}`,
      }}
    >
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 8, gap: 12, flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span
            style={{
              width: 10, height: 10, borderRadius: '50%', background: color,
              display: 'inline-block',
            }}
          />
          <span style={{ fontSize: 13, fontWeight: 700 }}>{bucket.label}</span>
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            {bucket.count} card{bucket.count === 1 ? '' : 's'}
            {bucket.totalWeight !== bucket.count && ` · weight ${bucket.totalWeight}`}
          </span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          Mean EV: <strong>{bucket.meanEv.toFixed(1)}</strong> chips
        </div>
      </div>

      {/* Stacked strategy bar */}
      {entries.length > 0 && (
        <div
          style={{
            display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden',
            marginBottom: 8, background: 'var(--color-bg-tertiary)',
          }}
        >
          {entries.map((e) => (
            <div
              key={e.label}
              title={`${e.label} ${e.pct}`}
              style={{
                width: `${e.num}%`,
                background: actionColor(e.label),
              }}
            />
          ))}
        </div>
      )}

      {/* Strategy pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
        {entries.map((e) => (
          <span
            key={e.label}
            style={{
              padding: '3px 8px', borderRadius: 999,
              background: actionColor(e.label), color: '#fff',
              fontSize: 10, fontWeight: 600,
            }}
          >
            {e.label} {e.pct}
          </span>
        ))}
      </div>

      {/* Card list */}
      <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>
        {bucket.cards.map((c) => c.toUpperCase()).join(' · ')}
      </div>
    </div>
  );
}
