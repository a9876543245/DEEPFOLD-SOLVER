/**
 * 1326 Combo Drill — Day 3 modal.
 *
 * Click any 169 label → see the 4/6/12 specific combos in that class with
 * **per-combo blocker analysis** ranked best-to-worst. Strategy mix and EV
 * are class-shared (engine doesn't yet emit per-specific-combo values) and
 * labelled as such so users don't misread them.
 *
 * UX:
 *   - Header: title + class search + close
 *   - Class summary card: cards icon, strategy bar, mean EV, range weight
 *   - Combo grid: 4/6/12 cards (live first, sorted by blocker % desc),
 *     dead combos faded
 *   - Per combo: cards with suit colors, blocker pct + bar, top 3 blocked
 *     opp classes ("blocks AsKs ×3"), strategy bar (shared with class)
 *
 * See HANDOFF_NEXT_SESSION.md for the per-combo emission punt.
 */

import { useMemo, useState } from 'react';
import { X, Search, Layers, AlertCircle } from 'lucide-react';
import {
  analyzeClass,
  sortDrilledByBlocker,
  rangedLabels,
  type DrilledCombo,
} from '../lib/comboDrill';
import {
  SUIT_SYMBOLS,
  SUIT_COLORS,
  getActionColor,
  type ComboStrategy,
} from '../lib/poker';

interface Props {
  /** Hero per-class strategy mixes for the current node. */
  comboStrategies: Record<string, ComboStrategy> | undefined;
  /** Hero per-class EVs for the current node. */
  comboEvs: Record<string, number> | undefined;
  /** Opponent range at this node (drives blocker math). */
  opponentRange: Record<string, number> | undefined;
  /** Board cards (3..5 entries). */
  boardCards: string[];
  /** Default class label (e.g. from target_combo_analysis). */
  initialLabel?: string;
  onClose: () => void;
}

export function ComboDrillPanel({
  comboStrategies,
  comboEvs,
  opponentRange,
  boardCards,
  initialLabel,
  onClose,
}: Props) {
  // Pick a sensible default class: caller's hint, else hero's most-played
  // (highest sum of non-fold actions), else "AKs" as a stable fallback.
  const defaultLabel = useMemo(() => {
    if (initialLabel) return initialLabel;
    if (comboStrategies) {
      const sorted = Object.entries(comboStrategies)
        .map(([label, mix]) => {
          let sum = 0;
          for (const [a, v] of Object.entries(mix)) {
            if (a === 'Not in range' || a === 'Fold') continue;
            sum += v;
          }
          return { label, sum };
        })
        .sort((a, b) => b.sum - a.sum);
      if (sorted.length > 0 && sorted[0].sum > 0) return sorted[0].label;
    }
    return 'AKs';
  }, [initialLabel, comboStrategies]);

  const [label, setLabel] = useState(defaultLabel);
  const [filter, setFilter] = useState('');

  const drilled = useMemo(
    () => sortDrilledByBlocker(analyzeClass(label, opponentRange, boardCards)),
    [label, opponentRange, boardCards],
  );
  const classMix = comboStrategies?.[label];
  const classEv = comboEvs?.[label];
  const heroFreqInRange = classMix
    ? Object.entries(classMix)
        .filter(([a]) => a !== 'Not in range' && a !== 'Fold')
        .reduce((s, [, v]) => s + v, 0)
    : 0;

  const liveCount = drilled.filter((d) => !d.combo.isDead).length;
  const totalCount = drilled.length;

  // Range stats for the picker
  const ranged = useMemo(() => rangedLabels(opponentRange), [opponentRange]);
  const rangedSet = useMemo(() => new Set(ranged.map((r) => r.label)), [ranged]);

  const filteredLabels = useMemo(() => {
    if (!comboStrategies) return [];
    const all = Object.keys(comboStrategies);
    const f = filter.trim().toLowerCase();
    return all
      .filter((l) => f === '' || l.toLowerCase().startsWith(f))
      .slice(0, 60);
  }, [comboStrategies, filter]);

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
          width: '100%', maxWidth: 880, maxHeight: '90vh',
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
            <div style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers size={16} /> Combo Drill — {label}
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
              {liveCount} of {totalCount} combos live ·
              {' '}{ranged.length} opp classes in range
            </div>
          </div>
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

        {/* Class picker */}
        <div
          style={{
            padding: '12px 24px',
            borderBottom: '1px solid var(--color-border)',
            background: 'var(--color-bg-primary)', flexShrink: 0,
          }}
        >
          <div style={{ position: 'relative', marginBottom: 8 }}>
            <Search
              size={14}
              style={{
                position: 'absolute', left: 10, top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--color-text-tertiary)',
              }}
            />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter class (e.g. AK, 99, AKs)…"
              style={{
                width: '100%', padding: '6px 10px 6px 30px',
                background: 'var(--color-bg-tertiary)',
                border: '1px solid var(--color-border)',
                borderRadius: 6, color: 'var(--color-text-primary)',
                fontSize: 12,
              }}
            />
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxHeight: 80, overflowY: 'auto' }}>
            {filteredLabels.map((l) => {
              const inRange = rangedSet.has(l);
              const active = l === label;
              return (
                <button
                  key={l}
                  onClick={() => setLabel(l)}
                  title={inRange ? `${l} is in opp range` : `${l} not in opp range`}
                  style={{
                    padding: '3px 8px', borderRadius: 4,
                    border: '1px solid ' + (active ? 'var(--color-accent, #3b82f6)' : 'var(--color-border)'),
                    background: active
                      ? 'var(--color-accent, #3b82f6)'
                      : (inRange ? 'var(--color-glass)' : 'transparent'),
                    color: active ? '#fff' : (inRange ? 'var(--color-text-primary)' : 'var(--color-text-tertiary)'),
                    fontSize: 10, fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  {l}
                </button>
              );
            })}
            {filteredLabels.length === 0 && (
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                No matching classes.
              </span>
            )}
          </div>
        </div>

        {/* Class summary */}
        {classMix && (
          <div
            style={{
              padding: '14px 24px', borderBottom: '1px solid var(--color-border)',
              background: 'var(--color-bg-primary)', flexShrink: 0,
            }}
          >
            <div
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                marginBottom: 8, gap: 12, flexWrap: 'wrap',
              }}
            >
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
                Class strategy ({(heroFreqInRange * 100).toFixed(1)}% in range
                {classEv !== undefined ? ` · class EV ${classEv.toFixed(2)}` : ''})
              </div>
            </div>
            <ClassMixBar mix={classMix} />
          </div>
        )}

        {/* Caveat strip */}
        <div
          style={{
            padding: '8px 24px',
            background: 'var(--color-bg-tertiary)', flexShrink: 0,
            display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 11, color: 'var(--color-text-tertiary)',
            borderBottom: '1px solid var(--color-border)',
          }}
        >
          <AlertCircle size={12} />
          Strategy &amp; EV are class-shared. Blocker analysis is per-combo —
          use it to break mixed-strategy ties.
        </div>

        {/* Combo grid */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: 12,
            }}
          >
            {drilled.map((d) => (
              <ComboCard key={d.combo.combo} drilled={d} classMix={classMix} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Class strategy bar
// ─────────────────────────────────────────────────────────────────────

function ClassMixBar({ mix }: { mix: ComboStrategy }) {
  const entries = Object.entries(mix)
    .filter(([k, v]) => k !== 'Not in range' && v > 0.005)
    .sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return (
      <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
        Not in range.
      </div>
    );
  }
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return (
    <div>
      <div
        style={{
          display: 'flex', height: 14, borderRadius: 4, overflow: 'hidden',
          border: '1px solid var(--color-glass-border)',
          marginBottom: 6,
        }}
      >
        {entries.map(([action, freq]) => {
          const pct = (freq / total) * 100;
          return (
            <div
              key={action}
              title={`${action} ${pct.toFixed(1)}%`}
              style={{
                width: `${pct}%`,
                background: getActionColor(action),
                minWidth: 1,
              }}
            />
          );
        })}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {entries.map(([action, freq]) => {
          const pct = (freq / total) * 100;
          return (
            <span
              key={action}
              style={{
                padding: '2px 7px', borderRadius: 999,
                background: getActionColor(action), color: '#fff',
                fontSize: 10, fontWeight: 700,
              }}
            >
              {action} {pct.toFixed(0)}%
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Per-combo card
// ─────────────────────────────────────────────────────────────────────

function ComboCard({ drilled, classMix }: { drilled: DrilledCombo; classMix?: ComboStrategy }) {
  const { combo, blocker } = drilled;
  const isDead = combo.isDead;
  const pct = blocker.blockedPct;

  // Color the blocker bar by strength: low/mid/high
  const blockerColor = pct >= 8 ? '#dc2626' : pct >= 4 ? '#f59e0b' : '#10b981';

  return (
    <div
      style={{
        padding: 12,
        background: isDead ? 'rgba(0,0,0,0.25)' : 'var(--color-glass)',
        border: '1px solid var(--color-glass-border)',
        borderRadius: 10,
        opacity: isDead ? 0.5 : 1,
        display: 'flex', flexDirection: 'column', gap: 8,
      }}
    >
      {/* Card icons + dead label */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div
          style={{
            fontSize: 18, fontWeight: 800,
            fontFamily: 'var(--font-mono, monospace)',
            display: 'flex', alignItems: 'center', gap: 2,
          }}
        >
          <span>{combo.rank1}</span>
          <span style={{ color: SUIT_COLORS[combo.suit1 as keyof typeof SUIT_COLORS] || '#fff', fontSize: 16 }}>
            {SUIT_SYMBOLS[combo.suit1 as keyof typeof SUIT_SYMBOLS] || combo.suit1}
          </span>
          <span style={{ marginLeft: 4 }}>{combo.rank2}</span>
          <span style={{ color: SUIT_COLORS[combo.suit2 as keyof typeof SUIT_COLORS] || '#fff', fontSize: 16 }}>
            {SUIT_SYMBOLS[combo.suit2 as keyof typeof SUIT_SYMBOLS] || combo.suit2}
          </span>
        </div>
        {isDead && (
          <span
            style={{
              padding: '2px 8px', borderRadius: 4,
              background: 'rgba(220,38,38,0.2)', color: '#fca5a5',
              fontSize: 9, fontWeight: 700, letterSpacing: 0.5,
            }}
          >
            DEAD
          </span>
        )}
      </div>

      {!isDead && (
        <>
          {/* Blocker bar */}
          <div>
            <div
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                fontSize: 10, color: 'var(--color-text-secondary)', marginBottom: 3,
              }}
            >
              <span>Blocks opp range</span>
              <span style={{ color: blockerColor, fontWeight: 700, fontSize: 12 }}>
                {pct.toFixed(1)}%
              </span>
            </div>
            <div
              style={{
                height: 6, borderRadius: 3,
                background: 'var(--color-bg-tertiary)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(pct * 5, 100)}%`,  // ×5 amp so ~20% maxes
                  background: blockerColor,
                }}
              />
            </div>
          </div>

          {/* Top blocked classes */}
          {blocker.topBlocked.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>
              <div style={{ marginBottom: 3 }}>Top blocked:</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {blocker.topBlocked.slice(0, 3).map((b) => (
                  <span
                    key={b.label}
                    title={`Blocks ${b.blockedCount} of ${b.label} (opp weight ${(b.weight * 100).toFixed(0)}%)`}
                    style={{
                      padding: '1px 6px', borderRadius: 3,
                      background: 'var(--color-bg-tertiary)',
                      color: 'var(--color-text-secondary)',
                      fontSize: 10, fontWeight: 600,
                    }}
                  >
                    {b.label} ×{b.blockedCount}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Class-shared mini strategy bar */}
          {classMix && <MiniMixBar mix={classMix} />}
        </>
      )}

      {isDead && (
        <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>
          Card on board — not playable.
        </div>
      )}
    </div>
  );
}

function MiniMixBar({ mix }: { mix: ComboStrategy }) {
  const entries = Object.entries(mix)
    .filter(([k, v]) => k !== 'Not in range' && v > 0.005);
  if (entries.length === 0) return null;
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return (
    <div
      title="Strategy shared with class"
      style={{
        display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden',
        background: 'var(--color-bg-tertiary)', opacity: 0.65,
      }}
    >
      {entries.map(([action, freq]) => (
        <div
          key={action}
          style={{
            width: `${(freq / total) * 100}%`,
            background: getActionColor(action),
            minWidth: 1,
          }}
        />
      ))}
    </div>
  );
}
