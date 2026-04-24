/**
 * RunoutPicker — Path B UI: picks which canonical runout to view.
 *
 * Renders only when the current strategy node has cached runout options
 * (i.e. iso enumeration was engaged at the immediate prior chance). The
 * currently-active runout is highlighted; clicking another swaps the cache
 * lookup to that runout's strategies.
 *
 * Cards are color-coded by suit (♣=green, ♦=blue, ♥=red, ♠=black-on-light).
 */
import React from 'react';
import type { RunoutOption } from '../lib/poker';

interface Props {
  /** The runout options at this node. From `result.runout_options`. */
  options: RunoutOption[];
  /** Currently active runout, e.g. "2c". From `result.dealt_cards[last]`. */
  active: string | null;
  /** Called with the new card when the user picks a different runout. */
  onPick: (card: string) => void;
}

const SUIT_COLOR: Record<string, string> = {
  c: '#16a34a',  // clubs - green
  d: '#2563eb',  // diamonds - blue
  h: '#dc2626',  // hearts - red
  s: '#1f2937',  // spades - dark gray
};

const SUIT_GLYPH: Record<string, string> = {
  c: '\u2663', d: '\u2666', h: '\u2665', s: '\u2660',
};

export function RunoutPicker({ options, active, onPick }: Props) {
  if (!options || options.length === 0) return null;

  return (
    <div style={{
      padding: '8px 12px',
      background: 'var(--color-glass, rgba(255,255,255,0.05))',
      borderRadius: 6,
      marginBottom: 8,
    }}>
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary, #9ca3af)', marginBottom: 6 }}>
        Runout shown: <strong style={{ color: '#fff' }}>{active ?? '(default)'}</strong>
        {' '}— click below to view a different canonical runout
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {options.map(opt => {
          const isActive = opt.card === active;
          const rank = opt.card[0];
          const suit = opt.card[1];
          return (
            <button
              key={opt.card}
              onClick={() => onPick(opt.card)}
              title={`${opt.card} — represents ${opt.weight} runout${opt.weight > 1 ? 's' : ''}`}
              style={{
                minWidth: 36, padding: '4px 6px',
                background: isActive ? '#fff' : '#fff',
                border: isActive ? '2px solid #2563eb' : '1px solid #d1d5db',
                color: SUIT_COLOR[suit] ?? '#000',
                fontWeight: 600, fontSize: 13, cursor: 'pointer',
                borderRadius: 4, fontFamily: 'inherit',
                position: 'relative',
              }}
            >
              {rank}{SUIT_GLYPH[suit] ?? suit}
              {opt.weight > 1 && (
                <span style={{
                  position: 'absolute', top: -6, right: -6,
                  background: '#7c3aed', color: '#fff', fontSize: 9,
                  borderRadius: 8, padding: '1px 4px', minWidth: 14, textAlign: 'center',
                }}>×{opt.weight}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
