import React, { useMemo } from 'react';
import { useT } from '../lib/i18n';

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const SUITS = [
  { key: 's', symbol: '♠', color: '#E8E8E8' },
  { key: 'h', symbol: '♥', color: '#FF453A' },
  { key: 'd', symbol: '♦', color: '#0A84FF' },
  { key: 'c', symbol: '♣', color: '#30D158' },
];

interface Props {
  /** Which street card is being selected */
  street: 'turn' | 'river';
  /** Cards already on the board (e.g. "AsKd7c" or "AsKd7c2h") */
  currentBoard: string;
  /** Callback when a card is selected */
  onCardSelect: (card: string) => void;
  /** Callback to cancel / go back */
  onCancel?: () => void;
}

/** Parse board string into individual card strings */
function parseBoardCards(board: string): string[] {
  const cards: string[] = [];
  for (let i = 0; i < board.length; i += 2) {
    cards.push(board.substring(i, i + 2));
  }
  return cards;
}

/**
 * TurnRiverCardSelector — displays a card picker for dealing Turn or River.
 *
 * Shows all 52 cards as a 13×4 grid (ranks × suits), with dealt cards disabled.
 * Designed to match the aesthetic of BoardSelector.
 */
export function TurnRiverCardSelector({ street, currentBoard, onCardSelect, onCancel }: Props) {
  const t = useT();
  const usedCards = useMemo(() => {
    const set = new Set<string>();
    parseBoardCards(currentBoard).forEach(c => set.add(c));
    return set;
  }, [currentBoard]);

  const streetLabel = street === 'turn' ? t('board.turn') : t('board.river');
  const streetColor = street === 'turn' ? '#FF9F0A' : '#FF453A';

  return (
    <div
      className="glass-panel animate-fade-in"
      style={{ padding: 16 }}
    >
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 24, height: 24, borderRadius: 6,
            background: `${streetColor}25`,
            color: streetColor, fontSize: 12, fontWeight: 800,
          }}>
            {street === 'turn' ? '4' : '5'}
          </span>
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-primary)' }}>
            {t('deal')} {streetLabel}
          </span>
        </div>

        {onCancel && (
          <button
            onClick={onCancel}
            style={{
              background: 'rgba(99,99,102,0.15)',
              border: '1px solid rgba(99,99,102,0.3)',
              borderRadius: 6, padding: '4px 10px',
              color: 'var(--color-text-secondary)', fontSize: 11, fontWeight: 600,
              cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            {t('cancel')}
          </button>
        )}
      </div>

      {/* Prompt */}
      <div style={{
        fontSize: 11, color: 'var(--color-text-tertiary)',
        marginBottom: 12, lineHeight: 1.4,
      }}>
        {t('deal.selectCard', { street: streetLabel })}
      </div>

      {/* Current board display */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        marginBottom: 14, padding: '8px 10px',
        background: 'rgba(0,0,0,0.2)', borderRadius: 8,
      }}>
        <span style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginRight: 4 }}>
          {t('deal.board')}:
        </span>
        {parseBoardCards(currentBoard).map((card, i) => {
          const rank = card[0];
          const suit = SUITS.find(s => s.key === card[1]);
          return (
            <span
              key={i}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 1,
                padding: '2px 5px', borderRadius: 4,
                background: 'rgba(255,255,255,0.08)',
                fontSize: 12, fontWeight: 700,
                color: suit?.color || '#fff',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {rank}{suit?.symbol}
            </span>
          );
        })}
        <span style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 28, height: 24, borderRadius: 4,
          border: `2px dashed ${streetColor}50`,
          color: streetColor, fontSize: 11, fontWeight: 700,
        }}>
          ?
        </span>
      </div>

      {/* Card grid: 13 columns (ranks) × 4 rows (suits) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(13, 1fr)',
        gap: 3,
      }}>
        {SUITS.map((suit) =>
          RANKS.map((rank) => {
            const card = `${rank}${suit.key}`;
            const isUsed = usedCards.has(card);

            return (
              <button
                key={card}
                onClick={() => !isUsed && onCardSelect(card)}
                disabled={isUsed}
                style={{
                  display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  padding: '6px 2px',
                  background: isUsed
                    ? 'rgba(99,99,102,0.1)'
                    : 'rgba(255,255,255,0.04)',
                  border: isUsed
                    ? '1px solid transparent'
                    : '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 5,
                  cursor: isUsed ? 'not-allowed' : 'pointer',
                  opacity: isUsed ? 0.2 : 1,
                  transition: 'all 120ms ease',
                  fontFamily: 'var(--font-mono)',
                  minWidth: 0,
                }}
                onMouseOver={(e) => {
                  if (!isUsed) {
                    (e.currentTarget as HTMLButtonElement).style.background = `${suit.color}20`;
                    (e.currentTarget as HTMLButtonElement).style.borderColor = `${suit.color}60`;
                    (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1.1)';
                  }
                }}
                onMouseOut={(e) => {
                  if (!isUsed) {
                    (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.04)';
                    (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.08)';
                    (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1)';
                  }
                }}
              >
                <span style={{
                  fontSize: 11, fontWeight: 700,
                  color: isUsed ? 'var(--color-text-tertiary)' : '#fff',
                  lineHeight: 1,
                }}>
                  {rank}
                </span>
                <span style={{
                  fontSize: 10, lineHeight: 1,
                  color: isUsed ? 'var(--color-text-tertiary)' : suit.color,
                }}>
                  {suit.symbol}
                </span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
