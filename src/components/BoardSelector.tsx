import React from 'react';
import { RANKS, SUITS, SUIT_SYMBOLS, SUIT_COLORS } from '../lib/poker';
import type { Rank, Suit } from '../lib/poker';
import { useT } from '../lib/i18n';

interface Props {
  board: string;
  onBoardChange: (board: string) => void;
}

// Suit display order and label (GTO Wizard convention: s, h, d, c top-to-bottom)
const SUIT_ORDER: Suit[] = ['s', 'h', 'd', 'c'];

const SUIT_LABEL_STYLE: Record<Suit, React.CSSProperties> = {
  s: { color: '#EBEBF5' },
  h: { color: '#FF453A' },
  d: { color: '#0A84FF' },
  c: { color: '#30D158' },
};

/**
 * Board Selector — GTO Wizard-style card picker.
 *
 * Design patterns from GTO Wizard:
 * - Card picker is ALWAYS visible (not a popup toggle)
 * - 4 rows (suits) × 13 columns (ranks) grid
 * - Suit symbols as row labels, rank letters as column headers
 * - Click to add card to board, click board card to remove it
 * - Dead cards (already on board) are dimmed
 * - Board cards shown prominently above the picker
 * - Street separators between Flop | Turn | River
 * - Random Flop button for quick study
 */
export function BoardSelector({ board, onBoardChange }: Props) {
  const t = useT();
  // Parse board string into card pairs
  const boardCards: string[] = [];
  for (let i = 0; i < board.length; i += 2) {
    boardCards.push(board.substring(i, i + 2));
  }

  const deadCards = new Set(boardCards);

  // ---- Actions ----
  const addCard = (rank: Rank, suit: Suit) => {
    const card = `${rank}${suit}`;
    if (deadCards.has(card) || boardCards.length >= 5) return;
    onBoardChange(board + card);
  };

  const removeCard = (index: number) => {
    // Remove a specific card and all cards after it (like GTO Wizard: removing turn also removes river)
    onBoardChange(board.substring(0, index * 2));
  };

  const clearBoard = () => onBoardChange('');

  const randomFlop = () => {
    const deck: string[] = [];
    for (const r of RANKS) {
      for (const s of SUITS) {
        deck.push(`${r}${s}`);
      }
    }
    // Fisher-Yates shuffle
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    onBoardChange(deck[0] + deck[1] + deck[2]);
  };

  // ---- Street label ----
  const getStreetLabel = () => {
    if (boardCards.length < 3) return t('board.selectFlop');
    if (boardCards.length === 3) return t('board.flop');
    if (boardCards.length === 4) return t('board.turn');
    return t('board.river');
  };

  const getNextStreet = () => {
    if (boardCards.length < 3) return t('board.selectMore', { n: 3 - boardCards.length });
    if (boardCards.length === 3) return t('board.selectTurn');
    if (boardCards.length === 4) return t('board.selectRiver');
    return t('board.complete');
  };

  return (
    <div className="glass-panel" style={{ padding: 16 }}>
      {/* Header row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>{t('board')}</span>
          <span style={{
            fontSize: 11, fontWeight: 600,
            color: boardCards.length >= 3 ? 'var(--color-green)' : 'var(--color-accent)',
          }}>
            {getStreetLabel()}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn-secondary" onClick={randomFlop}
            style={{ padding: '4px 10px', fontSize: 11, gap: 4 }}
            title="Generate a random flop"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M1 4v6h6" /><path d="M23 20v-6h-6" />
              <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
            </svg>
            {t('board.random')}
          </button>
          <button className="btn-secondary" onClick={clearBoard}
            style={{ padding: '4px 10px', fontSize: 11 }}>
            {t('board.clear')}
          </button>
        </div>
      </div>

      {/* ================================================================
          Board Cards Display (GTO Wizard style: large, prominent)
          ================================================================ */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        marginBottom: 14, minHeight: 72,
        padding: '10px 0',
      }}>
        {/* Flop (3 cards) */}
        {[0, 1, 2].map((i) => (
          <div
            key={`flop-${i}`}
            onClick={() => boardCards[i] && removeCard(i)}
            style={{
              width: 56, height: 72,
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              background: boardCards[i]
                ? 'linear-gradient(145deg, #3A3A3C, #2C2C2E)'
                : 'var(--color-bg-tertiary)',
              border: boardCards[i]
                ? '2px solid var(--color-accent)'
                : '2px dashed rgba(255,255,255,0.15)',
              borderRadius: 'var(--radius-md)',
              cursor: boardCards[i] ? 'pointer' : 'default',
              transition: 'all 200ms ease',
              position: 'relative',
              userSelect: 'none',
            }}
            title={boardCards[i] ? t('board.clickToRemove') : undefined}
          >
            {boardCards[i] ? (
              <>
                <span style={{
                  fontSize: 22, fontWeight: 800, lineHeight: 1,
                  color: SUIT_COLORS[boardCards[i][1] as Suit],
                }}>
                  {boardCards[i][0]}
                </span>
                <span style={{
                  fontSize: 16, lineHeight: 1, marginTop: 2,
                  color: SUIT_COLORS[boardCards[i][1] as Suit],
                }}>
                  {SUIT_SYMBOLS[boardCards[i][1] as Suit]}
                </span>
              </>
            ) : (
              <span style={{ color: 'var(--color-text-tertiary)', fontSize: 14, opacity: 0.5 }}>
                {i + 1}
              </span>
            )}
          </div>
        ))}

        {/* Vertical separator before Turn */}
        <div style={{
          width: 1, height: 48,
          background: 'var(--color-glass-border)',
          margin: '0 4px',
          opacity: boardCards.length >= 3 ? 1 : 0.3,
        }} />

        {/* Turn */}
        <div
          onClick={() => boardCards[3] && removeCard(3)}
          style={{
            width: 56, height: 72,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            background: boardCards[3]
              ? 'linear-gradient(145deg, #3A3A3C, #2C2C2E)'
              : 'var(--color-bg-tertiary)',
            border: boardCards[3]
              ? '2px solid var(--color-accent)'
              : '2px dashed rgba(255,255,255,0.10)',
            borderRadius: 'var(--radius-md)',
            cursor: boardCards[3] ? 'pointer' : 'default',
            opacity: boardCards.length >= 3 ? 1 : 0.4,
            transition: 'all 200ms ease',
            userSelect: 'none',
          }}
        >
          {boardCards[3] ? (
            <>
              <span style={{
                fontSize: 22, fontWeight: 800, lineHeight: 1,
                color: SUIT_COLORS[boardCards[3][1] as Suit],
              }}>
                {boardCards[3][0]}
              </span>
              <span style={{
                fontSize: 16, lineHeight: 1, marginTop: 2,
                color: SUIT_COLORS[boardCards[3][1] as Suit],
              }}>
                {SUIT_SYMBOLS[boardCards[3][1] as Suit]}
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--color-text-tertiary)', fontSize: 10, opacity: 0.5 }}>T</span>
          )}
        </div>

        {/* Vertical separator before River */}
        <div style={{
          width: 1, height: 48,
          background: 'var(--color-glass-border)',
          margin: '0 4px',
          opacity: boardCards.length >= 4 ? 1 : 0.2,
        }} />

        {/* River */}
        <div
          onClick={() => boardCards[4] && removeCard(4)}
          style={{
            width: 56, height: 72,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            background: boardCards[4]
              ? 'linear-gradient(145deg, #3A3A3C, #2C2C2E)'
              : 'var(--color-bg-tertiary)',
            border: boardCards[4]
              ? '2px solid var(--color-accent)'
              : '2px dashed rgba(255,255,255,0.08)',
            borderRadius: 'var(--radius-md)',
            cursor: boardCards[4] ? 'pointer' : 'default',
            opacity: boardCards.length >= 4 ? 1 : 0.3,
            transition: 'all 200ms ease',
            userSelect: 'none',
          }}
        >
          {boardCards[4] ? (
            <>
              <span style={{
                fontSize: 22, fontWeight: 800, lineHeight: 1,
                color: SUIT_COLORS[boardCards[4][1] as Suit],
              }}>
                {boardCards[4][0]}
              </span>
              <span style={{
                fontSize: 16, lineHeight: 1, marginTop: 2,
                color: SUIT_COLORS[boardCards[4][1] as Suit],
              }}>
                {SUIT_SYMBOLS[boardCards[4][1] as Suit]}
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--color-text-tertiary)', fontSize: 10, opacity: 0.5 }}>R</span>
          )}
        </div>
      </div>

      {/* Hint text */}
      <div style={{
        fontSize: 11, color: 'var(--color-text-tertiary)',
        marginBottom: 12, textAlign: 'center',
      }}>
        {getNextStreet()}
        {boardCards.length > 0 && ` · ${t('board.clickToRemove')}`}
      </div>

      {/* ================================================================
          Card Picker Grid (ALWAYS visible, GTO Wizard 4×13 layout)
          ================================================================ */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {/* Rank column headers */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '28px repeat(13, 1fr)',
          gap: 2,
          marginBottom: 2,
        }}>
          <div /> {/* Empty corner */}
          {RANKS.map((rank) => (
            <div key={rank} style={{
              textAlign: 'center',
              fontSize: 11, fontWeight: 700,
              color: 'var(--color-text-secondary)',
              padding: '2px 0',
            }}>
              {rank}
            </div>
          ))}
        </div>

        {/* 4 suit rows × 13 rank columns */}
        {SUIT_ORDER.map((suit) => (
          <div
            key={suit}
            style={{
              display: 'grid',
              gridTemplateColumns: '28px repeat(13, 1fr)',
              gap: 2,
              marginBottom: 2,
            }}
          >
            {/* Suit label */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 16, ...SUIT_LABEL_STYLE[suit],
            }}>
              {SUIT_SYMBOLS[suit]}
            </div>

            {/* 13 rank buttons */}
            {RANKS.map((rank) => {
              const card = `${rank}${suit}`;
              const isDead = deadCards.has(card);
              const isNextSlot = boardCards.length < 5;

              return (
                <button
                  key={card}
                  onClick={() => addCard(rank, suit)}
                  disabled={isDead || !isNextSlot}
                  style={{
                    height: 34,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: isDead
                      ? 'rgba(10, 132, 255, 0.15)'
                      : 'var(--color-bg-tertiary)',
                    border: isDead
                      ? '1.5px solid var(--color-accent)'
                      : '1.5px solid transparent',
                    borderRadius: 4,
                    color: isDead
                      ? 'var(--color-accent)'
                      : SUIT_COLORS[suit],
                    fontSize: 13,
                    fontWeight: 700,
                    cursor: isDead || !isNextSlot ? 'default' : 'pointer',
                    opacity: isDead ? 0.5 : (!isNextSlot ? 0.4 : 1),
                    transition: 'all 120ms ease',
                    padding: 0,
                    fontFamily: 'inherit',
                  }}
                  onMouseOver={(e) => {
                    if (!isDead && isNextSlot) {
                      (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.12)';
                      (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1.08)';
                    }
                  }}
                  onMouseOut={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background = isDead
                      ? 'rgba(10, 132, 255, 0.15)'
                      : 'var(--color-bg-tertiary)';
                    (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1)';
                  }}
                >
                  {rank}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
