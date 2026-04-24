import React, { useState, useMemo } from 'react';
import { X, Info, AlertTriangle } from 'lucide-react';
import { useT } from '../lib/i18n';
import { parseBoardCards, SUIT_SYMBOLS, SUIT_COLORS, getActionColor } from '../lib/poker';
import type { Suit } from '../lib/poker';
import { MATCHUPS } from '../lib/ranges';
import type { PotType } from '../lib/ranges';
import {
  BOARD_TEMPLATES,
  getSpotsByMatchup,
  isRealSolverAvailable,
} from '../lib/presolvedSpots';
import type { PresolvedSpot } from '../lib/presolvedSpots';

// ============================================================================
// Props
// ============================================================================

interface Props {
  onSelectSpot: (spot: PresolvedSpot) => void;
  onClose: () => void;
}

// ============================================================================
// Helpers
// ============================================================================

/** Render a single card with rank + colored suit symbol */
function BoardCardDisplay({ board }: { board: string }) {
  const cards = parseBoardCards(board);
  return (
    <span style={{ display: 'inline-flex', gap: 3 }}>
      {cards.map((card, i) => {
        const rank = card[0];
        const suit = card[1] as Suit;
        return (
          <span
            key={i}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              fontSize: 15,
              fontWeight: 800,
              fontFamily: 'var(--font-mono, monospace)',
              letterSpacing: -0.5,
            }}
          >
            <span style={{ color: 'var(--color-text-primary)' }}>{rank}</span>
            <span style={{ color: SUIT_COLORS[suit] }}>
              {SUIT_SYMBOLS[suit]}
            </span>
          </span>
        );
      })}
    </span>
  );
}

/** Get the top N actions from a global strategy, sorted by percentage descending */
function getTopActions(
  globalStrategy: Record<string, string>,
  n: number,
): { action: string; pct: number }[] {
  return Object.entries(globalStrategy)
    .map(([action, pctStr]) => ({
      action,
      pct: parseFloat(pctStr),
    }))
    .filter((a) => a.action !== 'Not in range')
    .sort((a, b) => b.pct - a.pct)
    .slice(0, n);
}

/** Shorten action labels for the mini bar display */
function shortLabel(action: string): string {
  if (action === 'Check') return 'Ck';
  if (action === 'Call') return 'Ca';
  if (action === 'Fold') return 'Fo';
  if (action.startsWith('Bet 33')) return 'B33';
  if (action.startsWith('Bet 75')) return 'B75';
  if (action.startsWith('Bet_33')) return 'B33';
  if (action.startsWith('Bet_75')) return 'B75';
  if (action === 'All-in') return 'AI';
  if (action.startsWith('Raise')) return 'Ra';
  if (action.startsWith('Bet')) return 'Bet';
  return action.slice(0, 3);
}

// ============================================================================
// Unique matchup labels for the dropdown
// ============================================================================

interface MatchupOption {
  idx: number;
  label: string;
  potType: PotType;
}

function buildMatchupOptions(): MatchupOption[] {
  return MATCHUPS.map((m, idx) => ({
    idx,
    label: m.label,
    potType: m.potType,
  }));
}

// ============================================================================
// SpotLibrary Component
// ============================================================================

export function SpotLibrary({ onSelectSpot, onClose }: Props) {
  const t = useT();
  const allOptions = useMemo(buildMatchupOptions, []);
  const realSolver = useMemo(isRealSolverAvailable, []);

  // Filters
  const [potFilter, setPotFilter] = useState<PotType>('SRP');
  const [selectedMatchupIdx, setSelectedMatchupIdx] = useState<number>(0);

  // Filter options by pot type
  const filteredOptions = useMemo(
    () => allOptions.filter((o) => o.potType === potFilter),
    [allOptions, potFilter],
  );

  // Ensure selected matchup is valid for current pot type
  const activeMatchupIdx = useMemo(() => {
    const found = filteredOptions.find((o) => o.idx === selectedMatchupIdx);
    if (found) return found.idx;
    return filteredOptions.length > 0 ? filteredOptions[0].idx : 0;
  }, [filteredOptions, selectedMatchupIdx]);

  // Generate spots for the active matchup
  const spots = useMemo(
    () => getSpotsByMatchup(activeMatchupIdx),
    [activeMatchupIdx],
  );

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: 'rgba(0,0,0,0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        className="animate-slide-up"
        style={{
          width: '100%',
          maxWidth: 860,
          maxHeight: '90vh',
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
          borderRadius: 16,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 32px 80px rgba(0,0,0,0.5)',
          overflow: 'hidden',
        }}
      >
        {/* ---- Header ---- */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '18px 24px',
            borderBottom: '1px solid var(--color-border)',
            background: 'var(--color-bg-primary)',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 15,
                fontWeight: 800,
                color: '#fff',
              }}
            >
              S
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>
                {t('spots.title') === 'spots.title'
                  ? 'Spot Library'
                  : t('spots.title')}
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: 'var(--color-text-tertiary)',
                }}
              >
                {t('spots.subtitle') === 'spots.subtitle'
                  ? 'Pre-solved GTO strategies for common boards'
                  : t('spots.subtitle')}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              padding: 8,
              border: 'none',
              background: 'var(--color-glass)',
              borderRadius: 8,
              cursor: 'pointer',
              color: 'var(--color-text-secondary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* ---- Filters ---- */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '14px 24px',
            borderBottom: '1px solid var(--color-border)',
            background: 'var(--color-bg-primary)',
            flexShrink: 0,
            flexWrap: 'wrap',
          }}
        >
          {/* Matchup dropdown */}
          <select
            value={activeMatchupIdx}
            onChange={(e) => setSelectedMatchupIdx(Number(e.target.value))}
            style={{
              padding: '6px 12px',
              borderRadius: 8,
              border: '1px solid var(--color-glass-border)',
              background: 'var(--color-glass)',
              color: 'var(--color-text-primary)',
              fontSize: 13,
              fontWeight: 600,
              fontFamily: 'inherit',
              cursor: 'pointer',
              outline: 'none',
              minWidth: 160,
            }}
          >
            {filteredOptions.map((opt) => (
              <option key={opt.idx} value={opt.idx}>
                {opt.label}
              </option>
            ))}
          </select>

          {/* Pot type toggle */}
          <div
            style={{
              display: 'flex',
              borderRadius: 8,
              overflow: 'hidden',
              border: '1px solid var(--color-glass-border)',
            }}
          >
            {(['SRP', '3BET'] as PotType[]).map((pt) => (
              <button
                key={pt}
                onClick={() => setPotFilter(pt)}
                style={{
                  padding: '6px 16px',
                  border: 'none',
                  fontSize: 12,
                  fontWeight: 700,
                  fontFamily: 'inherit',
                  cursor: 'pointer',
                  transition: 'all 150ms ease',
                  background:
                    potFilter === pt
                      ? 'var(--color-accent)'
                      : 'var(--color-glass)',
                  color:
                    potFilter === pt
                      ? '#fff'
                      : 'var(--color-text-secondary)',
                }}
              >
                {pt === 'SRP' ? 'SRP' : '3BP'}
              </button>
            ))}
          </div>

          {/* Info: matchup details */}
          <div
            style={{
              marginLeft: 'auto',
              fontSize: 11,
              color: 'var(--color-text-tertiary)',
            }}
          >
            {MATCHUPS[activeMatchupIdx]?.oop} (OOP) vs{' '}
            {MATCHUPS[activeMatchupIdx]?.ip} (IP)
            {'  '}|{'  '}
            {t('spots.boards') === 'spots.boards'
              ? `${BOARD_TEMPLATES.length} boards`
              : t('spots.boards', { n: String(BOARD_TEMPLATES.length) })}
          </div>
        </div>

        {/* ---- Mode banner ---- */}
        {!realSolver ? (
          <div
            style={{
              padding: '10px 24px',
              background: 'rgba(255,149,0,0.08)',
              borderBottom: '1px solid rgba(255,149,0,0.25)',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              fontSize: 12,
              color: 'var(--color-orange, #FF9500)',
              fontWeight: 600,
              flexShrink: 0,
            }}
          >
            <AlertTriangle size={14} style={{ flexShrink: 0 }} />
            <span>{t('spots.demoMode')}</span>
          </div>
        ) : (
          <div
            style={{
              padding: '10px 24px',
              background: 'rgba(10,132,255,0.06)',
              borderBottom: '1px solid rgba(10,132,255,0.2)',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              fontSize: 12,
              color: 'var(--color-accent)',
              fontWeight: 500,
              flexShrink: 0,
            }}
          >
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>{t('spots.clickToSolve')}</span>
          </div>
        )}

        {/* ---- Spot Grid ---- */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px 24px',
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 14,
            }}
          >
            {spots.map((spot) => {
              const topActions = getTopActions(spot.globalStrategy, 2);
              return (
                <button
                  key={spot.id}
                  onClick={() => onSelectSpot(spot)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                    padding: 16,
                    background: 'var(--color-glass)',
                    border: '1px solid var(--color-glass-border)',
                    borderRadius: 12,
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontFamily: 'inherit',
                    transition: 'all 200ms ease',
                    color: 'var(--color-text-primary)',
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor =
                      'var(--color-accent)';
                    e.currentTarget.style.boxShadow =
                      '0 4px 20px rgba(10,132,255,0.2)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor =
                      'var(--color-glass-border)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  {/* Board cards */}
                  <BoardCardDisplay board={spot.board} />

                  {/* Texture label */}
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: 'var(--color-text-tertiary)',
                      textTransform: 'uppercase',
                      letterSpacing: 0.5,
                    }}
                  >
                    {spot.boardTexture}
                  </div>

                  {/* Mini strategy bars */}
                  {topActions.length > 0 && (
                    <div
                      style={{
                        display: 'flex',
                        gap: 6,
                        marginTop: 2,
                      }}
                    >
                      {topActions.map((a, i) => (
                        <div
                          key={i}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                          }}
                        >
                          <div
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: 2,
                              background: getActionColor(a.action),
                              flexShrink: 0,
                            }}
                          />
                          <span
                            style={{
                              fontSize: 10,
                              fontWeight: 700,
                              color: 'var(--color-text-secondary)',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {shortLabel(a.action)} {Math.round(a.pct)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* ---- Footer ---- */}
        <div
          style={{
            padding: '14px 24px',
            borderTop: '1px solid var(--color-border)',
            background: 'var(--color-bg-primary)',
            flexShrink: 0,
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: '10px 40px',
              borderRadius: 10,
              border: 'none',
              background: 'var(--color-glass)',
              color: 'var(--color-text-secondary)',
              fontSize: 14,
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 200ms ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--color-glass-border)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--color-glass)';
            }}
          >
            {t('cancel') || 'Close'}
          </button>
        </div>
      </div>
    </div>
  );
}
