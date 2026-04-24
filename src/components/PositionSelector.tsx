import React, { useState, useMemo } from 'react';
import { MATCHUPS, rangePercentage, countRangeCombos } from '../lib/ranges';
import type { Position, PotType, PositionMatchup } from '../lib/ranges';
import { useT } from '../lib/i18n';

interface Props {
  selectedMatchup: PositionMatchup | null;
  onMatchupChange: (matchup: PositionMatchup, heroPos: Position) => void;
  onReset?: () => void;
  onEditRange?: (isIP: boolean) => void;
}

const POSITION_COLORS: Record<string, string> = {
  UTG: '#FF6B6B', MP: '#FFA94D', CO: '#FFD43B',
  BTN: '#69DB7C', SB: '#74C0FC', BB: '#B197FC',
};

const ALL_POSITIONS: Position[] = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];

function getOpponents(hero: Position, potType: PotType) {
  const filtered = MATCHUPS.filter(m => m.potType === potType);
  const results: { opponent: Position; matchup: PositionMatchup; heroIsIP: boolean }[] = [];
  for (const m of filtered) {
    if (m.ip === hero) results.push({ opponent: m.oop, matchup: m, heroIsIP: true });
    else if (m.oop === hero) results.push({ opponent: m.ip, matchup: m, heroIsIP: false });
  }
  return results;
}

export function PositionSelector({ selectedMatchup, onMatchupChange, onReset, onEditRange }: Props) {
  const t = useT();
  const [heroPosition, setHeroPosition] = useState<Position | null>(null);
  const [potType, setPotType] = useState<PotType>('SRP');

  const positionsWithMatchups = useMemo(() => {
    const set = new Set<Position>();
    for (const m of MATCHUPS.filter(m => m.potType === potType)) {
      set.add(m.ip);
      set.add(m.oop);
    }
    return set;
  }, [potType]);

  const opponents = useMemo(() => {
    if (!heroPosition) return [];
    return getOpponents(heroPosition, potType);
  }, [heroPosition, potType]);

  const handlePotTypeChange = (type: PotType) => {
    setPotType(type);
    setHeroPosition(null);
  };

  const handleHeroSelect = (pos: Position) => {
    if (heroPosition === pos) { setHeroPosition(null); return; }
    setHeroPosition(pos);
  };

  const handleOpponentSelect = (opp: { opponent: Position; matchup: PositionMatchup; heroIsIP: boolean }) => {
    onMatchupChange(opp.matchup, heroPosition!);
  };

  const handleReset = () => { setHeroPosition(null); onReset?.(); };

  return (
    <div className="glass-panel" style={{ padding: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>{t('position')}</span>
          <span style={{
            fontSize: 11, fontWeight: 600,
            color: selectedMatchup ? 'var(--color-green)' : 'var(--color-text-tertiary)',
          }}>
            {selectedMatchup ? selectedMatchup.label : t('position.6max')}
          </span>
        </div>
        {heroPosition && (
          <button className="btn-secondary" onClick={handleReset}
            style={{ padding: '3px 8px', fontSize: 10 }}>{t('reset')}</button>
        )}
      </div>

      {/* Pot Type Tabs */}
      <div style={{
        display: 'flex', gap: 4, marginBottom: 12,
        background: 'var(--color-bg-tertiary)', borderRadius: 8, padding: 3,
      }}>
        {(['SRP', '3BET'] as PotType[]).map(type => (
          <button key={type} onClick={() => handlePotTypeChange(type)}
            style={{
              flex: 1, padding: '6px 0', borderRadius: 6, cursor: 'pointer',
              fontFamily: 'inherit', fontSize: 11, fontWeight: 600, transition: 'all 150ms ease',
              background: potType === type
                ? (type === 'SRP' ? 'rgba(10,132,255,0.25)' : 'rgba(191,90,242,0.25)')
                : 'transparent',
              color: potType === type
                ? (type === 'SRP' ? 'var(--color-accent)' : 'var(--color-purple)')
                : 'var(--color-text-tertiary)',
              border: potType === type
                ? `1px solid ${type === 'SRP' ? 'rgba(10,132,255,0.4)' : 'rgba(191,90,242,0.4)'}`
                : '1px solid transparent',
            }}>
            {type === 'SRP' ? t('position.srp') : t('position.3bp')}
          </button>
        ))}
      </div>

      {/* Step 1: Select Hero */}
      {!heroPosition && (
        <div className="animate-fade-in">
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 10, fontWeight: 500 }}>
            {t('position.select')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 6 }}>
            {ALL_POSITIONS.map(pos => {
              const has = positionsWithMatchups.has(pos);
              return (
                <button key={pos} onClick={() => has && handleHeroSelect(pos)} disabled={!has}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                    padding: '12px 4px', background: 'var(--color-bg-tertiary)',
                    border: '2px solid transparent', borderRadius: 8,
                    cursor: has ? 'pointer' : 'not-allowed', opacity: has ? 1 : 0.35,
                    transition: 'all 150ms ease', fontFamily: 'inherit',
                  }}
                  onMouseOver={e => { if (has) { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.borderColor = POSITION_COLORS[pos]; }}}
                  onMouseOut={e => { e.currentTarget.style.background = 'var(--color-bg-tertiary)'; e.currentTarget.style.borderColor = 'transparent'; }}>
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    width: 38, height: 24, borderRadius: 6, background: POSITION_COLORS[pos],
                    color: '#000', fontSize: 12, fontWeight: 800,
                  }}>{pos}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Step 2: Select Opponent */}
      {heroPosition && !selectedMatchup && (
        <div className="animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)', fontWeight: 500 }}>{t('position.hero')}:</span>
            <span style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 42, height: 24, borderRadius: 6, background: POSITION_COLORS[heroPosition],
              color: '#000', fontSize: 12, fontWeight: 800,
            }}>{heroPosition}</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 10, fontWeight: 500 }}>
            {t('position.selectOpp')}
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: opponents.length <= 2 ? 'repeat(2, 1fr)' : 'repeat(3, 1fr)',
            gap: 6,
          }}>
            {opponents.map(({ opponent, matchup, heroIsIP }) => (
              <button key={opponent} onClick={() => handleOpponentSelect({ opponent, matchup, heroIsIP })}
                style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                  padding: '12px 8px', background: 'var(--color-bg-tertiary)',
                  border: '2px solid transparent', borderRadius: 8, cursor: 'pointer',
                  transition: 'all 150ms ease', fontFamily: 'inherit',
                }}
                onMouseOver={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.borderColor = POSITION_COLORS[opponent]; }}
                onMouseOut={e => { e.currentTarget.style.background = 'var(--color-bg-tertiary)'; e.currentTarget.style.borderColor = 'transparent'; }}>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 42, height: 24, borderRadius: 6, background: POSITION_COLORS[opponent],
                  color: '#000', fontSize: 12, fontWeight: 800,
                }}>{opponent}</span>
                <span style={{ fontSize: 9, fontWeight: 600, color: heroIsIP ? 'var(--color-green)' : 'var(--color-orange)' }}>
                  {heroIsIP ? t('position.heroIP') : t('position.heroOOP')}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Selected Matchup Details */}
      {selectedMatchup && heroPosition && (
        <div className="animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 14 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 44, height: 26, borderRadius: 6, background: POSITION_COLORS[heroPosition],
              color: '#000', fontSize: 12, fontWeight: 800,
            }}>{heroPosition}</span>
            <span style={{ fontSize: 14, fontWeight: 300, color: 'var(--color-text-tertiary)' }}>vs</span>
            {(() => {
              const villain = selectedMatchup.ip === heroPosition ? selectedMatchup.oop : selectedMatchup.ip;
              return (
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 44, height: 26, borderRadius: 6, background: POSITION_COLORS[villain],
                  color: '#000', fontSize: 12, fontWeight: 800,
                }}>{villain}</span>
              );
            })()}
          </div>

          {/* Pot type badge */}
          <div style={{ textAlign: 'center', marginBottom: 10 }}>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
              background: selectedMatchup.potType === 'SRP' ? 'rgba(10,132,255,0.2)' : 'rgba(191,90,242,0.2)',
              color: selectedMatchup.potType === 'SRP' ? 'var(--color-accent)' : 'var(--color-purple)',
            }}>
              {selectedMatchup.potType === 'SRP' ? t('position.srpPot') : t('position.3bpPot')}
            </span>
          </div>

          <div style={{ padding: 12, background: 'var(--color-bg-tertiary)', borderRadius: 8 }}>
            <RangeBar position={selectedMatchup.ip} label={t('position.ipRaiser')}
              rangeStr={selectedMatchup.ipRange} isHero={selectedMatchup.ip === heroPosition}
              onEdit={() => onEditRange?.(true)} />
            <div style={{ height: 10 }} />
            <RangeBar position={selectedMatchup.oop}
              label={selectedMatchup.potType === '3BET' ? t('position.oop3bettor') : t('position.oopCaller')}
              rangeStr={selectedMatchup.oopRange} isHero={selectedMatchup.oop === heroPosition}
              onEdit={() => onEditRange?.(false)} />
          </div>
        </div>
      )}
    </div>
  );
}

function RangeBar({ position, label, rangeStr, isHero, onEdit }: {
  position: Position; label: string; rangeStr: string; isHero: boolean; onEdit?: () => void;
}) {
  const t = useT();
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 34, height: 20, borderRadius: 4, background: POSITION_COLORS[position],
            color: '#000', fontSize: 10, fontWeight: 800,
          }}>{position}</span>
          <span style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontWeight: 500 }}>{label}</span>
          {isHero && (
            <span style={{
              fontSize: 9, fontWeight: 700, color: 'var(--color-accent)',
              background: 'var(--color-blue-dim)', padding: '1px 5px', borderRadius: 3,
            }}>YOU</span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="text-mono" style={{ fontSize: 11, fontWeight: 600, color: POSITION_COLORS[position] }}>
            {rangePercentage(rangeStr)}% ({countRangeCombos(rangeStr)})
          </span>
          {onEdit && (
            <button
              onClick={onEdit}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                padding: '3px 10px', borderRadius: 6, cursor: 'pointer',
                background: 'var(--color-accent)', color: '#fff',
                fontSize: 11, fontWeight: 700, border: 'none',
                transition: 'all 150ms ease', fontFamily: 'inherit',
                letterSpacing: '0.02em',
              }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--color-accent-hover)'; e.currentTarget.style.boxShadow = '0 2px 8px rgba(10,132,255,0.4)'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'var(--color-accent)'; e.currentTarget.style.boxShadow = 'none'; }}
            >
              {t('edit')}
            </button>
          )}
        </div>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div style={{
          height: '100%', borderRadius: 3, width: `${Math.min(100, rangePercentage(rangeStr))}%`,
          background: POSITION_COLORS[position], transition: 'width 300ms ease',
        }} />
      </div>
    </div>
  );
}
