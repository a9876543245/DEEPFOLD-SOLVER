import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X, RotateCcw, ArrowRight, Trophy, CheckCircle, AlertTriangle } from 'lucide-react';
import { SUIT_SYMBOLS, SUIT_COLORS, RANK_VALUES, getActionColor } from '../lib/poker';
import type { Suit } from '../lib/poker';
import { parseBoardCards } from '../lib/poker';
import {
  createDrillSession,
  scoreDrillAnswer,
  getSessionSummary,
} from '../lib/drill';
import type { DrillSession, DrillResult, DrillScenario } from '../lib/drill';
import { useT } from '../lib/i18n';

// ============================================================================
// Props
// ============================================================================

interface Props {
  onClose: () => void;
}

// ============================================================================
// Constants
// ============================================================================

const SCENARIO_COUNT = 10;

/** Action button styling config */
const ACTION_CONFIG: Record<string, { icon: string; color: string }> = {
  Check: { icon: '\u2713', color: '#30D158' },
  'Bet 33%': { icon: '\u2191', color: '#FF453A' },
  'Bet 75%': { icon: '\u21D1', color: '#FF6B35' },
  'All-in': { icon: '\u2605', color: '#FF9F0A' },
  Fold: { icon: '\u2715', color: '#636366' },
};

// ============================================================================
// Helpers
// ============================================================================

function getSuitColor(suit: string): string {
  return (SUIT_COLORS as Record<string, string>)[suit] || '#EBEBF5';
}

function getSuitSymbol(suit: string): string {
  return (SUIT_SYMBOLS as Record<string, string>)[suit] || suit;
}

/** Format seconds as M:SS */
function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Get star count (1-5) from percentage score */
function getStarRating(pct: number): number {
  if (pct >= 90) return 5;
  if (pct >= 75) return 4;
  if (pct >= 60) return 3;
  if (pct >= 40) return 2;
  return 1;
}

function getGradeLabel(stars: number): string {
  if (stars >= 5) return 'Perfect';
  if (stars >= 4) return 'Excellent';
  if (stars >= 3) return 'Good';
  if (stars >= 2) return 'Needs Work';
  return 'Keep Practicing';
}

/** Pick two specific cards for the hero combo display */
function getHeroCards(combo: string): { rank1: string; suit1: string; rank2: string; suit2: string } {
  const isPair = combo.length === 2 || (combo.length === 3 && combo[0] === combo[1] && !combo.endsWith('s') && !combo.endsWith('o'));
  const isSuited = combo.endsWith('s');
  const r1 = combo[0];
  const r2 = isPair ? combo[0] : combo[1];

  if (isPair) {
    return { rank1: r1, suit1: 'h', rank2: r2, suit2: 's' };
  } else if (isSuited) {
    return { rank1: r1, suit1: 'h', rank2: r2, suit2: 'h' };
  } else {
    return { rank1: r1, suit1: 'h', rank2: r2, suit2: 's' };
  }
}

// ============================================================================
// Sub-components
// ============================================================================

/** Rendered poker card */
function Card({ rank, suit, size = 'md' }: { rank: string; suit: string; size?: 'sm' | 'md' | 'lg' }) {
  const sizes = {
    sm: { w: 40, h: 54, rank: 15, suit: 12 },
    md: { w: 56, h: 76, rank: 22, suit: 16 },
    lg: { w: 68, h: 92, rank: 26, suit: 20 },
  };
  const s = sizes[size];

  return (
    <div style={{
      width: s.w,
      height: s.h,
      background: 'linear-gradient(145deg, #2C2C2E, #1C1C1E)',
      border: '1.5px solid rgba(255,255,255,0.15)',
      borderRadius: 10,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 1,
      boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
      userSelect: 'none',
    }}>
      <span style={{
        fontSize: s.rank,
        fontWeight: 800,
        lineHeight: 1,
        color: getSuitColor(suit),
      }}>
        {rank}
      </span>
      <span style={{
        fontSize: s.suit,
        lineHeight: 1,
        color: getSuitColor(suit),
      }}>
        {getSuitSymbol(suit)}
      </span>
    </div>
  );
}

/** Strategy frequency bar */
function StrategyBar({ action, freq, isUserPick, isOptimal }: {
  action: string;
  freq: number;
  isUserPick: boolean;
  isOptimal: boolean;
}) {
  const color = getActionColor(action);
  const pct = freq * 100;
  const barWidth = Math.max(2, pct);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '6px 0',
    }}>
      {/* Bar */}
      <div style={{
        flex: 1,
        height: 24,
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 6,
        overflow: 'hidden',
        position: 'relative',
      }}>
        <div style={{
          width: `${barWidth}%`,
          height: '100%',
          background: isUserPick || isOptimal
            ? color
            : `${color}88`,
          borderRadius: 6,
          transition: 'width 600ms cubic-bezier(0.16, 1, 0.3, 1)',
        }} />
      </div>

      {/* Label */}
      <div style={{
        width: 80,
        fontSize: 13,
        fontWeight: 600,
        color: isUserPick || isOptimal ? color : 'var(--color-text-secondary)',
        whiteSpace: 'nowrap',
      }}>
        {action}
      </div>

      {/* Percentage */}
      <div style={{
        width: 55,
        fontSize: 13,
        fontWeight: 600,
        fontFamily: 'var(--font-mono, monospace)',
        textAlign: 'right',
        color: isUserPick || isOptimal ? '#fff' : 'var(--color-text-tertiary)',
      }}>
        {pct.toFixed(1)}%
      </div>

      {/* Tag */}
      <div style={{ width: 80, fontSize: 11, fontWeight: 600 }}>
        {isOptimal && isUserPick && (
          <span style={{ color: '#30D158' }}>Your pick!</span>
        )}
        {isOptimal && !isUserPick && (
          <span style={{ color: '#FF9F0A' }}>Optimal</span>
        )}
        {!isOptimal && isUserPick && (
          <span style={{ color: 'var(--color-text-tertiary)' }}>You</span>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function DrillMode({ onClose }: Props) {
  const t = useT();

  // --- State ---
  const [session, setSession] = useState<DrillSession | null>(null);
  const [phase, setPhase] = useState<'question' | 'answer' | 'summary'>('question');
  const [selectedAction, setSelectedAction] = useState<string | null>(null);
  const [timer, setTimer] = useState(0);
  const [currentResult, setCurrentResult] = useState<DrillResult | null>(null);
  const [scoreAnim, setScoreAnim] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- Initialize session ---
  useEffect(() => {
    setSession(createDrillSession(SCENARIO_COUNT));
  }, []);

  // --- Timer ---
  useEffect(() => {
    if (phase === 'question') {
      timerRef.current = setInterval(() => {
        setTimer((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [phase]);

  // --- Handlers ---

  const handleSelectAction = useCallback((action: string) => {
    if (!session || phase !== 'question') return;

    const scenario = session.scenarios[session.currentIndex];
    const result = scoreDrillAnswer(action, scenario.correctStrategy);

    // Update session
    const newResults = [...session.results];
    newResults[session.currentIndex] = result;
    const newTotalScore = session.totalScore + result.score;

    setSession({
      ...session,
      results: newResults,
      totalScore: newTotalScore,
    });

    setSelectedAction(action);
    setCurrentResult(result);
    setPhase('answer');

    // Trigger score animation
    setScoreAnim(true);
    setTimeout(() => setScoreAnim(false), 500);
  }, [session, phase]);

  const handleNext = useCallback(() => {
    if (!session) return;

    const nextIndex = session.currentIndex + 1;
    if (nextIndex >= session.scenarios.length) {
      setPhase('summary');
      return;
    }

    setSession({ ...session, currentIndex: nextIndex });
    setPhase('question');
    setSelectedAction(null);
    setCurrentResult(null);
    setTimer(0);
  }, [session]);

  const handleRestart = useCallback(() => {
    setSession(createDrillSession(SCENARIO_COUNT));
    setPhase('question');
    setSelectedAction(null);
    setCurrentResult(null);
    setTimer(0);
  }, []);

  // --- Render guards ---
  if (!session) return null;

  const scenario = session.scenarios[session.currentIndex];
  const boardCards = parseBoardCards(scenario.board);
  const heroCards = getHeroCards(scenario.heroCombo);
  const questionNum = session.currentIndex + 1;

  // ============================================================================
  // RENDER: Question Phase
  // ============================================================================

  const renderQuestion = () => (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 32,
      padding: '40px 24px',
      maxWidth: 640,
      margin: '0 auto',
      width: '100%',
      animation: 'fadeIn 300ms ease-out',
    }}>
      {/* Matchup + Board Texture */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: 14,
          fontWeight: 600,
          color: 'var(--color-text-secondary)',
          marginBottom: 4,
        }}>
          {scenario.matchupLabel}
        </div>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span style={{
            padding: '2px 10px',
            borderRadius: 6,
            background: 'rgba(48,209,88,0.15)',
            color: '#30D158',
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}>
            Flop
          </span>
          <span style={{
            padding: '2px 10px',
            borderRadius: 6,
            background: scenario.heroPosition === 'IP'
              ? 'rgba(191,90,242,0.2)'
              : 'rgba(10,132,255,0.2)',
            color: scenario.heroPosition === 'IP'
              ? 'var(--color-purple)'
              : 'var(--color-accent)',
            fontSize: 11,
            fontWeight: 700,
          }}>
            {scenario.heroPosition}
          </span>
          <span style={{
            fontSize: 12,
            color: 'var(--color-text-tertiary)',
          }}>
            {scenario.boardTexture}
          </span>
        </div>
      </div>

      {/* Board Cards */}
      <div style={{
        display: 'flex',
        gap: 10,
        padding: '20px 32px',
        background: 'var(--color-glass)',
        backdropFilter: 'blur(40px)',
        borderRadius: 16,
        border: '1px solid var(--color-glass-border)',
      }}>
        {boardCards.map((card, i) => (
          <Card key={i} rank={card[0]} suit={card[1]} size="lg" />
        ))}
      </div>

      {/* Hero Hand */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
      }}>
        <span style={{
          fontSize: 13,
          fontWeight: 500,
          color: 'var(--color-text-tertiary)',
        }}>
          Your hand:
        </span>
        <div style={{ display: 'flex', gap: 6 }}>
          <Card rank={heroCards.rank1} suit={heroCards.suit1} size="md" />
          <Card rank={heroCards.rank2} suit={heroCards.suit2} size="md" />
        </div>
        <span style={{
          padding: '4px 12px',
          borderRadius: 8,
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.1)',
          fontSize: 15,
          fontWeight: 700,
          color: '#fff',
          fontFamily: 'var(--font-mono, monospace)',
        }}>
          {scenario.heroCombo}
        </span>
      </div>

      {/* Question prompt */}
      <div style={{
        fontSize: 18,
        fontWeight: 700,
        color: '#fff',
        letterSpacing: '-0.02em',
      }}>
        What should you do?
      </div>

      {/* Action Buttons */}
      <div style={{
        display: 'flex',
        gap: 10,
        flexWrap: 'wrap',
        justifyContent: 'center',
      }}>
        {scenario.availableActions.map((action) => {
          const cfg = ACTION_CONFIG[action] || { icon: '?', color: '#8E8E93' };
          return (
            <button
              key={action}
              onClick={() => handleSelectAction(action)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '14px 28px',
                background: `${cfg.color}18`,
                border: `1.5px solid ${cfg.color}50`,
                borderRadius: 12,
                cursor: 'pointer',
                transition: 'all 150ms ease',
                fontFamily: 'inherit',
                minWidth: 120,
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = `${cfg.color}30`;
                e.currentTarget.style.borderColor = cfg.color;
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 6px 20px ${cfg.color}25`;
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = `${cfg.color}18`;
                e.currentTarget.style.borderColor = `${cfg.color}50`;
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <span style={{ fontSize: 14, color: cfg.color }}>{cfg.icon}</span>
              <span style={{ fontSize: 15, fontWeight: 700, color: cfg.color }}>
                {action}
              </span>
            </button>
          );
        })}
      </div>

      {/* Timer */}
      <div style={{
        fontSize: 14,
        fontWeight: 500,
        color: 'var(--color-text-tertiary)',
        fontFamily: 'var(--font-mono, monospace)',
      }}>
        {formatTime(timer)}
      </div>
    </div>
  );

  // ============================================================================
  // RENDER: Answer Phase
  // ============================================================================

  const renderAnswer = () => {
    if (!currentResult || !selectedAction) return null;

    const isCorrect = currentResult.grade === 'optimal';
    const isGood = currentResult.grade === 'good';
    const headerColor = isCorrect ? '#30D158' : isGood ? '#FF9F0A' : '#FF453A';
    const headerIcon = isCorrect ? <CheckCircle size={22} /> : <AlertTriangle size={22} />;
    const headerText = isCorrect
      ? `Correct! (+${currentResult.score})`
      : isGood
        ? `Good (+${currentResult.score})`
        : currentResult.grade === 'acceptable'
          ? `Acceptable (+${currentResult.score})`
          : `Suboptimal (+${currentResult.score})`;

    // Sort strategy entries by frequency descending
    const strategyEntries = Object.entries(scenario.correctStrategy)
      .filter(([, freq]) => freq > 0.005)
      .sort((a, b) => b[1] - a[1]);

    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 28,
        padding: '40px 24px',
        maxWidth: 600,
        margin: '0 auto',
        width: '100%',
        animation: 'fadeIn 300ms ease-out',
      }}>
        {/* Result header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          color: headerColor,
        }}>
          {headerIcon}
          <span style={{ fontSize: 22, fontWeight: 800 }}>{headerText}</span>
        </div>

        {/* Context if wrong */}
        {!isCorrect && (
          <div style={{
            fontSize: 14,
            color: 'var(--color-text-secondary)',
            textAlign: 'center',
            lineHeight: 1.6,
          }}>
            You chose: <span style={{ fontWeight: 700, color: '#fff' }}>{selectedAction}</span>{' '}
            <span style={{ fontFamily: 'var(--font-mono, monospace)' }}>
              ({(currentResult.userFreq * 100).toFixed(1)}%)
            </span>
            <br />
            Optimal: <span style={{ fontWeight: 700, color: headerColor === '#FF453A' ? '#FF9F0A' : headerColor }}>{currentResult.correctAction}</span>{' '}
            <span style={{ fontFamily: 'var(--font-mono, monospace)' }}>
              ({(currentResult.correctFreq * 100).toFixed(1)}%)
            </span>
          </div>
        )}

        {/* Board recap (compact) */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          <span style={{
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--color-text-tertiary)',
          }}>
            GTO Strategy for {scenario.heroCombo} on
          </span>
          <div style={{ display: 'flex', gap: 4 }}>
            {boardCards.map((card, i) => (
              <Card key={i} rank={card[0]} suit={card[1]} size="sm" />
            ))}
          </div>
        </div>

        {/* Strategy bars */}
        <div style={{
          width: '100%',
          padding: '16px 20px',
          background: 'var(--color-glass)',
          backdropFilter: 'blur(40px)',
          borderRadius: 14,
          border: '1px solid var(--color-glass-border)',
        }}>
          {strategyEntries.map(([action, freq]) => (
            <StrategyBar
              key={action}
              action={action}
              freq={freq}
              isUserPick={action === selectedAction}
              isOptimal={action === currentResult.correctAction}
            />
          ))}
        </div>

        {/* Next button */}
        <button
          onClick={handleNext}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '14px 40px',
            borderRadius: 12,
            border: 'none',
            background: 'var(--color-accent)',
            color: '#fff',
            fontSize: 16,
            fontWeight: 700,
            cursor: 'pointer',
            fontFamily: 'inherit',
            transition: 'all 200ms ease',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'var(--color-accent-hover)';
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(10,132,255,0.4)';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'var(--color-accent)';
            e.currentTarget.style.boxShadow = 'none';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          {questionNum < SCENARIO_COUNT ? 'Next' : 'Results'}
          <ArrowRight size={18} />
        </button>
      </div>
    );
  };

  // ============================================================================
  // RENDER: Summary Phase
  // ============================================================================

  const renderSummary = () => {
    const summary = getSessionSummary(session);
    const pct = Math.round((summary.totalScore / summary.maxScore) * 100);
    const stars = getStarRating(pct);
    const acceptableCount = SCENARIO_COUNT - summary.optimalCount - summary.goodCount - summary.mistakeCount;

    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 32,
        padding: '48px 24px',
        maxWidth: 480,
        margin: '0 auto',
        width: '100%',
        animation: 'fadeIn 300ms ease-out',
      }}>
        {/* Trophy */}
        <div style={{
          width: 72,
          height: 72,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #FF9F0A, #FFD60A)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 8px 32px rgba(255,159,10,0.3)',
        }}>
          <Trophy size={36} color="#1C1C1E" strokeWidth={2.5} />
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: 24,
            fontWeight: 800,
            letterSpacing: '-0.02em',
          }}>
            Session Complete!
          </div>
        </div>

        {/* Score display */}
        <div style={{
          textAlign: 'center',
          padding: '24px 40px',
          background: 'var(--color-glass)',
          backdropFilter: 'blur(40px)',
          borderRadius: 16,
          border: '1px solid var(--color-glass-border)',
          minWidth: 280,
        }}>
          <div style={{
            fontSize: 48,
            fontWeight: 800,
            fontFamily: 'var(--font-mono, monospace)',
            letterSpacing: '-0.03em',
            background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            lineHeight: 1.1,
          }}>
            {summary.totalScore}
          </div>
          <div style={{
            fontSize: 16,
            color: 'var(--color-text-tertiary)',
            fontFamily: 'var(--font-mono, monospace)',
            marginTop: 4,
          }}>
            / {summary.maxScore}
          </div>

          {/* Stars */}
          <div style={{
            marginTop: 16,
            fontSize: 28,
            letterSpacing: 4,
            color: '#FFD60A',
          }}>
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i} style={{ opacity: i < stars ? 1 : 0.2 }}>
                {'\u2605'}
              </span>
            ))}
          </div>
          <div style={{
            fontSize: 14,
            fontWeight: 600,
            color: 'var(--color-text-secondary)',
            marginTop: 6,
          }}>
            {getGradeLabel(stars)}
          </div>
        </div>

        {/* Breakdown */}
        <div style={{
          display: 'flex',
          gap: 16,
          justifyContent: 'center',
        }}>
          <StatBadge
            label="Optimal"
            value={summary.optimalCount}
            color="#30D158"
          />
          <StatBadge
            label="Good"
            value={summary.goodCount}
            color="#0A84FF"
          />
          <StatBadge
            label="Acceptable"
            value={acceptableCount}
            color="#FF9F0A"
          />
          <StatBadge
            label="Mistake"
            value={summary.mistakeCount}
            color="#FF453A"
          />
        </div>

        {/* Buttons */}
        <div style={{
          display: 'flex',
          gap: 12,
        }}>
          <button
            onClick={handleRestart}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '12px 28px',
              borderRadius: 12,
              border: '1px solid var(--color-glass-border)',
              background: 'var(--color-glass)',
              color: '#fff',
              fontSize: 15,
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 200ms ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--color-glass-hover)';
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--color-glass)';
              e.currentTarget.style.borderColor = 'var(--color-glass-border)';
            }}
          >
            <RotateCcw size={16} />
            Play Again
          </button>
          <button
            onClick={onClose}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '12px 28px',
              borderRadius: 12,
              border: 'none',
              background: 'var(--color-accent)',
              color: '#fff',
              fontSize: 15,
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 200ms ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--color-accent-hover)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(10,132,255,0.4)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--color-accent)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            Back to Solver
          </button>
        </div>
      </div>
    );
  };

  // ============================================================================
  // RENDER: Main layout
  // ============================================================================

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 200,
      background: 'var(--color-bg-primary)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 20px',
        background: 'rgba(28,28,30,0.95)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--color-glass-border)',
        flexShrink: 0,
      }}>
        {/* Left: Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #FF9F0A, #FF453A)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 16,
            fontWeight: 800,
            color: '#fff',
          }}>
            D
          </div>
          <div>
            <div style={{
              fontSize: 15,
              fontWeight: 700,
              letterSpacing: '0.03em',
            }}>
              DRILL MODE
            </div>
            <div style={{
              fontSize: 10,
              fontWeight: 500,
              color: 'var(--color-text-tertiary)',
              letterSpacing: '0.05em',
            }}>
              GTO TRAINING
            </div>
          </div>
        </div>

        {/* Center: Score + Question counter */}
        {phase !== 'summary' && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 20,
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}>
              <span style={{
                fontSize: 12,
                fontWeight: 600,
                color: 'var(--color-text-tertiary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}>
                Score
              </span>
              <span
                style={{
                  fontSize: 20,
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono, monospace)',
                  color: '#fff',
                  transition: 'transform 200ms ease',
                  transform: scoreAnim ? 'scale(1.2)' : 'scale(1)',
                }}
              >
                {session.totalScore}
              </span>
            </div>
            <div style={{
              padding: '4px 12px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.1)',
              fontSize: 13,
              fontWeight: 600,
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--color-text-secondary)',
            }}>
              #{questionNum} / {SCENARIO_COUNT}
            </div>
          </div>
        )}

        {/* Right: Close button */}
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
            transition: 'all 150ms ease',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'var(--color-glass-hover)';
            e.currentTarget.style.color = '#fff';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'var(--color-glass)';
            e.currentTarget.style.color = 'var(--color-text-secondary)';
          }}
        >
          <X size={20} />
        </button>
      </div>

      {/* Main content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: phase === 'summary' ? 'center' : 'flex-start',
        paddingTop: phase === 'question' ? 32 : 24,
      }}>
        {phase === 'question' && renderQuestion()}
        {phase === 'answer' && renderAnswer()}
        {phase === 'summary' && renderSummary()}
      </div>

      {/* Progress dots */}
      {phase !== 'summary' && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 6,
          padding: '12px 20px',
          flexShrink: 0,
          borderTop: '1px solid rgba(255,255,255,0.06)',
        }}>
          {session.scenarios.map((_, i) => {
            const result = session.results[i];
            let dotColor = 'rgba(255,255,255,0.12)';
            if (result) {
              switch (result.grade) {
                case 'optimal': dotColor = '#30D158'; break;
                case 'good': dotColor = '#0A84FF'; break;
                case 'acceptable': dotColor = '#FF9F0A'; break;
                case 'mistake': dotColor = '#FF453A'; break;
              }
            }
            const isCurrent = i === session.currentIndex;

            return (
              <div
                key={i}
                style={{
                  width: isCurrent ? 24 : 8,
                  height: 8,
                  borderRadius: 4,
                  background: isCurrent && !result ? 'var(--color-accent)' : dotColor,
                  transition: 'all 300ms ease',
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Small stat badge for summary
// ============================================================================

function StatBadge({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 4,
      padding: '12px 16px',
      background: `${color}12`,
      border: `1px solid ${color}30`,
      borderRadius: 12,
      minWidth: 80,
    }}>
      <span style={{
        fontSize: 24,
        fontWeight: 800,
        fontFamily: 'var(--font-mono, monospace)',
        color,
      }}>
        {value}
      </span>
      <span style={{
        fontSize: 11,
        fontWeight: 600,
        color: 'var(--color-text-tertiary)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
      }}>
        {label}
      </span>
    </div>
  );
}
