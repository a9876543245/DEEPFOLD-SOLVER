import React, { useMemo } from 'react';
import type { SolverResponse, ComboStrategy } from '../lib/poker';
import { getActionColor, parseBoardCards, expandComboLabel, SUIT_SYMBOLS, SUIT_COLORS } from '../lib/poker';
import { parseRange } from '../lib/ranges';
import type { SolverProgress } from '../hooks/useSolver';
import { useT } from '../lib/i18n';

import { Lock } from 'lucide-react';

interface Props {
  result: SolverResponse | null;
  hoveredCombo: string | null;
  elapsed: number;
  loading?: boolean;
  progress?: SolverProgress | null;
  board?: string;
  heroRange?: string;
  onLockNode?: () => void;
}

/** Full-width strategy bar for a combo variant — taller than the old 6px
 *  mini-bar and shows per-action frequency labels inside each segment so
 *  users can read the exact mix without guessing colors. */
function VariantStrategyBar({ strategy }: { strategy: ComboStrategy }) {
  const entries = Object.entries(strategy)
    .filter(([k, v]) => k !== 'Not in range' && v > 0.005)
    .sort((a, b) => b[1] - a[1]);

  if (!entries.length) return null;

  const total = entries.reduce((s, [, v]) => s + v, 0);

  return (
    <div style={{
      display: 'flex', height: 18, borderRadius: 4, overflow: 'hidden',
      background: 'var(--color-bg-tertiary)', flex: 1,
      border: '1px solid var(--color-glass-border)',
    }}>
      {entries.map(([action, freq]) => {
        const pct = (freq / total) * 100;
        // Only show the inline label when the segment is wide enough to fit
        // readable text — below ~14% the text overflows and looks bad.
        const showLabel = pct >= 14;
        return (
          <div key={action} style={{
            width: `${pct}%`,
            background: getActionColor(action),
            minWidth: 2,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, fontWeight: 700,
            color: '#fff',
            textShadow: '0 1px 2px rgba(0,0,0,0.6)',
            whiteSpace: 'nowrap',
          }}>
            {showLabel && `${Math.round(pct)}%`}
          </div>
        );
      })}
    </div>
  );
}

/** Render a specific combo with colored suit symbols */
function ComboDisplay({ rank1, suit1, rank2, suit2, isDead }: {
  rank1: string; suit1: string; rank2: string; suit2: string; isDead: boolean;
}) {
  const s1Color = SUIT_COLORS[suit1 as keyof typeof SUIT_COLORS] || '#fff';
  const s2Color = SUIT_COLORS[suit2 as keyof typeof SUIT_COLORS] || '#fff';
  const sym1 = SUIT_SYMBOLS[suit1 as keyof typeof SUIT_SYMBOLS] || suit1;
  const sym2 = SUIT_SYMBOLS[suit2 as keyof typeof SUIT_SYMBOLS] || suit2;

  return (
    <span style={{
      fontFamily: 'var(--font-mono, monospace)', fontSize: 12, fontWeight: 700,
      opacity: isDead ? 0.3 : 1,
      display: 'inline-flex', alignItems: 'center', gap: 1,
      minWidth: 48,
    }}>
      <span>{rank1}</span><span style={{ color: s1Color, fontSize: 11 }}>{sym1}</span>
      <span>{rank2}</span><span style={{ color: s2Color, fontSize: 11 }}>{sym2}</span>
    </span>
  );
}

export function StrategyPanel({ result, hoveredCombo, elapsed, loading, progress, board, heroRange, onLockNode }: Props) {
  const t = useT();

  // Expand hovered/target combo into specific variants
  const comboVariants = useMemo(() => {
    const targetLabel = result?.target_combo_analysis?.combo || hoveredCombo;
    if (!targetLabel || !board) return null;
    const boardCards = parseBoardCards(board);
    const specifics = expandComboLabel(targetLabel, boardCards);
    if (!specifics.length) return null;

    return specifics.map(sc => ({
      ...sc,
      strategy: result?.combo_strategies?.[sc.combo] as ComboStrategy | undefined,
    }));
  }, [result, hoveredCombo, board]);

  // Loading / progress state
  if (loading && progress) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div className="glass-panel" style={{ padding: 20 }}>
          <span className="text-label" style={{ marginBottom: 12, display: 'block' }}>
            {t('solving')}
          </span>

          {/* Progress bar */}
          <div style={{
            height: 8, background: 'var(--color-bg-tertiary)',
            borderRadius: 'var(--radius-full)', overflow: 'hidden', marginBottom: 12,
          }}>
            <div className="progress-bar-fill" style={{
              height: '100%', width: `${progress.pct}%`,
              borderRadius: 'var(--radius-full)',
              transition: 'width 150ms ease-out',
            }} />
          </div>

          {/* Phase label */}
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8, color: 'var(--color-text-primary)' }}>
            {progress.phase}
          </div>

          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
              <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.progress')}</div>
              <div className="text-mono" style={{ fontSize: 16, fontWeight: 700 }}>
                {Math.round(progress.pct)}%
              </div>
            </div>
            <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
              <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.elapsed')}</div>
              <div className="text-mono" style={{ fontSize: 16, fontWeight: 700 }}>
                {(progress.elapsed / 1000).toFixed(1)}s
              </div>
            </div>
            <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
              <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.iterations')}</div>
              <div className="text-mono" style={{ fontSize: 16, fontWeight: 700 }}>
                {progress.iteration}<span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>/{progress.total}</span>
              </div>
            </div>
            <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
              <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.status')}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-orange)' }}>
                {t('panel.running')}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-panel" style={{ padding: 20, textAlign: 'center' }}>
        <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>&#9824;</div>
        <div style={{ color: 'var(--color-text-secondary)', fontSize: 13 }}>
          {t('panel.emptyHint')}
        </div>
      </div>
    );
  }

  const globalEntries = Object.entries(result.global_strategy);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Solver Stats */}
      <div className="glass-panel" style={{ padding: 14 }}>
        <span className="text-label" style={{ marginBottom: 8, display: 'block' }}>
          {t('panel.solverResult')}
        </span>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.iterations')}</div>
            <div className="text-mono" style={{ fontSize: 16, fontWeight: 700 }}>
              {result.iterations_run}
            </div>
          </div>
          <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.exploitability')}</div>
            <div className="text-mono" style={{
              fontSize: 16, fontWeight: 700,
              color: result.exploitability_pct < 0.5 ? 'var(--color-green)' : 'var(--color-orange)'
            }}>
              {result.exploitability_pct.toFixed(2)}%
            </div>
          </div>
          <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.time')}</div>
            <div className="text-mono" style={{ fontSize: 16, fontWeight: 700 }}>
              {(elapsed / 1000).toFixed(2)}s
            </div>
          </div>
          <div style={{ background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '6px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{t('panel.status')}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-green)' }}>
              {result.status === 'success' ? t('panel.converged') : result.status}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Read — simplified human-readable recommendation */}
      {(() => {
        const entries = Object.entries(result.global_strategy)
          .map(([a, s]) => ({ action: a, freq: parseFloat(s) }))
          .sort((a, b) => b.freq - a.freq);
        if (entries.length === 0) return null;

        const top = entries[0];
        const second = entries[1];
        const aggSum = entries
          .filter(e => !['Check', 'Call', 'Fold'].includes(e.action))
          .reduce((s, e) => s + e.freq, 0);
        const passSum = entries
          .filter(e => ['Check', 'Call'].includes(e.action))
          .reduce((s, e) => s + e.freq, 0);

        let headline = '';
        let subtext = '';
        let accentColor = getActionColor(top.action);

        if (top.freq > 60) {
          headline = `${t('qr.primarily')} ${top.action}`;
          subtext = top.action === 'Check' || top.action === 'Call'
            ? t('qr.passive')
            : t('qr.strongPref');
        } else if (top.freq > 40 && second && second.freq > 25) {
          headline = `${t('qr.mixed')} ${top.action} ${t('qr.or')} ${second.action}`;
          subtext = t('qr.useBoth');
        } else if (aggSum > 55) {
          headline = t('qr.aggressive');
          subtext = t('qr.leanAgg');
          accentColor = 'var(--color-red)';
        } else if (passSum > 55) {
          headline = t('qr.defensive');
          subtext = t('qr.checkCall');
          accentColor = 'var(--color-green)';
        } else {
          headline = `${t('qr.lean')} ${top.action}`;
          subtext = t('qr.mixedStrategy');
        }

        return (
          <div className="glass-panel animate-fade-in" style={{
            padding: 14, borderLeft: `3px solid ${accentColor}`,
          }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accentColor, marginBottom: 4 }}>
              {headline}
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginBottom: 10 }}>
              {subtext}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {entries.slice(0, 3).map(e => (
                <div key={e.action} style={{
                  flex: 1, padding: '6px 8px', borderRadius: 6,
                  background: 'var(--color-bg-tertiary)', textAlign: 'center',
                }}>
                  <div style={{ fontSize: 16, fontWeight: 800, fontFamily: 'var(--font-mono)', color: getActionColor(e.action) }}>
                    {e.freq.toFixed(0)}%
                  </div>
                  <div style={{ fontSize: 9, fontWeight: 600, color: 'var(--color-text-tertiary)', marginTop: 2 }}>
                    {e.action}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      {/* Global Strategy */}
      <div className="glass-panel" style={{ padding: 14 }}>
        <span className="text-label" style={{ marginBottom: 8, display: 'block' }}>
          {t('panel.globalStrategy')}
        </span>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {globalEntries.map(([action, freqStr]) => {
            const freq = parseFloat(freqStr);
            const color = getActionColor(action);
            return (
              <div key={action}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: 2,
                      background: color, flexShrink: 0
                    }} />
                    <span style={{ fontSize: 12, fontWeight: 500 }}>{action}</span>
                  </div>
                  <span className="text-mono" style={{ fontSize: 12, fontWeight: 600 }}>
                    {freqStr}
                  </span>
                </div>
                <div style={{
                  height: 5, background: 'var(--color-bg-tertiary)',
                  borderRadius: 'var(--radius-full)', overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%', width: `${freq}%`,
                    background: color, borderRadius: 'var(--radius-full)',
                    transition: 'width 500ms ease-out'
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Target Combo Analysis */}
      {result.target_combo_analysis && (() => {
        const tca = result.target_combo_analysis;
        // Check if this combo is in the hero's original preflop range
        const heroRangeMap = heroRange ? parseRange(heroRange) : null;
        const isOutOfRange = heroRangeMap ? (heroRangeMap[tca.combo] ?? 0) <= 0.001 : false;
        const hasStrategy = Object.keys(tca.strategy_mix).some(k => k !== 'Not in range');

        return (
          <div className="glass-panel animate-fade-in" style={{ padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="text-label" style={{ display: 'block' }}>
                  {tca.combo}
                </span>
                {isOutOfRange && (
                  <span style={{
                    fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                    background: 'var(--color-orange-dim)', color: 'var(--color-orange)',
                    textTransform: 'uppercase', letterSpacing: '0.3px',
                  }}>{t('panel.offRange')}</span>
                )}
              </div>
              {onLockNode && hasStrategy && (
                <button
                  onClick={onLockNode}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '3px 8px', background: 'var(--color-glass)',
                    border: '1px solid var(--color-glass-border)',
                    borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                    color: 'var(--color-text-secondary)', fontSize: 10, fontWeight: 700,
                    textTransform: 'uppercase',
                  }}
                  title={t('lock.tooltip')}
                >
                  <Lock size={10} />
                  {t('lock')}
                </button>
              )}
            </div>

            {/* Off-range warning banner */}
            {isOutOfRange && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 10px', marginTop: 6, marginBottom: 8, borderRadius: 6,
                background: 'var(--color-orange-dim)', border: '1px solid rgba(255,159,10,0.2)',
              }}>
                <span style={{ fontSize: 14 }}>&#9888;</span>
                <div style={{ fontSize: 10, color: 'var(--color-orange)', lineHeight: 1.4 }}>
                  {t('panel.offRangeWarning')}
                </div>
              </div>
            )}

            {/* Strategy breakdown (shown for both in-range and off-range) */}
            {hasStrategy && (
              <>
                <div style={{ marginBottom: 6, marginTop: isOutOfRange ? 0 : 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
                    {isOutOfRange ? t('panel.suggested') : t('panel.best')}
                  </span>
                  <span style={{
                    fontWeight: 700, fontSize: 13,
                    color: getActionColor(tca.best_action)
                  }}>
                    {tca.best_action}
                  </span>
                  {tca.ev !== 0 && (
                    <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginLeft: 8 }}>
                      EV: <span className="text-mono">{tca.ev.toFixed(2)}</span>
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {Object.entries(tca.strategy_mix)
                    .filter(([a]) => a !== 'Not in range')
                    .map(([action, freq]) => (
                    <div key={action} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: getActionColor(action) }}>{action}</span>
                      <span className="text-mono">{freq}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        );
      })()}

      {/* Combo Variants Breakdown */}
      {comboVariants && comboVariants.length > 0 && (
        <div className="glass-panel animate-fade-in" style={{ padding: 14 }}>
          <span className="text-label" style={{ marginBottom: 8, display: 'block' }}>
            {t('panel.comboVariants')} ({comboVariants.filter(c => !c.isDead).length} {t('panel.live')} / {comboVariants.length})
          </span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {comboVariants.map(cv => {
              const isNotInRange = cv.strategy?.['Not in range'];
              const isDead = cv.isDead || !!isNotInRange;

              return (
                <div key={cv.combo} style={{
                  display: 'flex', flexDirection: 'column', gap: 4,
                  padding: '8px 10px',
                  background: isDead ? 'transparent' : 'var(--color-bg-tertiary)',
                  borderRadius: 'var(--radius-sm)',
                  opacity: isDead ? 0.35 : 1,
                  transition: 'opacity 200ms ease',
                }}>
                  {/* Row 1: combo symbols + wide strategy bar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <ComboDisplay
                      rank1={cv.rank1} suit1={cv.suit1}
                      rank2={cv.rank2} suit2={cv.suit2}
                      isDead={isDead}
                    />
                    {isDead ? (
                      <span style={{
                        fontSize: 10, fontWeight: 700,
                        color: 'var(--color-text-tertiary)',
                        textTransform: 'uppercase', letterSpacing: '0.5px',
                      }}>
                        {cv.isDead ? t('panel.dead') : t('panel.out')}
                      </span>
                    ) : cv.strategy ? (
                      <VariantStrategyBar strategy={cv.strategy} />
                    ) : null}
                  </div>

                  {/* Row 2: full per-action breakdown (all actions, not just dominant) */}
                  {!isDead && cv.strategy && (() => {
                    const sorted = Object.entries(cv.strategy)
                      .filter(([k, v]) => k !== 'Not in range' && v > 0.005)
                      .sort((a, b) => b[1] - a[1]);
                    if (!sorted.length) return null;
                    return (
                      <div style={{
                        display: 'flex', flexWrap: 'wrap', gap: '4px 10px',
                        paddingLeft: 58,  // align under the bar, past ComboDisplay width
                        fontSize: 10, fontWeight: 600,
                      }}>
                        {sorted.map(([action, freq]) => (
                          <span key={action} style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                          }}>
                            <span style={{
                              width: 8, height: 8, borderRadius: 2,
                              background: getActionColor(action),
                              flexShrink: 0,
                            }} />
                            <span style={{ color: 'var(--color-text-secondary)' }}>
                              {action}
                            </span>
                            <span style={{
                              color: getActionColor(action),
                              fontWeight: 700,
                              fontVariantNumeric: 'tabular-nums',
                            }}>
                              {(freq * 100).toFixed(1)}%
                            </span>
                          </span>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Hovered Combo Info (when no target selected) */}
      {hoveredCombo && !result.target_combo_analysis && !comboVariants && (
        <div className="glass-panel animate-fade-in" style={{ padding: 10 }}>
          <div style={{
            display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 2,
          }}>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{hoveredCombo}</div>
            {/* EV from cache, if present (#3 from roadmap). Avoids forcing a
             *  re-solve just to show the per-combo number. */}
            {result.combo_evs && hoveredCombo in result.combo_evs && (
              <div style={{
                fontSize: 11,
                color: result.combo_evs[hoveredCombo] >= 0
                  ? 'var(--color-green)' : 'var(--color-red, #ef4444)',
                fontWeight: 700,
              }}>
                EV {result.combo_evs[hoveredCombo] >= 0 ? '+' : ''}
                {result.combo_evs[hoveredCombo].toFixed(2)}
              </div>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>
            {t('panel.clickToAnalyze')}
          </div>
        </div>
      )}
    </div>
  );
}
