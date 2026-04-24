import React from 'react';
import type { GameTreeNode, GameAction } from '../lib/gameTree';
import { getActionColor } from '../lib/poker';
import { useT } from '../lib/i18n';

interface Props {
  node: GameTreeNode;
  onAction: (action: GameAction) => void;
  loading: boolean;
}

/** Map action types to display colors */
const ACTION_TYPE_COLORS: Record<string, string> = {
  check: '#30D158',
  bet: '#FF453A',
  raise: '#BF5AF2',
  call: '#30D158',
  fold: '#636366',
  allin: '#FF9F0A',
};

/**
 * ActionBar — clickable action buttons displayed below the range grid.
 *
 * Shows available actions at the current game tree node.
 * OOP actions: Check / Bet 33% / Bet 75% / All-in
 * Facing bet:  Fold / Call / Raise 3x / All-in
 */
export function ActionBar({ node, onAction, loading }: Props) {
  const t = useT();
  if (node.isTerminal || node.actions.length === 0) {
    // Terminal node — show result
    const lastStep = node.path[node.path.length - 1];
    const isFold = lastStep?.action.type === 'fold';

    return (
      <div style={{
        padding: '14px 20px',
        background: 'var(--color-glass)',
        backdropFilter: 'blur(20px)',
        borderRadius: 12,
        border: '1px solid var(--color-glass-border)',
        textAlign: 'center',
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-secondary)' }}>
          {isFold ? (
            <>
              <span style={{ color: ACTION_TYPE_COLORS.fold }}>
                {lastStep.player}
              </span>{' '}
              {t('action.folds')} — {lastStep.player === 'OOP' ? 'IP' : 'OOP'} {t('action.wins')}{' '}
              <span className="text-mono" style={{ color: 'var(--color-green)' }}>
                {node.pot}
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--color-green)' }}>{t('action.showdown')}</span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      padding: '12px 16px',
      background: 'var(--color-glass)',
      backdropFilter: 'blur(20px)',
      borderRadius: 12,
      border: '1px solid var(--color-glass-border)',
    }}>
      {/* Node context header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Street badge */}
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            padding: '2px 7px', borderRadius: 4,
            background: node.street === 'flop' ? 'rgba(48,209,88,0.15)'
              : node.street === 'turn' ? 'rgba(255,159,10,0.15)'
              : 'rgba(255,69,58,0.15)',
            color: node.street === 'flop' ? '#30D158'
              : node.street === 'turn' ? '#FF9F0A'
              : '#FF453A',
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}>
            {node.street === 'flop' ? t('board.flop') : node.street === 'turn' ? t('board.turn') : t('board.river')}
          </span>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            padding: '2px 8px', borderRadius: 4,
            background: node.activePlayer === 'OOP' ? 'rgba(10,132,255,0.2)' : 'rgba(191,90,242,0.2)',
            color: node.activePlayer === 'OOP' ? 'var(--color-accent)' : 'var(--color-purple)',
            fontSize: 11, fontWeight: 700,
          }}>
            {node.activePlayer}
          </span>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            {node.awaitingDeal ? t('action.awaitDeal') : t('action.toAct')}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            Pot: <span className="text-mono" style={{ color: 'var(--color-green)', fontWeight: 600 }}>{node.pot}</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            Stack: <span className="text-mono" style={{ fontWeight: 600 }}>{node.effectiveStack}</span>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div style={{
        display: 'flex', gap: 6, flexWrap: 'wrap',
      }}>
        {node.actions.map((action) => {
          const color = ACTION_TYPE_COLORS[action.type] || '#8E8E93';
          const isFold = action.type === 'fold';

          return (
            <button
              key={action.label}
              onClick={() => !loading && onAction(action)}
              disabled={loading}
              style={{
                flex: action.type === 'check' || action.type === 'call' ? '1 1 auto' : '0 1 auto',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                padding: '10px 16px',
                background: isFold
                  ? 'rgba(99,99,102,0.15)'
                  : `${color}18`,
                border: `1.5px solid ${color}50`,
                borderRadius: 8,
                cursor: loading ? 'wait' : 'pointer',
                transition: 'all 150ms ease',
                fontFamily: 'inherit',
                opacity: loading ? 0.5 : 1,
              }}
              onMouseOver={(e) => {
                if (!loading) {
                  (e.currentTarget as HTMLButtonElement).style.background = `${color}30`;
                  (e.currentTarget as HTMLButtonElement).style.borderColor = color;
                }
              }}
              onMouseOut={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = isFold
                  ? 'rgba(99,99,102,0.15)'
                  : `${color}18`;
                (e.currentTarget as HTMLButtonElement).style.borderColor = `${color}50`;
              }}
            >
              {/* Action icon */}
              <span style={{ fontSize: 12, color }}>
                {action.type === 'check' ? '✓' :
                 action.type === 'bet' ? '↑' :
                 action.type === 'raise' ? '⇑' :
                 action.type === 'call' ? '✓' :
                 action.type === 'fold' ? '✕' :
                 '★'}
              </span>
              {/* Label */}
              <span style={{
                fontSize: 12, fontWeight: 600,
                color,
              }}>
                {action.label}
              </span>
              {/* Amount */}
              {action.amount !== undefined && action.type !== 'allin' && (
                <span className="text-mono" style={{
                  fontSize: 10, fontWeight: 500,
                  color: 'var(--color-text-tertiary)',
                }}>
                  ({action.amount})
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
