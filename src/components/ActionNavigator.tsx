import React from 'react';
import { useT } from '../lib/i18n';

interface ActionStep {
  label: string;
  player: 'OOP' | 'IP' | 'Deal';
}

interface Props {
  history: ActionStep[];
  onNavigate: (index: number) => void;
}

/**
 * Action Navigator — breadcrumb path for tree traversal.
 * Shows the action history: Flop → OOP: Check → IP: Bet 66% → ...
 * Each crumb is clickable for time-travel (re-render strategy at that node).
 */
export function ActionNavigator({ history, onNavigate }: Props) {
  const t = useT();
  if (history.length === 0) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '8px 12px', fontSize: 12, color: 'var(--color-text-tertiary)'
      }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        </svg>
        {t('nav.root')}
      </div>
    );
  }

  const playerColors: Record<string, string> = {
    OOP: 'var(--color-accent)',
    IP: 'var(--color-purple)',
    Deal: 'var(--color-orange)',
  };

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 2,
      padding: '6px 12px', overflowX: 'auto',
      flexWrap: 'nowrap',
    }}>
      {/* Root crumb */}
      <button
        onClick={() => onNavigate(-1)}
        style={{
          background: 'none', border: 'none',
          color: 'var(--color-text-secondary)', cursor: 'pointer',
          fontSize: 12, fontWeight: 500, padding: '4px 8px',
          borderRadius: 'var(--radius-sm)',
          transition: 'all 150ms ease',
        }}
        onMouseOver={(e) => {
          (e.target as HTMLButtonElement).style.background = 'var(--color-glass-hover)';
        }}
        onMouseOut={(e) => {
          (e.target as HTMLButtonElement).style.background = 'none';
        }}
      >
        {t('nav.root')}
      </button>

      {history.map((step, idx) => (
        <React.Fragment key={idx}>
          <span style={{ color: 'var(--color-text-tertiary)', fontSize: 10 }}>
            ›
          </span>
          <button
            onClick={() => onNavigate(idx)}
            style={{
              background: idx === history.length - 1 ? 'var(--color-glass)' : 'none',
              border: idx === history.length - 1 ? '1px solid var(--color-glass-border)' : 'none',
              color: idx === history.length - 1
                ? 'var(--color-text-primary)'
                : 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontSize: 12, fontWeight: idx === history.length - 1 ? 600 : 500,
              padding: '4px 10px',
              borderRadius: 'var(--radius-sm)',
              transition: 'all 200ms ease',
              whiteSpace: 'nowrap',
              display: 'flex', alignItems: 'center', gap: 4,
              animation: idx === history.length - 1 ? 'fadeIn 200ms ease-out' : 'none',
            }}
          >
            <span style={{
              fontSize: 9, fontWeight: 700,
              color: playerColors[step.player] || 'var(--color-text-tertiary)',
              textTransform: 'uppercase',
            }}>
              {step.player}
            </span>
            {step.label}
          </button>
        </React.Fragment>
      ))}
    </div>
  );
}
