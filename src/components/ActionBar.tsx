import type { EngineAction, NodeView } from '../lib/poker';
import { useT } from '../lib/i18n';

interface Props {
  /** The node shown (engine NodeInfo). */
  view: NodeView;
  onAction: (action: EngineAction) => void;
  loading: boolean;
}

/** Format internal chip amount as BB for display. 1 BB = 10 chips (same
 *  convention as MATCHUPS / derivePotStack). Trims trailing .0 so whole
 *  values like 22 BB show as "22" instead of "22.0". */
export function toBB(chips: number): string {
  const v = chips / 10;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

/** "Bet_75" → "Bet 75%", "Raise_150" → "Raise 150%"; others unchanged. */
export function prettyAction(label: string): string {
  const m = /^(Bet|Raise)_(.+)$/.exec(label);
  return m ? `${m[1]} ${m[2]}%` : label;
}

/** Map action types to display colors */
const ACTION_TYPE_COLORS: Record<string, string> = {
  check: '#30C8C0',  // teal — kept in sync with ACTION_COLORS in lib/poker.ts
  bet: '#FF453A',
  raise: '#BF5AF2',
  call: '#30D158',
  fold: '#0A84FF',   // blue — kept in sync with ACTION_COLORS in lib/poker.ts
  allin: '#FF9F0A',
};

const STREET_NAMES = ['flop', 'turn', 'river'] as const;
const STREET_COLORS = ['#30D158', '#FF9F0A', '#FF453A'];

/**
 * ActionBar — the actions of the node shown, exactly as the engine solved
 * them (labels, amounts, what each leads to), below the range grid.
 */
export function ActionBar({ view, onAction, loading }: Props) {
  const t = useT();
  if (view.kind === 'terminal') {
    const fold = view.terminal === 'fold_oop' || view.terminal === 'fold_ip';
    const folder = view.terminal === 'fold_oop' ? 'OOP' : 'IP';
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
          {fold ? (
            <>
              <span style={{ color: ACTION_TYPE_COLORS.fold }}>{folder}</span>{' '}
              {t('action.folds')} — {folder === 'OOP' ? 'IP' : 'OOP'} {t('action.wins')}{' '}
              <span className="text-mono" style={{ color: 'var(--color-green)' }}>
                {toBB(view.pot)} BB
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--color-green)' }}>
              {t('action.showdown')}{' '}
              <span className="text-mono">({toBB(view.pot)} BB)</span>
            </span>
          )}
        </div>
      </div>
    );
  }

  const street = STREET_NAMES[Math.min(2, view.street)] ?? 'flop';
  const streetColor = STREET_COLORS[Math.min(2, view.street)] ?? STREET_COLORS[0];
  const acting = view.acting ?? 'OOP';
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
        marginBottom: 10, flexWrap: 'wrap', gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            padding: '2px 7px', borderRadius: 4,
            background: `${streetColor}26`, color: streetColor,
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}>
            {t(`board.${street}`)}
          </span>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            padding: '2px 8px', borderRadius: 4,
            background: acting === 'OOP' ? 'rgba(10,132,255,0.2)' : 'rgba(191,90,242,0.2)',
            color: acting === 'OOP' ? 'var(--color-accent)' : 'var(--color-purple)',
            fontSize: 11, fontWeight: 700,
          }}>
            {acting}
          </span>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            {t('action.toAct')}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            Pot: <span className="text-mono" style={{ color: 'var(--color-green)', fontWeight: 600 }}>{toBB(view.pot)} BB</span>
          </div>
          {view.to_call > 0 && (
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
              {t('action.toCall')}: <span className="text-mono" style={{ fontWeight: 600 }}>{toBB(view.to_call)} BB</span>
            </div>
          )}
          {/* Each player's own stack: they differ while a bet is pending. */}
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            OOP <span className="text-mono" style={{ fontWeight: 600 }}>{toBB(view.stack_oop)}</span>
            {' / '}
            IP <span className="text-mono" style={{ fontWeight: 600 }}>{toBB(view.stack_ip)}</span> BB
          </div>
        </div>
      </div>

      {view.actions.length === 0 && (
        <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>
          {t('action.solveToExplore')}
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {view.actions.map((action, idx) => {
          const color = ACTION_TYPE_COLORS[action.type] || '#8E8E93';
          const isFold = action.type === 'fold';
          // Bets and raises show the street total they make (raise TO),
          // calls and all-ins the chips they put in.
          const shown = action.type === 'bet' || action.type === 'raise' ? action.raise_to : action.amount;
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
                background: isFold ? 'rgba(99,99,102,0.15)' : `${color}18`,
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
              {/* Keyboard hint — 1-9 trigger the Nth action (matches App.tsx) */}
              {idx < 9 && (
                <span className="text-mono" style={{
                  fontSize: 9, fontWeight: 700, lineHeight: 1,
                  padding: '1px 4px', borderRadius: 3,
                  background: `${color}22`, color,
                  opacity: 0.7,
                }}>
                  {idx + 1}
                </span>
              )}
              <span style={{ fontSize: 12, color }}>
                {action.type === 'check' ? '✓' :
                 action.type === 'bet' ? '↑' :
                 action.type === 'raise' ? '⇑' :
                 action.type === 'call' ? '✓' :
                 action.type === 'fold' ? '✕' :
                 '★'}
              </span>
              <span style={{ fontSize: 12, fontWeight: 600, color }}>
                {prettyAction(action.label)}
              </span>
              {shown > 0 && (
                <span className="text-mono" style={{
                  fontSize: 10, fontWeight: 500,
                  color: 'var(--color-text-tertiary)',
                }}>
                  ({toBB(shown)} BB)
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
