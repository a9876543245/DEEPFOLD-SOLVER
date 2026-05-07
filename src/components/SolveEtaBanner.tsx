/**
 * Pre-solve ETA banner.
 *
 * v1.2.2 introduced this; v1.3.0 reworked it around time budgets:
 *  - The headline number is now `min(estimate, time_budget)`. Showing
 *    "Estimated 5 hours" when the user-set budget will stop at 5 minutes
 *    is misleading. Show what they'll ACTUALLY wait.
 *  - When the estimate exceeds the budget significantly we surface that
 *    as a quality warning ("solve may not converge fully in budget —
 *    Standard mode usually fine; for more accuracy switch to Deep").
 *  - GPU-rejected users still see the fallback_reason inline.
 */

import type { EstimateResponse } from '../lib/poker';

interface Props {
  estimate: EstimateResponse | null;
  loading: boolean;
  /** v1.3.0: time budget the user picked (Quick=60, Standard=300, Deep=900).
   *  When set, the headline shows "up to <budget>" with the model estimate
   *  as a side note. */
  timeBudgetSeconds?: number;
}

function humanizeSeconds(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return '—';
  if (s < 1) return '< 1 second';
  if (s < 60) return `${s.toFixed(0)} second${s >= 1.5 ? 's' : ''}`;
  if (s < 3600) {
    const m = s / 60;
    if (m < 10) return `${m.toFixed(1)} minutes`;
    return `${Math.round(m)} minutes`;
  }
  const h = s / 3600;
  return `${h.toFixed(1)} hours`;
}

/** Shorten the verbose backend label for inline display.
 *  Input: "CUDA (NVIDIA GeForce RTX 5090, 32606 MB, CC 12.0)" → "CUDA"
 *         "CPU-DCFR" → "CPU" */
function shortBackend(b: string | undefined): string {
  if (!b) return 'unknown';
  if (b.toLowerCase().startsWith('cuda')) return 'GPU';
  if (b.toLowerCase().includes('cpu')) return 'CPU';
  return b;
}

export function SolveEtaBanner({ estimate, loading, timeBudgetSeconds }: Props) {
  if (!loading) return null;
  if (!estimate || estimate.status !== 'estimate') return null;

  const r = estimate.resources;
  const rawEstimate = r.estimated_solve_seconds ?? 0;
  const backend = shortBackend(r.backend_for_estimate);
  const isCpu = backend === 'CPU';
  const wasRejected = !!r.fallback_reason;

  // v1.3.0: when a time budget is set, the actual wait is `min(estimate,
  // budget)` — that's what we headline. The unbounded estimate goes in a
  // smaller side note for context.
  const headlineSeconds = timeBudgetSeconds && timeBudgetSeconds > 0
    ? Math.min(rawEstimate, timeBudgetSeconds)
    : rawEstimate;
  const willHitBudget = timeBudgetSeconds !== undefined && timeBudgetSeconds > 0
    && rawEstimate > timeBudgetSeconds;

  // Visual tier: budget guarantees the wait — once budget is set, urgency
  // tracks "will the budget produce a converged answer?" not "how long
  // will I wait?". Without budget (legacy), tier by raw estimate.
  let bg = 'var(--color-glass)';
  let border = 'var(--color-glass-border)';
  let icon = 'ⓘ';
  if (willHitBudget && isCpu) {
    bg = 'rgba(245, 158, 11, 0.15)'; border = '#f59e0b'; icon = '⏱';
  } else if (!timeBudgetSeconds && rawEstimate >= 1800 && isCpu) {
    bg = 'rgba(220, 38, 38, 0.15)'; border = '#dc2626'; icon = '⚠';
  } else if (!timeBudgetSeconds && rawEstimate >= 60) {
    bg = 'rgba(245, 158, 11, 0.15)'; border = '#f59e0b'; icon = '⏱';
  }

  return (
    <div
      style={{
        padding: '10px 14px',
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 8,
        fontSize: 12,
        color: 'var(--color-text-primary)',
        display: 'flex', alignItems: 'flex-start', gap: 10,
        lineHeight: 1.5,
      }}
    >
      <span style={{ fontSize: 16, lineHeight: 1, marginTop: 1 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600 }}>
          {timeBudgetSeconds && timeBudgetSeconds > 0
            ? <>Up to <strong>{humanizeSeconds(headlineSeconds)}</strong> on {backend}</>
            : <>Estimated <strong>{humanizeSeconds(rawEstimate)}</strong> on {backend}</>}
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 2 }}>
          {r.player_nodes.toLocaleString()} player nodes
          {willHitBudget && (
            <> · model estimate {humanizeSeconds(rawEstimate)}, budget will fire first</>
          )}
          {!willHitBudget && <> · ETA may vary ±2×</>}
        </div>
        {wasRejected && (
          <div style={{
            fontSize: 11, color: '#fca5a5', marginTop: 6,
            padding: '4px 8px', background: 'rgba(0,0,0,0.2)',
            borderRadius: 4,
          }}>
            <strong>Heads up:</strong> {r.fallback_reason}
          </div>
        )}
        {willHitBudget && isCpu && (
          <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 6 }}>
            Budget will stop the solve before convergence. Strategy will be
            usable but rougher than full solve. Consider Deep mode for
            longer budget, or wait for GPU support.
          </div>
        )}
      </div>
    </div>
  );
}
