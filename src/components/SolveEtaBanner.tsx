/**
 * v1.2.2: pre-solve ETA banner.
 *
 * Renders during the loading state when `useSolver().estimate` is populated
 * (set by the `estimate_solve` Tauri command, fires in parallel with the
 * actual solve). Surfaces three pieces of context users want BEFORE
 * committing to a long wait:
 *
 *   1. ETA: humanized — "1 second", "12 seconds", "2 minutes", "1.5 hours".
 *   2. Backend: which engine path will run ("CPU" or "CUDA <gpu>").
 *   3. Warning: when ETA > 60s on CPU AND GPU was rejected, point at the
 *      reject reason (e.g. "Pascal needs CUDA-12.x build") so the user
 *      knows whether they can switch.
 *
 * Designed to NOT be modal — never blocks the solve. Just informs.
 * The existing cancel button on the loading overlay (if any) still works
 * for explicit aborts.
 */

import type { EstimateResponse } from '../lib/poker';

interface Props {
  estimate: EstimateResponse | null;
  loading: boolean;
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

export function SolveEtaBanner({ estimate, loading }: Props) {
  if (!loading) return null;
  if (!estimate || estimate.status !== 'estimate') return null;

  const r = estimate.resources;
  const seconds = r.estimated_solve_seconds ?? 0;
  const backend = shortBackend(r.backend_for_estimate);
  const isCpu = backend === 'CPU';
  const isLong = seconds >= 60;
  const wasRejected = !!r.fallback_reason;

  // Tier the visual urgency. Solid neutral for OK, amber for "this'll take
  // a bit", red for "are you sure" (>30 min CPU).
  let bg = 'var(--color-glass)';
  let border = 'var(--color-glass-border)';
  let icon = 'ⓘ';
  if (seconds >= 1800 && isCpu) {
    bg = 'rgba(220, 38, 38, 0.15)'; border = '#dc2626'; icon = '⚠';
  } else if (isLong) {
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
          Estimated <strong>{humanizeSeconds(seconds)}</strong> on {backend}
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 2 }}>
          {r.player_nodes.toLocaleString()} player nodes · ETA may vary ±2× —
          DCFR can stop early on convergence.
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
        {seconds >= 1800 && isCpu && !wasRejected && (
          <div style={{
            fontSize: 11, color: '#fca5a5', marginTop: 6,
          }}>
            That's a long wait. Consider <code>--backend gpu</code> if you have
            a supported GPU, or reduce iterations / bet sizes / range width.
          </div>
        )}
      </div>
    </div>
  );
}
