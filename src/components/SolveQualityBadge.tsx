/**
 * v1.3.0: post-solve Quality badge.
 *
 * Renders next to the StrategyPanel header after a solve completes,
 * giving the user a one-glance read on "is this strategy usable?". The
 * heuristic is exploitability-based since that's what the engine
 * computes and what GTO literature uses for quality:
 *
 *   <  0.5%  → 🟢 High quality        (pro-grade convergence)
 *   0.5–1.5% → 🟡 Good for play       (casual play absolutely fine)
 *   1.5–5%   → 🟠 Rough               (look at general lean only)
 *   >  5%    → 🔴 Low confidence      (run longer / use GPU)
 *
 * If the solve hit `time_budget` we additionally surface "stopped at
 * budget — try Deep mode for tighter convergence" so users know the
 * fix when results are below their bar.
 */

import type { EarlyStopReason } from '../lib/poker';

interface Props {
  exploitability: number;          // percent (e.g. 0.5 means 0.5%)
  iterationsRun: number;
  earlyStopReason?: EarlyStopReason;
}

interface Tier {
  emoji: string;
  label: string;
  color: string;
  bg: string;
}

function tierFor(exploit: number): Tier {
  if (exploit < 0.5) {
    return { emoji: '🟢', label: 'High quality', color: '#10b981', bg: 'rgba(16,185,129,0.12)' };
  }
  if (exploit < 1.5) {
    return { emoji: '🟡', label: 'Good for play', color: '#eab308', bg: 'rgba(234,179,8,0.12)' };
  }
  if (exploit < 5) {
    return { emoji: '🟠', label: 'Rough — directional only', color: '#f59e0b', bg: 'rgba(245,158,11,0.15)' };
  }
  return { emoji: '🔴', label: 'Low confidence', color: '#ef4444', bg: 'rgba(239,68,68,0.15)' };
}

export function SolveQualityBadge({ exploitability, iterationsRun, earlyStopReason }: Props) {
  // Don't render before we have a meaningful number. Engine sometimes
  // emits exploitability=0 when postsolve is skipped (e.g. --postsolve
  // none) — in that case we have no signal so suppress the badge.
  if (!Number.isFinite(exploitability) || exploitability <= 0) return null;

  const tier = tierFor(exploitability);
  const stoppedEarly = earlyStopReason === 'time_budget';

  return (
    <div
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 8,
        padding: '4px 10px', borderRadius: 999,
        background: tier.bg, border: `1px solid ${tier.color}55`,
        fontSize: 11, fontWeight: 600,
      }}
      title={
        `Exploitability ${exploitability.toFixed(2)}% after ${iterationsRun} iterations` +
        (stoppedEarly ? ' (stopped at time budget — strategy is the running average so far)' : '')
      }
    >
      <span style={{ fontSize: 12 }}>{tier.emoji}</span>
      <span style={{ color: tier.color }}>{tier.label}</span>
      <span style={{ color: 'var(--color-text-tertiary)', fontWeight: 500 }}>
        · {exploitability.toFixed(2)}% in {iterationsRun}{stoppedEarly ? ' iter (budget)' : ' iter'}
      </span>
    </div>
  );
}
