import React, { useState } from 'react';
import { useT } from '../lib/i18n';
import {
  MEMORY_PROFILE_PRESETS, SOLVE_MODE_PRESETS,
  type MemoryProfile, type SolveMode, type EstimateResponse,
} from '../lib/poker';

export type BetSizingKey = 'lite' | 'standard' | 'polar' | 'small_ball';

interface Props {
  pot: number;
  stack: number;
  iterations: number;
  onPotChange: (val: number) => void;
  onStackChange: (val: number) => void;
  onIterationsChange: (val: number) => void;
  onSolve: () => void;
  loading: boolean;
  sizingKey?: BetSizingKey;
  onSizingChange?: (key: BetSizingKey) => void;
  /** Memory profile preset for the next solve. Defaults to 'balanced' if
   *  unset. Mirrors the C++ engine's `--memory-profile` CLI flag and the
   *  Rust `ResolvedMemoryBudget::from_profile` resolver. */
  memoryProfile?: MemoryProfile;
  onMemoryProfileChange?: (profile: MemoryProfile) => void;
  /** v1.3.0: solve mode preset (Quick/Standard/Deep). Drives iter cap +
   *  time budget. Defaults 'standard'. */
  solveMode?: SolveMode;
  onSolveModeChange?: (mode: SolveMode) => void;
  /** Stage 5: runout decomposition mode. 'off' (default) = legacy collapse
   *  gate (turn/river approximated on large rainbow boards). 'auto' = solve
   *  real runouts via subgame decomposition. Orthogonal to solveMode. */
  decomposeRunouts?: 'off' | 'auto';
  onDecomposeRunoutsChange?: (mode: 'off' | 'auto') => void;
  /** Roadmap ④: Exact feasibility pre-flight (--estimate-only with the
   *  decompose block). Fed by App's debounced estimate when Exact is on;
   *  null while unavailable (browser mode, estimating, or failed). */
  exactPreflight?: EstimateResponse | null;
  /** v1.3.0: called when the user clicks Stop during loading. Should
   *  invoke the cancel_solve Tauri command. */
  onStop?: () => void;
  /** Effective stack (BB) implied by the current GameContext. Used to
   *  compute how far the user-edited `stack` drifts from the depth the
   *  preflop range was built for; drives the SPR readout + deviation
   *  warning. Pass null when no context is active. */
  expectedEffectiveBB?: number | null;
}

/**
 * Solver Controls — Stack/Pot inputs, bet size toggles, and solve button.
 * Apple HIG style with numeric steppers and pill buttons.
 */
export function SolverControls({
  pot, stack, iterations,
  onPotChange, onStackChange, onIterationsChange,
  onSolve, loading,
  sizingKey = 'standard', onSizingChange,
  memoryProfile = 'balanced', onMemoryProfileChange,
  solveMode = 'standard', onSolveModeChange, onStop,
  decomposeRunouts = 'off', onDecomposeRunoutsChange,
  exactPreflight = null,
  expectedEffectiveBB = null,
}: Props) {
  const t = useT();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const betPresets: Array<{ key: BetSizingKey; label: string; flop: string; turn: string; river: string }> = [
    { key: 'lite',       label: t('config.lite'),      flop: '50',     turn: '50',     river: '50' },
    { key: 'standard',   label: t('config.standard'),  flop: '33/75',  turn: '33/75',  river: '33/75' },
    { key: 'polar',      label: t('config.polar'),     flop: '75/150', turn: '75/150', river: '75/150' },
    { key: 'small_ball', label: t('config.smallBall'), flop: '25/33',  turn: '25/33',  river: '33/50' },
  ];
  const activePreset = Math.max(0, betPresets.findIndex(p => p.key === sizingKey));

  // Memory profile presets — labels are translated, the numeric budgets come
  // from MEMORY_PROFILE_PRESETS so the UI never drifts from the actual values
  // the engine will use.
  const memProfiles: Array<{ key: MemoryProfile; label: string }> = [
    { key: 'safe',        label: t('config.memory.safe') },
    { key: 'balanced',    label: t('config.memory.balanced') },
    { key: 'performance', label: t('config.memory.performance') },
  ];
  const activeMemoryPreset = MEMORY_PROFILE_PRESETS[memoryProfile];

  // SPR + stack-depth deviation. Both feed the info row below the
  // pot/stack inputs. 1 BB = 10 chips (same convention as MATCHUPS and
  // derivePotStack).
  const BB_CHIPS = 10;
  const effectiveBBNow = stack / BB_CHIPS;
  const spr = pot > 0 ? stack / pot : 0;
  const deviation = expectedEffectiveBB != null && expectedEffectiveBB > 0
    ? Math.abs(effectiveBBNow - expectedEffectiveBB) / expectedEffectiveBB
    : 0;
  // > 40% → strong warning (red): preflop range is likely wrong for this depth
  // > 20% → soft warning (yellow): edges of the comfort zone
  // else  → quiet info readout
  const deviationLevel: 'red' | 'yellow' | null =
    expectedEffectiveBB == null ? null
    : deviation > 0.40 ? 'red'
    : deviation > 0.20 ? 'yellow'
    : null;

  return (
    <div className="glass-panel" style={{ padding: 16 }}>
      <span className="text-label" style={{ marginBottom: 12, display: 'block' }}>
        {t('config')}
      </span>

      {/* Pot Size — displayed/edited in BB. Internal state (pot prop) is
       *  still chips since the engine + rest of codebase use chips; only
       *  the input layer is BB. 1 BB = 10 chips. Step 1 BB on +/-. */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4, display: 'block' }}>
          {t('config.potSize')} ({t('gctx.bb')})
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onPotChange(Math.max(BB_CHIPS, pot - BB_CHIPS))}>-</button>
          <input
            className="input-field text-mono"
            type="number"
            step="0.5"
            value={pot / BB_CHIPS}
            onChange={(e) => {
              const bb = Number(e.target.value);
              if (!Number.isFinite(bb) || bb <= 0) return;
              onPotChange(Math.round(bb * BB_CHIPS));
            }}
            style={{ textAlign: 'center', width: 80 }}
          />
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onPotChange(pot + BB_CHIPS)}>+</button>
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{t('gctx.bb')}</span>
        </div>
      </div>

      {/* Effective Stack — same BB-input pattern. Step 5 BB on +/- (keeps
       *  the previous "25 chips" feel — 25 chips = 2.5 BB; we round to 5
       *  BB for a cleaner increment). */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4, display: 'block' }}>
          {t('config.effStack')} ({t('gctx.bb')})
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onStackChange(Math.max(BB_CHIPS, stack - 5 * BB_CHIPS))}>-</button>
          <input
            className="input-field text-mono"
            type="number"
            step="1"
            value={stack / BB_CHIPS}
            onChange={(e) => {
              const bb = Number(e.target.value);
              if (!Number.isFinite(bb) || bb <= 0) return;
              onStackChange(Math.round(bb * BB_CHIPS));
            }}
            style={{ textAlign: 'center', width: 80 }}
          />
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onStackChange(stack + 5 * BB_CHIPS)}>+</button>
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{t('gctx.bb')}</span>
        </div>
      </div>

      {/* Derived info: SPR + stack-depth deviation warning. SPR is the
       *  real driver of postflop strategy, so we surface it directly.
       *  (Effective BB itself is no longer printed here — the stack input
       *  above is already in BB, so duplicating it adds noise.) Deviation
       *  warns when the user has manually overridden the stack far from
       *  what the preflop range was solved for. */}
      <div style={{ marginBottom: 12, fontSize: 11, lineHeight: 1.4 }}>
        <div style={{ color: 'var(--color-text-tertiary)' }}>
          {t('config.spr')}: <span className="text-mono" style={{ color: 'var(--color-text-secondary)', fontWeight: 600 }}>
            {spr.toFixed(1)}
          </span>
        </div>
        {deviationLevel && (
          <div style={{
            marginTop: 4, padding: '4px 8px', borderRadius: 4,
            fontWeight: 600,
            background: deviationLevel === 'red'
              ? 'rgba(255, 69, 58, 0.12)'
              : 'rgba(255, 159, 10, 0.12)',
            color: deviationLevel === 'red'
              ? 'var(--color-red, #FF453A)'
              : 'var(--color-orange, #FF9F0A)',
            border: deviationLevel === 'red'
              ? '1px solid rgba(255, 69, 58, 0.3)'
              : '1px solid rgba(255, 159, 10, 0.3)',
          }}>
            {deviationLevel === 'red'
              ? t('config.stackDeviation.strong').replace('{bb}', String(expectedEffectiveBB))
              : t('config.stackDeviation.soft').replace('{bb}', String(expectedEffectiveBB))}
          </div>
        )}
      </div>

      {/* Bet Size Presets */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6, display: 'block' }}>
          {t('config.betSizing')}
        </label>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {betPresets.map((preset, i) => (
            <button
              key={preset.key}
              className={`btn-pill ${activePreset === i ? 'active' : ''}`}
              onClick={() => onSizingChange?.(preset.key)}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
          Flop: {betPresets[activePreset].flop}% / Turn: {betPresets[activePreset].turn}% / River: {betPresets[activePreset].river}%
        </div>
      </div>

      {/* v1.3.0: Solve mode (Quick / Standard / Deep) — picks iter cap +
          time budget. Displayed as 3 pills right above the Solve button so
          users see the trade-off (sanity check vs pro vs research). */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6, display: 'block' }}>
          {t('config.solveMode.label')}
        </label>
        <div style={{ display: 'flex', gap: 6 }} role="radiogroup">
          {(['quick','standard','deep'] as SolveMode[]).map(m => {
            const preset = SOLVE_MODE_PRESETS[m];
            const active = solveMode === m;
            return (
              <button
                key={m}
                role="radio"
                aria-checked={active}
                disabled={loading}
                className={`btn-pill ${active ? 'active' : ''}`}
                onClick={() => onSolveModeChange?.(m)}
                title={preset.description}
                style={{ flex: 1 }}
              >
                {t(`config.solveMode.${m}`)}
              </button>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
          {SOLVE_MODE_PRESETS[solveMode].description}
        </div>
      </div>

      {/* Stage 5: runout decomposition toggle. 'off' keeps the legacy collapse
          gate (turn/river equity approximated on rainbow boards too large to
          enumerate); 'auto' solves real runouts via flop-trunk + per-turn-card
          subgame decomposition. Independent of solveMode. */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6, display: 'block' }}>
          {t('config.decompose.label')}
        </label>
        <div style={{ display: 'flex', gap: 6 }} role="radiogroup">
          {(['off','auto'] as const).map(m => {
            const active = decomposeRunouts === m;
            return (
              <button
                key={m}
                role="radio"
                aria-checked={active}
                disabled={loading}
                className={`btn-pill ${active ? 'active' : ''}`}
                onClick={() => onDecomposeRunoutsChange?.(m)}
                style={{ flex: 1 }}
              >
                {t(`config.decompose.${m}`)}
              </button>
            );
          })}
        </div>
        {decomposeRunouts === 'off' ? (
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
            {t('config.decompose.hint.off')}
          </div>
        ) : (() => {
          // Roadmap ④: honest SPR-gated positioning. Prefer the engine's
          // tier from the pre-flight; fall back to the same thresholds
          // computed locally (mirrors solver_decomposed.h: ≤3 / ≤6 / above).
          const de = exactPreflight?.decompose;
          const tier = de?.ok
            ? de.quality_tier
            : (spr <= 3 ? 'high' : spr <= 6 ? 'medium' : 'navigation');
          const tierColor =
            tier === 'high' ? '#10b981'
            : tier === 'medium' ? 'var(--color-text-tertiary)'
            : '#f59e0b';
          const fmtDuration = (s: number): string => {
            if (!Number.isFinite(s) || s <= 0) return '—';
            if (s < 90) return `${Math.max(1, Math.round(s))}s`;
            if (s < 5400) return `${Math.round(s / 60)} min`;
            return `${(s / 3600).toFixed(1)} h`;
          };
          const isTauriEnv = !!(window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
          return (
            <div style={{ fontSize: 11, marginTop: 4, display: 'grid', gap: 3 }}>
              <div style={{ color: tierColor }}>
                {t(`config.decompose.hint.auto.${tier}`)}
              </div>
              {de?.ok && !de.would_engage && (
                <div style={{ color: 'var(--color-text-tertiary)' }}>
                  {t('config.decompose.preflight.noop')}
                </div>
              )}
              {de?.ok && de.would_engage && (
                <div className="text-mono" style={{ color: 'var(--color-text-secondary)' }}>
                  {t('config.decompose.preflight', {
                    time: fmtDuration(de.total_seconds),
                    leaves: de.leaves,
                    lo: de.expected_exploit_lo_pct,
                    hi: de.expected_exploit_hi_pct,
                  })}
                </div>
              )}
              {exactPreflight === null && isTauriEnv && (
                <div style={{ color: 'var(--color-text-tertiary)' }}>
                  {t('config.decompose.preflight.estimating')}
                </div>
              )}
            </div>
          );
        })()}
      </div>

      {/* Solve / Stop Button — switches role based on loading state.
          Stop is pure abort (no result preserved); time budget gives the
          "stop with what we have" semantics automatically when it fires. */}
      <button
        className={loading ? 'btn-secondary' : 'btn-primary'}
        onClick={loading ? onStop : onSolve}
        disabled={loading && !onStop}
        style={{
          width: '100%', height: 44, fontSize: 15,
          ...(loading && onStop ? { background: 'var(--color-red, #dc2626)', color: '#fff' } : {}),
        }}
      >
        {loading ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="1" />
            </svg>
            {t('solve.stop')}
          </>
        ) : (
          <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            {t('solve')}
          </>
        )}
      </button>

      {/* Advanced Settings */}
      <button
        className="btn-secondary"
        onClick={() => setShowAdvanced(!showAdvanced)}
        style={{ width: '100%', marginTop: 8, fontSize: 12 }}
      >
        {showAdvanced ? t('config.hide') : t('config.show')} {t('config.advanced')}
      </button>

      {showAdvanced && (
        <div className="animate-fade-in" style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{t('config.maxIter')}</label>
            <input
              className="input-field text-mono"
              type="number"
              value={iterations}
              onChange={(e) => onIterationsChange(Number(e.target.value))}
              style={{ marginTop: 4 }}
            />
          </div>

          {/* Memory Profile (Polish #1) */}
          <div>
            <label style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6, display: 'block' }}>
              {t('config.memory.label')}
            </label>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }} role="radiogroup">
              {memProfiles.map(({ key, label }) => {
                const active = memoryProfile === key;
                return (
                  <button
                    key={key}
                    role="radio"
                    aria-checked={active}
                    className={`btn-pill ${active ? 'active' : ''}`}
                    onClick={() => onMemoryProfileChange?.(key)}
                    title={`${label} — ${MEMORY_PROFILE_PRESETS[key].host_mb} MB host, ${MEMORY_PROFILE_PRESETS[key].json_mb} MB JSON, up to ${MEMORY_PROFILE_PRESETS[key].strategy_tree_max_nodes} strategy-tree nodes`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
              {t('config.memory.host')}: {(activeMemoryPreset.host_mb / 1024).toFixed(1)} GB
              {' · '}
              {t('config.memory.json')}: {activeMemoryPreset.json_mb} MB
              {' · '}
              {t('config.memory.nodes')}: {activeMemoryPreset.strategy_tree_max_nodes.toLocaleString()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
