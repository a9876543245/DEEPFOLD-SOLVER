import React, { useState } from 'react';
import { useT } from '../lib/i18n';
import { MEMORY_PROFILE_PRESETS, type MemoryProfile } from '../lib/poker';

export type BetSizingKey = 'standard' | 'polar' | 'small_ball';

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
}: Props) {
  const t = useT();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const betPresets: Array<{ key: BetSizingKey; label: string; flop: string; turn: string; river: string }> = [
    { key: 'standard',   label: t('config.standard'),  flop: '33/75',  turn: '33/75', river: '33/75' },
    { key: 'polar',      label: t('config.polar'),     flop: '75/150', turn: '75/150', river: '75/150' },
    { key: 'small_ball', label: t('config.smallBall'), flop: '25/33',  turn: '25/33', river: '33/50' },
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

  return (
    <div className="glass-panel" style={{ padding: 16 }}>
      <span className="text-label" style={{ marginBottom: 12, display: 'block' }}>
        {t('config')}
      </span>

      {/* Pot Size */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4, display: 'block' }}>
          {t('config.potSize')}
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onPotChange(Math.max(1, pot - 10))}>-</button>
          <input
            className="input-field text-mono"
            type="number"
            value={pot}
            onChange={(e) => onPotChange(Number(e.target.value))}
            style={{ textAlign: 'center', width: 80 }}
          />
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onPotChange(pot + 10)}>+</button>
        </div>
      </div>

      {/* Effective Stack */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4, display: 'block' }}>
          {t('config.effStack')}
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onStackChange(Math.max(1, stack - 25))}>-</button>
          <input
            className="input-field text-mono"
            type="number"
            value={stack}
            onChange={(e) => onStackChange(Number(e.target.value))}
            style={{ textAlign: 'center', width: 80 }}
          />
          <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 14 }}
            onClick={() => onStackChange(stack + 25)}>+</button>
        </div>
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

      {/* Solve Button */}
      <button
        className="btn-primary"
        onClick={onSolve}
        disabled={loading}
        style={{ width: '100%', height: 44, fontSize: 15 }}
      >
        {loading ? (
          <>
            <span className="animate-spin" style={{
              display: 'inline-block', width: 16, height: 16,
              border: '2px solid rgba(255,255,255,0.3)',
              borderTopColor: 'white', borderRadius: '50%'
            }} />
            {t('solving')}
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
