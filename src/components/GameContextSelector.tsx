/**
 * GameContextSelector — pick which preflop chart bucket drives default
 * IP/OOP ranges. Three cascading dropdowns:
 *
 *   1. Game type:    Cash / MTT
 *   2. Format:       depends on game type (e.g. 6max_100bb, vs_open_3b)
 *   3. Effective BB: depends on (game, format) — only stacks that have
 *                    at least one chart in the bundled library
 *
 * The available options are derived from `list_gto_scenarios` so they
 * always match what the chart library actually contains. If the library
 * isn't bundled (browser preview, missing install dir), the component
 * shows a one-line note and disables the dropdowns.
 *
 * Changing any field bubbles a fresh GameContext to the parent, which
 * (a) re-runs `useGtoAutoRange` to load matching IP/OOP ranges, and
 * (b) updates pot/stack to the chosen depth in chips.
 */
import React, { useEffect, useMemo, useState } from 'react';
import type { GameContext } from '../lib/poker';
import type { GtoScenario } from './GtoChartBrowser';
import { isTauri } from '../lib/tauriEnv';
import { useT } from '../lib/i18n';

interface Props {
  value: GameContext;
  onChange: (next: GameContext) => void;
}

export function GameContextSelector({ value, onChange }: Props) {
  const t = useT();
  const [scenarios, setScenarios] = useState<GtoScenario[] | null>(null);

  // Load scenario metadata once when this mounts (Tauri only).
  useEffect(() => {
    if (!isTauri()) return;
    (async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const list = await invoke<GtoScenario[]>('list_gto_scenarios');
        setScenarios(list);
      } catch {
        // Silent; selector stays in fallback state.
      }
    })();
  }, []);

  // Folder helpers. The data update made `scenario_type` a semantic field
  // ("RFI", "vs_Open", "vs_3B", ...), so the Format dropdown must derive
  // bucket choices from each chart's id path instead.
  // id format: "cash/6max_100bb/BB_100bb_K19" → folder = "6max_100bb".
  const folderOf = (id: string): string => id.split('/')[1] ?? '';

  // Distinct game types in the library (e.g. "Cash", "MTT").
  const gameTypes = useMemo(() => {
    if (!scenarios) return [];
    return Array.from(new Set(scenarios.map(s => s.game_type))).sort();
  }, [scenarios]);

  // Distinct folder buckets within the selected game type.
  const formats = useMemo(() => {
    if (!scenarios) return [];
    return Array.from(new Set(
      scenarios
        .filter(s => s.game_type === value.gameType)
        .map(s => folderOf(s.id)),
    )).filter(f => f).sort();
  }, [scenarios, value.gameType]);

  // Distinct effective_bb values within the selected (game, folder).
  const stacks: { sorted: number[]; hasNull: boolean; total: number } = useMemo(() => {
    if (!scenarios) return { sorted: [], hasNull: false, total: 0 };
    const matches = scenarios.filter(s =>
      s.game_type === value.gameType &&
      folderOf(s.id) === value.scenarioType,
    );
    const set = new Set<number>();
    let hasNull = false;
    for (const s of matches) {
      if (s.effective_bb == null) hasNull = true;
      else set.add(s.effective_bb);
    }
    const sorted = Array.from(set).sort((a, b) => b - a);
    return { sorted, hasNull, total: matches.length };
  }, [scenarios, value.gameType, value.scenarioType]);

  // Some format names encode the stack depth in the folder name itself
  // ("6max_100bb" → 100, "6max_200bb" → 200) — for those, the stack
  // picker is redundant and gets hidden. Other formats ("8max_straddle",
  // "vs_open_3b", "rfi", ...) don't encode stack, so the user picks
  // manually from whatever discrete stacks the bundled charts cover.
  const parseStackFromFormat = (st: string): number | null => {
    const m = st.match(/_(\d+)bb/i);
    return m ? parseInt(m[1], 10) : null;
  };
  const formatEncodesStack = (st: string): boolean => parseStackFromFormat(st) !== null;

  // When the user picks a new game type, snap format + stack to defaults
  // for that type so we don't leave the selector in an unreachable state.
  const handleGameType = (gt: string) => {
    const newFormats = Array.from(new Set(
      (scenarios ?? [])
        .filter(s => s.game_type === gt)
        .map(s => folderOf(s.id)),
    )).filter(f => f).sort();
    const newFormat = newFormats[0] ?? value.scenarioType;
    const newStack = formatEncodesStack(newFormat)
      ? parseStackFromFormat(newFormat)
      : null;
    onChange({ gameType: gt, scenarioType: newFormat, effectiveBB: newStack });
  };

  const handleFormat = (fmt: string) => {
    const newStack = formatEncodesStack(fmt) ? parseStackFromFormat(fmt) : null;
    onChange({ gameType: value.gameType, scenarioType: fmt, effectiveBB: newStack });
  };

  const handleStack = (bb: number | null) => {
    onChange({ ...value, effectiveBB: bb });
  };

  // Pretty label for a scenario_type. Cash buckets are self-describing
  // ("6max_100bb"); MTT buckets are short codes that need translation.
  const formatLabel = (st: string): string => {
    const key = `gctx.mtt.${st.toLowerCase()}`;
    const translated = t(key);
    if (translated && translated !== key) return `${translated}`;
    return st.replace(/_/g, ' ');
  };

  const disabled = !scenarios;

  return (
    <div className="glass-panel" style={{ padding: 16, marginBottom: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ fontSize: 14, fontWeight: 700 }}>{t('gctx')}</span>
        {scenarios && (
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)', fontWeight: 500 }}>
            {stacks.total} {t('gctx.format').toLowerCase()}
          </span>
        )}
      </div>

      {disabled && (
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
          {t('gctx.noScenarios')}
        </div>
      )}

      {!disabled && (
        <>
          {/* Game type tabs */}
          <Row label={t('gctx.gameType')}>
            <div style={{ display: 'flex', gap: 4, background: 'var(--color-bg-tertiary)', borderRadius: 8, padding: 3, flex: 1 }}>
              {gameTypes.map(gt => {
                const active = value.gameType === gt;
                const labelKey = `gctx.${gt.toLowerCase()}`;
                const label = t(labelKey) === labelKey ? gt : t(labelKey);
                return (
                  <button key={gt} onClick={() => handleGameType(gt)}
                    style={{
                      flex: 1, padding: '6px 0', borderRadius: 6, cursor: 'pointer',
                      fontFamily: 'inherit', fontSize: 11, fontWeight: 600, transition: 'all 150ms ease',
                      background: active ? 'rgba(124,58,237,0.25)' : 'transparent',
                      color: active ? '#c4b5fd' : 'var(--color-text-tertiary)',
                      border: active ? '1px solid rgba(124,58,237,0.4)' : '1px solid transparent',
                    }}>
                    {label}
                  </button>
                );
              })}
            </div>
          </Row>

          {/* Format dropdown */}
          <Row label={t('gctx.format')}>
            <select value={value.scenarioType} onChange={e => handleFormat(e.target.value)}
              style={selectStyle}>
              {formats.map(f => (
                <option key={f} value={f}>{formatLabel(f)}</option>
              ))}
            </select>
          </Row>

          {/* Stack depth — when the format name encodes stack in itself
           *  ("6max_100bb"), the row is hidden; otherwise (e.g. "8max_straddle"
           *  or any MTT bucket) the user picks from discrete stacks the
           *  bundled charts actually cover. */}
          {!formatEncodesStack(value.scenarioType) && (
            <Row label={t('gctx.stack')}>
              <select value={value.effectiveBB ?? ''} onChange={e => handleStack(e.target.value === '' ? null : Number(e.target.value))}
                style={selectStyle}>
                <option value="">{t('gctx.any')}</option>
                {stacks.sorted.map(bb => (
                  <option key={bb} value={bb}>{bb} {t('gctx.bb')}</option>
                ))}
              </select>
            </Row>
          )}
        </>
      )}
    </div>
  );
}

// Small helper — keeps each row's label-input layout consistent without a
// separate stylesheet entry.
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
      <span style={{ width: 90, fontSize: 11, color: 'var(--color-text-secondary)', fontWeight: 500 }}>
        {label}
      </span>
      {children}
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  flex: 1, padding: '6px 8px',
  background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)',
  border: '1px solid var(--color-border)', borderRadius: 6,
  fontSize: 12, fontFamily: 'inherit', cursor: 'pointer',
};
