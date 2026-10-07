import { useMemo, useState, type CSSProperties } from 'react';
import { X } from 'lucide-react';
import { useT } from '../lib/i18n';
import {
  STREETS, SIZING_PRESETS, presetSpec, cloneSpec, parseSizeList, formatSizeList, validateSpec,
  type BetSizingSpec, type SizingPresetKey, type StreetKey,
} from '../lib/betSizing';

interface Props {
  spec: BetSizingSpec;
  onSave: (spec: BetSizingSpec) => void;
  onClose: () => void;
}

type Side = 'oop' | 'ip';
type Field = 'bet' | 'raise' | 'donk';

/** Text of every menu, keyed "oop.flop.bet", while the user types. */
function textsOf(spec: BetSizingSpec): Record<string, string> {
  const out: Record<string, string> = {};
  for (const side of ['oop', 'ip'] as Side[]) {
    for (const st of STREETS) {
      for (const f of ['bet', 'raise', 'donk'] as Field[]) {
        out[`${side}.${st}.${f}`] = formatSizeList(spec[side][st][f]);
      }
    }
  }
  return out;
}

/**
 * Pio-style bet sizing editor: per street and player a bet menu, a raise
 * menu, OOP's donk menu and an all-in switch; a raise cap and an all-in
 * threshold. "33 75" = 33% and 75% of the pot; raises also take "2.5x"
 * (raise to 2.5 times the bet faced).
 */
export function BetSizingEditor({ spec, onSave, onClose }: Props) {
  const t = useT();
  const [draft, setDraft] = useState<BetSizingSpec>(() => cloneSpec(spec));
  const [texts, setTexts] = useState<Record<string, string>>(() => textsOf(spec));

  // Per-field parse errors, then whole-spec validation.
  const fieldErrors = useMemo(() => {
    const errs: Record<string, string> = {};
    for (const [key, text] of Object.entries(texts)) {
      const { error } = parseSizeList(text, key.endsWith('.raise'));
      if (error) errs[key] = error;
    }
    return errs;
  }, [texts]);
  const specError = Object.keys(fieldErrors).length === 0 ? validateSpec(draft) : null;
  const canSave = Object.keys(fieldErrors).length === 0 && !specError;

  const setText = (side: Side, st: StreetKey, f: Field, text: string) => {
    const key = `${side}.${st}.${f}`;
    setTexts(prev => ({ ...prev, [key]: text }));
    const { sizes, error } = parseSizeList(text, f === 'raise');
    if (!error) {
      setDraft(prev => {
        const next = cloneSpec(prev);
        next[side][st][f] = sizes;
        return next;
      });
    }
  };

  const setAllin = (side: Side, st: StreetKey, on: boolean) => {
    setDraft(prev => {
      const next = cloneSpec(prev);
      next[side][st].allin = on;
      return next;
    });
  };

  const load = (next: BetSizingSpec) => {
    setDraft(cloneSpec(next));
    setTexts(textsOf(next));
  };

  const copyOopToIp = () => {
    const next = cloneSpec(draft);
    for (const st of STREETS) {
      next.ip[st] = { ...cloneSpec(draft).oop[st], donk: [] };
    }
    load(next);
  };

  const inputStyle = (key: string): CSSProperties => ({
    width: '100%', padding: '5px 7px', fontSize: 12, fontFamily: 'var(--font-mono)',
    background: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)',
    border: `1px solid ${fieldErrors[key] ? 'var(--color-red, #FF453A)' : 'var(--color-glass-border)'}`,
    borderRadius: 6,
  });

  const cell = (side: Side, st: StreetKey, f: Field) => {
    const key = `${side}.${st}.${f}`;
    return (
      <input
        aria-label={`${side.toUpperCase()} ${st} ${f}`}
        value={texts[key] ?? ''}
        onChange={e => setText(side, st, f, e.target.value)}
        placeholder={f === 'donk' ? '—' : ''}
        title={fieldErrors[key] ?? ''}
        style={inputStyle(key)}
      />
    );
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: 16, background: 'rgba(0,0,0,0.6)',
    }} onClick={onClose}>
      <div className="glass-panel" style={{
        width: '100%', maxWidth: 760, maxHeight: '90vh', overflowY: 'auto', padding: 18,
        background: 'var(--color-bg-elevated, #1c1c1e)',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <span style={{ fontSize: 15, fontWeight: 700 }}>{t('sizing.editTitle')}</span>
          <button onClick={onClose} aria-label={t('cancel')} style={{
            background: 'none', border: 'none', color: 'var(--color-text-secondary)', cursor: 'pointer',
          }}><X size={18} /></button>
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 12, lineHeight: 1.5 }}>
          {t('sizing.desc')}
        </div>

        {/* Presets */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{t('sizing.loadPreset')}:</span>
          {(Object.keys(SIZING_PRESETS) as SizingPresetKey[]).map(k => (
            <button key={k} className="btn-pill" onClick={() => load(presetSpec(k))}>
              {t(`config.${k === 'small_ball' ? 'smallBall' : k}`)}
            </button>
          ))}
          <button className="btn-pill" onClick={copyOopToIp}>{t('sizing.copyOop')}</button>
        </div>

        {/* Menus: one block per street, OOP and IP rows */}
        <div style={{ display: 'grid', gap: 12 }}>
          {STREETS.map(st => (
            <div key={st} style={{ border: '1px solid var(--color-glass-border)', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6, color: 'var(--color-text-secondary)' }}>
                {t(`board.${st}`)}
              </div>
              <div style={{
                display: 'grid', gridTemplateColumns: '44px 1fr 1fr 1fr 64px', gap: 6, alignItems: 'center',
                fontSize: 11, color: 'var(--color-text-tertiary)',
              }}>
                <span />
                <span>{t('sizing.bet')}</span>
                <span>{t('sizing.raise')}</span>
                <span>{t('sizing.donk')}</span>
                <span>{t('sizing.allin')}</span>
                {(['oop', 'ip'] as Side[]).map(side => (
                  <div key={side} style={{ display: 'contents' }}>
                    <span style={{ fontWeight: 700, color: side === 'oop' ? 'var(--color-accent)' : 'var(--color-purple)' }}>
                      {side.toUpperCase()}
                    </span>
                    {cell(side, st, 'bet')}
                    {cell(side, st, 'raise')}
                    {side === 'oop' ? cell(side, st, 'donk') : <span style={{ textAlign: 'center' }}>—</span>}
                    <input
                      type="checkbox"
                      aria-label={`${side.toUpperCase()} ${st} all-in`}
                      checked={draft[side][st].allin}
                      onChange={e => setAllin(side, st, e.target.checked)}
                      style={{ justifySelf: 'center' }}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Global limits */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 14, fontSize: 12 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {t('sizing.raiseCap')}
            <input type="number" min={0} max={10} step={1} value={draft.raise_cap}
              onChange={e => setDraft(prev => ({ ...cloneSpec(prev), raise_cap: Number(e.target.value) }))}
              style={{ ...inputStyle('raise_cap'), width: 64 }} />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {t('sizing.allinThreshold')}
            <input type="number" min={0} max={100} step={1} value={draft.allin_threshold}
              onChange={e => setDraft(prev => ({ ...cloneSpec(prev), allin_threshold: Number(e.target.value) }))}
              style={{ ...inputStyle('allin_threshold'), width: 72 }} />
          </label>
        </div>

        {(Object.values(fieldErrors)[0] || specError) && (
          <div style={{ marginTop: 10, fontSize: 12, color: 'var(--color-red, #FF453A)' }}>
            {Object.entries(fieldErrors)[0]
              ? `${Object.entries(fieldErrors)[0][0]}: ${Object.entries(fieldErrors)[0][1]}`
              : specError}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button className="btn-secondary" onClick={onClose}>{t('cancel')}</button>
          <button className="btn-primary" disabled={!canSave} onClick={() => onSave(draft)}>
            {t('sizing.apply')}
          </button>
        </div>
      </div>
    </div>
  );
}
