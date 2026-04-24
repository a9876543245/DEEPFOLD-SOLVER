import React, { useState, useEffect, useMemo } from 'react';
import { GRID_LABELS, RANGE_TEMPLATES } from '../lib/poker';
import { parseRange } from '../lib/ranges';
import { X, Check, Type, Grid3X3 } from 'lucide-react';
import { useT } from '../lib/i18n';

interface Props {
  initialRangeStr?: string;
  onSave: (rangeStr: string) => void;
  onClose: () => void;
  title?: string;
}

const WEIGHTS = [0, 0.25, 0.5, 0.75, 1.0];
const WEIGHT_COLORS: Record<number, string> = {
  0: 'var(--color-bg-tertiary)',
  0.25: 'rgba(64, 156, 255, 0.25)',
  0.5: 'rgba(64, 156, 255, 0.50)',
  0.75: 'rgba(10, 132, 255, 0.75)',
  1.0: '#0A84FF',
};

/** Parse simple range text: supports "AA,AKs,KK" and "AA:1.0,AKs:0.5" */
function parseRangeText(text: string): Record<string, number> {
  const w: Record<string, number> = {};
  if (!text.trim()) return w;
  const validLabels = new Set(GRID_LABELS.flat());
  const tokens = text.split(',').map(t => t.trim()).filter(Boolean);
  for (const token of tokens) {
    const colonIdx = token.indexOf(':');
    if (colonIdx !== -1) {
      const label = token.substring(0, colonIdx).trim();
      const freq = parseFloat(token.substring(colonIdx + 1));
      if (validLabels.has(label) && !isNaN(freq)) {
        w[label] = Math.max(0, Math.min(1, freq));
      }
    } else {
      // No colon → assume 100% weight
      const label = token.trim();
      if (validLabels.has(label)) {
        w[label] = 1.0;
      }
    }
  }
  return w;
}

export function RangeEditorModal({ initialRangeStr, onSave, onClose, title }: Props) {
  const t = useT();
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [selectedWeight, setSelectedWeight] = useState<number>(1.0);
  const [isDragging, setIsDragging] = useState(false);
  const [inputMode, setInputMode] = useState<'grid' | 'text'>('grid');
  const [rangeText, setRangeText] = useState('');

  useEffect(() => {
    if (initialRangeStr) {
      setWeights(parseRange(initialRangeStr));
    } else {
      const all: Record<string, number> = {};
      GRID_LABELS.flat().forEach(lbl => all[lbl] = 1.0);
      setWeights(all);
    }
  }, [initialRangeStr]);

  // Compute combo count for display
  const comboCount = useMemo(() => {
    return Object.values(weights).filter(v => v > 0).length;
  }, [weights]);

  const handlePointerDown = (label: string) => {
    setIsDragging(true);
    setWeights(prev => ({ ...prev, [label]: selectedWeight }));
  };

  const handlePointerEnter = (label: string) => {
    if (isDragging) {
      setWeights(prev => ({ ...prev, [label]: selectedWeight }));
    }
  };

  const handlePointerUp = () => setIsDragging(false);

  const handleSave = () => {
    const parts: string[] = [];
    for (const label of GRID_LABELS.flat()) {
      const w = weights[label] ?? 0;
      if (w > 0) parts.push(`${label}:${w.toFixed(2)}`);
    }
    onSave(parts.join(','));
  };

  const applyTemplate = (tpl: typeof RANGE_TEMPLATES[number]) => {
    const newWeights: Record<string, number> = {};
    for (const [label, w] of Object.entries(tpl.weights)) newWeights[label] = w;
    setWeights(newWeights);
  };

  const applyText = () => {
    const parsed = parseRangeText(rangeText);
    if (Object.keys(parsed).length > 0) {
      setWeights(parsed);
      setInputMode('grid');
    }
  };

  // When switching to text mode, serialize current weights
  const switchToText = () => {
    const parts: string[] = [];
    for (const label of GRID_LABELS.flat()) {
      const w = weights[label] ?? 0;
      if (w >= 1.0) parts.push(label);
      else if (w > 0) parts.push(`${label}:${w.toFixed(2)}`);
    }
    setRangeText(parts.join(','));
    setInputMode('text');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
         style={{ background: 'rgba(0,0,0,0.6)' }}
         onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp}>
      <div className="relative flex flex-col w-full max-w-2xl overflow-hidden animate-slide-up"
           style={{
             background: 'var(--color-bg-elevated)',
             border: '1px solid var(--color-border)',
             borderRadius: 16, boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
           }}>

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '14px 20px', borderBottom: '1px solid var(--color-border)',
          background: 'var(--color-bg-secondary)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 15, fontWeight: 700 }}>{title || t('range.editTitle')}</span>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
              background: 'var(--color-blue-dim)', color: 'var(--color-accent)',
            }}>{comboCount} / 169</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* Grid / Text toggle */}
            <div style={{
              display: 'flex', background: 'var(--color-bg-tertiary)',
              borderRadius: 6, padding: 2, gap: 2,
            }}>
              <button onClick={() => setInputMode('grid')} style={{
                display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
                borderRadius: 4, border: 'none', cursor: 'pointer',
                background: inputMode === 'grid' ? 'var(--color-accent)' : 'transparent',
                color: inputMode === 'grid' ? '#fff' : 'var(--color-text-tertiary)',
                fontSize: 11, fontWeight: 600, fontFamily: 'inherit',
              }}>
                <Grid3X3 size={12} /> {t('range.grid')}
              </button>
              <button onClick={switchToText} style={{
                display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
                borderRadius: 4, border: 'none', cursor: 'pointer',
                background: inputMode === 'text' ? 'var(--color-accent)' : 'transparent',
                color: inputMode === 'text' ? '#fff' : 'var(--color-text-tertiary)',
                fontSize: 11, fontWeight: 600, fontFamily: 'inherit',
              }}>
                <Type size={12} /> {t('range.text')}
              </button>
            </div>
            <button onClick={onClose} style={{
              padding: 6, border: 'none', background: 'none', cursor: 'pointer',
              color: 'var(--color-text-secondary)', borderRadius: '50%',
            }}>
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Quick Templates Bar */}
        <div style={{
          display: 'flex', gap: 6, padding: '10px 20px', overflowX: 'auto',
          borderBottom: '1px solid var(--color-border)',
          background: 'rgba(255,255,255,0.02)',
        }}>
          {RANGE_TEMPLATES.map(tpl => (
            <button key={tpl.name} onClick={() => applyTemplate(tpl)} title={tpl.description}
              style={{
                padding: '5px 12px', borderRadius: 6, border: '1px solid var(--color-glass-border)',
                background: 'var(--color-glass)', color: 'var(--color-text-secondary)',
                fontSize: 11, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
                fontFamily: 'inherit', transition: 'all 150ms ease',
              }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--color-accent)'; e.currentTarget.style.color = '#fff'; e.currentTarget.style.borderColor = 'var(--color-accent)'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'var(--color-glass)'; e.currentTarget.style.color = 'var(--color-text-secondary)'; e.currentTarget.style.borderColor = 'var(--color-glass-border)'; }}
            >
              {tpl.name}
            </button>
          ))}
        </div>

        {/* Body */}
        {inputMode === 'grid' ? (
          <div style={{ display: 'flex', padding: 20, gap: 20 }}>
            {/* Weight Selector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: 110, flexShrink: 0 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                {t('range.brushWeight')}
              </div>
              {WEIGHTS.slice().reverse().map(w => (
                <button key={w} onClick={() => setSelectedWeight(w)} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px',
                  borderRadius: 8, border: selectedWeight === w ? '1px solid var(--color-accent)' : '1px solid var(--color-border)',
                  background: selectedWeight === w ? 'rgba(10,132,255,0.1)' : 'var(--color-bg-tertiary)',
                  cursor: 'pointer', fontFamily: 'inherit', transition: 'all 150ms ease',
                }}>
                  <span style={{ width: 14, height: 14, borderRadius: 3, background: WEIGHT_COLORS[w], border: '1px solid rgba(255,255,255,0.15)' }} />
                  <span style={{ fontSize: 12, fontWeight: 700, color: selectedWeight === w ? 'var(--color-accent)' : '#ccc' }}>
                    {w * 100}%
                  </span>
                </button>
              ))}
              <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                <button onClick={() => setWeights({})} style={{
                  padding: '6px 0', borderRadius: 6, border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-tertiary)', color: '#aaa', fontSize: 11, fontWeight: 600,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}>
                  {t('range.clearAll')}
                </button>
                <button onClick={() => {
                  const all: Record<string, number> = {};
                  GRID_LABELS.flat().forEach(lbl => all[lbl] = 1.0);
                  setWeights(all);
                }} style={{
                  padding: '6px 0', borderRadius: 6, border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-tertiary)', color: '#aaa', fontSize: 11, fontWeight: 600,
                  cursor: 'pointer', fontFamily: 'inherit',
                }}>
                  {t('range.allFull')}
                </button>
              </div>
            </div>

            {/* Grid */}
            <div style={{ flex: 1, background: 'var(--color-bg-tertiary)', padding: 3, borderRadius: 10, border: '1px solid var(--color-border)' }}>
              <div style={{ display: 'grid', gap: 2, width: '100%', aspectRatio: '1', gridTemplateColumns: 'repeat(13, 1fr)' }}>
                {GRID_LABELS.flat().map((label, idx) => {
                  const w = weights[label] ?? 0;
                  return (
                    <div key={idx}
                      onPointerDown={() => handlePointerDown(label)}
                      onPointerEnter={() => handlePointerEnter(label)}
                      style={{
                        position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        cursor: 'crosshair', userSelect: 'none', overflow: 'hidden',
                        background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)',
                        borderRadius: 2,
                      }}
                    >
                      <div style={{
                        position: 'absolute', bottom: 0, left: 0, right: 0,
                        height: `${w * 100}%`,
                        background: WEIGHT_COLORS[w] || WEIGHT_COLORS[1.0],
                        opacity: w > 0 ? 0.9 : 0, transition: 'all 150ms ease-out',
                      }} />
                      <span style={{
                        position: 'relative', zIndex: 2,
                        fontFamily: 'var(--font-mono, monospace)', fontSize: 9, fontWeight: 700,
                        color: w > 0.4 ? '#fff' : 'var(--color-text-secondary)',
                        textShadow: w > 0.4 ? '0 1px 2px rgba(0,0,0,0.5)' : 'none',
                      }}>
                        {label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          /* Text Input Mode */
          <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
              {t('range.pasteHint')}
            </div>
            <textarea
              value={rangeText}
              onChange={e => setRangeText(e.target.value)}
              placeholder="AA,KK,QQ,AKs,AKo,AQs..."
              spellCheck={false}
              style={{
                width: '100%', height: 120, padding: 12, borderRadius: 8,
                background: 'var(--color-bg-primary)', color: 'var(--color-text-primary)',
                border: '1px solid var(--color-border)', fontFamily: 'var(--font-mono, monospace)',
                fontSize: 13, resize: 'vertical', outline: 'none',
              }}
            />
            {/* Preview */}
            {rangeText && (() => {
              const parsed = parseRangeText(rangeText);
              const count = Object.keys(parsed).length;
              return (
                <div style={{ fontSize: 12, color: count > 0 ? 'var(--color-green)' : 'var(--color-red)' }}>
                  {count > 0 ? t('range.parsed', { n: String(count) }) : t('range.noValid')}
                </div>
              );
            })()}
            <button onClick={applyText} style={{
              alignSelf: 'flex-start', padding: '8px 20px', borderRadius: 8,
              background: 'var(--color-accent)', color: '#fff',
              border: 'none', fontSize: 13, fontWeight: 700, cursor: 'pointer',
              fontFamily: 'inherit',
            }}>
              {t('range.applyGrid')}
            </button>
          </div>
        )}

        {/* Footer */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
          padding: '12px 20px', borderTop: '1px solid var(--color-border)',
          background: 'var(--color-bg-secondary)', gap: 10,
        }}>
          <button onClick={onClose} style={{
            padding: '8px 16px', borderRadius: 8, border: '1px solid var(--color-border)',
            background: 'var(--color-bg-tertiary)', color: '#ccc',
            fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
          }}>
            {t('cancel')}
          </button>
          <button onClick={handleSave} style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '8px 20px', borderRadius: 8,
            background: 'var(--color-accent)', color: '#fff', border: 'none',
            fontSize: 13, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit',
          }}>
            <Check size={16} />
            {t('range.apply')}
          </button>
        </div>
      </div>
    </div>
  );
}
