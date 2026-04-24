import React, { useState, useEffect } from 'react';
import { X, Lock, RefreshCw, AlertCircle } from 'lucide-react';
import { getActionColor } from '../lib/poker';
import { useT } from '../lib/i18n';

interface Props {
  combo: string;
  actions: string[];
  initialStrategy?: Record<string, number>;
  onSave: (strategy: Record<string, number>) => void;
  onClose: () => void;
}

export function NodeLockEditor({ combo, actions, initialStrategy, onSave, onClose }: Props) {
  const t = useT();
  const [strategy, setStrategy] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialStrategy) {
      setStrategy({ ...initialStrategy });
    } else {
      // Initialize uniformly
      const uniform = 1.0 / actions.length;
      const init: Record<string, number> = {};
      actions.forEach(a => init[a] = uniform);
      setStrategy(init);
    }
  }, [actions, initialStrategy]);

  const total = Object.values(strategy).reduce((sum, val) => sum + val, 0);

  const normalize = () => {
    if (total === 0) return;
    const norm: Record<string, number> = {};
    for (const [action, val] of Object.entries(strategy)) {
      norm[action] = val / total;
    }
    setStrategy(norm);
    setError(null);
  };

  const handleSave = () => {
    // Check if sums to roughly 1.0
    if (Math.abs(total - 1.0) > 0.01) {
      setError(`Strategy frequencies must sum to 100% (currently ${(total * 100).toFixed(1)}%)`);
      return;
    }
    onSave(strategy);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative flex flex-col w-full max-w-sm bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-2xl shadow-2xl overflow-hidden animate-slide-up">
        
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
          <div className="flex items-center gap-2">
            <Lock size={18} className="text-[var(--color-primary-blue)]" />
            <h2 className="text-md font-bold text-white">{t('lock.title')}: <span className="font-mono text-[var(--color-primary-blue)]">{combo}</span></h2>
          </div>
          <button onClick={onClose} className="p-1.5 text-gray-400 hover:text-white rounded-full hover:bg-[var(--color-bg-tertiary)]">
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 flex flex-col gap-4">
          <p className="text-xs text-gray-400 mb-2">
            {t('lock.desc')}
          </p>

          <div className="space-y-4">
            {actions.map(action => {
              const val = strategy[action] || 0;
              const color = getActionColor(action);
              return (
                <div key={action} className="flex flex-col gap-1">
                  <div className="flex justify-between items-center text-sm">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
                      <span className="font-semibold">{action}</span>
                    </div>
                    <span className="font-mono text-gray-300">{(val * 100).toFixed(1)}%</span>
                  </div>
                  <input 
                    type="range"
                    min="0" max="100" step="1"
                    value={val * 100}
                    onChange={(e) => {
                      setStrategy(prev => ({ ...prev, [action]: parseInt(e.target.value) / 100.0 }));
                      setError(null);
                    }}
                    className="w-full accent-[var(--color-primary-blue)] cursor-ew-resize"
                  />
                </div>
              );
            })}
          </div>

          {/* Sum / Normalize */}
          <div className="flex items-center justify-between mt-2 pt-4 border-t border-[var(--color-border)]">
            <div className={`text-sm font-bold font-mono ${Math.abs(total - 1.0) < 0.01 ? 'text-green-400' : 'text-red-400'}`}>
              {t('lock.total')}: {(total * 100).toFixed(1)}%
            </div>
            <button 
              onClick={normalize}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-[var(--color-bg-tertiary)] hover:bg-[#444] text-gray-300 transition-colors"
            >
              <RefreshCw size={14} />
              {t('lock.autoBalance')}
            </button>
          </div>

          {error && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs mt-2">
              <AlertCircle size={14} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end p-4 border-t border-[var(--color-border)] bg-[var(--color-bg-secondary)] gap-3">
          <button 
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm font-semibold text-gray-300 hover:bg-[#444] hover:text-white transition-colors border border-transparent"
          >
            {t('cancel')}
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-lg text-sm font-bold text-white bg-[var(--color-primary-blue)] hover:bg-blue-500 transition-all shadow-[0_0_15px_rgba(10,132,255,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={Math.abs(total - 1.0) > 0.01}
          >
            {t('lock.save')}
          </button>
        </div>
      </div>
    </div>
  );
}
