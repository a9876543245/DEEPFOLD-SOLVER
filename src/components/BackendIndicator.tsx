import React, { useEffect, useState } from 'react';
import { Cpu, Zap, AlertTriangle } from 'lucide-react';
import { detectBackend } from '../lib/backend';
import type { BackendStatus } from '../lib/backend';
import { useT } from '../lib/i18n';

/**
 * Small pill in the header showing which execution backend is active.
 * Colors:
 *   green  = GPU functional
 *   blue   = CPU (normal operation)
 *   orange = Demo mode (browser, heuristic only)
 *   gray   = Detecting
 */
export function BackendIndicator() {
  const t = useT();
  const [status, setStatus] = useState<BackendStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    detectBackend().then((s) => {
      if (!cancelled) setStatus(s);
    });
    return () => { cancelled = true; };
  }, []);

  // ---- Detecting (initial state) ----
  if (!status) {
    return (
      <div
        title={t('backend.detecting')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          padding: '3px 8px',
          borderRadius: 'var(--radius-full)',
          background: 'var(--color-glass)',
          color: 'var(--color-text-tertiary)',
          fontSize: 10,
          fontWeight: 600,
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--color-text-tertiary)',
            opacity: 0.6,
          }}
        />
        …
      </div>
    );
  }

  // ---- Color / icon / label per mode ----
  let color = 'var(--color-text-tertiary)';
  let label = 'CPU';
  let icon: React.ReactNode = <Cpu size={11} />;

  if (status.mode === 'gpu') {
    color = '#32D74B'; // green
    label = t('backend.gpu');
    icon = <Zap size={11} />;
  } else if (status.mode === 'demo') {
    color = 'var(--color-orange, #FF9500)';
    label = 'DEMO';
    icon = <AlertTriangle size={11} />;
  } else if (status.mode === 'cpu') {
    color = 'var(--color-accent, #0A84FF)';
    label = t('backend.cpu');
    icon = <Cpu size={11} />;
  }

  return (
    <div
      title={status.description}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        padding: '3px 8px',
        borderRadius: 'var(--radius-full)',
        background: 'var(--color-glass)',
        color,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: 0.3,
        border: `1px solid ${color}33`,
        cursor: 'help',
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: color,
          boxShadow: `0 0 6px ${color}66`,
        }}
      />
      {icon}
      {label}
    </div>
  );
}
