import React, { useEffect, useState } from 'react';
import { ArrowDownCircle, X, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { isTauri } from '../lib/tauriEnv';
import { useT } from '../lib/i18n';

type UpdateState =
  | { kind: 'idle' }
  | { kind: 'available'; version: string; notes: string }
  | { kind: 'downloading'; downloaded: number; total: number }
  | { kind: 'installed' }
  | { kind: 'error'; message: string };

/**
 * Top-of-window banner that checks for Tauri updates on mount.
 *
 * Flow:
 *  1. On mount, if we're in Tauri, call `check()` from `@tauri-apps/plugin-updater`.
 *  2. If an update is available, show banner with version + "Install now" button.
 *  3. User clicks → downloadAndInstall with progress → restart when complete.
 *  4. Banner dismissible; state persisted in sessionStorage so it doesn't
 *     re-prompt every window open.
 *
 * In browser mode this component is a no-op.
 */
export function UpdateBanner() {
  const t = useT();
  const [state, setState] = useState<UpdateState>({ kind: 'idle' });
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!isTauri()) return;
    if (sessionStorage.getItem('deepfold-update-dismissed') === '1') {
      setDismissed(true);
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const { check } = await import('@tauri-apps/plugin-updater');
        const update = await check();
        if (cancelled) return;
        if (update) {
          setState({
            kind: 'available',
            version: update.version,
            notes: update.body || '',
          });
        }
      } catch (err) {
        // Silent fail — user doesn't need to know if the update check itself failed.
        // They'll just keep running their current version.
        // eslint-disable-next-line no-console
        console.warn('[Updater] check failed:', err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const install = async () => {
    try {
      const { check } = await import('@tauri-apps/plugin-updater');
      const update = await check();
      if (!update) {
        setState({ kind: 'error', message: 'No update available' });
        return;
      }

      let downloaded = 0;
      let total = 0;

      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case 'Started':
            total = event.data.contentLength ?? 0;
            setState({ kind: 'downloading', downloaded: 0, total });
            break;
          case 'Progress':
            downloaded += event.data.chunkLength;
            setState({ kind: 'downloading', downloaded, total });
            break;
          case 'Finished':
            setState({ kind: 'installed' });
            break;
        }
      });

      // Tauri auto-restarts the app once install completes.
      setState({ kind: 'installed' });
      // Give the UI a beat to show "installed" before the relaunch fires.
      setTimeout(async () => {
        try {
          const { relaunch } = await import('@tauri-apps/plugin-process');
          await relaunch();
        } catch {
          // plugin-process might not be installed; Tauri itself handles relaunch
          // for most update flows. Fall back silently.
        }
      }, 800);
    } catch (err) {
      setState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const dismiss = () => {
    setDismissed(true);
    sessionStorage.setItem('deepfold-update-dismissed', '1');
  };

  if (!isTauri()) return null;
  if (dismissed) return null;
  if (state.kind === 'idle') return null;

  // ---- Render by state ----
  // NOTE: the banner gets dropped into a CSS grid in App.tsx — without
  // `gridColumn: '1 / -1'` it lands in the first column only and the
  // sidebar squeezes the content. Spanning all columns keeps the banner
  // as a full-width horizontal bar at the top of the workspace.
  const base: React.CSSProperties = {
    gridColumn: '1 / -1',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 16px',
    borderRadius: 12,
    background: 'var(--color-glass)',
    border: '1px solid var(--color-glass-border)',
    color: 'var(--color-text-primary)',
    fontSize: 13,
    fontFamily: 'var(--font-sans)',
    margin: '8px 16px',
    minWidth: 0,  // allow content inside flex to truncate instead of overflowing
  };

  if (state.kind === 'available') {
    return (
      <div style={{ ...base, borderColor: 'rgba(10,132,255,0.4)' }}>
        <ArrowDownCircle size={18} color="#0A84FF" style={{ flexShrink: 0 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {t('update.available') || 'Update available'} — v{state.version}
          </div>
          {state.notes && (
            <div
              style={{
                fontSize: 11,
                color: 'var(--color-text-secondary)',
                marginTop: 2,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {state.notes.split('\n')[0]}
            </div>
          )}
        </div>
        <button
          onClick={install}
          style={{
            padding: '6px 14px',
            borderRadius: 8,
            border: 'none',
            background: 'linear-gradient(135deg,#0A84FF,#BF5AF2)',
            color: '#fff',
            fontWeight: 700,
            fontSize: 12,
            cursor: 'pointer',
          }}
        >
          {t('update.install') || 'Install now'}
        </button>
        <button
          onClick={dismiss}
          aria-label="Dismiss"
          style={{
            padding: 6,
            borderRadius: 6,
            border: 'none',
            background: 'transparent',
            color: 'var(--color-text-tertiary)',
            cursor: 'pointer',
          }}
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  if (state.kind === 'downloading') {
    const pct = state.total > 0 ? Math.min(100, (state.downloaded / state.total) * 100) : 0;
    return (
      <div style={base}>
        <Loader2
          size={18}
          color="#0A84FF"
          style={{ animation: 'spin 1s linear infinite' }}
        />
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>
            {t('update.downloading') || 'Downloading update…'} {pct.toFixed(0)}%
          </div>
          <div
            style={{
              height: 4,
              borderRadius: 2,
              background: 'var(--color-bg-tertiary)',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${pct}%`,
                background: 'linear-gradient(90deg,#0A84FF,#BF5AF2)',
                transition: 'width 200ms ease',
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  if (state.kind === 'installed') {
    return (
      <div style={{ ...base, borderColor: 'rgba(50,215,75,0.4)' }}>
        <CheckCircle size={18} color="#32D74B" />
        <div style={{ flex: 1, fontWeight: 700, color: '#32D74B' }}>
          {t('update.installed') || 'Update installed — restarting…'}
        </div>
      </div>
    );
  }

  if (state.kind === 'error') {
    return (
      <div style={{ ...base, borderColor: 'rgba(255,69,58,0.4)' }}>
        <AlertCircle size={18} color="#FF453A" />
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, color: '#FF453A' }}>
            {t('update.failed') || 'Update failed'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
            {state.message}
          </div>
        </div>
        <button
          onClick={dismiss}
          style={{
            padding: 6,
            borderRadius: 6,
            border: 'none',
            background: 'transparent',
            color: 'var(--color-text-tertiary)',
            cursor: 'pointer',
          }}
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  return null;
}
