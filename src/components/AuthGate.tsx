import React, { useState, useEffect } from 'react';
import type { AuthUser } from '../lib/auth';
import { loginAndVerify, restoreSession, getSavedToken, saveToken, clearSession } from '../lib/auth';
import { useT } from '../lib/i18n';
import { isTauri as isTauriEnv } from '../lib/tauriEnv';

interface Props {
  children: React.ReactNode;
  onAuth: (user: AuthUser) => void;
}

export function AuthGate({ children, onAuth }: Props) {
  const t = useT();
  const [status, setStatus] = useState<'loading' | 'login' | 'denied' | 'ok'>('loading');
  const [user, setUser] = useState<AuthUser | null>(null);
  const [error, setError] = useState<string | null>(null);

  // In browser dev mode (no Tauri), skip auth entirely
  const isTauri = isTauriEnv();

  useEffect(() => {
    if (!isTauri) {
      // Dev mode: auto-grant access
      const devUser: AuthUser = { email: 'dev@local', username: 'Developer', tier: 'pro', allowed: true, token: '' };
      setUser(devUser);
      setStatus('ok');
      onAuth(devUser);
      return;
    }

    // Try to restore session on mount
    const token = getSavedToken();
    if (token) {
      restoreSession(token).then(u => {
        if (u && u.allowed) {
          setUser(u);
          setStatus('ok');
          onAuth(u);
        } else if (u) {
          setUser(u);
          setStatus('denied');
        } else {
          clearSession();
          setStatus('login');
        }
      });
    } else {
      setStatus('login');
    }
  }, []);

  const handleLogin = async () => {
    setError(null);
    try {
      const u = await loginAndVerify();
      setUser(u);
      saveToken(u.token);
      if (u.allowed) {
        setStatus('ok');
        onAuth(u);
      } else {
        setStatus('denied');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  const handleLogout = () => {
    clearSession();
    setUser(null);
    setStatus('login');
  };

  // ---- Authenticated + PRO ----
  if (status === 'ok') {
    return <>{children}</>;
  }

  // ---- Full-screen gate ----
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 200,
      background: 'var(--color-bg-primary)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-sans)',
    }}>
      <div style={{
        width: '100%', maxWidth: 420, padding: 40, textAlign: 'center',
      }}>
        {/* Logo */}
        <div style={{
          width: 64, height: 64, borderRadius: 16, margin: '0 auto 24px',
          background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 28, fontWeight: 800, color: '#fff',
        }}>D</div>

        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>{t('app.title')}</div>
        <div style={{ fontSize: 13, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>
          {t('app.subtitle')}
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', opacity: 0.7, marginBottom: 32 }}>
          {t('app.subtitleNote')}
        </div>

        {/* Loading state */}
        {status === 'loading' && (
          <div style={{ color: 'var(--color-text-secondary)', fontSize: 14 }}>
            <div className="animate-pulse" style={{ marginBottom: 8 }}>Verifying session...</div>
          </div>
        )}

        {/* Login state */}
        {status === 'login' && (
          <div className="animate-fade-in">
            <div style={{
              padding: 24, borderRadius: 16,
              background: 'var(--color-glass)', border: '1px solid var(--color-glass-border)',
              marginBottom: 16,
            }}>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8, color: 'var(--color-text-primary)' }}>
                {t('auth.loginRequired')}
              </div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', lineHeight: 1.6, marginBottom: 20 }}>
                {t('auth.loginDesc')}
              </div>

              <button onClick={handleLogin} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                width: '100%', padding: '12px 20px', borderRadius: 10,
                background: '#fff', color: '#333', border: 'none',
                fontSize: 15, fontWeight: 600, cursor: 'pointer',
                fontFamily: 'inherit', transition: 'all 200ms ease',
              }}
                onMouseOver={e => { e.currentTarget.style.boxShadow = '0 4px 16px rgba(255,255,255,0.15)'; }}
                onMouseOut={e => { e.currentTarget.style.boxShadow = 'none'; }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                {t('auth.googleSignIn')}
              </button>
            </div>

            {error && (
              <div style={{
                padding: 10, borderRadius: 8, fontSize: 12,
                background: 'var(--color-red-dim)', color: 'var(--color-red)',
                border: '1px solid rgba(255,69,58,0.2)',
              }}>{error}</div>
            )}
          </div>
        )}

        {/* Denied state — logged in but not PRO */}
        {status === 'denied' && user && (
          <div className="animate-fade-in">
            <div style={{
              padding: 24, borderRadius: 16,
              background: 'var(--color-glass)', border: '1px solid var(--color-glass-border)',
              marginBottom: 16,
            }}>
              <div style={{
                width: 48, height: 48, borderRadius: '50%', margin: '0 auto 12px',
                background: 'var(--color-orange-dim)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 22,
              }}>&#128274;</div>

              <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4, color: 'var(--color-orange)' }}>
                {t('auth.proRequired')}
              </div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 12, lineHeight: 1.6 }}>
                {t('auth.proDesc')}
              </div>

              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '6px 14px', borderRadius: 8,
                background: 'var(--color-bg-tertiary)', marginBottom: 16,
              }}>
                <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>{user.email}</span>
                <span style={{
                  fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                  background: user.tier === 'basic' ? 'var(--color-blue-dim)' : 'var(--color-bg-tertiary)',
                  color: user.tier === 'basic' ? 'var(--color-accent)' : 'var(--color-text-tertiary)',
                  textTransform: 'uppercase',
                }}>{user.tier}</span>
              </div>

              <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                <a href="https://deepfold.co" target="_blank" rel="noopener" style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  padding: '10px 24px', borderRadius: 10,
                  background: 'linear-gradient(135deg, #0A84FF, #BF5AF2)',
                  color: '#fff', fontSize: 14, fontWeight: 700,
                  textDecoration: 'none', cursor: 'pointer',
                }}>
                  {t('auth.upgradePro')}
                </a>
                <button onClick={handleLogout} style={{
                  padding: '10px 16px', borderRadius: 10, border: '1px solid var(--color-glass-border)',
                  background: 'var(--color-glass)', color: 'var(--color-text-secondary)',
                  fontSize: 13, cursor: 'pointer', fontFamily: 'inherit',
                }}>
                  {t('auth.logout')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
