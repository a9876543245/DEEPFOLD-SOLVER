/**
 * Authentication module for DEEPFOLD-SOLVER.
 *
 * Two login paths, picked automatically at runtime:
 *
 *   1. **Tauri desktop** — system-browser OAuth (the right way for native apps):
 *      - Frontend invokes `start_google_oauth` command.
 *      - Rust opens Google auth URL in the user's system browser and starts
 *        a local callback server on an ephemeral port.
 *      - User signs in in the real browser; Google redirects back to the local
 *        server; Rust emits an `oauth-google-token` event with the id_token.
 *      - Frontend receives the id_token and exchanges it with the DEEPFOLD
 *        backend for a JWT.
 *
 *   2. **Browser dev mode** — Google Identity Services (GSI) in-page popup.
 *      Kept for `npm run dev` only. Won't work in Tauri webview because
 *      Tauri's origin (`tauri://localhost`) can't be authorised in the Google
 *      Cloud Console.
 */

import { isTauri } from './tauriEnv';

// ============================================================================
// Config
// ============================================================================

// API endpoint. Prod API is served from the same origin as the website
// (`https://deepfold.co/api/...`) — there is no `api.deepfold.co` subdomain.
// Override at build time via VITE_API_BASE (e.g. `http://localhost:8000` for
// testing against a local dev server).
const API_BASE = import.meta.env.VITE_API_BASE || 'https://deepfold.co';
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '230259880631-32c386966khp78gmqb62idd3ockr4vff.apps.googleusercontent.com';

// Max time to wait for the user to complete OAuth in the system browser.
const TAURI_OAUTH_TIMEOUT_MS = 5 * 60 * 1000; // 5 min

// ============================================================================
// Types
// ============================================================================

export interface AuthUser {
  email: string;
  username: string;
  tier: 'free' | 'basic' | 'pro';
  allowed: boolean;
  token: string;
}

// ============================================================================
// Tauri OAuth flow (desktop)
// ============================================================================

/**
 * Run Google OAuth via the system browser. Returns the Google id_token once
 * the user completes sign-in. Resolves/rejects based on Tauri event listener.
 */
async function googleSignInTauri(): Promise<string> {
  const [{ invoke }, { listen }] = await Promise.all([
    import('@tauri-apps/api/core'),
    import('@tauri-apps/api/event'),
  ]);

  // Set up listener BEFORE invoking so we don't miss the event if the server
  // fires it between invoke completion and listener registration.
  const tokenPromise = new Promise<string>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('OAuth timed out. Please try again.'));
    }, TAURI_OAUTH_TIMEOUT_MS);

    let unlistenToken: (() => void) | null = null;
    let unlistenManual: (() => void) | null = null;

    const cleanup = () => {
      clearTimeout(timeout);
      if (unlistenToken) unlistenToken();
      if (unlistenManual) unlistenManual();
    };

    listen<string>('oauth-google-token', (event) => {
      cleanup();
      if (typeof event.payload === 'string' && event.payload.length > 0) {
        resolve(event.payload);
      } else {
        reject(new Error('Empty token received from OAuth flow'));
      }
    }).then((un) => {
      unlistenToken = un;
    });

    // If the Rust side couldn't open the system browser, it emits a
    // `oauth-google-manual` event with the URL. Surface that as an error the
    // user can see — UI can pick this up separately if desired.
    listen<string>('oauth-google-manual', (event) => {
      cleanup();
      reject(
        new Error(
          `Could not open browser. Copy this URL into a browser manually:\n${event.payload}`
        )
      );
    }).then((un) => {
      unlistenManual = un;
    });
  });

  await invoke('start_google_oauth');
  return tokenPromise;
}

// ============================================================================
// Browser GSI flow (dev only)
// ============================================================================

let gsiLoaded = false;

/** Load Google Identity Services script */
function loadGoogleScript(): Promise<void> {
  if (gsiLoaded) return Promise.resolve();
  return new Promise((resolve, reject) => {
    if (document.getElementById('google-gsi')) {
      gsiLoaded = true;
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.id = 'google-gsi';
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.onload = () => { gsiLoaded = true; resolve(); };
    script.onerror = () => reject(new Error('Failed to load Google Sign-In'));
    document.head.appendChild(script);
  });
}

/** Trigger Google popup sign-in in a real browser (dev mode only) */
function googleSignInBrowser(): Promise<string> {
  return new Promise((resolve, reject) => {
    const google = (window as any).google;
    if (!google?.accounts?.id) {
      reject(new Error('Google Sign-In not loaded'));
      return;
    }

    google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: (response: any) => {
        if (response.credential) {
          resolve(response.credential);
        } else {
          reject(new Error('No credential returned'));
        }
      },
      auto_select: false,
    });

    google.accounts.id.prompt((notification: any) => {
      if (notification.isNotDisplayed?.() || notification.isSkippedMoment?.()) {
        // Render a button as fallback
        const btn = document.createElement('div');
        btn.id = 'google-btn-temp';
        btn.style.position = 'fixed';
        btn.style.top = '50%';
        btn.style.left = '50%';
        btn.style.transform = 'translate(-50%, -50%)';
        btn.style.zIndex = '9999';
        document.body.appendChild(btn);
        google.accounts.id.renderButton(btn, {
          type: 'standard',
          theme: 'filled_black',
          size: 'large',
          text: 'signin_with',
          shape: 'pill',
          width: 300,
        });
      }
    });
  });
}

// ============================================================================
// API calls (backend exchange)
// ============================================================================

/** Exchange Google ID token for DEEPFOLD JWT */
async function exchangeToken(googleIdToken: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/auth/google`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_token: googleIdToken }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Auth failed: ${err}`);
  }
  const data = await res.json();
  return data.access_token;
}

/** Check if the user has PRO solver access */
async function checkAccess(jwt: string): Promise<{ allowed: boolean; tier: string; email: string; username: string }> {
  const res = await fetch(`${API_BASE}/api/auth/solver-access`, {
    headers: { Authorization: `Bearer ${jwt}` },
  });
  if (!res.ok) throw new Error('Access check failed');
  return res.json();
}

// ============================================================================
// Main auth flow
// ============================================================================

/** Full login flow: Google Sign-In → exchange token → check PRO access */
export async function loginAndVerify(): Promise<AuthUser> {
  let googleToken: string;

  if (isTauri()) {
    googleToken = await googleSignInTauri();
  } else {
    await loadGoogleScript();
    googleToken = await googleSignInBrowser();
  }

  const jwt = await exchangeToken(googleToken);
  const access = await checkAccess(jwt);
  return {
    email: access.email,
    username: access.username,
    tier: access.tier as AuthUser['tier'],
    allowed: access.allowed,
    token: jwt,
  };
}

/** Restore session from saved token (skip Google sign-in) */
export async function restoreSession(jwt: string): Promise<AuthUser | null> {
  try {
    const access = await checkAccess(jwt);
    return {
      email: access.email,
      username: access.username,
      tier: access.tier as AuthUser['tier'],
      allowed: access.allowed,
      token: jwt,
    };
  } catch {
    return null;
  }
}

/** Persist token to localStorage */
export function saveToken(token: string) {
  try { localStorage.setItem('deepfold-solver-token', token); } catch {}
}

/** Read saved token */
export function getSavedToken(): string | null {
  try { return localStorage.getItem('deepfold-solver-token'); } catch { return null; }
}

/** Clear saved session */
export function clearSession() {
  try { localStorage.removeItem('deepfold-solver-token'); } catch {}
}

/** Kept exported for compatibility (no longer used internally). */
export { loadGoogleScript };

/** @deprecated use loginAndVerify — handles Tauri vs browser automatically. */
export const googleSignIn = googleSignInBrowser;
