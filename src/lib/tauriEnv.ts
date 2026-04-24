/**
 * Tauri environment detection.
 *
 * In Tauri v1, the global was `window.__TAURI__`.
 * In Tauri v2, it was renamed to `window.__TAURI_INTERNALS__` and the legacy
 * `__TAURI__` is only exposed when `app.withGlobalTauri = true` in
 * tauri.conf.json (which we don't set).
 *
 * Every runtime check in the app should go through this helper so we never
 * drift between files again.
 */

export function isTauri(): boolean {
  if (typeof window === 'undefined') return false;
  const w = window as unknown as Record<string, unknown>;
  return '__TAURI_INTERNALS__' in w || '__TAURI__' in w;
}
