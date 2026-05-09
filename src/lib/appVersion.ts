/**
 * App version constant — single source of truth for the frontend.
 *
 * Imports from package.json directly via Vite's JSON import. The Tauri build
 * also embeds this same version into the binary via tauri.conf.json (kept in
 * sync by scripts/prepare-release.ps1's version bump).
 *
 * Used by:
 *   - presolvedSpots.ts: invalidate bundled spots whose solver_version doesn't
 *     match (so the UI never shows a stale Deep-quality bundle after a
 *     kernel-affecting solver bump).
 *   - Anywhere we want to render a version badge.
 */

import pkg from '../../package.json';

export const APP_VERSION: string = pkg.version;
