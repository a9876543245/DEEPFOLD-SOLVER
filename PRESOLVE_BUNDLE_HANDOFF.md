# Pre-Solve Bundle — Continuation Handoff

Audience: future me / agent picking up after context compaction
Project: DEEPFOLD-SOLVER
Date written: 2026-05-10
Companion doc: `PRESOLVE_BUNDLE_PLAN.md` (full design)
Workspace memory: `workspace_layout.md` (sandbox vs repo split)

---

## 1. TL;DR — What's shipped, what's pending

**Already done in this session, committed + pushed to main:**

| Commit | What |
|--------|------|
| `8d0379d` | **v1.8.2 release** — A2 matchup encoding + Lite preset + 4 hygiene fixes + phase counters. **GitHub release artifact has STALE sidecar binary (April 29, v1.0.x).** |
| `d4d8866` | Phase 1: `scripts/bulk-presolve.mjs` (Node ESM, reads MATCHUPS via regex, shells out to deepsolver_core.exe) |
| `b9f1dee` | Phase 2 (`scripts/compact-presolved.mjs`) + Phase 3 (`read_bundled_presolve` Rust command + frontend `tryLoadBundled` + UI badge) |
| `34886ab` | Phase 3 polish: SpotLibrary auto-upgrades grid via `getBundledOrDemo` + App.tsx onSelectSpot bypasses live solve on bundled hit + Rust `find_presolve_dir()` (presolve-aware path search) + `scripts/dev.ps1` launcher |
| `4aacef3` | GUI iter caps bumped (Quick 100→500, Standard 300→**3000**, Deep 1000→**10000**) + bulk-presolve tiered (SRP 3000 / 3BET 1500 / Lite 1000) |

**Validated end-to-end:**
- SpotLibrary opens, shows `● PRE-SOLVED` green badge on the 2 bundled spots
- Click bundled spot → **instant 0.00s** load with cached strategy (5.39% exploit, 1000 iter Deep)
- Click non-bundled spot → falls through to live solve via Tauri `solve` command
- Live solve confirmed running on **CUDA RTX 5090** via `window.__TAURI_INTERNALS__.invoke('solve', ...)` test

**Bundle artifacts present** (just 2, smoke-tested):
- `D:/DEEPFOLD-SOLVER/gto_output/presolved/m0_b0_standard_defaultbb.json.gz` (1.4 MB)
- `D:/DEEPFOLD-SOLVER/gto_output/presolved/m0_b0_lite_defaultbb.json.gz` (231 KB)
- ALSO copied into `D:/DEEPFOLD-SOLVER/src-tauri/target/debug/gto_output/presolved/` (dev mode workaround — Rust `find_presolve_dir()` handles this so it's no longer needed after the next dev restart)

---

## 2. Three pending decisions for the user

User left off here. Pick A / B / C when resuming:

### A. Kick off the full bulk run (~10 hours on 5090, background)
```powershell
node D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.mjs
# 624 spots (26 matchups × 12 boards × 2 sizings)
# Tiered: SRP×Standard 3000 iter, 3BET×Standard 1500, Any×Lite 1000
# Output: D:\DEEPFOLD-SOLVER\gto_output\presolved\raw\
# After: node D:\DEEPFOLD-SOLVER\scripts\compact-presolved.mjs
# Final bundle: ~400 MB across 624 .json.gz files
# Quality: SRP ~1.5% exploit, 3BET ~0.5%, Lite ~1%
```
Once done, sync `gto_output/presolved/*.json.gz` to repo and ship v1.8.3.

### B. Test Standard mode 3000 iter in GUI first
User restarts dev, runs a Standard solve on a wide SRP scenario, confirms exploit drops to ~1-2% (vs the old ~5-13% at 300 iter). If acceptable, proceed to A. If too slow on user's machine, lower the iter caps before bulk.

### C. Ship v1.8.3 patch immediately (no bundles yet)
Just bundles GUI Phase 3 wiring + iter cap bumps + REFRESHED sidecar. Users with v1.8.2 auto-update get:
- Working A2 engine (the v1.8.2 sidecar was stale — see below)
- Lite preset working with proper engine
- Bumped iter caps for tighter convergence
- SpotLibrary bundled lookup wiring (no spots bundled yet — falls through to live solve gracefully)

Then later after bulk run, ship v1.8.4 with the actual bundled spots.

**Recommended order: B → A → ship v1.8.3 with bundles included** (skip the intermediate v1.8.3-without-bundles release).

---

## 3. CRITICAL: v1.8.2 installer has stale sidecar

**Problem found mid-session:** `D:\DEEPFOLD-SOLVER-repo\src-tauri\binaries\deepsolver_core-x86_64-pc-windows-msvc.exe` was the **April 29 v1.0.x binary** (622 KB, doesn't even understand `--benchmark standard`). When `prepare-release.ps1` ran for v1.8.2, it bundled this stale binary into the installer.

**End-user impact**: anyone who downloaded v1.8.2 from GitHub gets v1.8.2 frontend (Lite preset, etc) but v1.0.x engine (no A2, no levelized default, ~5x slower).

**Fixed in sandbox + repo** during session by copying fresh CUDA build:
```bash
cp D:/DEEPFOLD-SOLVER/core/build/Release/deepsolver_core.exe \
   D:/DEEPFOLD-SOLVER/src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe
cp D:/DEEPFOLD-SOLVER/core/build/Release/deepsolver_core.exe \
   D:/DEEPFOLD-SOLVER-repo/src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe
```
**These changes were NOT committed** (binaries don't go in git). Each release must refresh the sidecar BEFORE prepare-release.ps1 — add a check in prepare-release.ps1 to verify sidecar timestamp > some threshold.

**For v1.8.3 ship**: must refresh sidecar AGAIN before running prepare-release.ps1 (since the in-repo binary may have rotted).

---

## 4. Sandbox vs Repo workflow rule (memory)

Per `workspace_layout.md` saved memory:
- `D:\DEEPFOLD-SOLVER` = **dev workspace** (sandbox). All edits go here first. `npm run tauri dev` runs from here. Has `.env.local`.
- `D:\DEEPFOLD-SOLVER-repo` = **clean git repo** uploaded to GitHub. Sandbox files copy here for releases.

**Always edit sandbox first.** Mid-session I made the mistake of editing repo for Phase 1/2/3, which left sandbox stale. Then had to reverse-sync when user wanted to dev. Don't repeat.

---

## 5. Quick reference — paths + commands

### Files / locations
- Sandbox root: `D:\DEEPFOLD-SOLVER\`
- Repo root: `D:\DEEPFOLD-SOLVER-repo\` (origin: github.com/a9876543245/DEEPFOLD-SOLVER)
- OAuth secret: `D:\DEEPFOLD-SOLVER\.env.local` (length 35, DEEPFOLD_GOOGLE_CLIENT_SECRET=...)
- Dev launcher: `D:\DEEPFOLD-SOLVER\scripts\dev.ps1` (loads .env.local, runs npm run tauri dev)
- CUDA engine build: `D:\DEEPFOLD-SOLVER\core\build\Release\deepsolver_core.exe` (1.7 MB, last built 2026-05-10)
- CPU engine build: `D:\DEEPFOLD-SOLVER\core\build_cpu\Release\deepsolver_core.exe`
- Sidecar (used by Tauri): `D:\DEEPFOLD-SOLVER\src-tauri\binaries\deepsolver_core-x86_64-pc-windows-msvc.exe`

### Key commands
```powershell
# Dev (auto-loads .env.local)
powershell -ExecutionPolicy Bypass -File D:\DEEPFOLD-SOLVER\scripts\dev.ps1

# Engine rebuild (CUDA support, for 5090)
cmake --build D:/DEEPFOLD-SOLVER/core/build --config Release --target deepsolver_core

# Sidecar refresh (REQUIRED before any release ship)
cp D:/DEEPFOLD-SOLVER/core/build/Release/deepsolver_core.exe \
   D:/DEEPFOLD-SOLVER/src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe
cp D:/DEEPFOLD-SOLVER/core/build/Release/deepsolver_core.exe \
   D:/DEEPFOLD-SOLVER-repo/src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe

# Bulk presolve (5090)
node D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.mjs --dry-run    # plan only
node D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.mjs --limit 5    # smoke test
node D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.mjs              # full ~10 hr run
node D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.mjs --resume     # skip existing

# Compact bundles (after bulk-presolve)
node D:\DEEPFOLD-SOLVER\scripts\compact-presolved.mjs            # full Tier B (with strategy_tree)
node D:\DEEPFOLD-SOLVER\scripts\compact-presolved.mjs --no-tree  # Tier A (preview only, ~50× smaller)

# Engine smoke validation
ctest --test-dir D:/DEEPFOLD-SOLVER/core/build_cpu -C Release -L smoke -j 4

# Release ship (run from repo)
powershell -ExecutionPolicy Bypass -File D:\DEEPFOLD-SOLVER-repo\scripts\prepare-release.ps1 -Version 1.8.3
cd D:/DEEPFOLD-SOLVER-repo/release/1.8.3
gh release create v1.8.3 \
  "DEEPFOLD-SOLVER_1.8.3_x64-setup.exe" \
  "DEEPFOLD-SOLVER_1.8.3_x64-setup.exe.sig" \
  latest.json \
  --repo a9876543245/DEEPFOLD-SOLVER \
  --title "..." --notes "..."
```

---

## 6. Useful debug recipes (for the WebView console / DevTools)

### Direct Tauri command test (bypasses UI)
```js
// Check if sidecar is the new one + which backend AUTO picks
const invoke = window.__TAURI_INTERNALS__.invoke;
const r = await invoke('solve', { request: {
  board: 'KhQd4c',
  pot_size: 55, effective_stack: 975,
  iterations: 100, exploitability: 0.5,
  ip_range: 'AQs:0.20,KQs:0.62',
  oop_range: 'AA:1.0,KK:1.0,QQ:1.0',
  cpu_backend: 'levelized', cpu_simd: 'auto', cpu_threads: 0,
}});
console.log('backend:', r.backend);
// Expect: "CUDA (NVIDIA GeForce RTX 5090, 32606 MB, CC 12.0)" if sidecar fresh
// Expect: "CPU-DCFR-AVX2" + reference backend kind if stale v1.0.x sidecar
```

### Direct bundled lookup test
```js
const m = await import('/src/lib/presolvedSpots.ts');
const r = await m.tryLoadBundled(0, 0);
console.log('result:', r);
// Expect: PresolvedSpot object with source: 'bundled'
// If null + '[presolvedSpots] no bundle for ...' log → file missing at expected path
```

### Verify bundle file format end-to-end
```bash
node -e "
const fs=require('fs'),zlib=require('zlib');
const gz=fs.readFileSync('D:/DEEPFOLD-SOLVER/gto_output/presolved/m0_b0_lite_defaultbb.json.gz');
const b=JSON.parse(zlib.gunzipSync(gz).toString());
console.log({v:b.v, solver_version:b.solver_version, iter:b.iterations_run, exploit:b.exploitability_pct});
"
```

---

## 7. Known gotchas (don't repeat these)

1. **HMR is disabled** in this dev session (some browser shim → React Fast Refresh off). Code changes need **Ctrl+Shift+R hard reload** in webview, NOT just file save.

2. **Tauri dev copies `gto_output/` to `target/debug/gto_output/` at startup**. New files added later won't appear there until next dev restart. **`find_presolve_dir()` in `gto_charts.rs` now handles this** by preferring paths that contain `presolved/` subdir — but for one-off tests just copy:
   ```bash
   cp -rf D:/DEEPFOLD-SOLVER/gto_output/presolved/* \
          D:/DEEPFOLD-SOLVER/src-tauri/target/debug/gto_output/presolved/
   ```

3. **Bare ESM imports don't work in raw browser console** (`import('@tauri-apps/api/core')` → "Failed to resolve module specifier"). Use `window.__TAURI_INTERNALS__.invoke` directly, or `import('/src/lib/foo.ts')` (Vite-served paths).

4. **`option_env!()` is compile-time** — env var must be set BEFORE `cargo build` for OAuth secret to land in binary. `scripts/dev.ps1` handles this; manual cargo runs need `Set-Item Env:DEEPFOLD_GOOGLE_CLIENT_SECRET ...` first.

5. **Console.debug is filtered out** by default Chrome/Edge DevTools log level. Use `console.log` or `console.info` for user-visible diagnostics. We bumped all bundle-related debug logs to log already.

6. **Standard mode iter cap was just bumped from 300 to 3000** in commit 4aacef3. Old caps hit "iter_cap" early on wide trees; new caps let exploit converge below target. Slow hardware unaffected because `time_budget_seconds` (5min for Standard) caps wall time anyway.

---

## 8. PRESOLVE_BUNDLE_PLAN.md status (Phase tracking)

- ✅ **Phase 1** — `scripts/bulk-presolve.mjs` (Node ESM, reads MATCHUPS via regex, shells out to deepsolver_core.exe with tiered iter)
- ✅ **Phase 2** — `scripts/compact-presolved.mjs` (BundledSpot envelope schema v1, gzip)
- ✅ **Phase 3** — `read_bundled_presolve` Tauri command + `tryLoadBundled` frontend + `getBundledOrDemo` for grid + `● PRE-SOLVED` badge
- ⏳ **Phase 4** — Bundle into installer. Currently `gto_output/presolved/` is gitignored. For ship: must un-gitignore the .json.gz files (raw/ stays gitignored), git LFS evaluation, prepare-release.ps1 sidecar-version check.
- ⏳ **Phase 5** — UX polish: SpotLibrary header coverage stat, settings panel cache stats. Optional.

Phase 4 detail: when ready to ship bundles, edit `gto_output/presolved/.gitignore` to allow `*.json.gz`:
```
# inside gto_output/presolved/.gitignore (currently just has "raw/")
raw/
# Bundles ARE committed (used by installer):
!*.json.gz
```
Then `git add gto_output/presolved/*.json.gz`. ~400 MB total at Tier B — consider git-lfs if the repo grows uncomfortably.

---

## 9. v1.9.0 — Migration plan: bundles → GitHub Release artifact

**Decision recorded 2026-05-13**: v1.8.3 ships bundles via plain git commit (~440 MB
into repo, ~880 MB total). Accepted as one-time growth for solo dev workflow.

**v1.9.0 should migrate to**: bundles distributed as a separate Release artifact
(e.g. `presolved-1.9.0.tar.gz`), with first-time-launch download flow in-app.

### Why migrate
- Each bundle regen adds another ~440 MB to git history. After 3 regens we hit
  GitHub's 1 GB soft warning; after ~10 we approach 5 GB hard limit.
- Clone time for new dev / fresh CI ballooned to ~500 MB.
- Installer .exe currently ~500 MB (Tauri base ~5 MB + bundles 440 MB +
  sidecar 1.7 MB + frontend dist). Release-artifact split lets installer
  stay ~10 MB, optional bundle download per user preference.

### v1.9.0 work breakdown
1. **Repo prune** — `git filter-repo --path gto_output/presolved/ --invert-paths`
   to retroactively yank bundles from git history. Forces re-clone for everyone
   but `.git` drops back to ~5 MB.
2. **Update `tauri.conf.json`** — remove the `bundle.resources` entry for
   `gto_output/presolved/`. Installer stops bundling them.
3. **App-side bundle downloader** —
   - On first launch (or after solver_version bump), download `presolved-X.Y.Z.tar.gz`
     from `github.com/<org>/DEEPFOLD-SOLVER/releases/download/vX.Y.Z/`.
   - Verify SHA256 checksum (publish alongside .tar.gz).
   - Extract to user data dir (e.g. `%APPDATA%/co.deepfold.solver/presolved/`).
   - `gto_charts.rs:find_presolve_dir()` already walks `<exe>/resources/`, just
     add user-data-dir as a search candidate.
   - Show download progress UI on first run.
4. **Release pipeline** — `scripts/prepare-release.ps1` should:
   - Compress `gto_output/presolved/` to `release/X.Y.Z/presolved-X.Y.Z.tar.gz`
   - Compute SHA256 → write to `release/X.Y.Z/presolved-X.Y.Z.sha256`
   - `gh release create` picks both up alongside the .exe / .sig / latest.json
5. **`latest.json` schema bump** — embed bundle artifact URL + checksum so the
   updater grabs the matching bundle when migrating users from v1.8.3 → v1.9.0.

### Effort estimate
~1-2 days of focused work. The hardest part is the in-app download UI with
correct error states (no network, partial download, checksum mismatch, disk full).

---

## 10. When you resume — first 3 things to do

1. **Read this doc** (you're doing it now).
2. **Verify sandbox + repo are still in sync** for the recently-touched files:
   ```bash
   for f in src/lib/poker.ts src/lib/presolvedSpots.ts src/components/SpotLibrary.tsx \
            src/App.tsx src-tauri/src/gto_charts.rs scripts/bulk-presolve.mjs \
            scripts/compact-presolved.mjs scripts/dev.ps1; do
     diff -q "D:/DEEPFOLD-SOLVER/$f" "D:/DEEPFOLD-SOLVER-repo/$f"
   done
   ```
   No diff output = synced. Any output = sync needed before further changes.
3. **Ask user which path** (A/B/C from §2) they want to take next.
