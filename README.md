# DEEPFOLD-SOLVER

> GPU-accelerated GTO poker solver · CPU fallback · Trilingual UI · One-click installer

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[User Guide (English)](USER_GUIDE.md)** · **[使用說明 (中文)](USER_GUIDE.zh.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.1.0-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER is the desktop GTO solver from [DEEPFOLD](https://deepfold.co). It pairs a GPU-accelerated DCFR engine (with full CPU fallback) with **runout aggregation, per-combo blocker analysis, and a 2,500+ preflop chart browser** — all in a single Windows installer.

## What's new in v1.1.0

> v1.1.0 is a feature drop focused on **post-solve insight tools** that other
> solvers don't have. The engine itself also got a serious resource-safety pass.

### Three flagship features

- **Runout Report** — One click after a solve fans out every canonical turn
  card into a single 13×4 grid colored by dominant action. Switch to **By
  Class** view and the 23+ turns are grouped into **Pair / Flush / Straight /
  Overcard / Brick** texture buckets with weighted strategy + EV per bucket.
  Sort by Best EV / Worst EV / Most aggressive. Export to CSV.
- **1326 Combo Drill** — Click any 169-class label to expand the 4/6/12
  specific combos in that class with **per-combo blocker analysis**. See
  exactly how much of the opponent's range each specific hand removes,
  with the top-5 most-blocked opponent classes called out. The standard
  poker tie-breaker for mixed strategies, finally first-class in the UI.
- **Memory Profile presets** — `safe / balanced / performance` profiles
  pick host-RAM, JSON, and strategy-tree-node budgets up front. The
  solver respects the budget end-to-end — no more silent OOM kills.

### Engine resource safety

- **Pre-backend budget gate** — CPU host / GPU VRAM / AUTO fallback all
  evaluated *before* allocation. OOM scenarios become structured errors
  with a UI badge, not crashes.
- **CUDA exception-based error handling** — `CUDA_CHECK` throws
  `CudaError` instead of `exit()`. Partial allocations roll back cleanly
  on failure.
- **Chunked GPU matchup upload** — host-side `flat_ev` / `flat_valid`
  duplication eliminated. Upload happens per-runout via `cudaMemset +
  cudaMemcpy`. Lower peak host RAM during GPU prep.
- **Strategy tree EV emission modes** — `none | visible | full` lets you
  trim the JSON output for narrow workflows (e.g. headless benchmarks).
- **Test layering** — ctest now has labeled suites: `smoke` (~13s),
  `correctness` (~106s), `stress` (nightly), plus dedicated `gpu` and
  `memory` labels.

### Carried forward from v1.0.4–1.0.11

- Phase 2 suit isomorphism (3–7× speedup on monotone / three-of-suit boards)
- GPU per-runout matchup tables (6–10× over CPU on iso-engaged trees)
- Route A navigation cache (O(1) action switching, no re-solve)
- Path B runout selector (PioSolver-style chance-aware navigation)
- GameContextSelector — Cash 6max/8max + MTT + stack picker

## Download

**Windows 10 / 11 (x64)** — [Latest installer](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

After install, the app self-updates: a banner appears in the top-left when a new release is available; one click installs and restarts.

> ⚠️ **First-run Windows warning**: when you launch the installer, Windows
> will show a "**Windows protected your PC**" (SmartScreen) prompt. This is
> expected — DEEPFOLD-SOLVER doesn't yet ship with an EV code-signing
> certificate, so Windows doesn't recognize the publisher. Click
> **More info** → **Run anyway** to continue. Full walkthrough in the
> [User Guide — SmartScreen warning](USER_GUIDE.md#appendix-first-install-windows-smartscreen-warning).

## System requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | Dual-core | Quad-core or better |
| RAM | 4 GB | 8 GB+ |
| GPU | — (CPU fallback works) | NVIDIA RTX 2000 series or newer, 4 GB+ VRAM |
| Disk | 200 MB | 200 MB |

GPU detection is automatic. The status pill in the top-right shows **CUDA** / **CPU** at a glance.

## Getting started

1. Install and launch the app.
2. Click **Sign in with Google** — your system browser opens for OAuth.
3. DEEPFOLD PRO members land straight in the solver.

Not a member yet? Upgrade at [deepfold.co](https://deepfold.co).

## What's inside

| Feature | Description |
|---|---|
| **GTO Solver** | Discounted CFR with vectorized GPU kernels. Sub-percent exploitability in seconds for typical turn spots. |
| **Per-combo strategy grid** | 13×13 grid colored by action mix at the current decision node. Hover for suited-variant breakdown. |
| **Acting ↔ Opponent view** | Toggle between your strategy and the opponent's reach-weighted range at the same node. |
| **🆕 Runout Report (v1.1.0)** | After any solve, fan out all enumerated turn cards into a 13×4 grid + texture-bucket view + 4 sort modes + CSV export. See the [User Guide](USER_GUIDE.md#2-runout-report--see-every-turn-at-once). |
| **🆕 1326 Combo Drill (v1.1.0)** | Expand any 169-class into its 4/6/12 specific combos with per-combo blocker analysis vs the opponent's range. See the [User Guide](USER_GUIDE.md#3-combo-drill--break-169-classes-into-specific-combos). |
| **🆕 Memory Profile (v1.1.0)** | `safe / balanced / performance` presets to bound host-RAM, JSON, and strategy-tree-node budgets. No more silent OOM kills. |
| **Runout picker** | When iso enumeration is engaged, click any canonical river card to switch subtrees. |
| **GTO chart library** | 2,550+ bundled preflop scenarios browsable in-app. One click applies as IP / OOP range. |
| **Bet sizing presets** | Standard / Polar / Small Ball — flows through to the solver tree AND the UI buttons. |
| **Training mode** | 10-question drills that score your answers against the equilibrium. |
| **Pre-solved spot library** | 120+ common flop spots, one click to load. |
| **Range editor + node locking** | Override any combo frequency and re-solve. |

## Architecture

```
┌─────────────────────────────────────────────┐
│  React + TypeScript UI (Tauri webview)      │
│  ├── Strategy grid · Runout picker          │
│  └── GTO chart browser                      │
├─────────────────────────────────────────────┤
│  Rust (Tauri) — IPC + chart loader          │
├─────────────────────────────────────────────┤
│  C++ engine (deepsolver_core)               │
│  ├── DCFR (CPU)                             │
│  ├── CUDA kernels (GPU)                     │
│  └── Suit isomorphism + per-runout matchup  │
└─────────────────────────────────────────────┘
```

The engine is a standalone CLI (`deepsolver_core.exe`) shipped as a Tauri sidecar. Tauri spawns it per solve and parses the JSON result, including a full strategy tree for client-side navigation.

## Building from source

Requires:
- **Node.js 20+** + **npm**
- **Rust 1.78+** (`rustup`)
- **CMake 3.20+** + **MSVC 2022** (Windows)
- **CUDA Toolkit 12.x** (optional — CPU build works without)

```sh
git clone https://github.com/a9876543245/DEEPFOLD-SOLVER.git
cd DEEPFOLD-SOLVER
npm install

# Build C++ engine
cd core && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest

# Run dev app (engine binary picked up from ../core/build/Release/)
cd ../..
npm run tauri dev
```

> **Note**: the bundled GTO preflop chart library (`gto_output/`, ~31MB) and
> the precompiled engine sidecar (`src-tauri/binaries/`) are NOT in this
> repository — they ship inside the official installer. Builds from source
> won't have a populated chart browser unless you provide your own
> `gto_output/` directory at the repo root in the same JSON schema. Sign-in
> additionally requires `DEEPFOLD_GOOGLE_CLIENT_SECRET` set in the build
> environment (without it, OAuth will fail at runtime).

## Support

- **Bug reports / feature requests**: [open an issue](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **Membership questions**: [contact@deepfold.co](mailto:contact@deepfold.co)

When filing a bug, please include:
- App version (top-right of window, or **About**)
- Backend pill state at the time: **CUDA** / **CPU**
- Windows version (Settings → About)
- Screenshot or screen recording for UI issues

## FAQ

**Does it work without a GPU?**
Yes. Auto-detects and falls back to CPU. Slower but produces identical strategies.

**macOS / Linux support?**
Not in v1.0.x. On the roadmap.

**Are solver strategies uploaded anywhere?**
No. Everything runs locally. The only network call is the OAuth sign-in check against deepfold.co.

**How do updates work?**
On launch the app checks the latest GitHub release. If signed and newer, a banner offers one-click update.

## License

DEEPFOLD-SOLVER source is published for transparency. Installers are intended for DEEPFOLD PRO members. © DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)
