# DEEPFOLD-SOLVER

> GPU-accelerated GTO poker solver · CPU fallback · Trilingual UI · One-click installer

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.0.4-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER is the desktop GTO solver from [DEEPFOLD](https://deepfold.co). It pairs a GPU-accelerated DCFR engine (with full CPU fallback) with a chart browser drawing on **2,500+ preflop scenarios** extracted from training material — all in a single Windows installer.

## Highlights in v1.0.4

- **Phase 2 suit isomorphism** — runout enumeration mirrors PioSolver / GTO+ symmetry compression. Monotone & three-of-suit boards solve **3-7× faster** with no loss of strategy quality.
- **GPU per-runout matchup tables** — chance-node enumeration runs on CUDA. **6-10× speedup** vs CPU on iso-engaged trees.
- **Route A navigation cache** — solve once, click anywhere. Action navigation in the UI is now an **O(1) cache lookup** instead of a fresh solve. No more 8-second waits between clicks.
- **Path B runout selector** — when isomorphism enumerates multiple canonical river cards, the UI now shows a runout picker. Click any canonical card to switch to that subtree's strategy. PioSolver-style chance-aware navigation.
- **GTO chart library** — 2,550+ bundled preflop scenarios (Cash 6max/8max + MTT) browsable in-app, with one-click "apply as IP / OOP range".

## Download

**Windows 10 / 11 (x64)** — [Latest installer](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

After install, the app self-updates: a banner appears in the top-left when a new release is available; one click installs and restarts.

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
| **Runout picker (v1.0.4)** | When iso enumeration is engaged, click any canonical river card to switch subtrees. |
| **GTO chart library (v1.0.4)** | 2,550+ bundled preflop scenarios browsable in-app. One click applies as IP / OOP range. |
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
