# DEEPFOLD-SOLVER

> GPU-accelerated GTO poker solver · CPU fallback · Trilingual UI · One-click installer

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[User Guide (English)](USER_GUIDE.md)** · **[使用說明 (中文)](USER_GUIDE.zh.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU%20%28AVX2%2BMulti--core%29-orange)

DEEPFOLD-SOLVER is the desktop GTO solver from [DEEPFOLD](https://deepfold.co). It pairs a GPU-accelerated DCFR engine with a multi-core CPU backend that scales linearly across every available core, surrounded by **runout aggregation, per-combo blocker analysis, EV / aggression heatmaps, and a 2,500+ preflop chart browser** — all in a single Windows installer.

## What sets DEEPFOLD-SOLVER apart

### Engine that uses every core you have

- **Dual-backend DCFR** — GPU when you have one, CPU when you don't, identical numerical strategy either way. The GPU path ships native CUDA SASS for Turing / Ampere / Ada / Hopper, with PTX-JIT forward-compat for Blackwell. The CPU path is a BFS-flat "levelized" CFR backend that scales linearly across physical cores instead of capping at two threads.
- **Runtime CPUID dispatch** — AVX2 kernels on Haswell-and-newer, scalar fallback on older silicon. One binary, no separate build, no startup crashes on pre-2013 CPUs.
- **Per-commit parity gate** — every build re-verifies that `reference vs levelized`, `scalar vs AVX2`, and `1-thread vs N-thread` paths emit bit-identical strategies. The fast path can never silently drift from the correctness oracle.

### Solve-mode presets that actually stop on time

- **Quick / Standard / Deep** preset pills bundle iteration cap + time budget + exploitability target into one click. The solver stops at whichever fires first — and because CFR is anytime, the running average at any iteration *is* the strategy, so a budget-stopped solve is a usable strategy, not a half-baked one.
- **Pre-solve ETA banner** — clicking Solve fires a sub-second `--estimate-only` engine call that shows wall-clock time before iterations begin, calibrated against the standard benchmark. Surfaces AUTO fallback reasons (e.g. "Pascal needs CUDA-12.x build") right next to the estimate.
- **Stop button + Quality badge** — pure abort with no partial result preserved (use the time budget for "stop with what we have"). Result panel shows 🟢 high / 🟡 good / 🟠 rough / 🔴 low confidence based on final exploitability%.
- **`--time-budget-seconds`** is checked **every iter**, so even slow per-iter spots stop precisely at budget instead of overshooting.

### Post-solve insight tools other solvers don't have

- **Runout Report** — one click after a solve fans out every canonical turn card into a 13×4 grid colored by dominant action. Switch to **By Class** view and the 23+ turns get grouped into **Pair / Flush / Straight / Overcard / Brick** texture buckets with weighted strategy + EV per bucket. Sort by Best EV / Worst EV / Most aggressive. CSV export.
- **1326 Combo Drill** — click any 169-class label to expand the 4 / 6 / 12 specific combos in that class with **per-combo blocker analysis**: how much of the opponent's range each specific hand removes, plus the top-5 most-blocked opponent classes. The standard tie-breaker for mixed strategies, finally first-class in the UI.
- **Strategy grid view modes** — toolbar above the 169 grid switches between Strategy Mix (default multi-action gradient), **EV** (per-class heatmap, red→grey→green normalized to in-range EV span), **Aggression** (Bet/Raise/All-in frequency cool→hot), and single-action Heatmap. EV mode surfaces "which combos are profit centres vs which are losing" at a glance.

### Memory you can trust

- **Memory Profile presets** — `safe / balanced / performance` pick host-RAM, JSON, and strategy-tree-node budgets up front, with a live preview of each. The solver respects the budget end-to-end — pre-backend gates evaluate CPU host / GPU VRAM / AUTO fallback **before** allocation, so OOM scenarios become structured errors with a UI badge instead of crashes.
- **Common host budget gate** applies on GPU backend too — matchup tables, strategy-tree EV cache, and JSON response all live on host RAM regardless of backend, and all are checked. The diagnostic clarifies that switching to GPU won't fix common-host overflows.
- **Chunked GPU matchup upload** eliminates host-side `flat_ev` / `flat_valid` duplication; upload happens per-runout via `cudaMemset + cudaMemcpy`, lowering peak host RAM during GPU prep.

### Built-in content

- **2,550+ preflop scenarios** browsable in-app. One click applies as IP / OOP range.
- **120+ pre-solved flop spots** in a one-click library.
- **Bet sizing presets** — Standard / Polar / Small Ball — flow through to both the solver tree and the UI buttons.
- **Range editor + node locking** — override any combo frequency and re-solve.
- **Training mode** — 10-question drills that score your answers against the equilibrium.

### Operational polish

- **Trilingual UI** — English / 中文 / 日本語, switchable at any time.
- **Auto-update** — banner-driven one-click installer refresh, signed releases, install-mode `passive`.
- **Suit isomorphism** delivers 3–7× speedup on monotone / three-of-suit boards automatically; per-runout matchup tables on GPU give 6–10× over CPU on iso-engaged trees.
- **Route A navigation cache** — O(1) action switching, no re-solve. **Path B runout selector** for PioSolver-style chance-aware navigation.
- **Reproducible benchmarks** — `deepsolver_core --benchmark standard` runs an AsKd7c rainbow / 100-iter scenario and emits compact perf-tracking JSON (`iterations_per_sec`, `nodes_per_sec`, `memory_estimate_mb`, full timing breakdown). Greppable for CI regression tracking.

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
| CPU | x86-64, dual-core (any year) | 4+ physical cores, AVX2 (Haswell 2013 / Excavator 2015 or newer) |
| RAM | 4 GB | 8 GB+ |
| GPU | — *(CPU backend is fully featured)* | NVIDIA RTX 2000 series or newer, 4 GB+ VRAM |
| Disk | 200 MB | 200 MB |

GPU and SIMD detection are both automatic. The status pill in the top-right shows **CUDA** / **CPU** at a glance, and the CPU backend prints an `AVX2` or `scalar` tag based on what your hardware supports — pre-Haswell CPUs run the scalar kernels and never see an AVX2 opcode.

## Getting started

1. Install and launch the app.
2. Click **Sign in with Google** — your system browser opens for OAuth.
3. DEEPFOLD PRO members land straight in the solver.

Not a member yet? Upgrade at [deepfold.co](https://deepfold.co).

## Feature reference

| Feature | Description |
|---|---|
| **GTO Solver** | Discounted CFR with vectorized GPU kernels and a multi-core CPU backend. Sub-percent exploitability in seconds for typical turn spots. |
| **Per-combo strategy grid** | 13×13 grid colored by action mix at the current decision node. Hover for suited-variant breakdown. |
| **Acting ↔ Opponent view** | Toggle between your strategy and the opponent's reach-weighted range at the same node. |
| **Grid view modes** | Toolbar above the 169 grid switches between Strategy Mix / **EV** / **Aggression** / single-action heatmap. EV mode normalizes red→grey→green across in-range cells so profit centres jump out. |
| **Memory Profile selector** | UI pills for `safe / balanced / performance` in advanced settings with live budget preview. Threads through to the engine via `--memory-profile`. |
| **Benchmark CLI** | `deepsolver_core --benchmark standard` runs a reproducible AsKd7c+100iter scenario and emits compact perf-tracking JSON. |
| **Runout Report** | After any solve, fan out all enumerated turn cards into a 13×4 grid + texture-bucket view + 4 sort modes + CSV export. See the [User Guide](USER_GUIDE.md#2-runout-report--see-every-turn-at-once). |
| **1326 Combo Drill** | Expand any 169-class into its 4/6/12 specific combos with per-combo blocker analysis vs the opponent's range. See the [User Guide](USER_GUIDE.md#3-combo-drill--break-169-classes-into-specific-combos). |
| **Memory Profile** | `safe / balanced / performance` presets to bound host-RAM, JSON, and strategy-tree-node budgets. No more silent OOM kills. |
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
Not currently. On the roadmap.

**Are solver strategies uploaded anywhere?**
No. Everything runs locally. The only network call is the OAuth sign-in check against deepfold.co.

**How do updates work?**
On launch the app checks the latest GitHub release. If signed and newer, a banner offers one-click update.

## License

DEEPFOLD-SOLVER source is published for transparency. Installers are intended for DEEPFOLD PRO members. © DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)
