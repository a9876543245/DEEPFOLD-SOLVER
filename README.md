# DEEPFOLD-SOLVER

> GPU-accelerated GTO poker solver · CPU fallback · Trilingual UI · One-click installer

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.0.1-green)
![License](https://img.shields.io/badge/license-Commercial-lightgrey)

DEEPFOLD-SOLVER is the desktop GTO solver from [DEEPFOLD](https://deepfold.co).
It ships with a drill mode, a pre-solved spot library, and the full CFR engine
— automatically on your GPU when one is available, on CPU when not.

---

## Download

**Windows 10 / 11 (x64)**

- [Download latest installer (.exe)](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)
  ← NSIS, recommended
- Also available: `.msi` (enterprise)

After the first install, future updates download and apply automatically —
you'll see a small banner in the app's top-left corner the next time a new
version is available.

---

## System requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | Dual-core | Quad-core or better |
| RAM | 4 GB | 8 GB+ |
| GPU | — (CPU fallback works) | NVIDIA RTX 2000 series or newer, 4 GB+ VRAM |
| Disk | 200 MB | 200 MB |
| Internet | Required for sign-in | Required for sign-in |

**GPU detection is automatic.** If you have a supported NVIDIA card, solves
run on it; otherwise they fall back to CPU. A pill in the top-right corner
tells you which mode you're in.

---

## Getting started

1. Install and launch the app.
2. Click **Sign in with Google**. Your system browser will open; sign in with
   the same Google account you use on [deepfold.co](https://deepfold.co).
3. DEEPFOLD PRO members land straight in the solver.

No PRO membership yet? Upgrade at [deepfold.co](https://deepfold.co).

---

## Why DEEPFOLD-SOLVER

- **GPU-first** — NVIDIA CUDA backend runs the full CFR hot loop on your
  card. On an RTX 5090, a typical turn solve at 100 iterations finishes in
  ~2 seconds. CPU fallback keeps the app usable on any machine.
- **Real GTO, not a lookup table** — Discounted CFR with vectorized kernels.
  Every solve produces a fresh equilibrium for YOUR pot size, stack depth,
  and ranges. No cached approximations.
- **Desktop, not SaaS** — No monthly fee per seat, no server round-trips.
  Solves stay on your machine.
- **Auto-updates** — Signed with minisign, verified at launch. A banner
  appears when a new version is ready; one click installs and relaunches.
- **Trilingual UI** — English / 中文 / 日本語 switchable on the fly.

## What's inside

- **GTO Solver** — full DCFR implementation with GPU acceleration. Common
  turn spots converge to sub-percent exploitability in seconds.
- **Per-combo strategy grid** — 13×13 grid shows each hand's action mix at
  the current decision node. Hover to see suited-variant breakdown.
- **Acting ↔ Opponent view** — toggle between your strategy at the node
  and the opponent's reach-weighted range at the same node (heatmap).
  See exactly how a line has narrowed villain's range.
- **Flexible bet sizing** — pick Standard / Polar / Small Ball presets.
  The sizing flows through to the solver tree AND the UI action buttons,
  so what you click is what gets solved.
- **Training mode** — 10-question drill sessions that quiz you on GTO play
  and score your answers against the equilibrium.
- **Pre-solved spot library** — 120+ common flop spots, one click to load.
- **Range editor + node locking** — override any combo frequency and re-solve.

See the full [feature breakdown on deepfold.co](https://deepfold.co/solver).

---

## Support

- **Bug reports / feature requests**:
  [open an issue](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **Purchase / membership questions**:
  [contact@deepfold.co](mailto:contact@deepfold.co)
- **Discord community**: (link TBD)

When filing a bug, please include:
- Your app version (top-right corner of the window, or in **About**)
- The backend pill state at the time: **GPU** / **CPU** / **DEMO**
- OS version (Windows → Settings → About → Windows specifications)
- Screenshot or screen recording if it's a UI issue

---

## Frequently asked questions

**Does the app work without a GPU?**
Yes. It auto-detects and falls back to CPU. Solves are slower but numerically
identical.

**Does it run on macOS or Linux?**
Not yet. Windows only in v1.0.x. macOS / Linux builds are on the roadmap.

**Is my Google account being shared with this app?**
No. Sign-in runs through your system browser using OAuth 2.0 (standard OIDC).
The app only receives a one-time Google id_token that it hands to the DEEPFOLD
API to verify PRO status. No password is ever visible to the app.

**How big is the install?**
~4 MB installer. The solver binary is ~400 KB (CUDA libraries load from your
system). Total disk footprint after install: ~200 MB.

**Are solver strategies uploaded anywhere?**
No. All solves run locally on your machine. Nothing is sent to DEEPFOLD
servers other than the sign-in check.

**How do updates work?**
On launch, the app quietly checks this repo's latest release. If there's a
newer version with a valid signature, a banner appears at the top of the
window — one click to install, app restarts automatically.

---

## License

DEEPFOLD-SOLVER is commercial software. Installers distributed here are for
DEEPFOLD PRO members only. Unauthorized redistribution is prohibited.

The source code is not published in this repository.

© DEEPFOLD · All rights reserved
