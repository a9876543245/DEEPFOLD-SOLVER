# DEEPFOLD-SOLVER User Guide

> Complete walkthrough of the v1.1.0 features — Runout Report, Combo Drill, Memory Profile

**[English](USER_GUIDE.md) · [中文](USER_GUIDE.zh.md)**

---

## Contents

1. [First run](#1-first-run)
2. [Runout Report — see every turn at once](#2-runout-report--see-every-turn-at-once)
   - [2.1 By Card view (13×4 grid)](#21-by-card-view-134-grid)
   - [2.2 By Class view (5 texture buckets)](#22-by-class-view-5-texture-buckets)
   - [2.3 Sort modes](#23-sort-modes)
   - [2.4 CSV export](#24-csv-export)
3. [Combo Drill — break 169 classes into specific combos](#3-combo-drill--break-169-classes-into-specific-combos)
   - [3.1 How to open](#31-how-to-open)
   - [3.2 Class picker](#32-class-picker)
   - [3.3 Reading the blocker numbers](#33-reading-the-blocker-numbers)
4. [Memory Profile — solver resource policy](#4-memory-profile--solver-resource-policy)
5. [FAQ](#5-faq)
6. [Appendix: First-install Windows SmartScreen warning](#appendix-first-install-windows-smartscreen-warning)

---

## 1. First run

| Step | Action |
|---|---|
| 1 | Install → launch |
| 2 | Click **Sign in with Google** (system browser opens for OAuth) |
| 3 | Set board (e.g. `AsKsQs`) + ranges + bet sizings |
| 4 | Click **Solve** |
| 5 | Wait for the progress bar (small spots: seconds, turn spots: tens of seconds, river full tree: depends on memory profile) |

After the solve completes, the right-hand **StrategyPanel** shows two new buttons:

```
┌───────────────────────────────────┐
│  [📊 Runout Report   23 turns]    │  ← see Section 2
│  [🎴 Combo Drill     AKs    ]    │  ← see Section 3
└───────────────────────────────────┘
```

> 💡 **Runout Report** only appears when the board enumerated multiple
> turn cards (monotone / two-tone / paired flops enumerate; rainbow
> flops collapse to a single chance child to bound memory).

> 💡 **Combo Drill** is always available after a solve. If you previously
> clicked a class in the main grid, the button shows that class as a chip
> (e.g. `AKs`) and the modal opens directly into it.

---

## 2. Runout Report — see every turn at once

**Problem solved**: traditional solver UIs show you "the strategy at the
current turn" — to see how strategy changes across all 23 possible turn
cards you'd click 23 times. Runout Report fans out every turn into a
single view. **The whole turn street, in one second.**

### 2.1 By Card view (13×4 grid)

Default mode. 13 columns (rank) × 4 rows (suit). Each cell is colored by
the **dominant action** on that turn.

```
       2  3  4  5  6  7  8  9  T  J  Q  K  A
  ♣  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♦  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♥  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♠  [.][.][.][.][.][.][.][.][.][.][.][.][.]
```

**Color key**:

| Color | Action |
|---|---|
| Grey | Check / Call |
| Dark grey | Fold |
| Green | Bet ≤ 33% pot |
| Orange | Bet 34–75% pot |
| Red | Bet > 75% pot |
| Purple | Raise |
| Deep red | All-in |
| Blue | Other |

**Cell content**:

- Big number = dominant action's frequency (e.g. `67%`)
- Small `×N` = this turn is an iso class representing N real cards
  (e.g. on a monotone-spade flop, hearts/diamonds/clubs collapse into
  one canonical rep with weight 3 by symmetry)

**Flop-card cells**: the 3 cards already on the flop are rendered as
dashed boxes labeled `flop` (impossible to deal again on the turn).

**Hover for detail**: hovering any cell pops a DetailPane below the grid:

```
┌─────────────────────────────────────────┐
│  9♠   IP acts · iso class ×1           │
│  [Check 40%] [Bet_33 35%] [All-in 25%] │
│  Mean EV: 8.4 chips                    │
└─────────────────────────────────────────┘
```

### 2.2 By Class view (5 texture buckets)

Click the **By Class** tab in the header → all 23 turns are grouped by
board texture into 5 buckets, with weighted strategy and EV per bucket:

| Bucket | Definition | Example (AsKsQs flop) |
|---|---|---|
| **Pair** | Turn rank matches a flop rank | A♣ K♣ Q♣ |
| **Flush completion** | Turn suit has ≥ 2 on flop | 2♠ 3♠ 4♠ … J♠ T♠ |
| **Straight completion** | Turn rank is in `[min flop rank − 1, max flop rank + 1]` | J♣ |
| **Overcard** | Turn rank strictly above all flop ranks | (none here — A is already top) |
| **Brick** | None of the above | 2♣–9♣, T♣ |

Each bucket renders as a card with a colored left border:

```
┃ ⬤ Flush completion       10 cards · weight 10
┃ Mean EV: 7.3 chips
┃ ▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░  ← stacked strategy bar
┃ [Check 32%] [Bet_33 33%] [All-in 35%]
┃ 2S · 3S · 4S · 5S · 6S · 7S · 8S · 9S · TS · JS
```

**Weighting**: each turn contributes `iso_weight × strategy_freq` to the
bucket sum, then divided by total weight. A weight-3 canonical rep affects
the bucket average like 3 weight-1 turns. EV is weighted the same way.

> ⚠️ **The straight bucket is a coarse approximation.** The simplified
> rule only checks if the turn rank lies in `[min−1, max+1]` and doesn't
> verify that 5 cards can actually run together. On AKQ flops, T turn
> can complete TJQKA with a J in hand, but the simplified rule classes
> it as Brick. Accurate straight detection needs full hand-shape
> analysis — slated for the post-Day-3 polish round.

### 2.3 Sort modes

In **By Card** mode, the toolbar's right side has 4 sort buttons:

| Button | Use | Sort key |
|---|---|---|
| **Card** | Default | suit (♣♦♥♠) → rank (2→A) |
| **Best EV** | Find sweetest turns | meanEV desc |
| **Worst EV** | Defense planning | meanEV asc |
| **Aggressive** | Find most-bet turns | sum(Bet/Raise/All-in freq) desc |

Switching to a non-Card sort flips the layout from 13×4 to a 1D wrap so
the suit/rank positions don't lie. Each cell now shows `card icon` + freq
+ EV.

### 2.4 CSV export

The header **CSV** button downloads a `runout_<board>_<history>.csv` file:

```csv
card,acting,All-in_pct,Bet_33_pct,Check_pct,mean_ev,iso_weight
2c,OOP,30.9,34.4,34.6,9.40,3
2s,OOP,72.3,15.8,11.9,1.20,1
3c,OOP,28.1,35.2,36.8,9.21,3
...
```

- `<action>_pct` columns are plain numbers (no `%`) so spreadsheets /
  pandas can do math on them
- `iso_weight` lets you reconstruct true frequency (multiply by weight)
- Filename includes board + history so multi-spot runs don't clash

---

## 3. Combo Drill — break 169 classes into specific combos

**Problem solved**: a 169 hand class like `AKs` covers 4 specific combos
(A♠K♠ / A♥K♥ / A♦K♦ / A♣K♣). When the class shows a mixed strategy
(e.g. 55% bet / 45% check), you have to decide which **specific combo
in your actual hand** takes which line.

The standard poker tie-breaker is the **blocker effect**: how much of the
opponent's range your hole cards remove. Combo Drill ranks the 4/6/12
specific combos by board-aware blocker strength.

### 3.1 How to open

Two paths:

| Path A | Path B |
|---|---|
| Click a class in the main **169 grid** (e.g. `AKs`) | Click the **Combo Drill** button in StrategyPanel |
| → StrategyPanel shows the target analysis | → Modal opens, picks hero's most-played class as default |
| → **Combo Drill** button label becomes `AKs` | |
| → Click it → modal opens with initial class = AKs | |

Both paths land in the same modal.

### 3.2 Class picker

The top of the modal has a search box + 169 class buttons:

```
🔍 [Filter class (e.g. AK, 99, AKs)…]

[22] [33] [44] [55] [66] [77] [88] [99] [TT] [JJ] [QQ] [KK] [AA]
[A2s][A3s][A4s][A5s][A6s][A7s][A8s][A9s][ATs][AJs][AQs][AKs]
[A2o][A3o][A4o][A5o][A6o][A7o][A8o][A9o][ATo][AJo][AQo][AKo]
...
```

- **Highlighted button**: currently selected class
- **Solid background**: class is in opponent's range (weight > 0)
- **Outline only**: opp has 0 weight here (selectable anyway)

Type in the search box to filter (e.g. `AK` → shows `AKs` `AKo`).

### 3.3 Reading the blocker numbers

The grid below shows the class's 4/6/12 specific combos, sorted by
blocker% desc with dead combos at the end. Each combo card:

```
┌──────────────────────────────┐
│  A♥ K♥                  DEAD │  ← Marked DEAD if combo uses
├──────────────────────────────┤   any board card
│  Blocks opp range    8.1%    │
│  ▓▓▓▓░░░░░░░░░░░░░░░░░░     │  ← blocker bar
│                              │
│  Top blocked:                │
│  [AKo ×4] [AJo ×3] [KJo ×3] │
│                              │
│  ▓▓▓▓▓▓░░░░░░░░░░░░░░       │  ← class-shared strategy bar
└──────────────────────────────┘
```

**Blocker% formula**:

```
total_live_opp_combos = sum(opp_range[label] × live_combos_in_label)
                        across all 169 labels (live = after board removal)

blocked_opp_combos = sum(opp_range[label] × combos_using_hero_cards)
                     across all labels

blocker% = blocked_opp_combos / total_live_opp_combos × 100
```

**Top blocked**: opp classes ranked by `blocked_count × range_weight`,
top 5. `AKo ×4` means: of opp's 12 specific AKo combos, 4 of them share
a card with you.

**Color tiers**:

| Color | blocker% | Reading |
|---|---|---|
| 🟢 Green | < 4% | Weak blocker — fine to give up |
| 🟡 Yellow | 4–8% | Neutral |
| 🔴 Red | ≥ 8% | Strong blocker — favor aggressive line |

### Important caveat

The modal has a center caveat strip:

> ⓘ Strategy & EV are class-shared. Blocker analysis is per-combo —
> use it to break mixed-strategy ties.

Meaning: the strategy bar / EV shown for each of the 4 AKs combos is the
**class average**, not per-specific-combo truth. Per-specific-combo
strategy/EV requires engine work (on the roadmap). **Blocker is the only
truly per-combo signal** in this build, and it's the right tool for
choosing among mixed-strategy combos.

---

## 4. Memory Profile — solver resource policy

Choose a memory profile in advanced settings before solving (CLI flag:
`--memory-profile`):

| Profile | Host RAM budget | JSON budget | Strategy tree node cap | Use when |
|---|---|---|---|---|
| `safe` | 2 GB | 50 MB | 500 | Old laptop / testing / running other heavy apps |
| `balanced` | 6 GB | 100 MB | 2,000 | **Default** — general use |
| `performance` | 12 GB | 150 MB | 5,000 | 16+ GB RAM workstation / want deepest strategy_tree |

**Failsafe**: if the estimated footprint exceeds budget, the solver
**won't OOM** — it tries:

1. GPU AUTO fallback (GPU OOM but CPU budget OK → switch to CPU)
2. Byte-based runout cap to trim chance enumeration
3. Truncate strategy_tree (emit up to cap, set `truncated: true`)
4. If all paths fail → structured error, UI badge with reason

**How to see if you got truncated**: a resource badge appears in
StrategyPanel (only when truncate / fallback / non-ok decision):

```
┃ ⚠ Strategy tree truncated to 2000/4321 nodes
┃   (50 MB JSON budget)
┃ ⚠ GPU OOM → fell back to CPU backend
```

---

## 5. FAQ

**Q: Why doesn't the Runout Report button always show?**

A: That board didn't enumerate multiple turn cards. Rainbow flops
(3 different suits + no pair) collapse to a single chance child to bound
memory — there's no per-turn data to aggregate. Try a monotone, two-tone,
or paired flop, or solve the turn spot directly.

---

**Q: Why is the strategy / EV in Combo Drill the same as the 169 grid?**

A: Correct. The current build doesn't yet emit per-specific-combo
strategy / EV (engine work). **Only the blocker number is truly
per-combo** — use it as the decision driver, treat strategy/EV as a
class-level reference. Per-combo emission is on the roadmap.

---

**Q: Why are some combos marked DEAD?**

A: At least one of the combo's two cards is on the board. E.g. with A♠
on the flop, A♠K♠ is DEAD (you can't hold a card that's already public).

---

**Q: What blocker% counts as "strong"?**

A: Rough rules of thumb:

| Range | Reading |
|---|---|
| < 3% | Almost no overlap with opp range |
| 3–6% | Neutral |
| 6–10% | Significant — usually take the aggressive line |
| > 10% | Rare, very strong — typically two-overcard scenarios |

On monotone flops, flush-completing turns will see denser 6–12% values
(hero holding a flush card blocks many opp flush combos).

---

**Q: My CSV opens with garbled characters.**

A: CSV is UTF-8 encoded. Excel defaults to ANSI. Use **Data → From
Text/CSV → pick UTF-8 → Load**. Or open with LibreOffice / Numbers /
pandas — those handle UTF-8 by default.

---

**Q: Memory profile = `performance` and still OOMs.**

A: Likely the spot is genuinely too big (extreme turn full tree + wide
ranges). Read the resource badge reason; use range-locking, fewer bet
sizes, or switch to flop-only solve to shrink the tree.

---

**Q: How do I know the solve converged?**

A: After solving, the top right shows `exploitability_pct`:
- `< 0.5%` — high quality
- `0.5–2%` — usable
- `> 2%` — too few iterations or ranges too wide; add iterations or
  narrow ranges

---

**Q: How long does a solve take?**

A: Depends on the board and backend:

| Board type | CPU | GPU |
|---|---|---|
| Flop, narrow ranges | 1–3s | < 1s |
| Flop, full ranges | 5–15s | 1–5s |
| Turn, full ranges | 30–120s | 5–30s |
| River full tree | not recommended | depends on VRAM |

---

## Appendix: First-install Windows SmartScreen warning

### What you'll see

When you double-click the installer, Windows shows a **full-screen blue
warning**:

```
┌──────────────────────────────────────────────┐
│  Windows protected your PC                   │
│                                              │
│  Microsoft Defender SmartScreen prevented an │
│  unrecognized app from starting. Running this │
│  app might put your PC at risk.              │
│                                              │
│  [More info]                                 │  ← click this (NOT "Don't run")
│                                              │
│              [Don't run]                     │
└──────────────────────────────────────────────┘
```

After clicking **More info**, two more lines + a new button appear:

```
┌──────────────────────────────────────────────┐
│  Windows protected your PC                   │
│  ...(same as above)                          │
│                                              │
│  App: DEEPFOLD-SOLVER_1.x.x_x64-setup.exe    │
│  Publisher: Unknown publisher                 │
│                                              │
│  [Run anyway]              [Don't run]       │  ← click "Run anyway"
└──────────────────────────────────────────────┘
```

Click **Run anyway** → the normal NSIS installer takes over.

### Why this warning appears

Windows SmartScreen warns about every `.exe` that **lacks a code signing
certificate**. It's not "your app is unsafe" — it's "Microsoft doesn't
recognize the publisher" (anyone can call themselves "DEEPFOLD" without
third-party verification).

To remove the warning entirely, we'd need an **EV (Extended Validation)
Code Signing Certificate** ($300–600 USD/year). **We're holding off on
that investment until the user base grows enough to justify the cost** —
in the meantime, please bear with the extra click.

### Safety

- DEEPFOLD-SOLVER's full source is **public on GitHub** for inspection:
  https://github.com/a9876543245/DEEPFOLD-SOLVER
- Every release page shows the installer's hash for verification
- Everything runs **locally** — the only network call is the OAuth
  membership check against deepfold.co
- Lack of warning ≠ safe; presence of warning ≠ unsafe. The right signal
  is **source transparency** + **download provenance**

### Will auto-update trigger the warning again?

**Yes.** Each version's installer has a different hash, and SmartScreen
reputation for unsigned executables is per-file, not per-publisher.
v1.1.0 → v1.1.1 will require another **Run anyway** click.

This is exactly why EV code signing is on the roadmap — once signed,
**all future versions** are trusted automatically.

### Still uneasy?

Build from source (full instructions in the README). The `.exe` you
build will lack publisher info but won't trigger the warning either —
SmartScreen doesn't apply to local builds you run yourself, only to
downloads from the internet.

---

For technical details → [README](README.md) / [中文 Guide](USER_GUIDE.zh.md)
