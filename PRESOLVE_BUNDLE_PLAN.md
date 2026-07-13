# Pre-Solve Bundle Plan

Audience: future me / agent picking this up after context compaction
Project: DEEPFOLD-SOLVER
Owner GPU: user has RTX 5090 (Blackwell, CUDA-13 build path)
Decision date: 2026-05-09
Status: planned, not yet started

---

## 1. Why this exists

The user owns an RTX 5090 and wants to pre-solve a curated set of common
postflop spots once on their machine, then bundle the solved strategies
into the Windows installer so end users get **instant, Deep-quality**
results for those spots — no live solve required for the SpotLibrary
grid or for follow-up runout / turn navigation.

Net win:
- SpotLibrary grid stops showing heuristic placeholders. Real strategies render instantly.
- Click-to-load a spot becomes O(disk read) instead of O(5–15 sec live solve).
- Runout / turn navigation inside a loaded spot stays instant (full strategy_tree is bundled).
- All users see Deep-quality solves (1000 iter @ 0.2% exploit), not the
  current 200 iter @ 1% exploit live solve. **Free quality upgrade for everyone.**

Budget agreed: **up to 500 MB installer size increase** (current installer is
~5 MB pre-bundle, so this is acceptable).

---

## 2. What already exists (do not rewrite)

| File | Role | Notes |
|------|------|-------|
| `gto_output/cash/...` | 852 preflop charts (3.7 MB) | Bundled today via `tauri.conf.json` `resources` |
| `src-tauri/tauri.conf.json` | Tauri bundle config | Already has `"resources": {"../gto_output": "gto_output"}` — bundle pipeline works |
| `src/lib/presolvedSpots.ts` | SpotLibrary spot resolver | `solveSpotReal()` does live solve via `invoke('solve', ...)`; cache is in-memory only, lost on app exit |
| `src/lib/ranges.ts` | `MATCHUPS` array | 27 preflop matchups (SRP / 3BET combos) |
| `src/components/SpotLibrary.tsx` | UI grid | 12 boards × 27 matchups = 324 base spots |
| `src/lib/gameTree.ts` | `BET_SIZINGS` | `lite` (NEW v1.8.2), `standard`, `polar`, `small_ball` |
| `scripts/cash-scenarios/`, `scripts/cash-align/` | Prior bulk solve experiments | Look here for prior art before writing new scripts |
| `core/build_cpu/Release/deepsolver_core.exe` | Solver CLI | Use with `--backend cuda` for 5090 acceleration |

**Key file paths (D:/DEEPFOLD-SOLVER/):**
- `BOARD_TEMPLATES`: `src/lib/presolvedSpots.ts:47`
- `MATCHUPS`: `src/lib/ranges.ts` (search for `defaultPot`)
- `BET_SIZINGS`: `src/lib/gameTree.ts:74`
- `solveSpotReal()`: `src/lib/presolvedSpots.ts:245`

---

## 3. Empirical sizing data (measured 2026-05-09)

### First measurement (narrow ranges, AsKd7c flop, 200 iter, max_tree=2000)
- Raw JSON: 2.6 MB
- gzipped: 521 KB

### Phase 1 actual measurement (2026-05-09 PM, after script written)

Tested via `scripts/bulk-presolve.mjs --limit 2` on actual MATCHUPS data
(UTG vs MP SRP, AsKd2c flop, 1000 iter @ 0.2% target on RTX 5090):

| Sizing | Wall time | Exploit reached | Raw JSON | Gzipped |
|--------|-----------|-----------------|----------|---------|
| Standard | **53s** | 5.39% (didn't hit 0.2% target) | **6.9 MB** | **1.5 MB** |
| Lite | **6.9s** | 3.87% | **1.2 MB** | **248 KB** |

**Key finding**: strategy_tree is **99.7%** of raw bytes (8.0 MB out of 8.02 MB
in the Standard sample). All other fields combined (timing, resources, global_strategy,
combo_strategies, opponent_range) are ~28 KB. **Compaction by stripping fluff
saves <1%.** Only effective lever: cap `--strategy-tree-max-nodes` lower.

### Bundle size projection (revised)

| Slice | Spots | Time on 5090 | Compressed bundle |
|-------|-------|--------------|-------------------|
| 312 × Standard × default stack | 312 | ~4.6 hours | **~470 MB** |
| 312 × Lite × default stack | 312 | ~36 min | **~78 MB** |
| Combined (default stack only) | 624 | **~5.2 hours** | **~548 MB** |

**~50 MB over 500 MB target.** Mitigations (pick one):
1. Drop `--strategy-tree-max-nodes` 2000 → 1500 ≈ 25% size cut → ~410 MB ✓
2. Drop Lite's strategy_tree (Tier A for Lite, Tier B for Standard) → 312 × 1.5 MB + 312 × ~10 KB = ~471 MB ✓
3. Skip the SRP wide-range matchups for Standard (worst offenders for tree size)
4. Bundle Standard ONLY (drop Lite entirely) → 470 MB ✓

**Recommended path**: Option 1 (lower max_tree_nodes to 1500). Loss in
navigability is minor — most users won't hit the 1500-node ceiling on a
flop spot, and the rare cases that would are exactly the spots that benefit
most from re-solving on the user's machine anyway. Re-run bulk-presolve with
`--max-tree-nodes 1500` after agreement.

---

## 4. Scope — what to bundle

**Bundle target: ~400–450 MB total (within 500 MB cap)**

| Slice | Spots | Size estimate |
|-------|-------|---------------|
| 324 spots × Standard sizing × 100bb | 324 | ~165 MB |
| 324 spots × Lite sizing × 100bb | 324 | ~165 MB |
| Top-100 spots × Standard sizing × 200bb deep | 100 | ~50 MB |
| Top-50 spots × Standard sizing × 50bb short | 50 | ~25 MB |
| **Total** | **798 solves** | **~405 MB** |

**Not bundled (live-solve fallback):**
- Polar / Small Ball sizings (uncommon)
- Custom user ranges (any config that doesn't match the bundled key)
- Stack depths outside {50, 100, 200} bb on the cold spots

**"Top-100 hot spots" definition** (no telemetry yet; pick by heuristic):
- All BTN-vs-blinds matchups across all 12 boards (BTN open / 3BET pot) ≈ 48 spots
- All blind-vs-blind ≈ 24 spots
- All button-vs-CO/MP across all boards ≈ 72 spots
- Pick top 100 by overlapping rank

If we get telemetry later (Phase E.D in original plan), refine the hot list.

**Quality**: 1000 iter @ 0.2% exploit target = "Deep" mode.

---

## 5. 5090 compute budget

| Slice | Solves | Time @ ~30 sec/spot on 5090 |
|-------|--------|------------------------------|
| 324 × Standard × 100bb | 324 | ~2.7 hours |
| 324 × Lite × 100bb | 324 | ~2.7 hours |
| 100 × Standard × 200bb | 100 | ~50 min |
| 50 × Standard × 50bb | 50 | ~25 min |
| **Total** | **798** | **~7 hours** |

User runs once overnight per release.

---

## 6. Implementation phases

### Phase 1 — Bulk solve script (DONE, 2026-05-09)

**Status**: ✓ Implemented as `scripts/bulk-presolve.mjs` (Node ESM, not PowerShell — easier
range-string passing, cross-platform). Smoke-tested on 2 spots end-to-end on the
5090 (CC 12.0 / Blackwell). Output validated, sizes measured (see §3).

**Auto-detects** the CUDA build at `D:/DEEPFOLD-SOLVER/core/build/Release/deepsolver_core.exe`
(falls back to CPU build at `core/build_cpu/Release/` if missing). Reads MATCHUPS /
BOARD_TEMPLATES / BET_SIZINGS via regex+`new Function()` from the actual TS sources
so there's no manual sync.

**Usage**:
```
node scripts/bulk-presolve.mjs --dry-run              # show plan
node scripts/bulk-presolve.mjs --limit 5 --iterations 100  # smoke
node scripts/bulk-presolve.mjs                        # full run (Deep, ~5-7 hours)
node scripts/bulk-presolve.mjs --resume               # skip existing files
node scripts/bulk-presolve.mjs --max-tree-nodes 1500  # smaller bundle
```

**Defaults**: `--iterations 1000 --exploitability 0.2 --max-tree-nodes 2000 --sizings standard,lite`.

Output: `gto_output/presolved/raw/m<i>_b<j>_<sizing>_<stack>bb.json` (raw, NOT yet
gzipped/compacted — that's Phase 2).

**Original spec (kept for reference)**: `scripts/bulk-presolve.ps1`

```powershell
# Inputs:
#   - $manifest = list of spots to solve (matchup_idx × board_idx × sizing × stack)
#     Source: derive from src/lib/ranges.ts (MATCHUPS) + src/lib/presolvedSpots.ts (BOARD_TEMPLATES)
#     + src/lib/gameTree.ts (BET_SIZINGS)
#   - $core_exe = path to deepsolver_core.exe (Release build)
#   - $out_dir = gto_output/presolved/
#
# Loop:
#   for each spot in $manifest:
#     $args = construct CLI args for this spot (--board, --pot, --stack,
#             --ip-range, --oop-range, --flop-sizes, --turn-sizes, --river-sizes,
#             --iterations 1000, --exploitability 0.2, --backend cuda,
#             --postsolve full, --strategy-tree-evs visible)
#     $output_path = "$out_dir/$(spot_id_for $spot).json"
#     & $core_exe @args > $output_path
#     gzip $output_path
#     log progress (current/total, elapsed, ETA)
#
# Output:
#   - One .json.gz per (matchup, board, sizing, stack) tuple
#   - run-summary.json with total time, exploit per spot, any failures
```

Spot ID scheme: `m<matchup_idx>_b<board_idx>_<sizing>_<stack>bb.json.gz`
Example: `m12_b3_standard_100bb.json.gz`

**Implementation notes**:
- Look at existing `scripts/import-pio-ranges.mjs` for prior art on iterating MATCHUPS
- Range strings: `MATCHUPS[i].ipRange` and `oopRange` are in range-string format — pass directly via `--ip-range` / `--oop-range`
- `defaultPot` and `defaultStack` in MATCHUPS are in BB; multiply by 10 for chip units (matches existing `solveSpotReal` line 259-260)
- Use `--backend cuda` for the GPU; verify availability via `--gpu-info` first
- Add `--postsolve full --strategy-tree-evs visible` so the bundled JSON has everything the UI needs
- Set `--strategy-tree-max-nodes 2000` (current default) — keeps file size predictable
- Resume support: skip files that already exist + match the manifest's expected solver_version
- Failure handling: log to `scripts/bulk-presolve-failures.log`, continue with next spot

### Phase 2 — Compaction script (0.5 day)

**Deliverable**: `scripts/compact-presolved.mjs`

Raw solve JSON has lots of fluff (timing, resources, debug info) that the UI doesn't need. Strip down to:

```typescript
interface BundledSpot {
  v: number;                  // schema version (start at 1)
  solver_version: string;     // from app version, e.g. "1.8.2"
  matchup_idx: number;
  board_idx: number;
  sizing_key: string;         // "lite" | "standard" | ...
  stack_bb: number;
  exploitability_pct: number; // for badge display
  iterations: number;         // for "Deep quality" badge
  global_strategy: Record<string, string>;
  combo_strategies: Record<string, ComboStrategy>;
  strategy_tree?: Record<string, StrategyTreeEntry>;  // FULL tree for navigation
  // Drop: timing, resources (most), runtime metadata, etc.
}
```

Should bring 2.6 MB raw → ~1.2 MB compact → ~400 KB gzipped.

If compact size still pushes total over 500 MB, drop `strategy_tree` from cold spots (keep only for top-100 hot spots).

### Phase 3 — Frontend lookup (1 day)

**File to edit**: `src/lib/presolvedSpots.ts`

Modify `solveSpotReal()` (currently at line 245):

```typescript
async function solveSpotReal(
  matchupIdx: number,
  boardIdx: number,
  sizingKey: string = 'standard',
  stackBb: number = 100,
): Promise<PresolvedSpot> {
  if (!isTauri()) {
    throw new Error('Real solver not available in browser mode');
  }

  // NEW v1.8.2: try bundled first
  const bundled = await tryLoadBundled(matchupIdx, boardIdx, sizingKey, stackBb);
  if (bundled) return bundled;

  // Existing live solve fallback unchanged
  // ... existing code ...
}

async function tryLoadBundled(
  matchupIdx: number,
  boardIdx: number,
  sizingKey: string,
  stackBb: number,
): Promise<PresolvedSpot | null> {
  const filename = `m${matchupIdx}_b${boardIdx}_${sizingKey}_${stackBb}bb.json.gz`;
  const resourcePath = `gto_output/presolved/${filename}`;
  try {
    const { readFile } = await import('@tauri-apps/plugin-fs');
    const { resolveResource } = await import('@tauri-apps/api/path');
    const fullPath = await resolveResource(resourcePath);
    const compressed = await readFile(fullPath);
    const decompressed = await gunzipBuffer(compressed);  // need a small util
    const bundled: BundledSpot = JSON.parse(decompressed);

    // Version check — invalidate on solver bump
    if (bundled.solver_version !== CURRENT_SOLVER_VERSION) return null;

    return convertBundledToSpot(bundled);
  } catch (e) {
    return null;  // missing file → fall through to live solve
  }
}
```

**Caller update**: `solveSpotReal()` callers need to pass `sizingKey` and `stackBb`. Currently (line 333) passes only `(matchupIdx, boardIdx)`. Find all callers via grep and add the new args. Default to `('standard', 100)` for backward-compat.

**UI badge**: in `SpotLibrary.tsx`, add a small indicator next to each spot:
- 🟢 `Pre-solved · Deep` if bundled
- ⏳ `Live-solving...` while in flight
- 🟡 `Live-solved · Standard` if no bundle hit but real solve completed

### Phase 4 — Bundle + release pipeline (0.5 day)

**Files to touch**:
- `.gitattributes`: add `gto_output/presolved/*.gz binary` (avoid line-ending diffs)
- `scripts/prepare-release.ps1`: add a `verify_presolved_bundle` step:
  - For each `.json.gz` in `gto_output/presolved/`, decompress, parse, check `solver_version` matches the version we're about to release
  - If any mismatch, abort with: "Re-run bulk-presolve before release: $count files have stale solver_version"

**Tauri config**: no change needed. `gto_output/` is already bundled wholesale.

**Git LFS evaluation**: 400 MB across ~800 files would inflate `.git` history fast. Two options:
1. **Git LFS**: `gto_output/presolved/*.gz` tracked by LFS. Keeps repo small but requires LFS on clone.
2. **Release artifact only**: don't commit `gto_output/presolved/` to repo; CI/release pipeline pulls it from a release artifact (S3 / GitHub release). Cleanest but requires release infra.

**Recommendation**: start with raw-commit (no LFS), see if repo size becomes a problem. If `.git` exceeds ~5 GB after a few releases, switch to LFS or artifact.

### Phase 5 — Quality / coverage UX (0.5 day)

**Files to touch**:
- `src/components/SpotLibrary.tsx`: add a header row showing "324 spots × 2 sizings pre-solved at Deep quality on RTX 5090"
- `src/components/SettingsPanel.tsx` (if exists, else add to Advanced): show cache stats (size, hit rate this session, last bundle update timestamp)
- `i18n.tsx`: add translation keys for the new UI strings

---

## 7. Validation criteria

Before declaring this done:

1. **Bundled lookup works**:
   ```
   - Open SpotLibrary
   - Pick a bundled spot (e.g., BTN vs BB SRP, AsKd2c)
   - Strategy displays instantly (<200 ms)
   - Console shows "loaded bundled spot" log
   ```

2. **Live fallback works**:
   ```
   - Pick an unbundled config (e.g., Polar sizing on a board)
   - Falls back to live solve, shows "Live-solving..." spinner
   - Result eventually displays with "Live-solved" badge
   ```

3. **Version invalidation works**:
   ```
   - Hand-edit one bundled JSON's solver_version to "0.0.0"
   - Reload SpotLibrary
   - That spot falls through to live solve
   ```

4. **Quality is real**:
   ```
   - Pick a bundled spot
   - exploitability_pct field shows ≤ 0.2 (Deep target)
   - Compare global_strategy to a live-solve at same config — should match within numerical noise
   ```

5. **Installer size**:
   ```
   - prepare-release.ps1 builds installer
   - Final .exe size: 405 MB ± 50 MB
   - First-install + first-launch works (Tauri can read bundled resources)
   ```

6. **Smoke / parity gates still green**:
   ```
   - ctest -L smoke -j 4 — under 10s, all pass
   - ctest -L parity — all pass
   - This bundle work is frontend + script only, shouldn't affect engine
   ```

---

## 8. Open decisions (resolve before / during Phase 1)

- **D1: spot_id scheme** — `m<i>_b<j>_<sizing>_<stack>bb` vs hash of full config. Hash is robust to MATCHUPS reordering but unreadable. Recommend: human-readable for now, switch to hash if MATCHUPS churns.
- **D2: hot spot selection** — without telemetry, pick by gut. Document the choice. Plan to revisit after 1-2 releases.
- **D3: 50bb / 200bb scope** — could skip entirely (only ship 100bb) for v1 to keep budget under 350 MB. Decide based on actual measured sizes after Phase 1.
- **D4: Lite vs Standard sizing priority** — both worth bundling. If forced to pick one, Standard (default).
- **D5: max_strategy_tree_nodes** — current default 2000. Higher = more navigable but bigger files. Lower = smaller bundle but limited navigation. 2000 seems right; revisit if user feedback says navigation hits limits.

---

## 9. Run commands (when ready to execute)

```powershell
# Phase 1: bulk solve (run on 5090 machine, ~7 hours)
pwsh D:\DEEPFOLD-SOLVER\scripts\bulk-presolve.ps1 `
  -OutDir D:\DEEPFOLD-SOLVER\gto_output\presolved `
  -CoreExe D:\DEEPFOLD-SOLVER\core\build_cpu\Release\deepsolver_core.exe `
  -SizingKeys @('standard','lite') `
  -StacksBb @(100) `
  -DeepStacksHot @(50, 200) `
  -Verbose

# Phase 2: compact + gzip outputs
node D:\DEEPFOLD-SOLVER\scripts\compact-presolved.mjs `
  --in D:\DEEPFOLD-SOLVER\gto_output\presolved\raw `
  --out D:\DEEPFOLD-SOLVER\gto_output\presolved

# Phase 4: build installer (uses prepare-release.ps1's existing flow)
pwsh D:\DEEPFOLD-SOLVER\scripts\prepare-release.ps1
```

---

## 10. Where this fits in the larger plan

This is on top of the recently-shipped (in dev, not yet released) work:

- **A2 encoding** (matchup_category_u8 + valid_f32): +12-21% iter throughput on monotone, neutral-to-+8% on standard. Bit-identical parity.
- **Phase counters** (showdown vs fold split): diagnostic infrastructure for future profiling.
- **Lite sizing preset**: 4× faster solve on standard fixture (1383 → 315 nodes). Frontend-only change.
- **POST_OPTIMIZATION_REVIEW hygiene** (4 findings): standard benchmark ~80% throughput recovery, smoke loop <10s, paired-bench warm-both, invalid-case exit code 1.

Pre-solve bundle is the **next logical user-visible win** — pure UX upgrade for SpotLibrary, ships at the same v1.8.2 release as the engine improvements above.

After this lands, candidates for the next thing:
- **SPR-canonical abstraction** (~3 weeks, big payoff): cache lookup by SPR rather than exact (pot, stack), so any (pot, stack) tuple matching a bundled SPR hits cache. Replaces "stack_bb" dimension in the cache key with continuous SPR.
- **Persistent in-session cache**: write user's live solves to disk too, not just bundled ones. So second-time queries hit even for non-bundled configs.
- **Telemetry-driven hot-spot list**: see what users actually query, refine the bundled set to maximize hit rate.

---

## 11. Resumption checklist (for the agent picking this up)

1. Read this entire file
2. Verify the file paths in §2 still exist (codebase may have moved)
3. Run Phase 1 dry-run on 1-2 spots manually to confirm the CLI args work
4. Check `scripts/cash-scenarios/` for any prior bulk-solve patterns to reuse
5. Estimate actual file size from a single Phase 1 + Phase 2 round-trip before committing to the 798-spot run
6. Get user confirmation before kicking off the 7-hour 5090 run
7. Execute phases in order, validate after each
