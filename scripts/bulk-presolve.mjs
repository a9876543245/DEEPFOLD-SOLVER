#!/usr/bin/env node
/**
 * @file scripts/bulk-presolve.mjs
 * @brief Phase 1 of PRESOLVE_BUNDLE_PLAN.md — bulk-solve common spots on
 *        the operator's GPU and stage them for installer bundling.
 *
 * Usage:
 *   node scripts/bulk-presolve.mjs                    # full run, default sizings
 *   node scripts/bulk-presolve.mjs --dry-run          # print plan, don't solve
 *   node scripts/bulk-presolve.mjs --sizings standard # subset
 *   node scripts/bulk-presolve.mjs --limit 5          # solve first 5 spots only (smoke)
 *   node scripts/bulk-presolve.mjs --resume           # skip files that already exist
 *
 * Output:
 *   gto_output/presolved/raw/m<i>_b<j>_<sizing>_<stack>bb.json
 *
 * Phase 2 (compact-presolved.mjs) compresses + strips fluff from these.
 *
 * Reads the actual TS source files for MATCHUPS / BOARD_TEMPLATES / BET_SIZINGS
 * via regex extraction so we don't need a TypeScript loader. The arrays are
 * valid JS object literals — only the type annotations are TS-specific, and
 * those live OUTSIDE the array body (as `: PositionMatchup[]` after the `=`).
 */

import { spawn } from 'node:child_process';
import {
  existsSync, mkdirSync, readFileSync, writeFileSync, statSync,
} from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// Paths
// ============================================================================

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const RANGES_TS    = join(REPO_ROOT, 'src/lib/ranges.ts');
const SPOTS_TS     = join(REPO_ROOT, 'src/lib/presolvedSpots.ts');
const SIZINGS_TS   = join(REPO_ROOT, 'src/lib/gameTree.ts');

// CUDA build with 5090 support — see PRESOLVE_BUNDLE_PLAN §2.
// Falls back to the CPU-only build if CUDA isn't built.
const CUDA_EXE = resolve(REPO_ROOT, '..', 'DEEPFOLD-SOLVER', 'core', 'build', 'Release', 'deepsolver_core.exe');
const CPU_EXE  = join(REPO_ROOT, 'core/build_cpu/Release/deepsolver_core.exe');

const OUT_RAW_DIR = join(REPO_ROOT, 'gto_output/presolved/raw');

// ============================================================================
// Args
// ============================================================================

const args = parseArgs(process.argv.slice(2));

function parseArgs(argv) {
  const out = {
    dryRun: false,
    resume: false,
    sizings: ['standard', 'lite'],
    stacksBb: ['default'],   // 'default' = use the matchup's defaultStack
    limit: 0,
    backend: 'auto',         // 'cuda' | 'cpu' | 'auto'
    // v1.8.3+ tiered iter caps. SRP wide ranges need more iter to converge
    // (1000 iter only reaches ~5-7% exploit; need 3000+ for ~1.5%). Narrow
    // 3-bet pots converge faster (1500 iter usually <1%). Lite sizing tree
    // is small so 1000 iter is enough. `--iterations N` overrides all tiers
    // with a single value (back-compat).
    iterations: 0,            // 0 = use tiered defaults below
    iterSrpStandard: 3000,    // SRP × Standard sizing — wide tree, needs the most
    iterThreeBetStandard: 1500,  // 3-bet × Standard — narrower range, faster convergence
    iterAnyLite: 1000,        // Any × Lite sizing — small tree, fast
    iterDefault: 1000,        // safety fallback for any tuple we didn't classify
    exploitability: 0.2,
    maxTreeNodes: 2000,
  };
  for (let i = 0; i < argv.length; ++i) {
    const a = argv[i];
    if (a === '--dry-run')        out.dryRun = true;
    else if (a === '--resume')    out.resume = true;
    else if (a === '--sizings')   out.sizings = argv[++i].split(',');
    else if (a === '--stacks')    out.stacksBb = argv[++i].split(',');
    else if (a === '--limit')     out.limit = parseInt(argv[++i], 10);
    else if (a === '--backend')   out.backend = argv[++i];
    else if (a === '--iterations') out.iterations = parseInt(argv[++i], 10);
    else if (a === '--exploitability') out.exploitability = parseFloat(argv[++i]);
    else if (a === '--max-tree-nodes') out.maxTreeNodes = parseInt(argv[++i], 10);
    else throw new Error(`unknown arg: ${a}`);
  }
  return out;
}

/** Pick iter cap based on (potType, sizingKey) — see parseArgs comments for
 *  the rationale. `--iterations N` overrides via args.iterations. */
function iterFor(spot) {
  if (args.iterations > 0) return args.iterations;
  const isLite = spot.sizingKey === 'lite';
  const isSrp  = spot.meta.pot_type === 'SRP';
  if (isLite)             return args.iterAnyLite;
  if (isSrp)              return args.iterSrpStandard;
  if (spot.meta.pot_type === '3BET') return args.iterThreeBetStandard;
  return args.iterDefault;
}

// ============================================================================
// Source extraction
// ============================================================================

/**
 * Pull a top-level `export const NAME ... = [ ... ];` block out of a TS file
 * and `eval` the array body as JS. Works because object/array literals are a
 * subset of TS that doesn't need type stripping.
 */
function extractArray(srcPath, name) {
  const src = readFileSync(srcPath, 'utf-8');
  // Match the array body up to the closing `];` at column 0 of a line.
  const re = new RegExp(
    `export const ${name}[^=]*=\\s*(\\[[\\s\\S]*?\\n\\]);`, 'm');
  const m = src.match(re);
  if (!m) throw new Error(`Could not extract ${name} from ${srcPath}`);
  // new Function returns the evaluated array literal.
  return new Function(`return ${m[1]}`)();
}

/** Same but for a Record<string, ...> object literal (BET_SIZINGS). */
function extractRecord(srcPath, name) {
  const src = readFileSync(srcPath, 'utf-8');
  const re = new RegExp(
    `export const ${name}[^=]*=\\s*(\\{[\\s\\S]*?\\n\\});`, 'm');
  const m = src.match(re);
  if (!m) throw new Error(`Could not extract ${name} from ${srcPath}`);
  return new Function(`return ${m[1]}`)();
}

const MATCHUPS        = extractArray(RANGES_TS, 'MATCHUPS');
const BOARD_TEMPLATES = extractArray(SPOTS_TS,  'BOARD_TEMPLATES');
const BET_SIZINGS     = extractRecord(SIZINGS_TS, 'BET_SIZINGS');

console.log(`Loaded ${MATCHUPS.length} matchups, ${BOARD_TEMPLATES.length} boards, ${Object.keys(BET_SIZINGS).length} sizings`);

// ============================================================================
// Plan generation
// ============================================================================

/**
 * One spot to solve. Filename is the cache key; mirrors §6.4 "Phase 1 spot ID
 * scheme" of PRESOLVE_BUNDLE_PLAN.md.
 */
function buildPlan() {
  const plan = [];
  for (let mi = 0; mi < MATCHUPS.length; ++mi) {
    const m = MATCHUPS[mi];
    for (let bi = 0; bi < BOARD_TEMPLATES.length; ++bi) {
      const board = BOARD_TEMPLATES[bi];
      for (const sizingKey of args.sizings) {
        if (!BET_SIZINGS[sizingKey]) {
          throw new Error(`unknown sizing key: ${sizingKey}`);
        }
        // For now: 'default' = use the matchup's own defaultStack.
        // Future: explicit '50' / '200' bb depths require adjusting pot too,
        // which means recomputing the SRP / 3BET pre-flop pot ratio. Out of
        // scope for v1 of the bundle.
        for (const stackBb of args.stacksBb) {
          const effStackBb = (stackBb === 'default') ? m.defaultStack : Number(stackBb);
          const potBb      = m.defaultPot;   // SRP/3BET pot stays as MATCHUPS encodes it
          plan.push({
            matchupIdx: mi,
            boardIdx:   bi,
            sizingKey,
            stackLabel: stackBb,             // 'default' or '50' / '200' string for filename
            // CLI args (chip units = BB × 10, matches solveSpotReal in presolvedSpots.ts:259-260)
            board:    board.board,
            pot:      Math.round(potBb * 10),
            stack:    Math.round(effStackBb * 10),
            ipRange:  m.ipRange,
            oopRange: m.oopRange,
            sizes:    BET_SIZINGS[sizingKey],
            // Metadata for the output filename + bundled JSON
            meta: {
              matchup_label: m.label,
              pot_type:      m.potType,
              ip_pos:        m.ip,
              oop_pos:       m.oop,
              board_texture: board.texture,
              effective_bb:  effStackBb,
              pot_bb:        potBb,
            },
          });
        }
      }
    }
  }
  if (args.limit > 0 && plan.length > args.limit) {
    return plan.slice(0, args.limit);
  }
  return plan;
}

function spotFilename(spot) {
  // m<matchup_idx>_b<board_idx>_<sizing>_<stack>bb.json
  return `m${spot.matchupIdx}_b${spot.boardIdx}_${spot.sizingKey}_${spot.stackLabel}bb.json`;
}

// ============================================================================
// Solver invocation
// ============================================================================

function pickExe() {
  if (args.backend === 'cpu')  return CPU_EXE;
  if (args.backend === 'cuda') return CUDA_EXE;
  // auto: prefer cuda build if it exists
  return existsSync(CUDA_EXE) ? CUDA_EXE : CPU_EXE;
}

function solveOne(exe, spot) {
  return new Promise((resolveFn, reject) => {
    const sizes = spot.sizes;
    const cmdArgs = [
      '--pot',           String(spot.pot),
      '--stack',         String(spot.stack),
      '--board',         spot.board,
      '--iterations',    String(iterFor(spot)),
      '--exploitability', String(args.exploitability),
      '--ip-range',      spot.ipRange,
      '--oop-range',     spot.oopRange,
      '--flop-sizes',    sizes.flopBetSizes.join(','),
      '--turn-sizes',    sizes.turnBetSizes.join(','),
      '--river-sizes',   sizes.riverBetSizes.join(','),
      '--postsolve',     'full',
      '--strategy-tree-evs', 'visible',
      '--strategy-tree-max-nodes', String(args.maxTreeNodes),
      '--backend',       (exe === CUDA_EXE) ? 'cuda' : 'cpu',
    ];
    const child = spawn(exe, cmdArgs, { windowsHide: true });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`solver exit ${code}: ${stderr.slice(-500)}`));
      }
      try {
        const parsed = JSON.parse(stdout);
        if (parsed.status !== 'success') {
          return reject(new Error(`solver returned non-success: ${parsed.message || 'unknown'}`));
        }
        resolveFn({ stdout, parsed });
      } catch (e) {
        reject(new Error(`solver output not JSON: ${e.message} | first 200 chars: ${stdout.slice(0, 200)}`));
      }
    });
  });
}

// ============================================================================
// Driver
// ============================================================================

async function main() {
  const exe = pickExe();
  if (!existsSync(exe)) {
    throw new Error(`solver exe not found: ${exe}\nBuild with: cmake --build ../DEEPFOLD-SOLVER/core/build --config Release`);
  }
  console.log(`Solver:  ${exe}`);
  console.log(`Output:  ${OUT_RAW_DIR}`);
  console.log(`Backend: ${args.backend}`);
  if (args.iterations > 0) {
    console.log(`Plan:    ${args.iterations} iter (override) × ${args.exploitability}% exploit target`);
  } else {
    console.log(`Plan (tiered):`);
    console.log(`  SRP × Standard sizing  → ${args.iterSrpStandard} iter`);
    console.log(`  3BET × Standard sizing → ${args.iterThreeBetStandard} iter`);
    console.log(`  Any × Lite sizing      → ${args.iterAnyLite} iter`);
    console.log(`  exploit target          ${args.exploitability}%`);
  }

  if (args.dryRun) {
    const plan = buildPlan();
    console.log(`\n=== DRY RUN — ${plan.length} spots planned ===`);
    for (const spot of plan.slice(0, 10)) {
      console.log(`  ${spotFilename(spot)}  [${spot.meta.matchup_label} ${spot.meta.pot_type} on ${spot.board}]`);
    }
    if (plan.length > 10) console.log(`  ... and ${plan.length - 10} more`);
    return;
  }

  mkdirSync(OUT_RAW_DIR, { recursive: true });
  const plan = buildPlan();
  console.log(`\n=== ${plan.length} spots to solve ===\n`);

  const t0 = Date.now();
  let done = 0;
  let skipped = 0;
  let failed = 0;
  const failures = [];

  for (const spot of plan) {
    const filename = spotFilename(spot);
    const outPath  = join(OUT_RAW_DIR, filename);
    if (args.resume && existsSync(outPath) && statSync(outPath).size > 0) {
      ++skipped;
      ++done;
      continue;
    }
    const tStart = Date.now();
    try {
      const { stdout, parsed } = await solveOne(exe, spot);
      // Write raw output. Phase 2 (compact-presolved.mjs) handles compression.
      writeFileSync(outPath, stdout);
      const elapsed = (Date.now() - tStart) / 1000;
      const eta = computeEta(t0, ++done, plan.length);
      const exploit = parsed.exploitability_pct?.toFixed(2) ?? '?';
      console.log(`[${done}/${plan.length}] ${filename}  exploit=${exploit}%  ${elapsed.toFixed(1)}s  ETA ${eta}`);
    } catch (e) {
      ++failed;
      failures.push({ spot: filename, error: e.message });
      console.error(`[${++done}/${plan.length}] FAIL ${filename}: ${e.message}`);
    }
  }

  const total = (Date.now() - t0) / 1000;
  console.log(`\n=== Done ===`);
  console.log(`Total time:  ${(total / 60).toFixed(1)} min`);
  console.log(`Solved:      ${done - failed - skipped}`);
  console.log(`Skipped:     ${skipped}`);
  console.log(`Failed:      ${failed}`);

  if (failures.length > 0) {
    const failPath = join(OUT_RAW_DIR, 'failures.log');
    writeFileSync(failPath, JSON.stringify(failures, null, 2));
    console.log(`Failures logged: ${failPath}`);
  }
}

function computeEta(t0, done, total) {
  if (done === 0) return '?';
  const elapsed = (Date.now() - t0) / 1000;
  const perSpot = elapsed / done;
  const remaining = perSpot * (total - done);
  if (remaining < 60) return `${Math.round(remaining)}s`;
  if (remaining < 3600) return `${(remaining / 60).toFixed(1)} min`;
  return `${(remaining / 3600).toFixed(1)} h`;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
