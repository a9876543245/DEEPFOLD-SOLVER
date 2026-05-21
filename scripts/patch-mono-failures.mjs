#!/usr/bin/env node
/**
 * @file scripts/patch-mono-failures.mjs
 * @brief Re-run the SRP × Standard × Monotone spots that OOM'd on GPU during
 *        bulk-presolve.mjs by switching to GPU backend with SINGLE flop sizing.
 *
 * Why this exists:
 *   The wide SRP ranges combined with monotone flops (Ah9h4h, Kd8d3d) +
 *   2-flop-size Standard tree blow up to ~4.6M nodes / ~165 GB GPU memory,
 *   which a 32 GB 5090 can't fit. CPU backend at 3000 iter would take ~4 hr
 *   per spot (~88 hr total) — too slow for v1.8.3 ship.
 *
 *   Compromise: drop the small 0.33 flop sizing for THESE 22 spots, keeping
 *   only the 0.75 flop sizing. Tree shrinks from ~4.6M to ~1M nodes and fits
 *   GPU comfortably. Each spot then takes ~150s on 5090 (matches bulk).
 *
 *   UX impact: these 22 monotone SRP spots show "Bet 75% / Check / Fold"
 *   instead of "Bet 33% / Bet 75% / Check / Fold". Acceptable — on monotone
 *   boards, GTO almost never uses small bets anyway (range advantage is
 *   reduced when both players hit the suit).
 *
 * Usage:
 *   node scripts/patch-mono-failures.mjs            # patch all OOM'd spots
 *   node scripts/patch-mono-failures.mjs --dry-run  # show plan only
 *
 * Detection:
 *   Reads gto_output/presolved/raw/, identifies expected SRP×Std×Mono spots
 *   by filename pattern (m{srpIdx}_b{6,7}_standard_defaultbb.json), and
 *   re-runs any that are missing or contain a {"status":"error"} body.
 */

import { spawn } from 'node:child_process';
import {
  existsSync, readFileSync, writeFileSync, statSync,
} from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const RANGES_TS = join(REPO_ROOT, 'src/lib/ranges.ts');
const SPOTS_TS  = join(REPO_ROOT, 'src/lib/presolvedSpots.ts');
const SIZINGS_TS = join(REPO_ROOT, 'src/lib/gameTree.ts');

const CUDA_EXE = resolve(REPO_ROOT, '..', 'DEEPFOLD-SOLVER', 'core', 'build', 'Release', 'deepsolver_core.exe');
const FALLBACK_CUDA_EXE = join(REPO_ROOT, 'core/build/Release/deepsolver_core.exe');

const OUT_RAW_DIR = join(REPO_ROOT, 'gto_output/presolved/raw');

const args = parseArgs(process.argv.slice(2));

function parseArgs(argv) {
  const out = {
    dryRun: false,
    iterations: 3000,
    exploitability: 0.2,
    dcfrSchedule: 'postflop',
    maxTreeNodes: 2000,
  };
  for (let i = 0; i < argv.length; ++i) {
    const a = argv[i];
    if (a === '--dry-run')          out.dryRun = true;
    else if (a === '--iterations')  out.iterations = parseInt(argv[++i], 10);
    else if (a === '--exploitability') out.exploitability = parseFloat(argv[++i]);
    else if (a === '--dcfr-schedule') out.dcfrSchedule = argv[++i];
    else throw new Error(`unknown arg: ${a}`);
  }
  return out;
}

function extractArray(srcPath, name) {
  const src = readFileSync(srcPath, 'utf-8');
  const re = new RegExp(`export const ${name}[^=]*=\\s*(\\[[\\s\\S]*?\\n\\]);`, 'm');
  const m = src.match(re);
  if (!m) throw new Error(`Could not extract ${name} from ${srcPath}`);
  return new Function(`return ${m[1]}`)();
}

function extractRecord(srcPath, name) {
  const src = readFileSync(srcPath, 'utf-8');
  const re = new RegExp(`export const ${name}[^=]*=\\s*(\\{[\\s\\S]*?\\n\\});`, 'm');
  const m = src.match(re);
  if (!m) throw new Error(`Could not extract ${name} from ${srcPath}`);
  return new Function(`return ${m[1]}`)();
}

const MATCHUPS        = extractArray(RANGES_TS, 'MATCHUPS');
const BOARD_TEMPLATES = extractArray(SPOTS_TS,  'BOARD_TEMPLATES');
const BET_SIZINGS     = extractRecord(SIZINGS_TS, 'BET_SIZINGS');

// Identify all (matchupIdx, boardIdx) tuples that need patching:
//   - SRP pot type
//   - Standard sizing
//   - Monotone board (boardIdx 6 or 7 in current BOARD_TEMPLATES)
function findFailedSpots() {
  const monoIndices = [];
  for (let bi = 0; bi < BOARD_TEMPLATES.length; ++bi) {
    if (BOARD_TEMPLATES[bi].texture === 'Monotone') monoIndices.push(bi);
  }
  const failures = [];
  for (let mi = 0; mi < MATCHUPS.length; ++mi) {
    if (MATCHUPS[mi].potType !== 'SRP') continue;
    for (const bi of monoIndices) {
      const filename = `m${mi}_b${bi}_standard_defaultbb.json`;
      const outPath = join(OUT_RAW_DIR, filename);
      let needsPatch = false;
      if (!existsSync(outPath) || statSync(outPath).size === 0) {
        needsPatch = true;
      } else {
        // File exists — check if it's an error body (the bulk script writes
        // stdout even for failures? actually it doesn't — it only writes on
        // success. So missing file = need patch).
        try {
          const body = readFileSync(outPath, 'utf-8');
          const parsed = JSON.parse(body);
          if (parsed.status !== 'success') needsPatch = true;
        } catch (_) {
          needsPatch = true;
        }
      }
      if (needsPatch) {
        failures.push({
          matchupIdx: mi,
          boardIdx:   bi,
          filename,
          outPath,
          spot: {
            board:    BOARD_TEMPLATES[bi].board,
            pot:      Math.round(MATCHUPS[mi].defaultPot * 10),
            stack:    Math.round(MATCHUPS[mi].defaultStack * 10),
            ipRange:  MATCHUPS[mi].ipRange,
            oopRange: MATCHUPS[mi].oopRange,
            sizes:    BET_SIZINGS.standard,
            meta: {
              matchup_label: MATCHUPS[mi].label,
              pot_type:      MATCHUPS[mi].potType,
              ip_pos:        MATCHUPS[mi].ip,
              oop_pos:       MATCHUPS[mi].oop,
              board_texture: BOARD_TEMPLATES[bi].texture,
              effective_bb:  MATCHUPS[mi].defaultStack,
              pot_bb:        MATCHUPS[mi].defaultPot,
            },
          },
        });
      }
    }
  }
  return failures;
}

function pickCudaExe() {
  if (existsSync(CUDA_EXE)) return CUDA_EXE;
  if (existsSync(FALLBACK_CUDA_EXE)) return FALLBACK_CUDA_EXE;
  throw new Error(`CUDA exe not found at ${CUDA_EXE} or ${FALLBACK_CUDA_EXE}\nBuild with: cmake --build ../DEEPFOLD-SOLVER/core/build --config Release`);
}

function solveOne(exe, item) {
  return new Promise((resolveFn, reject) => {
    const sizes = item.spot.sizes;
    // Override flop sizes to single 0.75 — see header comment for rationale.
    // Turn/river sizes stay at their normal Standard menu (only flop is the
    // tree-explosion bottleneck on monotone × wide ranges).
    const flopSizesStr = '0.75';
    const cmdArgs = [
      '--pot',           String(item.spot.pot),
      '--stack',         String(item.spot.stack),
      '--board',         item.spot.board,
      '--iterations',    String(args.iterations),
      '--exploitability', String(args.exploitability),
      '--ip-range',      item.spot.ipRange,
      '--oop-range',     item.spot.oopRange,
      '--flop-sizes',    flopSizesStr,
      '--turn-sizes',    sizes.turnBetSizes.join(','),
      '--river-sizes',   sizes.riverBetSizes.join(','),
      '--dcfr-schedule', args.dcfrSchedule,
      '--cpu-persistent-omp', '1',
      '--postsolve',     'full',
      '--strategy-tree-evs', 'visible',
      '--strategy-tree-max-nodes', String(args.maxTreeNodes),
      '--backend',       'cuda',
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
        reject(new Error(`solver output not JSON: ${e.message}`));
      }
    });
  });
}

async function main() {
  const exe = pickCudaExe();
  const failures = findFailedSpots();

  console.log(`CUDA exe: ${exe}`);
  console.log(`Output:   ${OUT_RAW_DIR}`);
  console.log(`Plan:     ${args.iterations} iter, ${args.exploitability}% exploit target, single 0.75 flop sizing`);
  console.log(`\n=== ${failures.length} mono SRP×Std spots to patch (GPU + single flop sizing) ===\n`);

  if (args.dryRun) {
    for (const f of failures) {
      console.log(`  ${f.filename}  [${f.spot.meta.matchup_label} ${f.spot.meta.pot_type} on ${f.spot.board}]`);
    }
    return;
  }

  if (failures.length === 0) {
    console.log('Nothing to patch.');
    return;
  }

  const t0 = Date.now();
  let done = 0, ok = 0, failed = 0;
  const errors = [];
  for (const item of failures) {
    const tStart = Date.now();
    try {
      const { stdout, parsed } = await solveOne(exe, item);
      writeFileSync(item.outPath, stdout);
      const elapsed = (Date.now() - tStart) / 1000;
      ++ok;
      ++done;
      const exploit = parsed.exploitability_pct?.toFixed(2) ?? '?';
      console.log(`[${done}/${failures.length}] ${item.filename}  exploit=${exploit}%  ${elapsed.toFixed(1)}s`);
    } catch (e) {
      ++failed;
      ++done;
      errors.push({ spot: item.filename, error: e.message });
      console.error(`[${done}/${failures.length}] FAIL ${item.filename}: ${e.message}`);
    }
  }

  const total = (Date.now() - t0) / 1000;
  console.log(`\n=== Done ===`);
  console.log(`Total time:  ${(total / 60).toFixed(1)} min`);
  console.log(`Patched:     ${ok}`);
  console.log(`Failed:      ${failed}`);
  if (errors.length > 0) {
    const errPath = join(OUT_RAW_DIR, 'patch_failures.log');
    writeFileSync(errPath, JSON.stringify(errors, null, 2));
    console.log(`Failures logged: ${errPath}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
