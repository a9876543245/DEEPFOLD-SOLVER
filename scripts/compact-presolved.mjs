#!/usr/bin/env node
/**
 * @file scripts/compact-presolved.mjs
 * @brief Phase 2 of PRESOLVE_BUNDLE_PLAN.md — compact + gzip raw bulk-presolve
 *        output into the BundledSpot envelope the frontend expects.
 *
 * Input:  gto_output/presolved/raw/m<i>_b<j>_<sizing>_<stack>bb.json
 * Output: gto_output/presolved/m<i>_b<j>_<sizing>_<stack>bb.json.gz
 *
 * Usage:
 *   node scripts/compact-presolved.mjs                # compact all
 *   node scripts/compact-presolved.mjs --no-tree      # root strategy only (preview)
 *   node scripts/compact-presolved.mjs --in <dir>     # override input dir
 *   node scripts/compact-presolved.mjs --out <dir>    # override output dir
 *
 * Schema v2 (2026-10-07): a spot is a presolve PACK — every flop decision
 * (`strategy_tree`, ~10-60 KB gzipped) and the exact ranges where each line
 * ends the flop (`chance_ranges`), from which the app re-solves the turn and
 * river locally. v1 shipped a 2000-node strategy tree the app never read
 * (99.7% of the bytes). Raw files that are not packs (no chance_ranges:
 * solved before bulk-presolve passed --flop-pack) are skipped.
 */

import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, basename, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

// ----------------------------------------------------------------------------
// Paths + args
// ----------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');

const args = parseArgs(process.argv.slice(2));

function parseArgs(argv) {
  const out = {
    inDir:  join(REPO_ROOT, 'gto_output/presolved/raw'),
    outDir: join(REPO_ROOT, 'gto_output/presolved'),
    includeTree: true,
  };
  for (let i = 0; i < argv.length; ++i) {
    const a = argv[i];
    if      (a === '--in')      out.inDir  = resolve(argv[++i]);
    else if (a === '--out')     out.outDir = resolve(argv[++i]);
    else if (a === '--no-tree') out.includeTree = false;
    else throw new Error(`unknown arg: ${a}`);
  }
  return out;
}

// ----------------------------------------------------------------------------
// Solver version (for invalidation key) — read from package.json
// ----------------------------------------------------------------------------

const PKG = JSON.parse(readFileSync(join(REPO_ROOT, 'package.json'), 'utf-8'));
const SOLVER_VERSION = PKG.version;
const SCHEMA_VERSION = 2;

// ----------------------------------------------------------------------------
// Filename parsing — mirror the scheme in scripts/bulk-presolve.mjs:spotFilename
//   m<matchupIdx>_b<boardIdx>_<sizingKey>_<stackLabel>bb.json
// ----------------------------------------------------------------------------

function parseFilename(name) {
  const m = name.match(/^m(\d+)_b(\d+)_([a-z_]+)_([a-z0-9]+)bb\.json$/);
  if (!m) return null;
  return {
    matchupIdx: parseInt(m[1], 10),
    boardIdx:   parseInt(m[2], 10),
    sizingKey:  m[3],
    stackLabel: m[4],
  };
}

// ----------------------------------------------------------------------------
// Compaction
// ----------------------------------------------------------------------------

/**
 * Strip the raw solver output down to what the frontend needs to render
 * a SpotLibrary spot. PRESOLVE_BUNDLE_PLAN.md §6.2 BundledSpot schema.
 */
function buildBundledSpot(raw, parsed, includeTree) {
  const out = {
    v: SCHEMA_VERSION,
    solver_version: SOLVER_VERSION,
    matchup_idx: parsed.matchupIdx,
    board_idx:   parsed.boardIdx,
    sizing_key:  parsed.sizingKey,
    stack_label: parsed.stackLabel,
    iterations_run:    raw.iterations_run ?? 0,
    exploitability_pct: raw.exploitability_pct ?? null,
    early_stop_reason:  raw.early_stop_reason ?? null,
    // Turn/river equity approximated in the solve (the UI warns), and the
    // pot-percentage menus it was solved with (the turn re-solve uses them).
    runout_approximated: !!raw.runout_approximated,
    sizes:              raw.presolve_sizes ?? null,
    global_strategy:    raw.global_strategy ?? {},
    combo_strategies:   raw.combo_strategies ?? {},
    acting_player:      raw.acting_player ?? null,
    opponent_side:      raw.opponent_side ?? null,
    opponent_range:     raw.opponent_range ?? [],
  };
  // The flop pack: every flop decision + the exact ranges ending each line.
  if (includeTree) {
    out.strategy_tree = raw.strategy_tree;
    out.chance_ranges = raw.chance_ranges;
  }
  return out;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

function main() {
  if (!existsSync(args.inDir)) {
    console.error(`No input directory: ${args.inDir}`);
    console.error(`Run scripts/bulk-presolve.mjs first to populate it.`);
    process.exit(1);
  }
  const files = readdirSync(args.inDir).filter(f => f.endsWith('.json') && f !== 'failures.log');
  if (files.length === 0) {
    console.error(`No .json files found in ${args.inDir}`);
    process.exit(1);
  }

  mkdirSync(args.outDir, { recursive: true });

  console.log(`Solver version: ${SOLVER_VERSION}`);
  console.log(`Schema version: ${SCHEMA_VERSION}`);
  console.log(`Include tree:   ${args.includeTree ? 'yes (Tier B)' : 'no  (Tier A — preview only)'}`);
  console.log(`Compacting ${files.length} spots from ${args.inDir}\n`);

  let totalRaw = 0;
  let totalGz  = 0;
  let skipped  = 0;
  let failed   = 0;
  const failures = [];

  for (const file of files) {
    const inPath = join(args.inDir, file);
    const parsed = parseFilename(file);
    if (!parsed) {
      console.warn(`SKIP ${file} — filename does not match expected pattern`);
      ++skipped;
      continue;
    }
    try {
      const raw = JSON.parse(readFileSync(inPath, 'utf-8'));
      if (raw.status !== 'success') {
        console.warn(`SKIP ${file} — raw solve was not status:success`);
        ++skipped;
        continue;
      }
      if (!raw.chance_ranges || !raw.strategy_tree) {
        console.warn(`SKIP ${file} — not a presolve pack (re-solve it with the current bulk-presolve.mjs)`);
        ++skipped;
        continue;
      }
      const bundled = buildBundledSpot(raw, parsed, args.includeTree);
      const json = JSON.stringify(bundled);
      const gz = gzipSync(json, { level: 9 });
      const outName = file.replace(/\.json$/, '.json.gz');
      const outPath = join(args.outDir, outName);
      writeFileSync(outPath, gz);

      const rawSize = readFileSync(inPath).length;
      totalRaw += rawSize;
      totalGz  += gz.length;

      const ratio = (100 * gz.length / rawSize).toFixed(1);
      console.log(`  ${outName.padEnd(40)} ${(rawSize/1024/1024).toFixed(2).padStart(6)} MB → ${(gz.length/1024).toFixed(0).padStart(5)} KB  (${ratio}%)`);
    } catch (e) {
      ++failed;
      failures.push({ file, error: e.message });
      console.error(`  FAIL ${file}: ${e.message}`);
    }
  }

  console.log('');
  console.log(`Total raw:    ${(totalRaw/1024/1024).toFixed(1)} MB`);
  console.log(`Total gz:     ${(totalGz/1024/1024).toFixed(1)} MB  (avg ${((totalGz/files.length)/1024).toFixed(0)} KB/spot)`);
  console.log(`Bundle slot:  500 MB target (per PRESOLVE_BUNDLE_PLAN §4)`);
  if (totalGz > 500 * 1024 * 1024) {
    console.log(`              ${((totalGz - 500*1024*1024)/1024/1024).toFixed(0)} MB over budget — try --no-tree`);
  } else {
    console.log(`              ${((500*1024*1024 - totalGz)/1024/1024).toFixed(0)} MB under budget`);
  }
  if (skipped) console.log(`Skipped:      ${skipped}`);
  if (failed) {
    console.log(`Failed:       ${failed}`);
    const failPath = join(args.outDir, '_compact-failures.log');
    writeFileSync(failPath, JSON.stringify(failures, null, 2));
    console.log(`              logged to ${failPath}`);
  }
}

main();
