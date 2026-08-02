#!/usr/bin/env node
// Bit-exact A/B harness (ROADMAP §5 rule 1).
//
// Runs the same fixture set through a baseline binary and the current build,
// strips the fields that are ALLOWED to move (wall-clock timing, measured and
// estimated resource telemetry, the backend/device identity string), and
// byte-compares everything else. A refactor that should not change math must
// produce byte-identical stripped output on every fixture.
//
// This is the instrument that caught the lift-overflow bug and every
// regression since. It lived in a session-scoped scratchpad for three
// sessions and was lost each time; it lives here now.
//
// Usage:
//   node scripts/ab.mjs                          # newest baseline/*.exe vs current build
//   node scripts/ab.mjs --baseline <exe>         # pin a specific baseline
//   node scripts/ab.mjs --candidate <exe>
//   node scripts/ab.mjs --only cpu_flop_mono,gpu_river
//   node scripts/ab.mjs --dump                   # write both stripped JSONs on drift
//
// Refreshing the baseline after a ship:
//   cp src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe \
//      baseline/deepsolver_core_v<VERSION>.exe
// The newest file there wins, so nothing else needs updating. Keep the old
// ones or delete them; they are just snapshots. The directory is sandbox-only
// (multi-MB binaries do not belong in git).
//
// Exit code 0 = every fixture byte-identical, 1 = any drift or run failure.

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, '..');
const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : dflt;
};
const has = (name) => argv.includes(`--${name}`);

/** Newest .exe in baseline/ — so a ship only has to drop a file there. */
function newestBaseline() {
  const dir = join(ROOT, 'baseline');
  if (!existsSync(dir)) return null;
  const exes = readdirSync(dir)
    .filter((f) => f.toLowerCase().endsWith('.exe'))
    .map((f) => ({ f, mtime: statSync(join(dir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
  return exes.length ? join(dir, exes[0].f) : null;
}

const baselineExe = resolve(opt('baseline', newestBaseline() ?? ''));
const candidateExe = resolve(
  opt('candidate', join(ROOT, 'core', 'build', 'Release', 'deepsolver_core.exe')));
const only = opt('only', '').split(',').filter(Boolean);
const dumpDir = join(ROOT, 'baseline', 'dump');

// ---------------------------------------------------------------------------
// Fields allowed to differ between two builds of the same math.
// Everything NOT listed here is compared byte-for-byte.
// ---------------------------------------------------------------------------
const STRIP_TOP_LEVEL = [
  'timing',      // wall-clock, per-phase ms
  'resources',   // measured RSS/VRAM + every estimator field
  'backend',     // carries the device name / VRAM string
];

const POT = ['--pot', '100'];
const STACK = ['--stack', '500'];
const QUIET = ['--no-progress'];

// CPU and GPU over singleton-iso (rainbow/paired) and iso-engaged (monotone)
// boards, flop/turn/river, with postsolve full so combo_evs and the strategy
// tree are exercised — that is where every past delta showed up.
//
// NOTE: these use DEFAULT ranges, i.e. 100% live hands. That is deliberate for
// a math-identity check, but it means they cannot prove anything about
// range-restricted paths (B1b). Anything touching live-hand handling needs a
// fixture with real --ip-range/--oop-range on top of this set.
const FIXTURES = [
  { name: 'cpu_flop_rainbow',   args: [...POT, ...STACK, '--board', 'AsKd7c',     '--iterations', '60', '--exploitability', '0', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
  { name: 'cpu_flop_mono',      args: [...POT, ...STACK, '--board', 'AsKsQs',     '--iterations', '20', '--exploitability', '0', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
  { name: 'cpu_turn_singleton', args: [...POT, ...STACK, '--board', 'AsKd7c2h',   '--iterations', '40', '--exploitability', '0', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
  { name: 'cpu_river',          args: [...POT, ...STACK, '--board', 'AsKd7c2h3s', '--iterations', '80', '--exploitability', '0', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
  { name: 'cpu_paired',         args: [...POT, ...STACK, '--board', 'AsAd7c',     '--iterations', '40', '--exploitability', '0', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
  { name: 'cpu_ref_backend',    args: [...POT, ...STACK, '--board', 'AsKd7c',     '--iterations', '40', '--exploitability', '0', '--backend', 'cpu', '--cpu-backend', 'reference', '--postsolve', 'full', ...QUIET] },
  { name: 'gpu_flop_rainbow',   args: [...POT, ...STACK, '--board', 'AsKd7c',     '--iterations', '60', '--exploitability', '0', '--backend', 'gpu', '--postsolve', 'full', ...QUIET] },
  { name: 'gpu_flop_mono',      args: [...POT, ...STACK, '--board', 'AsKsQs',     '--iterations', '20', '--exploitability', '0', '--backend', 'gpu', '--postsolve', 'full', ...QUIET] },
  { name: 'gpu_turn_singleton', args: [...POT, ...STACK, '--board', 'AsKd7c2h',   '--iterations', '40', '--exploitability', '0', '--backend', 'gpu', '--postsolve', 'full', ...QUIET] },
  { name: 'gpu_river',          args: [...POT, ...STACK, '--board', 'AsKd7c2h3s', '--iterations', '80', '--exploitability', '0', '--backend', 'gpu', '--postsolve', 'full', ...QUIET] },
  // Exercises the exploitability probe loop and the early-stop path.
  { name: 'cpu_early_stop',     args: [...POT, ...STACK, '--board', 'AsKd7c',     '--iterations', '400', '--exploitability', '1.5', '--backend', 'cpu', '--postsolve', 'full', ...QUIET] },
];

function run(exe, args, label) {
  try {
    return execFileSync(exe, args, {
      maxBuffer: 512 * 1048576,
      stdio: ['ignore', 'pipe', 'ignore'],
    }).toString();
  } catch (e) {
    throw new Error(`${label} exit ${e.status ?? '?'}`);
  }
}

function strip(raw) {
  const j = JSON.parse(raw);
  for (const k of STRIP_TOP_LEVEL) delete j[k];
  return JSON.stringify(j, null, 2);
}

const sha1 = (p) => createHash('sha1').update(readFileSync(p)).digest('hex').slice(0, 12);

if (!baselineExe || !existsSync(baselineExe)) {
  console.error(
    'no baseline binary.\n' +
    'Drop the shipped sidecar into baseline/, e.g.\n' +
    '  cp src-tauri/binaries/deepsolver_core-x86_64-pc-windows-msvc.exe \\\n' +
    '     baseline/deepsolver_core_v2.6.0.exe\n' +
    'or pass --baseline <exe>.');
  process.exit(2);
}
if (!existsSync(candidateExe)) {
  console.error(`candidate binary not found: ${candidateExe}`);
  process.exit(2);
}

console.log(`baseline : ${baselineExe}  (sha1 ${sha1(baselineExe)})`);
console.log(`candidate: ${candidateExe}  (sha1 ${sha1(candidateExe)})`);
console.log(`stripping top-level: ${STRIP_TOP_LEVEL.join(', ')}\n`);

let bad = 0, ok = 0;
for (const fx of FIXTURES) {
  if (only.length && !only.includes(fx.name)) continue;
  let a, b;
  try {
    a = strip(run(baselineExe, fx.args, 'baseline'));
    b = strip(run(candidateExe, fx.args, 'candidate'));
  } catch (e) {
    console.log(`${fx.name.padEnd(22)} ERROR      ${e.message}`);
    bad++;
    continue;
  }
  if (a === b) {
    ok++;
    console.log(`${fx.name.padEnd(22)} identical  (${a.length} B)`);
  } else {
    bad++;
    const la = a.split('\n'), lb = b.split('\n');
    let i = 0;
    while (i < la.length && i < lb.length && la[i] === lb[i]) i++;
    console.log(`${fx.name.padEnd(22)} DRIFT      first diff at line ${i + 1}`);
    console.log(`    baseline : ${(la[i] ?? '<eof>').trim().slice(0, 120)}`);
    console.log(`    candidate: ${(lb[i] ?? '<eof>').trim().slice(0, 120)}`);
    if (has('dump')) {
      mkdirSync(dumpDir, { recursive: true });
      writeFileSync(join(dumpDir, `${fx.name}.base.json`), a);
      writeFileSync(join(dumpDir, `${fx.name}.cand.json`), b);
      console.log(`    dumped to ${dumpDir}`);
    }
  }
}

console.log(`\n${ok}/${ok + bad} byte-identical` + (bad ? `  — ${bad} DRIFTED` : ''));
process.exit(bad ? 1 : 0);
