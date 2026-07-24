#!/usr/bin/env node
// Benchmark matrix runner (FABLE_SOLVER_OPTIMIZATION_REPORT §12 / PR-1,
// hardened per the GPT Phase-0 review P1-3/P1-5/P2).
//
// Contract enforced per fixture:
//   - tree_mode must equal the fixture's expected_tree_mode (collapsed runs
//     can never be quoted as enumerated baselines) — hard fail.
//   - effective threads must equal expected_threads when declared (a
//     silently clamped 16T→8T run can never be quoted as 16T) — hard fail.
//   - any run failure fails the whole fixture: no partial medians.
//   - process exits non-zero if ANY fixture failed.
// Stats: median + min + max over N runs (5 samples cannot honestly support
// a p95 — the old "p95" landed on the 4th order statistic ≈ p80).
//
// Usage:
//   node scripts/bench-matrix.mjs [--exe <path>] [--runs 5] [--only a,b] [--out <dir>]
//
// Output: bench-results/<UTC timestamp>.jsonl + console summary. Each record
// carries exe sha1 + mtime so results are never compared across unknown
// binaries (report §14.8).

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : dflt;
};

const exe = resolve(opt('exe', join(here, '..', 'core', 'build', 'Release', 'deepsolver_core.exe')));
const runs = Math.max(1, parseInt(opt('runs', '5'), 10));
const only = opt('only', '').split(',').filter(Boolean);
const outDir = resolve(opt('out', join(here, '..', 'bench-results')));

const manifest = JSON.parse(readFileSync(join(here, 'bench-fixtures.json'), 'utf8'));
const exeStat = statSync(exe);
const exeSha1 = createHash('sha1').update(readFileSync(exe)).digest('hex').slice(0, 12);

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mb = (bytes) => Math.round((bytes / 1048576) * 10) / 10;
const r3 = (x) => Math.round(x * 1000) / 1000;

mkdirSync(outDir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = join(outDir, `${stamp}.jsonl`);
const records = [];
let anyFailed = false;

for (const fx of manifest.fixtures) {
  if (only.length && !only.includes(fx.name)) continue;
  const rates = [], walls = [], procWalls = [], rss = [], vram = [];
  let last = null;
  let failed = null;
  for (let k = 0; k < runs && !failed; ++k) {
    let out;
    const t0 = Date.now();
    try {
      out = execFileSync(exe, fx.args, { maxBuffer: 256 * 1048576, stdio: ['ignore', 'pipe', 'ignore'] }).toString();
    } catch (e) {
      failed = `run ${k + 1}: exit ${e.status}`;
      break;
    }
    const procWall = Date.now() - t0;  // externally measured: covers
                                       // serialization + I/O the child's
                                       // own timing cannot see
    const j = JSON.parse(out);
    if (j.status !== 'success') { failed = `run ${k + 1}: status=${j.status}`; break; }
    // ---- contract gates (hard failures, not annotations) ----
    const mode = j.tree_mode ?? j.resources?.tree_mode ?? null;
    if (mode === 'decomposed') {
      // Decomposed telemetry is NOT aggregated yet — timing/resources
      // describe the monolithic prefix solve only. Reject unconditionally
      // until aggregation lands (ROADMAP §3 2b), even if a fixture asks.
      failed = 'tree_mode=decomposed: telemetry is monolithic-prefix only — rejected until aggregation lands';
      break;
    }
    if (fx.expected_tree_mode && mode !== fx.expected_tree_mode) {
      failed = `tree_mode "${mode}" != expected "${fx.expected_tree_mode}"`;
      break;
    }
    if (fx.expected_threads !== undefined) {
      const eff = j.resources?.cpu_threads_effective ?? 0;
      if (eff !== fx.expected_threads) {
        failed = `threads clamped: requested ${fx.expected_threads}, effective ${eff}`;
        break;
      }
    }
    const iterMs = j.timing?.iterations_ms ?? 0;
    rates.push(iterMs > 0 ? (j.iterations_run * 1000) / iterMs : 0);
    walls.push(j.timing?.total_ms ?? 0);
    procWalls.push(procWall);
    rss.push(j.resources?.measured_peak_rss_bytes ?? 0);
    vram.push(j.resources?.measured_peak_vram_bytes ?? 0);
    last = j;
  }
  if (failed) {
    // No partial stats of ANY kind — a fixture either delivers its full
    // run set under contract or reports nothing (review round 2).
    anyFailed = true;
    rates.length = walls.length = procWalls.length = 0;
    rss.length = vram.length = 0;
    last = null;
  }
  const reqThreadsIdx = fx.args.indexOf('--cpu-threads');
  const rec = {
    fixture: fx.name,
    family: fx.family,
    exe_sha1: exeSha1,
    exe_mtime: exeStat.mtime.toISOString(),
    runs: rates.length,
    failed,
    backend: last?.backend ?? null,
    tree_mode: last?.tree_mode ?? last?.resources?.tree_mode ?? null,
    threads_requested: reqThreadsIdx >= 0
      ? parseInt(fx.args[reqThreadsIdx + 1], 10) : null,
    threads_effective: last?.resources?.cpu_threads_effective ?? null,
    tree_nodes: last?.timing?.tree_nodes ?? null,
    canonical_combos: last?.resources?.canonical_combos ?? null,
    iter_per_sec_median: rates.length ? r3(median(rates)) : null,
    iter_per_sec_min: rates.length ? r3(Math.min(...rates)) : null,
    iter_per_sec_max: rates.length ? r3(Math.max(...rates)) : null,
    total_ms_median: walls.length ? Math.round(median(walls)) : null,
    proc_wall_ms_median: procWalls.length ? Math.round(median(procWalls)) : null,
    peak_rss_mb_max: rss.length ? mb(Math.max(...rss)) : null,
    peak_vram_mb_max: vram.length ? mb(Math.max(...vram)) : null,
    est_peak_host_mb: last ? mb(last.resources?.estimated_peak_host_bytes ?? 0) : null,
    est_overhead_mb: last ? mb(last.resources?.estimated_overhead_bytes ?? 0) : null,
    est_matchup_mb: last ? mb(last.resources?.estimated_matchup_bytes ?? 0) : null,
    est_cpu_state_mb: last ? mb(last.resources?.estimated_cpu_state_bytes ?? 0) : null,
    est_gpu_state_mb: last ? mb(last.resources?.estimated_gpu_state_bytes ?? 0) : null,
    est_gpu_matchup_mb: last ? mb(last.resources?.estimated_gpu_matchup_bytes ?? 0) : null,
    est_device_total_mb: last ? mb(last.resources?.estimated_device_total_bytes ?? 0) : null,
    allocated_state_mb: last ? mb(last.resources?.allocated_state_bytes ?? 0) : null,
    allocated_device_total_mb: last ? mb(last.resources?.allocated_device_total_bytes ?? 0) : null,
  };
  records.push(rec);
  const isGpuRow = (rec.backend ?? '').startsWith('CUDA');
  const stateEst = isGpuRow ? rec.est_gpu_state_mb : rec.est_cpu_state_mb;
  console.log(
    `${rec.fixture.padEnd(20)} ${String(rec.iter_per_sec_median ?? 'FAIL').padStart(10)} it/s med  ` +
    `rss ${String(rec.peak_rss_mb_max ?? '-').padStart(8)} MB (est peak ${rec.est_peak_host_mb ?? '-'})  ` +
    `vram ${String(rec.peak_vram_mb_max ?? '-').padStart(8)} MB  ` +
    `state est/alloc ${stateEst}/${rec.allocated_state_mb} MB` +
    (rec.failed ? `  FAILED: ${rec.failed}` : ''));
}

writeFileSync(outPath, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`\nwrote ${records.length} records → ${outPath}` + (anyFailed ? '  (WITH FAILURES)' : ''));
process.exit(anyFailed ? 1 : 0);
