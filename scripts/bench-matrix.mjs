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
//
// Fixture kinds (T0):
//   default                  — iterations/sec at a fixed iteration count.
//   mode: "time_to_target"   — wall-clock and iterations to reach each accuracy
//                              in `targets` (% of pot), read off the solve's
//                              --convergence-log curve. This is the currency a
//                              PioSOLVER comparison is quoted in; iterations/sec
//                              is not. The tightest declared target must equal
//                              the --exploitability argument, and not reaching
//                              it is a hard fail (a capped run must never be
//                              quoted as a time-to-target).
//   estimate_only: true      — --estimate-only pre-flight; records the memory
//                              estimate instead of a solve. tree_mode comes
//                              from resources.runout_approximated.
//   known_gap: "<reason>"    — this fixture is EXPECTED to fail until the named
//                              work lands. Reported as a GAP, does not set the
//                              process exit code. Its whole purpose is to be
//                              red today and green when the gap closes.
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

/** First probe at or below `target`, or null. Pio-style first-reach. */
const firstReach = (probes, target) =>
  probes.find((p) => p.exploitability_pct <= target) ?? null;

for (const fx of manifest.fixtures) {
  if (only.length && !only.includes(fx.name)) continue;
  const isEstimate = fx.estimate_only === true;
  const isT2T = fx.mode === 'time_to_target';
  // targetHits[target] = array of {iteration, elapsed_ms} across runs
  const targetHits = new Map((fx.targets ?? []).map((t) => [t, []]));
  const probeOverheads = [];
  const rates = [], walls = [], procWalls = [], rss = [], vram = [];
  let last = null;
  let failed = null;
  let completed = 0;
  // An estimate is a deterministic pre-flight; repeating it measures nothing.
  const fxRuns = isEstimate ? 1 : runs;
  for (let k = 0; k < fxRuns && !failed; ++k) {
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
    const wantStatus = isEstimate ? 'estimate' : 'success';
    if (j.status !== wantStatus) { failed = `run ${k + 1}: status=${j.status}`; break; }
    // ---- contract gates (hard failures, not annotations) ----
    // --estimate-only emits no tree_mode; runout_approximated is the same
    // decision under its pre-flight name.
    const mode = isEstimate
      ? (j.resources?.runout_approximated === false ? 'enumerated' : 'collapsed')
      : (j.tree_mode ?? j.resources?.tree_mode ?? null);
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
    if (isT2T) {
      const targets = [...targetHits.keys()];
      const tightest = Math.min(...targets);
      const declared = parseFloat(fx.args[fx.args.indexOf('--exploitability') + 1]);
      if (!(Math.abs(declared - tightest) < 1e-9)) {
        failed = `tightest target ${tightest}% != --exploitability ${declared}%`;
        break;
      }
      const probes = j.convergence?.probes;
      if (!Array.isArray(probes)) {
        failed = `no convergence block — fixture needs --convergence-log`;
        break;
      }
      if (!firstReach(probes, tightest)) {
        // A capped run reports the cap's wall time, not a time-to-target.
        failed = `run ${k + 1}: never reached ${tightest}% ` +
                 `(stopped "${j.early_stop_reason}" at iter ${j.iterations_run}, ` +
                 `best ${Math.min(...probes.map((p) => p.exploitability_pct))}%)`;
        break;
      }
      for (const t of targets) {
        const hit = firstReach(probes, t);
        if (hit) targetHits.get(t).push(hit);
      }
      probeOverheads.push(j.convergence.probe_overhead_ms ?? 0);
    }
    if (!isEstimate) {
      const iterMs = j.timing?.iterations_ms ?? 0;
      rates.push(iterMs > 0 ? (j.iterations_run * 1000) / iterMs : 0);
      walls.push(j.timing?.total_ms ?? 0);
      procWalls.push(procWall);
      rss.push(j.resources?.measured_peak_rss_bytes ?? 0);
      vram.push(j.resources?.measured_peak_vram_bytes ?? 0);
    }
    completed++;
    last = j;
  }
  if (failed) {
    // No partial stats of ANY kind — a fixture either delivers its full
    // run set under contract or reports nothing (review round 2).
    // A known_gap fixture is EXPECTED to be red until its blocker lands, so
    // it is reported but does not condemn the run.
    if (!fx.known_gap) anyFailed = true;
    rates.length = walls.length = procWalls.length = 0;
    rss.length = vram.length = 0;
    probeOverheads.length = 0;
    for (const hits of targetHits.values()) hits.length = 0;
    completed = 0;
    last = null;
  }
  const reqThreadsIdx = fx.args.indexOf('--cpu-threads');
  const timeToTarget = isT2T && completed ? [...targetHits.entries()].map(([t, hits]) => ({
    target_pct: t,
    reached: hits.length === completed,
    // median across runs, so a single scheduling hiccup cannot set the number
    elapsed_ms_median: hits.length ? Math.round(median(hits.map((h) => h.elapsed_ms))) : null,
    iterations_median: hits.length ? Math.round(median(hits.map((h) => h.iteration))) : null,
  })) : null;

  const rec = {
    fixture: fx.name,
    family: fx.family,
    kind: isEstimate ? 'estimate' : (isT2T ? 'time_to_target' : 'throughput'),
    known_gap: fx.known_gap ?? null,
    exe_sha1: exeSha1,
    exe_mtime: exeStat.mtime.toISOString(),
    runs: completed,
    failed,
    time_to_target: timeToTarget,
    // Probe cost is real wall time the user waits, so it is INCLUDED in the
    // elapsed figures above and reported separately rather than netted out.
    probe_overhead_ms_median: probeOverheads.length ? Math.round(median(probeOverheads)) : null,
    final_exploit_pct: last?.exploitability_pct ?? null,
    early_stop_reason: last?.early_stop_reason ?? null,
    backend: last?.backend ?? null,
    tree_mode: last
      ? (isEstimate
          ? (last.resources?.runout_approximated === false ? 'enumerated' : 'collapsed')
          : (last.tree_mode ?? last.resources?.tree_mode ?? null))
      : null,
    terminal_representation: last?.resources?.terminal_representation ?? null,
    threads_requested: reqThreadsIdx >= 0
      ? parseInt(fx.args[reqThreadsIdx + 1], 10) : null,
    threads_effective: last?.resources?.cpu_threads_effective ?? null,
    tree_nodes: last?.timing?.tree_nodes ?? null,
    canonical_combos: last?.resources?.canonical_combos ?? null,
    // B1b inc 2: the index space the solve allocated. Equal to the line above
    // whenever the ranges are full, which is why a default-range row cannot
    // tell you whether compaction engaged.
    live_combos: last?.resources?.live_combos ?? null,
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
  const status = rec.failed
    ? `  ${fx.known_gap ? `GAP (${fx.known_gap})` : 'FAILED'}: ${rec.failed}`
    : '';
  if (isT2T) {
    const cols = (timeToTarget ?? []).map((r) =>
      `${r.target_pct}%: ${r.reached ? `${(r.elapsed_ms_median / 1000).toFixed(2)}s/${r.iterations_median}it` : '—'}`
    ).join('  ');
    console.log(`${rec.fixture.padEnd(20)} ${cols}` +
      (rec.probe_overhead_ms_median !== null ? `  (probe ${rec.probe_overhead_ms_median} ms)` : '') + status);
  } else if (isEstimate) {
    console.log(`${rec.fixture.padEnd(20)} estimate  mode ${rec.tree_mode ?? '-'}  ` +
      `host ${rec.est_peak_host_mb ?? '-'} MB  device total ${rec.est_device_total_mb ?? '-'} MB` + status);
  } else {
    console.log(
      `${rec.fixture.padEnd(20)} ${String(rec.iter_per_sec_median ?? 'FAIL').padStart(10)} it/s med  ` +
      `rss ${String(rec.peak_rss_mb_max ?? '-').padStart(8)} MB (est peak ${rec.est_peak_host_mb ?? '-'})  ` +
      `vram ${String(rec.peak_vram_mb_max ?? '-').padStart(8)} MB  ` +
      `state est/alloc ${stateEst}/${rec.allocated_state_mb} MB` + status);
  }
}

writeFileSync(outPath, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
const gaps = records.filter((r) => r.failed && r.known_gap);
if (gaps.length) {
  console.log(`\nopen gaps (expected red, not counted as failures):`);
  for (const g of gaps) console.log(`  ${g.fixture} — blocked on ${g.known_gap}`);
}
console.log(`\nwrote ${records.length} records → ${outPath}` + (anyFailed ? '  (WITH FAILURES)' : ''));
process.exit(anyFailed ? 1 : 0);
