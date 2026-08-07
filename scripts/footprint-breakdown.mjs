#!/usr/bin/env node
// Decompose a CPU solve's host footprint into the terms a Pio-style DFS solver
// would and would not carry. This is the arithmetic behind ROADMAP §0-pre
// ("Why their RAM is smaller"), so a new session can re-derive it in one
// command instead of trusting a table.
//
// It runs --estimate-only. The CPU state estimator is byte-exact against the
// real allocation (Phase 0 contract), so the totals below are measurements;
// the two SPLITS are derived here and labelled as such.
//
// Usage: node scripts/footprint-breakdown.mjs [--exe <path>] [--board X,Y]

import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const EXE = opt('exe', join(ROOT, 'core', 'build', 'Release', 'deepsolver_core.exe'));
const BOARDS = opt('board', 'Td9d6h,AsKd7c').split(',');

// The §0-pre anchor: 100bb start, SB 3bets to 10, BB calls → pot 20bb, 90bb
// behind (SPR 4.5), chip units ×10. Full menu on every street.
const POT = '200', STACK = '900';
const SB_3BET =
  'AA:1,KK:1,QQ:1,JJ:1,TT:0.5,99:0.3,88:0.2,77:0.2,66:0.2,55:0.2,' +
  'AKs:1,AQs:1,AJs:1,ATs:0.7,A5s:1,A4s:1,A3s:0.6,A2s:0.4,' +
  'KQs:1,KJs:0.8,KTs:0.5,QJs:0.6,QTs:0.4,JTs:0.6,T9s:0.3,98s:0.3,' +
  'AKo:1,AQo:0.8,AJo:0.4,KQo:0.4';
const BB_CALL =
  'QQ:0.5,JJ:1,TT:1,99:1,88:1,77:1,66:0.8,55:0.6,44:0.4,33:0.4,22:0.4,' +
  'AQs:1,AJs:1,ATs:1,A9s:0.5,A5s:0.8,A4s:0.6,KQs:1,KJs:1,KTs:0.8,' +
  'QJs:1,QTs:0.8,JTs:1,T9s:0.8,98s:0.6,87s:0.5,76s:0.4,65s:0.3,' +
  'AQo:1,AJo:0.6,KQo:0.8';

// Mirrors memory_budget.h. If either constant changes there this file goes
// quietly wrong, so it re-derives the split from published totals rather than
// hard-coding any byte count.
const LANE = 8;                    // kCpuActionLaneFloats
const STRAT_ARRAYS = 3;            // kCpuStateArraysPerNode
const laneStride = (nc) => Math.ceil(nc / LANE) * LANE;

const MB = (b) => b / 1048576;
const fmt = (v) => v.toFixed(0).padStart(6);

for (const board of BOARDS) {
  const args = ['--board', board, '--pot', POT, '--stack', STACK,
    '--iterations', '4000', '--exploitability', '0.005', '--backend', 'cpu',
    '--no-progress', '--flop-sizes', '0.33,0.75', '--turn-sizes', '0.33,0.75',
    '--river-sizes', '0.33,0.75', '--host-memory-mb', '24000',
    '--postsolve', 'none', '--no-strategy-tree', '--estimate-only',
    '--oop-range', SB_3BET, '--ip-range', BB_CALL];
  const r = JSON.parse(execFileSync(EXE, args,
    { maxBuffer: 1 << 30, encoding: 'utf8' })).resources;

  const N = r.tree_nodes, live = r.live_combos, stride = laneStride(live);
  if (!N) {
    console.log(`${board}: this binary does not report resources.tree_nodes, ` +
      `so the split cannot be derived. Needs a post-2026-08-05 build.`);
    continue;
  }

  // DERIVED: the estimator publishes one cpu_state total. Its halves are
  // bytes_for_levelized_cpu_extra(N, nc) — closed form, recomputed here — and
  // the strat arrays, which are the remainder.
  const levelized = N * stride * 3 * 4;
  const strat = r.estimated_cpu_state_bytes - levelized;
  const perArray = strat / STRAT_ARRAYS;
  const slots = strat / (stride * STRAT_ARRAYS * 4);
  // bytes_for_final_strategy × kFinalStrategyCopiesCpu (2.0); rows un-padded.
  const finalStrat = 2 * (slots * live * 4 + r.player_nodes * 32);
  const tree = N * 31 * 2 + N * 9 * 2;    // 31 B/node + 9 B/edge, 2× growth

  console.log(`\n${board}  ${N.toLocaleString()} nodes  ` +
    `${r.player_nodes.toLocaleString()} player nodes  ` +
    `nc ${r.canonical_combos} live ${live} (stride ${stride})  ` +
    `~${(slots / r.player_nodes).toFixed(2)} slots/player node`);
  const row = (label, bytes, note) =>
    console.log(`  ${label.padEnd(36)} ${fmt(MB(bytes))}   ${note}`);
  row('regrets + strategy_sum', perArray * 2, 'yes — irreducible CFR state');
  row('current_strategy (3rd array)', perArray, 'NO  — GPU dropped it, B1a inc 3');
  row('levelized 3 x N x nc buffers', levelized, 'NO  — a DFS keeps these on the stack');
  row('2x materialized final strategy', finalStrat, 'partly (peak-only term)');
  row('matchup tables', r.estimated_matchup_bytes, 'partly');
  row('flat game tree', tree, 'yes, smaller');
  row('process baseline', r.estimated_overhead_bytes, 'yes');
  console.log(`  ${'ESTIMATED PEAK HOST'.padEnd(36)} ` +
    `${fmt(MB(r.estimated_peak_host_bytes))}`);

  const a = r.estimated_peak_host_bytes - perArray;      // Phase 2b item 1
  const b = a - levelized;                               // + item 2 (CPU B3)
  const c = b - perArray;                                // + item 3 (i16 pair)
  console.log(`  → drop current_strategy:        ${fmt(MB(a))} MB`);
  console.log(`  → + DFS, no flat buffers:       ${fmt(MB(b))} MB`);
  console.log(`  → + i16 on the surviving pair:  ${fmt(MB(c))} MB` +
    `   (Pio publishes 634 MB — different tree)`);
}
