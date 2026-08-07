#!/usr/bin/env node
// B1b inc 2 — convergence equivalence, the bar that replaces the byte compare.
//
// Per-node strategies cannot discriminate here: `noise.mjs` showed a
// provably-equivalent pair (FORCE_DENSE vs default, same index space, same
// math) already moves whole-tree strategies by up to 47pp at 200 iterations,
// because DCFR + regret matching is chaotic in the summation order. What must
// hold is that the compacted solve is the same GAME solved equally well:
// exploitability tracks iteration-for-iteration, and it gets there faster.
//
// Prints the two curves side by side plus time-to-target for each threshold —
// the currency ROADMAP §0 settled on.

import { execFileSync } from 'node:child_process';

const EXE = 'D:\\DEEPFOLD-SOLVER\\core\\build\\Release\\deepsolver_core.exe';
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const ITERS = opt('iters', '2000');
const INTERVAL = opt('interval', '50');
const only = opt('only', '').split(',').filter(Boolean);

const UTG_OPEN =
  'AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,' +
  'AKs:1,AQs:1,AJs:1,ATs:1,A5s:1,A4s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,T9s:1,98s:1,' +
  'AKo:1,AQo:1,AJo:1,KQo:1';
const MP_CALL =
  'JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,' +
  'AQs:1,AJs:1,ATs:1,A9s:1,A8s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,J9s:1,T9s:1,98s:1,87s:1,76s:1,' +
  'AQo:1,AJo:1,KQo:1';
const SB_OPEN =
  'AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,' +
  'AKs:1,AQs:1,AJs:1,ATs:1,A9s:1,A8s:1,A7s:1,A6s:1,A5s:1,A4s:1,A3s:1,A2s:1,' +
  'KQs:1,KJs:1,KTs:1,K9s:1,K8s:1,QJs:1,QTs:1,Q9s:1,JTs:1,J9s:1,T9s:1,98s:1,87s:1,76s:1,65s:1,' +
  'AKo:1,AQo:1,AJo:1,ATo:1,KQo:1,KJo:1,QJo:1';
const BB_DEF =
  'TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,' +
  'AJs:1,ATs:1,A9s:1,A8s:1,A7s:1,A6s:1,A5s:1,A4s:1,A3s:1,A2s:1,' +
  'KJs:1,KTs:1,K9s:1,K8s:1,K7s:1,QTs:1,Q9s:1,J9s:1,T8s:1,97s:1,86s:1,75s:1,64s:1,54s:1,' +
  'AJo:1,ATo:1,A9o:1,KJo:1,KTo:1,QJo:1,QTo:1,JTo:1,T9o:1';

const FIXTURES = [
  { name: 'utg_mp_turn',      board: 'AsKd7c2h',   oop: UTG_OPEN, ip: MP_CALL, host: 4096 },
  { name: 'utg_mp_river',     board: 'AsKd7c2h9s', oop: UTG_OPEN, ip: MP_CALL, host: 4096 },
  { name: 'utg_mp_flop_rain', board: 'AsKd7c',     oop: UTG_OPEN, ip: MP_CALL, host: 512 },
  { name: 'sbbb_turn_paired', board: '7c7d2hKs',   oop: SB_OPEN,  ip: BB_DEF,  host: 4096 },
];

const TARGETS = [2.0, 1.0, 0.5, 0.3];

function run(fx, env) {
  const args = ['--board', fx.board, '--pot', '100', '--stack', '500',
    '--iterations', ITERS, '--backend', 'cpu', '--no-progress',
    '--convergence-log', '--exploitability-interval', INTERVAL,
    '--exploitability', '0.0001',            // never early-stop; we want the curve
    '--host-memory-mb', String(fx.host),
    '--oop-range', fx.oop, '--ip-range', fx.ip];
  return JSON.parse(execFileSync(EXE, args,
    { env: { ...process.env, ...env }, maxBuffer: 1 << 30, encoding: 'utf8' }));
}

/** First probe at or below `t`, linearly interpolated in elapsed_ms. */
function timeToTarget(probes, t) {
  for (let i = 0; i < probes.length; i++) {
    if (probes[i].exploitability_pct <= t) return probes[i].elapsed_ms;
  }
  return null;
}

for (const fx of FIXTURES) {
  if (only.length && !only.includes(fx.name)) continue;
  const on = run(fx, {});
  const off = run(fx, { DEEPSOLVER_B1B_COMPACT: '0' });
  const ctrl = run(fx, { DEEPSOLVER_B1B_COMPACT: '0', DEEPSOLVER_FORCE_DENSE: '1' });
  const pOn = on.convergence.probes, pOff = off.convergence.probes, pC = ctrl.convergence.probes;

  const r = on.resources;
  console.log(`\n${fx.name}  nc=${r.canonical_combos} live=${r.live_combos} ` +
    `(${(100 * r.live_combos / r.canonical_combos).toFixed(1)}%)  ` +
    `tree_mode ${off.tree_mode}/${on.tree_mode}`);

  // Curve agreement: max |Δexploitability| at matched iterations, and the same
  // for the known-equivalent control, so the two are read on one scale.
  const byIter = (ps) => new Map(ps.map((p) => [p.iteration, p.exploitability_pct]));
  const mOff = byIter(pOff), mC = byIter(pC);
  let dB1b = 0, dCtrl = 0;
  for (const p of pOn) {
    if (mOff.has(p.iteration)) dB1b = Math.max(dB1b, Math.abs(p.exploitability_pct - mOff.get(p.iteration)));
  }
  for (const p of pC) {
    if (mOff.has(p.iteration)) dCtrl = Math.max(dCtrl, Math.abs(p.exploitability_pct - mOff.get(p.iteration)));
  }
  const last = (ps) => ps.length ? ps[ps.length - 1] : null;
  console.log(`  final @${ITERS}: full ${last(pOff)?.exploitability_pct}%  ` +
    `compact ${last(pOn)?.exploitability_pct}%  ctrl(dense) ${last(pC)?.exploitability_pct}%`);
  console.log(`  max curve Δ vs full:  compaction ${dB1b.toFixed(4)}pp   ` +
    `FORCE_DENSE control ${dCtrl.toFixed(4)}pp`);
  const row = TARGETS.map((t) => {
    const a = timeToTarget(pOff, t), b = timeToTarget(pOn, t);
    if (a == null && b == null) return `${t}%: —`;
    if (a == null) return `${t}%: full never / compact ${(b / 1000).toFixed(2)}s`;
    if (b == null) return `${t}%: full ${(a / 1000).toFixed(2)}s / compact NEVER`;
    return `${t}%: ${(a / 1000).toFixed(2)}→${(b / 1000).toFixed(2)}s (${(a / b).toFixed(2)}×)`;
  });
  console.log(`  time-to-target  ${row.join('   ')}`);
}
