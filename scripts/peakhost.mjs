#!/usr/bin/env node
// Peak-host estimate vs measured (ROADMAP §1a-quater's instrument, rebuilt).
//
// The hard host gate rejects solves whose predicted peak RSS exceeds the
// budget, so the model may be conservative but must never be materially UNDER.
// This prints estimate-vs-measured with the RSS checkpoints next to it, which
// is what localizes a gap: prepare → iterations → finalize says whether the
// missing bytes belong to the tree, the mid-loop exploitability probe, or the
// finalized strategy.
//
// Runs each spot with the probe both ON and OFF, because the probe calls
// finalize() mid-loop and that materializes a strategy copy the finalize-delta
// calibration cannot see.
//
// Usage: node scripts/peakhost.mjs [--only name,name] [--iters 60]

import { execFileSync } from 'node:child_process';

const EXE = 'D:\\DEEPFOLD-SOLVER\\core\\build\\Release\\deepsolver_core.exe';
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const only = opt('only', '').split(',').filter(Boolean);
const ITERS = opt('iters', '60');

const UTG_OPEN =
  'AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,' +
  'AKs:1,AQs:1,AJs:1,ATs:1,A5s:1,A4s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,T9s:1,98s:1,' +
  'AKo:1,AQo:1,AJo:1,KQo:1';
const MP_CALL =
  'JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,' +
  'AQs:1,AJs:1,ATs:1,A9s:1,A8s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,J9s:1,T9s:1,98s:1,87s:1,76s:1,' +
  'AQo:1,AJo:1,KQo:1';

// Narrow ranges throughout: with B1b inc 2 the full-range spots no longer
// exercise the regime the gate now operates in.
const SPOTS = [
  { name: 'mono_flop',    board: 'Ah9h4h',     host: 16384 },
  { name: 'rainbow_flop', board: 'AsKd7c',     host: 16384 },
  { name: 'paired_flop',  board: '7c7d2h',     host: 16384 },
  { name: 'turn',         board: 'AsKd7c2h',   host: 16384 },
  { name: 'river',        board: 'AsKd7c2h9s', host: 16384 },
];

function run(spot, backend, probe) {
  const args = ['--board', spot.board, '--pot', '100', '--stack', '500',
    '--iterations', ITERS, '--backend', backend, '--no-progress',
    '--host-memory-mb', String(spot.host),
    '--oop-range', UTG_OPEN, '--ip-range', MP_CALL];
  if (!probe) args.push('--exploitability', '0');
  const out = execFileSync(EXE, args, { maxBuffer: 1 << 30, encoding: 'utf8' });
  const j = JSON.parse(out);
  const r = j.resources;
  const MB = (x) => (x / 1048576);
  return {
    tree: j.tree_mode,
    nodes: j.timing.tree_nodes, edges: j.timing.tree_edges,
    nc: r.canonical_combos, live: r.live_combos,
    est: MB(r.estimated_peak_host_bytes),
    meas: MB(r.measured_peak_rss_bytes),
    prep: MB(r.measured_rss_after_prepare_bytes),
    iter: MB(r.measured_rss_after_iterations_bytes),
    fin: MB(r.measured_rss_after_finalize_bytes),
  };
}

const f1 = (x) => x.toFixed(1).padStart(7);
let worst = 0, worstWhat = '';
console.log('spot          backend probe  tree_mode   nodes     est     meas    err%   | prepare   iters  finalize');
for (const spot of SPOTS) {
  if (only.length && !only.includes(spot.name)) continue;
  for (const backend of ['cpu', 'gpu']) {
    for (const probe of [true, false]) {
      let r;
      try { r = run(spot, backend, probe); }
      catch (e) { console.log(`${spot.name.padEnd(13)} ${backend} ${probe ? 'on ' : 'off'}  FAILED`); continue; }
      const err = 100 * (r.est - r.meas) / r.meas;
      if (err < worst) { worst = err; worstWhat = `${spot.name}/${backend}/probe=${probe ? 'on' : 'off'}`; }
      console.log(
        `${spot.name.padEnd(13)} ${backend.padEnd(7)} ${probe ? 'on ' : 'off'}    ` +
        `${r.tree.padEnd(11)}${String(r.nodes).padStart(8)} ${f1(r.est)} ${f1(r.meas)} ` +
        `${err.toFixed(1).padStart(7)}   |${f1(r.prep)} ${f1(r.iter)} ${f1(r.fin)}` +
        `${err < 0 ? '   <-- UNDER' : ''}`);
    }
  }
}
console.log(`\nworst under-estimate: ${worst.toFixed(1)}%  (${worstWhat || 'none'})`);
process.exit(worst < -2 ? 1 : 0);
