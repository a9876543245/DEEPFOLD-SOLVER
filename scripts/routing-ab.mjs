#!/usr/bin/env node
// Routing A/B for B1b inc 2 (ROADMAP §5 rule 4).
//
// The usual routing matrix runs DEFAULT ranges, where compaction is the
// identity — it would show zero flips and prove nothing. This one carries real
// ranges and toggles DEEPSOLVER_B1B_COMPACT on the SAME binary, which isolates
// the routing effect of the index space from every other difference between
// two builds.
//
// What it must rule out: any spot becoming STRICTER. Compaction only lowers
// estimates, so a solve that ran before must still run.

import { execFileSync } from 'node:child_process';

const EXE = 'D:\\DEEPFOLD-SOLVER\\core\\build\\Release\\deepsolver_core.exe';

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

const BOARDS = [
  ['mono_flop',    'Ah9h4h'],
  ['rainbow_flop', 'AsKd7c'],
  ['twotone_flop', 'Kd8d3c'],
  ['paired_flop',  '7c7d2h'],
  ['turn_single',  'AsKd7c2h'],
  ['turn_paired',  '7c7d2hKs'],
  ['river',        'AsKd7c2h9s'],
];
const MATCHUPS = [['utg_mp', UTG_OPEN, MP_CALL], ['sb_bb', SB_OPEN, BB_DEF]];
const BUDGETS = [512, 2048, 6144, 16384];
const MODES = ['solve_cpu', 'solve_auto', 'estimate'];

function probe(board, oop, ip, host, mode, compact) {
  const args = ['--board', board, '--pot', '100', '--stack', '500',
    '--iterations', '5', '--exploitability', '0', '--no-progress',
    '--no-strategy-tree', '--postsolve', 'none',
    '--host-memory-mb', String(host), '--oop-range', oop, '--ip-range', ip];
  if (mode === 'solve_cpu') args.push('--backend', 'cpu');
  else if (mode === 'solve_auto') args.push('--backend', 'auto');
  else args.push('--estimate-only');
  const env = { ...process.env };
  if (!compact) env.DEEPSOLVER_B1B_COMPACT = '0';
  let out, rc = 0;
  try {
    out = execFileSync(EXE, args, { env, maxBuffer: 1 << 30, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (e) {
    rc = e.status ?? -1;
    out = e.stdout || '';
  }
  let j = null;
  try { j = JSON.parse(out); } catch { /* non-JSON failure */ }
  if (!j) return { rc, decision: `rc=${rc}/unparseable` };
  const r = j.resources || {};
  return {
    rc,
    decision: [
      j.status ?? 'estimate',
      j.tree_mode ?? r.tree_mode ?? '-',
      r.budget_decision ?? j.budget_decision ?? '-',
      (j.backend ?? '-').split(' ')[0],
    ].join('|'),
    live: r.live_combos, nc: r.canonical_combos,
  };
}

let probes = 0, flips = 0;
const classes = new Map();
for (const [mname, oop, ip] of MATCHUPS) {
  for (const [bname, board] of BOARDS) {
    for (const host of BUDGETS) {
      for (const mode of MODES) {
        const off = probe(board, oop, ip, host, mode, false);
        const on = probe(board, oop, ip, host, mode, true);
        probes++;
        if (off.decision === on.decision) continue;
        flips++;
        const key = `${off.decision}  →  ${on.decision}`;
        if (!classes.has(key)) classes.set(key, []);
        classes.get(key).push(`${mname}/${bname}/${host}MB/${mode} (live ${on.live}/${on.nc})`);
      }
    }
  }
}
console.log(`${probes} probes, ${flips} flips\n`);
for (const [k, v] of [...classes.entries()].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`[${v.length}] ${k}`);
  for (const s of v.slice(0, 6)) console.log(`      ${s}`);
  if (v.length > 6) console.log(`      ... +${v.length - 6} more`);
}
