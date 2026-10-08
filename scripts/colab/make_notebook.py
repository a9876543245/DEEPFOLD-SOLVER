"""Writes scripts/colab/DEEPFOLD_presolve.ipynb (run: python scripts/colab/make_notebook.py).

The notebook runs scripts/bulk-presolve.mjs on a Colab A100 80GB: it builds the
engine for sm_80, checks the GPU build against the CPU build on a small spot,
then solves the spots listed in colab_spots.txt into Google Drive (resumable).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# DEEPFOLD presolve on a Colab A100 80GB

Solves presolve packs (`scripts/bulk-presolve.mjs`) for the spots in `colab_spots.txt`.

**Before running:**
1. Runtime → Change runtime type → **A100 GPU** with **High-RAM** on (the 80 GB A100).
2. In Google Drive, create `MyDrive/deepfold_presolve/` and upload `deepfold_colab_src.tar.gz`
   and `colab_spots.txt` (made by `scripts/colab/prepare.sh` on the dev machine).

Then **Runtime → Run all**. Each solved spot is written to `MyDrive/deepfold_presolve/raw/` as soon as it
finishes, so a disconnect loses at most the spot in progress: reconnect and **Run all** again (`--resume`).
"""),
    code("""
import subprocess
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True).stdout.strip()
print(gpu)
assert int(gpu.split(',')[-1]) >= 79000, 'Needs the A100 80GB: Runtime > Change runtime type > A100 + High-RAM'
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
WORK = '/content/drive/MyDrive/deepfold_presolve'
RAW = WORK + '/raw'
import os
os.makedirs(RAW, exist_ok=True)
print(sorted(os.listdir(WORK)))
"""),
    code("""
%%bash -s "$WORK"
set -e
rm -rf /content/deepfold && mkdir -p /content/deepfold && cd /content/deepfold
tar -xzf "$1/deepfold_colab_src.tar.gz"
if [ ! -x /content/node/bin/node ]; then
  mkdir -p /content/node
  curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar -xJ -C /content/node --strip-components=1
fi
/content/node/bin/node --version
nvcc --version | tail -1
pip install -q --upgrade cmake > /dev/null && hash -r && cmake --version | head -1
# CMAKE_CUDA_COMPILER skips check_language(CUDA), whose probe project cannot
# find make on Colab and silently falls back to a CPU-only build.
cmake -S core -B build -DCMAKE_BUILD_TYPE=Release -DDEEPSOLVER_CUDA_ARCHS=80   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /content/cmake.log
grep -q "CUDA Enabled: TRUE" /content/cmake.log || { echo "CUDA not enabled:"; grep -i cuda /content/cmake.log; exit 1; }
cmake --build build --target deepsolver_core -j "$(nproc)" > /content/build.log 2>&1 || { tail -40 /content/build.log; exit 1; }
ls -la build/deepsolver_core
"""),
    md("""
**Check the GPU build:** the same small spot on the GPU and on the CPU must agree (same engine, two backends).
"""),
    code("""
import json, subprocess
EXE = '/content/deepfold/build/deepsolver_core'
spot = ['--pot', '100', '--stack', '400', '--board', 'Td9d6h',
        '--oop-range', 'AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,QJs,JTs,T9s,98s,AKo,AQo',
        '--ip-range', 'QQ,JJ,TT,99,88,77,AQs,AJs,KQs,KJs,QJs,JTs,T9s,98s,87s,AQo,KQo',
        '--flop-sizes', '0.33,0.75', '--turn-sizes', '0.75', '--river-sizes', '0.75',
        '--iterations', '300', '--exploitability', '0', '--no-progress', '--flop-pack']
res = {}
for backend in ('gpu', 'cpu'):
    p = subprocess.run([EXE, *spot, '--backend', backend], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    res[backend] = json.loads(p.stdout)
g, c = res['gpu'], res['cpu']
print('backend:', g['backend'])
print('exploitability gpu/cpu: %.4f / %.4f' % (g['exploitability_pct'], c['exploitability_pct']))
assert g['backend'].startswith('CUDA'), 'the solve did not run on the GPU'
assert abs(g['exploitability_pct'] - c['exploitability_pct']) < 0.05
# Aggregate frequencies only: single hands that (almost) never reach a node
# have no unique strategy, so per-hand mixes differ between backends.
gdiff = max(abs(float(c['global_strategy'][k].rstrip('%')) - float(g['global_strategy'][k].rstrip('%')))
            for k in c['global_strategy'])
print('largest root action frequency difference: %.2f pp' % gdiff)
assert gdiff < 2.0
assert set(g['chance_ranges']) == set(c['chance_ranges']) and len(g['chance_ranges']) > 0
print('chance lines in the pack:', len(g['chance_ranges']))
print('GPU build OK')
"""),
    md("""
**Solve the listed spots** (hours; progress prints one line per spot). Re-running skips the spots already in `raw/`.
"""),
    code("""
import os
host_mb = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**20 * 0.85)
print('host budget MB:', host_mb)
!cd /content/deepfold && /content/node/bin/node scripts/bulk-presolve.mjs --only "{WORK}/colab_spots.txt" --out "{RAW}" --resume --exe build/deepsolver_core --backend cuda --gpu-memory-mb 76000 --host-memory-mb {host_mb} 2>&1 | tee -a "{WORK}/presolve.log"
"""),
    code("""
listed = [l.strip() for l in open(WORK + '/colab_spots.txt') if l.strip()]
done = set(f for f in os.listdir(RAW) if f.endswith('.json'))
missing = [f for f in listed if f not in done]
print(f'{len(listed) - len(missing)} / {len(listed)} solved')
if missing:
    print('missing (run again):', missing[:10])
else:
    !cd "{WORK}" && rm -f raw.zip && zip -qr raw.zip raw && ls -la raw.zip
    print('Done: download MyDrive/deepfold_presolve/raw.zip to the dev machine.')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(HERE, 'DEEPFOLD_presolve.ipynb')
with open(out, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote', out)
