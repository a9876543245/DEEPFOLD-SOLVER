#!/usr/bin/env bash
# Makes the files to upload to Google Drive MyDrive/deepfold_presolve/ for
# scripts/colab/DEEPFOLD_presolve.ipynb (a Colab A100 80GB presolve run):
#   deepfold_colab_src.tar.gz  engine sources + bulk-presolve.mjs + the TS files its plan reads
#   colab_spots.txt            the spots Colab solves (one bundle filename per line)
#   DEEPFOLD_presolve.ipynb    the notebook (open it in Colab: File > Upload notebook)
#
# usage: scripts/colab/prepare.sh [out dir] [spot list]
#   default spot list: every single-raised-pot spot (the ones too big for a 32 GB card)
set -e
cd "$(dirname "$0")/../.."
out="${1:-colab_upload}"
mkdir -p "$out"
tar -czf "$out/deepfold_colab_src.tar.gz" core/CMakeLists.txt core/include core/cuda core/src core/tests \
  scripts/bulk-presolve.mjs src/lib/ranges.ts src/lib/presolvedSpots.ts src/lib/betSizing.ts
if [ -n "$2" ]; then
  tr -d '\r' < "$2" > "$out/colab_spots.txt"
else
  node scripts/bulk-presolve.mjs --list | awk -F'\t' '$2 == "SRP" { print $1 }' > "$out/colab_spots.txt"
fi
cp scripts/colab/DEEPFOLD_presolve.ipynb "$out/"
echo "$out: $(wc -l < "$out/colab_spots.txt") spots for Colab"
ls -la "$out"
