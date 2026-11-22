#!/usr/bin/env bash
# Read-only inventory; no downloads, job submission, or environment modifications.
set -euo pipefail
cd "${1:-$HOME/Crys_METH}"
hostname
pwd
sinfo -o '%P %a %l %D %G'
squeue -u "$USER"
for path in .venv-comparison/bin/python .venv-megnet/bin/python .venv-chgnet/bin/python \
  data/eval/hull/mp.lmdb data/eval/release/exp_logs/base/mp_20/eval \
  utils/eval/reference_baselines/__init__.py \
  data/eval/graphs/physical.npz \
  data/eval/graphs/cosine.npz; do
  if [[ -e "$path" ]]; then ls -ld "$path"; else echo "MISSING: $path"; fi
done
