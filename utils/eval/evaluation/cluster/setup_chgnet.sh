#!/usr/bin/env bash
# Run only inside an allocated compute node. Existing environments are never upgraded.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}" || { echo 'Request a compute allocation first' >&2; exit 2; }
test -f utils/eval/evaluation/requirements-chgnet.txt || { echo 'Run from repository root' >&2; exit 2; }
env_dir="${CHGNET_ENV:-$PWD/.venv-chgnet}"
if [[ -e "$env_dir" ]]; then
  "$env_dir/bin/python" -m pip check
  echo 'Existing environment retained. Run smoke; do not reinstall during a resumable experiment.'
  exit 0
fi
"${PYTHON_BOOTSTRAP:-/usr/bin/python3.12}" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install 'torch==2.6.0' --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"$env_dir/bin/python" -m pip install -r utils/eval/evaluation/requirements-chgnet.txt
"$env_dir/bin/python" -m pip check
mkdir -p data/eval/environment
"$env_dir/bin/python" -m pip freeze > data/eval/environment/chgnet.installed.txt
echo 'Installation recorded. The real-weight smoke test remains required.'
