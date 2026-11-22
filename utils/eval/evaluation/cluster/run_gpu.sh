#!/usr/bin/env bash
#SBATCH --job-name=crys-eval-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=evaluation_gpu_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
stage="${1:?Supply smoke or predict}"; model="${2:?Supply megnet or chgnet}"
shift 2
case "$model" in
  megnet) interpreter="${MEGNET_PYTHON:-$PWD/.venv-megnet/bin/python}" ;;
  chgnet) interpreter="${CHGNET_PYTHON:-$PWD/.venv-chgnet/bin/python}" ;;
  *) echo 'Choose megnet or chgnet' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/utils/eval${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/crys-eval-${SLURM_JOB_ID:-interactive}-XXXXXX")"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
exec "$interpreter" -m evaluation.cli "$stage" --model "$model" "$@"
