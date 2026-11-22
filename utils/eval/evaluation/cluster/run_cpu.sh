#!/usr/bin/env bash
#SBATCH --job-name=crys-eval-cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=evaluation_cpu_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/utils/eval${PYTHONPATH:+:$PYTHONPATH}"
exec "${EVAL_PYTHON:-$PWD/.venv-comparison/bin/python}" -m evaluation.cli "$@"
