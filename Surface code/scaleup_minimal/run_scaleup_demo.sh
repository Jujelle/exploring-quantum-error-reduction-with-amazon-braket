#!/bin/bash
# Small surface-code scale-up sweep for AWS PCS.
#
# Submit:
#   sbatch run_scaleup_demo.sh
#
# Current sweep comes from sweep_config.py:
#   d = 3, 5, 7, 9
#   p = 0.005
#   4 replicas per d
#   total tasks = 16

#SBATCH -J scaleup
#SBATCH -o logs/scaleup_%A_%a.out
#SBATCH -e logs/scaleup_%A_%a.err
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH -p demo

set -euo pipefail

mkdir -p logs results

VENV_DIR=${VENV_DIR:-/shared/error_correction/qec_py311}
PYTHON_BIN=${PYTHON_BIN:-python3.11}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[warn] $PYTHON_BIN not found; falling back to python3"
    PYTHON_BIN=python3
fi

echo "[env] using python executable: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python3 - <<'PY' || {
import importlib.util
missing = [
    name for name in ["numpy", "stim", "pymatching"]
    if importlib.util.find_spec(name) is None
]
raise SystemExit(1 if missing else 0)
PY
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy
    python3 -m pip install --only-binary=:all: stim pymatching
}

export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}

read -r DISTANCE ERROR_RATE REPLICA SEED < <(python3 sweep_config.py "$TASK_ID")
SHOTS=$(python3 sweep_config.py --shots)
OUT="results/scaleup_${JOB_ID}_${TASK_ID}.json"

echo "[start] $(date -Iseconds) host=$(hostname) job=$JOB_ID task=$TASK_ID d=$DISTANCE p=$ERROR_RATE replica=$REPLICA shots=$SHOTS seed=$SEED"

srun --unbuffered python3 surface_code_hpc.py \
    --distance "$DISTANCE" \
    --error-rate "$ERROR_RATE" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --replica "$REPLICA" \
    --output "$OUT"

echo "[done] $(date -Iseconds) task=$TASK_ID -> $OUT"
