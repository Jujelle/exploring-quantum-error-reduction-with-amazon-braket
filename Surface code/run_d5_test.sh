#!/bin/bash
# Single d=5 AWS PCS smoke test.
#
# Submit from the PCS shared directory with:
#   sbatch run_d5_test.sh

#SBATCH -J d5_test
#SBATCH -o logs/d5_test_%j.out
#SBATCH -e logs/d5_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
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

JOB_ID=${SLURM_JOB_ID:-local}
OUT="results/d5_test_${JOB_ID}.json"

echo "[start] $(date -Iseconds) host=$(hostname) job=${JOB_ID}"

srun --unbuffered python3 d5_single_job.py \
    --p 0.005 \
    --rounds 5 \
    --shots 100000 \
    --seed 12345 \
    --output "$OUT"

echo "[done] $(date -Iseconds) -> $OUT"
