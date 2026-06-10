#!/bin/bash
# Simple AWS PCS / Slurm job-array script for the surface-code threshold sweep.
#
# Put these files in the same cluster folder, for example:
#   /shared/error_correction/
#
# Required files:
#   run_threshold_sweep.sh
#   sweep_config.py
#   surface_code_hpc.py
#   aggregate.py
#
# Submit from that folder:
#   sbatch run_threshold_sweep.sh
#
# After all array tasks finish, aggregate results manually:
#   python3 aggregate.py --results-dir . --prefix threshold \
#       --csv threshold_summary.csv --plot threshold_curve.png

# ================================================================
# 1. PCS / Slurm resources
#    Beginners usually only edit this block.
# ================================================================
#SBATCH -J threshold
#SBATCH -p demo
#SBATCH --array=0-127%16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH -o threshold_%A_%a.out
#SBATCH -e threshold_%A_%a.err

set -euo pipefail

# ================================================================
# 2. Region, Python, and packages
# ================================================================
export AWS_DEFAULT_REGION="us-east-1"
PYTHON_BIN="python3"

# Install these once in your cluster environment before submitting:
#   python3 -m pip install numpy matplotlib
#   python3 -m pip install --only-binary=:all: stim pymatching
REQUIRED_PACKAGES="numpy matplotlib stim pymatching"

# Keep numerical libraries from overusing CPU threads inside each Slurm task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Make sure the script runs in the folder where sbatch was submitted.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "[env] folder: $(pwd)"
echo "[env] host: $(hostname)"
echo "[env] region: $AWS_DEFAULT_REGION"
echo "[env] python: $($PYTHON_BIN --version)"
echo "[env] required packages: $REQUIRED_PACKAGES"

# Stop early with a clear message if required Python packages are missing.
"$PYTHON_BIN" - <<'PY'
import importlib.util

required = ["numpy", "matplotlib", "stim", "pymatching"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall them once before submitting:\n"
        + "  python3 -m pip install numpy matplotlib\n"
        + "  python3 -m pip install --only-binary=:all: stim pymatching"
    )
PY

# ================================================================
# 3. Read this array task's parameters from sweep_config.py
# ================================================================
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

read -r DISTANCE ERROR_RATE REPLICA SEED < <("$PYTHON_BIN" sweep_config.py "$TASK_ID")
SHOTS="$("$PYTHON_BIN" sweep_config.py --shots)"

# One JSON file per Slurm array task. surface_code_hpc.py writes elapsed_sec.
OUTPUT_JSON="threshold_${JOB_ID}_${TASK_ID}.json"

echo "[task] job=$JOB_ID task=$TASK_ID"
echo "[task] d=$DISTANCE p=$ERROR_RATE replica=$REPLICA shots=$SHOTS seed=$SEED"
echo "[task] output=$OUTPUT_JSON"
echo "[start] $(date -Iseconds)"

# ================================================================
# 4. Run one Monte Carlo replica
# ================================================================
srun --unbuffered "$PYTHON_BIN" surface_code_hpc.py \
    --distance "$DISTANCE" \
    --error-rate "$ERROR_RATE" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --replica "$REPLICA" \
    --output "$OUTPUT_JSON"

echo "[done] $(date -Iseconds) output=$OUTPUT_JSON"
