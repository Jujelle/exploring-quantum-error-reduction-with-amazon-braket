#!/bin/bash
#SBATCH -J multi
#SBATCH -o multi.out
#SBATCH -e multi.err
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -p demo

if [ ! -d "/shared/error_correction/my_env" ]; then
    python3 -m venv /shared/error_correction/my_env
fi
source /shared/error_correction/my_env/bin/activate

pip install --upgrade pip
pip install amazon-braket-sdk matplotlib

export AWS_DEFAULT_REGION="us-east-1"

srun python3 test.py