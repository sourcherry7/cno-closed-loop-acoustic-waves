#!/bin/bash
#SBATCH --job-name=inv_cno_train
#SBATCH --output=results/logs/%j_inv_cno_train.out    # STDOUT log file
#SBATCH --error=results/logs/%j_inv_cno_train.err     # STDERR log file
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores
#SBATCH --mem-per-cpu=8G                 # 8GB RAM per core
#SBATCH --gpus=rtx_4090:1                # Request 1x RTX 4090
#
# Train the inverse CNO (configuration in cno_closed_loop/inverse/train.py).
# Submit from the repository root:
#   mkdir -p results/logs && VENV=/path/to/venv sbatch scripts/slurm/train_inverse.sh

# --- 1. SETUP ENVIRONMENT ---
# Cluster-specific module stack (ETH Euler); adapt or remove for other systems.
module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy

# Python environment with the packages from requirements.txt
source "${VENV:?set VENV to the path of a Python virtual environment}/bin/activate"

# --- 2. DEFINE PATHS ---
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- 3. RUN ---
echo "=================================================="
echo "Train the inverse CNO (configuration in cno_closed_loop/inverse/train.py)."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

python -m cno_closed_loop.inverse.train

echo "=================================================="
echo "Job Finished."
