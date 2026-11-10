#!/bin/bash
#SBATCH --job-name=cno_train
#SBATCH --output=results/logs/%j_cno_train.out    # STDOUT log file
#SBATCH --error=results/logs/%j_cno_train.err     # STDERR log file
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores
#SBATCH --mem-per-cpu=8G                 # 8GB RAM per core
#SBATCH --gpus=rtx_4090:1                # 1x RTX 4090
#
# Train the forward CNO (configuration in cno_closed_loop/forward/train.py).
# Submit from the repository root:
#   mkdir -p results/logs && VENV=/path/to/venv sbatch scripts/slurm/train_forward.sh

# Environment
# Cluster-specific module stack (ETH Euler); adapt or remove for other systems.
module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy

# Python environment with the packages from requirements.txt
source "${VENV:?set VENV to the path of a Python virtual environment}/bin/activate"

# Paths
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Run
echo "=================================================="
echo "Train the forward CNO (configuration in cno_closed_loop/forward/train.py)."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

python -m cno_closed_loop.forward.train

echo "=================================================="
echo "Job Finished."
