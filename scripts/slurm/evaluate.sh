#!/bin/bash
#SBATCH --job-name=cno_evaluate
#SBATCH --output=results/logs/%j_cno_evaluate.out    # STDOUT log file
#SBATCH --error=results/logs/%j_cno_evaluate.err     # STDERR log file
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores
#SBATCH --mem-per-cpu=8G                 # 8GB RAM per core
#SBATCH --gpus=rtx_4090:1                # 1x RTX 4090
#
# Evaluate the trained runs set in cno_closed_loop/paths.py: forward rollout, inverse diagnostics, closed loop.
# Submit from the repository root:
#   mkdir -p results/logs && VENV=/path/to/venv sbatch scripts/slurm/evaluate.sh

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
echo "Evaluate the trained runs set in cno_closed_loop/paths.py: forward rollout, inverse diagnostics, closed loop."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

set -e
python -m cno_closed_loop.forward.evaluate_rollout
python -m cno_closed_loop.inverse.evaluate
python -m cno_closed_loop.closed_loop.compare_cno_vs_fdtd

echo "=================================================="
echo "Job Finished."
