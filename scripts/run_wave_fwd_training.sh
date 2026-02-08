#!/bin/bash
#SBATCH --job-name=cno_train
#SBATCH --output=results/logs/%j.out     # STDOUT log file
#SBATCH --error=results/logs/%j.err      # STDERR log file
#SBATCH --time=06:30:00                  # 6 hours 30 min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores
#SBATCH --mem-per-cpu=8G                 # 8GB RAM
#SBATCH --gpus=rtx_4090:1                # Request 1x RTX 4090

# --- 1. SETUP ENVIRONMENT ---
# Load the module stack
module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy

# Activate your venv (Path confirmed from your logs)
source /cluster/home/svisnjic/venvs/operatorlearning/bin/activate

# --- 2. DEFINE PATHS ---
PROJECT_ROOT="/cluster/home/svisnjic/Thalassa"
CODE_DIR="$PROJECT_ROOT/src/external/CNO/CNO2d_time_dependent_&_foundation_model"

# Export PYTHONPATH (Crucial so it finds CNO_timeModule_CIN)
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Move to the code directory
cd "$CODE_DIR"

# --- 3. RUN TRAINING ---
echo "=================================================="
echo "Starting Training"
echo "Experiment: Configured manually wave_gauss inside TrainCNO_time_L.py"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

# Run without arguments
python TrainCNO_time_L.py

echo "=================================================="
echo "Job Finished."
