#!/bin/bash
#SBATCH --job-name=corr_analysis
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/correlation/corr_analysis_%j.out
#SBATCH --error=logs/slurm/correlation/corr_analysis_%j.err

# Create log directory if it doesn't exist
mkdir -p logs/slurm/correlation

# Use direct path to your conda environment's Python
PYTHON_PATH="/u/mbanisharifdehkordi/.conda/envs/gnn4_env/bin/python"

# Verify environment
echo "Python location: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"

# Run the correlation analysis
echo "Starting correlation analysis at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

srun $PYTHON_PATH /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/scripts/other/correlation.py

echo "Correlation analysis completed at $(date)"