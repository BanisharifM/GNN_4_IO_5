#!/bin/bash
#SBATCH --job-name=GNN4_IO_4
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# === Setup environment ===
module load cuda

python scripts/sample_dataset.py \
  --input_csv data/sample_train_total_normalized.csv \
  --output_csv data/sample_6M.csv \
  --n_rows 6000000
