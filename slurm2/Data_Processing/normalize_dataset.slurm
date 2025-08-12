#!/bin/bash
#SBATCH --job-name=normalize_csv
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/normalize_csv_%j.out
#SBATCH --error=logs/slurm/normalize_csv_%j.err

CONDA_ENV_PATH="$HOME/.conda/envs/gnn4_env/bin/python"

INPUT_PATH="data/sample_train_total.csv"
OUTPUT_PATH="data/sample_train_total_normalized.csv"

$CONDA_ENV_PATH scripts/normalize_dataset.py \
    --input_csv "$INPUT_PATH" \
    --output_csv "$OUTPUT_PATH"
