#!/bin/bash
#SBATCH --job-name=data_normalizing
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/data_normalizing_%j.out
#SBATCH --error=logs/slurm/data_normalizing_%j.err


srun python scripts/preprocessing/data_normalizing2.py
