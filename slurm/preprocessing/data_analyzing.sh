#!/bin/bash
#SBATCH --job-name=data_analyzing
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/data_analyzing_%j.out
#SBATCH --error=logs/slurm/data_analyzing_%j.err


srun python scripts/preprocessing/data_analyzing.py