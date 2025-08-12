#!/bin/bash
#SBATCH --job-name=data_analyzing
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/data_analyzing_%j.out
#SBATCH --error=logs/slurm/data_analyzing_%j.err


srun python scripts/data_analyzing2.py