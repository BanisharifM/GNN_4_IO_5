#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compute_similarity_100K_%j.out
#SBATCH --error=logs/slurm/compute_similarity_100K_%j.err


srun python scripts/compute_similarity.py