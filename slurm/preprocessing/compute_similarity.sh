#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/compute_similarity_total_%j.out
#SBATCH --error=logs/slurm/compute_similarity_total_%j.err


srun python scripts/compute_similarity_1M.py