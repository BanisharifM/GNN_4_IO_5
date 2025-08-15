#!/bin/bash
#SBATCH --job-name=data_sampeling
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/neighbor_finding_%j.out
#SBATCH --error=logs/slurm/neighbor_finding_%j.err


srun python scripts/test_neighbor_finding.py