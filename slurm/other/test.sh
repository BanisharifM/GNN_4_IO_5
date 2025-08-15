#!/bin/bash
#SBATCH --job-name=data_sampeling
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/darshan/job_processor_simple_%j.out
#SBATCH --error=logs/slurm/darshan/job_processor_simple_%j.err


srun python /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/scripts/test_job_processor_simple.py