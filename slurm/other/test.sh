#!/bin/bash
#SBATCH --job-name=data_sampeling
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/darshan/ior_interpret_pattern1_%j.out
#SBATCH --error=logs/slurm/darshan/ior_interpret_pattern1_%j.err


PYTHON_PATH="/u/mbanisharifdehkordi/.conda/envs/gnn4_env/bin/python"

srun $PYTHON_PATH /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/scripts/ior/ior_interpret_chart.py