#!/bin/bash
#SBATCH --job-name=merge_similarity_1M
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/merge_sim_%j.out
#SBATCH --error=logs/slurm/merge_sim_%j.err

module load cuda
source activate gnn4_env

INPUT_PREFIX="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/1M/"
OUTPUT_DIR="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/1M/merged_batches"
WORLD_SIZE=4

srun python scripts/merge_similarity_dicts.py \
  --input_dir_prefix "$INPUT_PREFIX" \
  --world_size $WORLD_SIZE \
  --output_dir "$OUTPUT_DIR"
