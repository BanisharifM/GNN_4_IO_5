#!/bin/bash
#SBATCH --job-name=compute_similarity_10K
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:2                
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compute_similarity_10K_%j.out
#SBATCH --error=logs/slurm/compute_similarity_10K_%j.err

# === Load environment ===
module load cuda
source activate gnn4_env

# === Set distributed environment variables ===
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + RANDOM % 20000)) 
export WORLD_SIZE=2  

CSV_PATH="data/10K/sample_10K.csv"
OUTPUT_PATH="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/10K/"
BATCH_SIZE=10408
CHUNK_SIZE=10408
TOP_K=30

srun python src/compute_similarity.py \
  --input_csv "$CSV_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size $BATCH_SIZE \
  --chunk_size $CHUNK_SIZE \
  --top_k $TOP_K \
  --world_size $WORLD_SIZE
