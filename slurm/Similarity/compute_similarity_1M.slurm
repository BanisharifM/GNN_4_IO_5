#!/bin/bash
#SBATCH --job-name=compute_similarity_1M
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/compute_similarity_1M_%j.out
#SBATCH --error=logs/slurm/compute_similarity_1M_%j.err

# === Load environment ===
module load cuda
source activate gnn4_env

# === Set distributed environment variables ===
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + RANDOM % 20000)) 
export WORLD_SIZE=4  

CSV_PATH="data/1M/sample_1M.csv"
OUTPUT_PATH="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/1MM/"
BATCH_SIZE=10408
CHUNK_SIZE=10408
TOP_K=30
SAVE_BATCH_SIZE=1000

srun python src/compute_similarity.py \
  --input_csv "$CSV_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size $BATCH_SIZE \
  --chunk_size $CHUNK_SIZE \
  --top_k $TOP_K \
  --world_size $WORLD_SIZE \
  --save_batch_size $SAVE_BATCH_SIZE
