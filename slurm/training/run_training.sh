#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/training_100K_%j.out
#SBATCH --error=logs/slurm/training_100K_%j.err


# Set paths
# 1M dataset paths
# SIMILARITY_PT="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt"
# SIMILARITY_NPZ="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/similarity_output_0.75/similarity_graph_20250812_043913.npz"
# FEATURES_CSV="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000_normalized.csv"

# 100K dataset paths
SIMILARITY_PT="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/similarity_output_0.75/similarity_graph_20250812_002240.pt"
SIMILARITY_NPZ="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/similarity_output_0.75/similarity_graph_20250812_002240.npz"
FEATURES_CSV="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/aiio_sample_100000_normalized.csv"

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./experiments/gat_exp_${TIMESTAMP}"

# Create the directory first
mkdir -p "$SAVE_DIR"

# Set Config path
CONFIG="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config2.yaml"

# Run training
python scripts/train_gat.py \
    --similarity-pt "$SIMILARITY_PT" \
    --similarity-npz "$SIMILARITY_NPZ" \
    --features-csv "$FEATURES_CSV" \
    --config "$CONFIG" \
    --save-dir "$SAVE_DIR" \
    --run-name "gat_100k_${TIMESTAMP}" \
    --force-mode mini \
    --interpret \
    --plot \
    2>&1 | tee "${SAVE_DIR}/training.log"

echo "Training complete! Results saved to ${SAVE_DIR}"