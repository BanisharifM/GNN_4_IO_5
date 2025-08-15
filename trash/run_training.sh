#!/bin/bash

# Run script for GAT training on I/O performance prediction

# Set paths
SIMILARITY_PT="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/similarity_output_0.75/similarity_graph_20250812_002240.pt"
SIMILARITY_NPZ="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/similarity_output_0.75/similarity_graph_20250812_002240.npz"
FEATURES_CSV="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/aiio_sample_100000_normalized.csv"

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./experiments/gat_exp_${TIMESTAMP}"

# Run training
python scripts/train_gat.py \
    --similarity-pt "$SIMILARITY_PT" \
    --similarity-npz "$SIMILARITY_NPZ" \
    --features-csv "$FEATURES_CSV" \
    --epochs 200 \
    --lr 0.001 \
    --seed 42 \
    --save-dir "$SAVE_DIR" \
    --run-name "gat_100k_${TIMESTAMP}" \
    --interpret \
    --plot \
    2>&1 | tee "${SAVE_DIR}/training.log"

echo "Training complete! Results saved to ${SAVE_DIR}"