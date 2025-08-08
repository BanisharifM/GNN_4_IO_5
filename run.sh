#!/bin/bash
# Run script for GNN4_IO_4

# Check if data path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <data_path> [output_dir] [target_column]"
    echo "Example: $0 /path/to/data.csv ./output performance_tag"
    exit 1
fi

# Set default values
DATA_PATH=$1
OUTPUT_DIR=${2:-"./output"}
TARGET_COLUMN=${3:-"TARGET"}

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the training script
echo "Running GNN4_IO_4 training with TabGNN model..."
'~/.conda/envs/gnn4_env/bin/python' scripts/train.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --target_column $TARGET_COLUMN \
    --model_type tabgnn \
    --gnn_type gcn \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --patience 10 \
    --seed 42

echo "Training complete. Results saved to $OUTPUT_DIR"
