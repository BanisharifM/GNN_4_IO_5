#!/bin/bash
#SBATCH --job-name=GNN4_IO_4_tune
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/GNN4_IO_4_tune_%j.out
#SBATCH --error=logs/slurm/GNN4_IO_4_tune_%j.err

# Load required modules
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs/slurm
mkdir -p logs/tuning

# Set variables for the workflow
DATA_PATH="data/sample_train_100.csv"
OUTPUT_DIR="logs/tuning/GNN4_IO_4"
TARGET_COLUMN="tag"
TUNE_COMPONENT="all"  # Options: graph, gnn, tabular, all

# Define important features for graph construction
IMPORTANT_FEATURES=(
    "POSIX_SEQ_WRITES"
    "POSIX_SIZE_READ_1K_10K"
    "POSIX_SIZE_READ_10K_100K"
    "POSIX_unique_bytes"
    "POSIX_SIZE_READ_0_100"
    "POSIX_SIZE_WRITE_100K_1M"
    "POSIX_write_only_bytes"
    "POSIX_MEM_NOT_ALIGNED"
    "POSIX_FILE_NOT_ALIGNED"
)

# Convert array to space-separated string
FEATURES_STR=$(IFS=" "; echo "${IMPORTANT_FEATURES[*]}")

# Ray Tune parameters
NUM_SAMPLES=20
MAX_EPOCHS=50
GPUS_PER_TRIAL=0.5
CPUS_PER_TRIAL=8

echo "Starting GNN4_IO_4 hyperparameter tuning at $(date)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run Ray Tune optimization
python -c "
import os
import sys
sys.path.append('$(pwd)')
from src.data import IODataProcessor
from src.raytune_optimizer import RayTuneOptimizer

# Create data processor
processor = IODataProcessor(
    data_path='$DATA_PATH',
    important_features=['$FEATURES_STR'.split()],
    similarity_thresholds={f: 0.05 for f in '$FEATURES_STR'.split()}
)

# Load and preprocess data
processor.load_data()
processor.preprocess_data()

# Create Ray Tune optimizer
optimizer = RayTuneOptimizer(
    data_processor=processor,
    target_column='$TARGET_COLUMN',
    output_dir='$OUTPUT_DIR',
    num_samples=$NUM_SAMPLES,
    max_num_epochs=$MAX_EPOCHS,
    gpus_per_trial=$GPUS_PER_TRIAL,
    cpus_per_trial=$CPUS_PER_TRIAL
)

# Run optimization based on component
if '$TUNE_COMPONENT' == 'graph' or '$TUNE_COMPONENT' == 'all':
    print('Optimizing graph construction parameters...')
    graph_config = optimizer.optimize_graph_construction()
    print(f'Best graph construction config: {graph_config}')

if '$TUNE_COMPONENT' == 'all':
    # Create PyG data with best graph config
    data = processor.create_combined_pyg_data(target_column='$TARGET_COLUMN')
    data = processor.train_val_test_split(data)

if '$TUNE_COMPONENT' == 'gnn' or '$TUNE_COMPONENT' == 'all':
    print('Optimizing GNN hyperparameters...')
    gnn_config = optimizer.optimize_gnn(data)
    print(f'Best GNN config: {gnn_config}')

if '$TUNE_COMPONENT' == 'tabular' or '$TUNE_COMPONENT' == 'all':
    print('Optimizing tabular model hyperparameters...')
    for model_type in ['lightgbm', 'catboost', 'xgboost', 'mlp', 'tabnet']:
        tabular_config = optimizer.optimize_tabular(data, model_type)
        print(f'Best {model_type} config: {tabular_config}')

if '$TUNE_COMPONENT' == 'all':
    print('Running full optimization pipeline...')
    best_configs = optimizer.run_optimization()
    print(f'Optimization completed. Best configurations saved to {os.path.join(\"$OUTPUT_DIR\", \"best_configs.json\")}')
"

echo "Hyperparameter tuning completed at $(date)"

# Train models with best hyperparameters
if [ -f "$OUTPUT_DIR/best_configs.json" ]; then
    echo "Training models with best hyperparameters..."
    
    python scripts/train_with_best_config.py \
        --data_path $DATA_PATH \
        --output_dir "$OUTPUT_DIR/models" \
        --target_column $TARGET_COLUMN \
        --config_path "$OUTPUT_DIR/best_configs.json" \
        --device cuda
    
    echo "Model training with best hyperparameters completed."
fi

echo "All tuning jobs completed at $(date)"
