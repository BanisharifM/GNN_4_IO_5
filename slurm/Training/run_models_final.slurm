#!/bin/bash
#SBATCH --job-name=GNN4_IO_5
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/Experiment13/run_all_models_100K_%j.out
#SBATCH --error=logs/slurm/Experiment13/run_all_models_100K_%j.err

# === Setup environment ===
module load cuda
# source ~/.conda/etc/profile.d/conda.sh
# conda activate gnn4_env

# === Config ===
CONDA_ENV_PATH="$HOME/.conda/envs/gnn4_env/bin/python"
CONFIG_FILE="configs/experiment13.yml"
EXPERIMENT_NAME="Experiment13"
BASE_OUTPUT_DIR="logs/training/all/${EXPERIMENT_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"

# === Run all models manually (skip 'all' keyword in YAML) ===
for model in tabgnn lightgbm catboost xgboost mlp tabnet combined; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${model}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Running $model model..."
    $CONDA_ENV_PATH scripts/train.py \
        --config_file "$CONFIG_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda \
        --model_type "$model" \
        --precomputed_similarity_path data/100K/similarity_output_merged_100K.pt

    echo "$model model completed."
done

# === Optionally run comparison script ===
$CONDA_ENV_PATH scripts/compare_models.py \
    --results_dir "logs/training" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_file "logs/training/all/${EXPERIMENT_NAME}/comparison_report_${EXPERIMENT_NAME}.json" \
    --plot_file "logs/training/all/${EXPERIMENT_NAME}/model_comparison_${EXPERIMENT_NAME}.png"

echo "All jobs completed at $(date)"
