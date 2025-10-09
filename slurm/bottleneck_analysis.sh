#!/bin/bash
#SBATCH --job-name=io_bottleneck_analysis
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:03:00
#SBATCH --output=logs/slurm/bottleneck/analysis_%j.out
#SBATCH --error=logs/slurm/bottleneck/analysis_%j.err

# Load any necessary modules
# module load python/3.9  # Uncomment if needed

# Set paths
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5"
DATA_DIR="${PROJECT_DIR}/data/1M"
MODEL_PATH="${PROJECT_DIR}/data/1M/best_model.pt"
OUTPUT_DIR="${PROJECT_DIR}/benchmark_evaluation/IO500/ION_V1/Config5/analysis_results"
TEST_FILE="${PROJECT_DIR}/benchmark_evaluation/IO500/ION_V1/Config5/Config5_parsed.csv"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Add project to Python path
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Run the analysis
srun python -m io_bottleneck_analyzer.run_analysis \
    "${TEST_FILE}" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --verbose