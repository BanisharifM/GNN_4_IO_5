#!/bin/bash
#SBATCH --job-name=interpretability
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/interpretability_1M_%j.out
#SBATCH --error=logs/slurm/interpretability_1M_%j.err


srun python analysis/interpretability/test_methods.py \
    --checkpoint /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/experiments/gat_exp_20250813_020454/checkpoints/best_model.pt \
    --config /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config4.yaml \
    --similarity-pt /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt \
    --similarity-npz /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/similarity_output_0.75/similarity_graph_20250812_043913.npz \
    --features-csv /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000_normalized.csv \
    --device cuda \
    --output-dir /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/analysis/results \
    --visualize-nodes 25 38 30 229 56 86 19