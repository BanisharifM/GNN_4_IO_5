#!/bin/bash
#SBATCH --job-name=interpretability
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/interpretability_1M_C6_%j.out
#SBATCH --error=logs/slurm/interpretability_1M_C6_%j.err


srun python analysis/interpretability/test_methods_ior.py \
    --checkpoint /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt \
    --config /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config6.yaml \
    --similarity-pt data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt \
    --similarity-npz data/1M/similarity_output_0.75/similarity_graph.npz \
    --features-csv data/1M/aiio_sample_1000000_normalized.csv \
    --ior-csv test.csv \
    --ior-neighbors 50