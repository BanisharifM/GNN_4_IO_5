#!/bin/bash
#SBATCH --job-name=darshan_parser
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=logs/slurm/darshan/darshan_parser_%j.out
#SBATCH --error=logs/slurm/darshan/darshan_parser_%j.err


# Single file processing
srun python scripts/preprocessing/darshan_parser.py \
    /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_log/darshan_log_E2E/e2e_pathological_11642889_64procs_1stripe_64kb.darshan \
    /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/darshan/E2E/e2e_features.csv \
    --config configs/darshan_features.json

# Batch processing entire directory
# srun python scripts/preprocessing/darshan_parser.py \
#     darshan_log/darshan_log_ior/ \
#     data/all_ior_features.csv \
#     --config configs/darshan_features.json \
#     --batch
