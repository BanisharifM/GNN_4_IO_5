import torch
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from torch.multiprocessing import spawn
import time
import math


def setup(rank, world_size):
    if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
        raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set in the environment by SLURM script.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup():
    dist.destroy_process_group()


def compute_cosine_similarity_distributed(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Load and normalize data
    df = pd.read_csv(args.input_csv)
    data = torch.tensor(df.values, dtype=torch.float32).to(device)
    data = torch.nn.functional.normalize(data, p=2, dim=1)

    num_rows = data.size(0)
    chunk_size = args.chunk_size
    top_k = args.top_k
    save_batch_size = args.save_batch_size

    # Output directory
    row_output_dir = f"{args.output_path}_rows_rank{rank}"
    os.makedirs(row_output_dir, exist_ok=True)

    # Find completed batches
    completed_batches = {
        int(f.split(".")[0]) for f in os.listdir(row_output_dir)
        if f.endswith(".pt") and f.split(".")[0].isdigit()
    }

    # Distribute work among ranks
    local_indices = list(range(rank, num_rows, world_size))
    total_batches = math.ceil(len(local_indices) / save_batch_size)

    print(f"[Rank {rank}] Total rows: {len(local_indices)}, Batches: {total_batches}")

    for batch_idx in range(total_batches):
        start = batch_idx * save_batch_size
        end = min(start + save_batch_size, len(local_indices))
        batch_rows = local_indices[start:end]
        batch_file = os.path.join(row_output_dir, f"{start:07d}.pt")

        if os.path.exists(batch_file):
            continue

        batch_results = {}
        for i in batch_rows:
            row = data[i].unsqueeze(0)
            similarities = []

            for j in range(0, num_rows, chunk_size):
                chunk = data[j:j + chunk_size]
                sims = torch.matmul(row, chunk.T).squeeze(0)
                dst_indices = torch.arange(j, j + chunk.size(0), device=device)

                if top_k:
                    vals, indices = torch.topk(sims, min(top_k + 1, sims.size(0)))
                    filtered = [(int(dst_indices[idx]), float(vals[k]))
                                for k, idx in enumerate(indices) if dst_indices[idx] != i]
                    similarities.extend(filtered[:top_k])
                else:
                    similarities.extend([(int(dst_indices[k]), float(sims[k]))
                                         for k in range(sims.size(0)) if dst_indices[k] != i])

            batch_results[i] = similarities
            del row, similarities, sims
            torch.cuda.empty_cache()

        try:
            torch.save(batch_results, batch_file)
        except Exception as e:
            print(f"[Rank {rank}] Failed to save batch {batch_file}: {e}")
            try:
                with open(f"{row_output_dir}/failures_rank{rank}.log", "a") as logf:
                    logf.write(f"Batch {start:07d} failed: {e}\n")
            except:
                pass

        if batch_idx % 10 == 0:
            time.sleep(0.1)

    print(f"[Rank {rank}] Finished writing batches to {row_output_dir}")
    cleanup()


def main_worker(rank, world_size, args):
    compute_cosine_similarity_distributed(rank, world_size, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--save_batch_size", type=int, default=1000)
    args = parser.parse_args()

    spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
