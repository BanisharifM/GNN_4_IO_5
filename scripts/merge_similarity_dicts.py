import torch
import os
import argparse
from glob import glob
from tqdm import tqdm


def merge_distributed_row_batches_streaming(input_dir_prefix, world_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    temp_batch_size = 100_000  # number of entries per partial file to save
    buffer = {}
    buffer_count = 0
    file_index = 0

    for rank in range(world_size):
        rank_dir = f"{input_dir_prefix}_rows_rank{rank}"
        if not os.path.isdir(rank_dir):
            print(f" Rank directory not found: {rank_dir}")
            continue

        pt_files = sorted(glob(os.path.join(rank_dir, "*.pt")))
        print(f"📂 [Rank {rank}] Found {len(pt_files)} batch files in {rank_dir}")

        for pt_file in tqdm(pt_files, desc=f"Merging rank {rank}"):
            try:
                partial = torch.load(pt_file)
                buffer.update(partial)
                buffer_count += len(partial)
                del partial

                if buffer_count >= temp_batch_size:
                    save_path = os.path.join(output_dir, f"merged_{file_index:05d}.pt")
                    torch.save(buffer, save_path)
                    print(f"Saved batch to {save_path} with {buffer_count} entries")
                    buffer = {}
                    buffer_count = 0
                    file_index += 1

            except Exception as e:
                print(f"Failed to load {pt_file}: {e}")

    # Save remaining
    if buffer_count > 0:
        save_path = os.path.join(output_dir, f"merged_{file_index:05d}.pt")
        torch.save(buffer, save_path)
        print(f" Saved final batch to {save_path} with {buffer_count} entries")

    print(f"Streaming merge completed into directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_prefix", type=str, required=True,
                        help="Prefix of input folders like 'similarity_output' (expects _rows_rank0, _rows_rank1, ...)")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of ranks used in the job")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to store merged batches")
    args = parser.parse_args()

    merge_distributed_row_batches_streaming(args.input_dir_prefix, args.world_size, args.output_dir)

