import torch
import argparse
import os
from glob import glob
from tqdm import tqdm

def merge_similarity_dicts(input_dir, output_path):
    merged = {}

    pt_files = sorted(glob(os.path.join(input_dir, "similarity_output_total.pt.rank*.pt")))
    print(f"Found {len(pt_files)} partial similarity files.")

    for pt_file in pt_files:
        partial = torch.load(pt_file)
        merged.update(partial)  # Assumes no key overlap
        print(f"Merged {pt_file} with {len(partial)} entries.")

    torch.save(merged, output_path)
    print(f"Final merged similarity saved to: {output_path}")
    print(f"Total rows: {len(merged)}")

def merge_all_rank_rows(base_path_prefix, world_size, output_path):
    merged = {}
    for rank in range(world_size):
        folder = f"{base_path_prefix}_rows_rank{rank}"
        row_files = sorted(glob(os.path.join(folder, "*.pt")))
        for f in tqdm(row_files, desc=f"Merging rank {rank}"):
            row_result = torch.load(f)
            merged.update(row_result)

    torch.save(merged, output_path)
    print(f" Merged all rows to {output_path} with {len(merged)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["rank_files", "row_files"], required=True,
                        help="Merge mode: 'rank_files' for .pt.rank*.pt files, 'row_files' for row-level folders")
    parser.add_argument("--input_dir", type=str, required=True, help="Parent directory for similarity files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged .pt file")
    parser.add_argument("--world_size", type=int, default=4, help="Number of ranks (only used for row_files)")
    args = parser.parse_args()

    if args.mode == "rank_files":
        merge_similarity_dicts(args.input_dir, args.output_path)
    elif args.mode == "row_files":
        merge_all_rank_rows(os.path.join(args.input_dir, os.path.basename(args.output_path)), args.world_size, args.output_path)


# python merge_similarity_dicts.py \
#   --input_dir data/ \
#   --output_path data/similarity_output_merged_100K.pt

# python merge_similarity_dicts.py \
#   --mode row_files \
#   --input_dir data/10K \
#   --output_path data/10K/similarity_output_10K.pt \
#   --world_size 2

