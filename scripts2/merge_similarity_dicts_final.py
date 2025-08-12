import torch
import os
import argparse
from glob import glob
from tqdm import tqdm
import gc  # For garbage collection

def merge_final_batches_streaming(input_dir, output_path):
    pt_files = sorted(glob(os.path.join(input_dir, "merged_*.pt")))
    print(f"Found {len(pt_files)} merged batch files.")

    if os.path.exists(output_path):
        merged = torch.load(output_path)
        print(f"Loaded existing merged file with {len(merged)} entries.")
    else:
        merged = {}

    for pt_file in tqdm(pt_files, desc="Merging final batches"):
        try:
            partial = torch.load(pt_file)
            merged.update(partial)
            torch.save(merged, output_path)  # Save incrementally
            del partial
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load {pt_file}: {e}")

    print(f"Final merged file saved at: {output_path} with {len(merged)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing merged_*.pt files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to final merged .pt file")
    args = parser.parse_args()

    merge_final_batches_streaming(args.input_dir, args.output_path)
