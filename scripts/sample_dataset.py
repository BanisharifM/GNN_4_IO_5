import pandas as pd
import argparse
import os

def sample_dataset(input_csv, output_csv, n_rows, seed=42):
    # Load data
    df = pd.read_csv(input_csv)

    # Clamp n_rows to max available
    n_rows = min(n_rows, len(df))

    # Sample with fixed seed for reproducibility
    sampled_df = df.sample(n=n_rows, random_state=seed)

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sampled_df.to_csv(output_csv, index=False)
    print(f"Sampled {n_rows} rows -> {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to full dataset (CSV)")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save sampled dataset")
    parser.add_argument("--n_rows", type=int, required=True, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default=42)")
    args = parser.parse_args()

    sample_dataset(args.input_csv, args.output_csv, args.n_rows, args.seed)


# python sample_dataset.py \
#   --input_csv data/sample_train_total_normalized.csv \
#   --output_csv data/sample_10K.csv \
#   --n_rows 10000

