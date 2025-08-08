import pandas as pd
import torch
import argparse
import os

def normalize_csv(input_path: str, output_path: str):
    print(f"🔹 Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("🔹 Converting to tensor and normalizing...")
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    normalized_tensor = torch.nn.functional.normalize(data_tensor, p=2, dim=1)

    print("🔹 Converting back to DataFrame and saving...")
    normalized_df = pd.DataFrame(normalized_tensor.cpu().numpy(), columns=df.columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalized_df.to_csv(output_path, index=False)

    print(f"✅ Normalized data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to save normalized CSV")

    args = parser.parse_args()
    normalize_csv(args.input_csv, args.output_csv)
