#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

def minmax(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def zscore(x, mean, std):
    return (x - mean) / std

def main():
    p = argparse.ArgumentParser(
        description="Compare min-max vs z-score after invert-log10"
    )
    p.add_argument("csv", help="Path to normalized CSV")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--frac", type=float, default=0.002,
                   help="Fraction to sample per chunk")
    p.add_argument("--columns", nargs="+",
                   default=["POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN", "tag"],
                   help="Columns to test")
    p.add_argument("--xmin", type=float, required=True)
    p.add_argument("--xmax", type=float, required=True)
    p.add_argument("--mean", type=float, required=True)
    p.add_argument("--std", type=float, required=True)
    args = p.parse_args()

    # sample up to ~100k rows
    samples = []
    total = 0
    for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
        samp = chunk.sample(frac=args.frac, random_state=1)
        samples.append(samp)
        total += len(samp)
        if total > 100_000:
            break
    df = pd.concat(samples)
    print(f"Sampled {len(df)} rows from {args.csv}")

    for col in args.columns:
        if col not in df.columns:
            print(f"⚠️ Column {col!r} not found, skipping.")
            continue

        print(f"\n▶️ Testing column {col!r}:")
        norm = df[col].values
        # invert assumed log10: raw = 10**norm - 1
        raw = (10 ** norm) - 1
        logvals = np.log10(raw + 1)

        mm = minmax(logvals, args.xmin, args.xmax)
        zs = zscore(logvals, args.mean, args.std)

        mse_mm = mean_squared_error(norm, mm)
        mse_zs = mean_squared_error(norm, zs)

        print(f"  MSE(log→min-max) = {mse_mm:.6f}")
        print(f"  MSE(log→z-score) = {mse_zs:.6f}")

if __name__ == "__main__":
    main()
