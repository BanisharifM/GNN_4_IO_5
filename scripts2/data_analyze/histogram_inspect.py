#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to sample_train_total.csv")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--frac", type=float, default=0.005,
                   help="Fraction to sample per chunk")
    args = p.parse_args()

    samps = []
    for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
        samps.append(chunk.sample(frac=args.frac, random_state=42))
    df = pd.concat(samps)

    inspect = ["nprocs", "POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN", "tag"]
    for col in inspect:
        plt.figure()
        df[col].hist(bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"hist_{col}.png")
        print(f"→ saved hist_{col}.png")

if __name__ == "__main__":
    main()
