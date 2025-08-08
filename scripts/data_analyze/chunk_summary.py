#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to sample_train_total.csv")
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()

    cols = None
    agg = {}

    for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
        if cols is None:
            cols = chunk.columns.tolist()
            agg = {c: {"count":0, "sum":0.0, "sumsq":0.0, "min":np.inf, "max":-np.inf}
                   for c in cols}
        for c in cols:
            vals = chunk[c].dropna().astype(float).values
            cnt = vals.size
            s = vals.sum()
            ss = (vals*vals).sum()
            agg[c]["count"] += cnt
            agg[c]["sum"]   += s
            agg[c]["sumsq"] += ss
            agg[c]["min"]   = min(agg[c]["min"], vals.min())
            agg[c]["max"]   = max(agg[c]["max"], vals.max())

    # compute final stats
    print(f"{'column':30s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}")
    print("-"*70)
    for c, v in agg.items():
        n = v["count"]
        mean = v["sum"] / n
        var  = (v["sumsq"] / n) - mean*mean
        std  = np.sqrt(var)
        print(f"{c:30s} {v['min']:10.4f} {v['max']:10.4f} {mean:10.4f} {std:10.4f}")

if __name__ == "__main__":
    main()
