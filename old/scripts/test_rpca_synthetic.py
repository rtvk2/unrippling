"""Stage 3 check: the RPCA solver recovers a known low-rank + sparse split (pipeline.md asks for error < 1e-3).

    python scripts/test_rpca_synthetic.py

Setup as in Candes et al. (2011): L0 = X Y^T with N(0, 1/n) factors, S0 with a fraction of entries set to
+-1, lambda = 1/sqrt(max(n, T)). Square cases match the paper; tall-thin cases have the shape of a
registered stack (many pixels, few frames). Writes results/rpca_synthetic/metrics.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd

from unripple import paths
from unripple.rpca import rpca

# (n, T, rank, sparse fraction)
CASES = [(500, 500, 25, 0.05), (500, 500, 50, 0.10), (20000, 100, 5, 0.05), (20000, 100, 5, 0.10), (100000, 200, 3, 0.05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for n, T, r, frac in CASES:
        L0 = rng.standard_normal((n, r)) @ rng.standard_normal((r, T)) / np.sqrt(n)
        S0 = np.where(rng.random((n, T)) < frac, rng.choice([-1.0, 1.0], (n, T)), 0.0)
        for dtype, tol in [(np.float64, 1e-7), (np.float32, 1e-6)]:
            t = time.perf_counter()
            L, S, hist = rpca((L0 + S0).astype(dtype), tol=tol)
            row = {"n": n, "T": T, "rank": r, "sparse_frac": frac, "dtype": np.dtype(dtype).name,
                   "iters": len(hist), "rank_L": hist[-1]["rank"],
                   "err_L": np.linalg.norm(L - L0) / np.linalg.norm(L0),
                   "err_S": np.linalg.norm(S - S0) / np.linalg.norm(S0),
                   "runtime_s": time.perf_counter() - t}
            row["pass"] = bool(row["err_L"] < 1e-3)
            rows.append(row)
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    out = paths.RESULTS / "rpca_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print(f"\n{df['pass'].sum()}/{len(df)} cases recover L with relative error < 1e-3")


if __name__ == "__main__":
    main()
