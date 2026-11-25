"""Stage 4 diagnosis: is lucky patch fusion limited by the score or by the registered stack itself?

    python scripts/diagnose_fusion.py            # dev clips, PSNR/SSIM only (fast)

For every clip, an ORACLE score ranks each frame's patch by its true error against the ground truth (impossible
in practice, an upper bound on any frame-selection score). Oracle fusion at several keep fractions shows how much
headroom selection has on the registered stack. For each real score term, the per-location Spearman correlation
with the oracle ranking and the overlap of its top-k with the oracle top-k show how well it finds the good frames.
Writes results/fusion_diagnosis/{scores,summary}.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd

from cache_registered import load_cached
from run_fusion import clip_list
from unripple import paths
from unripple.fusion import _cells, _geometry, fuse_from_terms, score_terms
from unripple.metrics import all_metrics
from unripple.registration import to_gray

FRACS = [0.01, 0.05, 0.1, 0.2, 0.5]


def oracle_term(reg, gt, patch, chunk=32):
    """-(patch SSE vs gt) per frame per location (T, Ly, Lx); higher = better frame."""
    H, W = reg.shape[1:3]
    s, pads = _geometry(H, W, patch)
    g = to_gray(np.asarray(gt, np.float32)[None])
    out = []
    for a in range(0, len(reg), chunk):
        f = to_gray(np.asarray(reg[a:a + chunk], np.float32) / 255)
        out.append(-_cells((f - g) ** 2, s, pads))
    return np.concatenate(out)


def rank_stats(term, oracle, k):
    """Mean per-location Spearman correlation, and mean fraction of the term's top-k in the oracle top-k."""
    r = lambda x: np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
    a, b = r(term), r(oracle)
    a -= a.mean(0); b -= b.mean(0)
    rho = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-12)
    T = len(term)
    top = lambda x: np.argpartition(-x, k - 1, axis=0)[:k]
    mt, mo = np.zeros(term.shape, bool), np.zeros(term.shape, bool)
    np.put_along_axis(mt, top(term), True, 0); np.put_along_axis(mo, top(oracle), True, 0)
    return float(rho.mean()), float((mt & mo).sum(0).mean() / k), k / T  # last = chance overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--datasets", nargs="+")
    args = ap.parse_args()
    out = paths.RESULTS / "fusion_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    cl = [c for c in clip_list(None) if not args.datasets or c[0].startswith(tuple(args.datasets))]
    for ci, (dataset, name) in enumerate(cl, 1):
        reg, tmpl, flows, gt = load_cached(dataset, name)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name}
        m = lambda im: all_metrics(np.clip(im, 0, 1), gt, with_lpips=False, crop=args.crop)
        rows.append({**base, "method": "registered", **m(tmpl)})
        terms = score_terms(reg, tmpl, flows, args.patch)
        terms["oracle"] = oracle_term(reg, gt, args.patch)
        k5 = max(1, round(0.05 * len(reg)))
        for f in FRACS:
            for wt in ["uniform", "softmax"]:
                if wt == "softmax":  # oracle SSE has no unit scale: z-score it like the other terms
                    o = terms["oracle"]; t = (o - o.mean(0)) / (o.std(0) + 1e-8)
                    im = fuse_from_terms(reg, {"o": t}, {"o": 1.0}, args.patch, f, "softmax")
                else:
                    im = fuse_from_terms(reg, terms, {"oracle": 1.0}, args.patch, f, "uniform")
                rows.append({**base, "method": f"oracle_{wt}", "frac": f, **m(im)})
        im = fuse_from_terms(reg, terms, {"flow": 1.0}, args.patch, 0.05)
        rows.append({**base, "method": "fusion_flow", "frac": 0.05, **m(im)})
        for name_t in ["sharp", "sim", "flow"]:
            rho, ov, chance = rank_stats(terms[name_t], terms["oracle"], k5)
            rows.append({**base, "method": f"rank_{name_t}", "spearman": rho, "top5pct_overlap": ov, "chance": chance})
        print(f"[{ci}/{len(cl)}, {time.time() - t0:5.0f} s] {dataset} {name}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    df = pd.DataFrame(rows)
    df["frac"] = df.get("frac", np.nan)
    reg = df[df.method == "registered"].set_index(["dataset", "clip"])["psnr"]
    df["gain_psnr"] = df["psnr"] - df.set_index(["dataset", "clip"]).index.map(reg)
    fus = df[df.psnr.notna() & (df.method != "registered")]
    s1 = fus.groupby(["method", "frac", "dataset"])["gain_psnr"].mean().unstack("dataset")
    s2 = df[df.method.str.startswith("rank_")].groupby(["method", "dataset"])[["spearman", "top5pct_overlap", "chance"]].mean()
    print("\nPSNR gain over the registered mean (dB):\n", s1.round(2).to_string())
    print("\nAgreement of each score term with the oracle ranking:\n", s2.round(3).to_string())
    s1.to_csv(out / "summary_psnr.csv"); s2.to_csv(out / "summary_rank.csv")


if __name__ == "__main__":
    main()
