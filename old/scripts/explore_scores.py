"""Stage 4 follow-up (dev only): candidate frame scores that target the residual misalignment the oracle measures.

    python scripts/explore_scores.py [--datasets ...]

Candidates (higher = better frame at a patch location, z-scored across frames):
  flow     the frozen score (little registration correction; a proxy, not the residual error)
  ssd      -patch SSD to the registered mean (template); unlike NCC it is not contrast-invariant
  ssd_med  -patch SSD to the per-pixel median of the registered stack
  resid    -patch mean |residual flow|, Farneback template -> registered frame, small window, no smoothing
  iterN    bootstrapped lucky imaging: fuse with ssd, then re-score by SSD to the fused image, N times
For each: per-location Spearman with the oracle ranking, top-5 % overlap, and fused PSNR/SSIM at a few fracs.
Writes results/fusion_explore/{scores,summary_psnr,summary_rank}.csv.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from cache_registered import load_cached
from diagnose_fusion import oracle_term, rank_stats
from run_fusion import clip_list
from unripple import paths
from unripple.fusion import _cells, _geometry, _zscore, fuse_from_terms, score_terms
from unripple.metrics import all_metrics
from unripple.registration import _u8, to_gray

FRACS = [0.05, 0.1, 0.2]


def ssd_term(reg, ref, patch, chunk=32):
    H, W = reg.shape[1:3]
    s, pads = _geometry(H, W, patch)
    r = to_gray(np.asarray(ref, np.float32)[None])
    out = [-_cells((to_gray(np.asarray(reg[a:a + chunk], np.float32) / 255) - r) ** 2, s, pads)
           for a in range(0, len(reg), chunk)]
    return _zscore(np.concatenate(out))


def resid_term(reg, tmpl, patch, winsize=11):
    H, W = reg.shape[1:3]
    s, pads = _geometry(H, W, patch)
    t = _u8(to_gray(tmpl[None])[0])
    fb = dict(pyr_scale=0.5, levels=3, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

    def one(f):
        g = _u8(to_gray(f[None].astype(np.float32) / 255)[0])
        return np.linalg.norm(cv2.calcOpticalFlowFarneback(t, g, None, winsize=winsize, **fb), axis=-1)

    with ThreadPoolExecutor() as pool:
        mag = np.stack(list(pool.map(one, reg)))
    return _zscore(-_cells(mag, s, pads))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--datasets", nargs="+")
    args = ap.parse_args()
    out = paths.RESULTS / "fusion_explore"
    out.mkdir(parents=True, exist_ok=True)
    rows, t0, p = [], time.time(), args.patch
    cl = [c for c in clip_list(None) if not args.datasets or c[0].startswith(tuple(args.datasets))]
    for ci, (dataset, name) in enumerate(cl, 1):
        reg, tmpl, flows, gt = load_cached(dataset, name)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name}
        m = lambda im: all_metrics(np.clip(im, 0, 1), gt, with_lpips=False, crop=args.crop)
        rows.append({**base, "method": "registered", **m(tmpl)})
        oracle = oracle_term(reg, gt, p)
        k5 = max(1, round(0.05 * len(reg)))
        cand = {"flow": score_terms(reg, tmpl, flows, p)["flow"],
                "ssd": ssd_term(reg, tmpl, p),
                "ssd_med": ssd_term(reg, np.median(reg, axis=0).astype(np.float32) / 255, p),
                "resid": resid_term(reg, tmpl, p)}
        cand["resid+ssd"] = (cand["resid"] + cand["ssd"]) / 2
        cand["flow+ssd"] = (cand["flow"] + cand["ssd"]) / 2
        ref = fuse_from_terms(reg, cand, {"ssd": 1.0}, p, 0.1)
        for it in range(1, args.iters + 1):  # bootstrap: the fused image is a sharper reference than the mean
            cand[f"iter{it}"] = ssd_term(reg, ref, p)
            ref = fuse_from_terms(reg, cand, {f"iter{it}": 1.0}, p, 0.1)
        for c, term in cand.items():
            rho, ov, chance = rank_stats(term, oracle, k5)
            rows.append({**base, "method": c, "spearman": rho, "top5pct_overlap": ov, "chance": chance})
            for f in FRACS:
                rows.append({**base, "method": c, "frac": f, **m(fuse_from_terms(reg, {c: term}, {c: 1.0}, p, f))})
        print(f"[{ci}/{len(cl)}, {time.time() - t0:5.0f} s] {dataset} {name}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    df = pd.DataFrame(rows)
    reg = df[df.method == "registered"].set_index(["dataset", "clip"])
    idx = df.set_index(["dataset", "clip"]).index
    for c in ["psnr", "ssim"]:
        df[f"gain_{c}"] = df[c] - idx.map(reg[c])
    fus = df[df.psnr.notna() & (df.method != "registered")]
    s1 = fus.groupby(["method", "frac", "dataset"])["gain_psnr"].mean().unstack("dataset")
    s1["mean"] = s1.mean(axis=1)
    s1["worst"] = s1.drop(columns="mean").min(axis=1)
    s2 = df[df.spearman.notna()].groupby(["method", "dataset"])[["spearman", "top5pct_overlap"]].mean() \
        .unstack("dataset")["top5pct_overlap"]
    print("\nPSNR gain over the registered mean (dB):\n", s1.round(2).to_string())
    print("\nTop-5 % overlap with the oracle (chance 0.05):\n", s2.round(3).to_string())
    s1.to_csv(out / "summary_psnr.csv"); s2.to_csv(out / "summary_rank.csv")


if __name__ == "__main__":
    main()
