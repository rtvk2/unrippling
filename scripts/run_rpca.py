"""Stage 3: robust PCA of the registered stack (Oreifej-style baseline), compared against the registered mean.

    python scripts/run_rpca.py
    python scripts/run_rpca.py --datasets james2019 --max-frames 50
    python scripts/run_rpca.py --split test      # score once on the test split (MFIR backgrounds 8-15)

Registers every clip with the frozen stage 2 setting, runs inexact-ALM RPCA on the registered frames
(lambda = 1/sqrt(max(H*W*C, T)), no tuning) and scores the temporal mean of the low-rank part. Writes
results/rpca/<dataset>/<clip>_{rpca,sparse}.png (sparse = mean |S|, scaled to its maximum),
<clip>_comparison.png (input | median | registered | rpca | gt), rpca_iters.csv and metrics.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd

from run_baselines import clips
from run_registration import labelled
from tune_registration import clips as split_clips
from unripple import data, paths
from unripple.baselines import temporal_median
from unripple.metrics import all_metrics
from unripple.registration import register
from unripple.rpca import rpca_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--mfir-amplitudes", nargs="+", default=["low", "mid", "high", "extreme"])
    ap.add_argument("--datasets", nargs="+", help="only dataset names starting with one of these (e.g. james2019 mfir_shallow)")
    ap.add_argument("--max-frames", type=int, help="use only the first N frames (faster)")
    ap.add_argument("--lam", type=float, help="sparsity weight; default 1/sqrt(max(n, T))")
    ap.add_argument("--tol", type=float, default=1e-4, help="stopping tolerance; see rpca_image")
    ap.add_argument("--crop", type=int, default=4, help="border pixels removed before scoring, same as run_baselines.py")
    ap.add_argument("--tag", default="", help="write to results/rpca_<tag>/ so runs do not overwrite each other")
    ap.add_argument("--split", choices=["test"], help="use the frozen test split (pipeline.md data split rule) "
                    "instead of all development clips; writes results/rpca_test/")
    args = ap.parse_args()

    tag = args.tag or (args.split or "")
    out = paths.RESULTS / ("rpca" + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows, iters = [], []
    clip_iter = split_clips(args.split, args.max_frames or 199) if args.split else clips(args.mfir_amplitudes)
    for dataset, name, load in clip_iter:
        if args.datasets and not dataset.startswith(tuple(args.datasets)):
            continue
        frames, gt = load()
        if args.max_frames:
            frames = frames[:args.max_frames]
        print(f"{dataset} {name}: {frames.shape}", flush=True)

        t = time.perf_counter()
        registered, template, _, _ = register(frames)  # frozen stage 2 defaults
        reg_runtime = time.perf_counter() - t
        t = time.perf_counter()
        est, _, S, hist = rpca_image(registered, lam=args.lam, tol=args.tol)
        rpca_runtime = time.perf_counter() - t
        iters += [{"dataset": dataset, "clip": name, **h} for h in hist]
        print(f"  rpca: {len(hist)} iters, rank {hist[-1]['rank']}, sparsity {hist[-1]['sparsity']:.3f}, "
              f"residual {hist[-1]['residual']:.2e}, {rpca_runtime:.1f} s", flush=True)

        panels = [labelled(frames[0], "input (frame 0)")]
        for method, im, runtime in [("median", temporal_median(frames), 0.0), ("registered", template, reg_runtime),
                                    ("rpca", est, reg_runtime + rpca_runtime)]:
            m = all_metrics(im.clip(0, 1), gt, with_lpips=not args.no_lpips, crop=args.crop)
            row = {"dataset": dataset, "clip": name, "method": method, "frames": len(frames), **m, "runtime_s": runtime}
            if method == "rpca":
                row.update(rpca_iters=len(hist), rank=hist[-1]["rank"], sparsity=hist[-1]["sparsity"],
                           rpca_runtime_s=rpca_runtime)
            rows.append(row)
            print(f"  {method:10s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            panels.append(labelled(im.clip(0, 1), f"{method}  {m['psnr']:.2f} dB / SSIM {m['ssim']:.3f}"))
        sparse = np.abs(S).mean(axis=0)
        data.save_image(out / dataset / f"{name}_rpca.png", est.clip(0, 1))
        data.save_image(out / dataset / f"{name}_sparse.png", sparse / max(sparse.max(), 1e-12))
        data.save_image(out / dataset / f"{name}_comparison.png", np.concatenate(panels + [labelled(gt, "ground truth")], axis=1))
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)  # saved after every clip
        pd.DataFrame(iters).to_csv(out / "rpca_iters.csv", index=False)

    df = pd.DataFrame(rows)
    print("\nAverages:\n", df.groupby(["dataset", "method"])[["psnr", "ssim", "nmi", "rrmse", "runtime_s"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
