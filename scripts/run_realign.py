"""Stage 4b prototype (dev only): per-patch realignment of the cached registered stacks, alone and with fusion.

    python scripts/run_realign.py [--datasets ...] [--no-lpips]
    python scripts/run_realign.py --split test    # frozen choice scored once: registered, fusion, realign_i3, realign3+fusion

Methods per clip (all scored like every other stage: gray, crop 4, LPIPS-VGG):
  registered            stage 2 output (mean of the registered stack)
  fusion                frozen stage 4 (flow score, patch 32, keep 5 %)
  extra_round_s{0,8}    CONTROL: one more Farneback round of the registered frames to the template (winsize 11,
                        flow smoothing 0 or 8) and the mean; realignment must beat this to be more than "more registration"
  realign_i{1,2,3}      variant A: mean of the realigned stack after 1-3 iterations
  realign3+fusion       variant B: frozen fusion setting on the realigned stack (flow score from the cached flows)
  realign3+ssd_f{05,20} variant B: score = -patch SSD to the realigned reference, keep 5 / 20 %
  oracle / realign3+oracle  oracle top-5 % on the registered / realigned stack (headroom left)
Writes results/realign/{metrics.csv,summary.csv} and per clip <dataset>/<clip>_comparison.png.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from cache_registered import load_cached
from diagnose_fusion import oracle_term
from explore_scores import ssd_term
from run_fusion import DEFAULT, clip_list
from run_registration import labelled
from unripple import data, paths
from unripple.fusion import fuse_from_terms, score_terms
from unripple.metrics import all_metrics
from unripple.realign import realign
from unripple.registration import farneback, to_gray, warp


def extra_round(reg, tmpl, sigma):
    t = to_gray(tmpl[None])[0]
    acc = np.zeros(reg.shape[1:], np.float64)
    for f in reg:
        f = f.astype(np.float32) / 255
        acc += warp(f, farneback(t, to_gray(f[None])[0], 11, sigma))
    return (acc / len(reg)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--split", choices=["test"])
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--crop", type=int, default=4)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = paths.RESULTS / ("realign_test" if args.split else "realign")
    full = not args.split  # the test split only gets the rows needed for the decision (no oracle, controls)
    out.mkdir(parents=True, exist_ok=True)
    p, frac, wts = DEFAULT["patch"], DEFAULT["frac"], DEFAULT["weights"]
    rows, t0 = [], time.time()
    cl = [c for c in clip_list(args.split) if not args.datasets or c[0].startswith(tuple(args.datasets))]
    for ci, (dataset, name) in enumerate(cl, 1):
        reg, tmpl, flows, gt = load_cached(dataset, name, args.split)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name}
        ims = {}

        def add(method, im, runtime=np.nan):
            im = np.clip(im, 0, 1)
            ims[method] = im
            rows.append({**base, "method": method, "runtime_s": runtime,
                         **all_metrics(im, gt, with_lpips=not args.no_lpips, crop=args.crop)})

        add("registered", tmpl)
        terms = score_terms(reg, tmpl, flows, p)
        add("fusion", fuse_from_terms(reg, terms, wts, p, frac))
        if full:
            add("oracle", fuse_from_terms(reg, {"o": oracle_term(reg, gt, p)}, {"o": 1}, p, frac, "uniform"))
        for sigma in ((0, 8) if full else ()):
            t = time.perf_counter()
            add(f"extra_round_s{sigma}", extra_round(reg, tmpl, sigma), time.perf_counter() - t)

        t = time.perf_counter()
        stacks = {}
        realign(reg, iters=3, on_iter=lambda k, st, ref: (add(f"realign_i{k}", ref, time.perf_counter() - t),
                                                         stacks.__setitem__(k, (st, ref))))
        ra, ref = stacks[3]
        info = None
        add("realign3+fusion", fuse_from_terms(ra, score_terms(ra, ref, flows, p), wts, p, frac))
        if full:
            sterm = {"s": ssd_term(ra, ref, p)}
            add("realign3+ssd_f05", fuse_from_terms(ra, sterm, {"s": 1}, p, 0.05))
            add("realign3+ssd_f20", fuse_from_terms(ra, sterm, {"s": 1}, p, 0.20))
            add("realign3+oracle", fuse_from_terms(ra, {"o": oracle_term(ra, gt, p)}, {"o": 1}, p, frac, "uniform"))
        del stacks, ra, info

        last = {r["method"]: r for r in rows if r["dataset"] == dataset and r["clip"] == name}
        show = ["registered", "fusion", "realign_i3", "realign3+fusion"]
        panels = [labelled(ims[m], f"{m}  {last[m]['psnr']:.2f} dB") for m in show]
        data.save_image(out / dataset / f"{name}_comparison.png", np.concatenate(panels + [labelled(gt, "ground truth")], 1))
        print(f"[{ci}/{len(cl)}, {time.time() - t0:5.0f} s] {dataset} {name}: " +
              "  ".join(f"{m} {last[m]['psnr']:.2f}" for m in last), flush=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    cols = [c for c in ["psnr", "ssim", "lpips_vgg"] if c in df]
    order = list(dict.fromkeys(df.method))
    summ = df.groupby(["dataset", "method"])[cols].mean().unstack("dataset").reindex(order)
    summ.round(4).to_csv(out / "summary.csv")
    print("\nMeans per group:\n", summ.round(3).T.to_string())


if __name__ == "__main__":
    main()
