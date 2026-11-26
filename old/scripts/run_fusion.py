"""Stage 4: lucky patch fusion on the cached registered stacks (run scripts/cache_registered.py first).

    python scripts/run_fusion.py                       # ablation ladder, development clips, default setting
    python scripts/run_fusion.py --grid                # tuning grid (dev only): patch size x frac x score weights
    python scripts/run_fusion.py --sharpen-grid        # unsharp amount sweep on the frozen setting (dev only)
    python scripts/run_fusion.py --split test          # score the frozen setting once on the test split

Ladder (pipeline.md, stage 4), all at the frozen patch size and frac: registered mean -> fusion (sharpness only)
-> + template similarity -> + flow stability, plus similarity only and flow only; 'fusion' is the frozen method
(= flow only), 'fusion_sharpen' is added only with --sharpen > 0. Median and RPCA rows are copied from
results/rpca[_test]/metrics.csv (+ lpips_check.csv), same clips and same frozen registration, not recomputed.
Scores: grayscale, crop 4, LPIPS-VGG (GPU if available), as every other stage.

Ladder writes results/fusion[_<tag>]/metrics.csv and per clip <dataset>/<clip>_{fusion,comparison}.png.
Grid writes results/fusion/grid_scores.csv and grid_summary.csv (gain over the registered mean in PSNR/SSIM,
averaged over groups, i.e. James and each MFIR amplitude count equally, plus the worst group).
"""

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from cache_registered import cache_dir, load_cached
from run_baselines import clips
from run_registration import labelled
from tune_registration import clips as split_clips
from unripple import data, paths
from unripple.fusion import fuse_from_terms, score_terms, unsharp
from unripple.metrics import all_metrics

# FROZEN setting (2026-11-19, dev grid results/fusion/grid_summary.csv + sharpen_summary.csv, rule in pipeline.md):
# flow stability only, 32 px patches, keep 5 % of the frames, no sharpening. Pre-tuning default was patch 32,
# frac 0.15, weights 1/1/1, sharpen 0.5 (results/fusion_default/).
DEFAULT = dict(patch=32, frac=0.05, weights={"sharp": 0.0, "sim": 0.0, "flow": 1.0}, sharpen=0.0)
# Ladder rungs, all at the frozen patch/frac: (name, weights, sharpen); weights None = the frozen/CLI weights,
# sharpen None = the CLI --sharpen amount (rung skipped when it is 0).
LADDER = [("fusion_sharp", {"sharp": 1.0}, 0.0),
          ("fusion_sharp_sim", {"sharp": 1.0, "sim": 1.0}, 0.0),
          ("fusion_sharp_sim_flow", {"sharp": 1.0, "sim": 1.0, "flow": 1.0}, 0.0),
          ("fusion_sim", {"sim": 1.0}, 0.0),
          ("fusion_flow", {"flow": 1.0}, 0.0),
          ("fusion", None, 0.0),
          ("fusion_sharpen", None, None)]
GRID_PATCH = [16, 32, 64]
GRID_FRAC = [0.05, 0.1, 0.2, 0.3]
GRID_WEIGHTS = [{"sharp": a, "sim": b, "flow": c} for a, b, c in itertools.product([0, 0.5, 1], [0, 1], [0, 0.5, 1])
                if a or b or c]


def clip_list(split):
    cl = split_clips("test", 199) if split else clips(["low", "mid", "high", "extreme"])
    for dataset, name, _ in cl:
        if (cache_dir(dataset, name, split) / "gt.npy").exists():
            yield dataset, name
        else:
            print(f"{dataset} {name}: not cached, skipped (run cache_registered.py)", flush=True)


def wname(w):
    return "+".join(f"{k}{v:g}" for k, v in w.items() if v) or "none"


def run_grid(args):
    out = paths.RESULTS / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    cl = list(clip_list(None))
    for ci, (dataset, name) in enumerate(cl, 1):
        reg, tmpl, flows, gt = load_cached(dataset, name)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name, "frames": len(reg)}
        lp = not args.no_lpips
        m = all_metrics(tmpl.clip(0, 1), gt, with_lpips=lp, crop=args.crop)
        rows.append({**base, "method": "registered", **m})
        for patch in GRID_PATCH:
            terms = score_terms(reg, tmpl, flows, patch)
            for frac, w in itertools.product(GRID_FRAC, GRID_WEIGHTS):
                im = fuse_from_terms(reg, terms, w, patch, frac).clip(0, 1)
                rows.append({**base, "method": "fusion", "patch": patch, "frac": frac, "weights": wname(w),
                             **all_metrics(im, gt, with_lpips=lp, crop=args.crop)})
        print(f"[{ci}/{len(cl)}, {time.time() - t0:6.0f} s] {dataset} {name}", flush=True)
        pd.DataFrame(rows).to_csv(out / "grid_scores.csv", index=False)
    summarize_grid(pd.DataFrame(rows), out, args.max_psnr_loss)


def run_sharpen_grid(args):
    """Unsharp-mask amount (sigma 1) on top of the setting given by --patch/--frac/--w-*; dev only, same rule."""
    out = paths.RESULTS / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    w = {"sharp": args.w_sharp, "sim": args.w_sim, "flow": args.w_flow}
    rows = []
    for dataset, name in clip_list(None):
        reg, tmpl, flows, gt = load_cached(dataset, name)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name, "frames": len(reg)}
        rows.append({**base, "method": "registered", **all_metrics(tmpl.clip(0, 1), gt, crop=args.crop)})
        im = fuse_from_terms(reg, score_terms(reg, tmpl, flows, args.patch), w, args.patch, args.frac).clip(0, 1)
        for amount in SHARPEN_GRID:
            s = unsharp(im, amount) if amount else im
            rows.append({**base, "method": "fusion", "patch": args.patch, "frac": args.frac, "weights": wname(w),
                         "sharpen": amount, **all_metrics(s, gt, crop=args.crop)})
        print(f"{dataset} {name}", flush=True)
    pd.DataFrame(rows).to_csv(out / "sharpen_scores.csv", index=False)
    summarize_grid(pd.DataFrame(rows), out, args.max_psnr_loss, stem="sharpen")


SHARPEN_GRID = [0.0, 0.1, 0.25, 0.5]


def summarize_grid(df, out, max_psnr_loss, stem="grid"):
    """Selection rule (decided 2026-11-18, before the grid): among settings whose PSNR loss against the registered
    mean is at most `max_psnr_loss` dB (mean over groups), pick the best mean LPIPS gain. Groups (James, each MFIR
    amplitude, demo) count equally. PSNR-only ranking would drift back to the registered mean (keep every frame)."""
    metrics = [c for c in ["psnr", "ssim", "lpips_vgg"] if c in df]
    reg = df[df.method == "registered"].set_index(["dataset", "clip"])[metrics]
    fu = df[df.method == "fusion"].join(reg, on=["dataset", "clip"], rsuffix="_reg")
    for c in metrics:  # gains: positive = better (LPIPS is lower-better, so its sign is flipped)
        fu[f"gain_{c}"] = (fu[c] - fu[f"{c}_reg"]) * (-1 if c == "lpips_vgg" else 1)
    keys = [k for k in ["patch", "frac", "weights", "sharpen"] if k in fu]
    gains = [f"gain_{c}" for c in metrics]
    per_group = fu.groupby(keys + ["dataset"])[metrics + gains].mean()
    summary = per_group.groupby(keys).mean().add_prefix("overall_")
    for g in gains:
        summary = summary.join(per_group.groupby(keys)[g].min().rename(f"worst_group_{g}"))
    summary = summary.join(per_group["gain_psnr"].unstack("dataset").add_prefix("gain_psnr_"))
    if "gain_lpips_vgg" in per_group:
        summary = summary.join(per_group["gain_lpips_vgg"].unstack("dataset").add_prefix("gain_lpips_"))
        summary["eligible"] = summary.overall_gain_psnr >= -max_psnr_loss
        summary = summary.sort_values(["eligible", "overall_gain_lpips_vgg"], ascending=False)
        print(f"\nSettings ranked by mean LPIPS gain over the registered mean, PSNR loss <= {max_psnr_loss} dB first:\n")
    else:
        summary = summary.sort_values("overall_gain_psnr", ascending=False)
        print("\nSettings ranked by mean PSNR gain over the registered mean (dB), no LPIPS:\n")
    summary.round(4).to_csv(out / f"{stem}_summary.csv")
    show = [c for c in summary if c.startswith(("overall_gain", "worst_group", "eligible"))]
    print(summary[show].head(25).round(3).to_string())


def run_ladder(args):
    out = paths.RESULTS / ("fusion" + (f"_{args.tag or args.split}" if (args.tag or args.split) else ""))
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = paths.RESULTS / ("rpca_test" if args.split else "rpca")
    ref = pd.read_csv(ref_dir / "metrics.csv") if (ref_dir / "metrics.csv").exists() else \
        pd.DataFrame(columns=["dataset", "clip", "method"])
    lp = ref_dir / "lpips_check.csv"  # the dev RPCA run used --no-lpips; its LPIPS came later from score_saved.py
    if "lpips_vgg" not in ref and lp.exists():
        ref = ref.merge(pd.read_csv(lp)[["dataset", "clip", "method", "lpips_vgg"]], how="left",
                        on=["dataset", "clip", "method"])
    w_full = {"sharp": args.w_sharp, "sim": args.w_sim, "flow": args.w_flow}
    rows, t0 = [], time.time()
    cl = list(clip_list(args.split))
    for ci, (dataset, name) in enumerate(cl, 1):
        reg, tmpl, flows, gt = load_cached(dataset, name, args.split)
        reg = np.asarray(reg)
        base = {"dataset": dataset, "clip": name, "frames": len(reg)}
        sel = (ref["dataset"] == dataset) & (ref["clip"] == name) & ref["method"].isin(["median", "rpca"])
        for _, r in ref[sel].iterrows():  # ref["clip"], not ref.clip (that is DataFrame.clip)
            rows.append({**base, "method": r["method"], **{k: r[k] for k in ["psnr", "ssim", "nmi", "rrmse", "lpips_vgg"]
                                                          if k in r.index and pd.notna(r[k])}})
        t = time.perf_counter()
        terms = score_terms(reg, tmpl, flows, args.patch)
        t_terms = time.perf_counter() - t
        images = [("registered", tmpl.clip(0, 1), 0.0)]
        for method, w, sharpen in LADDER:
            if sharpen is None and not args.sharpen:
                continue
            t = time.perf_counter()
            im = fuse_from_terms(reg, terms, w or w_full, args.patch, args.frac).clip(0, 1)
            if sharpen is None:
                im = unsharp(im, args.sharpen)
            images.append((method, im, t_terms + time.perf_counter() - t))
        panels = [labelled(reg[0].astype(np.float32) / 255, "registered frame 0")]
        rpca_png = ref_dir / dataset / f"{name}_rpca.png"
        scored = {r["method"]: r for r in rows[-2:] if r["dataset"] == dataset and r["clip"] == name}
        for method, im, runtime in images:
            m = all_metrics(im, gt, with_lpips=not args.no_lpips, crop=args.crop)
            rows.append({**base, "method": method, **m, "fusion_runtime_s": runtime})
            if method == "registered" and rpca_png.exists() and "rpca" in scored:
                panels.append(labelled(im, f"registered  {m['psnr']:.2f} dB / SSIM {m['ssim']:.3f}"))
                r = scored["rpca"]
                panels.append(labelled(data.read_image(rpca_png, gray=False), f"rpca  {r['psnr']:.2f} dB / SSIM {r['ssim']:.3f}"))
            elif method in ("registered", "fusion"):
                panels.append(labelled(im, f"{method}  {m['psnr']:.2f} dB / SSIM {m['ssim']:.3f}"))
        fused = dict((k, v) for k, v, _ in images)["fusion"]
        data.save_image(out / dataset / f"{name}_fusion.png", fused)
        data.save_image(out / dataset / f"{name}_comparison.png", np.concatenate(panels + [labelled(gt, "ground truth")], 1))
        last = rows[-len(images):]
        print(f"[{ci}/{len(cl)}, {time.time() - t0:5.0f} s] {dataset} {name}: " +
              "  ".join(f"{r['method'].replace('fusion_', 'f_')} {r['psnr']:.2f}" for r in last), flush=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    cols = [c for c in ["psnr", "ssim", "lpips_vgg"] if c in df]
    order = ["median", "registered", "rpca"] + [m for m, _, _ in LADDER if m in set(df["method"])]
    print("\nMeans per group:\n", df.groupby(["dataset", "method"])[cols].mean().unstack("dataset")
          .reindex(order).round(3).T.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="run the tuning grid (development split only)")
    ap.add_argument("--split", choices=["test"])
    ap.add_argument("--patch", type=int, default=DEFAULT["patch"])
    ap.add_argument("--frac", type=float, default=DEFAULT["frac"])
    ap.add_argument("--w-sharp", type=float, default=DEFAULT["weights"]["sharp"])
    ap.add_argument("--w-sim", type=float, default=DEFAULT["weights"]["sim"])
    ap.add_argument("--w-flow", type=float, default=DEFAULT["weights"]["flow"])
    ap.add_argument("--sharpen", type=float, default=DEFAULT["sharpen"], help="unsharp-mask amount for the last rung")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--max-psnr-loss", type=float, default=0.5,
                    help="grid selection: settings losing more PSNR (dB, mean over groups) than this are not eligible")
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sharpen-grid", action="store_true", help="unsharp amount sweep on the given setting (dev only)")
    args = ap.parse_args()
    if (args.grid or args.sharpen_grid) and args.split:
        ap.error("grids are for the development split only; the test split scores one frozen setting")
    run_grid(args) if args.grid else run_sharpen_grid(args) if args.sharpen_grid else run_ladder(args)


if __name__ == "__main__":
    main()
