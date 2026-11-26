"""Stage 2 tuning: pick one registration setting on the DEVELOPMENT set only.

    python scripts/tune_registration.py                  # development grid (default)
    python scripts/tune_registration.py --split test --flow-sigma 2 --average mean --rounds 3
                                                         # score ONE frozen setting on the test set, once

Data split (pipeline.md, "Data split rule"):
  development: MFIR shallow profile_6, backgrounds 0-2 (sorted order), all 4 amplitudes + all 14 James clips
  test:        MFIR shallow profile_6, backgrounds 8-15, all 4 amplitudes (never used for choices)

Grid: flow smoothing x averaging, each run for 4 rounds with early stopping off; the template after
every round is scored, so rounds 1-4 come from the same run. Scores: grayscale, crop 4, no LPIPS, and
the temporal median of the same frames as the reference. One setting for everything: no per-clip or
per-amplitude choices (a real video does not come labelled with its amplitude).

Writes results/tuning/<split>[_<tag>]_scores.csv (one row per clip x setting x round) and _summary.csv.
"""

import argparse
import itertools
import time

import pandas as pd

from unripple import data, paths
from unripple.baselines import temporal_median
from unripple.metrics import all_metrics
from unripple.registration import register

SPLITS = {"dev": range(0, 3), "test": range(8, 16)}
AMPLITUDES = ["low", "mid", "high", "extreme"]


def clips(split, frames):
    bgs = data.mfir_backgrounds()
    for profile in data.mfir_profiles("shallow"):
        for i in SPLITS[split]:
            for amp in AMPLITUDES:
                yield f"mfir_shallow_{amp}", f"{bgs[i][:-4]}_{profile}", \
                    lambda b=bgs[i], p=profile, a=amp: data.load_mfir(b, "shallow", p, a, L=frames)
    if split == "dev":  # all 14 James clips have been seen already, so they can only be development data
        for name in data.james_names():
            yield "james2019", name, lambda n=name: data.load_james(n, max_frames=frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    ap.add_argument("--frames", type=int, default=50, help="frames per clip (first N; James clips have 101, MFIR 199)")
    ap.add_argument("--flow-sigma", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0])
    ap.add_argument("--average", nargs="+", default=["mean", "median"], choices=["mean", "median", "trimmed"])
    ap.add_argument("--rounds", type=int, default=4, help="max rounds; every round count up to this is scored")
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the output files, so runs do not overwrite each other")
    args = ap.parse_args()

    grid = list(itertools.product(args.flow_sigma, args.average))
    if args.split == "test" and len(grid) > 1:
        ap.error("the test split scores one frozen setting: pass a single --flow-sigma and --average")

    out = paths.RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    rows, t_start = [], time.time()
    stem = args.split + (f"_{args.tag}" if args.tag else "")
    clip_list = list(clips(args.split, args.frames))
    for ci, (dataset, name, load) in enumerate(clip_list, 1):
        frames, gt = load()
        base = {"split": args.split, "dataset": dataset, "clip": name, "frames": len(frames)}
        m = all_metrics(temporal_median(frames), gt, with_lpips=False, crop=args.crop)
        rows.append({**base, "method": "median", "flow_sigma": None, "average": None, "rounds": 0, **m, "runtime_s": 0.0})
        for sigma, avg in grid:
            t = time.perf_counter()

            def score(k, template):
                rows.append({**base, "method": "registered", "flow_sigma": sigma, "average": avg, "rounds": k,
                             **all_metrics(template.clip(0, 1), gt, with_lpips=False, crop=args.crop),
                             "runtime_s": time.perf_counter() - t})

            register(frames, rounds=args.rounds, average=avg, flow_sigma=sigma, tol=0.0, on_round=score)
            r = rows[-1]
            print(f"[{ci}/{len(clip_list)}, {time.time() - t_start:6.0f} s] {dataset} {name} sigma={sigma} {avg:6s}: "
                  f"PSNR {r['psnr']:.2f} SSIM {r['ssim']:.3f} after {args.rounds} rounds (median {m['psnr']:.2f})", flush=True)
        pd.DataFrame(rows).to_csv(out / f"{stem}_scores.csv", index=False)  # saved after every clip

    df = pd.DataFrame(rows)
    reg = df[df.method == "registered"]
    med = df[df.method == "median"].set_index(["dataset", "clip"])[["psnr", "ssim"]]
    reg = reg.join(med, on=["dataset", "clip"], rsuffix="_median")
    reg["gain_psnr"], reg["gain_ssim"] = reg.psnr - reg.psnr_median, reg.ssim - reg.ssim_median
    keys = ["flow_sigma", "average", "rounds"]
    per_group = reg.groupby(keys + ["dataset"])[["psnr", "ssim", "gain_psnr", "gain_ssim"]].mean()
    # overall = mean over groups (each amplitude and James count equally, whatever their clip count)
    summary = per_group.groupby(keys).mean().add_prefix("overall_")
    wide = per_group["gain_psnr"].unstack("dataset").add_prefix("gain_psnr_")
    summary = summary.join(wide).join(per_group.groupby(keys)["gain_psnr"].min().rename("worst_group_gain_psnr"))
    summary = summary.sort_values("overall_gain_psnr", ascending=False)
    summary.round(4).to_csv(out / f"{stem}_summary.csv")
    print("\nSettings ranked by mean PSNR gain over the median (dB), averaged over groups:\n")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
