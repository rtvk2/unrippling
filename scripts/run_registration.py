"""Stage 2: iterative registration to a template, compared against the mean and median baselines.

    python scripts/run_registration.py --no-lpips
    python scripts/run_registration.py --rounds 5 --average trimmed --datasets james2019

Writes results/registration/<dataset>/<clip>_{mean,median,registered}.png, a per-round diagnostics table
(registration_rounds.csv) and the metrics (metrics.csv). Same clips as run_baselines.py.
"""

import argparse
import time

import pandas as pd

from run_baselines import clips
from unripple import data, paths
from unripple.baselines import temporal_mean, temporal_median
from unripple.metrics import all_metrics
from unripple.registration import register


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--mfir-amplitudes", nargs="+", default=["low", "mid"])
    ap.add_argument("--datasets", nargs="+", help="only dataset names starting with one of these (e.g. james2019 mfir_demo)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--init", choices=["mean", "median"], default="median")
    ap.add_argument("--average", choices=["mean", "median", "trimmed"], default="mean")
    ap.add_argument("--flow-sigma", type=float, default=0.0, help="Gaussian smoothing of the flow (px)")
    ap.add_argument("--max-frames", type=int, help="use only the first N frames (faster)")
    ap.add_argument("--crop", type=int, default=4, help="border pixels removed before scoring, same as run_baselines.py")
    args = ap.parse_args()

    out = paths.RESULTS / "registration"
    rows, rounds = [], []
    for dataset, name, load in clips(args.mfir_amplitudes):
        if args.datasets and not dataset.startswith(tuple(args.datasets)):
            continue
        frames, gt = load()
        if args.max_frames:
            frames = frames[:args.max_frames]
        print(f"{dataset} {name}: {frames.shape}")

        t = time.perf_counter()
        _, est, _, hist = register(frames, rounds=args.rounds, init=args.init, average=args.average,
                                   flow_sigma=args.flow_sigma, gt=gt, verbose=True)
        reg_runtime = time.perf_counter() - t
        rounds += [{"dataset": dataset, "clip": name, **h} for h in hist]

        for method, im, runtime in [("mean", temporal_mean(frames), 0.0), ("median", temporal_median(frames), 0.0),
                                    ("registered", est, reg_runtime)]:
            data.save_image(out / dataset / f"{name}_{method}.png", im)
            m = all_metrics(im.clip(0, 1), gt, with_lpips=not args.no_lpips, crop=args.crop)
            rows.append({"dataset": dataset, "clip": name, "method": method, "frames": len(frames),
                         **m, "runtime_s": runtime})
            print(f"  {method:10s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(rounds).to_csv(out / "registration_rounds.csv", index=False)
    print("\nAverages:\n", df.groupby(["dataset", "method"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
