"""Mean and median baselines (proposal step 1) on every dataset that is present locally.

    python scripts/run_baselines.py                 # James et al. clips (+ MFIR / own if downloaded)
    python scripts/run_baselines.py --no-lpips      # skip LPIPS (faster)

Writes results/baselines/<dataset>/<clip>_{mean,median}.png and results/baselines/metrics.csv.
"""

import argparse
import time

import pandas as pd

from unripple import data, paths
from unripple.baselines import temporal_mean, temporal_median
from unripple.metrics import all_metrics


def clips(mfir_amplitudes):
    for name in data.james_names():
        yield "james2019", name, lambda n=name: data.load_james(n)
    for wave in ["ocean", "shallow", "sine", "ripple"]:
        for profile in data.mfir_profiles(wave):
            for bg in data.mfir_backgrounds()[:3]:
                for amp in mfir_amplitudes:
                    yield f"mfir_{wave}_{amp}", f"{bg[:-4]}_{profile}", \
                        lambda b=bg, w=wave, p=profile, a=amp: data.load_mfir(b, w, p, a)
    for name in data.own_names():
        yield "own", name, lambda n=name: data.load_own(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--mfir-amplitudes", nargs="+", default=["low", "mid"])
    args = ap.parse_args()

    out = paths.RESULTS / "baselines"
    rows = []
    for dataset, name, load in clips(args.mfir_amplitudes):
        frames, gt = load()
        for method, fn in [("mean", temporal_mean), ("median", temporal_median)]:
            t = time.perf_counter()
            est = fn(frames)
            runtime = time.perf_counter() - t
            data.save_image(out / dataset / f"{name}_{method}.png", est)
            m = all_metrics(est, gt, with_lpips=not args.no_lpips)
            rows.append({"dataset": dataset, "clip": name, "method": method, "frames": len(frames),
                         "size": "x".join(map(str, gt.shape[:2])), **m, "runtime_s": runtime})
            print(f"{dataset:22s} {name:28s} {method:6s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print("\nAverages:\n", df.groupby(["dataset", "method"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
