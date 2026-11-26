"""Dev test: does a mean-POSITION starting reference (unripple.reference) fix the high/extreme registration failure?

    python scripts/test_reference.py [--limit N]     # resumable; results/reference_test/{metrics.csv,summary.csv}

Dev clips only (profiles 1, 2 x background 3; not in the sweep, not in the benchmark's official subset):
high + extreme: 4 wave types x profiles 1, 2 (16 clips); low + mid: 4 wave types x profile 1 (8 clips). 199 frames.
Methods (scored like every stage: gray, crop 4, LPIPS-VGG):
  median            plain median
  reg               registration as frozen (starts from the median)
  ref_mp1 / ref_mp4 the mean-position reference itself, 1 or 4 anchors
  reg_mp1 / reg_mp4 frozen registration started from that reference
Decision rule (fixed before the run): worth pursuing (cheap version, then stage 5) if reg_mp* beats reg by >= 0.5 dB
PSNR or clearly on LPIPS at high/extreme, without losing > 0.05 dB / 0.002 LPIPS at low/mid.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from unripple import data, paths
from unripple.metrics import all_metrics
from unripple.reference import mean_position_reference
from unripple.registration import register

OUT = paths.RESULTS / "reference_test"


def jobs():
    bg = data.mfir_backgrounds()[3]
    j = [(w, p, a, bg) for a in ("high", "extreme") for w in ("shallow", "ocean", "ripple", "sine")
         for p in ("profile_1", "profile_2")]
    return j + [(w, "profile_1", a, bg) for a in ("low", "mid") for w in ("shallow", "ocean", "ripple", "sine")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "metrics.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["wave"], r["profile"], r["amplitude"]) for r in rows}
    todo = [j for j in jobs()[:args.limit] if j[:3] not in done]
    t0 = time.time()
    for k, (w, p, a, bg) in enumerate(todo, 1):
        frames, gt = data.load_mfir(bg, w, p, a, L=199, device="cuda")
        base = {"wave": w, "profile": p, "amplitude": a, "background": bg[:-4]}
        ims, rt = {"median": np.median(frames, 0)}, {}
        t = time.perf_counter()
        ims["reg"] = register(frames)[1]
        rt["reg"] = time.perf_counter() - t
        for n in (1, 4):
            t = time.perf_counter()
            ref, info = mean_position_reference(frames, anchors=n)
            rt[f"ref_mp{n}"] = time.perf_counter() - t
            ims[f"ref_mp{n}"] = ref
            t = time.perf_counter()
            ims[f"reg_mp{n}"] = register(frames, init=ref)[1]
            rt[f"reg_mp{n}"] = time.perf_counter() - t + rt[f"ref_mp{n}"]
        new = [{**base, "method": m, "runtime_s": rt.get(m, np.nan),
                **all_metrics(np.clip(im, 0, 1), gt, with_lpips=True, crop=4)} for m, im in ims.items()]
        rows += new
        pd.DataFrame(rows).to_csv(csv, index=False)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}, {el / 60:4.1f} min, ETA {el / k * (len(todo) - k) / 60:4.1f} min] {w} {p} {a}: " +
              "  ".join(f"{r['method']} {r['psnr']:.2f}/{r['lpips_vgg']:.3f}" for r in new), flush=True)

    df = pd.DataFrame(rows)
    order = ["median", "reg", "ref_mp1", "ref_mp4", "reg_mp1", "reg_mp4"]
    for col in ("psnr", "lpips_vgg"):
        s = df.pivot_table(index=["amplitude", "wave"], columns="method", values=col)[order]
        s.round(4).to_csv(OUT / f"summary_{col}.csv")
        print(f"\n{col}:\n", s.round(3 if "lpips" in col else 2).to_string())
    n = df.pivot_table(index=["wave", "profile", "amplitude"], columns="method", values="psnr")
    for m in ("reg_mp1", "reg_mp4"):
        print(f"{m} - reg PSNR by amplitude:", (n[m] - n["reg"]).groupby(level="amplitude").mean().round(2).to_dict())


if __name__ == "__main__":
    main()
