"""Clean runtime table per stage (block E): one process, one clip at a time, nothing else running.

    python scripts/runtime_table.py            # 14 James clips + 4 MFIR clips; results/runtime/{runtime.csv,summary.csv}

Stages are timed inside unripple.pipeline.restore (all methods, frozen settings): mean, median, registration
(Farneback, CPU threads), rpca (GPU, torch), fusion, realign (3 rounds) and the fusion after realignment.
rpca_numpy is the CPU version (identical output to 1e-6) for comparison. Loading/rendering is not timed.
Final method time = registration + realign + realign_fusion_fuse.
"""

import os
import platform
import time
import warnings

import numpy as np
import pandas as pd

from unripple import data, paths
from unripple.pipeline import METHODS, default_device, restore
from unripple.registration import register
from unripple.rpca import rpca_image

OUT = paths.RESULTS / "runtime"


def clips():
    for n in data.james_names():
        yield "james", n, lambda n=n: data.load_james(n)[0]
    bg = data.mfir_backgrounds()[8]
    for w in ("shallow", "ocean", "ripple", "sine"):
        yield "mfir", f"{w}_profile_6_high", lambda w=w: data.load_mfir(bg, w, "profile_6", "high", L=199,
                                                                        device=default_device())[0]


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"CPU {platform.processor()} ({os.cpu_count()} threads), GPU {dev}", flush=True)
    rows = []
    restore(np.random.rand(10, 64, 64, 3).astype(np.float32), methods=METHODS)  # warm-up (CUDA init, LPIPS-free)
    for k, (ds, name, get) in enumerate(clips(), 1):
        frames = get()
        T, H, W = frames.shape[:3]
        _, info = restore(frames, methods=METHODS)
        rows.append({"dataset": ds, "clip": name, "frames": T, "height": H, "width": W,
                     "megapixel_frames": T * H * W / 1e6, **info["times"]})
        reg = register(frames)[0]  # numpy RPCA on a registered stack; registration itself is not counted
        t = time.perf_counter()
        rpca_image(reg)
        rows[-1]["rpca_numpy"] = time.perf_counter() - t
        r = rows[-1]
        r["final_method"] = r["registration"] + r["realign"] + r["realign_fusion_fuse"]
        pd.DataFrame(rows).to_csv(OUT / "runtime.csv", index=False)
        print(f"[{k}/18] {ds} {name} {T}x{H}x{W}: " + ", ".join(
            f"{c} {r[c]:.1f}" for c in ("registration", "rpca", "rpca_numpy", "fusion", "realign", "final_method")),
              flush=True)

    df = pd.DataFrame(rows)
    stages = ["mean", "median", "registration", "rpca", "rpca_numpy", "fusion", "realign", "realign_fusion_fuse",
              "final_method"]
    s = df.groupby("dataset")[stages + ["megapixel_frames"]].mean()
    per = df[stages].div(df["megapixel_frames"], axis=0).groupby(df["dataset"]).mean().add_suffix("_per_MPframe")
    out = pd.concat([s, per], axis=1).T.round(3)
    out.to_csv(OUT / "summary.csv")
    print("\nmean seconds per clip (and per megapixel-frame):\n", out.to_string())


if __name__ == "__main__":
    main()
