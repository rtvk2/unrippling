"""Register every clip once with the frozen stage 2 setting and cache the result for stage 4 (fusion).

    python scripts/cache_registered.py                 # all development clips (same 29 as run_rpca.py)
    python scripts/cache_registered.py --split test    # test split (MFIR backgrounds 8-15)

Fusion needs the whole registered stack and the flows, not just the template, and it is tuned with grids,
so registering once and reusing the stack keeps each fusion run to seconds per clip. Per clip, writes
<paths.CACHE>/<dataset>/<clip>/ with
  registered.npy  (T, H, W, 3) uint8   registered frames (8-bit, like the input videos)
  template.npy    (H, W, 3) float32    final template = registered mean (the stage 2 output)
  flows.npy       (T, h, w, 2) float16 last-round flow, downsampled by FLOW_STEP (the flow is Gaussian-smoothed
                                       with sigma 8 px, so nothing is lost), values in full-resolution pixels
  gt.npy          (H, W, 3) float32
Clips already in the cache are skipped. The cache lives outside OneDrive (paths.CACHE, ~150 MB per clip).
"""

import argparse
import time

import cv2
import numpy as np

from run_baselines import clips
from tune_registration import clips as split_clips
from unripple import paths
from unripple.registration import register

FLOW_STEP = 4


def cache_dir(dataset, name, split=None):
    return paths.CACHE / ("test" if split == "test" else "dev") / dataset / name


def load_cached(dataset, name, split=None):
    """Returns (registered uint8 (T,H,W,3), template float32, flows float32 at 1/FLOW_STEP resolution, gt)."""
    d = cache_dir(dataset, name, split)
    return (np.load(d / "registered.npy", mmap_mode="r"), np.load(d / "template.npy"),
            np.load(d / "flows.npy").astype(np.float32), np.load(d / "gt.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test"])
    ap.add_argument("--datasets", nargs="+", help="only dataset names starting with one of these")
    args = ap.parse_args()

    clip_list = list(split_clips("test", 199) if args.split else clips(["low", "mid", "high", "extreme"]))
    t0 = time.time()
    for i, (dataset, name, load) in enumerate(clip_list, 1):
        if args.datasets and not dataset.startswith(tuple(args.datasets)):
            continue
        d = cache_dir(dataset, name, args.split)
        if (d / "gt.npy").exists():
            print(f"[{i}/{len(clip_list)}] {dataset} {name}: cached, skipped", flush=True)
            continue
        frames, gt = load()
        t = time.perf_counter()
        registered, template, flows, _ = register(frames)  # frozen stage 2 defaults
        reg_s = time.perf_counter() - t
        h, w = flows.shape[1:3]
        small = np.stack([cv2.resize(f, (w // FLOW_STEP, h // FLOW_STEP), interpolation=cv2.INTER_AREA) for f in flows])
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "registered.npy", np.clip(registered * 255 + 0.5, 0, 255).astype(np.uint8))
        np.save(d / "template.npy", template.astype(np.float32))
        np.save(d / "flows.npy", small.astype(np.float16))
        np.save(d / "gt.npy", np.asarray(gt, dtype=np.float32))  # written last: marks the clip as complete
        print(f"[{i}/{len(clip_list)}, {time.time() - t0:6.0f} s] {dataset} {name}: {frames.shape}, "
              f"registration {reg_s:.1f} s", flush=True)


if __name__ == "__main__":
    main()
