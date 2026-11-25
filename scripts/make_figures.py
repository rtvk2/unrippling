"""Block E3: full-resolution qualitative figures, GIFs and failure examples (frozen settings, nothing tuned).

    python scripts/make_figures.py            # resumable; results/figures/

Clips (all from the stage 7 sweep's test backgrounds or James; scores are the same as in the sweep):
  overview/  MFIR background 8, profile 6: every wave type x amplitude (16 clips)
  failure/   the sweep's worst clips for the final method vs the median (ripple/sine extreme), plus the
             ocean bg10 profile 10 extreme folding case analysed in pipeline.md stage 7
  james/     4 real tank clips (BlueTiles, Eye, Math, Middle)
Per clip, in results/figures/<group>/<clip>/:
  grid.png    full resolution, 2 x 3: input frame, median, registration / RPCA, final method, ground truth,
              each labelled with PSNR / LPIPS-VGG (our protocol: gray, crop 4)
  zoom.png    the same six panels cropped to the central quarter and enlarged 2x (fine detail)
  input_vs_output.gif   the distorted video next to the final output (and ground truth), 256 px tiles
  <method>.png          each output at full resolution
results/figures/index.csv lists every clip with its scores.
"""

import time
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from run_registration import labelled
from unripple import data, paths
from unripple.metrics import all_metrics
from unripple.pipeline import default_device, restore

OUT = paths.RESULTS / "figures"
SHOW = ["median", "register", "rpca", "realign_fusion"]
NAMES = {"median": "median", "register": "registration", "rpca": "RPCA", "realign_fusion": "final (realign+fusion)"}


def clips():
    bgs = data.mfir_backgrounds()
    for w in ("shallow", "ocean", "ripple", "sine"):
        for a in ("low", "mid", "high", "extreme"):
            yield "overview", (bgs[8], w, "profile_6", a)
    for b, w, p in [(10, "ripple", "profile_6"), (8, "ripple", "profile_10"), (10, "sine", "profile_6"),
                    (10, "ocean", "profile_10")]:
        yield "failure", (bgs[b], w, p, "extreme")
    for n in ("BlueTiles", "Eye", "Math", "Middle"):
        yield "james", n


def load(spec):
    if isinstance(spec, str):
        f, g = data.load_james(spec)
        return spec, f, g
    bg, w, p, a = spec
    f, g = data.load_mfir(bg, w, p, a, L=199, device=default_device())
    return f"{bg[:-4]}_{w}_{p}_{a}", f, g


def gif(path, frames, out, gt, size=256, n=60):
    def small(im):
        h, w = im.shape[:2]
        s = size / max(h, w)
        return cv2.resize(np.clip(im, 0, 1), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    right = [labelled(small(out), "final output"), labelled(small(gt), "ground truth")]
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    ims = [Image.fromarray((np.concatenate([labelled(small(frames[i]), "input video")] + right, 1) * 255 + .5)
                           .astype(np.uint8)) for i in idx]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=80, loop=0)


def main():
    warnings.filterwarnings("ignore")
    rows = []
    todo = list(clips())
    t0 = time.time()
    for k, (group, spec) in enumerate(todo, 1):
        name, frames, gt = load(spec)
        d = OUT / group / name
        csv = d / "scores.csv"
        if csv.exists():
            rows += pd.read_csv(csv).to_dict("records")
            continue
        ims, info = restore(frames, methods=SHOW)
        sc = {m: all_metrics(ims[m], gt, with_lpips=True, crop=4) for m in SHOW}
        mid = frames[len(frames) // 2]
        panels = [("input (one frame)", mid)] + \
                 [(f"{NAMES[m]}  {sc[m]['psnr']:.1f} dB / {sc[m]['lpips_vgg']:.3f}", ims[m]) for m in SHOW] + \
                 [("ground truth", gt)]
        lab = [labelled(im, t) for t, im in panels]
        grid = np.concatenate([np.concatenate(lab[:3], 1), np.concatenate(lab[3:], 1)], 0)
        h, w = gt.shape[:2]
        y, x = h // 4, w // 4
        zl = [labelled(cv2.resize(im[y:y + h // 2, x:x + w // 2], (w, h), interpolation=cv2.INTER_CUBIC), t)
              for t, im in panels]
        zoom = np.concatenate([np.concatenate(zl[:3], 1), np.concatenate(zl[3:], 1)], 0)
        d.mkdir(parents=True, exist_ok=True)
        data.save_image(d / "grid.png", grid)
        data.save_image(d / "zoom.png", np.clip(zoom, 0, 1))
        for m in SHOW:
            data.save_image(d / f"{m}.png", ims[m])
        gif(d / "input_vs_output.gif", frames, ims["realign_fusion"], gt)
        new = [{"group": group, "clip": name, "method": m, **sc[m]} for m in SHOW]
        pd.DataFrame(new).to_csv(csv, index=False)
        rows += new
        el = time.time() - t0
        print(f"[{k}/{len(todo)}, {el / 60:4.1f} min] {group} {name}: " +
              "  ".join(f"{m} {sc[m]['psnr']:.2f}/{sc[m]['lpips_vgg']:.3f}" for m in SHOW), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
