"""Stage 0 checks on the James et al. clips: ground-truth alignment and distortion level.

    python scripts/check_james.py

Alignment: search for the global translation that maximises PSNR between the ground truth and (a) the
temporal median of the wavy frames (the water's zero-mean distortion averages out, so the median sits on
the true geometry up to blur) and (b) the sharper registered output from run_registration.py, if present.
Integer grid of +-max_shift px, then a 0.25 px refinement around the best. A best shift well above ~0.5 px
that both images agree on means the ground truth is offset, and the scores of every method on that clip
are biased; the PSNR gain from undoing it is reported. (Phase correlation was tried first and was
unreliable here: the median is blurry and Checker/Bricks are periodic.)

Distortion level: mean Farneback flow magnitude (px) from the median template to each frame, the same
flow as registration round 1. Also the 95th percentile, since the mean hides local extremes.

Writes results/james_check.csv.
"""

import cv2
import numpy as np
import pandas as pd

from unripple import data, paths
from unripple.baselines import temporal_median
from unripple.metrics import psnr
from unripple.registration import DEFAULT_WINSIZES, farneback, to_gray


def shift(im, dx, dy):
    """out(x) = im(x - (dx, dy))."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, m, im.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def best_shift(im, gt, crop, max_shift=10):
    """(dx, dy, psnr) maximising psnr(shift(im, dx, dy), gt) on the region `crop` px inside the border."""
    c = (slice(crop, -crop), slice(crop, -crop))
    score = lambda dx, dy: psnr(shift(im, dx, dy)[c], gt[c])
    cands = [(dx, dy) for dx in range(-max_shift, max_shift + 1) for dy in range(-max_shift, max_shift + 1)]
    bx, by = max(cands, key=lambda d: score(*d))
    for step in (0.5, 0.25):  # refine around the integer optimum
        cands = [(bx + i * step, by + j * step) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        bx, by = max(cands, key=lambda d: score(*d))
    return bx, by, score(bx, by)


def main(crop=16):
    rows = []
    for name in data.james_names():
        frames, gt = data.load_james(name)
        g, gt_g = to_gray(frames), to_gray(gt[None])[0]
        if g.shape[1:] != gt_g.shape:
            print(f"{name}: frames {g.shape[1:]} vs gt {gt_g.shape}, size mismatch, skipped")
            rows.append({"clip": name, "frames_hw": g.shape[1:], "gt_hw": gt_g.shape})
            continue
        med = temporal_median(g)

        c = (slice(crop, -crop), slice(crop, -crop))
        dx, dy, med_best = best_shift(med, gt_g, crop)
        reg_png = paths.RESULTS / "registration" / "james2019" / f"{name}_registered.png"
        if reg_png.exists():
            reg = data.read_image(reg_png)
            rdx, rdy, reg_best = best_shift(reg, gt_g, crop)
            reg_cols = {"reg_shift_dx": rdx, "reg_shift_dy": rdy, "reg_psnr": psnr(reg[c], gt_g[c]), "reg_psnr_aligned": reg_best}
        else:
            reg_cols = {}

        mags = np.concatenate([np.linalg.norm(farneback(med, f, DEFAULT_WINSIZES[0]), axis=-1).ravel() for f in g])
        rows.append({"clip": name, "frames": len(g), "frames_hw": g.shape[1:], "gt_hw": gt_g.shape,
                     "med_shift_dx": dx, "med_shift_dy": dy, "med_psnr": psnr(med[c], gt_g[c]), "med_psnr_aligned": med_best,
                     **reg_cols,
                     "flow_mean_px": float(mags.mean()), "flow_p95_px": float(np.percentile(mags, 95))})
        r = rows[-1]
        msg = f"{name:12s} median: shift ({dx:+.2f}, {dy:+.2f}) PSNR {r['med_psnr']:.2f} -> {med_best:.2f}"
        if reg_cols:
            msg += f" | registered: shift ({rdx:+.2f}, {rdy:+.2f}) PSNR {r['reg_psnr']:.2f} -> {reg_best:.2f}"
        print(msg + f" | flow mean {r['flow_mean_px']:.2f} px, p95 {r['flow_p95_px']:.2f} px")

    df = pd.DataFrame(rows)
    out = paths.RESULTS / "james_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
