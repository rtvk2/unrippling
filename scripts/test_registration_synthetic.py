"""Stage 2 sanity check: register frames warped by KNOWN smooth flows and see whether the flows and the image are recovered.

    python scripts/test_registration_synthetic.py
    python scripts/test_registration_synthetic.py --amplitudes 1 2 4 --rounds 6

Frame t is I(x + u_t(x)) with a smooth random u_t that averages to zero over t (otherwise the template is only
defined up to a global shift). Flows are compared with the true inverse flow f_t = -u_t(x + f_t), which is what
register() should return (frame -> template grid). Scores are cropped by --crop pixels, like every other score.

Writes results/synthetic/: metrics.csv, rounds.csv and, per amplitude, amp_<A>px/{input,frame0,mean,median,registered,
comparison}.png (input = the clean image the frames were made from, comparison = all of them side by side).
"""

import argparse

import cv2
import numpy as np
import pandas as pd
from skimage import data as skdata

from unripple import paths
from unripple.data import save_image
from unripple.baselines import temporal_mean, temporal_median
from unripple.metrics import psnr, ssim
from unripple.registration import register, warp


def smooth_flows(T, h, w, amp, corr, rng):
    """(T, H, W, 2) smooth random flows, RMS magnitude ~amp px, zero mean over t."""
    u = rng.standard_normal((T, h, w, 2)).astype(np.float32)
    u = np.stack([cv2.GaussianBlur(f, (0, 0), corr) for f in u])
    u -= u.mean(axis=0)
    return u * (amp / np.sqrt((u ** 2).sum(-1).mean()))


def inverse_flow(u, iters=10):
    """f with f(x) = -u(x + f(x)), by fixed-point iteration."""
    f = -u
    for _ in range(iters):
        f = -warp(u, f)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="RMS flow magnitude in px")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--corr", type=float, default=8.0, help="spatial correlation length of the flow (px)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--crop", type=int, default=16, help="border pixels ignored (flows of size amp move content across the edge)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img = cv2.resize(skdata.astronaut(), (args.size, args.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    c = args.crop
    cr = lambda a: a[c:-c, c:-c]

    out = paths.RESULTS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    rows, rounds = [], []
    for amp in args.amplitudes:
        u = smooth_flows(args.frames, args.size, args.size, amp, args.corr, rng)
        frames = np.stack([warp(img, f) for f in u])
        f_true = np.stack([inverse_flow(f) for f in u])

        reg, tmpl, f_est, hist = register(frames, rounds=args.rounds, gt=img)
        epe = np.linalg.norm(f_est - f_true, axis=-1)
        print(f"\n=== RMS flow {amp} px: {args.frames} frames {args.size}x{args.size} ===")
        print(" round  winsize  mean_flow  template_change  template_PSNR")
        for h in hist:
            print(f"  {h['round']:3d}  {h['winsize']:7d}  {h['mean_flow_px']:9.3f}  {h['template_change']:15.5f}  {h['template_psnr']:13.2f}")
        print(f" flow error (EPE, px): mean {cr(epe).mean():.3f}  (true flow magnitude {np.linalg.norm(cr(f_true), axis=-1).mean():.3f})")
        d = out / f"amp_{amp:g}px"
        results = [("input", img), ("frame0", frames[0])]
        rounds += [{"amp_px": amp, **h} for h in hist]
        for name, im in [("mean", temporal_mean(frames)), ("median", temporal_median(frames)), ("registered", tmpl)]:
            im = np.clip(im, 0, 1)
            p, s_ = psnr(cr(im), cr(img)), ssim(cr(im), cr(img))
            print(f" {name:10s} PSNR {p:6.2f}  SSIM {s_:.4f}")
            rows.append({"amp_px": amp, "method": name, "psnr": p, "ssim": s_,
                         "flow_epe_px": float(cr(epe).mean()) if name == "registered" else np.nan,
                         "true_flow_px": float(np.linalg.norm(cr(f_true), axis=-1).mean())})
            results.append((name, im))
        for name, im in results:
            save_image(d / f"{name}.png", im)
        save_image(d / "comparison.png", np.concatenate([im for _, im in results], axis=1))
        print(f" saved {d} (left to right: {', '.join(n for n, _ in results)})")

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(rounds).to_csv(out / "rounds.csv", index=False)


if __name__ == "__main__":
    main()
