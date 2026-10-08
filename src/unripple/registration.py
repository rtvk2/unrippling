"""Step 2 of the proposal: iterative registration of every frame to a template (Farneback flow + remap).

Each frame is the true scene warped by an unknown smooth flow. Estimate the flow from the template to
every frame, pull the frame back onto the template grid, average, and repeat.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Coarse-to-fine: a large window early (the template is still blurry), a smaller one later.
DEFAULT_WINSIZES = (31, 21, 15, 11)
FARNEBACK = dict(pyr_scale=0.5, levels=4, iterations=3, poly_n=7, poly_sigma=1.5)


def to_gray(frames):
    """(T, H, W[, 3]) float in [0, 1] -> (T, H, W) float32."""
    frames = np.asarray(frames, dtype=np.float32)
    return frames if frames.ndim == 3 else frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _u8(im):
    return np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)


def farneback(template, frame, winsize, smooth_sigma=0.0):
    """Flow (H, W, 2) with template(x) ~ frame(x + flow(x)); inputs are float grayscale in [0, 1]."""
    flow = cv2.calcOpticalFlowFarneback(_u8(template), _u8(frame), None, winsize=winsize, flags=0, **FARNEBACK)
    if smooth_sigma > 0:  # water surfaces are smooth
        flow = cv2.GaussianBlur(flow, (0, 0), smooth_sigma)
    return flow


def warp(frame, flow):
    """Backward map onto the template grid: out(x) = frame(x + flow(x)). No holes; the border is replicated."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(np.asarray(frame, dtype=np.float32), gx + flow[..., 0], gy + flow[..., 1],
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def robust_average(stack, how="mean", trim=0.1):
    """Collapse a (T, H, W[, C]) stack: 'mean', 'median' or 'trimmed' (drops the lowest and highest `trim` fraction)."""
    if how == "mean":
        return stack.mean(axis=0)
    if how == "median":
        return np.median(stack, axis=0)
    if how == "trimmed":
        k = int(len(stack) * trim)
        s = np.sort(stack, axis=0)
        return s[k:len(stack) - k].mean(axis=0)
    raise ValueError(f"unknown average '{how}'")


def register(frames, rounds=4, winsizes=DEFAULT_WINSIZES, init="median", average="mean", trim=0.1,
             flow_sigma=0.0, tol=1e-3, gt=None, workers=None, verbose=False):
    """Iteratively register `frames` (T, H, W[, 3], floats in [0, 1]) to their own template.

    Flow is computed on grayscale and applied to every channel. Returns (registered, template, flows, history):
      registered: (T, H, W[, 3]) float32     template: (H, W[, 3]) float32
      flows:      (T, H, W, 2) float32, the flow of the last round (frame -> template grid)
      history:    one dict per round with mean/max flow magnitude (px), template change and,
                  when `gt` is given, template PSNR.
    Stops early once the template changes by less than `tol` (RMS, in [0, 1] units) between rounds.
    """
    frames = np.asarray(frames, dtype=np.float32)
    template = (np.median if init == "median" else np.mean)(frames, axis=0).astype(np.float32)
    winsizes = list(winsizes) + [winsizes[-1]] * max(0, rounds - len(winsizes))
    history, registered, flows = [], frames, None

    with ThreadPoolExecutor(workers) as pool:  # cv2 releases the GIL, so threads parallelise the flows
        for k in range(rounds):
            tmpl_gray = to_gray(template[None])[0]

            def one(frame):
                # Always register the ORIGINAL frame to the template: warps do not compound, so the
                # output is a single resampling of the input and does not get blurrier each round.
                flow = farneback(tmpl_gray, to_gray(frame[None])[0], winsizes[k], flow_sigma)
                return warp(frame, flow), flow

            results = list(pool.map(one, frames))
            registered = np.stack([r for r, _ in results])
            flows = np.stack([f for _, f in results]).astype(np.float32)
            new_template = robust_average(registered, average, trim).astype(np.float32)

            mag = np.linalg.norm(flows, axis=-1)
            row = {"round": k + 1, "winsize": winsizes[k], "mean_flow_px": float(mag.mean()),
                   "max_flow_px": float(mag.max()),
                   "template_change": float(np.sqrt(np.mean((new_template - template) ** 2)))}
            if gt is not None:
                from .metrics import psnr
                row["template_psnr"] = float(psnr(np.clip(new_template, 0, 1), gt))
            history.append(row)
            if verbose:
                print("  " + "  ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))
            template = new_template
            if row["template_change"] < tol:
                break

    return registered, template, flows, history
