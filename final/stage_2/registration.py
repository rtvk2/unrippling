"""Step 2 of the proposal: iterative registration of every frame to a template (Farneback flow + remap).

Each frame is the true scene warped by an unknown smooth flow. Estimate the flow from the template to
every frame, pull the frame back onto the template grid, average, and repeat.
"""

from concurrent.futures import ThreadPoolExecutor

from stage_1.baselines import checked
from .warping import warp, validity, masked_mean

import cv2
import numpy as np

# Coarse-to-fine: a large window early (the template is still blurry), a smaller one later.
DEFAULT_WINSIZES = (31, 21, 15, 11)
FARNEBACK = dict(pyr_scale=0.5, levels=4, iterations=3, poly_n=7, poly_sigma=1.5)


def to_gray(frames):
    """(T, H, W[, 3]) float in [0, 1] -> (T, H, W) float32."""
    frames = np.asarray(frames, dtype=np.float32)
    return (
        frames
        if frames.ndim == 3
        else frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    )


def _u8(im):
    return np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)


def farneback(template, frame, winsize, smooth_sigma=0.0):
    """Flow (H, W, 2) with template(x) ~ frame(x + flow(x)); inputs are float grayscale in [0, 1]."""
    flow = cv2.calcOpticalFlowFarneback(
        _u8(template), _u8(frame), None, winsize=winsize, flags=0, **FARNEBACK
    )
    if smooth_sigma > 0:  # water surfaces are smooth
        flow = cv2.GaussianBlur(flow, (0, 0), smooth_sigma)
    return flow


def robust_average(stack, how="mean", trim=0.1, valid=None, fallback=None):
    """Collapse a (T, H, W[, C]) stack: 'mean', 'median' or 'trimmed' (drops the lowest and highest `trim` fraction)."""
    if how not in ("mean", "median", "trimmed") or not 0 <= trim < 0.5:
        raise ValueError("invalid average or trim fraction")
    if valid is None:
        valid = np.ones(stack.shape[:3], bool)
        fallback = stack[0]
    if how == "mean":
        return masked_mean(stack, valid, fallback)
    if stack.ndim == 4:
        valid = valid[..., None]
    count = valid.sum(axis=0)
    ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
    if how == "median":
        first = np.take_along_axis(ordered, ((count - 1) // 2).clip(0)[None], axis=0)[0]
        last = np.take_along_axis(ordered, (count // 2)[None], axis=0)[0]
        return np.where(count > 0, (first + last) / 2, fallback)
    cut = (trim * count).astype(int)
    rank = np.arange(len(stack)).reshape((-1,) + (1,) * (stack.ndim - 1))
    included = (rank >= cut) & (rank < count - cut)
    return np.divide(
        np.where(included, ordered, 0).sum(axis=0),
        count - 2 * cut,
        out=fallback.copy(),
        where=count > 0,
    )


# Frozen defaults (pipeline.md, stage 2 tuning; results/tuning/dev_summary.csv and dev_full_summary.csv, dev split only):
# mean beat median averaging in every setting; flow_sigma 8 beat 0/2/4 (50 frames) and 16 (199 frames), tied with 12
# (+3.39 vs +3.41 dB over the median) but with a better worst group; 4 rounds was best for sigma 8. tol=0 runs all rounds,
# as in the tuning.
def register(
    frames,
    rounds=4,
    winsizes=DEFAULT_WINSIZES,
    init="median",
    average="mean",
    trim=0.1,
    flow_sigma=8.0,
    tol=0.0,
    gt=None,
    workers=None,
    verbose=False,
    on_round=None,
    diagnostics=False,
):
    """Iteratively register `frames` (T, H, W[, 3], floats in [0, 1]) to their own template.

    Flow is computed on grayscale and applied to every channel. Returns (registered, template, flows, history):
      registered: (T, H, W[, 3]) float32     template: (H, W[, 3]) float32
      flows:      (T, H, W, 2) float32, the flow of the last round (template coordinates -> frame sampling locations)
      history:    one dict per round with mean/max flow magnitude (px), template change and,
                  when `gt` is given, template PSNR.
    Stops early once the template changes by less than `tol` (RMS, in [0, 1] units) between rounds.
    on_round(k, template), if given, is called after every round (k from 1) with that round's template,
    so one run can score every round count (used by scripts/tune_registration.py).
    """
    frames = checked(frames)
    if not winsizes or any(w < 3 for w in winsizes) or flow_sigma < 0 or tol < 0:
        raise ValueError("invalid flow window, smoothing or tolerance")
    if rounds < 1 or init not in ("median", "mean"):
        raise ValueError("positive rounds and mean/median initialization required")
    template = (np.median if init == "median" else np.mean)(frames, axis=0).astype(
        np.float32
    )
    winsizes = list(winsizes) + [winsizes[-1]] * max(0, rounds - len(winsizes))
    history, registered, flows = [], frames, None

    with ThreadPoolExecutor(
        workers
    ) as pool:  # cv2 releases the GIL, so threads parallelise the flows
        for k in range(rounds):
            tmpl_gray = to_gray(template[None])[0]

            def one(frame):
                # Always register the ORIGINAL frame to the template: warps do not compound, so the
                # output is a single resampling of the input and does not get blurrier each round.
                flow = farneback(
                    tmpl_gray, to_gray(frame[None])[0], winsizes[k], flow_sigma
                )
                return warp(frame, flow).clip(0, 1), flow

            results = list(pool.map(one, frames))
            registered = np.stack([r for r, _ in results])
            flows = np.stack([f for _, f in results]).astype(np.float32)
            valid = np.stack([validity(f) for f in flows])
            new_template = robust_average(registered, average, trim, valid, template)

            mag = np.linalg.norm(flows, axis=-1)
            row = {
                "round": k + 1,
                "winsize": winsizes[k],
                "mean_flow_px": float(mag.mean()),
                "max_flow_px": float(mag.max()),
                "template_change": float(
                    np.sqrt(np.mean((new_template - template) ** 2))
                ),
            }
            if gt is not None:
                from utils.metrics import psnr

                row["template_psnr"] = float(psnr(np.clip(new_template, 0, 1), gt))
            history.append(row)
            if verbose:
                print(
                    "  "
                    + "  ".join(
                        f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}"
                        for k_, v in row.items()
                    )
                )
            template = new_template
            if on_round is not None:
                on_round(k + 1, template)
            if row["template_change"] < tol:
                break

    if diagnostics:
        confidence = []
        for frame, flow, mask in zip(frames, flows, valid):
            reverse = farneback(
                to_gray(frame[None])[0], tmpl_gray, winsizes[k], flow_sigma
            )
            error = np.linalg.norm(flow + warp(reverse, flow), axis=-1)
            confidence.append(mask * np.exp(-0.5 * error**2))
        return registered, template, flows, history, valid, np.stack(confidence)
    return registered, template, flows, history
