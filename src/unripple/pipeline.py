"""One entry point for every method, with the frozen settings (pipeline.md stages 1-4; nothing tuned here).

    image, info = restore(frames, "realign_fusion")

Methods:
  mean            stage 1, temporal mean of the raw frames
  median          stage 1, temporal median of the raw frames
  register        stage 2, iterative Farneback registration; output = final template (mean of registered frames)
  rpca            stage 3, low-rank part of the registered stack (GPU when available, = numpy to 1e-6)
  fusion          stage 4, lucky patch fusion of the registered stack (frozen: flow-stability score, 32 px, 5 %)
  realign_fusion  stage 4 as adopted (final method): 3 rounds of per-patch realignment, then fusion

The registered stack is rounded to uint8 and the flows to 1/4 resolution float16 before fusion, exactly as in the
registration cache and scripts/run_sweep.py, so outputs match every reported fusion number.
info = {"times": {stage: seconds}, "device": ...}.
"""

import time

import cv2
import numpy as np

from .baselines import temporal_mean, temporal_median
from .fusion import fuse_from_terms, score_terms
from .realign import realign
from .registration import register
from .rpca import rpca_image

METHODS = ["mean", "median", "register", "rpca", "fusion", "realign_fusion"]
FINAL = "realign_fusion"
# Frozen stage 4 setting (= DEFAULT in scripts/run_fusion.py; sharpen 0)
FUSION = dict(patch=32, frac=0.05, weights={"sharp": 0.0, "sim": 0.0, "flow": 1.0})
REALIGN_ITERS = 3
FLOW_STEP = 4


def estimate_distortion(frames, every=4, border=16):
    """Ground-truth-free distortion estimate (px): mean magnitude of the frozen first-round registration flow
    (median template -> frame, winsize 31, flow sigma 8) over every `every`-th frame, `border` px dropped.
    Underestimates when the water folds the image (no flow describes such frames)."""
    from .registration import DEFAULT_WINSIZES, farneback, to_gray
    gray = to_gray(np.asarray(frames, dtype=np.float32))
    med = np.median(gray, axis=0)
    mags = [np.linalg.norm(farneback(med, g, DEFAULT_WINSIZES[0], 8.0), axis=-1)[border:-border, border:-border]
            for g in gray[::every]]
    return float(np.mean(mags))


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else None
    except ImportError:
        return None


def _sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def restore(frames, method=FINAL, device="auto", methods=None):
    """frames (T, H, W[, 3]) float in [0, 1]. Returns (image, info) for one method, or ({method: image}, info)
    when `methods` (a list) is given; shared stages (registration) then run once."""
    want = list(methods) if methods is not None else [method]
    bad = set(want) - set(METHODS)
    if bad:
        raise ValueError(f"unknown method(s) {sorted(bad)}; choose from {METHODS}")
    device = default_device() if device == "auto" else device
    frames = np.asarray(frames, dtype=np.float32)
    times, out = {}, {}

    def timed(name, fn):
        t = time.perf_counter()
        r = fn()
        _sync(device)
        times[name] = time.perf_counter() - t
        return r

    if "mean" in want:
        out["mean"] = timed("mean", lambda: temporal_mean(frames))
    if "median" in want:
        out["median"] = timed("median", lambda: temporal_median(frames))
    if not set(want) & {"register", "rpca", "fusion", "realign_fusion"}:
        return (out if methods is not None else out[method]), {"times": times, "device": device}

    registered, tmpl, flows, _ = timed("registration", lambda: register(frames))
    out["register"] = tmpl
    if "rpca" in want:
        out["rpca"] = timed("rpca", lambda: rpca_image(registered, device=device)[0])
    if set(want) & {"fusion", "realign_fusion"}:
        h, w = flows.shape[1:3]
        small = np.stack([cv2.resize(f, (w // FLOW_STEP, h // FLOW_STEP), interpolation=cv2.INTER_AREA)
                          for f in flows]).astype(np.float16).astype(np.float32)
        reg8 = np.clip(registered * 255 + 0.5, 0, 255).astype(np.uint8)
        p, frac, wts = FUSION["patch"], FUSION["frac"], FUSION["weights"]
        if "fusion" in want:
            out["fusion"] = timed("fusion", lambda: fuse_from_terms(reg8, score_terms(reg8, tmpl, small, p), wts, p, frac))
        if "realign_fusion" in want:
            ra, ref, _ = timed("realign", lambda: realign(reg8, iters=REALIGN_ITERS))
            out["realign_fusion"] = timed("realign_fusion_fuse",
                                          lambda: fuse_from_terms(ra, score_terms(ra, ref, small, p), wts, p, frac))
    out = {m: np.clip(out[m], 0, 1).astype(np.float32) for m in want}
    return (out if methods is not None else out[method]), {"times": times, "device": device}
