"""Seeded smooth pull warps for known-motion validation."""

import cv2
import numpy as np
from stage_2.warping import warp


def smooth_flows(T, h, w, amp, corr, rng):
    """(T, H, W, 2) smooth random flows, RMS magnitude ~amp px, zero mean over t."""
    u = rng.standard_normal((T, h, w, 2)).astype(np.float32)
    u = np.stack([cv2.GaussianBlur(f, (0, 0), corr) for f in u])
    u -= u.mean(axis=0)
    return u * (amp / np.sqrt((u**2).sum(-1).mean()))


def inverse_flow(u, iters=10):
    """f with f(x) = -u(x + f(x)), by fixed-point iteration."""
    f = -u
    for _ in range(iters):
        f = -warp(u, f)
    return f
