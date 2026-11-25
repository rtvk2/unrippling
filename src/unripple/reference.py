"""Experimental: a starting reference built by averaging pixel POSITIONS instead of pixel VALUES.

Diagnosis (pipeline.md, stage 7): at high/extreme distortion the median/mean image that registration starts from is a
smear, registration aligns every frame to that wrong picture, and the rounds cannot escape it (ocean bg10 p10
extreme: 5.4 px misalignment left from the median start, 1.7 px from a ground-truth start).

Idea (the core of stage 5 / James et al., without the sparse Fourier model): pick an anchor frame a and estimate the
flow f_t from it to every frame t, with a(x) ~ I_t(x + f_t(x)). The scene point shown at anchor pixel x appears at
x + f_t(x) in frame t; if the water's distortion is zero-mean over time, its true position is the average of those,
x + m(x) with m = mean_t f_t. So the true image J satisfies J(x + m(x)) ~ a(x), i.e. J(z) ~ a(z - m(z)) (backward
approximation of the forward warp): a sharp image at the mean positions. With several anchors the references are
averaged. Flows are computed at half resolution (the waves are smooth: 45-130 px wavelength) with a large capture range.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .registration import _u8, to_gray, warp

CAPTURE = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)


def _flow_half(a, b):
    """Flow at half resolution, upsampled to full resolution in full-resolution pixels: a(x) ~ b(x + f(x))."""
    h, w = a.shape
    s = lambda im: cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(_u8(s(a)), _u8(s(b)), None, **CAPTURE)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * 2


def mean_position_reference(frames, anchors=1, every=2, smooth_sigma=8.0, workers=None):
    """frames (T, H, W[, 3]) in [0, 1]. Returns (reference image like one frame, info dict).
    Anchors = the `anchors` frames closest (MSE) to the median image."""
    frames = np.asarray(frames, dtype=np.float32)
    gray = to_gray(frames)
    med = np.median(gray, axis=0)
    order = np.argsort(((gray - med) ** 2).mean(axis=(1, 2)))
    refs, shifts = [], []
    others = gray[::every]
    with ThreadPoolExecutor(workers) as pool:
        for a in order[:anchors]:
            flows = list(pool.map(lambda g: _flow_half(gray[a], g), others))
            m = np.mean(flows, axis=0).astype(np.float32)
            if smooth_sigma > 0:
                m = cv2.GaussianBlur(m, (0, 0), smooth_sigma)
            refs.append(warp(frames[a], -m))
            shifts.append(float(np.linalg.norm(m, axis=-1).mean()))
    return np.mean(refs, axis=0).astype(np.float32), {"anchors": [int(a) for a in order[:anchors]],
                                                      "mean_correction_px": shifts}
