"""Step 1 of the proposal: temporal mean and median ("long exposure") baselines."""

import numpy as np


def checked(frames):
    f = np.asarray(frames, dtype=np.float32)
    if f.ndim not in (3, 4) or (f.ndim == 4 and f.shape[-1] != 3) or not f.size:
        raise ValueError("expected nonempty (T,H,W) or (T,H,W,3) frames")
    if not np.isfinite(f).all():
        raise ValueError("frames must be finite")
    return f


def temporal_mean(frames):
    """frames: (T, H, W[, C]) array. Returns float32 image in the input's value range."""
    return checked(frames).mean(axis=0)


def temporal_median(frames):
    return np.median(checked(frames), axis=0)
