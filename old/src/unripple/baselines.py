"""Step 1 of the proposal: temporal mean and median ("long exposure") baselines."""

import numpy as np


def temporal_mean(frames):
    """frames: (T, H, W[, C]) array. Returns float32 image in the input's value range."""
    return np.asarray(frames, dtype=np.float32).mean(axis=0)


def temporal_median(frames):
    return np.median(np.asarray(frames, dtype=np.float32), axis=0)
