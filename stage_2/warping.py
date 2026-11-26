"""Pull sampling on the reference lattice. Displacements are (dx, dy) in pixels."""

import cv2
import numpy as np


def coordinates(flow):
    y, x = np.indices(flow.shape[:2], dtype=np.float32)
    return x + flow[..., 0], y + flow[..., 1]


def validity(flow):
    x, y = coordinates(flow)
    h, w = flow.shape[:2]
    return (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)


def warp(frame, flow):
    return cv2.remap(
        np.asarray(frame, np.float32),
        *coordinates(flow),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def masked_mean(stack, valid, fallback):
    if stack.ndim == 4:
        valid = valid[..., None]
    count = valid.sum(axis=0)
    return np.divide(
        (stack * valid).sum(axis=0), count, out=fallback.copy(), where=count > 0
    )
