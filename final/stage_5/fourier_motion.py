"""Sparse Fourier registration and split-track sensitivity diagnostics."""

import warnings
import cv2
import numpy as np
from stage_1.baselines import checked
from stage_2.registration import farneback, to_gray
from stage_2.warping import warp, validity, masked_mean
from .feature_tracking import trajectories, observations
from .fista import recover


def fit(tracks, h, w, step, lam, max_iter, subset):
    values, mask = observations(tracks, h, w, step)
    gh, gw = mask.shape[1:]
    y, x = np.indices((h, w), dtype=np.float32)
    maps = (x * ((gw - 1) / (w - 1)), y * ((gh - 1) / (h - 1)))
    flow = np.empty((len(tracks), h, w, 2), np.float32)
    history = []
    for axis in (0, 1):
        field, rows = recover(values[..., axis][mask], mask, lam, max_iter)
        history.extend(
            dict(
                subset=subset,
                axis=axis,
                tracks=tracks.shape[1],
                samples=int(mask.sum()),
                **r,
            )
            for r in rows
        )
        if not rows[-1]["converged"]:
            warnings.warn(
                f"Fourier {subset} axis {axis} reached iteration cap, stationarity {rows[-1]['stationarity']:.3g}"
            )
        for t, grid in enumerate(field):
            flow[t, :, :, axis] = cv2.remap(grid, *maps, cv2.INTER_LINEAR)
    return flow, history


def resample(frames, flow):
    valid = np.stack([validity(f) for f in flow])
    stack = np.stack([warp(im, f).clip(0, 1) for im, f in zip(frames, flow)])
    return stack, masked_mean(stack, valid, frames.mean(axis=0)), valid


def register(frames, step=8, lam=0.06, max_iter=500, diagnostics=False, refine=False):
    frames = checked(frames)
    if step < 1:
        raise ValueError("grid step must be positive")
    tracks = trajectories(frames)
    h, w = frames.shape[1:3]

    def estimate(part, label):
        return fit(part, h, w, step, lam, max_iter, label)

    flow, history = estimate(tracks, "all")
    stack, template, valid = resample(frames, flow)
    confidence = None
    if diagnostics and not refine and tracks.shape[1] >= 8:
        # Disjoint groups assess sensitivity to which features constrain the solution.
        # The half-difference has the same variance as the mean of two independent fits.
        groups = np.array_split(
            np.random.default_rng(0).permutation(tracks.shape[1]), 2
        )
        first, rows = estimate(tracks[:, groups[0]], "half_a")
        history.extend(rows)
        second, rows = estimate(tracks[:, groups[1]], "half_b")
        history.extend(rows)
        variance = ((first - second) ** 2).sum(axis=-1) * 0.25
        confidence = valid / (1 + variance)  # one pixel squared sets the scale
        confidence = confidence.astype(np.float32)
    if refine:
        # Local quadratic expansion estimates residual motion on the CS-registered frames.
        ref = to_gray(template[None])[0]
        delta = np.stack([farneback(ref, f, 21) for f in to_gray(stack)])
        delta -= delta.mean(axis=0)
        intermediate_valid = np.stack([validity(d) for d in delta])
        flow = np.stack([d + warp(f, d) for f, d in zip(flow, delta)])
        stack, template, valid = resample(frames, flow)
        valid &= intermediate_valid
        template = masked_mean(stack, valid, template)
    return stack, template, flow, history, valid, confidence
