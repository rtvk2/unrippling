"""Stage 4b (experimental): per-patch realignment of the registered stack before averaging / fusion.

The fusion diagnosis (pipeline.md, block D) showed that picking the right frames per patch could gain 3-6 dB, but
no score finds them. Realignment fixes the frames instead of choosing them: for every patch location and frame,
estimate the small leftover shift between the frame's patch and a reference patch (phase correlation), and undo it.

Patch grid as in fusion.py (patch p, stride s = p/2, reflect padding by s): patch (i, j) is centred on original
pixel (i*s, j*s). Shifts on that grid are interpolated bilinearly to a dense per-pixel field, and each frame is
resampled once with it (equivalent to shifting each patch and Hann-blending, without seams, and the result can go
straight into fusion.fuse_from_terms). Unreliable shifts (weak correlation peak, e.g. flat regions) and shifts
larger than `max_shift` are set to 0. Iterating re-estimates the shifts of the ORIGINAL registered frames against
the newest reference (the realigned mean), so resamplings do not compound.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .fusion import _geometry, _hann
from .registration import to_gray


def _phase_shifts(frames, ref, patch, lowpass=0.25):
    """frames (T, H, W) and ref (H, W) float gray. Returns shifts (T, Ly, Lx, 2) as (dx, dy) with
    frame(x) ~ ref(x - d), and the normalised peak height (T, Ly, Lx) as a confidence."""
    T, H, W = frames.shape
    s, pads = _geometry(H, W, patch)
    fp = np.pad(frames, ((0, 0),) + pads, mode="reflect")
    rp = np.pad(ref, pads, mode="reflect")
    win = _hann(patch)
    fy = np.fft.fftfreq(patch)[:, None]; fx = np.fft.fftfreq(patch)[None, :]
    lp = np.exp(-(fx ** 2 + fy ** 2) / (2 * lowpass ** 2)).astype(np.float32)  # down-weight noisy high frequencies
    Ly, Lx = (fp.shape[1] - patch) // s + 1, (fp.shape[2] - patch) // s + 1
    shifts = np.zeros((T, Ly, Lx, 2), np.float32)
    conf = np.zeros((T, Ly, Lx), np.float32)
    for i in range(Ly):
        y = i * s
        P = sliding_window_view(fp[:, y:y + patch], (patch, patch), axis=(1, 2))[:, 0, ::s][:, :Lx]  # (T, Lx, p, p)
        R = sliding_window_view(rp[y:y + patch], (patch, patch))[0, ::s][:Lx]  # (Lx, p, p)
        P = (P - P.mean(axis=(-2, -1), keepdims=True)) * win
        R = (R - R.mean(axis=(-2, -1), keepdims=True)) * win
        X = np.fft.fft2(P) * np.conj(np.fft.fft2(R))[None]
        X = X / (np.abs(X) + 1e-9) * lp
        c = np.fft.ifft2(X).real / lp.mean()  # peak 1 for a pure shift (ifft2 already divides by p*p)
        flat = c.reshape(T, Lx, -1)
        k = flat.argmax(-1)
        py, px = np.unravel_index(k, (patch, patch))
        t_idx, l_idx = np.meshgrid(np.arange(T), np.arange(Lx), indexing="ij")
        peak = c[t_idx, l_idx, py, px]

        def sub(dm, d0, dp):  # parabolic sub-pixel refinement
            den = dm - 2 * d0 + dp
            return np.where(np.abs(den) > 1e-9, 0.5 * (dm - dp) / den, 0.0)

        ox = sub(c[t_idx, l_idx, py, (px - 1) % patch], peak, c[t_idx, l_idx, py, (px + 1) % patch])
        oy = sub(c[t_idx, l_idx, (py - 1) % patch, px], peak, c[t_idx, l_idx, (py + 1) % patch, px])
        wrap = lambda v: np.where(v > patch / 2, v - patch, v)
        shifts[:, i, :, 0] = wrap(px) + ox
        shifts[:, i, :, 1] = wrap(py) + oy
        conf[:, i] = peak
    return shifts, conf


def _dense(grid, H, W, s):
    """(Ly, Lx) values at original pixels (i*s, j*s) -> (H, W) by bilinear interpolation."""
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32) / s, np.arange(H, dtype=np.float32) / s)
    return cv2.remap(grid.astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def realign(registered, reference=None, patch=32, iters=2, max_shift=3.0, min_conf=0.15, verbose=False, on_iter=None):
    """Per-patch realignment. registered: (T, H, W[, 3]) uint8 or float in [0, 1]; reference defaults to the mean.
    on_iter(k, stack, reference), if given, is called after every iteration (k from 1).
    Returns (realigned stack, same dtype as the input; reference after the last iteration (float32); info dict)."""
    reg = np.asarray(registered)
    scale = 1 / 255 if reg.dtype == np.uint8 else 1.0
    T, H, W = reg.shape[:3]
    s, _ = _geometry(H, W, patch)
    gray = to_gray(reg.astype(np.float32) * scale)
    ref = reg.astype(np.float32).mean(0) * scale if reference is None else np.asarray(reference, np.float32)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    info = []
    out = reg
    for it in range(iters):
        d, conf = _phase_shifts(gray, to_gray(ref[None])[0], patch)
        ok = (conf >= min_conf) & (np.linalg.norm(d, axis=-1) <= max_shift)
        d[~ok] = 0
        frames = []
        for t in range(T):  # out(x) = frame(x + d): undo frame(x) = ref(x - d)
            mx, my = _dense(d[t, ..., 0], H, W, s), _dense(d[t, ..., 1], H, W, s)
            frames.append(cv2.remap(reg[t].astype(np.float32), gx + mx, gy + my, cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE))
        out = np.stack(frames)
        if reg.dtype == np.uint8:
            out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        ref = out.astype(np.float32).mean(0) * scale
        row = {"iter": it + 1, "used": float(ok.mean()), "mean_shift_px": float(np.linalg.norm(d, axis=-1)[ok].mean())
               if ok.any() else 0.0, "conf_median": float(np.median(conf))}
        info.append(row)
        if on_iter is not None:
            on_iter(it + 1, out, ref)
        if verbose:
            print("   ", row, flush=True)
    return out, ref, info
