"""Stage 4 (our extension): lucky patch fusion of the registered stack.

After registration, different frames are good in different places (the water was locally flat there at that
moment). For every patch location, score each frame's patch, keep the best `frac` of the frames, average them
with score weights, and blend the overlapping patches with a Hann window so there are no seams.

Patch grid: patches of size p = `patch`, stride s = p/2 (50 % overlap). The image is reflect-padded by s so
that every real pixel is covered by four full windows; a periodic Hann window at 50 % overlap sums to exactly 1.

Score terms (computed on grayscale, one value per frame per patch location), each z-scored across frames at
that location before combining, so only the ranking of frames within a location matters:
  sharp: log gradient energy of the frame patch minus that of the template patch (detail present in the frame)
  sim:   normalised cross-correlation with the template patch (the patch is in the right place; penalises
         ghost copies and misregistered frames, which also raise gradient energy)
  flow:  -(z(mean |flow|) + z(mean |Jacobian of flow|)) / 2 over the patch: frames that needed little
         correction, and whose correction was locally a near-rigid shift, i.e. the water was locally flat.
         (Temporal variance of the flow at a location is the same for every frame, so it cannot rank frames.)
score = w_sharp * sharp + w_sim * sim + w_flow * flow.
"""

import cv2
import numpy as np

from .registration import to_gray

TERMS = ("sharp", "sim", "flow")


def _hann(p):
    n = np.arange(p)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / p)  # periodic, sums to 1 at 50 % overlap
    return np.outer(w, w).astype(np.float32)


def _geometry(h, w, patch):
    if patch % 2:
        raise ValueError("patch size must be even")
    s = patch // 2
    hp, wp = -(-(h + 2 * s) // s) * s, -(-(w + 2 * s) // s) * s  # padded size, a multiple of s
    return s, ((s, hp - h - s), (s, wp - w - s))


def _cells(maps, s, pads):
    """(N, H, W) per-pixel maps -> (N, Ly, Lx) sums over every p x p patch (p = 2s) of the padded grid."""
    m = np.pad(maps, ((0, 0),) + pads, mode="reflect")
    n, hp, wp = m.shape
    c = m.reshape(n, hp // s, s, wp // s, s).sum(axis=(2, 4), dtype=np.float64)
    return c[:, :-1, :-1] + c[:, 1:, :-1] + c[:, :-1, 1:] + c[:, 1:, 1:]


def _grad_energy(g):
    gy, gx = np.gradient(g, axis=(-2, -1))
    return gx * gx + gy * gy


def _zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def score_terms(registered, template, flows=None, patch=32, chunk=32):
    """Raw score terms, each (T, Ly, Lx), z-scored across frames per location.

    registered: (T, H, W[, 3]) float in [0, 1] or uint8; template: (H, W[, 3]) float.
    flows: optional (T, h, w, 2) flow of the last registration round, at full or reduced resolution (values in
    full-resolution pixels); without it the 'flow' term is absent."""
    T, H, W = registered.shape[:3]
    s, pads = _geometry(H, W, patch)
    n = float(patch * patch)
    scale = 1 / 255 if registered.dtype == np.uint8 else 1.0
    t = to_gray(np.asarray(template, np.float32)[None])
    st, stt, et = (_cells(x, s, pads)[0] for x in (t, t * t, _grad_energy(t)))
    vt = stt - st * st / n

    sharp, sim = [], []
    for a in range(0, T, chunk):
        g = to_gray(np.asarray(registered[a:a + chunk], np.float32) * scale)
        sf, sff, sft, ef = (_cells(x, s, pads) for x in (g, g * g, g * t, _grad_energy(g)))
        vf = sff - sf * sf / n
        sim.append((sft - sf * st / n) / np.sqrt(np.maximum(vf * vt, 0) + 1e-6 * n * n))
        sharp.append(np.log(ef + 1e-6 * n) - np.log(et + 1e-6 * n))
    terms = {"sharp": _zscore(np.concatenate(sharp)), "sim": _zscore(np.concatenate(sim))}

    if flows is not None:
        flows = np.asarray(flows, np.float32)
        step = W / flows.shape[2]
        mag = np.linalg.norm(flows, axis=-1)
        d = np.gradient(flows, axis=(1, 2))  # derivatives per low-res pixel -> per full-res pixel: / step
        jac = np.sqrt(sum((x ** 2).sum(-1) for x in d)) / step
        up = lambda m: np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in m])
        terms["flow"] = -(_zscore(_cells(up(mag), s, pads)) + _zscore(_cells(up(jac), s, pads))) / 2
    return terms


def fuse_from_terms(registered, terms, weights, patch=32, frac=0.15, weighting="softmax", tau=1.0):
    """Fuse with precomputed `terms` (from score_terms with the same patch size).

    weights: {"sharp": a, "sim": b, "flow": c}; terms with weight 0 or missing are ignored.
    frac: fraction of frames kept per location (frac=1 with weighting='uniform' gives the registered mean).
    weighting: 'softmax' (weights exp(score / tau) over the kept frames) or 'uniform'."""
    T, H, W = registered.shape[:3]
    s, pads = _geometry(H, W, patch)
    score = sum(w * terms[k] for k, w in weights.items() if w)
    if np.isscalar(score):
        score = np.zeros_like(next(iter(terms.values())))
    k = int(np.clip(round(frac * T), 1, T))
    idx = np.argpartition(-score, k - 1, axis=0)[:k] if k < T else np.broadcast_to(np.arange(T)[:, None, None],
                                                                                    score.shape)
    kept = np.take_along_axis(score, idx, axis=0)
    if weighting == "softmax":
        wts = np.exp((kept - kept.max(axis=0)) / tau)
    elif weighting == "uniform":
        wts = np.ones_like(kept)
    else:
        raise ValueError(f"unknown weighting '{weighting}'")
    wts = (wts / wts.sum(axis=0)).astype(np.float32)

    scale = 1 / 255 if registered.dtype == np.uint8 else 1.0
    color = registered.ndim == 4
    stack = np.pad(np.asarray(registered), ((0, 0),) + pads + (((0, 0),) if color else ()), mode="reflect")
    out = np.zeros(stack.shape[1:], np.float32)
    win = _hann(patch)[..., None] if color else _hann(patch)
    p = patch
    for i in range(score.shape[1]):
        for j in range(score.shape[2]):
            y, x = i * s, j * s
            pts = stack[idx[:, i, j], y:y + p, x:x + p].astype(np.float32)
            w = wts[:, i, j].reshape((-1,) + (1,) * (pts.ndim - 1))
            out[y:y + p, x:x + p] += win * (w * pts).sum(axis=0)
    return out[s:s + H, s:s + W] * scale


def unsharp(im, amount=0.5, sigma=1.0):
    """Mild sharpening: im + amount * (im - Gaussian(im, sigma)), clipped to [0, 1]."""
    im = np.asarray(im, np.float32)
    return np.clip(im + amount * (im - cv2.GaussianBlur(im, (0, 0), sigma)), 0, 1)


def fuse(registered, template, flows=None, weights=None, patch=32, frac=0.15, weighting="softmax", tau=1.0,
         sharpen=0.0, sharpen_sigma=1.0):
    """Lucky patch fusion: returns an (H, W[, 3]) float32 image in [0, 1]. See the module docstring."""
    weights = weights or {"sharp": 1.0, "sim": 1.0, "flow": 1.0 if flows is not None else 0.0}
    terms = score_terms(registered, template, flows, patch)
    out = fuse_from_terms(registered, terms, weights, patch, frac, weighting, tau)
    return unsharp(out, sharpen, sharpen_sigma) if sharpen > 0 else np.clip(out, 0, 1)
