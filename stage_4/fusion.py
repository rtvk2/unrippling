import cv2
import numpy as np
from stage_1.baselines import checked
from stage_2.registration import to_gray
from .patch_blending import starts, window
from .patch_scoring import gradient_energy, scores


def fuse(
    stack,
    template,
    valid=None,
    confidence=None,
    variant=2,
    patch=32,
    stride=16,
    fraction=0.1,
    weights=(0.4, 0.4, 0.2),
    sharpen=0.0,
):
    stack = checked(stack)
    gray = to_gray(stack)
    ref = to_gray(template[None])[0]
    valid = np.ones(gray.shape, bool) if valid is None else np.asarray(valid, bool)
    if template.shape != stack.shape[1:] or valid.shape != gray.shape:
        raise ValueError("template or validity shape mismatch")
    if variant not in (1, 2, 3) or not 0 < fraction <= 1 or not 0 < stride <= patch:
        raise ValueError("invalid variant, top fraction, stride or patch size")
    if variant == 3 and (confidence is None or confidence.shape != valid.shape):
        raise ValueError("V3 requires a measured confidence stack")
    if (
        len(weights) != 3
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or not sum(weights[:variant])
    ):
        raise ValueError(
            "provide three nonnegative weights with an active scoring term"
        )
    if variant == 3 and (
        not np.isfinite(confidence).all()
        or confidence.min() < 0
        or confidence.max() > 1
    ):
        raise ValueError("confidence must be finite and in [0,1]")
    energy = gradient_energy(gray, valid)
    h, w = ref.shape
    blend = window(min(h, patch), min(w, patch))
    ys, xs = starts(h, patch, stride), starts(w, patch, stride)
    k = int(np.ceil(fraction * len(stack)))
    selected = np.full((len(ys), len(xs), k), -1, dtype=np.int32)
    result = np.zeros_like(template)
    total = np.zeros((h, w), np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            area = np.s_[y : y + patch, x : x + patch]
            indices = (slice(None),) + area
            mask = valid[indices]
            if not mask.any():
                continue
            quality = scores(
                gray[indices],
                ref[area],
                energy[indices],
                mask,
                confidence[indices] if variant == 3 else None,
                variant,
                weights,
            )
            best = np.argsort(-quality, kind="stable")[:k]
            best = best[np.isfinite(quality[best])]
            selected[iy, ix, : len(best)] = best
            votes = blend * mask[best] * np.maximum(quality[best], 1e-6)[:, None, None]
            total[area] += votes.sum(axis=0)
            if stack.ndim == 4:
                votes = votes[..., None]
            result[area] += (votes * stack[indices][best]).sum(axis=0)
    if stack.ndim == 4:
        total = total[..., None]
    image = np.divide(result, total, out=template.copy(), where=total > 0)
    if sharpen:
        image += sharpen * (image - cv2.GaussianBlur(image, (0, 0), 1))
    return image.clip(0, 1), selected
