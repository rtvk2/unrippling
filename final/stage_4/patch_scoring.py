import cv2
import numpy as np


def gradient_energy(gray, valid):
    energy = []
    for image, mask in zip(gray, valid):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        energy.append((gx * gx + gy * gy) * interior)
    return np.stack(energy)


def scores(patches, template, energy, mask, confidence, variant, weights):
    axes = (1, 2)
    count = mask.sum(axis=axes).clip(1)
    sharp = (energy * mask).sum(axis=axes) / count
    usable = mask.any(axis=axes)
    lo, hi = sharp[usable].min(), sharp[usable].max()
    sharp = (sharp - lo) / max(hi - lo, 1e-9)
    # Masked normalized cross correlation; a constant patch has no correlation evidence.
    a = (patches - ((patches * mask).sum(axis=axes) / count)[:, None, None]) * mask
    b = (template - ((template * mask).sum(axis=axes) / count)[:, None, None]) * mask
    denom = np.sqrt((a * a).sum(axis=axes) * (b * b).sum(axis=axes))
    corr = np.divide(
        (a * b).sum(axis=axes),
        denom,
        where=denom > 1e-8,
        out=np.zeros_like(count, dtype=float),
    )
    score = weights[0] * sharp
    if variant >= 2:
        score += weights[1] * (1 + corr.clip(-1, 1)) / 2
    if variant == 3:
        score += weights[2] * (confidence * mask).sum(axis=axes) / count
    return np.where(usable, score * mask.mean(axis=axes), -np.inf)
