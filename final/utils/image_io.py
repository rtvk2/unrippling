"""Image reading and writing. Images are float arrays in [0, 1], shape (H, W) or (H, W, 3).

Copied verbatim from the historical src/unripple/data.py so that final/ is self-contained.
"""

import cv2
import numpy as np


def read_image(path, gray=True):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"could not read {path}")
    if not gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255


def save_image(path, im):
    im = np.clip(np.asarray(im) * 255 + 0.5, 0, 255).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im)
