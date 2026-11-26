"""Sanity checks for unripple.fusion (no real data needed).

    python scripts/test_fusion_synthetic.py

1. Identity: keeping every frame with uniform weights must give the plain temporal mean (checks the patch
   grid, padding and Hann blending; error should be ~1e-6), for every patch size, gray and RGB, odd sizes.
2. Lucky regions: a clean image where every frame is corrupted (blurred, or a shifted ghost copy blended
   in) in random patches. The mean mixes the corruption in; fusion should pick the clean frames per
   location and beat the mean clearly.
"""

import cv2
import numpy as np
from skimage import data as skdata

from unripple.fusion import fuse, fuse_from_terms, score_terms
from unripple.metrics import psnr

rng = np.random.default_rng(0)
ok = True

# 1. identity
for shape in [(20, 96, 128, 3), (20, 101, 77), (5, 64, 64, 3)]:
    stack = rng.random(shape, dtype=np.float32)
    for patch in (16, 32):
        terms = score_terms(stack, stack.mean(0), patch=patch)
        out = fuse_from_terms(stack, terms, {"sharp": 1.0}, patch=patch, frac=1.0, weighting="uniform")
        err = float(np.abs(out - stack.mean(0)).max())
        ok &= err < 1e-5
        print(f"identity {shape} patch {patch}: max error {err:.1e}")

# 2. lucky regions
img = cv2.resize(skdata.astronaut(), (256, 256)).astype(np.float32) / 255
T, frames = 40, []
for t in range(T):
    f = img.copy()
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    ghost = 0.5 * img + 0.5 * np.roll(img, (6, -5), axis=(0, 1))
    for _ in range(12):  # 12 corrupted 48x48 regions per frame (~40 % of the area)
        y, x = rng.integers(0, 256 - 48, 2)
        f[y:y + 48, x:x + 48] = (blur if rng.random() < 0.5 else ghost)[y:y + 48, x:x + 48]
    frames.append(f)
frames = np.stack(frames)
mean = frames.mean(0)
print(f"\nlucky regions: mean {psnr(mean, img):.2f} dB")
for name, w in [("sharp", {"sharp": 1}), ("sim", {"sim": 1}), ("sharp+sim", {"sharp": 1, "sim": 1})]:
    out = fuse(frames, mean, weights=w, patch=32, frac=0.2)
    print(f"  fusion {name:10s} {psnr(out, img):.2f} dB")
    if name == "sharp+sim":
        ok &= psnr(out, img) > psnr(mean, img) + 3

print("\nPASS" if ok else "\nFAIL")
