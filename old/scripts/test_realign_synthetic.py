"""Synthetic check for realign.py: frames = clean image warped by known smooth sub-pixel shift fields (~1.5 px RMS,
like the leftover misalignment after registration). Realignment must beat the plain mean and recover the shifts."""
import cv2
import numpy as np
from skimage import data as skd

from unripple.metrics import psnr
from unripple.realign import realign

rng = np.random.default_rng(0)
img = cv2.resize(skd.astronaut(), (256, 256)).astype(np.float32) / 255
H, W = img.shape[:2]
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
frames, true = [], []
for t in range(40):
    f = [cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 24) for _ in range(2)]
    f = [x / x.std() * 1.5 for x in f]
    frames.append(cv2.remap(img, gx + f[0], gy + f[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    true.append(np.stack(f, -1))
frames = np.clip(np.stack(frames), 0, 1)
u8 = (frames * 255 + 0.5).astype(np.uint8)
c = slice(8, -8)
m = lambda x: psnr(np.clip(x, 0, 1)[c, c], img[c, c])
print(f"mean {m(frames.mean(0)):.2f} dB")
for iters in (1, 2, 3):
    out, ref, info = realign(u8, iters=iters)
    print(f"realign iters={iters}: {m(ref):.2f} dB  frame0 {m(out[0] / 255):.2f} vs before {m(frames[0]):.2f}  {info[-1]}")
