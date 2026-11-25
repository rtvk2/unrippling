"""Decode the first N observations and take the same central square on both inputs."""

from itertools import islice
from pathlib import Path
import cv2
import numpy as np
from utils.image_io import read_image

ROOT = Path(__file__).resolve().parents[1]


def square(im, size):
    h, w = im.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    return cv2.resize(
        im[y : y + side, x : x + side],
        (size, size),
        interpolation=cv2.INTER_AREA if size <= side else cv2.INTER_LINEAR,
    )


def load(dataset, count, size, source=None, amplitude=2.0):
    if count < 1 or size < 8:
        raise ValueError("positive frame count and size >= 8 required")
    if dataset == "synthetic" and not source:
        from skimage.data import camera
        from stage_2.warping import warp
        from utils.synthetic import smooth_flows

        if count < 2 or amplitude < 0:
            raise ValueError(
                "synthetic generation needs two frames and nonnegative amplitude"
            )
        ref = square(camera().astype(np.float32) / 255, size)
        fields = smooth_flows(count, size, size, amplitude, 8, np.random.default_rng(0))
        return np.stack([warp(ref, f).clip(0, 1) for f in fields]), ref
    path = Path(source) if source else ROOT / "data/real" / (dataset.title() + ".avi")
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: run python data/download_data.py or pass --source"
        )
    frames = []
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for p in islice(files, count):
            image = read_image(p)
            frames.append(square(image, size))
    else:
        video = cv2.VideoCapture(str(path))
        try:
            for _ in range(count):
                ok, decoded = video.read()
                if not ok:
                    break
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                frames.append(square(image.astype(np.float32) / 255, size))
        finally:
            video.release()
    if len(frames) != count:
        raise ValueError(f"{path} supplied {len(frames)} frames, requested {count}")
    gt = None
    if not source:
        ref = read_image(path.with_suffix(".png"))
        if image.shape == ref.shape:
            gt = square(ref, size)
    return np.stack(frames), gt
