"""Dataset loaders. Each load_* returns (frames, gt):
frames: float32 (T, H, W) grayscale or (T, H, W, 3) RGB, in [0, 1]
gt:     float32 (H, W) or (H, W, 3), in [0, 1]
"""

import cv2
import numpy as np

from . import paths


def read_video(path, gray=True, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"could not read any frames from {path}")
    return np.stack(frames).astype(np.float32) / 255


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


# ---------------------------------------------------------------- James et al. [1]

def james_names():
    return sorted(p.stem for p in paths.JAMES_VIDEOS.glob("*.avi"))


def load_james(name, gray=True, max_frames=None):
    frames = read_video(paths.JAMES_VIDEOS / f"{name}.avi", gray, max_frames)
    gt = read_image(paths.JAMES_GT / f"{name}.png", gray)
    return frames, gt


# ---------------------------------------------------------------- refractive MFIR benchmark [3]

def mfir_backgrounds():
    return sorted(p.name for p in (paths.MFIR / "backgrounds").glob("*.png"))


def mfir_profiles(wave_type, min_frames=199):
    """Downloaded profiles of this wave type with at least min_frames frames (skips half-streamed ones)."""
    from .third_party.mfir_refraction import paths as wave_dirs
    d = paths.MFIR / "wave_profiles" / wave_dirs[wave_type]
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and len(list(p.glob("*.npy"))) >= min_frames)


def _pad_center_crop(im, size):
    """Same as the benchmark's albumentations PadIfNeeded(constant 0) + CenterCrop."""
    h, w = im.shape[:2]
    ph, pw = max(size - h, 0), max(size - w, 0)
    im = np.pad(im, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    h, w = im.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return im[y:y + size, x:x + size]


def load_mfir(background, wave_type, profile, amplitude="low", L=199, size=512, chunk=25):
    """Generate a distorted RGB clip the same way the benchmark's eval.py does
    (val mode: centre crop, no flips, first L frames of the profile, fixed amplitude).
    background: file name from mfir_backgrounds(); profile: name from mfir_profiles(wave_type)."""
    import torch
    from .third_party.mfir_refraction import amplitudes, gen_video, paths as wave_dirs

    gt = _pad_center_crop(read_image(paths.MFIR / "backgrounds" / background, gray=False), size)
    pdir = paths.MFIR / "wave_profiles" / wave_dirs[wave_type] / profile
    files = sorted(pdir.glob("*.npy"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))[:L]
    if len(files) < L:
        raise ValueError(f"{pdir} has only {len(files)} frames, need {L}")

    amp = amplitudes[wave_type][amplitude]
    x = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None, None]
    clip = []
    for s in range(0, L, chunk):  # scale/depth are fixed in eval mode, so chunking over frames is exact
        norms = torch.from_numpy(np.stack([np.load(f) for f in files[s:s + chunk]])).float()
        clip.append(gen_video(x, norms.permute(0, 3, 1, 2)[None], scale=amp, depth=amp)[0])
    return torch.cat(clip).permute(0, 2, 3, 1).numpy(), gt


# ---------------------------------------------------------------- our own recordings

def own_names():
    return sorted(p.name for p in paths.OWN.iterdir() if p.is_dir() and p.name != "raw") if paths.OWN.exists() else []


def load_own(name, gray=True, max_frames=None):
    """data/own/<name>/still.* (calm water, ground truth) and wavy.* (disturbed water), same fixed camera.
    The ground truth is the temporal median of the still clip, which removes sensor noise."""
    d = paths.OWN / name
    still = next(d.glob("still.*"))
    wavy = next(d.glob("wavy.*"))
    gt = np.median(read_video(still, gray), axis=0)
    return read_video(wavy, gray, max_frames), gt
