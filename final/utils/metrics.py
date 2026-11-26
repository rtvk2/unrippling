"""Image quality metrics. Inputs are float arrays in [0, 1], shape (H, W) or (H, W, 3).

psnr, ssim and lpips are copied verbatim from the historical src/unripple/metrics.py so that final/ is
self-contained. LPIPS uses a pretrained network, so it is only ever used for scoring, never inside the method.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_models = {}


def psnr(x, gt):
    return peak_signal_noise_ratio(gt, x, data_range=1.0)


def ssim(x, gt):
    return structural_similarity(gt, x, data_range=1.0, channel_axis=-1 if x.ndim == 3 else None)


def lpips(x, gt, net="vgg"):
    """LPIPS distance (lower is better). [3] reports LPIPS-VGG; net='alex' is also available."""
    import torch
    import lpips as lpips_pkg

    if net not in _lpips_models:
        _lpips_models[net] = lpips_pkg.LPIPS(net=net, verbose=False).eval()

    def to_tensor(im):
        im = np.asarray(im, dtype=np.float32)
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return torch.from_numpy(im).permute(2, 0, 1)[None] * 2 - 1

    with torch.no_grad():
        return float(_lpips_models[net](to_tensor(x), to_tensor(gt)))


def measure(image, reference, crop, perceptual):
    if reference is None:
        return {}
    if crop < 0 or min(image.shape[:2]) - 2 * crop < 7:
        raise ValueError("evaluation crop must retain at least seven pixels per axis")
    region = slice(crop, -crop or None)
    x, y = image[region, region], reference[region, region]
    result = dict(psnr=float(psnr(x, y)), ssim=float(ssim(x, y)))
    if perceptual:
        result["lpips_vgg"] = lpips(x, y)
    return result


def spatial_crop(stack, crop):
    """Crop H and W of a (T,H,W,...) measurement, preserving all observations."""
    if stack.ndim < 3 or crop < 0 or 2 * crop >= min(stack.shape[1:3]):
        raise ValueError("invalid spatial crop")
    region = slice(crop, -crop or None)
    return stack[:, region, region]
