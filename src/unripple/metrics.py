"""Image quality metrics. Inputs are float arrays in [0, 1], shape (H, W) or (H, W, 3).

PSNR/SSIM/LPIPS are what [3] reports; NMI/relative RMSE (plus SSIM) are what [1] reports.
LPIPS uses a pretrained network, so it is only ever used for scoring, never inside the method.
"""

import numpy as np
from skimage.metrics import normalized_mutual_information, peak_signal_noise_ratio, structural_similarity

_lpips_models = {}


def psnr(x, gt):
    return peak_signal_noise_ratio(gt, x, data_range=1.0)


def ssim(x, gt):
    return structural_similarity(gt, x, data_range=1.0, channel_axis=-1 if x.ndim == 3 else None)


def rrmse(x, gt):
    """Relative RMSE, ||x - gt|| / ||gt||, as in [1]."""
    return float(np.linalg.norm(x - gt) / np.linalg.norm(gt))


def nmi(x, gt, bins=100):
    """Studholme normalised mutual information, (H(x) + H(gt)) / H(x, gt), in [1, 2].
    Check against the definition in [1] before comparing with their tables."""
    return normalized_mutual_information(gt, x, bins=bins)


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


def all_metrics(x, gt, with_lpips=True):
    x = np.clip(np.asarray(x, dtype=np.float64), 0, 1)
    gt = np.asarray(gt, dtype=np.float64)
    out = {"psnr": psnr(x, gt), "ssim": ssim(x, gt), "nmi": nmi(x, gt), "rrmse": rrmse(x, gt)}
    if with_lpips:
        out["lpips_vgg"] = lpips(x, gt)
    return out
