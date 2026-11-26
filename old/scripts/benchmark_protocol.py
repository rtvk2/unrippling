"""Score our methods with the PUBLISHED papers' protocols, so our numbers can sit next to theirs (stage 7, item 3).

    python scripts/benchmark_protocol.py mfir     # test split (MFIR shallow, backgrounds 8-15), Shugaev et al. protocol
    python scripts/benchmark_protocol.py james    # 14 James clips, James et al. (ICCV 2019) protocol

MFIR protocol (external/refractive-mfir-benchmark/eval.py): RGB, full image (no crop), torchmetrics PSNR and SSIM
(data_range 1, Gaussian 11x11 window), LPIPS VGG and Alex (their LPIPS_metric). Methods: pixel average (mean of all
frames, their 'Pixel average'), median, registered (stage 2), fusion (frozen stage 4), realign3+fusion (stage 4
with realignment), and their own non-learned 'Grid registration' (img_gen.gen_Grid_registration, default
settings, run on our clips): 'grid_reg*' = metrics per output frame then averaged (as in their tables),
'grid_reg_mean' = the mean of its output frames scored as one image (like our single-image outputs).

James protocol (arXiv 1908.01940): frames resized to 256x256 after a centre crop to keep the aspect ratio
(256x256 = square crop of the shorter side), grayscale, NMI (Studholme, in [1, 2]), SSIM (Gaussian sigma 1.5),
relative RMSE ||x - gt|| / ||gt||. Raw ground truth (no shift correction).

Outputs: results/benchmark_protocol/{mfir,james}_metrics.csv (resumable; done clips are skipped) and per clip
<dataset>/<clip>_<method>.png.
"""

import argparse
import sys
import time
import types
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import normalized_mutual_information, structural_similarity

from cache_registered import load_cached
from run_fusion import DEFAULT
from tune_registration import clips as split_clips
from unripple import data, paths
from unripple.fusion import fuse_from_terms, score_terms
from unripple.realign import realign
from unripple.registration import to_gray

BENCH = paths.EXTERNAL / "refractive-mfir-benchmark"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def our_methods(reg, tmpl, flows):
    """Our single-image outputs from the cached registered stack: registered, fusion, realign3+fusion."""
    p, frac, w = DEFAULT["patch"], DEFAULT["frac"], DEFAULT["weights"]
    out = {"registered": tmpl, "fusion": fuse_from_terms(reg, score_terms(reg, tmpl, flows, p), w, p, frac)}
    ra, ref, _ = realign(reg, iters=3)
    out["realign3+fusion"] = fuse_from_terms(ra, score_terms(ra, ref, flows, p), w, p, frac)
    return {k: np.clip(v, 0, 1).astype(np.float32) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ MFIR

def bench_modules():
    sys.path.insert(0, str(BENCH))
    stub = types.ModuleType("DATUM")  # img_gen imports the learned DATUM model at the top; it is never used here
    stub.Model = None
    sys.modules.setdefault("DATUM", stub)
    import img_gen
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    import lpips as lp

    class LP:  # = metrics.LPIPS_metric of the benchmark (inputs in [0, 1], mapped to [-1, 1])
        def __init__(self, net):
            self.m = lp.LPIPS(pnet_tune=False, verbose=False, net=net).to(dev)

        def __call__(self, x, y):
            with torch.no_grad():
                return self.m(2 * x - 1, 2 * y - 1).view(-1)

    return img_gen, PeakSignalNoiseRatio(data_range=1.0).to(dev), \
        StructuralSimilarityIndexMeasure(data_range=1.0).to(dev), LP("vgg"), LP("alex")


def run_mfir(args):
    img_gen, psnr_m, ssim_m, lp_vgg, lp_alex = bench_modules()
    grid = img_gen.gen_Grid_registration(device=dev)
    out = paths.RESULTS / "benchmark_protocol"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "mfir_metrics.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["dataset"], r["clip"]) for r in rows}

    def score(pred, gt):  # pred (B, 3, H, W) in [0, 1], gt (3, H, W): torchmetrics over the batch, LPIPS averaged
        # Chunked update()/compute(): identical to one batch call (PSNR pools the MSE over all pixels of the batch,
        # SSIM averages per image), but fits a 6 GB GPU for the 199-frame grid registration output.
        r, lv, la = {}, [], []
        with torch.no_grad():
            for s in range(0, len(pred), 16):
                p = pred[s:s + 16].to(dev)
                g = gt[None].expand_as(p)
                psnr_m.update(p, g)
                ssim_m.update(p, g)
                lv.append(lp_vgg(p, g).cpu())
                la.append(lp_alex(p, g).cpu())
            r["psnr"], r["ssim"] = psnr_m.compute().item(), ssim_m.compute().item()
            psnr_m.reset()
            ssim_m.reset()
        r["lpips_vgg"], r["lpips_alex"] = torch.cat(lv).mean().item(), torch.cat(la).mean().item()
        torch.cuda.empty_cache()
        return r

    t_ = lambda im: torch.from_numpy(np.ascontiguousarray(np.asarray(im, np.float32))).permute(2, 0, 1).to(dev)
    cl = list(split_clips("test", 199))
    t0 = time.time()
    for ci, (dataset, name, load) in enumerate(cl, 1):
        if (dataset, name) in done:
            continue
        frames, gt = load()  # (T, H, W, 3) float [0, 1], exactly the benchmark's generation (data.load_mfir)
        reg, tmpl, flows, _ = load_cached(dataset, name, "test")
        reg = np.asarray(reg)
        ims = {"pixel_average": frames.mean(0), "median": np.median(frames, 0)}
        ims.update(our_methods(reg, tmpl, flows))
        t = time.perf_counter()
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().to(dev)  # L C H W, on the
        # device like eval.py's generated ref_video (project_all samples it with the GPU grid)
        gr = grid({"ref_video": video}).cpu().float() / 255  # their output: registered video (L, 3, H, W) uint8
        t_grid = time.perf_counter() - t
        g = t_(gt)
        base = {"dataset": dataset, "clip": name}
        for m, im in ims.items():
            im = (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)  # 8-bit output, as their pipeline (pred is uint8)
            data.save_image(out / dataset / f"{name}_{m}.png", im.astype(np.float32) / 255)
            rows.append({**base, "method": m, **score(t_(im.astype(np.float32) / 255)[None], g)})
        rows.append({**base, "method": "grid_reg*", "runtime_s": t_grid, **score(gr, g)})
        gm = gr.mean(0, keepdim=True)
        gm = torch.round(gm * 255) / 255
        data.save_image(out / dataset / f"{name}_grid_reg_mean.png", gm[0].permute(1, 2, 0).cpu().numpy())
        rows.append({**base, "method": "grid_reg_mean", **score(gm, g)})
        del gr, video
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(csv, index=False)
        last = [r for r in rows if r["dataset"] == dataset and r["clip"] == name]
        print(f"[{ci}/{len(cl)}, {time.time() - t0:5.0f} s] {dataset} {name} (grid {t_grid:.0f} s): " +
              "  ".join(f"{r['method']} {r['psnr']:.2f}/{r['lpips_vgg']:.3f}" for r in last), flush=True)
    summarize(pd.DataFrame(rows), out / "mfir_summary.csv", ["psnr", "ssim", "lpips_vgg", "lpips_alex"])


# ------------------------------------------------------------------------------------------------ James

def james_prep(im):
    """Centre square crop of the shorter side, resize to 256x256, grayscale (James et al. protocol)."""
    im = np.asarray(im, np.float32)
    h, w = im.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    im = cv2.resize(im[y:y + s, x:x + s], (256, 256), interpolation=cv2.INTER_AREA)
    return np.clip(to_gray(im[None])[0] if im.ndim == 3 else im, 0, 1).astype(np.float64)


def run_james(args):
    out = paths.RESULTS / "benchmark_protocol"
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    names = data.james_names()
    for ci, name in enumerate(names, 1):
        reg, tmpl, flows, gt = load_cached("james2019", name)
        reg = np.asarray(reg)
        raw = data.load_james(name)[0]  # same frames that were registered (cache_registered.py)
        ims = {"mean": raw.mean(0), "median": np.median(raw, 0)}
        del raw
        ims.update(our_methods(reg, tmpl, flows))
        g = james_prep(gt)
        for m, im in ims.items():
            x = james_prep(im)
            rows.append({"dataset": "james2019", "clip": name, "method": m,
                         "nmi": normalized_mutual_information(g, x, bins=100),
                         "ssim": structural_similarity(g, x, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                                       use_sample_covariance=False),
                         "rrmse": float(np.linalg.norm(x - g) / np.linalg.norm(g))})
        print(f"[{ci}/{len(names)}, {time.time() - t0:5.0f} s] {name}: " +
              "  ".join(f"{r['method']} {r['ssim']:.3f}" for r in rows[-len(ims):]), flush=True)
        pd.DataFrame(rows).to_csv(out / "james_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    print("\nPer clip SSIM:\n", df.pivot(index="clip", columns="method", values="ssim").round(3).to_string())
    summarize(df, out / "james_summary.csv", ["nmi", "ssim", "rrmse"])


def summarize(df, path, cols):
    s = df.groupby(["dataset", "method"])[cols].mean().round(4)
    s.to_csv(path)
    print("\nMeans per group:\n", s.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["mfir", "james"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    (run_mfir if args.which == "mfir" else run_james)(args)


if __name__ == "__main__":
    main()
