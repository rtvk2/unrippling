"""Un-ripple one clip with any method (frozen settings; see unripple.pipeline).

Input (one of):
    python scripts/run_pipeline.py --video my_clip.mp4                       # any video OpenCV can read
    python scripts/run_pipeline.py --frames folder_of_pngs/                   # image sequence, sorted by name
    python scripts/run_pipeline.py --james BlueTiles                          # James et al. clip (with gt)
    python scripts/run_pipeline.py --mfir 8 ocean profile_6 high             # MFIR clip: background index, wave, profile, amplitude
    python scripts/run_pipeline.py --own tub1                                 # data/own/<name>/{still,wavy}.*
Options:
    --method realign_fusion (default, final method) | mean | median | register | rpca | fusion | all
    --gt path.png        ground truth for --video/--frames (scored if given)
    --max-frames N       --gray       --out DIR (default results/pipeline/<clip>)

Writes <out>/<method>.png, <out>/compare.png (input frame, outputs, ground truth if known) and <out>/info.json
(estimated distortion without ground truth, runtimes per stage and, with a ground truth, PSNR/SSIM/LPIPS-VGG with our protocol: gray, crop 4).
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from unripple import data, paths
from unripple.pipeline import FINAL, METHODS, default_device, estimate_distortion, restore

# Bands for the ground-truth-free distortion estimate (results/distortion/: estimate vs measured distortion on the
# 128 sweep clips, Spearman 0.96; James 0.90). Report-only, never changes the method. Estimate > 3 px <=> measured
# > 4 px on 91 % of clips; > 9.5 px <=> measured > 12 px on 92 %. It underestimates ripple (about half) and sine.
BANDS = [(3.0, "mild: all methods work, the final method is best on both PSNR and sharpness"),
         (9.5, "moderate/strong: the final method is sharpest (LPIPS) but registration may score higher PSNR"),
         (1e9, "very strong: no method recovers the scene reliably, results are unreliable")]


def load(args):
    if args.video:
        return Path(args.video).stem, data.read_video(args.video, args.gray, args.max_frames), None
    if args.frames:
        files = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        files = files[:args.max_frames]
        if not files:
            raise SystemExit(f"no images in {args.frames}")
        return Path(args.frames).name, np.stack([data.read_image(f, args.gray) for f in files]), None
    if args.james:
        return args.james, *data.load_james(args.james, args.gray, args.max_frames)
    if args.own:
        return args.own, *data.load_own(args.own, args.gray, args.max_frames)
    bg, wave, prof, amp = args.mfir
    bgname = data.mfir_backgrounds()[int(bg)]
    frames, gt = data.load_mfir(bgname, wave, prof, amp, L=args.max_frames or 199, device=default_device())
    if args.gray:
        from unripple.registration import to_gray
        frames, gt = to_gray(frames), to_gray(gt[None])[0]
    return f"mfir_{bgname[:-4]}_{wave}_{prof}_{amp}", frames, gt


def tile(im, text, size):
    from run_registration import labelled
    h, w = im.shape[:2]
    s = size / max(h, w)
    return labelled(cv2.resize(np.clip(im, 0, 1), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames")
    src.add_argument("--james")
    src.add_argument("--own")
    src.add_argument("--mfir", nargs=4, metavar=("BG", "WAVE", "PROFILE", "AMP"))
    ap.add_argument("--method", default=FINAL, choices=METHODS + ["all"])
    ap.add_argument("--gt")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--size", type=int, default=384, help="tile size of compare.png")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    name, frames, gt = load(args)
    if args.gt:
        gt = data.read_image(args.gt, frames.ndim == 3)
    out = Path(args.out) if args.out else paths.RESULTS / "pipeline" / name
    methods = METHODS if args.method == "all" else [args.method]
    print(f"{name}: {frames.shape[0]} frames of {frames.shape[1]}x{frames.shape[2]}, methods {methods}", flush=True)

    est = estimate_distortion(frames)
    band = next(text for lim, text in BANDS if est <= lim)
    print(f"estimated distortion {est:.1f} px -> {band} (estimate reads low for ripple-like waves)", flush=True)
    t = time.perf_counter()
    ims, info = restore(frames, methods=methods)
    info["total_s"] = time.perf_counter() - t
    info.update(clip=name, frames=list(frames.shape), methods=methods, estimated_distortion_px=est, band=band)

    tiles = [tile(frames[len(frames) // 2], "input (middle frame)", args.size)]
    scores = {}
    for m, im in ims.items():
        data.save_image(out / f"{m}.png", im)
        label = m
        if gt is not None:
            from unripple.metrics import all_metrics
            scores[m] = all_metrics(im, gt, with_lpips=True, crop=4)
            label += f" {scores[m]['psnr']:.1f} dB / {scores[m]['lpips_vgg']:.3f}"
        tiles.append(tile(im, label, args.size))
    if gt is not None:
        tiles.append(tile(gt, "ground truth", args.size))
        info["scores"] = scores
    data.save_image(out / "compare.png", np.concatenate(tiles, 1))
    (out / "info.json").write_text(json.dumps(info, indent=2))

    print("times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in info["times"].items()) + f"; total {info['total_s']:.1f}")
    for m, s in scores.items():
        print(f"  {m:15s} PSNR {s['psnr']:.2f}  SSIM {s['ssim']:.3f}  LPIPS {s['lpips_vgg']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
