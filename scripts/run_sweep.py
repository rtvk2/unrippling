"""Stage 7 sweep: every method, frozen settings, on unseen MFIR clips of all four wave types.

    python scripts/run_sweep.py                  # 2 worker processes, resumable
    python scripts/run_sweep.py --workers 1 --limit 2   # quick smoke test

Clips: every extracted profile of every wave type (data.mfir_profiles: Shallow/Ocean/Ripples/Sine, profile_6 and
profile_10) x 4 amplitudes x test backgrounds 8, 10, 12, 14 (backgrounds 0-2 were seen during tuning). Ordered
background-first, so a half-finished run already covers every wave type / amplitude with fewer backgrounds.

Methods (nothing retuned): pixel_average, median, registered (stage 2), rpca (stage 3, GPU version, = numpy
version to 1e-6), fusion (frozen stage 4 selection), realign3+fusion (stage 4 as adopted). Registered frames
and flows are rounded exactly like the cache (uint8 frames, flow at 1/4 resolution in float16) before
realignment/fusion, so fusion numbers are made the same way as in run_fusion.py / run_realign.py.

Scored with both protocols:
  ours (psnr, ssim, lpips_vgg): gray, crop 4, as every earlier stage (metrics.all_metrics)
  MFIR (mfir_psnr, mfir_ssim, mfir_lpips_vgg, mfir_lpips_alex): RGB, full image, 8-bit output, torchmetrics
  PSNR/SSIM and LPIPS VGG/Alex, as in benchmark_protocol.py / the benchmark's eval.py
GPU work (rendering, RPCA, scoring) is serialised across workers with a lock (RPCA needs 4.4 GB of the 6 GB);
registration, realignment and fusion run on the CPU in parallel. Per-stage runtimes are recorded but measured
with workers competing for the CPU, so they are not a clean runtime table.

Outputs: results/sweep/metrics.csv (one row per clip x method, appended after every clip; done clips are skipped
on restart), results/sweep/summary_*.csv, results/sweep/<wave>/<clip>_<amp>.png comparison strips.
"""

import argparse
import multiprocessing as mp
import time
import warnings

import cv2
import numpy as np
import pandas as pd

from unripple import data, paths

BGS = [8, 10, 12, 14]
WAVES = ["shallow", "ocean", "ripple", "sine"]
AMPS = ["low", "mid", "high", "extreme"]
PROFILES = ["profile_6", "profile_10"]
METHODS = ["pixel_average", "median", "registered", "rpca", "fusion", "realign3+fusion"]
OUT = paths.RESULTS / "sweep"
FLOW_STEP = 4  # as cache_registered.py

_lock = None
_scorers = None


def clip_list():
    bgs = data.mfir_backgrounds()
    # Fixed profile list (not data.mfir_profiles): profiles 1 and 2 were extracted later as the ripple/sine dev set
    return [(wt, prof, amp, bgs[b]) for b in BGS for wt in WAVES for prof in PROFILES for amp in AMPS]


def _init(lock):
    global _lock
    _lock = lock
    warnings.filterwarnings("ignore")


def _mfir_scorers():
    global _scorers
    if _scorers is None:
        import lpips as lp
        import torch
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        torch.backends.cuda.matmul.allow_tf32 = False
        nets = {n: lp.LPIPS(pnet_tune=False, verbose=False, net=n).eval().cuda() for n in ("vgg", "alex")}
        _scorers = PeakSignalNoiseRatio(data_range=1.0).cuda(), StructuralSimilarityIndexMeasure(data_range=1.0).cuda(), nets
    return _scorers


def mfir_metrics(im, gt):
    """MFIR protocol on one image: 8-bit RGB, full frame (= benchmark_protocol.py for a single image)."""
    import torch
    psnr_m, ssim_m, nets = _mfir_scorers()
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1)[None].cuda()
    p = t((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8) / 255.0)
    g = t(gt)
    with torch.no_grad():
        r = {"mfir_psnr": psnr_m(p, g).item(), "mfir_ssim": ssim_m(p, g).item()}
        for n, m in nets.items():
            r[f"mfir_lpips_{n}"] = m(2 * p - 1, 2 * g - 1).item()
    psnr_m.reset()
    ssim_m.reset()
    return r


def run_clip(job):
    wt, prof, amp, bg = job
    from run_fusion import DEFAULT
    from run_registration import labelled
    from unripple.fusion import fuse_from_terms, score_terms
    from unripple.metrics import all_metrics
    from unripple.realign import realign
    from unripple.registration import register
    from unripple.rpca import rpca_image

    import torch

    rt = {}
    t = time.perf_counter()
    with _lock:  # every locked GPU section ends with empty_cache(): RPCA needs 4.4 GB of the 6 GB, so memory the
        # other worker's allocator keeps cached after rendering/scoring caused an OOM at clip 120 of the first run
        frames, gt = data.load_mfir(bg, wt, prof, amp, L=199, device="cuda")
        torch.cuda.empty_cache()
    rt["render"] = time.perf_counter() - t

    t = time.perf_counter()
    registered, tmpl, flows, _ = register(frames)  # frozen stage 2
    rt["registration"] = time.perf_counter() - t
    h, w = flows.shape[1:3]
    small = np.stack([cv2.resize(f, (w // FLOW_STEP, h // FLOW_STEP), interpolation=cv2.INTER_AREA)
                      for f in flows]).astype(np.float16).astype(np.float32)
    del flows
    reg8 = np.clip(registered * 255 + 0.5, 0, 255).astype(np.uint8)
    ims = {"pixel_average": frames.mean(0), "median": np.median(frames, 0), "registered": tmpl}
    del frames

    t = time.perf_counter()
    with _lock:
        ims["rpca"] = rpca_image(registered, device="cuda")[0]  # stage 3 as run_rpca.py (float registered stack)
    rt["rpca"] = time.perf_counter() - t
    del registered

    p, frac, wts = DEFAULT["patch"], DEFAULT["frac"], DEFAULT["weights"]
    t = time.perf_counter()
    ims["fusion"] = fuse_from_terms(reg8, score_terms(reg8, tmpl, small, p), wts, p, frac)
    rt["fusion"] = time.perf_counter() - t
    t = time.perf_counter()
    ra, ref, _ = realign(reg8, iters=3)
    ims["realign3+fusion"] = fuse_from_terms(ra, score_terms(ra, ref, small, p), wts, p, frac)
    rt["realign3+fusion"] = time.perf_counter() - t
    del ra, reg8

    rows = []
    t = time.perf_counter()
    with _lock:
        for m in METHODS:
            im = np.clip(ims[m], 0, 1).astype(np.float32)
            rows.append({"wave": wt, "profile": prof, "amplitude": amp, "background": bg[:-4], "method": m,
                         **all_metrics(im, gt, with_lpips=True, crop=4), **mfir_metrics(im, gt)})
        torch.cuda.empty_cache()
    rt["scoring"] = time.perf_counter() - t
    for r in rows:
        r.update({f"t_{k}": v for k, v in rt.items()})

    strip = [labelled(cv2.resize(np.clip(ims[m], 0, 1), (256, 256), interpolation=cv2.INTER_AREA),
                      f"{m} {r['mfir_psnr']:.1f}") for m, r in zip(METHODS, rows)]
    strip.append(labelled(cv2.resize(gt, (256, 256), interpolation=cv2.INTER_AREA), "ground truth"))
    data.save_image(OUT / wt / f"{bg[:-4]}_{prof}_{amp}.png", np.concatenate(strip, 1))
    return job, rows


def summarize(df):
    cols = ["psnr", "ssim", "lpips_vgg", "mfir_psnr", "mfir_ssim", "mfir_lpips_vgg", "mfir_lpips_alex"]
    order = {m: i for i, m in enumerate(METHODS)}
    by = df.groupby(["wave", "amplitude", "method"])[cols].mean()
    by = by.sort_index(key=lambda s: s.map(order) if s.name == "method" else
                       s.map({a: i for i, a in enumerate(AMPS)}) if s.name == "amplitude" else s)
    by.round(4).to_csv(OUT / "summary_wave_amplitude.csv")
    df.groupby(["amplitude", "method"])[cols].mean().round(4).to_csv(OUT / "summary_amplitude.csv")
    piv = df.pivot_table(index=["wave", "amplitude"], columns="method", values="mfir_psnr")[METHODS]
    lp = df.pivot_table(index=["wave", "amplitude"], columns="method", values="mfir_lpips_vgg")[METHODS]
    print("\nMFIR-protocol PSNR (mean over clips):\n", piv.round(2).to_string(), flush=True)
    print("\nMFIR-protocol LPIPS-VGG:\n", lp.round(3).to_string(), flush=True)
    n = df.pivot_table(index=["wave", "profile", "amplitude", "background"], columns="method", values="mfir_lpips_vgg")
    print(f"\nrealign3+fusion vs rpca, LPIPS-VGG better on {(n['realign3+fusion'] < n['rpca']).sum()}/{len(n)} clips")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "metrics.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["wave"], r["profile"], r["amplitude"], r["background"]) for r in rows}
    jobs = clip_list()[:args.limit]
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3][:-4]) not in done]
    print(f"{len(jobs)} clips, {len(jobs) - len(todo)} already done, {len(todo)} to run, {args.workers} workers",
          flush=True)
    t0 = time.time()
    lock = mp.Lock()
    with mp.Pool(args.workers, initializer=_init, initargs=(lock,)) as pool:
        for k, (job, new) in enumerate(pool.imap_unordered(run_clip, todo), 1):
            rows += new
            pd.DataFrame(rows).to_csv(csv, index=False)
            el = time.time() - t0
            r = {x["method"]: x for x in new}
            t = new[0]
            print(f"[{k}/{len(todo)}, {el / 60:5.1f} min, ETA {el / k * (len(todo) - k) / 60:5.1f} min] "
                  f"{job[0]} {job[1]} {job[2]} {job[3][:-4]} (reg {t['t_registration']:.0f} s, rpca {t['t_rpca']:.0f} s, "
                  f"realign+fus {t['t_realign3+fusion']:.0f} s) MFIR PSNR/LPIPS: " +
                  "  ".join(f"{m} {r[m]['mfir_psnr']:.2f}/{r[m]['mfir_lpips_vgg']:.3f}" for m in METHODS), flush=True)
    summarize(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
