"""Why does registration fail on ripple/sine waves? Flow smoothing sweep on a NEW dev set (stage 7 follow-up).

    python scripts/diagnose_waves.py            # resumable; results/wave_diagnosis/{metrics.csv,summary.csv}

Dev set (never used for the test sweep, not in the benchmark's official subset.pickle): profiles 1 and 2 of each
wave type x backgrounds 3, 4, 5. Ripple and sine at all 4 amplitudes (48 clips); shallow and ocean at high and
extreme only (24 clips), as a check that a fix does not break the wave types that already work.
50 frames per clip (as tune_registration.py), 4 rounds, mean template, flow_sigma in {0, 2, 4, 8} (8 = frozen).

Per clip and sigma: PSNR/SSIM/LPIPS of the registered mean (our protocol: gray, crop 4), plus mean and median of
the raw frames. Frame-only signals recorded for the later fixes (nothing here uses the ground truth):
  flow_wavelength_px  dominant wavelength of the last-round flow field at sigma 0 (power-weighted mean spatial
                      frequency of the flow, DC removed) -> candidate rule for adaptive smoothing (fix A)
  agree_ratio         temporal std of the registered gray frames / temporal std of the raw frames, averaged over
                      pixels: < 1 means registration made the frames agree more -> candidate failure test (fix B)
"""

import time
import warnings

import numpy as np
import pandas as pd

from unripple import data, paths
from unripple.metrics import all_metrics
from unripple.registration import register, to_gray

WAVES = {"ripple": ["low", "mid", "high", "extreme"], "sine": ["low", "mid", "high", "extreme"],
         "shallow": ["high", "extreme"], "ocean": ["high", "extreme"]}
PROFILES = ["profile_1", "profile_2"]
BGS = [3, 4, 5]
SIGMAS = [0.0, 2.0, 4.0, 8.0]
FRAMES = 50
OUT = paths.RESULTS / "wave_diagnosis"


def flow_wavelength(flows, every=5):
    """Power-weighted mean spatial frequency (cycles/px) of the flow field -> wavelength in px."""
    f = flows[::every].astype(np.float64)
    f = f - f.mean(axis=(1, 2), keepdims=True)
    H, W = f.shape[1:3]
    P = (np.abs(np.fft.fft2(f, axes=(1, 2))) ** 2).sum(axis=(0, 3))  # (H, W), both flow components, all frames
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    P[0, 0] = 0
    return float(1 / ((P * r).sum() / P.sum()))


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "metrics.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["wave"], r["profile"], r["amplitude"], r["background"]) for r in rows}
    bgs = data.mfir_backgrounds()
    jobs = [(w, p, a, bgs[b]) for w, amps in WAVES.items() for p in PROFILES for b in BGS for a in amps]
    t0, k = time.time(), 0
    for w, p, a, bg in jobs:
        if (w, p, a, bg[:-4]) in done:
            continue
        k += 1
        frames, gt = data.load_mfir(bg, w, p, a, L=FRAMES, device="cuda")
        base = {"wave": w, "profile": p, "amplitude": a, "background": bg[:-4]}
        raw_std = to_gray(frames).std(0).mean()
        new = [{**base, "method": "mean", **all_metrics(frames.mean(0), gt, crop=4)},
               {**base, "method": "median", **all_metrics(np.median(frames, 0), gt, crop=4)}]
        wl = np.nan
        for s in SIGMAS:
            t = time.perf_counter()
            reg, tmpl, flows, hist = register(frames, flow_sigma=s)
            rt = time.perf_counter() - t
            if s == 0:
                wl = flow_wavelength(flows)
            new.append({**base, "method": f"reg_s{s:g}", "sigma": s, "runtime_s": rt,
                        "agree_ratio": float(to_gray(reg).std(0).mean() / raw_std),
                        "flow_mean_px": hist[-1].get("mean_flow", np.nan) if hist else np.nan,
                        **all_metrics(np.clip(tmpl, 0, 1), gt, crop=4)})
        for r in new:
            r["flow_wavelength_px"] = wl
        rows += new
        pd.DataFrame(rows).to_csv(csv, index=False)
        el = time.time() - t0
        left = sum((j[0], j[1], j[2], j[3][:-4]) not in done for j in jobs) - k
        print(f"[{k}, {el / 60:4.1f} min, ETA {el / k * left / 60:4.1f} min] {w} {p} {a} {bg[:-4]} "
              f"(wavelength {wl:.0f} px): " + "  ".join(f"{r['method']} {r['psnr']:.2f}" for r in new), flush=True)

    df = pd.DataFrame(rows)
    order = ["mean", "median"] + [f"reg_s{s:g}" for s in SIGMAS]
    s = df.pivot_table(index=["wave", "amplitude"], columns="method", values="psnr")[order]
    l = df.pivot_table(index=["wave", "amplitude"], columns="method", values="lpips_vgg")[order]
    s.round(3).to_csv(OUT / "summary_psnr.csv")
    l.round(4).to_csv(OUT / "summary_lpips.csv")
    print("\nPSNR (gray, crop 4):\n", s.round(2).to_string())
    print("\nLPIPS-VGG:\n", l.round(3).to_string())
    print("\nflow wavelength (px) by wave/amplitude:\n",
          df.groupby(["wave", "amplitude"]).flow_wavelength_px.mean().round(1).to_string())


if __name__ == "__main__":
    main()
