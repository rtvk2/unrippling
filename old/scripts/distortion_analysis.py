"""Block E4: score vs MEASURED distortion, and the James clips grouped by distortion level.

    python scripts/distortion_analysis.py          # resumable; results/distortion/

Distortion of a clip = how far the water moves the scene: Farneback flow (large capture range) from the ground truth
to every 4th frame, magnitude in pixels, central region (16 px border dropped); we report the mean and the 95th
percentile. Diagnostic only (uses the ground truth, nothing is tuned). It saturates when the water folds the image
(extreme ripple/sine), where no flow describes the frame; those clips read lower than their true distortion.
James clips: also contains the small fixed offset between their ground truth and the video (james_check.csv).

MFIR: the 128 sweep clips (scores from results/sweep/metrics.csv, our protocol). James: all 14 clips, every method run
here with unripple.pipeline (frozen settings). Outputs: distortion.csv, james_scores.csv, james_groups.csv,
score_vs_distortion.png, gain_vs_distortion.png.
"""

import warnings

import cv2
import numpy as np
import pandas as pd

from unripple import data, paths
from unripple.pipeline import default_device, restore
from unripple.registration import _u8, to_gray

OUT = paths.RESULTS / "distortion"
BIG = dict(pyr_scale=0.5, levels=6, iterations=5, poly_n=7, poly_sigma=1.5, winsize=41, flags=0)
# sweep method name -> pipeline method name
MAP = {"median": "median", "registered": "register", "rpca": "rpca", "realign3+fusion": "realign_fusion"}
LABEL = {"median": "median", "register": "registration", "rpca": "RPCA", "realign_fusion": "final (realign+fusion)"}
COLOR = {"median": "#2a78d6", "register": "#eb6834", "rpca": "#1baf7a", "realign_fusion": "#eda100"}
MARK = {"median": "o", "register": "s", "rpca": "^", "realign_fusion": "D"}
WAVE_COLOR = {"shallow": "#2a78d6", "ocean": "#eb6834", "ripple": "#1baf7a", "sine": "#eda100"}


def distortion(frames, gt):
    g = _u8(to_gray(gt[None])[0])
    m = np.stack([np.linalg.norm(cv2.calcOpticalFlowFarneback(g, _u8(f), None, **BIG), axis=-1)[16:-16, 16:-16]
                  for f in to_gray(frames[::4])])
    return float(m.mean()), float(np.percentile(m, 95))


def measure():
    csv = OUT / "distortion.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["dataset"], r["clip"]) for r in rows}
    sw = pd.read_csv(paths.RESULTS / "sweep" / "metrics.csv")
    clips = sw[["wave", "profile", "amplitude", "background"]].drop_duplicates().itertuples(index=False)
    jrows = pd.read_csv(OUT / "james_scores.csv").to_dict("records") if (OUT / "james_scores.csv").exists() else []
    for n in data.james_names():
        if ("james", n) in done:
            continue
        frames, gt = data.load_james(n)
        ims, _ = restore(frames, methods=list(LABEL))
        from unripple.metrics import all_metrics
        jrows += [{"clip": n, "method": m, **all_metrics(ims[m], gt, with_lpips=True, crop=4)} for m in LABEL]
        pd.DataFrame(jrows).to_csv(OUT / "james_scores.csv", index=False)
        d = distortion(frames, gt)
        rows.append({"dataset": "james", "clip": n, "wave": "james", "amplitude": "", "dist_mean": d[0], "dist_p95": d[1]})
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"james {n}: distortion {d[0]:.1f} px (p95 {d[1]:.1f})", flush=True)
    for k, (w, p, a, bg) in enumerate(clips, 1):
        name = f"{bg}_{w}_{p}_{a}"
        if ("mfir", name) in done:
            continue
        frames, gt = data.load_mfir(bg + ".png", w, p, a, L=199, device=default_device())
        d = distortion(frames, gt)
        rows.append({"dataset": "mfir", "clip": name, "wave": w, "amplitude": a, "dist_mean": d[0], "dist_p95": d[1]})
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{k}/128] {name}: distortion {d[0]:.1f} px (p95 {d[1]:.1f})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(jrows)


def scores(dist, js):
    sw = pd.read_csv(paths.RESULTS / "sweep" / "metrics.csv")
    sw = sw[sw.method.isin(MAP)].assign(method=lambda d: d.method.map(MAP),
                                        clip=lambda d: d.background + "_" + d.wave + "_" + d.profile + "_" + d.amplitude)
    m = sw[["clip", "method", "psnr", "lpips_vgg"]].merge(dist[dist.dataset == "mfir"], on="clip")
    j = js[["clip", "method", "psnr", "lpips_vgg"]].merge(dist[dist.dataset == "james"], on="clip")
    return m, j


def plots(m, j):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6})

    # 1) score vs distortion: MFIR binned means (8 equal-count bins), James per clip; same method colours/markers
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for col, (metric, name) in enumerate([("psnr", "PSNR (dB, higher is better)"), ("lpips_vgg", "LPIPS-VGG (lower is better)")]):
        for row, (df, title) in enumerate([(m, "MFIR sweep, 128 clips (mean per distortion bin)"), (j, "James real tank clips (one point per clip)")]):
            a = ax[row, col]
            for meth in LABEL:
                d = df[df.method == meth].sort_values("dist_mean")
                if row == 0:
                    d = d.assign(bin=pd.qcut(d.dist_mean, 8, labels=False)).groupby("bin")[["dist_mean", metric]].mean()
                a.plot(d.dist_mean, d[metric], marker=MARK[meth], ms=7, lw=2 if row == 0 else 0, color=COLOR[meth],
                       label=LABEL[meth],
                       markeredgecolor="white", markeredgewidth=1)
            a.set_title(title, fontsize=10, loc="left")
            a.set_ylabel(name)
            if row == 1:
                a.set_xlabel("measured distortion: mean displacement ground truth -> frame (px)")
    ax[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_distortion.png", dpi=150)
    plt.close(fig)

    # 2) final minus median vs distortion, per wave type: where does the final method stop helping?
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, (metric, name, sign) in enumerate([("psnr", "PSNR gain over median (dB)", 1), ("lpips_vgg", "LPIPS gain over median (median - final)", -1)]):
        a = ax[col]
        for src, color, lab in [(m, None, None), (j, "#6250d6", "james")]:
            p = src.pivot_table(index=["clip", "wave", "dist_mean"], columns="method", values=metric).reset_index()
            p["gain"] = sign * (p["realign_fusion"] - p["median"])
            for w, d in p.groupby("wave"):
                a.scatter(d.dist_mean, d.gain, s=36, color=WAVE_COLOR.get(w, color), label=w if col == 0 else None,
                          edgecolor="white", linewidth=0.8, marker="D" if w == "james" else "o", zorder=3)
        a.axhline(0, color="#555", lw=1)
        a.set_xlabel("measured distortion (px)")
        a.set_ylabel(name)
    ax[0].legend(frameon=False, title="wave type", loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "gain_vs_distortion.png", dpi=150)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    dist, js = measure()
    m, j = scores(dist, js)
    plots(m, j)

    # James grouped by measured distortion (terciles)
    jd = dist[dist.dataset == "james"].sort_values("dist_mean")
    jd["group"] = pd.qcut(jd.dist_mean.rank(method="first"), 3, labels=["low", "mid", "high"])
    g = j.merge(jd[["clip", "group"]], on="clip")
    tab = g.pivot_table(index="group", columns="method", values=["psnr", "lpips_vgg"])[
        [(v, k) for v in ("psnr", "lpips_vgg") for k in LABEL]]
    tab.round(3).to_csv(OUT / "james_groups.csv")
    print("\nJames clips by measured distortion:")
    for grp, d in jd.groupby("group"):
        print(f"  {grp}: " + ", ".join(f"{r.clip} {r.dist_mean:.1f}" for r in d.itertuples()))
    print(tab.round(3).to_string())

    # MFIR: distortion by wave x amplitude, and final - median per bin
    print("\nMFIR measured distortion (mean px) by wave x amplitude:")
    print(dist[dist.dataset == "mfir"].pivot_table(index="wave", columns="amplitude", values="dist_mean")[
        ["low", "mid", "high", "extreme"]].round(1).to_string())
    p = m.pivot_table(index=["clip", "dist_mean"], columns="method", values="psnr").reset_index()
    p["bin"] = pd.cut(p.dist_mean, [0, 2, 4, 6, 8, 12, 100])
    print("\nMFIR PSNR by measured distortion bin (px):")
    print(p.groupby("bin", observed=True)[list(LABEL)].agg("mean").assign(n=p.groupby("bin", observed=True).size()).round(2).to_string())


if __name__ == "__main__":
    main()
