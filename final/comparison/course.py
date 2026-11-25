"""Build the course comparison from explicitly selected stage runs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.image_io import read_image
from utils.data_loader import ROOT
from utils.visualization import montage, plt

COURSE = ROOT / "comparison/results/course"
PRIMARY = {
    "optical_flow RPCA": (3, "optical_flow"),
    "optical_flow fusion V2": (4, "optical_flow/v2"),
    "fourier RPCA": (3, "fourier"),
    "fourier fusion V2": (4, "fourier/v2"),
}
MEASURES = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips_vgg": "LPIPS-VGG",
    "total_s": "Restoration time (s)",
}


def stage_path(n, dataset, run):
    return ROOT / f"stage_{n}/results" / dataset / run


def figures(rows, output):
    public = rows[rows.dataset != "synthetic"]
    axes = (
        public.groupby("method", sort=False)[list(MEASURES)]
        .mean()
        .plot.bar(
            subplots=True, layout=(2, 2), figsize=(12, 9), legend=False, sharex=False
        )
    )
    for ax, label in zip(axes.flat, MEASURES.values()):
        ax.set(title=label, xlabel="", ylabel=label)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig = axes.flat[0].figure
    fig.suptitle("Six public sequences · equal weight per clip")
    fig.tight_layout()
    fig.savefig(output / "public_means.png", dpi=130)
    plt.close(fig)

    sweep = rows[rows.dataset == "synthetic"]
    if sweep.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (key, label) in zip(axes.flat, MEASURES.items()):
        for method, group in sweep.groupby("method", sort=False):
            group = group.sort_values("rms_px")
            ax.plot(group.rms_px, group[key], marker="o", label=method)
        ax.set(ylabel=label, xticks=sorted(sweep.rms_px.unique()))
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Imposed RMS displacement (px)")
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        fontsize=8,
    )
    fig.suptitle("Controlled distortion · one scene and motion realization")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(output / "distortion_levels.png", dpi=140)
    plt.close(fig)


def summarize(manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    metrics, extra, counts = [], [], []
    for item in manifest:
        dataset, run = item["dataset"], item["run_id"]
        path = ROOT / "comparison/results" / dataset / run
        config = json.loads((path / "config.json").read_text())
        frame_count = config["frames"]
        amplitude = config["amplitude"] if dataset == "synthetic" else None
        label = dataset if amplitude is None else f"synthetic_{amplitude:g}px"
        df = pd.read_csv(path / "metrics.csv")
        if (
            df.main.sum() != 8
            or df.method.duplicated().any()
            or not set(PRIMARY) <= set(df[df.main].method)
        ):
            raise ValueError(f"Incomplete main comparison at {path}")
        metrics.append(
            df.assign(
                dataset=dataset,
                case=label,
                rms_px=float("nan") if amplitude is None else amplitude,
                run_id=run,
            )
        )
        images = {
            method: read_image(
                stage_path(n, dataset, run) / suffix / "reconstructed.png"
            )
            for method, (n, suffix) in PRIMARY.items()
        }
        montage(images, path / "primary.png")

        ref_rows = item.get("refined", [])
        for n, ref_run in zip((5, 3, 4), ref_rows, strict=bool(ref_rows)):
            folder = stage_path(n, dataset, ref_run)
            if n == 3:
                folder /= "fourier"
            elif n == 4:
                folder /= "fourier/v2"
            meta = json.loads((folder / "config.json").read_text())
            if meta["frame_sha256"] != config["frame_sha256"] or (
                n != 5 and meta["parent_run"] != ref_rows[0]
            ):
                raise ValueError(f"Refinement input mismatch at {folder}")
            extra.append(
                pd.read_csv(folder / "metrics.csv").assign(
                    dataset=dataset, run_id=ref_run
                )
            )
        counts.append(
            dict(
                case=label,
                frames=frame_count,
                side_px=config["size"],
                primary=4,
                baseline=4,
                other_fusion=len(df) - 8,
                refinement_images=len(ref_rows),
                registered_frames=2 * frame_count,
                refined_frames=frame_count if ref_rows else 0,
            )
        )
    rows = pd.concat(metrics, ignore_index=True)
    rows.to_csv(output / "combined.csv", index=False)
    main = rows[rows.main]
    main[main.dataset != "synthetic"].groupby("method", sort=False)[
        list(MEASURES)
    ].mean().to_csv(output / "public_means.csv")
    rows[rows.method.str.contains("fusion")].to_csv(
        output / "fusion_ablation.csv", index=False
    )
    if extra:
        pd.concat(extra, ignore_index=True).to_csv(
            output / "refinement.csv", index=False
        )
    pd.DataFrame(counts).to_csv(output / "counts.csv", index=False)
    figures(main, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=COURSE / "runs.json")
    parser.add_argument("--output", type=Path, default=COURSE)
    args = parser.parse_args()
    summarize(json.loads(args.manifest.read_text()), args.output)
