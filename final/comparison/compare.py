"""Regenerate comparison figures from one experiment's measured rows."""

import argparse
from pathlib import Path
import pandas as pd
from utils.visualization import plt
from utils.data_loader import ROOT


def charts(path):
    df = pd.read_csv(path / "metrics.csv")
    df[df.method.str.contains("fusion")].to_csv(path / "ablation.csv", index=False)
    main = df[df.main]
    for key, label in [
        ("psnr", "PSNR (dB)"),
        ("ssim", "SSIM"),
        ("lpips_vgg", "LPIPS (VGG)"),
        ("total_s", "Restoration time (s)"),
    ]:
        if key not in main or main[key].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(main.method, main[key])
        ax.set(xlabel=label, title=path.parent.name.capitalize())
        fig.tight_layout()
        fig.savefig(path / (key + ".png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="?", type=Path)
    args = p.parse_args()
    runs = (
        [args.run]
        if args.run
        else [p.parent for p in (ROOT / "comparison/results").glob("*/*/metrics.csv")]
    )
    if not runs:
        p.error("run python main.py --method all first")
    for run in runs:
        charts(run)
