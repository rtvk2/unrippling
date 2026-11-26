"""Score images that a stage already saved, with LPIPS included, without rerunning the method.

    python scripts/score_saved.py registration
    python scripts/score_saved.py rpca

For every clip in results/<stage>/<dataset>/ with a saved result, scores the median, the registered mean and
(for rpca) the RPCA image against the ground truth. Median, registered and gt images come from
results/registration/ (run_rpca.py uses the same frozen registration, so they are the same images).
Same convention as everywhere else: grayscale, crop 4. Writes results/<stage>/lpips_check.csv.
"""

import argparse

import pandas as pd

from unripple import data, paths
from unripple.metrics import all_metrics

STAGES = {"registration": ("registered", ["median", "registered"]), "rpca": ("rpca", ["median", "registered", "rpca"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--crop", type=int, default=4)
    args = ap.parse_args()

    marker, methods = STAGES[args.stage]
    stage_dir, reg_dir = paths.RESULTS / args.stage, paths.RESULTS / "registration"
    rows = []
    for path in sorted(stage_dir.glob(f"*/*_{marker}.png")):
        dataset, clip = path.parent.name, path.name[:-len(f"_{marker}.png")]
        gt_path = reg_dir / dataset / f"{clip}_gt.png"
        if not gt_path.exists():
            print(f"skip {dataset} {clip}: no ground truth in {reg_dir / dataset}")
            continue
        gt = data.read_image(gt_path, gray=False)
        for method in methods:
            src = stage_dir if method == "rpca" else reg_dir
            im = data.read_image(src / dataset / f"{clip}_{method}.png", gray=False)
            rows.append({"dataset": dataset, "clip": clip, "method": method,
                         **all_metrics(im, gt, with_lpips=True, crop=args.crop)})
        print(f"{dataset} {clip}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "lpips_check.csv", index=False)
    print("\n", df.pivot_table(index="dataset", columns="method", values=["psnr", "ssim", "lpips_vgg"]).round(3).to_string())
    if "rpca" in methods:
        p = df.pivot_table(index=["dataset", "clip"], columns="method", values="lpips_vgg")
        print("\nRPCA better LPIPS than registered:\n", (p.rpca < p.registered).groupby("dataset").agg(["sum", "count"]))


if __name__ == "__main__":
    main()
