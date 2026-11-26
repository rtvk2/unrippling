"""Restore one video or ordered image directory with classical reconstruction methods."""

import argparse
import hashlib
import json
import platform
import sys
import uuid
import warnings
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from comparison.compare import charts
from stage_1.baselines import temporal_mean, temporal_median
from stage_2.registration import register as optical
from stage_3.rpca import rpca_image
from stage_4.fusion import fuse
from stage_5.fourier_motion import register as fourier
from utils.data_loader import ROOT, PUBLIC, load
from utils.image_io import save_image
from utils.metrics import measure
from utils.visualization import montage, motion_figure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=sorted([*PUBLIC, "synthetic"]), default="checker"
    )
    p.add_argument("--source", help="custom video or lexically sorted frame directory")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--crop", type=int, default=4)
    p.add_argument(
        "--method",
        choices=["all", "baselines", "registration", "rpca", "fusion", "fourier"],
        default="all",
    )
    p.add_argument(
        "--branch", choices=["optical_flow", "fourier"], default="optical_flow"
    )
    p.add_argument(
        "--registered",
        type=Path,
        help="reuse a stage 2/5 directory made with --save-stack",
    )
    p.add_argument("--save-stack", action="store_true")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--flow-sigma", type=float, default=8)
    p.add_argument("--grid-step", type=int, default=8)
    p.add_argument("--lam", type=float, default=0.06, help="Fourier L1 weight")
    p.add_argument("--iterations", type=int, default=500, help="Fourier iteration cap")
    p.add_argument(
        "--refine",
        action="store_true",
        help="optional residual polynomial flow after Fourier recovery",
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=2.0,
        help="generated synthetic RMS displacement in pixels",
    )
    p.add_argument("--rpca-lambda", type=float, default=None)
    p.add_argument("--rpca-iterations", type=int, default=200)
    p.add_argument("--save-components", action="store_true")
    p.add_argument("--init", choices=["mean", "median"], default="median")
    p.add_argument("--rpca-tol", type=float, default=1e-4)
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--variant", type=int, choices=[1, 2, 3], default=2)
    p.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=(0.4, 0.4, 0.2),
        metavar=("SHARP", "SIM", "CONF"),
    )
    p.add_argument("--sharpen", type=float, default=0)
    p.add_argument("--no-lpips", action="store_true")
    return p


def run(a):
    cv2.setNumThreads(1)
    start = perf_counter()
    frames, gt = load(a.dataset, a.frames, a.size, a.source, a.amplitude)
    load_s = perf_counter() - start
    if not a.no_lpips and gt is not None and a.size - 2 * a.crop < 32:
        raise ValueError(
            "LPIPS needs a larger retained crop; use --no-lpips for small tests"
        )
    dataset = Path(a.source).stem if a.source else a.dataset
    run_id = uuid.uuid4().hex[:12]
    config = dict(
        vars(a),
        run_id=run_id,
        frame_shape=frames.shape,
        frame_order="first N decoded frames, ascending time",
        frame_sha256=hashlib.sha256(frames).hexdigest(),
        load_s=load_s,
        python=platform.python_version(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        argv=sys.argv,
        reference_available=gt is not None,
    )
    rows, images = [], {"Input frame 0": frames[0]}
    if gt is not None:
        images["Reference"] = gt

    def directory(stage, *parts):
        path = ROOT / stage / "results" / dataset / run_id
        path = path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config, default=str, indent=2))
        return path

    def record(
        stage,
        method,
        image,
        seconds,
        shared=0.0,
        main=True,
        parts=(),
        name="reconstructed",
    ):
        path = directory(stage, *parts)
        image = image.clip(0, 1)
        save_image(path / (name + ".png"), image)
        np.save(path / (name + ".npy"), image)
        row = dict(
            method=method,
            main=main,
            stage_s=seconds,
            registration_s=shared,
            total_s=seconds + shared,
            decode_and_restore_s=load_s + seconds + shared,
            **measure(image, gt, a.crop, not a.no_lpips),
        )
        rows.append(row)
        pd.DataFrame(
            [r for r in rows if r["method"] in ("mean", "median")]
            if stage == "stage_1"
            else [row]
        ).to_csv(path / "metrics.csv", index=False)
        if main:
            images[method] = image
        print(method, row, flush=True)
        return path

    if a.method in ("all", "baselines"):
        for name, fn in [("mean", temporal_mean), ("median", temporal_median)]:
            t = perf_counter()
            im = fn(frames)
            record("stage_1", name, im, perf_counter() - t, name=name)
    branches = {
        "all": ["optical_flow", "fourier"],
        "registration": ["optical_flow"],
        "fourier": ["fourier"],
        "fusion": [a.branch],
        "rpca": [a.branch],
        "baselines": [],
    }
    if a.registered and a.method not in ("rpca", "fusion"):
        raise ValueError("--registered applies to rpca or fusion")
    for branch in branches[a.method]:
        is_optical = branch == "optical_flow"
        stage = "stage_2" if is_optical else "stage_5"
        if a.registered:
            path = a.registered
            meta = json.loads((path / "config.json").read_text())
            if (
                meta["frame_sha256"] != config["frame_sha256"]
                or meta.get("registration_branch") != branch
            ):
                raise ValueError("registered artifact input or branch mismatch")
            if not (path / "registered_frames.npy").exists():
                raise ValueError(
                    "missing registered frames; generate the registration with --save-stack"
                )
            stack, template, valid = [
                np.load(path / (n + ".npy"))
                for n in ("registered_frames", "reconstructed", "valid")
            ]
            confidence = (
                np.load(path / "confidence.npy")
                if (path / "confidence.npy").is_file()
                else None
            )
            if (
                stack.shape != frames.shape
                or template.shape != frames.shape[1:]
                or valid.shape != frames.shape[:3]
                or valid.dtype != bool
            ):
                raise ValueError(
                    "inconsistent registered artifact dimensions or mask type"
                )
            if confidence is not None and confidence.shape != valid.shape:
                raise ValueError("inconsistent persisted confidence shape")
            config["registration_branch"] = branch
            reg_s = float(pd.read_csv(path / "metrics.csv").stage_s.iloc[0])
            config["parent_run"] = meta["run_id"]
        else:
            tracks = []
            t = perf_counter()
            if is_optical:
                result = optical(
                    frames,
                    rounds=a.rounds,
                    init=a.init,
                    flow_sigma=a.flow_sigma,
                    workers=2,
                    diagnostics=True,
                )
            else:
                result = fourier(
                    frames,
                    a.grid_step,
                    a.lam,
                    a.iterations,
                    diagnostics=True,
                    refine=a.refine,
                    on_tracks=tracks.append,
                )
            stack, template, fields, history, valid, confidence = result
            reg_s = perf_counter() - t
            config["registration_branch"] = branch
            path = record(stage, branch + " average", template, reg_s)
            pd.DataFrame(history).to_csv(path / "history.csv", index=False)
            motion_figure(
                template,
                fields[0],
                valid,
                path / "motion.png",
                tracks[0] if tracks else None,
            )
            if confidence is not None:
                save_image(path / "confidence_map.png", confidence.mean(axis=0))
            if a.save_stack:
                for name, arr in [
                    ("registered_frames", stack),
                    ("motion_fields", fields),
                    ("valid", valid),
                    ("confidence", confidence),
                ]:
                    if arr is not None:
                        np.save(path / (name + ".npy"), arr)
        if a.method in ("all", "rpca"):
            t = perf_counter()
            filled = np.where(valid, stack, template)
            image, low, sparse, hist = rpca_image(
                filled, tol=a.rpca_tol, lam=a.rpca_lambda, max_iter=a.rpca_iterations
            )
            path = record(
                "stage_3",
                branch + " RPCA",
                image,
                perf_counter() - t,
                reg_s,
                parts=(branch,),
            )
            pd.DataFrame(hist).to_csv(path / "history.csv", index=False)
            if not hist[-1]["converged"]:
                warnings.warn("RPCA reached its iteration cap; inspect history.csv")
            if a.save_components:
                np.save(path / "low_rank.npy", low)
                np.save(path / "sparse.npy", sparse)
            del low, sparse, filled
        if a.method in ("all", "fusion"):
            variants = (1, 2, 3) if confidence is not None else (1, 2)
            for v in variants if a.method == "all" else [a.variant]:
                t = perf_counter()
                image, idx = fuse(
                    stack,
                    template,
                    valid,
                    confidence,
                    v,
                    a.patch,
                    a.stride,
                    a.fraction,
                    weights=a.weights,
                    sharpen=a.sharpen,
                )
                path = record(
                    "stage_4",
                    f"{branch} fusion V{v}",
                    image,
                    perf_counter() - t,
                    reg_s,
                    main=v == 2 if a.method == "all" else True,
                    parts=(branch, f"v{v}"),
                )
                if confidence is not None and v == 3:
                    save_image(path / "confidence_map.png", confidence.mean(axis=0))
                np.save(path / "patch_selection.npy", idx)
                save_image(
                    path / "patch_selection.png", (idx[:, :, 0] + 1) / len(stack)
                )
    if a.method == "all":
        config.pop("registration_branch", None)
        config["elapsed_before_comparison_s"] = perf_counter() - start
        path = directory("comparison")
        pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)
        from comparison.course import PRIMARY

        montage({key: images[key] for key in PRIMARY}, path / "primary.png")
        montage(images, path / "comparison.png")
        save_image(path / "input.png", frames[0])
        if gt is not None:
            save_image(path / "reference.png", gt)
        charts(path)
    return run_id


def entry(stage=None):
    p = parser()
    a = p.parse_args()
    if stage:
        a.method = stage
    try:
        print("Run", run(a))
    except (ValueError, FileNotFoundError, MemoryError, ImportError) as err:
        p.exit(2, f"Experiment failed: {err}\n")


if __name__ == "__main__":
    entry()
