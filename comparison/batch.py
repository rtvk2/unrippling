"""Run one fixed configuration per public clip and one controlled synthetic sweep."""

import argparse
import json
from pathlib import Path
import uuid
from comparison.course import summarize
from main import parser as restoration_parser, run
from utils.data_loader import ROOT, PUBLIC


def evaluate(datasets, amplitudes, refine, frames=100, size=256):
    def execute(arguments):
        return run(restoration_parser().parse_args(arguments))

    records = []
    for dataset in datasets:
        args = ["--dataset", dataset, "--frames", str(frames), "--size", str(size)]
        record = dict(dataset=dataset, run_id=execute(args))
        if refine:
            reg = execute(args + ["--method", "fourier", "--refine", "--save-stack"])
            path = ROOT / "stage_5/results" / dataset / reg
            record["refined"] = [reg] + [
                execute(
                    args
                    + [
                        "--method",
                        method,
                        "--branch",
                        "fourier",
                        "--registered",
                        str(path),
                    ]
                )
                for method in ("rpca", "fusion")
            ]
        records.append(record)
    for amplitude in amplitudes:
        rid = execute(
            [
                "--dataset",
                "synthetic",
                "--amplitude",
                str(amplitude),
                "--frames",
                "20",
                "--size",
                "128",
            ]
        )
        records.append(dict(dataset="synthetic", run_id=rid, rms_px=amplitude))
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(PUBLIC),
        default=["checker", "middle", "cartoon", "eye", "small", "vision"],
    )
    p.add_argument("--amplitudes", nargs="*", type=float, default=[1, 2, 3, 6])
    p.add_argument("--refine", action="store_true")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    output = ROOT / "comparison/results/batches" / uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    rows = evaluate(a.datasets, a.amplitudes, a.refine, a.frames, a.size)
    (output / "runs.json").write_text(json.dumps(rows, indent=2))
    summarize(rows, output)
    print(output)
