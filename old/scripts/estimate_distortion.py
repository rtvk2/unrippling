"""Can the pipeline tell, WITHOUT ground truth, how distorted an input is? (block E4 add-on)

    python scripts/estimate_distortion.py      # resumable; results/distortion/estimate.csv

For the 128 MFIR sweep clips and the 14 James clips, computes unripple.pipeline.estimate_distortion (frozen
first-round registration flow from the median, no ground truth). CPU only (rendering included) so it can run next
to distortion_analysis.py, whose RPCA needs most of the GPU. Compare against the measured (ground-truth) distortion
in results/distortion/distortion.csv afterwards.
"""

import time
import warnings

import pandas as pd

from unripple import data, paths
from unripple.pipeline import estimate_distortion

OUT = paths.RESULTS / "distortion"


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "estimate.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {r["clip"] for r in rows}
    sw = pd.read_csv(paths.RESULTS / "sweep" / "metrics.csv")
    jobs = [("james", n, None) for n in data.james_names()] + \
           [("mfir", f"{bg}_{w}_{p}_{a}", (bg, w, p, a)) for w, p, a, bg in
            sw[["wave", "profile", "amplitude", "background"]].drop_duplicates().itertuples(index=False)]
    t0 = time.time()
    for k, (ds, name, spec) in enumerate(jobs, 1):
        if name in done:
            continue
        if spec is None:
            frames = data.load_james(name)[0]
        else:
            bg, w, p, a = spec
            frames = data.load_mfir(bg + ".png", w, p, a, L=199, device=None)[0]
        t = time.perf_counter()
        est = estimate_distortion(frames)
        rows.append({"dataset": ds, "clip": name, "estimate_px": est, "estimate_s": time.perf_counter() - t})
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{k}/{len(jobs)}, {(time.time() - t0) / 60:4.1f} min] {name}: estimate {est:.1f} px", flush=True)


if __name__ == "__main__":
    main()
