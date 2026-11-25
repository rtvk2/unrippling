# Un-Rippling

Recover a stationary scene seen through moving water by combining differently distorted video frames. All restoration stages use classical computer vision. A pretrained VGG network is used exclusively to score the restored images.

```text
original video ── temporal mean and median (1)
     ├── iterative Farnebäck with median initialization (2) ── registered stack A
     └── KLT tracks + sparse 3D Fourier recovery (5) ───────── registered stack B
each stack ──┬── valid-pixel average
             ├── robust PCA (3) ────────── mean low-rank image
             └── lucky patch fusion (4) ── blended local selections
```

On the two captured development clips, Fourier registration followed by RPCA reached 23.81 dB on Checker and 25.24 dB on Middle. Fusion gave different tradeoffs on PSNR, SSIM and LPIPS. See the full eight-method tables, confidence ablations, severity study and refinement measurements in [the experiment report](docs/experiments.md).

## Setup and execution

Use Python 3.10+ in a virtual environment. Activate it with `source .venv/bin/activate` on Linux or `.venv\Scripts\activate` on Windows. Every command below is run from inside this `final/` folder, which is self-contained and imports no Python code from outside it.

```sh
cd final
python -m venv .venv
# Activate before installing
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python data/download_data.py
python -m pytest
python main.py --dataset checker --method all --frames 100 --size 256 --save-stack
python main.py --dataset middle --method all --frames 100 --size 256 --save-stack
python -m comparison.compare
```

The first LPIPS call downloads VGG weights. Use `--no-lpips` for a quick offline run after installing dependencies. Limit BLAS and OpenMP to two threads to reproduce cloud timing. On Linux set `OPENBLAS_NUM_THREADS=2` and `OMP_NUM_THREADS=2` in the environment.

```sh
python -m stage_1.run_stage_1 --dataset checker
python -m stage_2.run_stage_2 --dataset checker --save-stack
python -m stage_3.run_stage_3 --dataset checker --branch optical_flow
python -m stage_4.run_stage_4 --dataset middle --branch fourier --variant 3
python -m stage_5.run_stage_5 --dataset middle --save-stack
python -m pytest stage_4/
```

Standalone RPCA and fusion can compute the chosen registration or accept its persisted outputs via `--registered stage_5/results/middle/RUN_ID --branch fourier`. Match the dataset, frame count and side length to the parent run. Each run prints its ID. All stage commands accept `--help`. Use `--source PATH` for video or a frame folder; use `--dataset synthetic --amplitude 3 --size 128 --frames 20` for generated distortions. Setting `--fraction` to at most 1/T selects one patch per location. Main comparison uses V2 for both branches and also measures the available V1/V3 variants.

## Repository and outputs

| Path | Contents |
|---|---|
| stage_1/ through stage_5/ | Each contains the algorithm, a run_stage_N entry point, pytest tests and results/. Stage 2 holds the shared pull warper. Stage 5 includes tracker, FISTA solver and dense field recovery. |
| main.py, utils/ | Orchestration, decoding, image I/O, generated motion, metrics and visualization. |
| comparison/ | compare.py regenerates plots; results/ holds run-specific images, metrics, configurations and plots. The two final captured runs and compact synthetic tables are preserved. |
| data/ | Pinned downloader (download_data.py), dataset provenance, data/real/ for the downloaded clips and data/synthetic/. |
| docs/ | Proposal, handoff, audit, mathematical description and measured experiments. |

Outside `final/`, the repository root keeps the historical development material: `src/unripple/` (original dataset loaders, metric APIs and the licensed MFIR refraction generator), `scripts/` (earlier experiments and baseline verification), `notebooks/` and the original experiment log `pipeline.md`. Historical scores use different protocols. Nothing in `final/` imports them.

Each stage writes float reconstructions and viewable PNGs under its own results/dataset/run directory. RPCA and fusion have branch subdirectories; fusion also separates variants. Registration can persist one frame, validity map and xy field per input time index, with confidence when available. Shared registration time is included once in each downstream method's total. Large data and intermediate stacks are locally generated.

The Fourier model assumes smooth motion observed at trackable corners. Sparse tracks and repeated texture weaken recovery. Optical flow can retain template bias, and high gradient energy can favor a distorted edge. These development clips provide no held-out generalization estimate. See [dataset provenance](data/README.md), [mathematics and research references](docs/pipeline.md), [audit](docs/audit.md) and [original proposal and bibliography](docs/proposal.pdf).
