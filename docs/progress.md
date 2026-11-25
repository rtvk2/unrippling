# Un-Rippling handoff to Codex Cloud

**Paused on 2026-11-25 during the initial audit and data preparation.** The original Stage 1–3 code is present. Two synthetic baseline scripts ran. The only new implementation file is a draft downloader. No algorithm fixes, new stage runners, pytest regression tests, real-footage experiments, or eight-method comparisons have been completed.

## Git and migration

- Repository `https://github.com/rtvk2/unrippling`. Local root `C:\Users\saipo\Desktop\sem_5\Digital Image Processing\project\unrippling`. Windows PowerShell.
- The starting tree was clean on local branch `complete-unrippling` at `487986334cf86a4451f6052b8be1a796c4083b3c` (“Stage 3 done”). At inspection, local `main` and the remote-tracking ref `origin/main` pointed to that commit. No commit, staging operation, or push occurred.
- **Transfer both local source files when migrating.** `docs/progress.md` is untracked. `data/download_data.py` is also untracked and the original `.gitignore` ignores all of `data/`. A clone of the starting commit or a plain `git diff` omits both. Downloader SHA-256 is `ad716a22dbc078aa0ae7371f10ba3bedc19c30eb3745c45aeeae0ef2eae9154e` (3,937 bytes). No tracked source, notebook, PDF, or configuration file changed.
- The baseline CSVs and PNGs listed below are ignored under `results/`. Preserve those separately if transferring the actual measurements. A Windows virtual environment is local to this machine. The new local environment, Python caches and pytest cache are disposable for migration.

## Audit coverage and existing pipeline

The starting inventory comprised 25 tracked files. There was no local `data/`, `external/`, or `results/` tree at the start. Historical figures and tables embedded in the notebook and prose in `pipeline.md` remain intact. Historical experiment CSVs mentioned there were absent.

| Original files | Inspection and use |
|---|---|
| `README.md`, `pipeline.md`, `proposal_doc.pdf`, `pyproject.toml`, `requirements.txt`, `.gitignore` | Inspected setup, package layout, experiment plan and historical findings. Read all four proposal pages, including its runtime target, parallel RPCA/fusion branches, references and evaluation goals. The PDF still has its original name. |
| `notebooks/01_baselines.ipynb` | Read cells and textual outputs. Historical Checker output used a 101 × 512 × 752 grayscale stack. Two saved figure outputs are present. The figures were identified from notebook metadata, without a pixel-level visual review or notebook rerun. |
| `src/unripple/{__init__,baselines,data,metrics,paths,registration,rpca}.py` | Traced frame loading, conversions, temporal reducers, registration, RPCA, metrics and save paths. |
| `src/unripple/third_party/{__init__,mfir_refraction}.py` | Inspected the attributed PyTorch refraction generator, including the MIT notice, fixed-amplitude profiles and clamped output. It remains intact. No MFIR run was performed. |
| `scripts/{check_james,fetch_external,run_baselines,run_registration,run_rpca,score_saved,tune_registration,test_registration_synthetic,test_rpca_synthetic}.py` | Reviewed dataset access, alignment check, split selection, timing, PNG rescoring and synthetic checks. Baseline scripts exercised below. The full per-issue audit and independent verification of historical tuning claims remain outstanding. |

Existing loaders decode 8-bit video/images into float32 in [0,1], with grayscale or RGB channel handling. James inputs currently resolve under `external/CompressiveFlows/Dataset/`. The new downloader targets `data/real/`; loader integration is pending. Other existing loaders handle MFIR physics generation, side-by-side demo GIFs and paired still/wavy recordings.

| Stage | Exact implementation status at pause |
|---|---|
| 1 | `src/unripple/baselines.py` computes temporal mean/median along axis 0. A legacy script saves images and metrics. Input-validation tests and the requested runner/output contract are pending. |
| 2 | `registration.register` computes Farneback flow from the current template to each original frame, samples at grid + flow, and updates the template. It returns the last ordered registered stack, template, fields and round history in memory. Initial mean/median choice already exists. The synthetic example improved image quality. Validity-aware updates, confidence, persisted runs and signed-field regression tests are pending. |
| 3 | `rpca.rpca` and `rpca_image` implement inexact ALM and average low-rank columns. The solver uses chunked float64 Gram accumulation and reusable buffers. The legacy runner repeats registration. Zero input fails, invalid borders are untreated and reaching the iteration cap has no explicit failure status. Direct-SVD equivalence remains to be tested. |
| 4 | Lucky Patch Fusion is described in the original proposal/planning document. No implementation or ablations exist. |
| 5 | No Python KLT/Fourier/FISTA registration branch exists. Research began on the paper and reference repository. No solver, tracker, warper integration or numerical verification has been implemented. |

The requested top-level `stage_1/` through `stage_5/`, `utils/`, `comparison/`, `main.py`, `docs/audit.md`, `docs/pipeline.md`, `docs/experiments.md` and `data/README.md` are pending. The original root `pipeline.md` is a historical plan and experiment log.

## Confirmed issues and verified behavior

All source references below refer to the starting commit. **Fixes applied to existing code: none.**

| Status | Location and evidence | Work still needed |
|---|---|---|
| Reproduced bug | `scripts/test_registration_synthetic.py:main` uses the image crop `a[c:-c,c:-c]` on EPE shaped (T,H,W) and flow shaped (T,H,W,2). Eight frames and crop four produced empty means and NaN; crop zero also yields empty slices. | Crop H/W explicitly, preserve time/components, validate crop and assert EPE on asymmetric known fields. Historical EPE numbers require recomputation. |
| Reproduced bug | `src/unripple/rpca.py:rpca`, dual initialization and initial penalty. A zero 6 × 2 matrix emitted invalid/divide-by-zero warnings, exhausted 500 iterations and reported NaN residual. | Return the exact zero solution, validate inputs, expose convergence and test zero, capped and regular cases. |
| Confirmed gap | `registration.warp` uses cubic remap with replicated borders. `register` includes all samples in its template update. No mask identifies samples outside the frame. Cubic overshoot is possible; its magnitude has not been measured. | Specify sampling support and output range, add masks, validity-aware averaging and analytical border tests. Keep vector-field sampling separate from intensity clipping. |
| Correct by inspection, partly exercised | `registration.farneback` calls OpenCV in template-to-frame order and `register` resamples original frames every round. The root planning pseudocode reverses the displayed flow arguments. | Retain the implementation direction, correct the documentation, test the warper and flow estimator independently. |
| Confirmed evaluation risks | `scripts/tune_registration.py:clips` loops over every downloaded shallow profile although the historical dev/test rule specifies `profile_6`. The test callback also scores intermediate rounds. `run_registration.py` and `run_rpca.py` include placeholder zero baseline times. `score_saved.py` reads quantized PNGs and assumes outputs of separate runs share settings. | Enforce the named profile and frozen test setting, bind artifacts to run identities, measure consistent scopes and evaluate float arrays. Actual historical leakage or mismatched runs could not be determined from this checkout. |
| Confirmed test gap | Both legacy synthetic checks are scripts with `main()`. One prints metrics, the other records a Boolean. Neither includes pytest assertions for the claimed behavior. | Convert the numerical claims into deterministic regression tests while retaining useful demos. |
| Open checks | Direct-SVD comparison, peak RPCA memory, signed EPE accuracy, native GT dimensions/alignment for the selected footage, preprocessing/crop and official MFIR score parity. Notebook prose describes grayscale where current `load_james` defaults to RGB. | Check real files and mathematical invariants before assigning further bug or performance claims. Keep internal and official metrics separately identified. |

## Baseline commands and actual outcomes

Local environment used Python 3.12.2 with system site packages, NumPy 2.5.2, SciPy 1.18.0, opencv-python 5.0.0.93 and scikit-image 0.26.0. Base Python lacked pytest. A local `.venv` with system site packages was created, and pytest 9.2.0 and lpips 0.1.4 were installed there. LPIPS weights and scores were not verified. For reference, these were the baseline commands from the repository root, before implementation changes.

```powershell
$env:PYTHONPATH = 'src'
$env:OPENBLAS_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -m pytest -q
python scripts/test_registration_synthetic.py --frames 8 --size 80 --rounds 2 --amplitudes 2 --crop 4
python scripts/test_rpca_synthetic.py
python -c "import numpy as np; from unripple.rpca import rpca; print(rpca(np.zeros((6,2)))[2][-1])"
```

- **Pytest** collected zero tests. The environment printed one Dash plugin deprecation warning. No passing pytest suite exists at this checkpoint.
- **Registration demo** ran on eight synthetic 80 × 80 frames with 2 px RMS input distortion. Cropped PSNR/SSIM were 19.6583 dB / 0.75068 for mean, 20.7218 / 0.82466 for median and 24.3155 / 0.91374 for registered average. EPE was undefined due to the crop bug. Outputs are `results/synthetic/metrics.csv`, `results/synthetic/rounds.csv`, and `results/synthetic/amp_2px/{input,frame0,mean,median,registered,comparison}.png`.
- **RPCA demo** ran all five matrix cases in float32 and float64. All ten recorded relative low-rank errors below 0.001, ranging from 5.40e-8 to 6.07e-5. Individual times ranged from 1.20 to 14.92 s. Exact case dimensions, tolerances implicit in the script, ranks, iterations, errors and timings are recoverable from the script and `results/rpca_synthetic/metrics.csv`. The separate zero-input probe returned iteration 500, rank 0, sparsity 1.0 and residual NaN.
- No real-footage reconstruction, official benchmark score, one-minute target measurement, cross-stage integration check or LPIPS measurement was produced in this session.

## Data, transfer budget and reference research

The pinned CompressiveFlows Git tree at `78d9320d50f00803b085c2a095094fc1796db241` identified the following individual assets. Sizes and Git blob hashes were checked against the manifest. Neither native video nor its paired reference was available locally for decoding.

| Upstream path under Dataset | Bytes | Git blob SHA-1 |
|---|---:|---|
| DistortedVideos/Checker.avi | 16,341,942 | 76434f2fe4eb04fb59091f2b27bd81eae513855f |
| GroundTruthImages/Checker.png | 317,842 | 2395ea9ef2de11af9bde22b6600116286e83abd2 |
| DistortedVideos/Middle.avi | 3,263,352 | baab278549104cab660e388230d32d7f6ee6a368 |
| GroundTruthImages/Middle.png | 44,408 | f6c271768abdb260c7fde72e67204e7a9113a781 |

- **Dataset transfer at handoff: 0 bytes evidenced; 19,967,544 bytes selected against a 5,000,000,000-byte cap.** A downloader invocation was requested before the interruption. At handoff there was no `data/real/` directory, lock, ledger, partial file or downloaded asset. The downloader creates that directory and records a reservation before opening a dataset connection. Dependency wheels and source metadata are separate from dataset payloads.
- The draft downloader checks existing local/upstream-clone assets, uses pinned individual file URLs, verifies expected byte counts and Git blob hashes, writes temporary files and keeps a locked cumulative ledger. An interrupted file is retried from the beginning. No transfer or interruption path has been tested. It imports `filelock`, which still needs declaring as a direct dependency. No archive is required for these four files. The legacy optional MFIR tar extractor still needs safe member-path validation.
- The authors classify **Checker as Real1 and Middle as Real2**. Middle is credited to Tian and Narasimhan. Both have reference PNG entries. Plan to add a deterministic generated sequence with known motion to obtain the synthetic counterpart, and keep the captured clips as development examples under the historical split. No synthetic/real designation was inferred from checkerboard content. Dataset redistribution rights and alignment still need verification. [James et al., Tables 1–2](https://openaccess.thecvf.com/content_ICCV_2019/papers/James_Restoration_of_Non-Rigidly_Distorted_Underwater_Images_Using_a_Combination_of_ICCV_2019_paper.pdf) · [Authors’ dataset](https://github.com/jeringeo/CompressiveFlows/tree/master/Dataset)
- The paper estimates undistorted feature positions using trajectory centers. Its CS+PEOF result adds local quadratic polynomial optical-flow refinement to CS restoration. The upstream manifest located the MATLAB tracking utility, FFT operator, CS and PEOF routines. An attempt to retrieve the pinned source bodies was interrupted; implementation-level parity and the reproduce/adapt/omit table remain open. [MATLAB reference](https://github.com/jeringeo/CompressiveFlows/tree/master/Matlab_Implementation)
- Verified MFIR's public benchmark repository and Zenodo record. Zenodo lists four parts totaling 43.8 GB. No archive part was downloaded. The existing generator and its license remain in place. Historical development used shallow `profile_6`, sorted backgrounds 0–2 and the James clips; held-out backgrounds were 8–15 on the same water, with other profiles/wave types reserved for unseen-water evaluation. No tuning or held-out scoring occurred here. [Benchmark](https://github.com/iafoss/refractive-mfir-benchmark) · [Zenodo record](https://zenodo.org/records/19390086)

## Remaining work after migration

1. Bring over both local source files and the baseline artifacts as needed. Finish `docs/audit.md`, inspect outstanding reference sources and figures, and preserve the original notebook, scripts, physics generator, attribution and historical results.
2. When data work resumes, test the downloader and its budget accounting, then validate the selected assets, provenance, licenses, frame counts, dimensions and GT alignment. Add local video/frame-folder ingestion and deterministic synthetic inputs with known motion and amplitude. Keep the full MFIR archive behind the user's separate approval.
3. Implement the requested package layout and Stage 1–3 repairs, with stage-local tests and results. Introduce the shared registration contract and validated save/reload with frame order, dataset identity, configuration and generation commands.
4. Implement and test Stage 4 scoring, normalization, top-K selection, confidence, overlap blending and edge cases. Independently verify Stage 5 sampled FFT/adjoint and complex FISTA on known signals, then integrate KLT, reference coordinates, dense motion and original-frame warping.
5. Exercise each stage, both persisted branches in separate processes, all eight main methods and separate V1/V2/V3 ablations. Save float reconstructions, PNGs, diagnostics, comparable runtimes and applicable metrics. Report failure cases and explicitly label unavailable metrics.
6. Complete the README, pipeline/audit/experiment documents, dataset instructions and progress ledger. Run all documented entry points and the final pytest suite, review the diff and report the actual artifact paths, results and bytes transferred.
