# Un-Rippling: Pipeline Flow

Goal: take a video `frames (T, H, W[, C])` of a scene seen through wavy water and output one sharp, undistorted image, using only classical methods. Every stage is evaluated on the same data so the effect of each addition is measurable.

Status legend: **[DONE]** implemented, **[TODO]** not yet, **[STRETCH]** only if time allows.

```
 video frames
     |
     v
 [0] Load + preprocess ----------------------------------------------.
     |                                                                |
     v                                                                |
 [1] Mean / median baseline  ---> template T0, "long exposure"        |
     |                                                                |
     v                                                                |
 [2] Iterative registration to template (Farneback flow + remap)      |
     |        registered stack R (T,H,W), template T_k, flow fields   |
     |------------------------------+                                 |
     v                              v                                 |
 [3] Robust PCA (low-rank part)   [4] Lucky patch fusion  <-- ours    |
     |  = Oreifej baseline            |  = proposed method            |
     v                              v                                 |
 baseline image                  final image (+ mild sharpening)      |
     |                              |                                 |
     '--------------+---------------'                                 |
                    v                                                 |
 [6] Evaluation: PSNR, SSIM, LPIPS, NMI, rRMSE, runtime  <------------'
                    |
                    v
 [7] Experiments by distortion level, ablations, failure analysis

 [5] STRETCH: track corners (KLT) -> sparse 3D-Fourier motion (FISTA)
     replaces or initialises stage 2
```

---

## Stage 0: Data and preprocessing [DONE]

We use existing datasets only; no new videos are recorded.

**Inputs**
- James et al. clips (14 videos + ground truth), incl. Tian & Narasimhan clips.
- Shugaev benchmark: 4 wave types (ocean, shallow, sine, ripple) x 4 amplitudes (low, mid, high, extreme), generated with `data.load_mfir`.

**Work**
1. Verify the ground truth of the James clips is aligned with the wavy frames (check with phase correlation, apply a global translation if needed).
2. Decide conventions once and use them everywhere:
   - grayscale for flow/RPCA/scoring, RGB applied by warping each channel with the same flow **(fixed: loaders return RGB, scoring in grayscale)**;
   - fixed resize (256x256 for the runtime target, native size for quality tables) **(provisional, not implemented)**;
   - border crop (a few pixels) before metrics, since warping creates edge artefacts **(provisional: 4 px)**.
3. Define a **distortion level** for the James clips: mean flow magnitude between each frame and the median template (pixels). For the benchmark use the given amplitude labels.

**Outputs**: `frames` float32 in [0,1], `gt`, a distortion-level tag per clip.

Files: `src/unripple/data.py`, `paths.py`, `scripts/fetch_external.py`.

---

## Stage 1: Baselines [DONE]

- `temporal_mean(frames)`, `temporal_median(frames)`.
- Purpose: blurry long-exposure reference, and the initial template for stage 2 (median is the better start: robust to outliers).
- Run: `python scripts/run_baselines.py`, which writes `results/baselines/metrics.csv`.

Files: `src/unripple/baselines.py`, `scripts/run_baselines.py`.

---

## Stage 2: Iterative registration to a template [IN PROGRESS, week 2]

Core idea: each frame is the true scene warped by an unknown smooth flow. Estimate the flow from every frame to a template, undo it, average, and repeat.

**Algorithm**
```
template = median(frames)                  # or mean
for k in 1..K (K ~ 3-6):
    for each frame t:
        flow_t = Farneback(frame_t -> template)     # dense, cv2.calcOpticalFlowFarneback
        reg_t  = remap(frame_t, flow_t)             # cv2.remap, bicubic
    template = robust_average(reg)                  # mean or trimmed mean / median
return reg (T,H,W), template, flows
```

**Design decisions to settle and ablate**
- **Direction of flow.** Warp the frame onto the template grid (backward map) to avoid holes.
- **Template drift.** Early templates are blurry, so the flow is biased toward blur. Mitigations: start from the median; use a coarse-to-fine schedule (larger Farneback window early, smaller later); use a robust update instead of a plain mean; optionally smooth the flow temporally.
- **Farneback parameters:** `pyr_scale`, `levels`, `winsize`, `iterations`, `poly_n`, `poly_sigma`. Pick on a small validation set, not the test clips.
- **Flow regularisation.** Optional Gaussian smoothing of the flow, since water surfaces are smooth.
- **Convergence check.** Track mean flow magnitude and template change per round; stop early when they plateau.
- **Colour.** Compute flow on grayscale, warp all channels.
- **Memory/runtime.** Process frames in chunks; flow per frame is independent so it parallelises over CPU cores.

**Outputs**: registered stack, template, per-round diagnostics (flow magnitude, template PSNR vs gt if gt available).

**Check**: registered mean should beat mean/median at low amplitude. If it doesn't, debug flow direction and drift before moving on.

Files: `src/unripple/registration.py`, `scripts/run_registration.py`, `scripts/test_registration_synthetic.py`.

---

## Stage 3: Robust PCA baseline [TODO, week 3]

Core idea: registered frames are approximately the same image (low rank) plus sparse errors (glints, local misalignment).

**Algorithm (inexact ALM, Lin et al.)**
1. Build matrix `D` of shape (H*W, T), one registered frame per column.
2. Solve `min ||L||_* + lambda ||S||_1  s.t. D = L + S` with `lambda = 1/sqrt(max(H*W, T))`.
3. Loop: SVT on `D - S + Y/mu` for L, soft-threshold for S, update `Y` and `mu`.
4. Output image = temporal mean of the columns of `L` (reshape to (H, W)).

**Notes**
- Use a truncated or randomised SVD for speed (the matrix is tall and thin, so an SVD of `D^T D` or of the (T x T) Gram matrix is cheap).
- Per proposal: **do not feed the RPCA output into stage 4**, because the sparse part absorbs the frame-specific sharp detail that fusion needs.
- Validate the solver first on synthetic low-rank + sparse data with known `L`, `S` (recover error < 1e-3).
- Steps 2 + 3 together are the reproduction of Oreifej et al. and are the **classical baseline to beat**.

Planned file: `src/unripple/rpca.py`.

---

## Stage 4: Lucky patch fusion [TODO, week 3] (our extension)

Core idea: after registration, different frames are sharp in different regions (water was flat there at that moment). Pick or weight the best patches per location.

**Algorithm**
1. Take the registered stack `R` (from stage 2, not RPCA) and final template `T_k`.
2. Split into overlapping patches (e.g. 32x32, 50% overlap; ablate 16/32/64).
3. For each frame/patch compute a **quality score**, combining:
   - sharpness: gradient energy or Laplacian variance of the patch;
   - template similarity: negative patch SSD or NCC vs the template patch;
   - optional: **flow stability**, i.e. small flow magnitude and low temporal variance of the flow at the patch (the "flat water" cue).
   Normalise each term (rank or z-score within the patch location across frames) before combining with weights `alpha, beta, gamma`.
4. Keep the top-k frames per patch location (k about 10-20% of T).
5. Blend the kept patches with a score-weighted average, then combine overlapping patches with a Hann/Gaussian window so there are no seams.
6. Mild sharpening: unsharp mask (or Wiener deconvolution with a small Gaussian PSF). Keep the strength as a parameter and report results with and without.

**Ablation ladder (report all rows)**
```
mean -> median -> registered mean (stage 2) -> RPCA (stage 3)
     -> fusion (sharpness only) -> + template similarity -> + flow stability
     -> + sharpening
```
Also ablate patch size, k, and the score weights.

**Pitfalls**: selection bias (picking sharp patches can pick noisy ones, so use sharpness relative to the template), colour (apply the same selection to all channels), and seams at patch boundaries.

Planned file: `src/unripple/fusion.py`.

---

## Stage 5: Sparse motion in the Fourier domain [STRETCH, week 4]

Only start once stages 0-4 and 6 work end to end. Pick this **or** extra ablations, not both.

1. Detect strong corners in the template (`cv2.goodFeaturesToTrack`).
2. Track them through the video with KLT (`cv2.calcOpticalFlowPyrLK`), forward-backward check to drop bad tracks.
3. Model the dense motion field `u(x, y, t)` as sparse in the 3D Fourier basis. Solve `min ||c||_1  s.t. ||A c - u_tracked||_2 <= eps` where `A` samples the inverse 3D DFT at tracked points, using our own FISTA (`NumPy FFT` for the transforms, step size from the Lipschitz constant, momentum).
4. Evaluate the recovered dense field at every pixel and frame, warp, and use it in place of (or as initialisation for) stage 2's Farneback flow.
5. Compare to the MATLAB reference of James et al. on a few clips to check that your numbers are near the reported ones.

Planned file: `src/unripple/sparse_motion.py`.

---

## Stage 6: Evaluation [mostly DONE]

- Metrics: PSNR, SSIM, LPIPS-VGG (scoring only), plus NMI and relative RMSE for comparability with James et al. [1]. **[DONE]** in `metrics.py`.
- Runtime per clip, measured for 256x256 on a CPU. **[TODO]** (timing only exists in `run_baselines.py`).
- Rules for fairness: same crop, same grayscale/RGB convention, same frame count and same clip subset for every method, and report which subset of [3]'s benchmark was used when comparing to their published numbers.
- Register a global brightness/offset check: if the output is shifted relative to gt by a sub-pixel translation, PSNR/SSIM collapse. Optionally also report metrics after a global alignment, labelled as such.
- Learning-based results (DATUM, V-cache, Li et al.) are taken from the papers, never rerun.

Planned: `scripts/run_pipeline.py` that takes `--method {mean,median,register,rpca,fusion,sparse}` and writes `results/<method>/...` and `metrics.csv`.

---

## Stage 7: Experiments and analysis [TODO, weeks 4-5]

1. **Distortion sweep.** Run all methods on the benchmark at low/mid/high/extreme for each wave type (as many profiles as disk allows) and on the James clips grouped by measured distortion. Plot PSNR/SSIM/LPIPS vs amplitude.
2. **Breakdown point.** Identify the amplitude where each classical method falls below mean/median or below the grid-registration number in [3]. Show example frames, flow fields, and failure images (multi-path refraction, repeated copies, discontinuities).
3. **Gap to learned methods.** Tabulate our numbers next to the reported numbers from [1] and [3]; at medium amplitude report how much of the gap is closed rather than claiming a win.
4. **Qualitative demo.** Before/after images on the James clips, plus the James MATLAB reference results for comparison.
5. **Runtime table** for each stage (flow, RPCA, fusion).

---

## Schedule

| Week | Stages | Exit criterion |
|---|---|---|
| 1 | 0, 1, 6 (partial) | Datasets downloaded, baselines run, metrics CSV produced |
| 2 | 2 | Registered mean beats mean/median at low amplitude; flow parameters chosen |
| 3 | 3, 4 | Minimum pipeline end to end; RPCA validated; fusion beats registered mean on at least the benchmark low amplitude |
| 4 | 7 (sweeps), 5 (stretch) | Full result grid; stretch only if the grid is done |
| 5 | 6, 7 | Tables, failure analysis, runtime numbers, report |

## Order of work

Tick boxes as you go. Do the items in order; each block ends with a check that must pass before moving on.

### A. Data and baselines (stages 0, 1)
- [x] Implement loaders (`data.py`, `paths.py`) and metrics (`metrics.py`)
- [x] Implement mean/median baselines and `run_baselines.py`
- [x] Clone reference repos: `python scripts/fetch_external.py repos`
- [ ] Download MFIR benchmark subset: `python scripts/fetch_external.py mfir --types shallow --max-profiles 1`
- [ ] Check James clip ground truth is aligned with the wavy frames (phase correlation)
- [~] Fix conventions:
  - [x] Grayscale vs RGB (fixed): every loader returns RGB (`load_james`/`load_own` now default `gray=False`, as MFIR and the demo already did); methods return RGB images; flow/RPCA run on grayscale internally (`registration.to_gray`) and the flow is applied to all channels; scoring is on grayscale, `all_metrics(gray=True)` with BT.601 luma, `--score-rgb` in `run_baselines.py` to override
  - [x] Border crop (provisional): `all_metrics(crop=4)`, `--crop` in `run_baselines.py`; tested (border-only error: PSNR inf with crop, 11 dB without). 4 px is a guess, set it from the max flow magnitude in the registration `history` once registration has run on real data
  - [ ] 256x256 resize (provisional, comes from the proposal's runtime target; demo GIFs are 360x360): not implemented, native size for quality tables
- [x] Run baselines on the 3 demo GIFs (`--no-lpips`, crop 4, gray scoring): mean PSNR 17.00 / median 17.26 dB, `results/baselines/metrics.csv` written. Re-run when MFIR/James/own data exist
- [ ] Compute a distortion level for every James clip
- [ ] Run baselines on all clips and write `results/baselines/metrics.csv`
- [x] Add runtime timing to the metrics output (`runtime_s` column in `run_baselines.py`; later stages need their own timing)

### B. Registration (stage 2)
- [x] Write `registration.py`: Farneback flow to template + remap, K rounds (+ `scripts/run_registration.py`)
- [x] Test on a known synthetic warp (`scripts/test_registration_synthetic.py`, astronaut image, 40 frames, smooth zero-mean flows of 1/2/4 px RMS). Registered template beats median by 3.5/6.3/3.6 dB PSNR and mean by 6.8/8.5/5.1 dB. Template PSNR peaks at round 4 (36.6/32.7/24.9 dB), then drops ~0.2-0.4 dB by round 6 (drift), while `template_change` keeps falling. Outputs are saved in `results/synthetic/` (`metrics.csv`, `rounds.csv`, and per amplitude `amp_<A>px/{input,frame0,mean,median,registered,comparison}.png`, input = the clean source image, `skimage.data.astronaut()` at 256x256). Flow recovery is only partial: mean end-point error 0.55/1.08/2.62 px against a true magnitude of 0.84/1.69/3.46 px (about 65-75% of the flow), so the image gain is real but the flows are not accurate; check whether textureless regions (aperture problem) explain it
- [ ] **Later (needs James clips):** choose Farneback parameters, round count, flow smoothing and averaging mode on a small validation set kept out of the reported results. Write `scripts/tune_registration.py` to grid over them and report mean PSNR/SSIM per setting. More rounds are not always better (template drift), so pick the round count where template PSNR peaks.
- [x] Add template-drift mitigation (median start, coarse-to-fine, robust update)
- [x] Add convergence tracking (flow magnitude, template change per round)
- [~] **Check:** registered mean beats mean/median at low amplitude. Not yet testable (no low-amplitude data). Interim, on the 3 Ocean_extreme demo GIFs (`run_registration.py --rounds 8`, crop 4, gray scoring): registered 17.99 dB vs median 17.26 / mean 17.00 PSNR and better rRMSE (0.219 vs 0.238), but SSIM is lower than median (0.523 vs 0.549) and NMI too, so it is mixed on extreme distortion
- [x] Demo-GIF diagnostics (8 rounds): mean flow 5-13 px, max flow 90-146 px, so the provisional 4 px crop is far too small for extreme amplitude (decide crop from flow magnitude per amplitude); template PSNR peaks at round 2-4 then declines, while `template_change` is still decreasing at round 8 (0.005-0.012, above the 1e-3 `tol`), so `tol` alone does not stop at the best round; the reported "registered" number uses the last round, not the best one
- [x] `run_registration.py` now uses the same crop convention as `run_baselines.py` (default 4, passed to `all_metrics`)

### C. Robust PCA baseline (stage 3)
- [ ] Write `rpca.py`: inexact ALM with SVT and soft-thresholding
- [ ] Validate on synthetic low-rank + sparse data (error < 1e-3)
- [ ] Run on registered stacks, output = mean of the low-rank part
- [ ] **Check:** RPCA numbers reproduce the expected Oreifej-style gain over registered mean

### D. Lucky patch fusion (stage 4)
- [ ] Write `fusion.py`: overlapping patches, quality score, top-k selection, windowed blend
- [ ] Score terms: sharpness, template similarity, flow stability
- [ ] Add optional unsharp-mask sharpening
- [ ] Run the ablation ladder (mean, median, registered mean, RPCA, fusion variants, + sharpening)
- [ ] Ablate patch size, k, and score weights
- [ ] **Check:** fusion beats registered mean at benchmark low amplitude

### E. Evaluation and experiments (stages 6, 7)
- [ ] Write `scripts/run_pipeline.py` with `--method {mean,median,register,rpca,fusion,sparse}`
- [ ] Distortion sweep: all wave types x low/mid/high/extreme, plus James clips by distortion level
- [ ] Find the breakdown amplitude for each method
- [ ] Collect failure examples (multi-path refraction, repeated copies, discontinuities)
- [ ] Tabulate against reported numbers from [1] and [3] (taken from the papers, not rerun)
- [ ] Qualitative before/after figures and James MATLAB reference comparison
- [ ] Runtime table per stage (flow, RPCA, fusion)

### F. Stretch (stage 5), only if E is complete
- [ ] KLT corner tracking with forward-backward check
- [ ] FISTA sparse 3D-Fourier motion model in `sparse_motion.py`
- [ ] Plug into stage 2 as replacement or initialisation and compare

### G. Report
- [ ] Final tables and figures
- [ ] Write the report