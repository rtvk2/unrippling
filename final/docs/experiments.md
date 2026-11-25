# Cloud experiments

## Development protocol

Both captured clips used the first 100 of 101 frames and the same center-square resize to 256 × 256. Metrics use the inner 248 × 248 floating-point grayscale image. VGG-LPIPS receives three replicated channels in [−1,1]. No optimizer or selector reads the supplied reference. The main fusion row uses V2, fixed before comparing the variants. Both sources are development examples. No final-test or unseen-water samples entered parameter selection. Source attribution, native dimensions and transfer bytes are in [data/README.md](../data/README.md).

Real1 has a still-water reference. The Real2 paper and source provide a matching text reference. At working resolution the zero-shift median/reference correlations were 0.875 and 0.833. A diagnostic ±10-pixel search with quarter-pixel refinement (16-pixel interior) favored moving the median by (+0.75,+1.00) pixels for Checker and (0,+0.50) for Middle. The optical/Fourier average diagnostics favored (+0.50,+1.00)/(−0.25,+0.25) on Checker and (+0.25,+0.50) for both on Middle. This method dependence on Checker is consistent with template bias. Both montages show the same scene and ordering. All reported methods use the unshifted references. Small score differences can be sensitive to residual alignment. No perfectly registered clean coordinate system has been independently measured.

Hardware was an AMD EPYC 9V74 cloud host with a two-vCPU quota, 8 GiB cgroup memory and about 7.5 GiB available disk at startup. CPU only, two BLAS/OpenMP threads, one OpenCV thread and two frame workers. Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, OpenCV headless 5.1.0.95, scikit-image 0.26.0, torch 2.15.0+cpu, torchvision 0.29.2+cpu and lpips 0.1.4. The first evaluation downloaded standard ImageNet VGG16 weights; weights were cached for the measured runs.

Stage seconds time reconstruction on decoded arrays. Total seconds add the shared registration once to each downstream method. Registration includes confidence diagnostics. The full-process time additionally includes imports, decoding, metric evaluation, image/array I/O, ablations and comparison figures. Stage times exclude those operations. RPCA invalid samples receive the final registration reference. No sharpening or polynomial refinement is used in the main comparison.

## Checker

Run `9d47ebe6af67` took 38.03 s wall time and peaked at 1524 MiB RSS for the full process. Per-image scores and stage timings follow. [Comparison image](../comparison/results/checker/9d47ebe6af67/comparison.png) · [Full-precision CSV](../comparison/results/checker/9d47ebe6af67/metrics.csv) · [Configuration](../comparison/results/checker/9d47ebe6af67/config.json)

| Method | PSNR (dB) | SSIM | LPIPS ↓ | Stage (s) | Total (s) |
|---|---|---|---|---|---|
| mean | 16.5029 | 0.5792 | 0.5466 | 0.0040 | 0.0040 |
| median | 16.9738 | 0.7403 | 0.4005 | 0.1435 | 0.1435 |
| optical_flow average | 18.6537 | 0.8010 | 0.0860 | 6.2522 | 6.2522 |
| optical_flow RPCA | 18.6081 | 0.7993 | 0.0796 | 2.3355 | 8.5877 |
| optical_flow fusion V2 | 18.4184 | 0.7937 | 0.0740 | 0.2867 | 6.5389 |
| fourier average | 23.4291 | 0.9175 | 0.0913 | 8.9954 | 8.9954 |
| fourier RPCA | 23.8101 | 0.9274 | 0.0731 | 2.5224 | 11.5178 |
| fourier fusion V2 | 22.2125 | 0.9025 | 0.0934 | 0.2373 | 9.2327 |

### Fusion ablation

| Method | PSNR (dB) | SSIM | LPIPS ↓ | Stage (s) | Total (s) |
|---|---|---|---|---|---|
| optical_flow fusion V1 | 18.4076 | 0.7934 | 0.0741 | 0.2335 | 6.4857 |
| optical_flow fusion V2 | 18.4184 | 0.7937 | 0.0740 | 0.2867 | 6.5389 |
| optical_flow fusion V3 | 18.4215 | 0.7939 | 0.0739 | 0.2919 | 6.5441 |
| fourier fusion V1 | 21.4004 | 0.8926 | 0.1060 | 0.2325 | 9.2279 |
| fourier fusion V2 | 22.2125 | 0.9025 | 0.0934 | 0.2373 | 9.2327 |
| fourier fusion V3 | 22.9190 | 0.9132 | 0.0826 | 0.3665 | 9.3619 |

Fourier retained 306 trajectories and 25,300 distinct space-time observations. The full xy solutions met stationarity tolerance at iterations 251 and 275. All four half-fit solves and both downstream RPCA solves converged.

## Middle

Run `4d86b0ef1cba` took 35.38 s wall time and peaked at 1526 MiB RSS for the full process. Per-image scores and stage timings follow. [Comparison image](../comparison/results/middle/4d86b0ef1cba/comparison.png) · [Full-precision CSV](../comparison/results/middle/4d86b0ef1cba/metrics.csv) · [Configuration](../comparison/results/middle/4d86b0ef1cba/config.json)

| Method | PSNR (dB) | SSIM | LPIPS ↓ | Stage (s) | Total (s) |
|---|---|---|---|---|---|
| mean | 19.2113 | 0.4495 | 0.5756 | 0.0041 | 0.0041 |
| median | 18.7589 | 0.4977 | 0.4668 | 0.1468 | 0.1468 |
| optical_flow average | 22.6752 | 0.7979 | 0.1672 | 6.2146 | 6.2146 |
| optical_flow RPCA | 22.5378 | 0.8049 | 0.1547 | 2.4154 | 8.6300 |
| optical_flow fusion V2 | 22.0135 | 0.7858 | 0.1475 | 0.2494 | 6.4640 |
| fourier average | 24.4203 | 0.8170 | 0.1900 | 9.3193 | 9.3193 |
| fourier RPCA | 25.2432 | 0.8542 | 0.1589 | 2.2636 | 11.5829 |
| fourier fusion V2 | 23.1198 | 0.7864 | 0.1910 | 0.2386 | 9.5579 |

### Fusion ablation

| Method | PSNR (dB) | SSIM | LPIPS ↓ | Stage (s) | Total (s) |
|---|---|---|---|---|---|
| optical_flow fusion V1 | 21.9813 | 0.7791 | 0.1527 | 0.2576 | 6.4722 |
| optical_flow fusion V2 | 22.0135 | 0.7858 | 0.1475 | 0.2494 | 6.4640 |
| optical_flow fusion V3 | 22.1109 | 0.7900 | 0.1446 | 0.2774 | 6.4920 |
| fourier fusion V1 | 21.4796 | 0.7227 | 0.2294 | 0.2396 | 9.5589 |
| fourier fusion V2 | 23.1198 | 0.7864 | 0.1910 | 0.2386 | 9.5579 |
| fourier fusion V3 | 23.6319 | 0.8064 | 0.1761 | 0.3100 | 9.6293 |

Fourier retained 404 trajectories and 27,883 distinct space-time observations. The full xy solutions met stationarity tolerance at iterations 194 and 203. All four half-fit solves and both downstream RPCA solves converged.

## Interpretation

Fourier registration increased averaged-image PSNR and SSIM over Farnebäck on both captures. Fourier RPCA gave the highest PSNR and SSIM among the eight prespecified outputs. V2 patch selection lowered both measures relative to averaging on each branch. Adding similarity to sharpness helped particularly on Fourier-registered Middle. V3 improved over V2 on all three metrics on both branches, but still lost PSNR to the corresponding average. Farnebäck fusion improved LPIPS against Farnebäck averaging on both clips. The Checker montage shows a relatively straight Farnebäck grid alongside a lower pixel score, and residual border warping on the Fourier branch. On Middle the Fourier RPCA output preserves readable letter shapes. These examples support metric-specific conclusions, not a universal ordering.

### Optional residual polynomial refinement

Separate stage 5 runs composed residual polynomial flow with the full Fourier solution. These runs omit split-confidence refits, so their timing has a different scope. They measure the average of the refined stack.

| Clip | PSNR | SSIM | LPIPS | Stage (s) |
|---|---|---|---|---|
| checker | 23.8649 | 0.9241 | 0.0560 | 6.2101 |
| middle | 26.7961 | 0.8934 | 0.1386 | 10.4387 |

Refinement increased average PSNR on both captured clips. It also improved LPIPS and SSIM against plain Fourier averaging. The eight-method table keeps the pure Fourier branch so the effect of that extra operation remains measurable.

## Controlled severity sweep

Generated sequences used the same cameraman scene, random seed 0, 20 frames at 128 × 128, and a Gaussian spatial correlation parameter of eight pixels. Smooth random displacement fields were temporally centered and scaled to 1, 3 and 6 px RMS. The simulator is a geometric warp; it omits caustics, multipath and exposure effects. Entries below are PSNR in dB. Full PSNR/SSIM/LPIPS and timing rows are in each run CSV.

| Method | 1 px | 3 px | 6 px |
|---|---|---|---|
| mean | 29.2203 | 22.7302 | 20.0948 |
| median | 31.4400 | 23.8148 | 20.8689 |
| optical_flow average | 33.6421 | 26.7822 | 22.6355 |
| optical_flow RPCA | 33.6469 | 26.8860 | 22.5442 |
| optical_flow fusion V2 | 31.8604 | 24.2309 | 20.1403 |
| fourier average | 33.0605 | 25.3567 | 20.1049 |
| fourier RPCA | 33.0369 | 25.6681 | 20.4997 |
| fourier fusion V2 | 30.8163 | 21.7763 | 16.8538 |

Reliable Fourier track counts fell from 99 to 61 to 6. At 6 px there were too few tracks for the split diagnostic, so Fourier V3 is unavailable. Both full Fourier components reached 500 iterations with stationarity 1.07e−4 and 1.35e−4 against a 1e−4 threshold. The last iterates are reported. With sparse support, a stable reference and local detail cannot be recovered throughout the image. Fourier fusion fell below the original mean; optical averaging still improved it. This establishes a failure on one generated scene and seed, not a calibrated physical wave threshold. An earlier 250-iteration checker run also hit its cap. The final cap of 500 was chosen for numerical convergence on development inputs.

## Reproduction and verification

After installation as in README.md, execute from inside `final/`. On Windows set the thread environment variables using PowerShell before the same Python commands.

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python data/download_data.py
python -m pytest -q
python main.py --dataset checker --method all --frames 100 --size 256 --save-stack
python main.py --dataset middle --method all --frames 100 --size 256 --save-stack
python main.py --dataset synthetic --amplitude 1 --size 128 --frames 20
python main.py --dataset synthetic --amplitude 3 --size 128 --frames 20
python main.py --dataset synthetic --amplitude 6 --size 128 --frames 20
python -m stage_5.run_stage_5 --dataset checker --refine
python -m stage_5.run_stage_5 --dataset middle --refine
python -m comparison.compare
```

The two legacy synthetic verifiers belong to the historical development code outside `final/`. They reach the stage implementations through the `src/unripple` compatibility imports, so run them from the repository root with both folders on the path (`PYTHONPATH=src:final`, or `src;final` on Windows). They write to the root `results/` folder.

```sh
python scripts/test_rpca_synthetic.py
python scripts/test_registration_synthetic.py --frames 8 --size 80 --rounds 2 --amplitudes 1 2 4 --crop 4
```

The full suite passed 27 tests. Stage counts were 8, 6, 5, 4 and 4. The environment emitted one pytest plugin deprecation warning. Individual stage suites, each stage CLI on an 8-frame 80 × 80 Checker smoke case, both full capture runs and comparison plotting executed. Separate processes reloaded each saved Checker registration and regenerated RPCA and V2 fusion with maximum absolute pixel difference 0 for all four reconstructions. The original ten RPCA cases passed before and after refactoring. [Ten-case results](rpca_verification.csv) The corrected eight-frame regression reported EPE 0.440, 0.949 and 2.127 px for nominal 1, 2 and 4 px warps. Each amplitude in that legacy script draws a new random field. The smaller synthetic pytest cases also assert signed translation, nonrigid flow, FFT adjoints and partial Fourier recovery.

## Remaining research

Evaluate the frozen algorithm with the official RGB MFIR protocol on held-out backgrounds and unseen water. Measure multiple random seeds and scenes, identify track-support failure before fusion, and compare repeated timings on the target laptop. A native-resolution MATLAB parity study, new paired camera captures and OCR have not been run. The two real clips and one synthetic sweep cannot establish generalization or a physical distortion limit.
