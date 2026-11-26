# Un-Rippling

Recovering a sharp, undistorted image from a video shot through wavy water, without deep learning.

# To test 
NOTE : Currently the code is testing on temporary 3 images available in external.

# 1. Install (CPU torch first, then the rest, then the package itself)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```
# 2. Clone the reference repos (James et al. clips, MFIR code)
```bash
python scripts/fetch_external.py repos
```
## 2b. (Optional) Download the full MFIR benchmark archive (43.8 GB, resumable, MD5-checked)
Uses 16 parallel connections (Zenodo caps each one at ~0.3 MB/s); re-run the same command to resume.
Keep the archive outside OneDrive.
```bash
python scripts/fetch_external.py download --dest C:/mfir_archive
```
Then extract all wave profiles into data/mfir/ (normals saved as float32):
```bash
python scripts/fetch_external.py mfir --archive C:/mfir_archive --types ocean shallow sine ripple --max-profiles 1000 --budget-gb 200
```

check : 
```bash
    python -m pip install -e

    python -c "import unripple; print(unripple.__file__)"
```

# 3. Run the baselines (mean and median) on everything available locally  ## For baselines
```bash
python scripts/run_baselines.py --no-lpips   # fast check
python scripts/run_baselines.py              # full run with LPIPS
```
## For Registration
```bash
python scripts/test_registration_synthetic.py   # check on a known synthetic warp
python scripts/run_registration.py --no-lpips
python scripts/run_registration.py --rounds 5 --average trimmed --datasets james2019
python scripts/score_saved.py registration      # LPIPS of the saved images -> results/registration/lpips_check.csv
```
## For Registration tuning (dev split only)
```bash
python scripts/tune_registration.py                                       # 50-frame grid -> results/tuning/dev_*.csv
python scripts/tune_registration.py --frames 199 --flow-sigma 8 12 16 --average mean --tag full   # full-length check
python scripts/tune_registration.py --split test --frames 199 --flow-sigma 8 --average mean       # frozen setting, test split, run once
```

## For Robust PCA
```bash
python scripts/test_rpca_synthetic.py           # solver check on known low-rank + sparse data
python scripts/run_rpca.py --no-lpips           # register, then RPCA -> results/rpca/
python scripts/score_saved.py rpca              # LPIPS of the saved images -> results/rpca/lpips_check.csv
```
