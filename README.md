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
check : 
```bash
    python -m pip install -e

    python -c "import unripple; print(unripple.__file__)"
```

# 3. Run the baselines (mean and median) on everything available locally ##For baselines
```bash
python scripts/run_baselines.py --no-lpips   # fast check
python scripts/run_baselines.py              # full run with LPIPS
```
## For Registration
```bash
python scripts/run_registration.py --no-lpips
python scripts/run_registration.py --rounds 5 --average trimmed --datasets james2019
```
