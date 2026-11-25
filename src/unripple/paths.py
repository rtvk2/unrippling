import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
# Large intermediate arrays (registered stacks, ~150 MB per clip): kept OUTSIDE the OneDrive project folder so they
# are not uploaded. Override with the UNRIPPLE_CACHE environment variable.
CACHE = Path(os.environ.get("UNRIPPLE_CACHE", "C:/unripple_cache"))

# James et al. [1]: 14 distorted videos + ground truth (Middle/Small/Tiny are Tian & Narasimhan's [5] "Real2" clips)
JAMES = EXTERNAL / "CompressiveFlows" / "Dataset"
JAMES_VIDEOS = JAMES / "DistortedVideos"
JAMES_GT = JAMES / "GroundTruthImages"
JAMES_SUPPLEMENTAL = EXTERNAL / "CompressiveFlows" / "Supplemental_Material_ICCV2019"

# Shugaev et al. [3]: backgrounds + wave profiles streamed by scripts/fetch_external.py mfir
MFIR = DATA / "mfir"
# Stand-in until MFIR is downloaded: the 3 Ocean_extreme demo GIFs shipped in the benchmark repo (left: distorted, right: ground truth)
MFIR_DEMO = EXTERNAL / "refractive-mfir-benchmark" / "assets"

# Our own tray recordings
OWN = DATA / "own"
