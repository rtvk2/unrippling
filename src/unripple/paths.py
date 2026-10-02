from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# James et al. [1]: 14 distorted videos + ground truth (Middle/Small/Tiny are Tian & Narasimhan's [5] "Real2" clips)
JAMES = EXTERNAL / "CompressiveFlows" / "Dataset"
JAMES_VIDEOS = JAMES / "DistortedVideos"
JAMES_GT = JAMES / "GroundTruthImages"
JAMES_SUPPLEMENTAL = EXTERNAL / "CompressiveFlows" / "Supplemental_Material_ICCV2019"

# Shugaev et al. [3]: backgrounds + wave profiles streamed by scripts/fetch_external.py mfir
MFIR = DATA / "mfir"

# Our own tray recordings
OWN = DATA / "own"
