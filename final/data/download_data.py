"""Download individual, pinned CompressiveFlows assets. No archive is required."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
PIN = "78d9320d50f00803b085c2a095094fc1796db241"
BASE = f"https://raw.githubusercontent.com/jeringeo/CompressiveFlows/{PIN}/Dataset/"
ASSETS = json.loads((ROOT / "data/manifest.json").read_text())
FILES = {f"{name}.{suffix}": pair for name, asset in ASSETS.items()
         for suffix,pair in asset["files"].items()}


def valid(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest() == digest


def record(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def download(names, budget):
    dest = ROOT / "data/real"
    dest.mkdir(parents=True, exist_ok=True)
    ledger = dest / "downloads.json"
    with FileLock(str(ledger) + ".lock"):
        log = json.loads(ledger.read_text()) if ledger.exists() else {"charged_bytes": 0, "files": {}}
        for name, (size, digest) in FILES.items():
            if Path(name).stem.lower() not in names:
                continue
            asset_dir = ROOT / "data/public_synthetic" if ASSETS[Path(name).stem]["group"] == "Synthetic" else dest
            asset_dir.mkdir(parents=True, exist_ok=True)
            path = asset_dir / name
            folder = "DistortedVideos" if name.endswith(".avi") else "GroundTruthImages"
            if valid(path, size, digest):
                print(f"Verified {name}")
                continue
            legacy = ROOT / "external/CompressiveFlows/Dataset" / folder / name
            part = path.with_suffix(path.suffix + ".part")
            if valid(legacy, size, digest):
                shutil.copyfile(legacy, part)
            else:
                url = BASE + folder + "/" + name
                # Reserve the entire request first. A crash can overcharge, never undercharge.
                if log["charged_bytes"] + size > budget:
                    raise SystemExit(f"Budget exhausted before {name} ({size} bytes)")
                log["charged_bytes"] += size
                record(ledger, log)
                received = 0
                try:
                    with urllib.request.urlopen(url, timeout=60) as source, part.open("wb") as sink:
                        length = source.headers.get("Content-Length")
                        if length and int(length) != size:
                            raise IOError(f"Unexpected size for {url}: {length}")
                        while received < size:
                            block = source.read(min(1 << 18, size - received))
                            if not block:
                                break
                            received += len(block)
                            sink.write(block)
                finally:
                    log["charged_bytes"] -= size - received
                    record(ledger, log)
            if not valid(part, size, digest):
                raise IOError(f"Incomplete or corrupt {name}. Rerun to retry the file.")
            part.replace(path)
            log["files"][name] = {"bytes": size, "git_blob": digest, "url": BASE + folder + "/" + name}
            record(ledger, log)
        print(f"Cumulative dataset transfer: {log['charged_bytes']} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["checker", "middle"], choices=sorted(n.lower() for n in ASSETS))
    parser.add_argument("--budget-bytes", type=int, default=500_000_000)
    args = parser.parse_args()
    download(args.datasets, min(args.budget_bytes, 500_000_000))
