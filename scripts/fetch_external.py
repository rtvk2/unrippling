"""Fetch third-party code and data into external/ and data/.

    python scripts/fetch_external.py repos
        Clone the reference repos at pinned commits into external/.

    python scripts/fetch_external.py mfir --types shallow --max-profiles 1
        Stream the Zenodo archive of the refractive MFIR benchmark [3] and extract
        only the backgrounds plus the requested wave profiles into data/mfir/.

Why streaming: the archive is one .tar.gz split into 4 parts (12.9+12.9+12.9+5.2 GB).
A gzip stream cannot be entered half-way, so part04 alone is useless, and the whole
thing does not fit on a typical laptop disk. Here the parts are read as one HTTP
stream, decompressed on the fly, and only the wanted members are written.
Normals are stored as float32 (the benchmark casts them to float32 anyway), which
halves the disk cost: one 200-frame profile is ~630 MB.

Every profile directory seen in the stream is logged with its compressed byte
offset in data/mfir/archive_index.tsv, so you learn how far into the archive a
given wave type sits before committing to a long download.
"""

import argparse
import http.client
import io
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
MFIR_DIR = ROOT / "data" / "mfir"

REPOS = {
    # James et al. ICCV 2019 [1]: dataset (incl. Tian & Narasimhan clips), MATLAB code + bundled YALL1, supplemental results
    "CompressiveFlows": ("https://github.com/jeringeo/CompressiveFlows.git", "78d9320d50f00803b085c2a095094fc1796db241"),
    # Shugaev et al. CVPRW 2026 [3]: benchmark generator + eval code
    "refractive-mfir-benchmark": ("https://github.com/iafoss/refractive-mfir-benchmark.git", "45eee65f291eb4cdd82987f51af3ad850f130d47"),
}

ZENODO = "https://zenodo.org/api/records/19390086/files/dataset_refractive_mfir_benchmark.tar.gz.part0{}/content"
PARTS = [ZENODO.format(i) for i in range(1, 5)]
PREFIX = "dataset_refractive_mfir_benchmark/"
WAVE_DIRS = {"ocean": "Ocean_waves", "shallow": "Shallow_waves", "sine": "Sine_waves", "ripple": "Ripples"}


ARCHIVE_GB = 12.9 * 3 + 5.2  # total compressed size of the four parts


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fetch_repos():
    EXTERNAL.mkdir(exist_ok=True)
    for name, (url, sha) in REPOS.items():
        dest = EXTERNAL / name
        if not dest.exists():
            subprocess.run(["git", "init", "-q", str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", sha], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        print(f"{name}: {head}" + ("" if head == sha else f"  (WARNING: pinned {sha})"))


class PartsStream(io.RawIOBase):
    """The four archive parts as one readable stream, resuming dropped connections with HTTP Range.

    http.client reports a connection the server closed early as an ordinary 0-byte read, so a part
    only counts as finished once all of its bytes (size taken from the response headers) have
    arrived. Otherwise the next part would be spliced into the middle of this one and corrupt the
    gzip stream.
    """

    def __init__(self, urls, retries=20, on_progress=None):
        self.urls, self.retries = urls, retries
        self.idx, self.offset_in_part, self.total = 0, 0, 0
        self.part_size, self.resp = None, None
        self.on_progress, self._last_tick = on_progress, 0.0

    def readable(self):
        return True

    def _open(self):
        req = urllib.request.Request(self.urls[self.idx])
        if self.offset_in_part:
            req.add_header("Range", f"bytes={self.offset_in_part}-")
        resp = urllib.request.urlopen(req, timeout=120)
        if self.offset_in_part:
            if resp.status != 206:
                resp.close()
                raise OSError(f"server ignored the Range request (HTTP {resp.status})")
            self.part_size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
        else:
            self.part_size = int(resp.headers["Content-Length"])
        self.resp = resp

    def readinto(self, b):
        for attempt in range(self.retries):
            if self.idx >= len(self.urls):
                return 0
            err = None
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.readinto(b)
            except (OSError, http.client.HTTPException) as e:
                n, err = 0, e
            if n:
                self.offset_in_part += n
                self.total += n
                now = time.time()
                if self.on_progress and now - self._last_tick >= 1.0:  # throttled: at most once a second
                    self._last_tick = now
                    self.on_progress()
                return n
            if self.resp is not None:
                self.resp.close()
                self.resp = None
            if err is None and self.offset_in_part == self.part_size:  # part complete, go to the next one
                self.idx, self.offset_in_part = self.idx + 1, 0
                continue
            print(f"\n  connection dropped at {self.total / 1e9:.2f} GB ({err or 'closed early'}); "
                  f"resuming, retry {attempt + 1}/{self.retries}", file=sys.stderr)
            time.sleep(min(60, 2 ** attempt))
        raise OSError("too many connection failures")


def fetch_mfir(types, max_profiles, budget_gb):
    wanted = {WAVE_DIRS[t] for t in types}
    kept = {w: [] for w in wanted}  # wave dir -> profile names being kept, in stream order
    closed = set()                  # (wave dir, profile) that the stream has moved past
    written, budget = 0, budget_gb * 1e9
    MFIR_DIR.mkdir(parents=True, exist_ok=True)
    index_path = MFIR_DIR / "archive_index.tsv"
    indexed = set(index_path.read_text().split("\n")[1:]) if index_path.exists() else set()
    index = open(index_path, "a")
    if not indexed:
        index.write("compressed_offset_GB\tprofile_dir\n")
    current_dir = None

    def done():
        return all(len(kept[w]) >= max_profiles and all((w, p) in closed for p in kept[w]) for w in wanted)

    t0, note = time.time(), ""

    def progress():
        el = time.time() - t0
        gb = raw.total / 1e9
        speed = raw.total / el / 1e6 if el > 0 else 0.0
        print(f"\r  {gb:6.2f}/{ARCHIVE_GB:.1f} GB streamed ({100 * gb / ARCHIVE_GB:4.1f}%) | {speed:5.1f} MB/s | "
              f"elapsed {fmt_time(el)} | written {written / 1e9:.2f} GB | {note}\033[K", end="", flush=True)

    raw = PartsStream(PARTS, on_progress=progress)
    with tarfile.open(fileobj=io.BufferedReader(raw, 1 << 20), mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.startswith(PREFIX):
                continue
            rel = m.name[len(PREFIX):]
            match = re.match(r"wave_profiles/([^/]+)/([^/]+)/(.+\.npy)$", rel)

            if match is None:  # backgrounds, README, pickles: keep as-is (never unpickled here)
                out = MFIR_DIR / rel
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tar.extractfile(m).read())
                    written += m.size
                continue

            wave, profile, fname = match.groups()
            if (wave, profile) != current_dir:
                if current_dir is not None:
                    closed.add(current_dir)
                current_dir = (wave, profile)
                line = f"{raw.total / 1e9:.2f}\t{wave}/{profile}"
                if not any(l.endswith(f"\t{wave}/{profile}") for l in indexed):
                    index.write(line + "\n")
                    index.flush()
                print(f"\n[{raw.total / 1e9:6.2f} GB streamed] reached {wave}/{profile}")
                if done():
                    break
            if wave not in wanted:
                continue
            if profile not in kept[wave]:
                if len(kept[wave]) >= max_profiles:
                    continue
                kept[wave].append(profile)
            out = MFIR_DIR / "wave_profiles" / wave / profile / fname
            if not out.exists():
                if written + m.size / 2 > budget:
                    print(f"\nDisk budget of {budget_gb} GB reached, stopping.")
                    break
                arr = np.load(io.BytesIO(tar.extractfile(m).read()), allow_pickle=False)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr.astype(np.float32))
                written += out.stat().st_size
            note = f"{wave}/{profile}/{fname}"
            progress()
    index.close()
    el = time.time() - t0
    print(f"\nDone in {fmt_time(el)}. Streamed {raw.total / 1e9:.2f} GB ({raw.total / el / 1e6:.1f} MB/s avg), "
          f"wrote {written / 1e9:.2f} GB to {MFIR_DIR}")
    print("Kept profiles:", {w: p for w, p in kept.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("repos", help="clone pinned reference repos into external/")
    m = sub.add_parser("mfir", help="stream a subset of the MFIR benchmark data into data/mfir/")
    m.add_argument("--types", nargs="+", default=["shallow"], choices=list(WAVE_DIRS),
                   help="wave types to keep (archive order starts with shallow)")
    m.add_argument("--max-profiles", type=int, default=1, help="profiles to keep per wave type")
    m.add_argument("--budget-gb", type=float, default=10.0, help="stop once this much has been written")
    args = ap.parse_args()
    if args.cmd == "repos":
        fetch_repos()
    else:
        fetch_mfir(args.types, args.max_profiles, args.budget_gb)
