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

    python scripts/fetch_external.py download --dest D:/mfir_archive
        Download the four archive parts (43.8 GB) to disk with a progress bar. Resumable:
        re-run the same command after an interruption and it continues where it stopped.
        Each part is checked against the MD5 published on Zenodo. Zenodo limits speed per
        connection, so each part is fetched in segments over --connections (default 16) parallel
        requests; with ~4.6 MB/s line speed that saturates at ~3.8 MB/s (about 3.5 h total).

    python scripts/fetch_external.py mfir --archive D:/mfir_archive --types ocean shallow sine ripple --max-profiles 1000 --budget-gb 200
        Same extraction as above, but read from the downloaded parts instead of the network.

Every profile directory seen in the stream is logged with its compressed byte
offset in data/mfir/archive_index.tsv, so you learn how far into the archive a
given wave type sits before committing to a long download.
"""

import argparse
import hashlib
import http.client
import io
import json
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

ZENODO_RECORD = "https://zenodo.org/api/records/19390086"
ARCHIVE_NAME = "dataset_refractive_mfir_benchmark.tar.gz.part0{}"
ZENODO = ZENODO_RECORD + "/files/" + ARCHIVE_NAME + "/content"
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


def zenodo_meta():
    """{file name: (size in bytes, md5)} for the archive parts, from the Zenodo API."""
    with urllib.request.urlopen(ZENODO_RECORD, timeout=60) as r:
        return {f["key"]: (f["size"], f["checksum"].split(":", 1)[1]) for f in json.load(r)["files"]}


def fetch_range(url, lo, hi, retries=50, log=print, stop=None):
    """Bytes [lo, hi] of url, retried until all of them have arrived (a short read counts as a drop).
    Returns None early once the threading.Event stop is set."""
    buf = bytearray()
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo + len(buf)}-{hi}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status != 206:
                    raise OSError(f"server ignored the Range request (HTTP {resp.status})")
                while block := resp.read(1 << 20):
                    buf += block
                    if stop is not None and stop.is_set():
                        return None
            if len(buf) == hi - lo + 1:
                return bytes(buf)
            err = "closed early"
        except (OSError, http.client.HTTPException) as e:
            err = e
        wait = 120 if "429" in str(err) else min(60, 2 ** attempt)  # 429 = Zenodo rate limit
        log(f"\n  range {lo}-{hi}: dropped ({err}); retry {attempt + 1}/{retries} in {wait}s")
        time.sleep(wait)
    raise OSError(f"{url}: too many connection failures")


class PartsStream(io.RawIOBase):
    """The four archive parts as one readable stream, fetched over parallel HTTP Range requests.

    Zenodo caps each connection at ~0.1-0.4 MB/s, so a single sequential stream is very slow. Here the
    parts are cut into chunks; `connections` worker threads download the next chunks ahead of the
    reader, and readinto() hands them out strictly in order, so the gzip decoder still sees one
    contiguous stream. At most 2 * connections chunks are held in memory (~128 MB with the defaults).
    """

    def __init__(self, connections=16, chunk_mb=4, on_progress=None):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        meta = zenodo_meta()
        size = [meta[ARCHIVE_NAME.format(i)][0] for i in range(1, 5)]
        c = chunk_mb << 20
        self.chunks = [(ZENODO.format(i + 1), lo, min(lo + c, size[i]) - 1) for i in range(4) for lo in range(0, size[i], c)]
        self.pool, self.stop = ThreadPoolExecutor(connections), Event()
        self.ahead = 2 * connections
        self.futures, self.next_submit = {}, 0
        self.buf, self.pos, self.idx, self.total = b"", 0, 0, 0
        self.on_progress, self._last_tick = on_progress, 0.0

    def readable(self):
        return True

    def close(self):  # stop the prefetch threads, otherwise the interpreter waits for them on exit
        self.stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().close()

    def readinto(self, b):
        if self.pos == len(self.buf):
            if self.idx == len(self.chunks):
                return 0
            while self.next_submit < min(self.idx + self.ahead, len(self.chunks)):
                url, lo, hi = self.chunks[self.next_submit]
                self.futures[self.next_submit] = self.pool.submit(fetch_range, url, lo, hi, stop=self.stop)
                self.next_submit += 1
            self.buf, self.pos = self.futures.pop(self.idx).result(), 0
            self.idx += 1
        n = min(len(b), len(self.buf) - self.pos)
        b[:n] = self.buf[self.pos:self.pos + n]
        self.pos += n
        self.total += n
        now = time.time()
        if self.on_progress and now - self._last_tick >= 1.0:  # throttled: at most once a second
            self._last_tick = now
            self.on_progress()
        return n


class LocalPartsStream(io.RawIOBase):
    """The four downloaded archive parts read back to back as one stream (same interface as PartsStream)."""

    def __init__(self, files, on_progress=None):
        self.files, self.idx, self.fh, self.total = files, 0, None, 0
        self.on_progress, self._last_tick = on_progress, 0.0

    def readable(self):
        return True

    def readinto(self, b):
        while self.idx < len(self.files):
            if self.fh is None:
                self.fh = open(self.files[self.idx], "rb")
            n = self.fh.readinto(b)
            if n:
                self.total += n
                now = time.time()
                if self.on_progress and now - self._last_tick >= 1.0:
                    self._last_tick = now
                    self.on_progress()
                return n
            self.fh.close()
            self.fh, self.idx = None, self.idx + 1
        return 0


def _fetch_segment(url, out, lo, hi, bar, lock, retries=50, chunk=1 << 20):
    """Download bytes [lo, hi] of url into the file out, resuming from whatever out already holds."""
    want = hi - lo + 1
    for attempt in range(retries):
        pos = out.stat().st_size if out.exists() else 0
        if pos >= want:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo + pos}-{hi}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(out, "ab") as f:
                if resp.status != 206:
                    raise OSError(f"server ignored the Range request (HTTP {resp.status})")
                while block := resp.read(chunk):
                    f.write(block)
                    with lock:
                        bar.update(len(block))
        except (OSError, http.client.HTTPException) as e:
            wait = 120 if "429" in str(e) else min(60, 2 ** attempt)  # 429 = Zenodo rate limit
            with lock:
                bar.write(f"  {out.name}: dropped ({e}); retry {attempt + 1}/{retries} in {wait}s")
            time.sleep(wait)
    raise OSError(f"{out.name}: too many connection failures")


def download_archive(dest, parts=(1, 2, 3, 4), connections=16, segment_mb=256):
    """Download the archive parts (1-4) into dest with a tqdm progress bar.

    Zenodo caps the speed of each connection (~0.3-0.5 MB/s), so every part is cut into segments of
    segment_mb that are fetched over `connections` parallel HTTP Range requests into dest/.segments/.
    Segment files are resumed by their size, so re-running after an interruption continues where it
    stopped. A finished part is assembled from its segments while its MD5 is checked against Zenodo.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    from tqdm import tqdm

    dest = Path(dest)
    seg_dir = dest / ".segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    meta = zenodo_meta()

    seg = segment_mb << 20
    todo = []  # (part number, name, size, md5, [(segment file, lo, hi)])
    total = have = 0
    for i in parts:
        name = ARCHIVE_NAME.format(i)
        size, md5 = meta[name]
        total += size
        if (dest / (name + ".md5ok")).exists() and (dest / name).exists() and (dest / name).stat().st_size == size:
            have += size
            continue
        segs = [(seg_dir / f"{name}.{k:04d}", lo, min(lo + seg, size) - 1) for k, lo in enumerate(range(0, size, seg))]
        have += sum(min(f.stat().st_size, hi - lo + 1) for f, lo, hi in segs if f.exists())
        todo.append((i, name, size, md5, segs))
    print(f"Parts {list(parts)}: {total / 1e9:.1f} GB -> {dest}  (already on disk: {have / 1e9:.1f} GB), "
          f"{connections} connections")

    bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024, smoothing=0.05,
               dynamic_ncols=True, desc="download")
    lock = Lock()
    with ThreadPoolExecutor(connections) as pool:
        futures = {i: [pool.submit(_fetch_segment, ZENODO.format(i), f, lo, hi, bar, lock) for f, lo, hi in segs]
                   for i, _, _, _, segs in todo}
        for i, name, size, md5, segs in todo:  # assemble each part as soon as all of its segments are in
            for fut in futures[i]:
                fut.result()
            with lock:
                bar.write(f"  {name}: all segments in, assembling + checking MD5")
            h, out = hashlib.md5(), dest / name
            with open(out, "wb") as fo:
                for f, _, _ in segs:
                    with open(f, "rb") as fi:
                        for block in iter(lambda: fi.read(1 << 24), b""):
                            h.update(block)
                            fo.write(block)
            if out.stat().st_size != size or h.hexdigest() != md5:
                out.unlink()
                raise OSError(f"{name}: MD5/size mismatch; delete {seg_dir}/{name}.* and run again")
            (dest / (name + ".md5ok")).write_text(md5)
            for f, _, _ in segs:
                f.unlink()
            with lock:
                bar.write(f"  {name}: done, MD5 ok")
    bar.close()
    print(f"Parts {list(parts)} downloaded and verified in {dest}")


def fetch_mfir(types, max_profiles, budget_gb, archive=None, connections=16):
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

    if archive:
        files = [Path(archive) / ARCHIVE_NAME.format(i) for i in range(1, 5)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            sys.exit(f"missing archive parts in {archive}: {missing}; run the download command first")
        raw = LocalPartsStream(files, on_progress=progress)
    else:
        raw = PartsStream(connections, on_progress=progress)
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
    raw.close()
    index.close()
    el = time.time() - t0
    print(f"\nDone in {fmt_time(el)}. Streamed {raw.total / 1e9:.2f} GB ({raw.total / el / 1e6:.1f} MB/s avg), "
          f"wrote {written / 1e9:.2f} GB to {MFIR_DIR}")
    print("Kept profiles:", {w: p for w, p in kept.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("repos", help="clone pinned reference repos into external/")
    d = sub.add_parser("download", help="download the full MFIR archive (43.8 GB, resumable) to disk")
    d.add_argument("--dest", default=str(ROOT / "data" / "mfir_archive"), help="folder for the four .tar.gz parts")
    d.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4],
                   help="which parts to fetch (default: all)")
    d.add_argument("--connections", type=int, default=16, help="parallel HTTP connections (more is faster up to your line speed)")
    m = sub.add_parser("mfir", help="stream a subset of the MFIR benchmark data into data/mfir/")
    m.add_argument("--types", nargs="+", default=["shallow"], choices=list(WAVE_DIRS),
                   help="wave types to keep (archive order starts with shallow)")
    m.add_argument("--max-profiles", type=int, default=1, help="profiles to keep per wave type")
    m.add_argument("--budget-gb", type=float, default=10.0, help="stop once this much has been written")
    m.add_argument("--connections", type=int, default=16, help="parallel HTTP connections when streaming")
    m.add_argument("--archive", help="folder with the downloaded parts (from the download command); default: stream from Zenodo")
    args = ap.parse_args()
    if args.cmd == "repos":
        fetch_repos()
    elif args.cmd == "download":
        download_archive(args.dest, args.parts, args.connections)
    else:
        fetch_mfir(args.types, args.max_profiles, args.budget_gb, args.archive, args.connections)
