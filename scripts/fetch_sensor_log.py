"""Selective anonymous fetch from the public AV2 sensor bucket.

The full AV2 Sensor dataset is ~1.06 TB (1000 logs x ~1.06 GB) -- far past any
sane local budget. It does not have to be fetched in bulk: the bucket is a plain
public S3 prefix, listable and gettable per object over anonymous HTTPS, so the
demo pulls only what it uses.

  metadata slice  (annotations + ego pose + calibration + HD map)  ~2.3 MB/log
  one ring camera (319 JPEGs @ 20 Hz)                             ~96-129 MB/log
  lidar                                                            NEVER (unused)

Scene selection (scripts/pick_rgb_demo_scene.py) runs entirely on the metadata
slice, so pixels are only ever fetched for a (log, camera) pair already known to
be worth rendering.

Everything lands under data/ -- gitignored, never tracked (tests/test_license_guard.py).

Usage:
  # scout pool: metadata only for the first 40 val logs  (~92 MB)
  python scripts/fetch_sensor_log.py val --n-logs 40
  # pixels for one chosen (log, camera)
  python scripts/fetch_sensor_log.py val --log-id 0b86f508-... --cam ring_front_left --images
"""
from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BASE = "https://s3.amazonaws.com/argoverse"
SENSOR = "datasets/av2/sensor"


def list_prefix(prefix: str, delimiter: str = "") -> tuple[list[tuple[str, int]], list[str]]:
    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BASE}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            keys.append((c.findtext(f"{NS}Key"), int(c.findtext(f"{NS}Size"))))
        for c in root.findall(f"{NS}CommonPrefixes"):
            prefixes.append(c.findtext(f"{NS}Prefix"))
        if root.findtext(f"{NS}IsTruncated") != "true":
            break
        token = root.findtext(f"{NS}NextContinuationToken")
    return keys, prefixes


def get(key: str, dest: Path) -> int:
    """Idempotent: an existing file is left alone, so an interrupted fetch
    resumes at object granularity on re-run."""
    if dest.exists():
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BASE}/{urllib.parse.quote(key)}", timeout=300) as r, \
            tmp.open("wb") as f:
        n = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            n += len(chunk)
    tmp.replace(dest)
    return n


def log_ids(split: str, n: int) -> list[str]:
    _, prefixes = list_prefix(f"{SENSOR}/{split}/", delimiter="/")
    return [p.rstrip("/").rsplit("/", 1)[-1] for p in sorted(prefixes)][:n]


def fetch_log(split: str, log_id: str, out_root: Path, cam: str | None,
              images: bool, workers: int) -> None:
    prefix = f"{SENSOR}/{split}/{log_id}/"
    out_log = out_root / split / log_id

    meta = [k for k, _ in list_prefix(prefix)[0]
            if "/sensors/" not in k and k != prefix]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        n = sum(ex.map(lambda k: get(k, out_log / k[len(prefix):]), meta))
    print(f"[{log_id[:8]}] meta {len(meta)} files  {n / 1e6:6.2f} MB  {time.time() - t0:5.1f}s")

    if images:
        if cam is None:
            raise SystemExit("--images requires --cam")
        keys = [k for k, _ in list_prefix(f"{prefix}sensors/cameras/{cam}/")[0]]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            n = sum(ex.map(lambda k: get(k, out_log / k[len(prefix):]), keys))
        dt = max(time.time() - t0, 1e-9)
        print(f"[{log_id[:8]}] {cam} {len(keys)} imgs  {n / 1e6:6.2f} MB  "
              f"{dt:5.1f}s  ({n / 1e6 / dt:.2f} MB/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--log-id", default=None, help="one log; default = first --n-logs")
    ap.add_argument("--n-logs", type=int, default=10)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--images", action="store_true",
                    help="also fetch the --cam JPEGs (~96-129 MB/log)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "av2_sensor")
    a = ap.parse_args()

    ids = [a.log_id] if a.log_id else log_ids(a.split, a.n_logs)
    print(f"{len(ids)} log(s) -> {a.out_root / a.split}"
          f"{'  (metadata only)' if not a.images else ''}")
    for lid in ids:
        fetch_log(a.split, lid, a.out_root, a.cam, a.images, a.workers)


if __name__ == "__main__":
    main()
