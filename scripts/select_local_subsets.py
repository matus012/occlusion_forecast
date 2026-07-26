"""Select the *-local subsets for the D-N1-14 reduced-scale package.

Two deterministic, manifest-pinned orderings:

  * TRAIN POOL  -- every scenario in data/av2/train, ordered so that ANY prefix
    is city-stratified (each city appears in proportion to its share of the
    full split). The actual trained-on subset is a prefix of this pool whose
    length is frozen AFTER throughput measurement (D-N1-14a) -- selection and
    sizing are decoupled on purpose.
  * VAL SUBSET  -- every scenario in data/av2/val, ordered so that any prefix
    preserves the R-A pattern-cohort proportions (M1/M2/M3 as drawn by the
    N1-mask-v2 severity-independent pattern stream). The fixed eval subset is
    a prefix; per-pattern slicing (D-N1-10) then has cohort counts as close to
    proportional as integer rounding allows.

Ordering technique (both lists): within each stratum, ids are ranked by
sha256("{SPEC}|{split}|{scenario_id}") -- deterministic, platform-independent,
no RNG state. Each id gets score = (rank + 1) / stratum_size, and the global
order is ascending (score, hash) -- a largest-remainder-style interleave where
every prefix of length n holds ~n * stratum_share of each stratum.

Outputs:
  results/local/train_pool.json   (ordered [scenario_id, city]        -- gitignored)
  results/local/val_order.json    (ordered [scenario_id, pattern]     -- gitignored)
  results/local_subsets_manifest.json (committed: spec, sizes, stratum
      distributions, sha256 of both local files)

The full id lists stay local (regenerable by rerunning this script against the
same AV2 split listing); only the summary manifest is committed -- same
precedent as results/av2_manifests and results/mask_manifests/local.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from otraj.masking.generator import (  # noqa: E402
    MIX_RA,
    SPEC_VERSION,
    _draw_pattern,
    pattern_seed,
)

LOCAL_SPEC = "N1-local-v1"


def _hash_rank(split: str, scenario_id: str) -> int:
    digest = hashlib.sha256(f"{LOCAL_SPEC}|{split}|{scenario_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _read_city(scenario_dir: Path) -> tuple[str, str]:
    sid = scenario_dir.name
    pf = pq.ParquetFile(scenario_dir / f"scenario_{sid}.parquet")
    # city is constant per scenario; row-group column statistics give it
    # without reading any data pages.
    stats = pf.metadata.row_group(0).column(15).statistics
    city = stats.min if stats is not None else None
    if not isinstance(city, str) or not city:
        # fallback: read one value from the city column
        city = pf.read_row_group(0, columns=["city"]).column(0)[0].as_py()
    return sid, city


def _interleaved(ids_by_stratum: dict[str, list[str]], split: str) -> list[tuple[str, str]]:
    """Ascending (rank_in_stratum / stratum_size, hash) => stratified prefixes."""
    rows: list[tuple[float, int, str, str]] = []
    for stratum, ids in ids_by_stratum.items():
        ids_sorted = sorted(ids, key=lambda s: _hash_rank(split, s))
        n = len(ids_sorted)
        for rank, sid in enumerate(ids_sorted):
            rows.append(((rank + 1) / n, _hash_rank(split, sid), sid, stratum))
    rows.sort()
    return [(sid, stratum) for _, _, sid, stratum in rows]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "av2")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "local")
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "results" / "local_subsets_manifest.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- train pool (city-stratified) ----------------
    t0 = time.time()
    train_dirs = sorted(p for p in (args.data_root / "train").iterdir() if p.is_dir())
    print(f"[train] {len(train_dirs)} scenarios; reading cities...")
    cities: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for sid, city in ex.map(_read_city, train_dirs, chunksize=256):
            cities[sid] = city
    city_counts = Counter(cities.values())
    print(f"[train] cities: {dict(city_counts)} ({time.time() - t0:.1f}s)")

    by_city: dict[str, list[str]] = {}
    for sid, city in cities.items():
        by_city.setdefault(city, []).append(sid)
    train_pool = _interleaved(by_city, "train")

    train_pool_path = args.out_dir / "train_pool.json"
    train_pool_path.write_text(json.dumps(
        {"spec": LOCAL_SPEC, "split": "train", "order": train_pool}), encoding="utf-8")

    # ---------------- val order (pattern-cohort-stratified) ----------------
    t0 = time.time()
    val_ids = sorted(p.name for p in (args.data_root / "val").iterdir() if p.is_dir())
    print(f"[val] {len(val_ids)} scenarios; drawing R-A pattern cohorts...")
    patterns: dict[str, str] = {
        sid: _draw_pattern(np.random.default_rng(pattern_seed(sid, "R-A")), MIX_RA)
        for sid in val_ids
    }
    cohort_counts = Counter(patterns.values())
    print(f"[val] cohorts: {dict(cohort_counts)} ({time.time() - t0:.1f}s)")

    by_pattern: dict[str, list[str]] = {}
    for sid, pat in patterns.items():
        by_pattern.setdefault(pat, []).append(sid)
    val_order = _interleaved(by_pattern, "val")

    val_order_path = args.out_dir / "val_order.json"
    val_order_path.write_text(json.dumps(
        {"spec": LOCAL_SPEC, "split": "val", "mask_spec": SPEC_VERSION,
         "regime": "R-A", "order": val_order}), encoding="utf-8")

    manifest = {
        "spec": LOCAL_SPEC,
        "mask_spec": SPEC_VERSION,
        "created": "2026-07-27",
        "ordering": "ascending ((rank+1)/stratum_size, sha256-hash) within-stratum "
                    "interleave; any prefix is stratum-proportional",
        "hash_string": "sha256('{spec}|{split}|{scenario_id}')[:8] big-endian",
        "train": {
            "n_total": len(train_pool),
            "stratum": "city (parquet row-group statistics)",
            "city_counts": dict(sorted(city_counts.items())),
            "file": "results/local/train_pool.json",
            "sha256": _sha256_file(train_pool_path),
        },
        "val": {
            "n_total": len(val_order),
            "stratum": "N1-mask-v2 R-A pattern cohort (severity-independent draw)",
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "file": "results/local/val_order.json",
            "sha256": _sha256_file(val_order_path),
        },
        "note": "Trained-on subset and fixed eval subset are PREFIXES of these "
                "orderings; prefix lengths are frozen by measurement (D-N1-14) "
                "and recorded in results/local/budget.json.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
