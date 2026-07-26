"""AV2 motion forecasting split verification: structure + SHA256 manifest.

Emits results/av2_verification.json (committable summary: counts, structural
check, aggregate SHA) and results/av2_manifests/<split>_manifest.json.gz
(full per-file SHA256 manifest — large, kept local; its own SHA recorded in
the summary so integrity is still committable).

Expected AV2 MF structure per scenario dir:
  <scenario_id>/scenario_<scenario_id>.parquet
  <scenario_id>/log_map_archive_<scenario_id>.json

Usage: python scripts/verify_av2.py --split val [--data-root data/av2]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COUNTS = {"train": 199908, "val": 24988, "test": 24984}
SOURCE = "s3://argoverse/datasets/av2/motion-forecasting (public bucket, aws cli --no-sign-request)"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "av2")
    args = ap.parse_args()

    split_dir = args.data_root / args.split
    t0 = time.time()
    scenario_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())

    structural_errors: list[str] = []
    manifest: dict[str, dict[str, str | int]] = {}
    total_bytes = 0
    for d in scenario_dirs:
        sid = d.name
        expected = {f"scenario_{sid}.parquet", f"log_map_archive_{sid}.json"}
        actual = {f.name for f in d.iterdir() if f.is_file()}
        if actual != expected:
            structural_errors.append(f"{sid}: has {sorted(actual)}")
            if len(structural_errors) > 20:
                break
            continue
        for f in sorted(d.iterdir()):
            st = f.stat()
            total_bytes += st.st_size
            manifest[f"{sid}/{f.name}"] = {"sha256": sha256_file(f), "bytes": st.st_size}

    agg = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in sorted(manifest.items())).encode()
    ).hexdigest()

    man_dir = ROOT / "results" / "av2_manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / f"{args.split}_manifest.json.gz"
    with gzip.open(man_path, "wt", encoding="utf-8") as f:
        json.dump(manifest, f)

    n = len(scenario_dirs)
    summary = {
        "split": args.split,
        "source": SOURCE,
        "n_scenario_dirs": n,
        "official_count": OFFICIAL_COUNTS[args.split],
        "count_matches_official": n == OFFICIAL_COUNTS[args.split],
        "structural_errors": structural_errors,
        "structure_ok": not structural_errors,
        "n_files_hashed": len(manifest),
        "total_bytes": total_bytes,
        "aggregate_sha256": agg,
        "per_file_manifest": f"{man_path.relative_to(ROOT).as_posix()} (local-only, large)",
        "per_file_manifest_sha256": sha256_file(man_path),
        "wall_seconds": round(time.time() - t0, 1),
        "date": time.strftime("%Y-%m-%d"),
    }

    out = ROOT / "results" / "av2_verification.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing[args.split] = summary
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["structure_ok"] and summary["count_matches_official"] else 1


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
