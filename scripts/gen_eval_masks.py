"""Phase 3 (D-N1-2 / D-N1-9 / D-N1-10): generate deterministic eval mask
manifests for a data split.

Reads scenario ids from a directory listing under data/<split> (each
subdirectory name is one AV2 scenario id -- no scenario content is read) and
writes, per (severity, regime) bucket:

  - results/mask_manifests/eval_masks_{split}_{severity}_{regime}.json
    COMMITTABLE summary (spec_hash, aggregates, pattern counts, a path +
    sha256 pointer to the local labels file below). Kept small (< 1MB,
    tested) per the license-guard 2MB-staged ceiling -- see adversarial
    review B2.
  - results/mask_manifests/local/eval_masks_{split}_{severity}_{regime}_labels.json
    LOCAL ONLY (gitignored): {scenario_id: pattern}, regenerable any time via
    otraj.masking.manifest.build_labels(), never committed.

Masks themselves are never written anywhere -- only pattern labels +
aggregate stats (license-guard invariant: no per-scenario dataset-derived
tensors committed). JSON is written compactly (no indentation).

Usage:
  .venv/Scripts/python.exe scripts/gen_eval_masks.py --split val \
      --severities S1,S2,S3,S4 --regimes R-A,R-B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otraj.masking.manifest import build_labels, build_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("gen_eval_masks")


def _list_scenario_ids(data_root: Path, split: str) -> list[str]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    ids = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not ids:
        raise ValueError(f"no scenario directories found under {split_dir}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--severities", default="S1,S2,S3,S4")
    ap.add_argument("--regimes", default="R-A,R-B")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "av2")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "mask_manifests")
    args = ap.parse_args()

    severities = [s.strip() for s in args.severities.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    scenario_ids = _list_scenario_ids(args.data_root, args.split)
    log.info("split=%s n_scenarios=%d", args.split, len(scenario_ids))

    local_dir = args.out_dir / "local"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    for regime in regimes:
        for severity in severities:
            stem = f"eval_masks_{args.split}_{severity}_{regime}"

            labels = build_labels(scenario_ids, severity, regime)
            labels_encoded = json.dumps(labels, separators=(",", ":"))
            labels_path = local_dir / f"{stem}_labels.json"
            labels_path.write_text(labels_encoded, encoding="utf-8")
            labels_sha256 = hashlib.sha256(labels_encoded.encode("utf-8")).hexdigest()

            relative_labels_path = str(labels_path.relative_to(ROOT)).replace("\\", "/")
            summary = build_summary(
                labels, severity, regime,
                labels_path=relative_labels_path, labels_sha256=labels_sha256,
            )
            summary_path = args.out_dir / f"{stem}.json"
            summary_encoded = json.dumps(summary, separators=(",", ":"))
            summary_path.write_text(summary_encoded, encoding="utf-8")

            summary_size_kb = summary_path.stat().st_size / 1e3
            labels_size_mb = labels_path.stat().st_size / 1e6
            agg = summary["aggregate"]
            log.info(
                "%s: n=%d const_frac=%.4f const_n_masked=%d patterns=%s "
                "summary=%.2fKB labels(local)=%.2fMB -> %s",
                stem, summary["n_scenarios"], agg["constant_achieved_fraction"],
                agg["constant_n_masked"], agg["pattern_counts"],
                summary_size_kb, labels_size_mb, summary_path,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
