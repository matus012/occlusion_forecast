"""Phase 3 prep (D-N1-2): extract P2 CARLA occlusion-segment statistics that
parameterize the N1 mask generator.

Reads the sibling P2 project's CARLA scenario ground truth (read-only,
never modified) and writes results/p2_occlusion_stats.json -- an aggregate
stats-only artifact (no raw trajectories/frames, per the license-guard
invariant: only loaders/manifests/metric JSONs may be committed).

Usage: .venv/Scripts/python.exe scripts/extract_p2_stats.py [--scenarios-root PATH]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otraj.masking.p2_stats import build_p2_stats, dump_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("extract_p2_stats")

DEFAULT_SCENARIOS_ROOT = (
    ROOT.parent / "100_occlusion_mot" / "data" / "sim" / "scenarios"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    ap.add_argument(
        "--dest", type=Path, default=ROOT / "results" / "p2_occlusion_stats.json"
    )
    args = ap.parse_args()

    result = build_p2_stats(args.scenarios_root)
    # portable descriptor only — never an absolute local path (public history)
    descriptor = "ws/100_occlusion_mot/data/sim/scenarios (P1/P2 sibling repo, local; CARLA GT)"
    dump_json(result, args.dest, descriptor, extracted_date="2026-07-26")

    log.info("n_scenarios=%d n_agents_analyzed=%d n_segments=%d fps=%d",
              result.n_scenarios, result.n_agents_analyzed, len(result.runs), result.fps)
    log.info("duration_fit: %s", result.duration_fit.to_dict())
    log.info("pattern_mix counts=%s fractions=%s",
              result.pattern_counts, result.pattern_fractions)
    log.info("-> %s", args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
