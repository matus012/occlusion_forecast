"""Phase 3 eval mask manifests (D-N1-10, restructured per adversarial review
B2/M7).

A manifest bucket (one severity x regime combination) is split into two
artifacts, mirroring the existing results/av2_manifests precedent of keeping
committed metadata small and separate from bulk per-scenario data:

  (a) SUMMARY (committable, small): spec_version, spec_hash, severity,
      regime, n_scenarios, mix_used, rb_mix_note, seed-derivation
      description, aggregate stats (pattern counts + the CONSTANT achieved
      fraction/n_masked for this bucket -- see below), and a path + sha256
      pointer to (b). This is the file that goes under results/mask_manifests/.

  (b) PER-SCENARIO LABELS (local only, gitignored): {scenario_id: pattern}.
      NOT achieved_fraction/n_masked per scenario -- those are CONSTANT
      across every scenario in a given (severity, regime) bucket by
      construction (the fraction guarantee hits the target EXACTLY, see
      generator.py's module docstring), so storing them per-scenario would
      be redundant bulk data with zero information content. This file lives
      under results/mask_manifests/local/ (gitignored) and is regenerated
      by scripts/gen_eval_masks.py, never committed.

Masks themselves are NEVER stored in either artifact -- they are cheaply
regenerable from (scenario_id, severity, regime, spec_version) via
generator.generate_mask(), per the license-guard invariant (no per-scenario
dataset-derived tensors committed).

spec_hash covers every generator constant that affects the mask distribution
-- so any constant change (mix, clips, fit params, bucket targets, tolerance,
forced-tail length) changes the hash and invalidates stale manifests.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from otraj.masking.generator import (
    FIT_LOC,
    FIT_SCALE,
    FIT_SHAPE,
    FORCED_VISIBLE_TAIL,
    HZ,
    M1_CLIP_S,
    M3_CLIP_S,
    M3_MAX_BLOCKS,
    M3_MIN_BLOCKS,
    MIX_RA,
    MIX_RB,
    N_STEPS,
    RB_MIX_NOTE,
    SEED_DERIVATION_NOTE,
    SEVERITY_TARGETS,
    SEVERITY_TOLERANCE,
    SPEC_VERSION,
    generate_mask,
    mix_for_regime,
    target_steps_for_severity,
)


def spec_constants(spec_version: str = SPEC_VERSION) -> dict:
    """Every generator constant that affects the mask distribution. Changing
    ANY of these must change compute_spec_hash()'s output."""
    return {
        "spec_version": spec_version,
        "n_steps": N_STEPS,
        "hz": HZ,
        "forced_visible_tail": FORCED_VISIBLE_TAIL,
        "severity_targets": dict(SEVERITY_TARGETS),
        "severity_tolerance": SEVERITY_TOLERANCE,
        "mix_ra": dict(MIX_RA),
        "mix_rb": dict(MIX_RB),
        "m1_clip_s": list(M1_CLIP_S),
        "m3_clip_s": list(M3_CLIP_S),
        "m3_min_blocks": M3_MIN_BLOCKS,
        "m3_max_blocks": M3_MAX_BLOCKS,
        "duration_fit": {"shape": FIT_SHAPE, "loc": FIT_LOC, "scale": FIT_SCALE},
    }


def compute_spec_hash(spec_version: str = SPEC_VERSION) -> str:
    """sha256 over a canonical (sorted-key, compact) JSON encoding of ALL
    generator constants -- so any constant change changes the hash."""
    canonical = json.dumps(spec_constants(spec_version), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_labels(
    scenario_ids: Sequence[str], severity: str, regime: str,
    spec_version: str = SPEC_VERSION,
) -> dict[str, str]:
    """The local (gitignored) per-scenario {scenario_id: pattern} artifact.
    achieved_fraction/n_masked are NOT included here -- they are constant per
    bucket, reported once in the summary's aggregate stats instead."""
    return {
        sid: generate_mask(sid, severity, regime, spec_version=spec_version).pattern
        for sid in scenario_ids
    }


def build_summary(
    labels: dict[str, str], severity: str, regime: str,
    spec_version: str = SPEC_VERSION,
    labels_path: str | None = None,
    labels_sha256: str | None = None,
) -> dict:
    """The committable summary artifact for one (severity, regime) bucket.
    `labels` is the dict produced by build_labels() for the SAME bucket --
    passed in rather than recomputed so the caller controls exactly one
    generation pass over scenario_ids (labels_path/labels_sha256 describe the
    local file the caller writes `labels` to, if any)."""
    pattern_counts: dict[str, int] = {}
    for pattern in labels.values():
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    target = SEVERITY_TARGETS[severity]
    n_masked_constant = target_steps_for_severity(severity)
    constant_achieved_fraction = n_masked_constant / N_STEPS

    return {
        "spec_version": spec_version,
        "spec_hash": compute_spec_hash(spec_version),
        "severity": severity,
        "regime": regime,
        "n_scenarios": len(labels),
        "mix_used": mix_for_regime(regime),
        "rb_mix_note": RB_MIX_NOTE if regime == "R-B" else "",
        "seed_derivation": SEED_DERIVATION_NOTE,
        "per_scenario_labels_path": labels_path,
        "per_scenario_labels_sha256": labels_sha256,
        "aggregate": {
            "target_fraction": target,
            "tolerance": SEVERITY_TOLERANCE,
            # Constant across every scenario in this bucket by construction
            # (the fraction guarantee hits target_fraction EXACTLY) -- NOT a
            # per-scenario measurement, reported once here instead of
            # duplicated n_scenarios times in the labels file.
            "constant_achieved_fraction": constant_achieved_fraction,
            "constant_n_masked": n_masked_constant,
            "pattern_counts": pattern_counts,
        },
    }


def build_manifest(
    scenario_ids: Sequence[str], severity: str, regime: str,
    spec_version: str = SPEC_VERSION,
    labels_path: str | None = None,
    labels_sha256: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Convenience one-shot: build both the labels dict and the summary dict
    for one (severity, regime) bucket over `scenario_ids`. Returns
    (summary, labels); the caller is responsible for writing `labels` to
    results/mask_manifests/local/ (gitignored) and `summary` to
    results/mask_manifests/ (committable) -- see scripts/gen_eval_masks.py."""
    labels = build_labels(scenario_ids, severity, regime, spec_version=spec_version)
    summary = build_summary(
        labels, severity, regime, spec_version=spec_version,
        labels_path=labels_path, labels_sha256=labels_sha256,
    )
    return summary, labels
