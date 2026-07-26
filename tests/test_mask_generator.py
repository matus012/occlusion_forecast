"""Unit tests for otraj.masking.generator / otraj.masking.manifest (Phase 3,
D-N1-2 as amended by D-N1-9 / D-N1-10, and by the adversarial-review fixes
B1-B3 / M1-M7 / N2). Torch-free.

Some tests reach into module-private helpers (leading underscore) -- this is
deliberate: `generate_mask`/`draw_train_mask` draw their pattern internally
from the mix, so exercising a SPECIFIC pattern deterministically (needed for
the M1/M2/M3-shape tests and the clip-precedence test) requires calling the
lower-level `_build_pattern_mask` / `_draw_and_scale_lengths` directly with a
manually-seeded generator. Separately, `test_structure_matches_pattern_*`
covers the same shapes through the fully PUBLIC `generate_mask` API (review
B3c), so both the mechanism and the end-to-end path are exercised.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from otraj.masking import generator as gen
from otraj.masking import manifest as manifest_mod
from otraj.masking.manifest import (
    build_labels,
    build_manifest,
    build_summary,
    compute_spec_hash,
    spec_constants,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mask_manifests"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) inclusive, in index order."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# --------------------------------------------------------------------------
# Fraction-in-band guarantee: every severity x regime x (naturally drawn)
# pattern, over >=200 synthetic scenario ids.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("severity", list(gen.SEVERITIES))
@pytest.mark.parametrize("regime", list(gen.REGIMES))
def test_fraction_in_band_over_many_scenarios(severity: str, regime: str) -> None:
    n_ids = 300
    seen_patterns: set[str] = set()
    target = gen.SEVERITY_TARGETS[severity]
    for i in range(n_ids):
        sid = f"synthetic_{i:04d}"
        result = gen.generate_mask(sid, severity, regime)
        seen_patterns.add(result.pattern)
        assert abs(result.achieved_fraction - target) <= gen.SEVERITY_TOLERANCE, (
            f"{sid}/{severity}/{regime}: achieved={result.achieved_fraction} "
            f"target={target} pattern={result.pattern}"
        )
        assert result.n_masked == round(result.achieved_fraction * gen.N_STEPS)

    if severity == "S0":
        assert seen_patterns == {"none"}  # M5: S0 never draws a real pattern
    elif regime == "R-A":
        # all three patterns should show up across 300 draws of a 5% mix
        assert seen_patterns == {"M1", "M2", "M3"}
    else:
        # M2 is excluded from R-B by construction
        assert seen_patterns == {"M1", "M3"}


# --------------------------------------------------------------------------
# B3c: mask STRUCTURE matches the pattern label, through the PUBLIC API.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("regime", list(gen.REGIMES))
def test_structure_matches_pattern_through_public_api(regime: str) -> None:
    for severity in ["S1", "S2", "S3", "S4"]:
        for i in range(80):
            sid = f"struct_{regime}_{severity}_{i:03d}"
            result = gen.generate_mask(sid, severity, regime)
            runs = _runs(result.mask)
            if result.pattern == "M1":
                assert len(runs) == 1
            elif result.pattern == "M2":
                assert len(runs) == 1
                assert runs[0][0] == 0
            elif result.pattern == "M3":
                assert gen.M3_MIN_BLOCKS <= len(runs) <= gen.M3_MAX_BLOCKS
            else:
                raise AssertionError(f"unexpected pattern {result.pattern!r} for {sid}")


# --------------------------------------------------------------------------
# B1 (CRITICAL FIX): a scenario's pattern must be STABLE across severities
# within a regime, so per-pattern cohorts (D-N1-10) are the same population
# across the S1-S4 curve.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("regime", list(gen.REGIMES))
def test_pattern_stable_across_severities_within_regime(regime: str) -> None:
    n_ids = 400
    severities = ["S1", "S2", "S3", "S4"]
    n_stable = 0
    for i in range(n_ids):
        sid = f"cohort_{regime}_{i:04d}"
        patterns = {gen.generate_mask(sid, sev, regime).pattern for sev in severities}
        if len(patterns) == 1:
            n_stable += 1
    assert n_stable == n_ids, (
        f"{regime}: only {n_stable}/{n_ids} scenarios kept ONE pattern across S1-S4 "
        "(pattern must be drawn from a severity-independent stream)"
    )


def test_pattern_seed_is_severity_independent() -> None:
    assert gen.pattern_seed("scn_x", "R-A") == gen.pattern_seed("scn_x", "R-A")
    # placement_seed (severity-dependent) must still differ across severities
    assert (
        gen.placement_seed("scn_x", "S1", "R-A") != gen.placement_seed("scn_x", "S2", "R-A")
    )


# --------------------------------------------------------------------------
# B3b: empirical pattern mix matches MIX_RA / MIX_RB over many draws
# (binomial tolerance check -- catches a mix regression).
# --------------------------------------------------------------------------


def _binomial_tolerance(p: float, n: int, z: float = 6.0) -> float:
    """Generous (z=6 sigma) binomial-proportion tolerance: essentially zero
    false-positive rate at this n, while still catching a real mix
    regression (e.g. reverting to the v1 prior 0.6/0.25/0.15)."""
    se = (p * (1 - p) / n) ** 0.5
    return max(z * se, 0.01)


def test_empirical_mix_matches_target_ra() -> None:
    n = 6000
    counts = {"M1": 0, "M2": 0, "M3": 0}
    for i in range(n):
        result = gen.generate_mask(f"mixcheck_ra_{i:05d}", "S2", "R-A")
        counts[result.pattern] += 1
    for pattern, expected_p in gen.MIX_RA.items():
        observed_p = counts[pattern] / n
        tol = _binomial_tolerance(expected_p, n)
        assert abs(observed_p - expected_p) <= tol, (
            f"{pattern}: observed={observed_p:.4f} expected={expected_p:.4f} tol={tol:.4f}"
        )


def test_empirical_mix_matches_target_rb() -> None:
    n = 6000
    counts = {"M1": 0, "M3": 0}
    for i in range(n):
        result = gen.generate_mask(f"mixcheck_rb_{i:05d}", "S2", "R-B")
        counts[result.pattern] += 1
    assert "M2" not in counts or counts.get("M2", 0) == 0
    for pattern, expected_p in gen.MIX_RB.items():
        observed_p = counts[pattern] / n
        tol = _binomial_tolerance(expected_p, n)
        assert abs(observed_p - expected_p) <= tol, (
            f"{pattern}: observed={observed_p:.4f} expected={expected_p:.4f} tol={tol:.4f}"
        )


# --------------------------------------------------------------------------
# Determinism: same inputs -> identical masks across two FRESH processes.
# --------------------------------------------------------------------------


def test_determinism_across_fresh_processes() -> None:
    script = (
        "from otraj.masking.generator import generate_mask\n"
        "import hashlib\n"
        "combos = [('sid_a', 'S2', 'R-A'), ('sid_b', 'S4', 'R-B'), "
        "('sid_c', 'S1', 'R-A'), ('sid_d', 'S3', 'R-B')]\n"
        "parts = []\n"
        "for sid, sev, reg in combos:\n"
        "    r = generate_mask(sid, sev, reg)\n"
        "    parts.append(r.mask.tobytes() + r.pattern.encode() + str(r.seed).encode())\n"
        "print(hashlib.sha256(b''.join(parts)).hexdigest())\n"
    )
    out1 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60,
    )
    out2 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60,
    )
    assert out1.stdout.strip() != ""
    assert out1.stdout.strip() == out2.stdout.strip()


def test_determinism_same_process_repeat_call() -> None:
    r1 = gen.generate_mask("scn_x", "S3", "R-A")
    r2 = gen.generate_mask("scn_x", "S3", "R-A")
    assert np.array_equal(r1.mask, r2.mask)
    assert r1.pattern == r2.pattern
    assert r1.seed == r2.seed


def test_different_scenario_id_generally_differs() -> None:
    r1 = gen.generate_mask("scn_alpha", "S3", "R-A")
    r2 = gen.generate_mask("scn_beta", "S3", "R-A")
    assert r1.seed != r2.seed


# --------------------------------------------------------------------------
# R-A: indices 45..49 never masked. R-B: index 49 always masked.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("severity", ["S1", "S2", "S3", "S4"])
def test_ra_forced_tail_never_masked(severity: str) -> None:
    for i in range(100):
        result = gen.generate_mask(f"ra_tail_{i:04d}", severity, "R-A")
        assert not result.mask[gen.N_STEPS - gen.FORCED_VISIBLE_TAIL:].any()


@pytest.mark.parametrize("severity", ["S1", "S2", "S3", "S4"])
def test_rb_index_49_always_masked(severity: str) -> None:
    for i in range(100):
        result = gen.generate_mask(f"rb_t0_{i:04d}", severity, "R-B")
        assert result.mask[gen.N_STEPS - 1]


def test_s0_is_always_empty_regardless_of_regime() -> None:
    for regime in gen.REGIMES:
        result = gen.generate_mask("s0_scn", "S0", regime)
        assert not result.mask.any()
        assert result.n_masked == 0
        assert result.achieved_fraction == 0.0
        assert result.pattern == "none"  # M5: S0 draws no pattern at all


# --------------------------------------------------------------------------
# Pattern shape tests (force the pattern via the private core builder).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("severity", ["S1", "S2", "S3", "S4"])
@pytest.mark.parametrize("regime", ["R-A", "R-B"])
def test_m1_is_exactly_one_contiguous_run(severity: str, regime: str) -> None:
    for seed in range(30):
        rng = np.random.default_rng(seed)
        mask, achieved, n_masked = gen._build_pattern_mask(rng, "M1", severity, regime)
        runs = _runs(mask)
        assert len(runs) == 1
        start, end = runs[0]
        assert end - start + 1 == n_masked
        assert abs(achieved - gen.SEVERITY_TARGETS[severity]) <= gen.SEVERITY_TOLERANCE
        if regime == "R-B":
            assert end == gen.N_STEPS - 1


@pytest.mark.parametrize("severity", ["S1", "S2", "S3", "S4"])
def test_m2_prefix_always_starts_at_index_0(severity: str) -> None:
    for seed in range(30):
        rng = np.random.default_rng(seed)
        mask, achieved, n_masked = gen._build_pattern_mask(rng, "M2", severity, "R-A")
        runs = _runs(mask)
        assert len(runs) == 1
        start, end = runs[0]
        assert start == 0
        assert end == n_masked - 1
        assert abs(achieved - gen.SEVERITY_TARGETS[severity]) <= gen.SEVERITY_TOLERANCE


def test_m2_excluded_from_rb_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="R-B"):
        gen._build_pattern_mask(rng, "M2", "S2", "R-B")


@pytest.mark.parametrize("severity", ["S1", "S2", "S3", "S4"])
@pytest.mark.parametrize("regime", ["R-A", "R-B"])
def test_m3_produces_2_to_4_runs(severity: str, regime: str) -> None:
    seen_counts: set[int] = set()
    for seed in range(60):
        rng = np.random.default_rng(seed)
        mask, achieved, n_masked = gen._build_pattern_mask(rng, "M3", severity, regime)
        runs = _runs(mask)
        assert gen.M3_MIN_BLOCKS <= len(runs) <= gen.M3_MAX_BLOCKS
        seen_counts.add(len(runs))
        assert sum(e - s + 1 for s, e in runs) == n_masked
        assert abs(achieved - gen.SEVERITY_TARGETS[severity]) <= gen.SEVERITY_TOLERANCE
        if regime == "R-B":
            assert runs[-1][1] == gen.N_STEPS - 1
    assert seen_counts.issubset({2, 3, 4})


def test_m3_scaling_can_violate_per_block_clip_fraction_guarantee_wins() -> None:
    """S1 (target=10 steps) split across 4 M3 blocks averages 2.5 steps
    (0.25s) each -- comfortably inside [0.2, 2.0]s most of the time, so force
    a harder case directly: 4 blocks squeezed into a 4-step total (1.0s /
    4Hz-equivalent -> 1 step = 0.1s each), which is below the M3 clip's 0.2s
    (2-step) floor. This demonstrates the documented precedence: the exact-
    sum fraction guarantee wins over the per-block duration clip."""
    rng = np.random.default_rng(0)
    lengths = gen._draw_and_scale_lengths(rng, n_blocks=4, clip_lo_s=0.2, clip_hi_s=2.0,
                                           target_steps=4)
    assert sum(lengths) == 4
    assert len(lengths) == 4
    min_clip_steps = int(round(0.2 * gen.HZ))  # 2 steps
    assert any(length < min_clip_steps for length in lengths), (
        "expected at least one block below the M3 clip floor when squeezed to a tiny total"
    )


def test_ra_s4_feasible_within_45_step_domain() -> None:
    """R-A caps maskable steps at 45 (50 - forced-visible tail of 5); S4's
    target of 40 steps must fit (40 <= 45)."""
    assert gen.target_steps_for_severity("S4") == 40
    assert gen.N_STEPS - gen.FORCED_VISIBLE_TAIL == 45
    rng = np.random.default_rng(1)
    mask, achieved, n_masked = gen._build_pattern_mask(rng, "M1", "S4", "R-A")
    assert n_masked == 40
    assert not mask[45:].any()


# --------------------------------------------------------------------------
# M6: load-bearing invariants raise RuntimeError (not bare `assert`, which
# `python -O` strips) when the internal placement machinery misbehaves.
# --------------------------------------------------------------------------


def test_place_blocks_raises_valueerror_when_infeasible() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        gen._place_blocks(rng, [46], domain_size=45, anchor_last_to_end=False)


def test_ra_tail_violation_raises_runtime_error_not_assert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6: the R-A forced-visible-tail invariant is an explicit `raise
    RuntimeError`, not a bare `assert` (which `python -O` strips). Force it
    by monkeypatching _place_blocks to return a span that touches the tail
    (indices 45+), which _build_pattern_mask must then catch and reject."""

    def _bad_place_blocks(rng, lengths, domain_size, anchor_last_to_end):  # noqa: ANN001, ANN202
        return [(40, 46)]

    monkeypatch.setattr(gen, "_place_blocks", _bad_place_blocks)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="R-A"):
        gen._build_pattern_mask(rng, "M1", "S1", "R-A")


def test_rb_index49_violation_raises_runtime_error_not_assert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6: same for the R-B 'must cover index 49' invariant."""

    def _bad_place_blocks(rng, lengths, domain_size, anchor_last_to_end):  # noqa: ANN001, ANN202
        return [(0, 9)]  # never reaches index 49

    monkeypatch.setattr(gen, "_place_blocks", _bad_place_blocks)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="49"):
        gen._build_pattern_mask(rng, "M1", "S1", "R-B")


# --------------------------------------------------------------------------
# Mix values
# --------------------------------------------------------------------------


def test_mix_ra_values() -> None:
    assert gen.MIX_RA == {"M1": 0.40, "M2": 0.05, "M3": 0.55}
    assert sum(gen.MIX_RA.values()) == pytest.approx(1.0)


def test_mix_rb_renormalized_from_mix_ra() -> None:
    expected_m1 = 0.40 / 0.95
    expected_m3 = 0.55 / 0.95
    assert gen.MIX_RB["M1"] == pytest.approx(expected_m1)
    assert gen.MIX_RB["M3"] == pytest.approx(expected_m3)
    assert "M2" not in gen.MIX_RB
    assert sum(gen.MIX_RB.values()) == pytest.approx(1.0)
    # explicitly matches the task-brief-quoted approximations
    assert gen.MIX_RB["M1"] == pytest.approx(0.421, abs=1e-3)
    assert gen.MIX_RB["M3"] == pytest.approx(0.579, abs=1e-3)


def test_mix_for_regime() -> None:
    assert gen.mix_for_regime("R-A") == gen.MIX_RA
    assert gen.mix_for_regime("R-B") == gen.MIX_RB
    with pytest.raises(ValueError):
        gen.mix_for_regime("R-C")


# --------------------------------------------------------------------------
# M3 (fail-fast fit loading)
# --------------------------------------------------------------------------


def test_load_duration_fit_fails_fast_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        gen._load_duration_fit(tmp_path / "does_not_exist.json")


def test_load_duration_fit_fails_fast_on_malformed_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not_the_right_key": {}}), encoding="utf-8")
    with pytest.raises(KeyError):
        gen._load_duration_fit(bad)


def test_load_duration_fit_reads_real_committed_file() -> None:
    fit = gen._load_duration_fit()
    assert fit["shape"] == pytest.approx(gen.FIT_SHAPE)
    assert fit["scale"] == pytest.approx(gen.FIT_SCALE)


# --------------------------------------------------------------------------
# spec_hash sensitivity (N2: loop over EVERY spec_constants() key)
# --------------------------------------------------------------------------


def test_spec_hash_deterministic_and_stable() -> None:
    assert compute_spec_hash() == compute_spec_hash()


def test_spec_hash_changes_with_spec_version() -> None:
    assert compute_spec_hash("v1") != compute_spec_hash("v2")


_SPEC_HASH_MUTATIONS: list[tuple[str, object]] = [
    ("N_STEPS", 51),
    ("HZ", 11.0),
    ("FORCED_VISIBLE_TAIL", 6),
    ("SEVERITY_TARGETS", {**gen.SEVERITY_TARGETS, "S1": 0.99}),
    ("SEVERITY_TOLERANCE", 0.99),
    ("MIX_RA", {"M1": 0.5, "M2": 0.05, "M3": 0.45}),
    ("MIX_RB", {"M1": 0.6, "M3": 0.4}),
    ("M1_CLIP_S", (0.5, 4.5)),
    ("M3_CLIP_S", (0.5, 2.5)),
    ("M3_MIN_BLOCKS", 3),
    ("M3_MAX_BLOCKS", 5),
    ("FIT_SHAPE", 9.99),
    ("FIT_LOC", 0.5),
    ("FIT_SCALE", 9.99),
]

# manifest.py's spec_constants() key -> the module attribute name(s) that
# feed it, so this table can be checked for completeness against
# spec_constants() itself (see test below) -- "duration_fit" maps to three
# attrs (shape/loc/scale), all three must be covered.
_KEY_TO_ATTRS = {
    "n_steps": {"N_STEPS"},
    "hz": {"HZ"},
    "forced_visible_tail": {"FORCED_VISIBLE_TAIL"},
    "severity_targets": {"SEVERITY_TARGETS"},
    "severity_tolerance": {"SEVERITY_TOLERANCE"},
    "mix_ra": {"MIX_RA"},
    "mix_rb": {"MIX_RB"},
    "m1_clip_s": {"M1_CLIP_S"},
    "m3_clip_s": {"M3_CLIP_S"},
    "m3_min_blocks": {"M3_MIN_BLOCKS"},
    "m3_max_blocks": {"M3_MAX_BLOCKS"},
    "duration_fit": {"FIT_SHAPE", "FIT_LOC", "FIT_SCALE"},
}


@pytest.mark.parametrize("attr,new_value", _SPEC_HASH_MUTATIONS)
def test_spec_hash_changes_for_every_constant(
    attr: str, new_value: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = compute_spec_hash()
    monkeypatch.setattr(manifest_mod, attr, new_value)
    changed = compute_spec_hash()
    assert baseline != changed, f"mutating {attr} did not change spec_hash"


def test_spec_hash_mutations_cover_every_spec_constants_key() -> None:
    """N2: every key spec_constants() reports (besides spec_version, a
    function parameter rather than a module constant) has >=1 mutation
    exercised in _SPEC_HASH_MUTATIONS above."""
    covered_attrs = {attr for attr, _ in _SPEC_HASH_MUTATIONS}
    keys = set(spec_constants().keys()) - {"spec_version"}
    assert keys == set(_KEY_TO_ATTRS.keys()), "spec_constants() keys drifted from this test's map"
    for key, attrs in _KEY_TO_ATTRS.items():
        assert attrs.issubset(covered_attrs), f"{key} ({attrs}) not covered by a mutation test"


def test_spec_constants_contains_all_documented_categories() -> None:
    keys = set(spec_constants().keys())
    assert {
        "mix_ra", "mix_rb", "m1_clip_s", "m3_clip_s", "duration_fit",
        "severity_targets", "severity_tolerance", "forced_visible_tail",
    }.issubset(keys)


# --------------------------------------------------------------------------
# Train-time API
# --------------------------------------------------------------------------


def test_train_p_occ_respected_statistically() -> None:
    n = 1000
    n_masked_draws = 0
    for i in range(n):
        result = gen.draw_train_mask(f"train_scn_{i:05d}", epoch=0)
        if result.n_masked > 0:
            n_masked_draws += 1
    frac = n_masked_draws / n
    assert 0.45 <= frac <= 0.55, f"observed masked-draw fraction {frac} outside [0.45, 0.55]"


def test_train_ra_regime_safety_tail_never_masked() -> None:
    for i in range(300):
        result = gen.draw_train_mask(f"train_safety_{i:04d}", epoch=3)
        assert result.regime == "R-A"
        assert not result.mask[gen.N_STEPS - gen.FORCED_VISIBLE_TAIL:].any()


def test_train_epoch_changes_the_draw() -> None:
    masks = []
    for epoch in range(20):
        result = gen.draw_train_mask("fixed_scenario_id", epoch=epoch)
        masks.append(result.mask.tobytes())
    assert len(set(masks)) > 1, "varying epoch never changed the draw across 20 epochs"


def test_train_severity_drawn_from_s1_to_s4_only() -> None:
    for i in range(200):
        result = gen.draw_train_mask(f"train_sev_{i:04d}", epoch=7)
        if result.n_masked > 0:
            assert result.severity in gen.TRAIN_SEVERITIES
        else:
            assert result.severity == "S0"
            assert result.pattern == "none"


def test_train_determinism_same_epoch_same_scenario() -> None:
    r1 = gen.draw_train_mask("scn_repeat", epoch=5)
    r2 = gen.draw_train_mask("scn_repeat", epoch=5)
    assert np.array_equal(r1.mask, r2.mask)
    assert r1.pattern == r2.pattern


# --------------------------------------------------------------------------
# Manifest: build_labels (local-only) + build_summary (committable)
# --------------------------------------------------------------------------


def test_build_labels_shape() -> None:
    scenario_ids = [f"m_{i:03d}" for i in range(50)]
    labels = build_labels(scenario_ids, "S2", "R-A")
    assert set(labels.keys()) == set(scenario_ids)
    assert all(p in {"M1", "M2", "M3"} for p in labels.values())


def test_build_summary_shape_and_content() -> None:
    scenario_ids = [f"m_{i:03d}" for i in range(50)]
    labels = build_labels(scenario_ids, "S2", "R-A")
    summary = build_summary(
        labels, "S2", "R-A", labels_path="local/x.json", labels_sha256="deadbeef",
    )
    assert summary["severity"] == "S2"
    assert summary["regime"] == "R-A"
    assert summary["n_scenarios"] == 50
    assert summary["rb_mix_note"] == ""
    assert summary["spec_hash"] == compute_spec_hash()
    assert summary["per_scenario_labels_path"] == "local/x.json"
    assert summary["per_scenario_labels_sha256"] == "deadbeef"
    assert isinstance(summary["seed_derivation"], str) and summary["seed_derivation"]
    agg = summary["aggregate"]
    assert agg["constant_achieved_fraction"] == pytest.approx(0.4)
    assert agg["constant_n_masked"] == 20
    assert sum(agg["pattern_counts"].values()) == 50
    # masks/per-scenario fractions are NEVER in the committable summary
    assert "scenarios" not in summary


def test_build_summary_rb_carries_note_and_excludes_m2() -> None:
    scenario_ids = [f"rb_{i:03d}" for i in range(80)]
    labels = build_labels(scenario_ids, "S3", "R-B")
    summary = build_summary(labels, "S3", "R-B")
    assert summary["rb_mix_note"] != ""
    assert "M2" not in summary["aggregate"]["pattern_counts"]
    assert set(summary["mix_used"].keys()) == {"M1", "M3"}


def test_build_manifest_convenience_matches_separate_calls() -> None:
    scenario_ids = [f"j_{i:03d}" for i in range(20)]
    summary, labels = build_manifest(scenario_ids, "S1", "R-A")
    assert set(labels.keys()) == set(scenario_ids)
    assert summary["n_scenarios"] == 20


def test_summary_json_serializable() -> None:
    scenario_ids = [f"j_{i:03d}" for i in range(20)]
    labels = build_labels(scenario_ids, "S1", "R-A")
    summary = build_summary(labels, "S1", "R-A", labels_path="x", labels_sha256="y")
    decoded = json.loads(json.dumps(summary))
    assert decoded["severity"] == "S1"


def test_labels_json_serializable() -> None:
    scenario_ids = [f"j_{i:03d}" for i in range(20)]
    labels = build_labels(scenario_ids, "S1", "R-A")
    decoded = json.loads(json.dumps(labels))
    assert set(decoded.keys()) == set(scenario_ids)


def test_summary_size_independent_of_n_scenarios() -> None:
    """B2/M7: the committable summary holds only aggregate pattern counts
    (<=3 entries), never per-scenario data, so its size must not scale with
    n_scenarios beyond trivial counter growth."""
    small_labels = build_labels([f"a_{i}" for i in range(10)], "S2", "R-A")
    large_labels = build_labels([f"b_{i}" for i in range(5000)], "S2", "R-A")
    small_summary = build_summary(small_labels, "S2", "R-A")
    large_summary = build_summary(large_labels, "S2", "R-A")
    small_size = len(json.dumps(small_summary))
    large_size = len(json.dumps(large_summary))
    assert large_size - small_size < 200, (
        f"summary size grew {large_size - small_size} bytes over a 500x scenario-count "
        "increase -- per-scenario data may have leaked into the committable summary"
    )


# --------------------------------------------------------------------------
# B2: committed manifest files stay small (< 1MB each); B3a: not stale.
# Both skip cleanly if results/mask_manifests/ hasn't been generated yet.
# --------------------------------------------------------------------------


def _committed_summary_files() -> list[Path]:
    if not MANIFESTS_DIR.exists():
        return []
    return [p for p in MANIFESTS_DIR.glob("*.json") if p.is_file()]


def test_committed_manifest_files_under_1mb() -> None:
    files = _committed_summary_files()
    if not files:
        pytest.skip("no results/mask_manifests summary files present")
    for p in files:
        size = p.stat().st_size
        assert size < 1_000_000, f"{p.name}: {size} bytes >= 1MB ceiling (B2)"


def test_committed_manifest_spec_hash_not_stale() -> None:
    files = _committed_summary_files()
    if not files:
        pytest.skip("no results/mask_manifests summary files present")
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        # compare against the LIVE spec version, not the file's own — otherwise a
        # SPEC_VERSION bump without regeneration cancels out and goes undetected
        assert data["spec_version"] == gen.SPEC_VERSION, (
            f"{p.name}: manifest from old spec version (stale after SPEC_VERSION bump)"
        )
        assert data["spec_hash"] == compute_spec_hash(), f"{p.name}: stale spec_hash (B3a)"
