"""Tests for the G-N1-2 paired significance test (scripts/paired_test.py).

The test procedure was PRE-REGISTERED in D-N1-12 before any masked training
existed, so the thing worth pinning here is that the implementation matches the
registration -- direction of the alternative, seed-averaging before pairing,
Holm across exactly three severities, and the gate keys check_gates.py reads.

A statistical test that silently gets the direction backwards, or pairs
(scenario, seed) instead of scenario, would still emit a plausible p-value.
These tests are what stop that shipping.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_test import (  # noqa: E402
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    GATE_SEVERITY,
    SEVERITIES,
    bootstrap_ci,
    holm_bonferroni,
    load_arm,
    paired_wilcoxon,
    severity_record,
)


# --------------------------------------------------------------------------
# fixtures: eval JSONs in the scripts/eval_local.py output format
# --------------------------------------------------------------------------
def write_eval(path: Path, label: str, values: dict[str, dict[str, float]],
               patterns: dict[str, str] | None = None,
               mask_spec: str = "N1-mask-v2", regime: str = "R-A") -> Path:
    blob = {
        "label": label, "mode": "native", "regime": regime,
        "mask_spec": mask_spec, "ckpt": f"{label}.tar", "ckpt_epoch": 5,
        "severities": {
            sev: {"aggregate": {"n": len(sc)},
                  "per_scenario": [
                      {"sid": sid, "pattern": (patterns or {}).get(sid, "M1"),
                       "minade6": v / 2.0, "minfde6": v, "miss": v > 2.0}
                      for sid, v in sc.items()]}
            for sev, sc in values.items()},
    }
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


def arm_values(sids: list[str], base: float, offset: float = 0.0
               ) -> dict[str, dict[str, float]]:
    return {sev: {s: base + offset + 0.1 * i for i, s in enumerate(sids)}
            for sev in SEVERITIES}


# --------------------------------------------------------------------------
# Holm-Bonferroni
# --------------------------------------------------------------------------
def test_holm_scales_the_smallest_p_by_the_number_of_tests() -> None:
    adj = holm_bonferroni({"S2": 0.01, "S3": 0.04, "S4": 0.20})
    assert adj["S2"] == pytest.approx(0.03)   # 3 * 0.01
    assert adj["S3"] == pytest.approx(0.08)   # 2 * 0.04
    assert adj["S4"] == pytest.approx(0.20)   # 1 * 0.20


def test_holm_is_monotone_non_decreasing() -> None:
    """The running max matters: without it a later, larger raw p can produce a
    SMALLER adjusted p than an earlier one, which is not a valid Holm output."""
    adj = holm_bonferroni({"S2": 0.03, "S3": 0.031, "S4": 0.032})
    ordered = [adj["S2"], adj["S3"], adj["S4"]]
    assert ordered == sorted(ordered), ordered
    assert adj["S2"] == pytest.approx(0.09)
    assert adj["S3"] == pytest.approx(0.09)  # 2*0.031=0.062 < 0.09 -> clamped up


def test_holm_clips_at_one() -> None:
    adj = holm_bonferroni({"S2": 0.5, "S3": 0.6, "S4": 0.9})
    assert all(v <= 1.0 for v in adj.values())
    assert adj["S4"] == 1.0


# --------------------------------------------------------------------------
# Wilcoxon direction + the collapse case
# --------------------------------------------------------------------------
def test_alternative_is_c3_better_not_c1_better() -> None:
    """delta = C1 - C3, H1: C3 < C1. Positive deltas must give a SMALL p.
    Getting this backwards is the single most damaging possible bug here."""
    better = np.full(40, 0.5) + np.linspace(0, 0.1, 40)
    p_better, _, _ = paired_wilcoxon(better)
    p_worse, _, _ = paired_wilcoxon(-better)
    assert p_better < 0.001, p_better
    assert p_worse > 0.999, p_worse


def test_all_zero_deltas_report_p_one_and_flag_collapse() -> None:
    """A collapsed arm produces bit-identical metrics (D-N1-14c). scipy raises on
    an all-zero vector; the gate must get p=1.0 and a warning, not a traceback."""
    p, stat, note = paired_wilcoxon(np.zeros(30))
    assert p == 1.0
    assert stat is None
    assert "collapse" in note


def test_empty_input_does_not_crash() -> None:
    p, stat, note = paired_wilcoxon(np.array([]))
    assert p == 1.0 and note == "empty"


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_is_seeded_and_reproducible() -> None:
    d = np.random.default_rng(7).normal(0.3, 1.0, 200)
    assert bootstrap_ci(d) == bootstrap_ci(d)


def test_bootstrap_ci_brackets_the_sample_mean() -> None:
    d = np.random.default_rng(7).normal(0.3, 1.0, 400)
    lo, hi = bootstrap_ci(d)
    assert lo < d.mean() < hi
    assert BOOTSTRAP_RESAMPLES == 10_000  # pre-registered


# --------------------------------------------------------------------------
# seed aggregation — mean over seeds BEFORE pairing
# --------------------------------------------------------------------------
def test_seed_mean_is_taken_before_pairing(tmp_path: Path) -> None:
    sids = ["a", "b", "c"]
    paths = []
    for k, base in enumerate((1.0, 2.0, 6.0)):      # per-scenario mean = 3.0
        v = {sev: {s: base for s in sids} for sev in SEVERITIES}
        paths.append(write_eval(tmp_path / f"c1_s{k}.json", "C1-local", v))
    arm = load_arm(paths, "C1")
    assert arm["n_seeds"] == 3
    for sev in SEVERITIES:
        for s in sids:
            assert arm["mean"][sev][s] == pytest.approx(3.0)


def test_mismatched_scenario_sets_are_rejected(tmp_path: Path) -> None:
    a = write_eval(tmp_path / "a.json", "C1-local", arm_values(["a", "b"], 1.0))
    b = write_eval(tmp_path / "b.json", "C1-local", arm_values(["a", "c"], 1.0))
    with pytest.raises(SystemExit, match="scenario sets differ"):
        load_arm([a, b], "C1")


def test_mismatched_mask_specs_are_rejected(tmp_path: Path) -> None:
    a = write_eval(tmp_path / "a.json", "C1-local", arm_values(["a", "b"], 1.0))
    b = write_eval(tmp_path / "b.json", "C1-local", arm_values(["a", "b"], 1.0),
                   mask_spec="N1-mask-v1")
    with pytest.raises(SystemExit, match="mixed mask specs"):
        load_arm([a, b], "C1")


def test_missing_severity_is_rejected(tmp_path: Path) -> None:
    vals = arm_values(["a", "b"], 1.0)
    vals.pop("S4")
    a = write_eval(tmp_path / "a.json", "C1-local", vals)
    with pytest.raises(SystemExit, match="S4 missing"):
        load_arm([a], "C1")


# --------------------------------------------------------------------------
# per-severity record
# --------------------------------------------------------------------------
def test_severity_record_reports_direction_and_fraction(tmp_path: Path) -> None:
    sids = [f"s{i}" for i in range(20)]
    c1 = load_arm([write_eval(tmp_path / "c1.json", "C1-local",
                              arm_values(sids, 5.0))], "C1")
    c3 = load_arm([write_eval(tmp_path / "c3.json", "C3-local",
                              arm_values(sids, 4.0))], "C3")
    r = severity_record(c1, c3, "S2", sorted(sids))
    assert r["c3_beats_c1"] is True
    assert r["mean_delta_m"] == pytest.approx(1.0)
    assert r["frac_scenarios_c3_better"] == 1.0
    assert r["n_scenarios"] == 20


# --------------------------------------------------------------------------
# end to end — the artifact check_gates.py actually reads
# --------------------------------------------------------------------------
def _run(tmp_path: Path, c1_paths, c3_paths) -> dict:
    out = tmp_path / "g_n1_2_headline.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "paired_test.py"),
         "--c1", *[str(p) for p in c1_paths],
         "--c3", *[str(p) for p in c3_paths], "--out", str(out)],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_end_to_end_emits_the_three_gate_keys(tmp_path: Path) -> None:
    sids = [f"s{i}" for i in range(60)]
    c1 = [write_eval(tmp_path / f"c1_{k}.json", "C1-local", arm_values(sids, 5.0))
          for k in range(3)]
    c3 = [write_eval(tmp_path / f"c3_{k}.json", "C3-local", arm_values(sids, 4.0))
          for k in range(3)]
    blob = _run(tmp_path, c1, c3)

    # exactly the keys gates.yaml G_N1_2_headline resolves
    assert blob["n_seeds_per_arm"] == 3
    assert isinstance(blob["paired_p_s2"], float)
    assert blob["c3_beats_c1_s2"] is True
    assert blob["paired_p_s2"] <= ALPHA, blob["paired_p_s2"]

    # paired_p_s2 must be the HOLM-adjusted p at S2, not the raw one
    assert blob["paired_p_s2"] == pytest.approx(
        blob["severities"][GATE_SEVERITY]["wilcoxon_p_holm"])
    assert blob["severities_tested"] == list(SEVERITIES)
    assert blob["mask_spec"] == "N1-mask-v2"


def test_end_to_end_direction_key_requires_all_three_severities(tmp_path: Path) -> None:
    """c3_beats_c1_s2 is an ALL-of check over S2/S3/S4 (D-N1-12), so C3 winning
    at S2 alone must not set it."""
    sids = [f"s{i}" for i in range(40)]
    c1v = arm_values(sids, 5.0)
    c3v = arm_values(sids, 4.0)
    c3v["S4"] = {s: v + 3.0 for s, v in c3v["S4"].items()}   # C3 worse at S4 only
    c1 = [write_eval(tmp_path / "c1.json", "C1-local", c1v)]
    c3 = [write_eval(tmp_path / "c3.json", "C3-local", c3v)]
    blob = _run(tmp_path, c1, c3)
    assert blob["severities"]["S2"]["c3_beats_c1"] is True
    assert blob["severities"]["S4"]["c3_beats_c1"] is False
    assert blob["c3_beats_c1_s2"] is False


def test_end_to_end_reports_underpowered_seed_count_honestly(tmp_path: Path) -> None:
    """One seed per arm must still produce an artifact whose n_seeds_per_arm is 1,
    so check_gates fails the min:3 criterion. Silently padding, or refusing to
    run, would both hide the shortfall."""
    sids = [f"s{i}" for i in range(30)]
    c1 = [write_eval(tmp_path / "c1.json", "C1-local", arm_values(sids, 5.0))]
    c3 = [write_eval(tmp_path / "c3.json", "C3-local", arm_values(sids, 4.0))]
    blob = _run(tmp_path, c1, c3)
    assert blob["n_seeds_per_arm"] == 1


def test_end_to_end_collapsed_arm_does_not_crash_the_gate(tmp_path: Path) -> None:
    """C3-local collapsed to history-invariant (D-N1-14c). Identical arms must
    yield p=1.0 and c3_beats_c1_s2=False, not a traceback."""
    sids = [f"s{i}" for i in range(30)]
    vals = arm_values(sids, 5.0)
    c1 = [write_eval(tmp_path / "c1.json", "C1-local", vals)]
    c3 = [write_eval(tmp_path / "c3.json", "C3-local", vals)]
    blob = _run(tmp_path, c1, c3)
    assert blob["paired_p_s2"] == 1.0
    assert blob["c3_beats_c1_s2"] is False
    assert "collapse" in blob["severities"]["S2"]["note"]


def test_per_pattern_cohorts_are_present_and_marked_report_only(tmp_path: Path) -> None:
    sids = [f"s{i}" for i in range(30)]
    pats = {s: ("M1", "M2", "M3")[i % 3] for i, s in enumerate(sids)}
    c1 = [write_eval(tmp_path / "c1.json", "C1-local", arm_values(sids, 5.0), pats)]
    c3 = [write_eval(tmp_path / "c3.json", "C3-local", arm_values(sids, 4.0), pats)]
    blob = _run(tmp_path, c1, c3)
    assert "per_pattern_report_only" in blob
    assert set(blob["per_pattern_report_only"]["S2"]) == {"M1", "M2", "M3"}
    # and they must NOT leak into the gate keys
    assert set(blob) & {"per_pattern_p_s2", "m1_p_s2"} == set()
