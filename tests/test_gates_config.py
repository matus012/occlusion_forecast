"""Sanity checks on gates.yaml — torch-free, runs in CI.

Guards the gate-file contract check_gates.py relies on, so a malformed edit
fails fast instead of silently turning a gate into a no-op.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BOUND_KEYS = ("min", "max", "rel_min", "rel_max", "equals")
EXPECTED_GATES = (
    "G_N1_0_evalkit",
    "G_N1_1_repro",
    "G_N1_2_headline",
    "G_N1_3_no_clean_tax",
    "G_N1_4_ship",
    "G_quality",
)


def load_gates() -> dict:
    return yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))


def test_gates_yaml_parses_and_has_all_gates() -> None:
    cfg = load_gates()
    assert cfg.get("version") == 1
    for g in EXPECTED_GATES:
        assert g in cfg["gates"], f"missing gate {g}"


def test_every_criterion_is_evaluable() -> None:
    cfg = load_gates()
    for gname, gate in cfg["gates"].items():
        if gate.get("live") == "quality":
            continue
        for cname, crit in gate.get("criteria", {}).items():
            assert isinstance(crit, dict), f"{gname}.{cname}: not a mapping"
            assert any(k in crit for k in BOUND_KEYS), f"{gname}.{cname}: no evaluable bound"
            if "between" not in crit and "equals" not in crit:
                assert "key" in crit or "rel_min" in crit or "rel_max" in crit, (
                    f"{gname}.{cname}: no result key"
                )


def test_result_paths_are_relative_json() -> None:
    cfg = load_gates()
    for gname, gate in cfg["gates"].items():
        for alias, rel in (gate.get("results") or {}).items():
            assert not Path(rel).is_absolute(), f"{gname}.{alias}: absolute path"
            assert rel.endswith(".json"), f"{gname}.{alias}: not a json path"


def test_frozen_flag_is_explicit_bool() -> None:
    """Freezing is a deliberate act: every gate must carry an explicit boolean
    frozen flag (numerics freeze only by measurement, committed as 'G-N1-x: freeze')."""
    cfg = load_gates()
    for gname, gate in cfg["gates"].items():
        assert isinstance(gate.get("frozen"), bool), f"{gname}: frozen flag missing/non-bool"
