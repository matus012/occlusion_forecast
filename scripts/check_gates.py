"""Gate checker: the executable interpretation of gates.yaml (ported from P1).

Exit codes: 0 = ALL gates pass (mission complete), 1 = gates pending/failing (loop continues),
2 = HARD_FAIL marker present (unrecoverable; loop stops).

Semantics (see gates.yaml header):
- frozen: false + missing results        -> PENDING (never a hard failure)
- criterion with freeze_order == "last"  -> SKIPPED entirely while the gate is unfrozen
- a gate with `live: quality` is evaluated live (pytest + ruff + coverage), not from JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTEST_NO_TESTS_COLLECTED = 5

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("check_gates")


def load_results(path_map: dict[str, str]) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for alias, rel in path_map.items():
        p = ROOT / rel
        out[alias] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


def lookup(results: dict[str, dict[str, Any] | None], dotted: str) -> float | None:
    alias, _, key = dotted.partition(".")
    blob = results.get(alias)
    if blob is None or key not in blob:
        return None
    return float(blob[key])


def resolve_source(spec: str) -> float | None:
    """Resolve 'relative/path.json:key' to a float, or None if unavailable."""
    rel, _, key = spec.rpartition(":")
    p = ROOT / rel
    if not p.exists():
        return None
    blob = json.loads(p.read_text(encoding="utf-8"))
    return float(blob[key]) if key in blob else None


def eval_criterion(
    name: str, crit: dict[str, Any], results: dict[str, dict[str, Any] | None], frozen: bool
) -> tuple[str, str]:
    """Return (status, detail) where status in PASS/FAIL/PENDING/SKIP."""
    if crit.get("freeze_order") == "last" and not frozen:
        return "SKIP", f"{name}: freeze_order=last, gate unfrozen -> not evaluated"

    if "between" in crit:  # abs delta between two result keys
        a = lookup(results, crit["between"][0])
        b = lookup(results, crit["between"][1])
        if a is None or b is None:
            return "PENDING", f"{name}: results missing"
        delta = abs(a - b)
        ok = delta <= float(crit["max"])
        detail = f"{name}: |{a:.3f}-{b:.3f}|={delta:.3f} (max {crit['max']})"
        return ("PASS" if ok else "FAIL"), detail

    val = lookup(results, crit["key"]) if "key" in crit else None

    if "equals" in crit:
        alias, _, key = crit["key"].partition(".")
        blob = results.get(alias)
        if blob is None or key not in blob:
            return "PENDING", f"{name}: results missing"
        ok = blob[key] == crit["equals"]
        return ("PASS" if ok else "FAIL"), f"{name}: {blob[key]!r} (expected {crit['equals']!r})"

    if val is None:
        return "PENDING", f"{name}: results missing"

    if "rel_min" in crit or "rel_max" in crit:
        spec = crit.get("rel_min") or crit.get("rel_max")
        base = resolve_source(spec["source"])
        if base is None:
            return "PENDING", f"{name}: relative source {spec['source']} missing"
        bound = base + float(spec["offset"])
        ok = val >= bound if "rel_min" in crit else val <= bound
        op = ">=" if "rel_min" in crit else "<="
        detail = f"{name}: {val:.3f} {op} {bound:.3f} (baseline {base:.3f})"
        return ("PASS" if ok else "FAIL"), detail

    if "min" in crit:
        ok = val >= float(crit["min"])
        return ("PASS" if ok else "FAIL"), f"{name}: {val:.3f} >= {crit['min']}"
    if "max" in crit:
        ok = val <= float(crit["max"])
        return ("PASS" if ok else "FAIL"), f"{name}: {val:.3f} <= {crit['max']}"
    return "PENDING", f"{name}: no evaluable bound"


def eval_quality_gate(crit: dict[str, Any]) -> tuple[str, list[str]]:
    """Live quality gate: pytest + coverage + ruff."""
    lines: list[str] = []
    statuses: list[str] = []

    r = subprocess.run(
        [str(VENV_PY), "-m", "ruff", "check", "."],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    ruff_ok = r.returncode == 0
    statuses.append("PASS" if ruff_ok else "FAIL")
    lines.append(f"ruff: {'clean' if ruff_ok else 'FAIL'}")
    if not ruff_ok:
        lines.append((r.stdout + r.stderr)[-2000:])

    cov_cfg = crit.get("coverage_core", {})
    cov_path = (cov_cfg.get("paths") or ["src/otraj"])[0].replace("/", ".").removeprefix("src.")
    r = subprocess.run(
        [str(VENV_PY), "-m", "pytest", "-q", f"--cov={cov_path}", "--cov-report=xml"],
        capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    if r.returncode == PYTEST_NO_TESTS_COLLECTED:
        statuses.append("PENDING")
        lines.append("pytest: no tests collected yet")
    elif r.returncode == 0:
        statuses.append("PASS")
        lines.append("pytest: pass")
    else:
        statuses.append("FAIL")
        lines.append("pytest: FAIL")
        lines.append((r.stdout + r.stderr)[-3000:])

    cov_xml = ROOT / "coverage.xml"
    if cov_xml.exists() and "coverage_core" in crit:
        rate = float(ET.parse(cov_xml).getroot().get("line-rate", "0"))
        ok = rate >= float(cov_cfg.get("min", 0.8))
        statuses.append("PASS" if ok else "FAIL")
        lines.append(f"coverage: {rate:.1%} (min {cov_cfg.get('min', 0.8):.0%})")
    else:
        statuses.append("PENDING")
        lines.append("coverage: no coverage.xml yet")

    if "FAIL" in statuses:
        return "FAIL", lines
    if "PENDING" in statuses:
        return "PENDING", lines
    return "PASS", lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=None, help="also write report to this path")
    args = ap.parse_args()

    if (ROOT / "HARD_FAIL").exists():
        msg = "HARD_FAIL marker present:\n" + (ROOT / "HARD_FAIL").read_text(encoding="utf-8")
        log.info(msg)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(msg, encoding="utf-8")
        return 2

    cfg = yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))
    report: list[str] = [f"GATE REPORT (gates.yaml v{cfg.get('version')})", ""]
    gate_status: dict[str, str] = {}

    for gname, gate in cfg["gates"].items():
        frozen = bool(gate.get("frozen", False))
        crits = gate.get("criteria", {})
        if gate.get("live") == "quality":
            status, lines = eval_quality_gate(crits)
        else:
            results = load_results(gate.get("results", {}) or {})
            statuses, lines = [], []
            for cname, crit in crits.items():
                if not isinstance(crit, dict):
                    continue
                s, detail = eval_criterion(cname, crit, results, frozen)
                statuses.append(s)
                lines.append(detail)
            active = [s for s in statuses if s != "SKIP"]
            if not active or all(s == "PENDING" for s in active):
                status = "PENDING"
            elif "FAIL" in active:
                status = "FAIL"
            elif "PENDING" in active:
                status = "PENDING"
            else:
                status = "PASS"
        gate_status[gname] = status
        frozen_tag = "frozen" if frozen else "UNFROZEN"
        report.append(f"[{status}] {gname} ({frozen_tag})")
        report.extend(f"    {ln}" for ln in lines)
        report.append("")

    all_pass = all(s == "PASS" for s in gate_status.values())
    report.append("OVERALL: " + ("ALL GATES PASS" if all_pass else "IN PROGRESS"))
    text = "\n".join(report)
    log.info(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
