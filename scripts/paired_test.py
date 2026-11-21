"""G-N1-2 paired significance test — the PRE-REGISTERED procedure, executed.

The test was pre-registered in D-N1-12 on 2026-07-27, before any masked training
existed, and the spec text in gates.yaml is immutable from that commit. This
script implements it and nothing else; it does not get to choose a test.

PRE-REGISTERED PROCEDURE (D-N1-12 / gates.yaml G_N1_2_headline):
  - per-scenario value = MEAN over >=3 training seeds per arm, taken BEFORE pairing
  - pair by scenario id; delta = minFDE6(C1) - minFDE6(C3), so delta > 0 == C3 better
  - one-sided Wilcoxon signed-rank, H1: C3 < C1  (alternative="greater" on delta)
  - regime R-A, MIXED pattern condition, identical deterministic N1-mask-v2 masks
  - severities S2, S3, S4; alpha 0.05, Holm-Bonferroni across the three
  - gate key paired_p_s2 = HOLM-ADJUSTED p at S2
  - direction key c3_beats_c1_s2 = seed-mean minFDE6(C3) < minFDE6(C1) at ALL of
    S2, S3, S4
  - effect size = seeded 10k-resample bootstrap 95% CI of the mean delta
  - per-pattern cohort tests (D-N1-10): same procedure, REPORT-ONLY, never
    gate-blocking (degenerate-entropy caveats, D-N1-11d)

Input is one eval JSON per (arm, seed) in the scripts/eval_local.py output format.
Output is results/g_n1_2_headline.json, whose top-level keys are what
scripts/check_gates.py reads.

The script reports what it measured and lets check_gates.py decide pass/fail --
including the seed count. Running with <3 seeds produces an honest
n_seeds_per_arm and the gate fails on that criterion; the script does not
silently pass, and it does not refuse to compute.

Usage:
  python scripts/paired_test.py \
      --c1 results/matrix/eval_C1_seed{0,1,2}_native.json \
      --c3 results/matrix/eval_C3_seed{0,1,2}_native.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]

SEVERITIES: tuple[str, ...] = ("S2", "S3", "S4")   # pre-registered, do not widen
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
PATTERNS: tuple[str, ...] = ("M1", "M2", "M3")
GATE_SEVERITY = "S2"                                # paired_p_s2 / c3_beats_c1_s2


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_arm(paths: list[Path], label: str) -> dict[str, Any]:
    """One eval JSON per seed -> per-scenario seed-MEAN minFDE6 per severity.

    Averaging over seeds happens BEFORE pairing, per the pre-registration: the
    unit of analysis is the scenario, not the (scenario, seed) pair.
    """
    if not paths:
        raise SystemExit(f"[{label}] no eval JSONs given")
    per_seed, meta = [], []
    for p in paths:
        blob = json.loads(p.read_text(encoding="utf-8"))
        meta.append({"path": str(p), "label": blob.get("label"),
                     "mode": blob.get("mode"), "regime": blob.get("regime"),
                     "mask_spec": blob.get("mask_spec"),
                     "ckpt": blob.get("ckpt"), "ckpt_epoch": blob.get("ckpt_epoch")})
        rows: dict[str, dict[str, float]] = {}
        patterns: dict[str, str] = {}
        for sev in SEVERITIES:
            if sev not in blob.get("severities", {}):
                raise SystemExit(f"[{label}] {p.name}: severity {sev} missing")
            rows[sev] = {}
            for r in blob["severities"][sev]["per_scenario"]:
                rows[sev][r["sid"]] = float(r["minfde6"])
                patterns[r["sid"]] = r.get("pattern", "none")
        per_seed.append({"rows": rows, "patterns": patterns})

    # identical masks across arms/seeds is what makes the pairing legitimate
    specs = {m["mask_spec"] for m in meta}
    if len(specs) != 1:
        raise SystemExit(f"[{label}] mixed mask specs across seeds: {specs}")
    regimes = {m["regime"] for m in meta}
    if len(regimes) != 1:
        raise SystemExit(f"[{label}] mixed regimes across seeds: {regimes}")

    sids = set(per_seed[0]["rows"][SEVERITIES[0]])
    for s in per_seed:
        for sev in SEVERITIES:
            if set(s["rows"][sev]) != sids:
                raise SystemExit(f"[{label}] scenario sets differ across seeds/"
                                 f"severities — pairing would be invalid")

    mean = {sev: {sid: float(np.mean([s["rows"][sev][sid] for s in per_seed]))
                  for sid in sids}
            for sev in SEVERITIES}
    return {"mean": mean, "n_seeds": len(paths), "patterns": per_seed[0]["patterns"],
            "sids": sids, "meta": meta, "mask_spec": specs.pop(),
            "regime": regimes.pop()}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Step-down Holm adjustment, with the monotonicity enforced.

    Adjusted p_(i) = max over j<=i of (m - j) * p_(j), clipped at 1.0 -- the
    running max is what makes the adjusted sequence non-decreasing.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for j, (key, p) in enumerate(items):
        running = max(running, (m - j) * p)
        out[key] = float(min(running, 1.0))
    return out


def bootstrap_ci(delta: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean paired delta.

    Seeded so the reported interval is reproducible from the artifact alone.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_wilcoxon(delta: np.ndarray) -> tuple[float, float | None, str]:
    """One-sided Wilcoxon signed-rank, H1: delta > 0 (C3 beats C1).

    An all-zero delta vector is not a pathological input here -- it is exactly
    what a COLLAPSED arm produces (D-N1-14c: C3-local was bit-identical across
    S0-S4). scipy raises on it, so it is handled explicitly and reported as
    p = 1.0 rather than crashing the gate runner or, worse, being silently
    dropped.
    """
    if len(delta) == 0:
        return 1.0, None, "empty"
    if np.allclose(delta, 0.0):
        return 1.0, None, "all-zero deltas (arms identical — check for collapse)"
    res = wilcoxon(delta, alternative="greater", zero_method="wilcox")
    return float(res.pvalue), float(res.statistic), "ok"


def severity_record(c1: dict, c3: dict, sev: str, sids: list[str]) -> dict[str, Any]:
    a = np.array([c1["mean"][sev][s] for s in sids])
    b = np.array([c3["mean"][sev][s] for s in sids])
    delta = a - b                       # > 0 == C3 better
    p, stat, note = paired_wilcoxon(delta)
    lo, hi = bootstrap_ci(delta) if len(delta) else (float("nan"), float("nan"))
    return {
        "n_scenarios": len(sids),
        "c1_mean_minfde6": float(a.mean()), "c3_mean_minfde6": float(b.mean()),
        "mean_delta_m": float(delta.mean()),
        "median_delta_m": float(np.median(delta)),
        "frac_scenarios_c3_better": float((delta > 0).mean()),
        "wilcoxon_p_raw": p, "wilcoxon_statistic": stat, "note": note,
        "bootstrap_ci95_mean_delta_m": [lo, hi],
        "c3_beats_c1": bool(b.mean() < a.mean()),
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=Path, nargs="+", required=True,
                    help="one eval JSON per seed for the C1 arm")
    ap.add_argument("--c3", type=Path, nargs="+", required=True,
                    help="one eval JSON per seed for the C3 arm")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "g_n1_2_headline.json")
    args = ap.parse_args()

    c1 = load_arm(args.c1, "C1")
    c3 = load_arm(args.c3, "C3")
    if c1["sids"] != c3["sids"]:
        raise SystemExit("C1 and C3 evaluated different scenario sets — "
                         "the paired test is undefined")
    if c1["mask_spec"] != c3["mask_spec"]:
        raise SystemExit(f"arms used different mask specs: "
                         f"{c1['mask_spec']} vs {c3['mask_spec']}")
    if c1["n_seeds"] != c3["n_seeds"]:
        print(f"[warn] unequal seed counts: C1={c1['n_seeds']} C3={c3['n_seeds']}; "
              f"n_seeds_per_arm reports the MINIMUM")

    sids = sorted(c1["sids"])
    per_sev = {sev: severity_record(c1, c3, sev, sids) for sev in SEVERITIES}
    holm = holm_bonferroni({s: per_sev[s]["wilcoxon_p_raw"] for s in SEVERITIES})
    for sev in SEVERITIES:
        per_sev[sev]["wilcoxon_p_holm"] = holm[sev]
        per_sev[sev]["significant_at_alpha"] = bool(holm[sev] <= ALPHA)

    # per-pattern cohorts: REPORT-ONLY, never gate-blocking (D-N1-10 / D-N1-11d)
    per_pattern: dict[str, dict[str, Any]] = {}
    for sev in SEVERITIES:
        per_pattern[sev] = {}
        for pat in PATTERNS:
            cohort = [s for s in sids if c1["patterns"].get(s) == pat]
            if len(cohort) < 2:
                continue
            per_pattern[sev][pat] = severity_record(c1, c3, sev, cohort)

    n_seeds = min(c1["n_seeds"], c3["n_seeds"])
    out: dict[str, Any] = {
        # --- keys read by scripts/check_gates.py -----------------------------
        "n_seeds_per_arm": n_seeds,
        "paired_p_s2": per_sev[GATE_SEVERITY]["wilcoxon_p_holm"],
        "c3_beats_c1_s2": all(per_sev[s]["c3_beats_c1"] for s in SEVERITIES),
        # --- provenance -------------------------------------------------------
        "spec": "D-N1-12 pre-registration (2026-07-27), immutable in gates.yaml",
        "test": "one-sided Wilcoxon signed-rank on per-scenario paired minFDE6 "
                "deltas (C1 - C3), H1: C3 < C1",
        "seed_aggregation": "per-scenario MEAN over seeds, taken before pairing",
        "severities_tested": list(SEVERITIES),
        "alpha": ALPHA,
        "multiplicity_correction": "Holm-Bonferroni across the 3 severities",
        "gate_key_definition": {
            "paired_p_s2": f"Holm-adjusted p at {GATE_SEVERITY}",
            "c3_beats_c1_s2": "seed-mean minFDE6(C3) < minFDE6(C1) at ALL of "
                              + ", ".join(SEVERITIES),
        },
        "effect_size": f"seeded percentile bootstrap, {BOOTSTRAP_RESAMPLES} "
                       f"resamples (seed {BOOTSTRAP_SEED}), 95% CI of mean delta",
        "metric_authority": "av2.datasets.motion_forecasting.eval.metrics "
                            "(inherited from eval_local.py)",
        "mask_spec": c1["mask_spec"], "regime": c1["regime"],
        "n_scenarios": len(sids),
        "inputs": {"c1": c1["meta"], "c3": c3["meta"]},
        # --- results -----------------------------------------------------------
        "severities": per_sev,
        "per_pattern_report_only": per_pattern,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"G-N1-2 paired test · {len(sids)} scenarios · {n_seeds} seeds/arm · "
          f"{c1['mask_spec']} · {c1['regime']}")
    for sev in SEVERITIES:
        r = per_sev[sev]
        lo, hi = r["bootstrap_ci95_mean_delta_m"]
        print(f"  {sev}: C1 {r['c1_mean_minfde6']:.4f} vs C3 {r['c3_mean_minfde6']:.4f} "
              f"| delta {r['mean_delta_m']:+.4f} m [{lo:+.4f}, {hi:+.4f}] "
              f"| p_raw {r['wilcoxon_p_raw']:.4g} -> p_holm {r['wilcoxon_p_holm']:.4g}"
              f"{'  *' if r['significant_at_alpha'] else ''}"
              + (f"  [{r['note']}]" if r["note"] != "ok" else ""))
    print(f"  gate keys: n_seeds_per_arm={n_seeds} "
          f"paired_p_s2={out['paired_p_s2']:.4g} "
          f"c3_beats_c1_s2={out['c3_beats_c1_s2']}")
    if n_seeds < 3:
        print(f"[warn] {n_seeds} seed(s)/arm — G-N1-2 requires >=3; "
              f"check_gates will FAIL n_seeds_per_arm (reported honestly, not padded)")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
