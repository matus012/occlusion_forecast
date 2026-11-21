"""Mini degradation curve (D-N1-14b): the reduced-scale headline figure.

Reads the three *-local eval JSONs (scripts/eval_local.py output) and renders:

  * headline: minADE6 / minFDE6 / MR6 vs severity S0-S4, R-A, three arms
    (C1-local clean, C2-local imputation null, C3-local occlusion-aug),
    with seeded-bootstrap 95% CIs on each mean, WATERMARKED
    "reduced-scale local proof -- full matrix pending HPC".
  * per-pattern appendix: minFDE6 curves sliced M1/M2/M3 (D-N1-10),
    same watermark + cohort-size annotations.

Also writes results/local/curve_summary.json with paired per-scenario deltas
(C1-C3, C2-C3) + bootstrap CIs per severity. Single-seed DESCRIPTIVE numbers
only -- explicitly NOT the pre-registered G-N1-2 test (that requires >=3
seeds per arm, D-N1-12); the summary says so.

Figures land in reports/local_curves/ (gitignored -- visuals default-deny).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEVS = ["S0", "S1", "S2", "S3", "S4"]
WATERMARK = "reduced-scale local proof — full matrix pending HPC"

ARMS = {
    "C1-local": {"file": "eval_C1-local_native.json", "color": "#c44e52",
                 "label": "C1-local (clean-trained)"},
    "C2-local": {"file": "eval_C2-local_impute.json", "color": "#dd8452",
                 "label": "C2-local (clean + CV/linear imputation)"},
    "C3-local": {"file": "eval_C3-local_native.json", "color": "#4c72b0",
                 "label": "C3-local (occlusion-aug trained)"},
}
BOOT_N = 10_000
BOOT_SEED = 20260727


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(BOOT_N, values.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stamp(fig: plt.Figure) -> None:
    fig.text(0.5, 0.5, WATERMARK, fontsize=22, color="grey", alpha=0.18,
             ha="center", va="center", rotation=18, zorder=0)
    fig.text(0.99, 0.01, WATERMARK, fontsize=7, color="grey",
             ha="right", va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=REPO / "results" / "local")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "local_curves")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    evals: dict[str, dict] = {}
    for arm, meta in ARMS.items():
        path = args.eval_dir / meta["file"]
        evals[arm] = json.loads(path.read_text(encoding="utf-8"))

    # ---------------- headline: 3 metrics x 3 arms ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    metrics = [("minade6", "minADE$_6$ [m]"), ("minfde6", "minFDE$_6$ [m]"),
               ("mr6", "MR$_6$ (2 m)")]
    for ax, (mkey, mlabel) in zip(axes, metrics, strict=True):
        for arm, meta in ARMS.items():
            ys, lo, hi = [], [], []
            for sev in SEVS:
                per = evals[arm]["severities"][sev]["per_scenario"]
                vals = (np.array([r["miss"] for r in per], dtype=float)
                        if mkey == "mr6"
                        else np.array([r[mkey] for r in per]))
                ys.append(vals.mean())
                ci = _boot_ci(vals, rng)
                lo.append(ci[0])
                hi.append(ci[1])
            x = np.arange(len(SEVS))
            ax.plot(x, ys, "-o", color=meta["color"], label=meta["label"], ms=4)
            ax.fill_between(x, lo, hi, color=meta["color"], alpha=0.15, lw=0)
        ax.set_xticks(range(len(SEVS)), SEVS)
        ax.set_xlabel("occlusion severity (masked fraction of 50-step history)")
        ax.set_ylabel(mlabel)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")
    n = evals["C1-local"]["severities"]["S0"]["aggregate"]["n"]
    fig.suptitle(
        f"SIMPL under deterministic occlusion masks (N1-mask-v2), regime R-A, "
        f"val subset n={n}, 1 seed, 6-epoch truncated arms (D-N1-14b) — *-local",
        fontsize=11)
    fig.text(0.01, 0.01,
             "R-A severity labels understate forecast-relevant information loss "
             "(recency discount, D-N1-11d). Bands: seeded bootstrap 95% CI of the mean.",
             fontsize=7, color="dimgrey")
    _stamp(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    head_path = args.out_dir / "degradation_curve_local.png"
    fig.savefig(head_path, dpi=200)
    plt.close(fig)

    # ---------------- per-pattern appendix (D-N1-10) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, pat in zip(axes, ("M1", "M2", "M3"), strict=True):
        for arm, meta in ARMS.items():
            xs, ys = [], []
            for i, sev in enumerate(SEVS[1:], start=1):  # S0 has no pattern
                per = [r for r in evals[arm]["severities"][sev]["per_scenario"]
                       if r["pattern"] == pat]
                if per:
                    xs.append(i)
                    ys.append(float(np.mean([r["minfde6"] for r in per])))
            ax.plot(xs, ys, "-o", color=meta["color"], label=meta["label"], ms=4)
        n_pat = evals["C1-local"]["severities"]["S1"]["aggregate"].get(f"n_{pat}", 0)
        ax.set_title(f"{pat}-only cohort (n={n_pat})", fontsize=10)
        ax.set_xticks(range(len(SEVS)), SEVS)
        ax.set_xlabel("severity")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("minFDE$_6$ [m]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-pattern degradation (report-only slice, D-N1-10) — *-local",
                 fontsize=11)
    fig.text(0.01, 0.01,
             "Pattern cohorts are severity-stable by construction. M2 cohort is small "
             "(~5% mix); R-A M2 masks have degenerate placement entropy (D-N1-11d).",
             fontsize=7, color="dimgrey")
    _stamp(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    pat_path = args.out_dir / "degradation_per_pattern_local.png"
    fig.savefig(pat_path, dpi=200)
    plt.close(fig)

    # ---------------- paired-delta summary JSON ----------------
    summary: dict = {
        "note": "single-seed DESCRIPTIVE paired deltas; NOT the pre-registered "
                "G-N1-2 test (requires >=3 seeds/arm, D-N1-12)",
        "watermark": WATERMARK,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
        "figures": {"headline": str(head_path), "per_pattern": str(pat_path)},
        "deltas": {},
    }
    for ref_arm in ("C1-local", "C2-local"):
        for sev in SEVS:
            ref = {r["sid"]: r["minfde6"]
                   for r in evals[ref_arm]["severities"][sev]["per_scenario"]}
            c3 = {r["sid"]: r["minfde6"]
                  for r in evals["C3-local"]["severities"][sev]["per_scenario"]}
            sids = sorted(set(ref) & set(c3))
            d = np.array([ref[s] - c3[s] for s in sids])
            ci = _boot_ci(d, rng)
            summary["deltas"][f"{ref_arm}_minus_C3_{sev}"] = {
                "n_paired": len(sids), "mean_minfde6_delta": float(d.mean()),
                "boot95_lo": ci[0], "boot95_hi": ci[1],
            }
    out = args.eval_dir / "curve_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {head_path}\n       {pat_path}\n       {out}")


if __name__ == "__main__":
    main()
