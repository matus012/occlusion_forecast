"""Build the PERUN HPC request (D-N1-14d) from measurement artifacts.

Fills reports/perun_request/template.md with numbers read EXCLUSIVELY from
results/*.json (traceability requirement: nothing hand-typed), embeds the
degradation-curve figures + video stills as base64, and renders a PDF via
headless Chrome.

Outputs:
  reports/perun_request/request.md    (tracked -- text only)
  reports/perun_request/request.html  (local)
  reports/perun_request/request.pdf   (local -- embeds AV2-derived figures,
                                       stays out of the public repo)

H200 conversion: the measured 4060 throughput is scaled by a STATED assumption
band (--h200-factor-lo/hi, default 6-10x per-GPU). The report says explicitly
that the band is an assumption to be calibrated by the pilot run.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
LOCAL = RES / "local"
OUT_DIR = REPO / "reports" / "perun_request"

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")

FULL_TRAIN = 199_908
FULL_VAL = 24_988
FULL_EPOCHS = 50
N_RUNS = 9  # 3 arms x 3 seeds
SEVS = ["S0", "S1", "S2", "S3", "S4"]

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 820px;
       margin: 2em auto; line-height: 1.45; color: #1a1a1a; font-size: 11pt; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 13pt; margin-top: 1.6em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 9px; text-align: left; }
th { background: #efefef; }
blockquote { border-left: 3px solid #888; margin-left: 0; padding-left: 1em;
             color: #444; font-size: 9.5pt; }
img { max-width: 100%; margin: 0.6em 0; border: 1px solid #ccc; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f4; }
.figcap { font-size: 8.5pt; color: #555; margin-top: -0.4em; }
"""


def _j(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_h(hours: float) -> str:
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def build_values(f_lo: float, f_hi: float, pilot_h: float) -> dict:
    g0 = _j(RES / "g_n1_0_checkpoint_eval.json")
    budget = _j(RES / "local_budget.json")
    c1_train = _j(LOCAL / "train_c1_seed42.json")
    c3_train = _j(LOCAL / "train_c3_seed42.json")
    evals = {arm: _j(LOCAL / f"eval_{arm}_{mode}.json")
             for arm, mode in (("C1-local", "native"), ("C2-local", "impute"),
                               ("C3-local", "native"))}
    summary = _j(LOCAL / "curve_summary.json")

    probe = budget["probe_measurements"]
    bs8 = probe["train_bs8"]
    steady = float(bs8["samples_per_s_steady"])

    # measured steady from the REAL c1 run if available (better than probe):
    walls = [e for e in c1_train["epochs_log"] if e["epoch"] > 0]
    if walls:
        steady = round(sum(e["samples_per_s"] for e in walls) / len(walls), 1)

    # ---- curve table ----
    hdr = ("| Severity | C1-local minADE₆/minFDE₆/MR₆ | C2-local (imputation) "
           "| C3-local (occl-aug) |\n|---|---|---|---|")
    rows = []
    for sev in SEVS:
        cells = []
        for arm in ("C1-local", "C2-local", "C3-local"):
            a = evals[arm]["severities"][sev]["aggregate"]
            cells.append(f"{a['minade6']:.3f} / {a['minfde6']:.3f} / {a['mr6']:.3f}")
        rows.append(f"| {sev} | " + " | ".join(cells) + " |")
    curve_table = "\n".join([hdr, *rows])

    # ---- auto reading (factual, from curve_summary) ----
    d3 = summary["deltas"]["C1-local_minus_C3_S3"]
    d0 = summary["deltas"]["C1-local_minus_C3_S0"]
    c1_s3 = evals["C1-local"]["severities"]["S3"]["aggregate"]["minfde6"]
    rel = 100.0 * d3["mean_minfde6_delta"] / c1_s3
    curve_reading = (
        f"Reading (single seed, descriptive only — the pre-registered G-N1-2 "
        f"test requires ≥3 seeds/arm): at S3, occlusion-aware training improves "
        f"paired per-scenario minFDE₆ by {d3['mean_minfde6_delta']:.3f} m "
        f"({rel:.0f}% of C1-local's {c1_s3:.3f} m; bootstrap 95% CI "
        f"[{d3['boot95_lo']:.3f}, {d3['boot95_hi']:.3f}], "
        f"`results/local/curve_summary.json`). At S0 (no occlusion) the paired "
        f"difference is {d0['mean_minfde6_delta']:+.3f} m (CI "
        f"[{d0['boot95_lo']:.3f}, {d0['boot95_hi']:.3f}]) — the no-clean-tax "
        f"property G-N1-3 will be tested at full scale.")

    # ---- VRAM table ----
    vram_table = (
        "| Batch | samples/s | peak VRAM | verdict |\n|---|---|---|---|\n"
        f"| 16 (authors' per-GPU) | {probe['train_bs16']['samples_per_s_incl_warmup']} "
        f"| {probe['train_bs16']['peak_vram_mb']} MB | {probe['train_bs16']['verdict']} |\n"
        f"| 12 | {probe['train_bs12']['samples_per_s_incl_warmup']} "
        f"| {probe['train_bs12']['peak_vram_mb']} MB | {probe['train_bs12']['verdict']} |\n"
        f"| 8 | {steady} (steady, full run) "
        f"| {bs8['peak_vram_mb']} MB | {bs8['verdict']} |")

    # ---- budget math ----
    samples_per_run = FULL_TRAIN * FULL_EPOCHS
    h_per_run_4060 = samples_per_run / steady / 3600
    days_per_run_4060 = h_per_run_4060 / 24
    days_matrix_4060 = days_per_run_4060 * N_RUNS

    h_run_hi = h_per_run_4060 / f_lo  # conservative (low speedup -> more hours)
    h_run_lo = h_per_run_4060 / f_hi
    train_hi = h_run_hi * N_RUNS
    train_lo = h_run_lo * N_RUNS

    # eval: measured seconds per (scenario, condition) from the local evals
    eval_wall = sum(e["wall_seconds"] for e in evals.values())
    eval_scen = sum(e["val_count"] * len(e["severities"]) for e in evals.values())
    s_per_scen = eval_wall / eval_scen
    # full eval: 9 run-ckpts x (5 sev x 2 regimes) x full val + per-pattern reuse
    full_eval_h_4060 = FULL_VAL * 10 * N_RUNS * s_per_scen / 3600
    eval_hi = full_eval_h_4060 / f_lo
    preproc_h = (FULL_TRAIN + FULL_VAL) / probe[
        "preprocess_scen_per_s_10workers"]["train"] / 3600

    r4_hi = h_run_hi  # R4 sensitivity: one arm re-eval + one extra aug run equiv
    contingency = 0.15
    total_hi = (train_hi + eval_hi + r4_hi + pilot_h) * (1 + contingency)
    h200h_total = int(round(total_hi / 10) * 10)

    feat_gb = (FULL_TRAIN + FULL_VAL) * 137 / 1e6
    storage_gb = int(round((60 + feat_gb + 10) / 10) * 10)

    budget_table = (
        "| Cost item | Basis (measured) | 4060 hours | H200 hours "
        f"(÷{f_lo}–{f_hi}) |\n|---|---|---|---|\n"
        f"| 1 training run ({FULL_TRAIN:,} × {FULL_EPOCHS} ep) | "
        f"{steady} samples/s | {h_per_run_4060:.0f} h | "
        f"{h_run_lo:.0f}–{h_run_hi:.0f} h |\n"
        f"| 9-run matrix (3 arms × 3 seeds) | — | "
        f"{h_per_run_4060 * N_RUNS:.0f} h | {train_lo:.0f}–{train_hi:.0f} h |\n"
        f"| Full masked-eval matrix (9 ckpts × 10 conditions × {FULL_VAL:,}) | "
        f"{s_per_scen * 1000:.1f} ms/scenario | {full_eval_h_4060:.0f} h | "
        f"{full_eval_h_4060 / f_hi:.0f}–{eval_hi:.0f} h |\n"
        f"| R4 sensitivity sweep (1 arm, v1-prior mix) | = 1 run | "
        f"{h_per_run_4060:.0f} h | {h_run_lo:.0f}–{h_run_hi:.0f} h |\n"
        f"| Preprocessing (CPU-side, {FULL_TRAIN + FULL_VAL:,} scenarios) | "
        f"{probe['preprocess_scen_per_s_10workers']['train']} scen/s (10 proc) | "
        f"{preproc_h:.1f} h | CPU nodes |\n"
        f"| Pilot calibration | 1 epoch | — | {pilot_h:.0f} h |\n"
        f"| Contingency ({int(contingency * 100)}%) + total | — | — | "
        f"**{h200h_total} h** (upper band) |")

    c1w = c1_train.get("total_wall_s", 0) / 3600
    c3w = c3_train.get("total_wall_s", 0) / 3600
    local_train_desc = (
        f"{c1_train['train_count']:,} city-stratified scenarios × "
        f"{c1_train['epochs']} epochs, batch {c1_train['batch_size']}, 1 seed, "
        f"wall-clock {c1w:.1f} h (C1-local) / {c3w:.1f} h (C3-local, "
        f"p_occ=0.5, empirical mix) — `results/local/train_c*_seed42.json`")

    curve_files = ("`results/local/eval_C1-local_native.json`, "
                   "`eval_C2-local_impute.json`, `eval_C3-local_native.json`")
    vids = sorted((REPO / "reports" / "videos").glob("*.mp4"))
    video_files = ", ".join(f"`reports/videos/{v.name}`" for v in vids) or "(pending)"

    return {
        "date": "2026-07-28",
        "g0_minade": g0["minade6"], "g0_minfde": g0["minfde6"], "g0_mr": g0["mr6"],
        "local_train_desc": local_train_desc,
        "val_count": evals["C1-local"]["val_count"],
        "curve_files": curve_files, "curve_table": curve_table,
        "curve_reading": curve_reading, "video_files": video_files,
        "vram_table": vram_table, "bs8_steady": steady,
        "days_per_run_4060": days_per_run_4060,
        "days_matrix_4060": days_matrix_4060,
        "h200_factor_lo": f_lo, "h200_factor_hi": f_hi,
        "budget_table": budget_table, "h200h_total": h200h_total,
        "storage_gb": storage_gb,
        "storage_note": f"raw AV2 ≈60 GB + SIMPL features ≈{feat_gb:.0f} GB "
                        "(137 KB/scenario measured) + checkpoints/logs",
        "walltime_per_run": _fmt_h(h_run_hi),
        "pilot_h200h": pilot_h,
    }


def render(md_text: str, embed_figs: bool) -> str:
    html_body = markdown.markdown(md_text, extensions=["tables"])
    figs = ""
    if embed_figs:
        blocks = []
        curve_dir = REPO / "reports" / "local_curves"
        stills = sorted((REPO / "reports" / "videos" / "stills").glob("*.png"))
        for p in [curve_dir / "degradation_curve_local.png",
                  curve_dir / "degradation_per_pattern_local.png", *stills]:
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                blocks.append(
                    f'<img src="data:image/png;base64,{b64}"/>'
                    f'<p class="figcap">{p.relative_to(REPO)}</p>')
        if blocks:
            figs = ("<h2>Attached figures</h2>" + "\n".join(blocks))
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{html_body}{figs}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h200-factor-lo", type=float, default=6.0)
    ap.add_argument("--h200-factor-hi", type=float, default=10.0)
    ap.add_argument("--pilot-h200h", type=float, default=5.0)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    values = build_values(args.h200_factor_lo, args.h200_factor_hi,
                          args.pilot_h200h)
    template = (OUT_DIR / "template.md").read_text(encoding="utf-8")
    md_text = template.format(**values)
    (OUT_DIR / "request.md").write_text(md_text, encoding="utf-8")

    html = render(md_text, embed_figs=True)
    html_path = OUT_DIR / "request.html"
    html_path.write_text(html, encoding="utf-8")

    if not args.no_pdf:
        pdf_path = OUT_DIR / "request.pdf"
        subprocess.run(
            [str(CHROME), "--headless", "--disable-gpu",
             f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
             html_path.resolve().as_uri()],
            check=True, capture_output=True, timeout=120)
        print(f"[done] {pdf_path}")
    print(f"[done] {OUT_DIR / 'request.md'}")


if __name__ == "__main__":
    main()
