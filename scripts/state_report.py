"""N1 state-of-project visual report generator (user-directed, session-local).

Produces committable code only: this script. Its OUTPUTS are local and
gitignored (``reports/`` is default-denied in .gitignore) -- every figure is
generated fresh from real repo artifacts (checkpoints, AV2 val data, P2 stats,
gate results), never hand-entered numbers.

Section -> required venv:
  scenario           .venv-qcnet\\Scripts\\python.exe   (torch + PyG + av2 + QCNet)
  p2, mask, dash,    .venv\\Scripts\\python.exe          (matplotlib/numpy/scipy/
  assemble                                              pandas/pyarrow/yaml only,
                                                         NO torch import)

Usage (from repo root):
  .venv-qcnet\\Scripts\\python.exe scripts\\state_report.py --sections scenario
  .venv\\Scripts\\python.exe scripts\\state_report.py --sections p2,mask,dash,assemble

Outputs (reports\\state_2026wk1\\, LOCAL only, never committed):
  scenario_qcnet.png, p2_duration_fit.png, p2_pattern_mix.png,
  p2_visibility_timelines.png, mask_spec_mockup.png, dashboard.png, index.html
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "state_2026wk1"
P2_SCENARIOS_ROOT = ROOT.parent / "100_occlusion_mot" / "data" / "sim" / "scenarios"
NUM_HISTORICAL_STEPS = 50  # AV2 motion-forecasting convention (context.md "Data & protocol")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("state_report")


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _first_val_scenario_id() -> str:
    """Deterministic scenario choice: sorted(scenario dirs under data/av2/val)[0]."""
    val_dir = ROOT / "data" / "av2" / "val"
    dirs = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    if not dirs:
        raise RuntimeError(f"no scenario directories under {val_dir}")
    return dirs[0]


# ---------------------------------------------------------------------------
# SECTION: scenario  (.venv-qcnet only -- torch/PyG/av2)
# ---------------------------------------------------------------------------


def run_scenario(args: argparse.Namespace) -> None:
    """Proof the eval pipeline produces sane predictions end-to-end (QCNet, world frame)."""
    _ensure_out_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Batch

    sys.path.insert(0, str(ROOT / "third_party" / "QCNet"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))

    import eval_checkpoint_g_n1_0 as gmod  # TargetBuilderCompat / FOCAL_CATEGORY, not duplicated
    from av2.datasets.motion_forecasting.eval.metrics import compute_ade, compute_fde
    from av2.map.map_api import ArgoverseStaticMap
    from datasets import ArgoverseV2Dataset
    from predictors import QCNet

    from otraj.utils.seeding import resolve_device, set_seed

    set_seed(args.seed)
    scenario_id = _first_val_scenario_id()
    log.info("scenario_id (deterministic: sorted val dirs[0]) = %s", scenario_id)

    val_dir = ROOT / "data" / "av2" / "val" / scenario_id
    parquet_path = val_dir / f"scenario_{scenario_id}.parquet"
    map_path = val_dir / f"log_map_archive_{scenario_id}.json"
    ckpt_path = ROOT / "checkpoints" / "qcnet_av2.ckpt"

    device = resolve_device(args.device)
    log.info("device=%s", device)
    model = QCNet.load_from_checkpoint(checkpoint_path=str(ckpt_path), map_location=device)
    model.eval().to(device)

    # ABSOLUTE root required (D-N1-8: upstream path-join bug with relative roots).
    qcnet_root = str((ROOT / "data" / "qcnet_root").resolve())
    dataset = ArgoverseV2Dataset(
        root=qcnet_root,
        split="val",
        transform=gmod.TargetBuilderCompat(model.num_historical_steps, model.num_future_steps),
    )

    target_name = f"{scenario_id}.pkl"
    processed_names = dataset.processed_file_names
    if target_name not in processed_names:
        raise RuntimeError(
            f"{target_name} not found among {len(processed_names)} processed files "
            f"under {qcnet_root}/val/processed -- mapping assumption broken"
        )
    idx = processed_names.index(target_name)  # verified mapping, ordering never assumed
    data = dataset[idx]
    if str(data["scenario_id"]) != scenario_id:
        raise RuntimeError(
            f"dataset index mapping mismatch: idx {idx} -> scenario_id "
            f"{data['scenario_id']!r}, expected {scenario_id!r}"
        )

    batch = Batch.from_data_list([data]).to(device)
    batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]

    with torch.no_grad():
        pred = model(batch)
    traj_refine = pred["loc_refine_pos"][..., : model.output_dim]  # [N, K, T, 2], agent-centric

    category = batch["agent"]["category"]
    focal_matches = (category == gmod.FOCAL_CATEGORY).nonzero(as_tuple=True)[0]
    if focal_matches.numel() != 1:
        raise RuntimeError(f"expected exactly 1 focal agent, found {focal_matches.numel()}")
    focal_idx = int(focal_matches[0])

    nh = model.num_historical_steps
    origin = batch["agent"]["position"][focal_idx, nh - 1, :2].cpu().numpy()
    theta = float(batch["agent"]["heading"][focal_idx, nh - 1].cpu().numpy())
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # TargetBuilder convention (third_party/QCNet/transforms/target_builder.py):
    # target = (world - origin) @ rot_mat, rot_mat = [[cos, -sin], [sin, cos]].
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # MANDATORY SELF-CHECK: invert the world->agent transform and verify it reconstructs
    # the focal agent's own recorded world future. Crash loudly if this is ever wrong.
    target_xy = batch["agent"]["target"][focal_idx][:, :2].cpu().numpy()
    world_recon = origin + target_xy @ rot_mat.T
    world_gt_future = batch["agent"]["position"][focal_idx, nh:, :2].cpu().numpy()
    predict_mask = batch["agent"]["predict_mask"][focal_idx, nh:].cpu().numpy().astype(bool)
    if not predict_mask.any():
        raise RuntimeError("focal agent has no valid future timesteps to self-check against")
    err = np.linalg.norm(world_recon[predict_mask] - world_gt_future[predict_mask], axis=-1)
    max_err = float(err.max())
    log.info(
        "SELF-CHECK frame-inversion max abs error over %d valid steps: %.6e m",
        int(predict_mask.sum()),
        max_err,
    )
    if max_err >= 0.01:
        raise AssertionError(
            f"frame self-check FAILED: max abs error {max_err:.6e} m >= 0.01 m tolerance "
            f"(origin={origin}, theta={theta}) -- inverse transform is wrong, crashing loudly"
        )

    # Apply the SAME validated inverse to the 6 predicted trajectories.
    preds_agent = traj_refine[focal_idx].detach().cpu().numpy()  # [K, T, 2]
    preds_world = origin + preds_agent @ rot_mat.T

    gt_agent = batch["agent"]["target"][focal_idx][:, :2].cpu().numpy()
    fde_per_mode = compute_fde(preds_agent, gt_agent)
    compute_ade(preds_agent, gt_agent)  # cross-check metric, not plotted
    best_mode = int(np.argmin(fde_per_mode))
    best_minfde = float(fde_per_mode[best_mode])
    log.info(
        "per-mode FDE (official av2 kit; agent frame == world frame under rigid transform): %s",
        np.round(fde_per_mode, 4).tolist(),
    )
    log.info("scenario %s best-mode minFDE = %.4f m", scenario_id, best_minfde)

    # ---- world-frame plot ----
    map_api = ArgoverseStaticMap.from_json(map_path)
    fig, ax = plt.subplots(figsize=(11, 11))
    for ls in map_api.get_scenario_lane_segments():
        cl = map_api.get_lane_segment_centerline(ls.id)
        ax.plot(cl[:, 0], cl[:, 1], color="lightgray", linewidth=0.8, zorder=1)

    df = pd.read_parquet(parquet_path)
    focal_track_id = str(df["focal_track_id"].iloc[0])
    hist_df = df[df["timestep"] < model.num_historical_steps]
    for track_id, g in hist_df.groupby("track_id"):
        if str(track_id) == focal_track_id:
            continue
        g = g.sort_values("timestep")
        ax.plot(g["position_x"], g["position_y"], color="royalblue", alpha=0.15,
                 linewidth=0.8, zorder=2)

    focal_hist = hist_df[hist_df["track_id"].astype(str) == focal_track_id].sort_values("timestep")
    ax.plot(focal_hist["position_x"], focal_hist["position_y"], color="royalblue",
             linewidth=2.5, zorder=4, label="focal observed history")

    focal_fut = df[
        (df["track_id"].astype(str) == focal_track_id)
        & (df["timestep"] >= model.num_historical_steps)
    ].sort_values("timestep")
    ax.plot(focal_fut["position_x"], focal_fut["position_y"], color="green", linestyle="--",
             linewidth=2.0, zorder=4, label="focal GT future")

    for k in range(preds_world.shape[0]):
        ax.plot(preds_world[k, :, 0], preds_world[k, :, 1], color="darkorange", alpha=0.7,
                 linewidth=1.8, zorder=5)
        ax.scatter(preds_world[k, -1, 0], preds_world[k, -1, 1], color="darkorange", alpha=0.9,
                   s=35, zorder=6, marker="o", edgecolors="black", linewidths=0.5)
    ax.plot([], [], color="darkorange", alpha=0.7, linewidth=1.8, label="QCNet 6 predicted modes")

    # Crop to the focal agent's own neighborhood (history + GT future + all 6
    # predicted modes, + margin) -- the full map/all-agent extent dwarfs the
    # focal trajectory and makes the plot unreadable otherwise.
    focal_x = np.concatenate([
        focal_hist["position_x"].to_numpy(), focal_fut["position_x"].to_numpy(),
        preds_world[..., 0].ravel(),
    ])
    focal_y = np.concatenate([
        focal_hist["position_y"].to_numpy(), focal_fut["position_y"].to_numpy(),
        preds_world[..., 1].ravel(),
    ])
    margin = 30.0
    ax.set_xlim(focal_x.min() - margin, focal_x.max() + margin)
    ax.set_ylim(focal_y.min() - margin, focal_y.max() + margin)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"scenario {scenario_id}\nQCNet best-mode minFDE = {best_minfde:.3f} m "
        f"(official av2 kit; frame self-check max err {max_err:.2e} m)",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "scenario_qcnet.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# SECTION: p2  (.venv only)
# ---------------------------------------------------------------------------


def run_p2(args: argparse.Namespace) -> None:  # args unused (uniform dispatch signature)
    _ensure_out_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as scipy_stats

    sys.path.insert(0, str(ROOT / "src"))
    from otraj.masking.p2_stats import (
        _iter_scenario_dirs,
        build_p2_stats,
        classify_agent_pattern,
        extract_agent_runs,
        load_scenario_visibility,
    )

    ref_path = ROOT / "results" / "p2_occlusion_stats.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_fit = ref["duration_fit_lognormal"]

    log.info("recomputing P2 stats live over %s", P2_SCENARIOS_ROOT)
    result = build_p2_stats(P2_SCENARIOS_ROOT)
    live_fit = result.duration_fit.to_dict()

    for key in ("shape", "loc", "scale"):
        delta = abs(live_fit[key] - ref_fit[key])
        if delta >= 1e-9:
            raise AssertionError(
                f"P2 duration fit refit mismatch on {key}: live={live_fit[key]!r} "
                f"json={ref_fit[key]!r} delta={delta:.3e} (must be < 1e-9)"
            )
    log.info("live refit matches results/p2_occlusion_stats.json to 1e-9: %s", live_fit)

    durations_s = np.array([r.duration_s for r in result.runs], dtype=np.float64)

    # --- p2_duration_fit.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durations_s, bins=30, density=True, color="steelblue", alpha=0.7,
            label=f"real P2 segment durations (n={len(durations_s)})")
    xs = np.linspace(1e-3, durations_s.max() * 1.05, 500)
    pdf = scipy_stats.lognorm.pdf(xs, ref_fit["shape"], loc=ref_fit["loc"], scale=ref_fit["scale"])
    ax.plot(xs, pdf, color="firebrick", linewidth=2,
            label=f"lognormal fit (shape={ref_fit['shape']:.3f}, scale={ref_fit['scale']:.3f})")
    mot17_s = ref["mot17_sanity_anchor"]["median_gap_s"]
    ax.axvline(mot17_s, color="black", linestyle=":", linewidth=1.5,
               label=f"MOT17 anchor ~{mot17_s:.1f}s (sanity, not fit against)")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("density")
    ax.set_title("P2 CARLA occlusion segment duration: real histogram vs fitted lognormal")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p1 = OUT_DIR / "p2_duration_fit.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", p1)

    # --- p2_pattern_mix.png ---
    v1_prior = ref["pattern_mix"]["v1_prior"]
    empirical = ref["pattern_mix"]["fractions_of_M1_M2_M3_only"]
    patterns = ["M1", "M2", "M3"]
    x = np.arange(len(patterns))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [v1_prior[p] for p in patterns], width, label="v1 prior",
           color="lightgray", edgecolor="black")
    ax.bar(x + width / 2, [empirical[p] for p in patterns], width,
           label="empirical (P2, M1/M2/M3-only)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.set_ylabel("fraction")
    ax.set_title("Occlusion pattern mix: v1 prior vs P2 empirical (D-N1-9)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = OUT_DIR / "p2_pattern_mix.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", p2)

    # --- p2_visibility_timelines.png ---
    dirs = _iter_scenario_dirs(P2_SCENARIOS_ROOT)
    first_m1 = first_m3_3plus = first_m2 = first_trailing = None
    for sd in dirs:
        traces = load_scenario_visibility(sd)
        for agent_id in sorted(traces):
            trace = traces[agent_id]
            runs, is_artifact = extract_agent_runs(trace)
            if is_artifact:
                continue
            pattern = classify_agent_pattern(runs)
            if pattern == "M1" and first_m1 is None:
                first_m1 = (sd.name, agent_id, runs, trace)
            elif pattern == "M3" and len(runs) >= 3 and first_m3_3plus is None:
                first_m3_3plus = (sd.name, agent_id, runs, trace)
            elif pattern == "M2" and first_m2 is None:
                first_m2 = (sd.name, agent_id, runs, trace)
            elif pattern == "other_trailing" and first_trailing is None:
                first_trailing = (sd.name, agent_id, runs, trace)

    third = first_m2 if first_m2 is not None else first_trailing
    panels = [
        ("M1 (single contiguous block)", first_m1),
        ("M3 (flicker, >=3 runs)", first_m3_3plus),
        ("M2 (prefix)" if first_m2 is not None else "other_trailing (never recovers)", third),
    ]
    for label, pick in panels:
        if pick is None:
            raise RuntimeError(f"could not find a real P2 agent trace for panel {label!r}")

    vis_lo, vis_hi = 0.25, 0.5
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)))
    for ax, (label, pick) in zip(axes, panels, strict=True):
        scen_name, agent_id, runs, trace = pick
        t = trace.frames / trace.fps
        ax.plot(t, trace.vis, color="black", linewidth=1.0)
        ax.axhline(vis_lo, color="red", linestyle=":", linewidth=1, label="vis_lo=0.25")
        ax.axhline(vis_hi, color="orange", linestyle=":", linewidth=1, label="vis_hi=0.5")
        for r in runs:
            ax.axvspan(r.start_frame / r.fps, (r.end_frame + 1) / r.fps, color="red", alpha=0.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("visibility")
        ax.set_title(f"{label} — {scen_name} agent {agent_id} ({len(runs)} run(s))", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    p3 = OUT_DIR / "p2_visibility_timelines.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", p3)


# ---------------------------------------------------------------------------
# SECTION: mask  (.venv only)
# ---------------------------------------------------------------------------


def run_mask(args: argparse.Namespace) -> None:  # args unused (uniform dispatch signature)
    _ensure_out_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    scenario_id = _first_val_scenario_id()
    val_dir = ROOT / "data" / "av2" / "val" / scenario_id
    parquet_path = val_dir / f"scenario_{scenario_id}.parquet"
    df = pd.read_parquet(parquet_path)
    focal_track_id = str(df["focal_track_id"].iloc[0])

    hist = df[
        (df["track_id"].astype(str) == focal_track_id) & (df["timestep"] < NUM_HISTORICAL_STEPS)
    ].sort_values("timestep")
    if len(hist) != NUM_HISTORICAL_STEPS:
        raise RuntimeError(
            f"focal agent {focal_track_id} has {len(hist)} observed steps, "
            f"expected {NUM_HISTORICAL_STEPS}"
        )
    xy = hist[["position_x", "position_y"]].to_numpy()

    severities = [("S1", 0.2), ("S2", 0.4), ("S3", 0.6), ("S4", 0.8)]
    regimes = ["R-A", "R-B"]
    forced_visible_tail = 5

    fig, axes = plt.subplots(len(severities), len(regimes), figsize=(11, 14))
    for i, (sev_name, frac) in enumerate(severities):
        block_len = int(round(frac * NUM_HISTORICAL_STEPS))
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            seed = i * 10 + j  # deterministic, fixed per panel
            rng = np.random.default_rng(seed)
            masked = np.zeros(NUM_HISTORICAL_STEPS, dtype=bool)

            if regime == "R-A":
                # Last `forced_visible_tail` steps FORCED visible; block placed
                # uniformly at random in the remaining prefix (deterministic rng).
                max_onset = (NUM_HISTORICAL_STEPS - forced_visible_tail) - block_len
                if max_onset < 0:
                    raise RuntimeError(
                        f"{sev_name}/{regime}: block_len={block_len} does not fit before "
                        f"the forced-visible tail ({forced_visible_tail} steps)"
                    )
                onset = int(rng.integers(0, max_onset + 1))
            else:  # R-B: block extends through t=0 (the most recent observed step)
                onset = NUM_HISTORICAL_STEPS - block_len
            masked[onset : onset + block_len] = True

            ax.plot(xy[:, 0], xy[:, 1], color="lightgray", linewidth=1.0, zorder=1)
            vis_pts, mask_pts = xy[~masked], xy[masked]
            ax.scatter(vis_pts[:, 0], vis_pts[:, 1], color="royalblue", marker="o", s=18,
                       zorder=2, label="visible")
            ax.scatter(mask_pts[:, 0], mask_pts[:, 1], color="red",
                       marker="x", s=40, linewidths=1.3, zorder=3, label="masked")

            # unambiguous time direction: star + label at the most recent step
            ax.scatter([xy[-1, 0]], [xy[-1, 1]], marker="*", s=140, color="black", zorder=4)
            ax.annotate("t=0", xy=(xy[-1, 0], xy[-1, 1]), fontsize=8, fontweight="bold",
                        xytext=(6, 6), textcoords="offset points")
            ax.annotate("oldest", xy=(xy[0, 0], xy[0, 1]), fontsize=7, color="gray",
                        xytext=(4, -10), textcoords="offset points")

            achieved = float(masked.mean())
            ax.set_title(f"{sev_name} x {regime} (masked {achieved:.0%})", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if regime == "R-A":
                ax.annotate("last 5 steps forced VISIBLE", xy=(0.02, 0.02),
                            xycoords="axes fraction", fontsize=6.5, color="darkgreen")
            else:
                ax.annotate("block extends through t=0 (still occluded)", xy=(0.02, 0.02),
                            xycoords="axes fraction", fontsize=6.5, color="darkred")
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        f"D-N1-2 mask spec mockup (M1 block, S1–S4 x R-A/R-B) — "
        f"focal agent, scenario {scenario_id}",
        fontsize=12,
    )
    fig.text(
        0.5, 0.005,
        "static spec mockup — generator output replaces this (D-N1-2/D-N1-9 spec, "
        "mix M1 .40/M2 .05/M3 .55)",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_path = OUT_DIR / "mask_spec_mockup.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# SECTION: dash  (.venv only)
# ---------------------------------------------------------------------------


def run_dash(args: argparse.Namespace) -> None:  # args unused (uniform dispatch signature)
    _ensure_out_dir()
    import matplotlib
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    check_gates_py = ROOT / "scripts" / "check_gates.py"
    tmp_report = OUT_DIR / "_gate_report_tmp.txt"
    subprocess.run(
        [str(venv_py), str(check_gates_py), "--report", str(tmp_report)],
        capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    report_text = tmp_report.read_text(encoding="utf-8") if tmp_report.exists() else ""
    if tmp_report.exists():
        tmp_report.unlink()

    cfg = yaml.safe_load((ROOT / "gates.yaml").read_text(encoding="utf-8"))
    frozen_map = {name: bool(g.get("frozen", False)) for name, g in cfg["gates"].items()}

    line_pattern = re.compile(r"^\[(\w+)\]\s+(\S+)\s+\((\w+)\)\s*$")
    status_map: dict[str, str] = {}
    for line in report_text.splitlines():
        m = line_pattern.match(line)
        if m:
            status_map[m.group(2)] = m.group(1)

    gate_rows = [
        (gname, "yes" if frozen_map[gname] else "no", status_map.get(gname, "UNKNOWN"))
        for gname in cfg["gates"]
    ]

    val_dir = ROOT / "data" / "av2" / "val"
    train_dir = ROOT / "data" / "av2" / "train"
    val_count = sum(1 for e in os.scandir(val_dir) if e.is_dir()) if val_dir.exists() else 0
    train_count = sum(1 for e in os.scandir(train_dir) if e.is_dir()) if train_dir.exists() else 0
    val_official, train_official = 24988, 199908
    val_label = f"{val_count} ({'complete' if val_count == val_official else 'syncing'})"
    train_label = f"{train_count} ({'complete' if train_count == train_official else 'syncing'})"

    processed_dir = ROOT / "data" / "qcnet_root" / "val" / "processed"
    processed_count = (
        sum(1 for e in os.scandir(processed_dir) if e.name.endswith(".pkl"))
        if processed_dir.exists()
        else 0
    )

    ckpt_path = ROOT / "checkpoints" / "qcnet_av2.ckpt"
    # utf-8-sig: results/qcnet_ckpt_manifest.json carries a UTF-8 BOM on disk.
    ckpt_manifest = json.loads(
        (ROOT / "results" / "qcnet_ckpt_manifest.json").read_text(encoding="utf-8-sig")
    )
    sha_prefix = str(ckpt_manifest.get("sha256", ""))[:12]

    data_rows = [
        ("val scenario dirs", val_label),
        ("train scenario dirs", train_label),
        ("qcnet_root/val/processed .pkl count", str(processed_count)),
        ("checkpoint present", str(ckpt_path.exists())),
        ("checkpoint sha256 (first 12)", sha_prefix),
    ]

    fig, (ax_gates, ax_data) = plt.subplots(
        2, 1,
        figsize=(9, 3 + 0.4 * len(gate_rows) + 0.3 * len(data_rows)),
        gridspec_kw={"height_ratios": [len(gate_rows), len(data_rows) + 1]},
    )
    ax_gates.axis("off")
    table1 = ax_gates.table(
        cellText=[[n, fz, st] for n, fz, st in gate_rows],
        colLabels=["gate", "frozen?", "status"],
        cellLoc="left", loc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.4)
    ax_gates.set_title("Gate status (scripts/check_gates.py)", fontsize=11, loc="left")

    ax_data.axis("off")
    table2 = ax_data.table(
        cellText=[[k, v] for k, v in data_rows],
        colLabels=["data artifact", "value"],
        cellLoc="left", loc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.4)
    ax_data.set_title("Data pipeline snapshot", fontsize=11, loc="left")

    fig.suptitle(f"N1 state dashboard — {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / "dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# SECTION: assemble  (.venv only)
# ---------------------------------------------------------------------------


def run_assemble(args: argparse.Namespace) -> None:  # args unused (uniform dispatch signature)
    _ensure_out_dir()
    git_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    git_sha = git_result.stdout.strip() or "unknown"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    captions: list[tuple[str, str]] = [
        ("scenario_qcnet.png",
         "QCNet inference end-to-end on the deterministic first val scenario (sorted "
         "scenario dirs[0]): lane centerlines, every agent's observed history, the focal "
         "agent's bold history and dashed GT future, and QCNet's 6 predicted modes "
         "reprojected into world frame via the validated TargetBuilder inverse. The title "
         "reports the official av2-kit best-mode minFDE for this scenario."),
        ("p2_duration_fit.png",
         "Real P2 CARLA occlusion segment durations (recomputed live) against the exact "
         "lognormal fit committed in results/p2_occlusion_stats.json, with the MOT17 "
         "median-gap sanity anchor. The generator asserts the live refit matches the "
         "committed params to 1e-9 before plotting."),
        ("p2_pattern_mix.png",
         "v1 prior pattern mix (M1/M2/M3 = 0.6/0.25/0.15) vs the P2-derived empirical mix "
         "(D-N1-9): real occlusion is flicker-dominated (M3), not block-dominated as v1 assumed."),
        ("p2_visibility_timelines.png",
         "Real per-agent visibility-vs-time traces for one clean M1 (block) agent, one M3 "
         "(flicker, 3+ runs) agent, and one M2 (or other_trailing) agent, with the "
         "vis_lo/vis_hi thresholds and detected segments shaded."),
        ("mask_spec_mockup.png",
         "Static mockup of the D-N1-2 occlusion-masking spec on the same focal agent's real "
         "50-step history: severities S1–S4 x regimes R-A/R-B, M1-block placement only. "
         "The real Phase-3 mask generator (not yet implemented) replaces this."),
        ("dashboard.png",
         "Live gate status from scripts/check_gates.py plus a data-pipeline snapshot "
         "(val/train sync progress, QCNet processed cache size, checkpoint presence+hash)."),
    ]

    sections_html = "\n".join(
        f'<section><h2>{name}</h2>'
        f'<img src="{name}" alt="{name}" style="max-width:100%;border:1px solid #ccc;">'
        f"<p>{caption}</p></section>"
        for name, caption in captions
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>N1 state-of-project report — {timestamp}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto;
        padding: 0 1rem; }}
header {{ border-bottom: 3px solid #c0392b; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }}
header .badge {{ color: #c0392b; font-weight: bold; }}
section {{ margin-bottom: 2.5rem; }}
h2 {{ font-size: 1.1rem; }}
p {{ color: #333; }}
</style>
</head>
<body>
<header>
  <p class="badge">LOCAL REPORT — not committed; AV2-derived pixels assumed non-committable</p>
  <p>Generated {timestamp} &middot; git HEAD {git_sha} &middot; otraj N1
  (occlusion-aware trajectory prediction)</p>
</header>
{sections_html}
</body>
</html>
"""
    out_path = OUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SECTION_FUNCS: dict[str, Any] = {
    "scenario": run_scenario,
    "p2": run_p2,
    "mask": run_mask,
    "dash": run_dash,
    "assemble": run_assemble,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sections", type=str, required=True,
        help=f"comma-separated subset of {sorted(SECTION_FUNCS)}",
    )
    ap.add_argument("--device", type=str, default=None, help="scenario section only; default auto")
    ap.add_argument("--seed", type=int, default=2023, help="scenario section only")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in sections if s not in SECTION_FUNCS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}, choose from {sorted(SECTION_FUNCS)}")

    _ensure_out_dir()
    for name in sections:
        log.info("=== running section: %s ===", name)
        t0 = time.time()
        SECTION_FUNCS[name](args)
        log.info("=== section %s done in %.1fs ===", name, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
