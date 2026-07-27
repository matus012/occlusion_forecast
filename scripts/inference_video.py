"""Side-by-side inference video: C1-local vs C3-local (D-N1-14c/d).

BEV-styled (nuScenes/Waymo visual register, D-N1-14d(7) polish pass):
dark background, filled road polygons from the AV2 static map's lane
boundaries (transformed world -> scene via the pickle's ORIG/ROT), white
dashed lane centerlines, agents as filled oriented rectangles (type-sized,
heading from the per-actor frame), focal highlighted, per-mode prediction
trails alpha-graded along time and weighted by mode probability, dashed GT
future, occlusion rendered as ghost outline at the true-but-hidden pose +
dimmed vignette + OCCLUDED banner, corner HUD (t / severity / pattern /
regime) and legend.

Scenario picking is visually curated BY C1 DEGRADATION (largest per-scenario
C1 minFDE6 increase S3-vs-S0) — picking by C1-vs-C3 gap would oversell the
collapsed C3-local arm (D-N1-14c). Every frame carries the "illustrative
scenario" label; the C3 panel is captioned as the collapse illustration.

Outputs (reports/videos/, gitignored -- AV2-derived visuals never committed):
  {sid_short}_{sev}_c1_vs_c3.mp4 / .gif, stills/{...}_{tag}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SIMPL"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import _simpl_compat  # noqa: E402, F401
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from av2.map.map_api import ArgoverseStaticMap  # noqa: E402
from config.simpl_av2_cfg import AdvCfg  # noqa: E402
from eval_local import MaskedEvalDataset  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402

from otraj.masking.generator import generate_mask  # noqa: E402

OBS, FUT = 50, 60

# ---- BEV style ----
BG = "#0d1117"
ROAD_FILL = "#262b33"
ROAD_EDGE = "#3a4150"
LANE_LINE = "#aab4c0"
CROSSWALK = "#39414d"
AGENT_FILL = "#8b949e"
AGENT_EDGE = "#c9d1d9"
FOCAL_FILL = "#39c0ed"
FOCAL_TRAIL = "#39c0ed"
HIDDEN_GHOST = "#ff5964"
PRED_COLOR = "#ffb020"
GT_COLOR = "#3fe07f"
HUD_COLOR = "#e6edf3"
C1_ACCENT, C3_ACCENT = "#ff6b6b", "#748ffc"

# type one-hot index -> (length, width), av2_preprocess estimated dims
TYPE_DIMS = {0: (5.0, 2.0), 1: (0.5, 0.5), 2: (2.0, 0.8), 3: (2.0, 0.7),
             4: (7.0, 2.1), 5: (1.0, 1.0), 6: (1.0, 1.0)}


def world_to_scene(pts_xy: np.ndarray, orig: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (pts_xy - orig) @ rot


def actor_to_scene(norm_xy: np.ndarray, ctr: np.ndarray, vec: np.ndarray) -> np.ndarray:
    c, s = float(vec[0]), float(vec[1])
    rot = np.array([[c, -s], [s, c]])
    return norm_xy @ rot.T + ctr


def load_scene(pkl_path: Path, map_dir: Path) -> dict:
    df = pd.read_pickle(pkl_path)
    data = {k: df[k].values[0] for k in df.keys()}
    trajs = data["TRAJS"]
    ctrs, vecs = trajs["trajs_ctrs"], trajs["trajs_vecs"]
    n = len(ctrs)
    pos_scene = np.stack([
        actor_to_scene(trajs["trajs_pos"][i], ctrs[i], vecs[i]) for i in range(n)])
    theta = np.arctan2(vecs[:, 1], vecs[:, 0])
    ang_scene = trajs["trajs_ang"] + theta[:, None]  # [N, 110]

    sid = data["SEQ_ID"]
    orig, rot = data["ORIG"], data["ROT"]
    static_map = ArgoverseStaticMap.from_json(
        map_dir / sid / f"log_map_archive_{sid}.json")
    lane_polys, lane_centerlines = [], []
    for lane_id in static_map.vector_lane_segments:
        poly = static_map.get_lane_segment_polygon(lane_id)[:, :2]
        lane_polys.append(world_to_scene(poly, orig, rot))
        cl = static_map.get_lane_segment_centerline(lane_id)[:, :2]
        lane_centerlines.append(world_to_scene(cl, orig, rot))
    crosswalks = [world_to_scene(pc.polygon[:, :2], orig, rot)
                  for pc in static_map.vector_pedestrian_crossings.values()]

    return {
        "sid": sid,
        "pos_scene": pos_scene,              # [N, 110, 2]
        "ang_scene": ang_scene,              # [N, 110]
        "types": trajs["trajs_type"][:, OBS - 1],  # [N, 7] one-hot at t=0
        "has_flags": trajs["has_flags"].copy(),
        "lane_polys": lane_polys,
        "lane_centerlines": lane_centerlines,
        "crosswalks": crosswalks,
        "focal_ctr": ctrs[0], "focal_vec": vecs[0],
    }


@torch.no_grad()
def predict(net: Simpl, ds: MaskedEvalDataset, idx: int) -> tuple[np.ndarray, np.ndarray]:
    data = ds.collate_fn([ds[idx]])
    post = net.post_process(net(net.pre_process(data)))
    return post["traj_pred"][0].cpu().numpy(), post["prob_pred"][0].cpu().numpy()


def minfde6(traj_pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(traj_pred[:, -1] - gt[-1], axis=-1)))


def _agent_box(ax, center, heading, dims, face, edge, alpha=1.0, ghost=False,
               zorder=6):
    length, width = dims
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                        [-length / 2, -width / 2], [-length / 2, width / 2]])
    corners = corners @ rot.T + center
    if ghost:
        ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                                edgecolor=edge, lw=1.1, ls=(0, (2, 2)),
                                alpha=alpha, zorder=zorder))
    else:
        ax.add_patch(MplPolygon(corners, closed=True, facecolor=face,
                                edgecolor=edge, lw=0.8, alpha=alpha,
                                zorder=zorder))
        # heading tick
        nose = center + rot @ np.array([length / 2, 0.0])
        ax.plot([center[0], nose[0]], [center[1], nose[1]], color=edge,
                lw=1.0, alpha=alpha, zorder=zorder + 1)


def _gradient_trail(ax, pts, color, max_alpha, lw=2.2, zorder=5, dashed=False):
    """Polyline whose alpha ramps up toward the trail's end."""
    if len(pts) < 2:
        return
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    alphas = np.linspace(0.12, max_alpha, len(segs))
    lc = LineCollection(segs, colors=[color] * len(segs), alpha=alphas,
                        linewidths=lw, zorder=zorder,
                        linestyles=(0, (4, 2)) if dashed else "solid",
                        capstyle="round")
    ax.add_collection(lc)


def _dims_for(type_onehot: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(type_onehot)) if type_onehot.any() else 0
    return TYPE_DIMS.get(idx, (1.0, 1.0))


def render_take(scene: dict, mask: np.ndarray, preds: dict, sev: str,
                pattern: str, out_dir: Path, fps: int = 10) -> None:
    sid = scene["sid"]
    short = sid.split("-")[0]
    pos, ang = scene["pos_scene"], scene["ang_scene"]
    focal = pos[0]
    gt_fut = focal[OBS:]

    # camera: frame the region of interest (occluded segment + t=0 pose + GT
    # future), not the full track — BEV figures read best at ~60-120 m span
    roi = np.concatenate([focal[OBS - 1: OBS], gt_fut,
                          focal[:OBS][mask] if mask.any() else focal[OBS - 1: OBS]])
    lo = roi.min(0) - 20.0
    hi = roi.max(0) + 20.0
    span = float(np.clip(max(hi[0] - lo[0], hi[1] - lo[1]), 60.0, 120.0))
    cx, cy = (lo + hi) / 2.0
    xlim = (cx - span / 2, cx + span / 2)
    ylim = (cy - span / 2, cy + span / 2)

    pred_scene = {
        arm: (np.stack([actor_to_scene(t[m], scene["focal_ctr"], scene["focal_vec"])
                        for m in range(t.shape[0])]), p, f)
        for arm, (t, p, f) in preds.items()}

    schedule = ([("hist", t) for t in range(0, OBS, 1)]
                + [("fans", OBS - 1)] * 8
                + [("fut", j) for j in range(2, FUT + 1, 2)]
                + [("end", FUT)] * 6)
    mask_mid = int(np.flatnonzero(mask).mean()) if mask.any() else 25
    still_keys = {("hist", mask_mid): "occluded",
                  ("fans", OBS - 1): "fans", ("end", FUT): "final"}

    arm_meta = [("C1-local", "clean-trained", C1_ACCENT),
                ("C3-local", "occl-aug (collapse illustration,\nD-N1-14c: "
                             "mask-invariant)", C3_ACCENT)]

    legend_handles = [
        Line2D([], [], color=FOCAL_TRAIL, lw=2.4, label="focal history (observed)"),
        Line2D([], [], color=HIDDEN_GHOST, lw=1.2, ls="--",
               label="true pose (hidden from model)"),
        Line2D([], [], color=PRED_COLOR, lw=2.2, label="prediction (6 modes, α ∝ prob.)"),
        Line2D([], [], color=GT_COLOR, lw=2.2, ls="--", label="ground-truth future"),
    ]

    frames: list[np.ndarray] = []
    stills: dict[str, int] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), dpi=100)
    fig.patch.set_facecolor(BG)

    for f_i, (phase, t) in enumerate(schedule):
        for ax, (arm, desc, accent) in zip(axes, arm_meta, strict=True):
            ax.clear()
            ax.set_facecolor(BG)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(accent)
                spine.set_linewidth(1.6)

            # --- map: road fill, crosswalks, lane lines ---
            for poly in scene["lane_polys"]:
                ax.add_patch(MplPolygon(poly, closed=True, facecolor=ROAD_FILL,
                                        edgecolor=ROAD_EDGE, lw=0.4, zorder=1))
            for cw in scene["crosswalks"]:
                ax.add_patch(MplPolygon(cw, closed=True, facecolor=CROSSWALK,
                                        edgecolor="none", zorder=2, alpha=0.9))
            for cl in scene["lane_centerlines"]:
                ax.plot(cl[:, 0], cl[:, 1], color=LANE_LINE, lw=0.6,
                        ls=(0, (5, 5)), alpha=0.30, zorder=3)

            t_now = t if phase == "hist" else OBS - 1
            occluded_now = phase == "hist" and bool(mask[t_now])

            # --- other agents: oriented boxes + short trails ---
            for i in range(1, pos.shape[0]):
                if not scene["has_flags"][i, t_now]:
                    continue
                trail = pos[i, max(0, t_now - 12): t_now + 1]
                vis = scene["has_flags"][i, max(0, t_now - 12): t_now + 1] > 0
                _gradient_trail(ax, trail[vis], AGENT_FILL, 0.5, lw=1.2, zorder=4)
                _agent_box(ax, pos[i, t_now], ang[i, t_now],
                           _dims_for(scene["types"][i]),
                           AGENT_FILL, AGENT_EDGE, alpha=0.75, zorder=5)

            # --- focal: observed trail + box; hidden truth as ghost ---
            obs_vis = ~mask[: t_now + 1]
            seen_idx = np.flatnonzero(obs_vis)
            if seen_idx.size:
                _gradient_trail(ax, focal[: t_now + 1][obs_vis], FOCAL_TRAIL,
                                0.95, lw=2.6, zorder=6)
                last_obs = int(seen_idx[-1])
                _agent_box(ax, focal[last_obs], ang[0, last_obs],
                           _dims_for(scene["types"][0]),
                           FOCAL_FILL, "#d9f3ff", zorder=7)
            hid_idx = np.flatnonzero(mask[: t_now + 1])
            if hid_idx.size:
                for h in hid_idx[::4]:
                    _agent_box(ax, focal[h], ang[0, h],
                               _dims_for(scene["types"][0]), None,
                               HIDDEN_GHOST, alpha=0.55, ghost=True, zorder=6)
                if occluded_now:
                    _agent_box(ax, focal[t_now], ang[0, t_now],
                               _dims_for(scene["types"][0]), None,
                               HIDDEN_GHOST, alpha=0.95, ghost=True, zorder=8)

            # --- predictions + GT future ---
            if phase in ("fans", "fut", "end"):
                trajs_s, prob, fde = pred_scene[arm]
                pmax = float(prob.max())
                start = focal[OBS - 1][None, :]
                for m in np.argsort(prob):
                    a = 0.30 + 0.65 * float(prob[m]) / pmax
                    pts = np.concatenate([start, trajs_s[m]], axis=0)
                    _gradient_trail(ax, pts, PRED_COLOR, a, lw=2.0, zorder=6)
                    ax.plot(trajs_s[m, -1, 0], trajs_s[m, -1, 1], "o",
                            color=PRED_COLOR, ms=4, alpha=a, zorder=7,
                            mec="none")
                ax.text(0.03, 0.03, f"minFDE$_6$ = {fde:.2f} m",
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        color=accent, zorder=12)
            if phase in ("fut", "end"):
                j = t if phase == "fut" else FUT
                pts = np.concatenate([focal[OBS - 1][None, :], gt_fut[:j]], axis=0)
                _gradient_trail(ax, pts, GT_COLOR, 0.95, lw=2.4, zorder=7,
                                dashed=True)
                if j:
                    ax.plot(gt_fut[j - 1, 0], gt_fut[j - 1, 1], "*",
                            color=GT_COLOR, ms=13, zorder=8)

            # --- occlusion vignette + banner ---
            if occluded_now:
                ax.add_patch(Rectangle((xlim[0], ylim[0]), span, span,
                                       facecolor="black", alpha=0.32, zorder=9))
                ax.text(0.5, 0.90, "OCCLUDED", transform=ax.transAxes,
                        ha="center", fontsize=15, fontweight="bold",
                        color=HIDDEN_GHOST, alpha=0.9, zorder=10,
                        bbox={"facecolor": BG, "edgecolor": HIDDEN_GHOST,
                              "alpha": 0.65, "pad": 4})

            # --- HUD ---
            if phase == "hist":
                t_str = f"t = {(t_now - OBS + 1) / 10.0:+05.1f} s"
            elif phase == "fans":
                t_str = "t = +00.0 s  PREDICT"
            else:
                j = t if phase == "fut" else FUT
                t_str = f"t = +{j / 10.0:04.1f} s"
            ax.text(0.03, 0.955, f"{t_str}   {sev} · {pattern} · R-A",
                    transform=ax.transAxes, fontsize=9, family="monospace",
                    color=HUD_COLOR, va="top", zorder=12,
                    bbox={"facecolor": BG, "edgecolor": "#30363d", "alpha": 0.75,
                          "pad": 3})
            # timeline mask bar
            for k_step in range(0, OBS, 1):
                col = HIDDEN_GHOST if mask[k_step] else "#4d5561"
                ax.plot([0.70 + 0.26 * k_step / OBS], [0.965], "s", color=col,
                        ms=2.2, transform=ax.transAxes, zorder=12)
            if phase == "hist":
                ax.plot([0.70 + 0.26 * t_now / OBS], [0.940], "^",
                        color=HUD_COLOR, ms=4, transform=ax.transAxes, zorder=12)
            ax.set_title(f"{arm} — {desc}", color=accent, fontsize=9.5, pad=6)

        axes[0].legend(handles=legend_handles, loc="lower right", fontsize=6.5,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor=HUD_COLOR, framealpha=0.9)
        fig.suptitle(
            f"scenario {short}…  ·  identical deterministic N1-mask-v2 masks  ·  "
            "illustrative scenario — reduced-scale local checkpoints "
            "(6-epoch truncated arms; full matrix pending HPC)",
            fontsize=9.5, color=HUD_COLOR)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        if (phase, t) in still_keys and still_keys[(phase, t)] not in stills:
            stills[still_keys[(phase, t)]] = f_i
    plt.close(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stills").mkdir(exist_ok=True)
    base = f"{short}_{sev}_c1_vs_c3"
    imageio.mimwrite(out_dir / f"{base}.mp4", frames, fps=fps, codec="libx264",
                     quality=8)
    imageio.mimwrite(out_dir / f"{base}.gif", [f[::2, ::2] for f in frames[::2]],
                     fps=fps // 2, loop=0)
    for tag, f_i in stills.items():
        imageio.imwrite(out_dir / "stills" / f"{base}_{tag}.png", frames[f_i])
    print(f"[take] {base}.mp4: {len(frames)} frames, stills {list(stills)}")


def render_c1_degradation_still(scene: dict, mask: np.ndarray,
                                preds_by_sev: dict, sev: str, pattern: str,
                                out_dir: Path) -> None:
    """Committee still (D-N1-14d(7) judgment): C1-local on the SAME scenario,
    clean (S0) vs masked (sev) — the motivating degradation visual. The C3
    panel is omitted here: its collapse makes a side-by-side fans frame read
    as 'two similar methods' rather than a collapse illustration."""
    sid = scene["sid"]
    short = sid.split("-")[0]
    pos, ang = scene["pos_scene"], scene["ang_scene"]
    focal = pos[0]
    gt_fut = focal[OBS:]
    roi = np.concatenate([focal[OBS - 1: OBS], gt_fut,
                          focal[:OBS][mask] if mask.any() else focal[OBS - 1: OBS]])
    lo = roi.min(0) - 20.0
    hi = roi.max(0) + 20.0
    span = float(np.clip(max(hi[0] - lo[0], hi[1] - lo[1]), 60.0, 120.0))
    cx, cy = (lo + hi) / 2.0
    xlim = (cx - span / 2, cx + span / 2)
    ylim = (cy - span / 2, cy + span / 2)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), dpi=100)
    fig.patch.set_facecolor(BG)
    panels = [("no occlusion (S0)", np.zeros_like(mask), preds_by_sev["S0"]),
              (f"{int(mask.sum() / OBS * 100)}% of history occluded "
               f"({sev} · {pattern} · R-A)", mask, preds_by_sev[sev])]
    for ax, (desc, pmask, (trajs_s, prob, fde)) in zip(axes, panels, strict=True):
        ax.set_facecolor(BG)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C1_ACCENT)
            spine.set_linewidth(1.6)
        for poly in scene["lane_polys"]:
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=ROAD_FILL,
                                    edgecolor=ROAD_EDGE, lw=0.4, zorder=1))
        for cw in scene["crosswalks"]:
            ax.add_patch(MplPolygon(cw, closed=True, facecolor=CROSSWALK,
                                    edgecolor="none", zorder=2, alpha=0.9))
        for cl in scene["lane_centerlines"]:
            ax.plot(cl[:, 0], cl[:, 1], color=LANE_LINE, lw=0.6,
                    ls=(0, (5, 5)), alpha=0.30, zorder=3)
        t_now = OBS - 1
        for i in range(1, pos.shape[0]):
            if not scene["has_flags"][i, t_now]:
                continue
            _agent_box(ax, pos[i, t_now], ang[i, t_now],
                       _dims_for(scene["types"][i]), AGENT_FILL, AGENT_EDGE,
                       alpha=0.75, zorder=5)
        obs_vis = ~pmask
        _gradient_trail(ax, focal[:OBS][obs_vis], FOCAL_TRAIL, 0.95, lw=2.6,
                        zorder=6)
        _agent_box(ax, focal[t_now], ang[0, t_now], _dims_for(scene["types"][0]),
                   FOCAL_FILL, "#d9f3ff", zorder=7)
        for h in np.flatnonzero(pmask)[::4]:
            _agent_box(ax, focal[h], ang[0, h], _dims_for(scene["types"][0]),
                       None, HIDDEN_GHOST, alpha=0.55, ghost=True, zorder=6)
        pmax = float(prob.max())
        start = focal[OBS - 1][None, :]
        for m in np.argsort(prob):
            a = 0.30 + 0.65 * float(prob[m]) / pmax
            _gradient_trail(ax, np.concatenate([start, trajs_s[m]], axis=0),
                            PRED_COLOR, a, lw=2.0, zorder=6)
            ax.plot(trajs_s[m, -1, 0], trajs_s[m, -1, 1], "o", color=PRED_COLOR,
                    ms=4, alpha=a, zorder=7, mec="none")
        _gradient_trail(ax, np.concatenate([start, gt_fut], axis=0), GT_COLOR,
                        0.95, lw=2.4, zorder=7, dashed=True)
        ax.plot(gt_fut[-1, 0], gt_fut[-1, 1], "*", color=GT_COLOR, ms=13, zorder=8)
        ax.text(0.03, 0.03, f"minFDE$_6$ = {fde:.2f} m", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=C1_ACCENT, zorder=12)
        ax.set_title(f"C1-local (clean-trained) — {desc}", color=C1_ACCENT,
                     fontsize=10, pad=6)
        for k_step in range(OBS):
            col = HIDDEN_GHOST if pmask[k_step] else "#4d5561"
            ax.plot([0.70 + 0.26 * k_step / OBS], [0.965], "s", color=col,
                    ms=2.2, transform=ax.transAxes, zorder=12)
    fig.suptitle(
        f"scenario {short}…  ·  SAME scenario, SAME model — occlusion alone "
        "degrades the forecast  ·  illustrative scenario, 6-epoch truncated "
        "*-local arm (full matrix pending HPC)", fontsize=9.5, color=HUD_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    (out_dir / "stills").mkdir(parents=True, exist_ok=True)
    out = out_dir / "stills" / f"{short}_{sev}_c1_degradation.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"[still] {out.name}")


def auto_pick(eval_dir: Path, sev: str, pattern: str, features_dir: Path,
              map_dir: Path, min_travel_m: float, top: int) -> list[str]:
    """Largest per-scenario C1 degradation (C1@sev - C1@S0), pattern cohort,
    dynamic focal agents only. See module docstring for why NOT C1-vs-C3."""
    c1 = json.loads((eval_dir / "eval_C1-local_native.json").read_text("utf-8"))
    c1s = {r["sid"]: r for r in c1["severities"][sev]["per_scenario"]}
    c1s0 = {r["sid"]: r for r in c1["severities"]["S0"]["per_scenario"]}
    cands = sorted(
        ((r["minfde6"] - c1s0[sid]["minfde6"], sid) for sid, r in c1s.items()
         if r["pattern"] == pattern and sid in c1s0
         and r["minfde6"] > c1s0[sid]["minfde6"]),
        reverse=True)
    picked = []
    for gap, sid in cands:
        scene = load_scene(features_dir / f"{sid}.pkl", map_dir)
        travel = float(np.linalg.norm(
            scene["pos_scene"][0, -1] - scene["pos_scene"][0, 0]))
        if travel >= min_travel_m:
            picked.append(sid)
            print(f"[pick {sev}/{pattern}] {sid} C1-degradation={gap:.2f}m "
                  f"travel={travel:.0f}m")
        if len(picked) >= top:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-ckpt", type=Path,
                    default=REPO / "checkpoints" / "local" / "c1_seed42.tar")
    ap.add_argument("--c3-ckpt", type=Path,
                    default=REPO / "checkpoints" / "local" / "c3_seed42.tar")
    ap.add_argument("--takes", default="S3:M1:2",
                    help="comma list severity:pattern:count (D-N1-14b: 2 takes)")
    ap.add_argument("--sids", default="", help="explicit sid:severity pairs")
    ap.add_argument("--min-travel-m", type=float, default=25.0)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "simpl_features" / "local" / "val")
    ap.add_argument("--map-dir", type=Path, default=REPO / "data" / "av2" / "val")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "videos")
    args = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    cfg = AdvCfg()
    nets = {}
    for arm, ckpt_path in (("C1-local", args.c1_ckpt), ("C3-local", args.c3_ckpt)):
        net = Simpl(cfg.get_net_cfg(), device).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        nets[arm] = net

    if args.sids:
        takes = [tuple(x.split(":")) for x in args.sids.split(",")]
    else:
        takes = []
        for spec in args.takes.split(","):
            sev, pattern, count = spec.split(":")
            takes += [(sid, sev) for sid in auto_pick(
                REPO / "results" / "local", sev, pattern, args.features_dir,
                args.map_dir, args.min_travel_m, int(count))]

    for sid, sev in takes:
        pkl = args.features_dir / f"{sid}.pkl"
        scene = load_scene(pkl, args.map_dir)
        res = generate_mask(sid, sev, "R-A")
        ds = MaskedEvalDataset([str(pkl)], sev, "R-A", "native")
        gt = np.asarray(ds[0]["TRAJS"]["TRAJS_POS_FUT"][0])
        preds = {}
        for arm, net in nets.items():
            traj, prob = predict(net, ds, 0)
            preds[arm] = (traj, prob, minfde6(traj, gt))
        render_take(scene, res.mask, preds, sev, res.pattern, args.out_dir)

        # committee still: C1-local S0 vs sev on the same scenario, scene-frame
        # predictions from the C1 net under each condition
        c1_by_sev = {}
        for cond in ("S0", sev):
            ds_c = MaskedEvalDataset([str(pkl)], cond, "R-A", "native")
            traj, prob = predict(nets["C1-local"], ds_c, 0)
            traj_s = np.stack([
                actor_to_scene(traj[m], scene["focal_ctr"], scene["focal_vec"])
                for m in range(traj.shape[0])])
            c1_by_sev[cond] = (traj_s, prob, minfde6(traj, gt))
        render_c1_degradation_still(scene, res.mask, c1_by_sev, sev,
                                    res.pattern, args.out_dir)


if __name__ == "__main__":
    main()
