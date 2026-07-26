"""Side-by-side inference video: C1-local vs C3-local (D-N1-14c).

Renders, for each chosen scenario, a two-panel animation (left: C1-local
clean-trained, right: C3-local occlusion-aug trained) on IDENTICAL
deterministic N1-mask-v2 masks:

  phase A (5 s): observed history reveals step by step; the focal agent's
      occluded steps are NOT drawn as observations -- the true-but-hidden
      positions appear as hollow markers ("ground truth, hidden from model"),
      and a timeline bar shows the mask;
  phase B: at t=0 both models predict; the K=6 fans appear (alpha ~ mode
      probability), annotated with each arm's minFDE6;
  phase C (6 s): ground-truth future unfolds over the static fans.

Scenario picking is ALLOWED to be visually curated (user directive) and every
frame carries the label "illustrative scenario". Default takes: top
C1-vs-C3 minFDE6 gap among M1-cohort scenarios at S3/R-A with a sufficiently
dynamic focal agent, plus one S4 take. Requires the *-local eval JSONs
(scripts/eval_local.py) for auto-picking.

Outputs (reports/videos/, gitignored -- AV2-derived visuals never committed):
  {sid_short}_{sev}_c1_vs_c3.mp4  (imageio-ffmpeg, 10 fps)
  {sid_short}_{sev}_c1_vs_c3.gif  (downscaled, 5 fps)
  stills/{sid_short}_{sev}_{tag}.png  (occlusion mid / fans / final)
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
from config.simpl_av2_cfg import AdvCfg  # noqa: E402
from eval_local import MaskedEvalDataset  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402

from otraj.masking.generator import generate_mask  # noqa: E402

OBS, FUT = 50, 60
C1_COLOR, C3_COLOR = "#c44e52", "#4c72b0"
PRED_CMAP = "#e58606"


def actor_to_scene(norm_xy: np.ndarray, ctr: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Invert SIMPL's per-actor normalization: p_scene = p_norm @ R^T + ctr,
    with R = [[c,-s],[s,c]] built from vec = (cos t, sin t)."""
    c, s = float(vec[0]), float(vec[1])
    rot = np.array([[c, -s], [s, c]])
    return norm_xy @ rot.T + ctr


def load_scene(pkl_path: Path) -> dict:
    df = pd.read_pickle(pkl_path)
    data = {k: df[k].values[0] for k in df.keys()}
    trajs = data["TRAJS"]
    ctrs, vecs = trajs["trajs_ctrs"], trajs["trajs_vecs"]
    n = len(ctrs)
    pos_scene = np.stack([
        actor_to_scene(trajs["trajs_pos"][i], ctrs[i], vecs[i]) for i in range(n)])
    return {
        "sid": data["SEQ_ID"],
        "pos_scene": pos_scene,                      # [N, 110, 2]
        "has_flags": trajs["has_flags"].copy(),      # [N, 110]
        "lane_nodes": data["LANE_GRAPH"]["node_ctrs"],  # [n_lane, 10, 2]
        "focal_ctr": ctrs[0], "focal_vec": vecs[0],
    }


@torch.no_grad()
def predict(net: Simpl, ds: MaskedEvalDataset, idx: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (traj_pred [K,60,2] focal frame, prob [K])."""
    data = ds.collate_fn([ds[idx]])
    out = net(net.pre_process(data))
    post = net.post_process(out)
    return (post["traj_pred"][0].cpu().numpy(), post["prob_pred"][0].cpu().numpy())


def minfde6(traj_pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(traj_pred[:, -1] - gt[-1], axis=-1)))


def render_take(scene: dict, mask: np.ndarray, preds: dict, sev: str,
                pattern: str, out_dir: Path, fps: int = 10) -> None:
    sid = scene["sid"]
    short = sid.split("-")[0]
    pos = scene["pos_scene"]
    focal = pos[0]
    gt_fut_scene = focal[OBS:]

    # fixed camera: bbox of focal full track +/- 15 m, square
    all_pts = focal
    lo = all_pts.min(0) - 15.0
    hi = all_pts.max(0) + 15.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    cx, cy = (lo + hi) / 2.0
    xlim = (cx - span / 2, cx + span / 2)
    ylim = (cy - span / 2, cy + span / 2)

    pred_scene = {}
    for arm, (traj, prob, fde) in preds.items():
        k = traj.shape[0]
        pred_scene[arm] = (
            np.stack([actor_to_scene(traj[m], scene["focal_ctr"], scene["focal_vec"])
                      for m in range(k)]),
            prob, fde)

    frames: list[np.ndarray] = []
    stills: dict[str, int] = {}
    # frame schedule: history 0..49 (every step), then fans hold x8, then future
    schedule = ([("hist", t) for t in range(0, OBS, 1)]
                + [("fans", OBS - 1)] * 8
                + [("fut", j) for j in range(2, FUT + 1, 2)]
                + [("end", FUT)] * 6)
    mask_mid = int(np.flatnonzero(mask).mean()) if mask.any() else 25
    still_keys = {("hist", mask_mid): "occluded",
                  ("fans", OBS - 1): "fans", ("end", FUT): "final"}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100)
    arm_meta = [("C1-local", "clean-trained", C1_COLOR),
                ("C3-local", "occlusion-aug trained", C3_COLOR)]
    for f_i, (phase, t) in enumerate(schedule):
        for ax, (arm, desc, color) in zip(axes, arm_meta, strict=True):
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            # lanes
            for lane in scene["lane_nodes"]:
                ax.plot(lane[:, 0], lane[:, 1], color="0.85", lw=1, zorder=1)
            t_now = t if phase == "hist" else OBS - 1
            # other agents observed history
            for i in range(1, pos.shape[0]):
                vis = scene["has_flags"][i, :t_now + 1] > 0
                pts = pos[i, :t_now + 1][vis]
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], color="0.6", lw=1.2, zorder=2)
                    ax.plot(pts[-1, 0], pts[-1, 1], "o", color="0.5", ms=3, zorder=2)
            # focal observed history (masked steps NOT drawn as observations)
            obs_vis = ~mask[: t_now + 1]
            seen = focal[: t_now + 1]
            ax.plot(np.where(obs_vis, seen[:, 0], np.nan),
                    np.where(obs_vis, seen[:, 1], np.nan),
                    color="k", lw=2.2, zorder=4, label="focal history (observed)")
            # true-but-hidden positions
            hid = seen[mask[: t_now + 1]]
            if len(hid):
                ax.plot(hid[:, 0], hid[:, 1], "o", mfc="none", mec="crimson",
                        ms=5, mew=1.2, zorder=4,
                        label="ground truth (hidden from model)")
            cur = seen[obs_vis]
            if len(cur):
                ax.plot(cur[-1, 0], cur[-1, 1], "s", color="k", ms=7, zorder=5)
            # predictions + future
            if phase in ("fans", "fut", "end"):
                trajs_s, prob, fde = pred_scene[arm]
                pmax = float(prob.max())
                for m in np.argsort(prob):
                    a = 0.25 + 0.75 * float(prob[m]) / pmax
                    ax.plot(trajs_s[m, :, 0], trajs_s[m, :, 1], color=PRED_CMAP,
                            lw=1.8, alpha=a, zorder=3)
                    ax.plot(trajs_s[m, -1, 0], trajs_s[m, -1, 1], "D",
                            color=PRED_CMAP, alpha=a, ms=4, zorder=3)
                ax.text(0.02, 0.02, f"minFDE$_6$ = {fde:.2f} m",
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        color=color)
            if phase in ("fut", "end"):
                j = t if phase == "fut" else FUT
                ax.plot(gt_fut_scene[:j, 0], gt_fut_scene[:j, 1], color="#2ca02c",
                        lw=2.4, zorder=4, label="ground-truth future")
                if j:
                    ax.plot(gt_fut_scene[j - 1, 0], gt_fut_scene[j - 1, 1], "*",
                            color="#2ca02c", ms=12, zorder=5)
            ax.set_title(f"{arm} — {desc}", color=color, fontsize=12)
            # timeline mask bar
            bar_y = 0.965
            for k_step in range(OBS):
                col = ("crimson" if mask[k_step] else "0.8")
                ax.plot([0.02 + 0.20 * k_step / OBS], [bar_y], "s", color=col,
                        ms=2.5, transform=ax.transAxes, zorder=6)
            if phase == "hist":
                ax.plot([0.02 + 0.20 * t_now / OBS], [bar_y - 0.03], "^",
                        color="k", ms=4, transform=ax.transAxes, zorder=6)
        t_label = (f"t = {(t_now - OBS + 1) / 10.0:+.1f} s" if phase == "hist"
                   else ("t = 0: predict" if phase == "fans"
                         else f"future +{(t if phase == 'fut' else FUT) / 10.0:.1f} s"))
        fig.suptitle(
            f"scenario {short}…  |  pattern {pattern}, severity {sev}, regime R-A, "
            f"identical masks  |  {t_label}\n"
            "illustrative scenario — reduced-scale local checkpoints "
            "(full matrix pending HPC)", fontsize=10)
        axes[0].legend(loc="lower right", fontsize=7)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        if (phase, t) in still_keys and still_keys[(phase, t)] not in stills:
            stills[still_keys[(phase, t)]] = f_i
    plt.close(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stills").mkdir(exist_ok=True)
    base = f"{short}_{sev}_c1_vs_c3"
    mp4 = out_dir / f"{base}.mp4"
    imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=8)
    gif_frames = [f[::2, ::2] for f in frames[::2]]
    gif = out_dir / f"{base}.gif"
    imageio.mimwrite(gif, gif_frames, fps=fps // 2, loop=0)
    for tag, f_i in stills.items():
        imageio.imwrite(out_dir / "stills" / f"{base}_{tag}.png", frames[f_i])
    print(f"[take] {mp4.name}: {len(frames)} frames, stills {list(stills)}")


def auto_pick(eval_dir: Path, sev: str, pattern: str, features_dir: Path,
              min_travel_m: float, top: int) -> list[str]:
    c1 = json.loads((eval_dir / "eval_C1-local_native.json").read_text("utf-8"))
    c3 = json.loads((eval_dir / "eval_C3-local_native.json").read_text("utf-8"))
    c1s = {r["sid"]: r for r in c1["severities"][sev]["per_scenario"]}
    c3s = {r["sid"]: r for r in c3["severities"][sev]["per_scenario"]}
    cands = []
    for sid, r in c1s.items():
        if r["pattern"] != pattern or sid not in c3s:
            continue
        gap = r["minfde6"] - c3s[sid]["minfde6"]
        if gap <= 0:
            continue
        cands.append((gap, sid))
    cands.sort(reverse=True)
    picked = []
    for gap, sid in cands:
        scene = load_scene(features_dir / f"{sid}.pkl")
        travel = float(np.linalg.norm(
            scene["pos_scene"][0, -1] - scene["pos_scene"][0, 0]))
        if travel >= min_travel_m:
            picked.append(sid)
            print(f"[pick {sev}/{pattern}] {sid} gap={gap:.2f}m travel={travel:.0f}m")
        if len(picked) >= top:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-ckpt", type=Path,
                    default=REPO / "checkpoints" / "local" / "c1_seed42.tar")
    ap.add_argument("--c3-ckpt", type=Path,
                    default=REPO / "checkpoints" / "local" / "c3_seed42.tar")
    ap.add_argument("--takes", default="S3:M1:2,S4:M1:1",
                    help="comma list severity:pattern:count for auto-picking")
    ap.add_argument("--sids", default="",
                    help="explicit sid:severity pairs, overrides --takes")
    ap.add_argument("--min-travel-m", type=float, default=25.0)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "simpl_features" / "local" / "val")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "videos")
    args = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    cfg = AdvCfg()
    nets = {}
    for arm, ckpt_path in (("C1-local", args.c1_ckpt), ("C3-local", args.c3_ckpt)):
        net = Simpl(cfg.get_net_cfg(), device).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        nets[arm] = net

    takes: list[tuple[str, str]] = []  # (sid, severity)
    if args.sids:
        takes = [tuple(x.split(":")) for x in args.sids.split(",")]
    else:
        for spec in args.takes.split(","):
            sev, pattern, count = spec.split(":")
            for sid in auto_pick(REPO / "results" / "local", sev, pattern,
                                 args.features_dir, args.min_travel_m, int(count)):
                takes.append((sid, sev))

    for sid, sev in takes:
        pkl = args.features_dir / f"{sid}.pkl"
        scene = load_scene(pkl)
        res = generate_mask(sid, sev, "R-A")
        ds = MaskedEvalDataset([str(pkl)], sev, "R-A", "native")
        gt_focal_frame = None
        preds = {}
        for arm, net in nets.items():
            traj, prob = predict(net, ds, 0, device)
            if gt_focal_frame is None:
                item = ds[0]
                gt_focal_frame = np.asarray(item["TRAJS"]["TRAJS_POS_FUT"][0])
            preds[arm] = (traj, prob, minfde6(traj, gt_focal_frame))
        render_take(scene, res.mask, preds, sev, res.pattern, args.out_dir)


if __name__ == "__main__":
    main()
