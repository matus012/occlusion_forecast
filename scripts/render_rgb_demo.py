"""Rolling-prediction RGB demo clip: SIMPL predictions on real AV2 ring-camera
footage (D-N1-15).

For every annotation step the model re-predicts from a fresh 110-step window, so
the fan genuinely tracks the scene rather than being a frozen still panned over.
Camera frames run at 20 Hz against 10 Hz annotations, so each prediction is held
across the ~2 camera frames it covers.

CLIP LENGTH IS BOUNDED BY THE DATA, not by choice: a ~15 s log carries ~157
annotated frames and each prediction needs 110, giving ~48 rolling windows =
~4.8 s. Owner decision (D-N1-15): accept ~5 s clips, do not engineer around it.

CHECKPOINT-AGNOSTIC BY CONSTRUCTION: --ckpt and --ckpt-label are plain arguments
and nothing else in the pipeline knows which arm produced them, so re-rendering
every clip against a post-HPC checkpoint is one command per clip with no code
change. That is a hard requirement of this build, not a convenience.

Outputs land in reports/videos/rgb/ -- AV2-derived pixels, so local and untracked
for ever (ALLOWED_TRACKED_VISUALS stays empty; D-N1-14d precedent).

Usage:
  python scripts/render_rgb_demo.py --log-dir data/av2_sensor/val/<id> \
      --cam ring_front_left --focal <uuid> --ckpt checkpoints/local/c1_seed42.tar
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SIMPL"))
sys.path.insert(0, str(REPO / "third_party" / "SIMPL" / "data_av2"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from types import SimpleNamespace  # noqa: E402

import _simpl_compat  # noqa: E402, F401  (py3.11 shims; must precede simpl imports)
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from av2.geometry.camera.pinhole_camera import PinholeCamera  # noqa: E402
from av2.map.map_api import ArgoverseStaticMap  # noqa: E402
from av2_preprocess import ArgoPreprocAV2  # noqa: E402
from config.simpl_av2_cfg import AdvCfg  # noqa: E402
from eval_local import MaskedEvalDataset  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402

from otraj.sensor import (  # noqa: E402
    FUT,
    OBS,
    build_scenario,
    focal_windows,
    ground_height,
    load_poses,
    n_windows,
    pose_at,
    project_city_to_img,
)
from otraj.sensor.projection import nearest_index  # noqa: E402

# Spike style, frozen. Polish is explicitly out of scope (D-N1-14d lesson).
PRED_COLOR = "#ffb020"
GT_COLOR = "#3fe07f"
HIST_COLOR = "#39c0ed"
HUD_BG = "#0d1117"
HUD_FG = "#e6edf3"
HONESTY = ("under-trained local checkpoint — pipeline demonstration; "
           "full-scale results pending HPC")
LEGEND = [
    Line2D([], [], color=PRED_COLOR, lw=3.0, label="prediction (6 modes, α ∝ prob.)"),
    Line2D([], [], color=GT_COLOR, lw=2.4, ls="--", label="ground-truth future (6 s)"),
    Line2D([], [], color=HIST_COLOR, lw=2.6, label="observed history (5 s)"),
]


def load_net(ckpt: Path, device: torch.device) -> tuple[Simpl, dict]:
    net = Simpl(AdvCfg().get_net_cfg(), device).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(state["state_dict"])
    net.eval()
    return net, state


def focal_to_city(traj: np.ndarray, meta: dict) -> np.ndarray:
    """[..., 60, 2] focal-actor frame -> city frame.

    SIMPL is target-centric: predictions come out in the focal actor's own
    normalised frame. Undo actor normalisation, then scene normalisation. Both
    are rigid, so ADE/FDE are untouched (see eval_local docstring)."""
    ctr = meta["TRAJS"]["trajs_ctrs"][0]
    vec = meta["TRAJS"]["trajs_vecs"][0]
    c, s = float(vec[0]), float(vec[1])
    scene = traj @ np.array([[c, -s], [s, c]]).T + ctr
    return scene @ np.asarray(meta["ROT"]).T + np.asarray(meta["ORIG"])


@torch.no_grad()
def predict_window(net: Simpl, log_dir: Path, start: int, focal: str,
                   static_map: ArgoverseStaticMap, pre: ArgoPreprocAV2,
                   severity: str, regime: str, tmp: Path) -> dict:
    scenario, win_ts = build_scenario(log_dir, start, focal)
    data, headers = pre.process(scenario.scenario_id, scenario, static_map)
    pkl = tmp / f"{scenario.scenario_id}.pkl"
    df = pd.DataFrame(data, columns=headers)
    df.to_pickle(pkl)

    ds = MaskedEvalDataset([str(pkl)], severity, regime, "native")
    sample = ds[0]
    post = net.post_process(net(net.pre_process(ds.collate_fn([sample]))))
    meta = {k: df[k].values[0] for k in ("ORIG", "ROT", "TRAJS")}

    traj = post["traj_pred"][0].cpu().numpy()          # [K, 60, 2] focal frame
    prob = post["prob_pred"][0].cpu().numpy()
    gt = np.asarray(sample["TRAJS"]["TRAJS_POS_FUT"][0])  # [60, 2] focal frame
    fde = np.linalg.norm(traj[:, -1] - gt[-1], axis=-1)

    focal_track = next(t for t in scenario.tracks if t.track_id == focal)
    hist = np.array([s.position for s in focal_track.object_states[:OBS]])
    # a masked history step is NOT observed by the model -- do not draw it as if
    # it were (the ghost-outline treatment of D-N1-14d is out of scope here)
    keep = np.asarray(sample["TRAJS"]["PAD_FLAGS"][0][:OBS]).astype(bool) \
        if severity != "S0" else np.ones(OBS, dtype=bool)

    return {
        "start": start,
        "t_now": int(win_ts[OBS - 1]),
        "traj_city": focal_to_city(traj, meta),
        "gt_city": focal_to_city(gt[None], meta)[0],
        "hist_city": hist[keep],
        "prob": prob,
        "minfde6": float(fde.min()),
        "n_hist_observed": int(keep.sum()),
    }


def render_clip(a, device: torch.device) -> dict:
    n_avail = n_windows(a.log_dir)
    # A focal that drops even one annotated frame breaks the window containing
    # it. Clip to where it is actually observed and SAY SO, rather than dying
    # 44 windows into a render: a track ending partway through a log is ordinary
    # data, not an error.
    n_focal = focal_windows(a.log_dir, a.focal)
    n_win = n_avail if a.clip_windows <= 0 else min(a.clip_windows, n_avail)
    if n_focal < n_win:
        print(f"[clip] focal observed for {n_focal}/{n_avail} windows — "
              f"truncating clip from {n_win} to {n_focal}")
        n_win = n_focal
    if n_win < 2:
        raise SystemExit(f"log supports {n_avail} windows and the focal survives "
                         f"{n_focal} — too short for a clip")

    static_map = ArgoverseStaticMap.from_map_dir(a.log_dir / "map", build_raster=True)
    cam = PinholeCamera.from_feather(a.log_dir, a.cam)
    pose_ts, quats, xyz = load_poses(a.log_dir)
    net, state = load_net(a.ckpt, device)
    pre = ArgoPreprocAV2(
        SimpleNamespace(obs_len=OBS, pred_len=FUT, debug=False, viz=False, mode="val"),
        verbose=False)

    print(f"[clip] {a.log_dir.name[:8]} · {a.cam} · focal {a.focal[:8]} · "
          f"{n_win}/{n_avail} windows · {a.ckpt_label} (epoch {state.get('epoch')}) · "
          f"severity {a.severity}")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wins = []
        for s in range(n_win):
            wins.append(predict_window(net, a.log_dir, s, a.focal, static_map,
                                       pre, a.severity, a.regime, tmp))
            if (s + 1) % 10 == 0 or s == n_win - 1:
                print(f"  predict {s + 1}/{n_win}  "
                      f"({(s + 1) / (time.time() - t0):.1f} win/s)", flush=True)
    t_pred = time.time() - t0

    # --- map camera frames onto the windows they belong to -------------------
    cam_dir = a.log_dir / "sensors" / "cameras" / a.cam
    imgs = sorted(cam_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {cam_dir} — fetch them with "
                         f"scripts/fetch_sensor_log.py --images --cam {a.cam}")
    stamps = np.array([int(p.stem) for p in imgs])
    t_lo, t_hi = wins[0]["t_now"], wins[-1]["t_now"]
    sel = np.flatnonzero((stamps >= t_lo) & (stamps <= t_hi))
    win_t = np.array([w["t_now"] for w in wins])
    print(f"[frames] {len(sel)} camera frames over "
          f"{(t_hi - t_lo) / 1e9:.2f} s of log time")

    # --- render ---------------------------------------------------------------
    probe = plt.imread(imgs[sel[0]])
    h, w = probe.shape[:2]
    fig = plt.figure(figsize=(w / 200, h / 200 + 0.55), dpi=a.dpi)
    ax = fig.add_axes([0, 0.075, 1, 0.925])
    cap = fig.add_axes([0, 0, 1, 0.075])
    cap.axis("off")
    cap.set_facecolor(HUD_BG)

    seen = {"hist": 0, "modes": 0, "gt": 0}
    frames, t1 = [], time.time()
    for n, i in enumerate(sel):
        ts = int(stamps[i])
        wi = nearest_index(win_t, ts)
        wd = wins[wi]
        city_SE3_ego = pose_at(ts, pose_ts, quats, xyz)

        ax.clear()
        ax.imshow(plt.imread(imgs[i]))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

        def draw(xy, _ego=city_SE3_ego, **kw) -> bool:
            """True if the polyline actually reached the image."""
            if len(xy) < 2:
                return False
            uv, ok = project_city_to_img(xy, ground_height(static_map, xy),
                                         cam, _ego)
            if ok.sum() < 2:
                return False
            ax.plot(uv[ok, 0], uv[ok, 1], solid_capstyle="round", **kw)
            return True

        vis_hist = draw(wd["hist_city"], ls="-", lw=2.6, color=HIST_COLOR, zorder=4)
        vis_modes = sum(
            draw(wd["traj_city"][m], ls="-", lw=3.0, color=PRED_COLOR, zorder=3,
                 # prob is float32, so the max mode's own ratio can land at
                 # 1.0000001 and matplotlib rejects alpha > 1
                 alpha=float(np.clip(0.28 + 0.72 * wd["prob"][m] / wd["prob"].max(),
                                     0.0, 1.0)))
            for m in np.argsort(-wd["prob"]))
        vis_gt = draw(wd["gt_city"], ls="--", lw=2.4, color=GT_COLOR, zorder=5)
        seen["hist"] += int(vis_hist)
        seen["modes"] += vis_modes
        seen["gt"] += int(vis_gt)

        # Legend is built from fixed proxy artists, NOT from what happened to be
        # in frustum this frame -- otherwise it flickers as elements leave the
        # camera, and a missing entry reads as "the model produced nothing".
        ax.legend(handles=LEGEND, loc="upper left", framealpha=0.72, fontsize=8)
        ax.text(0.995, 0.012,
                f"{a.ckpt_label} · {a.cam} · win {wd['start']:03d}/{n_win - 1} · "
                f"t={(ts - t_lo) / 1e9:5.2f}s · minFDE6 {wd['minfde6']:5.2f} m · "
                f"{a.severity}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=HUD_FG, family="monospace",
                bbox={"facecolor": HUD_BG, "alpha": 0.75, "pad": 3, "edgecolor": "none"})
        cap.clear()
        cap.axis("off")
        cap.set_facecolor(HUD_BG)
        cap.text(0.5, 0.5, HONESTY, ha="center", va="center", fontsize=9,
                 color=HUD_FG, transform=cap.transAxes)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if (n + 1) % 20 == 0 or n == len(sel) - 1:
            print(f"  render {n + 1}/{len(sel)}  "
                  f"({(n + 1) / (time.time() - t1):.1f} fps)", flush=True)
    plt.close(fig)
    t_render = time.time() - t1

    # --- write ----------------------------------------------------------------
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{a.log_dir.name[:8]}_{a.cam}_{a.focal[:8]}_"
            f"{a.ckpt_label.replace('-', '')}_{a.severity}")
    mp4 = a.out_dir / f"{stem}.mp4"
    imageio.mimwrite(mp4, frames, fps=a.fps, quality=8, macro_block_size=1)
    outs = [mp4]
    if a.gif:
        gif = a.out_dir / f"{stem}.gif"
        imageio.mimwrite(gif, frames[::2], duration=2.0 / a.fps, loop=0)
        outs.append(gif)
    still = a.out_dir / "stills" / f"{stem}_mid.png"
    still.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(still, frames[len(frames) // 2])
    outs.append(still)

    meta = {
        "log": a.log_dir.name, "cam": a.cam, "focal": a.focal,
        "ckpt": str(a.ckpt), "ckpt_label": a.ckpt_label,
        "ckpt_epoch": state.get("epoch"), "severity": a.severity,
        "regime": a.regime, "windows": n_win, "windows_available": n_avail,
        "frames": len(frames), "fps": a.fps,
        "clip_seconds": round(len(frames) / a.fps, 2),
        "log_seconds_covered": round((t_hi - t_lo) / 1e9, 2),
        "minfde6_mean": round(float(np.mean([x["minfde6"] for x in wins])), 3),
        "minfde6_first": round(wins[0]["minfde6"], 3),
        "minfde6_last": round(wins[-1]["minfde6"], 3),
        "predict_seconds": round(t_pred, 1), "render_seconds": round(t_render, 1),
        # per-frame in-frustum counts: a clip where the fan is rarely visible is
        # a bad take, and this is how you tell without watching it
        "frames_with_history": seen["hist"],
        "frames_with_gt": seen["gt"],
        "mean_modes_visible": round(seen["modes"] / max(len(frames), 1), 2),
        "honesty_label": HONESTY,
        "note": "AV2-derived pixels — local/untracked (D-N1-14d precedent)",
    }
    (a.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(frames)} frames, {meta['clip_seconds']}s @ {a.fps}fps · "
          f"minFDE6 mean {meta['minfde6_mean']} m · "
          f"predict {t_pred:.0f}s + render {t_render:.0f}s")
    print(f"[visible] modes {meta['mean_modes_visible']}/6 per frame · "
          f"history {seen['hist']}/{len(frames)} · GT {seen['gt']}/{len(frames)}")
    for o in outs:
        print(f"  -> {o}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=Path, default=None,
                    help="manifest of takes to render (results/rgb_demo_takes.json); "
                         "with --ckpt this makes a full re-render one command")
    ap.add_argument("--data-root", type=Path,
                    default=REPO / "data" / "av2_sensor" / "val")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--focal", default=None, help="track_uuid, from pick_rgb_demo_scene")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ckpt-label", default="C1-local")
    ap.add_argument("--clip-windows", type=int, default=0,
                    help="0 = every window the log supports (~48 => ~4.8 s)")
    ap.add_argument("--severity", default="S0", help="S0..S4 occlusion severity")
    ap.add_argument("--regime", default="R-A")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "videos" / "rgb")
    a = ap.parse_args()

    # Environment probe before compute (refactored_method §5)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        raise SystemExit("[STOP] CPU fallback on GPU work (refactored_method §5)")
    print(f"[env] device={device} {torch.cuda.get_device_name(0)}")

    if a.takes:
        spec = json.loads(a.takes.read_text(encoding="utf-8"))
        metas = []
        for t in spec["takes"]:
            a.log_dir = a.data_root / t["log"]
            a.cam, a.focal = t["cam"], t["focal"]
            print(f"\n=== take {t['name']}")
            metas.append({"take": t["name"], "why": t["why"], **render_clip(a, device)})
        summary = a.out_dir / f"takes_{a.ckpt_label.replace('-', '')}.json"
        summary.write_text(json.dumps(metas, indent=2), encoding="utf-8")
        print(f"\n[all] {len(metas)} takes -> {summary}")
        return

    missing = [f for f in ("log_dir", "cam", "focal") if getattr(a, f) is None]
    if missing:
        raise SystemExit(f"single-take mode needs --{', --'.join(missing)} "
                         f"(or pass --takes)")
    render_clip(a, device)


if __name__ == "__main__":
    main()
