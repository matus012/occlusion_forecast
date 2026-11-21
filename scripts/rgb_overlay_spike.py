"""Single-frame RGB overlay + TRANSFORM PROOF (D-N1-15 spike tool).

Kept after the demo build for the one job the video renderer does not do:
`--boxes` projects the annotated 3D CUBOIDS through the identical
city -> ego@cam -> image path used for predictions. That check is independent of
both the model and the trajectory code, which is the point -- a trajectory drawn
from the same annotations that produced it looks plausible even when the
projection is wrong. If the boxes land on the vehicles, the chain is right.

Synthesis, projection and the camera model all come from src/otraj/sensor/;
this script is the thin single-frame driver. For clips use
scripts/render_rgb_demo.py.

Outputs are AV2-derived pixels -> reports/rgb_spike/, local and untracked.

Usage:
  python scripts/rgb_overlay_spike.py --log-dir data/av2_sensor/val/<id> \
      --cam ring_front_left --boxes
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SIMPL"))
sys.path.insert(0, str(REPO / "third_party" / "SIMPL" / "data_av2"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import _simpl_compat  # noqa: E402, F401  (py3.11 shims; must precede simpl imports)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from av2.datasets.motion_forecasting.data_schema import ObjectType, Track  # noqa: E402
from av2.geometry.camera.pinhole_camera import PinholeCamera  # noqa: E402
from av2.map.map_api import ArgoverseStaticMap  # noqa: E402
from av2_preprocess import ArgoPreprocAV2  # noqa: E402
from eval_local import MaskedEvalDataset  # noqa: E402
from render_rgb_demo import focal_to_city, load_net  # noqa: E402

from otraj.sensor import (  # noqa: E402
    FUT,
    OBS,
    build_scenario,
    ground_height,
    load_annotations,
    load_poses,
    pose_at,
    project_city_to_img,
)
from otraj.sensor.projection import (  # noqa: E402
    BOX_EDGES,
    cuboid_corners_city,
    nearest_index,
)
from otraj.sensor.scenario import WINDOW  # noqa: E402


def pick_focal(tracks: list[Track], score) -> str:
    """Fully-observed moving vehicle, ranked by frustum coverage then pixel span.

    Without the visibility term the natural pick is the fastest agent in the log,
    which is routinely behind or beside the ego; without the span term the pick
    is a fully-visible agent 90 m down the road, i.e. a 4 px smudge. Both were
    observed during the spike. For clips use scripts/pick_rgb_demo_scene.py,
    which scores this over whole logs from metadata alone.
    """
    cands = []
    for t in tracks:
        if t.track_id == "AV" or t.object_type != ObjectType.VEHICLE:
            continue
        if len(t.object_states) != WINDOW:
            continue
        p = np.array([s.position for s in t.object_states])
        d = float(np.linalg.norm(p[-1] - p[0]))
        if d < 2.0:  # parked
            continue
        frac, span = score(p[OBS:])
        cands.append((frac, span, d, t.track_id))
    if not cands:
        raise SystemExit("no fully-observed moving vehicle track in this window")
    cands.sort(reverse=True)
    frac, span, d, best = cands[0]
    print(f"[focal] {best}  travel={d:.1f} m  future-in-frustum={frac:.0%}  "
          f"span={span:.0f} px  ({len(cands)} candidates)")
    return best


def draw_annotation_boxes(ax, log_dir: Path, t_ann: int, cam: PinholeCamera,
                          city_SE3_ego_cam, city_SE3_ego_ann) -> int:
    ann = load_annotations(log_dir)
    ann = ann[ann["timestamp_ns"] == t_ann]
    n = 0
    for row in ann.itertuples():
        pts_city = cuboid_corners_city(row, city_SE3_ego_ann)
        pts_ego = city_SE3_ego_cam.inverse().transform_point_cloud(pts_city)
        uv, _, valid = cam.project_ego_to_img(pts_ego)
        if valid.sum() < 8:  # only fully-visible boxes, keeps the check honest
            continue
        n += 1
        for i, j in BOX_EDGES:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], "-", lw=1.2,
                    color="#ff5964", alpha=0.9, zorder=6)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "checkpoints" / "local" / "c1_seed42.tar")
    ap.add_argument("--ckpt-label", default="C1-local")
    ap.add_argument("--cam", default="ring_front_center")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--focal", default=None, help="track_uuid; default = pick_focal")
    ap.add_argument("--boxes", action="store_true",
                    help="overlay annotated 3D cuboids (transform-chain proof)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports" / "rgb_spike" / "overlay.png")
    args = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    print(f"[env] device={device}  cuda={torch.cuda.is_available()}")

    static_map = ArgoverseStaticMap.from_map_dir(args.log_dir / "map", build_raster=True)
    pose_ts, quats, xyz = load_poses(args.log_dir)
    cam = PinholeCamera.from_feather(args.log_dir, args.cam)

    # the camera rig is fixed for the log, so it resolves before the scenario
    from otraj.sensor import window_timestamps
    win_ts = window_timestamps(args.log_dir, args.start)
    t_now = int(win_ts[OBS - 1])
    cam_dir = args.log_dir / "sensors" / "cameras" / args.cam
    imgs = sorted(cam_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {cam_dir}")
    stamps = np.array([int(p.stem) for p in imgs])
    idx = nearest_index(stamps, t_now)
    img_path, cam_ts = imgs[idx], int(stamps[idx])
    print(f"[cam] {img_path.name}  dt={abs(cam_ts - t_now) / 1e6:.1f} ms · "
          f"{args.cam} {cam.width_px}x{cam.height_px}")
    city_SE3_ego = pose_at(cam_ts, pose_ts, quats, xyz)

    def score(xy: np.ndarray) -> tuple[float, float]:
        uv, valid = project_city_to_img(xy, ground_height(static_map, xy),
                                        cam, city_SE3_ego)
        if valid.sum() < 2:
            return 0.0, 0.0
        return float(valid.mean()), float(
            np.linalg.norm(uv[valid].max(0) - uv[valid].min(0)))

    focal = args.focal
    if focal is None:
        probe, _ = build_scenario(args.log_dir, args.start,
                                  _any_full_vehicle(args.log_dir, args.start))
        focal = pick_focal(probe.tracks, score)
    scenario, win_ts = build_scenario(args.log_dir, args.start, focal)
    print(f"[scenario] {scenario.scenario_id}  city={scenario.city_name}  "
          f"tracks={len(scenario.tracks)}  focal={focal[:8]}")

    pre = ArgoPreprocAV2(
        SimpleNamespace(obs_len=OBS, pred_len=FUT, debug=False, viz=False, mode="val"),
        verbose=False)
    data, headers = pre.process(scenario.scenario_id, scenario, static_map)
    net, state = load_net(args.ckpt, device)
    with tempfile.TemporaryDirectory() as td:
        pkl = Path(td) / f"{scenario.scenario_id}.pkl"
        df = pd.DataFrame(data, columns=headers)
        df.to_pickle(pkl)
        ds = MaskedEvalDataset([str(pkl)], "S0", "R-A", "native")
        sample = ds[0]
        with torch.no_grad():
            post = net.post_process(net(net.pre_process(ds.collate_fn([sample]))))
        meta = {k: df[k].values[0] for k in ("ORIG", "ROT", "TRAJS")}
    traj = post["traj_pred"][0].cpu().numpy()
    prob = post["prob_pred"][0].cpu().numpy()
    gt = np.asarray(sample["TRAJS"]["TRAJS_POS_FUT"][0])
    print(f"[simpl] {args.ckpt_label} epoch={state.get('epoch')}  "
          f"probs={np.round(prob, 3)}")

    traj_city = focal_to_city(traj, meta)
    gt_city = focal_to_city(gt[None], meta)[0]
    focal_track = next(t for t in scenario.tracks if t.track_id == focal)
    hist_city = np.array([s.position for s in focal_track.object_states[:OBS]])

    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(9, 12), dpi=110)
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")

    def draw(xy, **kw) -> bool:
        uv, ok = project_city_to_img(xy, ground_height(static_map, xy),
                                     cam, city_SE3_ego)
        if ok.sum() < 2:
            return False
        ax.plot(uv[ok, 0], uv[ok, 1], solid_capstyle="round", **kw)
        return True

    n_drawn = sum(
        draw(traj_city[m], ls="-", lw=3.0, color="#ffb020", zorder=3,
             alpha=float(0.30 + 0.70 * prob[m] / prob.max()),
             label="prediction (6 modes)" if m == int(np.argmax(prob)) else None)
        for m in np.argsort(-prob))
    draw(gt_city, ls="--", lw=2.4, color="#3fe07f", zorder=4,
         label="ground-truth future")
    draw(hist_city, ls="-", lw=2.4, color="#39c0ed", zorder=4,
         label="observed history (5 s)")

    n_box = 0
    if args.boxes:
        n_box = draw_annotation_boxes(ax, args.log_dir, t_now, cam, city_SE3_ego,
                                      pose_at(t_now, pose_ts, quats, xyz))
        print(f"[boxes] {n_box} fully-visible annotated cuboids projected")

    ax.set_title(
        f"{args.ckpt_label} on synthesized AV2-sensor scenario {scenario.scenario_id}\n"
        f"{args.cam} · focal {focal[:8]} · {n_drawn}/{len(prob)} modes in frustum"
        + (f" · {n_box} cuboids (transform proof)" if args.boxes else "") + "\n"
        "under-trained local checkpoint — pipeline demonstration; "
        "full-scale results pending HPC",
        color="#111", fontsize=10)
    ax.legend(loc="lower left", framealpha=0.75, fontsize=9).set_zorder(5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[out] {args.out}  ({n_drawn} modes drawn)")


def _any_full_vehicle(log_dir: Path, start: int) -> str:
    """Any fully-observed track, purely to get a scenario built so pick_focal can
    rank its tracks; the real focal replaces it immediately."""
    from otraj.sensor import VEHICLE_CATEGORIES, window_timestamps

    ts = set(int(t) for t in window_timestamps(log_dir, start))
    ann = load_annotations(log_dir)
    ann = ann[ann["timestamp_ns"].isin(ts)]
    for uuid, grp in ann.groupby("track_uuid", sort=True):
        if len(grp) == WINDOW and str(grp["category"].iloc[0]) in VEHICLE_CATEGORIES:
            return str(uuid)
    raise SystemExit("no fully-observed vehicle in this window")


if __name__ == "__main__":
    main()
