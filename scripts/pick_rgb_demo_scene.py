"""Rank (log, camera, focal-agent) candidates for the RGB overlay demo using ONLY
the ~2.3 MB metadata slice of each AV2 sensor log (D-N1-15).

Camera pixels are ~96-129 MB per (log, camera) but annotations + ego pose +
calibration + map are ~2.3 MB, and CALIBRATION IS IN THE METADATA -- so "is this
agent big and visible in camera X" is answerable before a single JPEG is fetched.
Scout on metadata, pull pixels for the winners only.

Scoring, per the D-N1-15 scene brief (owner): dense-ish traffic, a visibly large
focal agent, and occlusion-relevant geometry. Components are reported separately
rather than collapsed into one opaque number, so a scene can be argued with:

  vis_span  frac of the 6 s future inside the frustum x its pixel span
            -- a fully-visible agent 90 m down the road is a 4 px smudge
  n_moving  fully-observed moving vehicles in the window       (traffic density)
  n_parked  stationary vehicles within PARKED_RADIUS of the focal path
            (parked rows are what actually occlude in urban logs)
  isect     focal path passes within ISECT_RADIUS of an intersection lane

Candidates are scored for the ROLLING clip, i.e. the focal must be fully observed
in EVERY window of the clip, not just one -- that is what makes it renderable as
video rather than a still.

Usage:
  python scripts/pick_rgb_demo_scene.py --root data/av2_sensor/val --top 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from av2.geometry.camera.pinhole_camera import PinholeCamera  # noqa: E402
from av2.map.map_api import ArgoverseStaticMap  # noqa: E402

from otraj.sensor import (  # noqa: E402
    OBS,
    VEHICLE_CATEGORIES,
    ground_height,
    load_poses,
    pose_at,
    project_city_to_img,
)
from otraj.sensor.scenario import WINDOW, annotation_timestamps  # noqa: E402

RING_CAMS = ("ring_front_center", "ring_front_left", "ring_front_right",
             "ring_side_left", "ring_side_right", "ring_rear_left",
             "ring_rear_right")
PARKED_RADIUS_M = 30.0
PARKED_TRAVEL_M = 1.0
ISECT_RADIUS_M = 25.0
MIN_TRAVEL_M = 8.0


def _tracks_city(log_dir: Path, win_ts: np.ndarray, city_SE3_ego: dict,
                 ) -> dict[str, tuple[str, np.ndarray]]:
    """{uuid: (category, city-frame xy [T, 2])} for one window."""
    ann = pd.read_feather(log_dir / "annotations.feather")
    ann = ann[ann["timestamp_ns"].isin(win_ts)]
    out: dict[str, tuple[str, np.ndarray]] = {}
    for uuid, grp in ann.groupby("track_uuid", sort=True):
        grp = grp.sort_values("timestamp_ns")
        pos = np.stack([
            city_SE3_ego[int(t)].transform_point_cloud(np.array([[x, y, 0.0]]))[0][:2]
            for t, x, y in zip(grp["timestamp_ns"], grp["tx_m"], grp["ty_m"],
                               strict=True)])
        out[str(uuid)] = (str(grp["category"].iloc[0]), pos)
    return out


def score_log(log_dir: Path, cams: tuple[str, ...], clip_windows: int) -> list[dict]:
    """One row per (camera, focal) that survives the whole rolling clip."""
    all_ts = annotation_timestamps(log_dir)
    n_win = len(all_ts) - WINDOW + 1
    if clip_windows <= 0:
        clip_windows = n_win
    if n_win < clip_windows:
        return []
    # the clip runs windows [0, clip_windows); score it at its midpoint, where the
    # focal's visibility is most representative of the clip as a whole
    mid = clip_windows // 2
    win_ts = all_ts[mid: mid + WINDOW]

    pose_ts, quats, xyz = load_poses(log_dir)
    city_SE3_ego = {int(t): pose_at(int(t), pose_ts, quats, xyz) for t in win_ts}
    static_map = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)
    cameras = {c: PinholeCamera.from_feather(log_dir, c) for c in cams}
    ego_now = city_SE3_ego[int(win_ts[OBS - 1])]

    tracks = _tracks_city(log_dir, win_ts, city_SE3_ego)
    full = {u: (c, p) for u, (c, p) in tracks.items() if len(p) == WINDOW}
    veh = {u: (c, p) for u, (c, p) in full.items() if c in VEHICLE_CATEGORIES}
    travel = {u: float(np.linalg.norm(p[-1] - p[0])) for u, (_, p) in veh.items()}
    moving = {u for u, d in travel.items() if d >= MIN_TRAVEL_M}
    parked = np.array([veh[u][1][OBS - 1] for u, d in travel.items()
                       if d < PARKED_TRAVEL_M]) if veh else np.empty((0, 2))

    isect_ctrs = np.array([
        static_map.get_lane_segment_centerline(lid)[:, :2].mean(0)
        for lid, lane in static_map.vector_lane_segments.items()
        if lane.is_intersection]) if static_map.vector_lane_segments else np.empty((0, 2))

    # A focal must be fully observed in EVERY window of the clip, else the video
    # breaks mid-clip. Windows overlap heavily, so their union is one contiguous
    # span [0, clip_windows + WINDOW) -- checking that span is equivalent and
    # costs one pass instead of `clip_windows` passes.
    span_ts = set(int(t) for t in all_ts[: clip_windows - 1 + WINDOW])
    ann_all = pd.read_feather(log_dir / "annotations.feather")
    seen = (ann_all[ann_all["timestamp_ns"].isin(span_ts)]
            .groupby("track_uuid")["timestamp_ns"].nunique())
    survives = {u for u in moving if int(seen.get(u, 0)) == len(span_ts)}

    rows: list[dict] = []
    for uuid in sorted(survives):
        path = veh[uuid][1]
        fut = path[OBS:]
        z = ground_height(static_map, fut)
        n_parked = int((np.linalg.norm(parked[:, None] - path[None], axis=-1)
                        .min(1) < PARKED_RADIUS_M).sum()) if len(parked) else 0
        isect = bool(len(isect_ctrs) and
                     np.linalg.norm(isect_ctrs[:, None] - path[None], axis=-1)
                     .min() < ISECT_RADIUS_M)
        hist = path[:OBS]
        z_hist = ground_height(static_map, hist)
        for cname, cam in cameras.items():
            uv, valid = project_city_to_img(fut, z, cam, ego_now)
            if valid.sum() < 2:
                continue
            span = float(np.linalg.norm(uv[valid].max(0) - uv[valid].min(0)))
            # History visibility is scored SEPARATELY and is not implied by future
            # visibility: an agent driving away from the ego has a fully visible
            # future and a history entirely behind the camera. Scoring only the
            # future silently selects scenes where the history track never appears.
            _, valid_h = project_city_to_img(hist, z_hist, cam, ego_now)
            rows.append({
                "log": log_dir.name, "cam": cname, "focal": uuid,
                "n_windows": int(n_win), "travel_m": round(travel[uuid], 1),
                "frac_visible": round(float(valid.mean()), 3),
                "frac_visible_hist": round(float(valid_h.mean()), 3),
                "span_px": round(span, 1),
                "vis_span": round(float(valid.mean()) * span, 1),
                "n_moving": len(moving), "n_parked": n_parked, "isect": isect,
            })
    return rows


def composite(r: dict) -> float:
    """Visual size is the base; density, occlusion geometry and history
    visibility are multipliers.

    Deliberately blunt and reported alongside its components -- the point is to
    surface a shortlist a human picks from, not to automate the curation.
    """
    return (r["vis_span"]
            * (1.0 + 0.04 * min(r["n_moving"], 15))
            * (1.0 + 0.03 * min(r["n_parked"], 15))
            * (1.25 if r["isect"] else 1.0)
            # the overlay spec is fan + GT + history; a scene whose history never
            # reaches the image renders a legend entry that never appears
            * (0.4 + 0.6 * r["frac_visible_hist"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cams", default=",".join(RING_CAMS))
    ap.add_argument("--clip-windows", type=int, default=0,
                    help="rolling windows the focal must survive; 0 = every window "
                         "the log supports, which is what render_rgb_demo defaults to")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rgb_demo_scenes.json")
    a = ap.parse_args()

    cams = tuple(a.cams.split(","))
    logs = sorted(p for p in a.root.iterdir() if (p / "annotations.feather").exists())
    rows: list[dict] = []
    for log_dir in logs:
        try:
            r = score_log(log_dir, cams, a.clip_windows)
        except Exception as exc:
            print(f"[skip] {log_dir.name[:8]}: {type(exc).__name__}: {exc}")
            continue
        rows += r
        print(f"[{log_dir.name[:8]}] {len(r)} candidates", flush=True)

    for r in rows:
        r["score"] = round(composite(r), 1)
    rows.sort(key=lambda r: -r["score"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    hdr = (f"{'score':>8} {'log':10} {'cam':18} {'vis*span':>9} {'visF':>5} "
           f"{'visH':>5} {'span':>7} {'trav':>6} {'mov':>4} {'park':>5} "
           f"{'isect':>6}  focal")
    print(f"\n{len(rows)} candidates over {len(logs)} logs — top {a.top}:")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[: a.top]:
        print(f"{r['score']:8.1f} {r['log'][:8]:10} {r['cam']:18} {r['vis_span']:9.1f} "
              f"{r['frac_visible']:5.2f} {r['frac_visible_hist']:5.2f} "
              f"{r['span_px']:7.1f} {r['travel_m']:6.1f} "
              f"{r['n_moving']:4d} {r['n_parked']:5d} {str(r['isect']):>6}  "
              f"{r['focal'][:8]}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
