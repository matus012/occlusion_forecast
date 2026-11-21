"""Synthesize an `ArgoverseScenario` from an AV2 *sensor* log (D-N1-15).

Sensor annotations ride the lidar clock at 10 Hz -- exactly the motion-forecasting
rate -- and a ~15 s log carries ~157 annotated frames against the 110 (5 s history
+ 6 s future) a scenario needs. Windows are therefore carved directly out of the
log, converted ego -> city, and handed to the unmodified SIMPL preprocessor.

FIVE NAMED DISTRIBUTION SHIFTS (reports/rgb_demo_feasibility.md §2). These are
design decisions, not published correspondences, and every downstream surface
must stay honest about them:

1. `velocity` is absent from sensor annotations -> finite-differenced here.
   SIMPL consumes `ObjectState.velocity` directly (av2_preprocess.py:191), so a
   value must exist; MF's is dataset-provided and smoother.
2. `category` (FOCAL/SCORED/UNSCORED/FRAGMENT) is absent -> assigned by
   observation completeness. MF's SCORED tracks are curated for PREDICTABILITY;
   ours are merely fully observed. Largest single shift.
3. The `AV` track is not an annotation (SIMPL asserts on it) -> synthesized from
   ego pose.
4. 30 sensor classes -> 10 MF ObjectTypes -> 7 SIMPL one-hot slots, via the
   hand-authored map below.
5. MF scenarios are MINED for interesting focal agents; a sensor window is not.
   Focal choice is ours (see scripts/pick_rgb_demo_scene.py).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3

OBS, FUT = 50, 60
WINDOW = OBS + FUT
DT = 0.1  # AV2 sensor annotations are at the 10 Hz lidar rate == the MF rate

# Shift #4. No published correspondence exists between the sensor taxonomy and
# the forecasting one; this is a judgement call, recorded so it can be argued with.
CATEGORY_TO_OBJECT_TYPE: dict[str, ObjectType] = {
    "REGULAR_VEHICLE": ObjectType.VEHICLE,
    "LARGE_VEHICLE": ObjectType.VEHICLE,
    "TRUCK": ObjectType.VEHICLE,
    "TRUCK_CAB": ObjectType.VEHICLE,
    "BOX_TRUCK": ObjectType.VEHICLE,
    "VEHICULAR_TRAILER": ObjectType.VEHICLE,
    "RAILED_VEHICLE": ObjectType.VEHICLE,
    "BUS": ObjectType.BUS,
    "SCHOOL_BUS": ObjectType.BUS,
    "ARTICULATED_BUS": ObjectType.BUS,
    "PEDESTRIAN": ObjectType.PEDESTRIAN,
    "OFFICIAL_SIGNALER": ObjectType.PEDESTRIAN,
    "WHEELCHAIR": ObjectType.PEDESTRIAN,
    "STROLLER": ObjectType.PEDESTRIAN,
    "BICYCLIST": ObjectType.CYCLIST,
    "MOTORCYCLIST": ObjectType.MOTORCYCLIST,
    # unridden two-wheelers and street furniture are static in MF terms
    "BICYCLE": ObjectType.STATIC,
    "MOTORCYCLE": ObjectType.STATIC,
    "SIGN": ObjectType.STATIC,
    "STOP_SIGN": ObjectType.STATIC,
    "BOLLARD": ObjectType.STATIC,
    "MESSAGE_BOARD_TRAILER": ObjectType.STATIC,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": ObjectType.STATIC,
    "TRAFFIC_LIGHT_TRAILER": ObjectType.STATIC,
    "CONSTRUCTION_CONE": ObjectType.CONSTRUCTION,
    "CONSTRUCTION_BARREL": ObjectType.CONSTRUCTION,
    "ANIMAL": ObjectType.UNKNOWN,
    "DOG": ObjectType.UNKNOWN,
    "WHEELED_DEVICE": ObjectType.UNKNOWN,
    "WHEELED_RIDER": ObjectType.UNKNOWN,
}

# Sensor categories that map to SIMPL's vehicle/bus slots -- the only sensible
# focal candidates for a trajectory demo.
VEHICLE_CATEGORIES = frozenset(
    c for c, t in CATEGORY_TO_OBJECT_TYPE.items()
    if t in (ObjectType.VEHICLE, ObjectType.BUS)
)


# --------------------------------------------------------------------------
# ego pose
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _read_feather(path: str) -> pd.DataFrame:
    """A rolling render rebuilds ~48 windows from the SAME log, so the two
    per-log tables would otherwise be re-parsed 48 times each. Cached by path;
    callers must treat the result as read-only."""
    return pd.read_feather(path)


def load_annotations(log_dir: Path) -> pd.DataFrame:
    return _read_feather(str(log_dir / "annotations.feather"))


def load_poses(log_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """city_SE3_egovehicle, sampled at ~170 Hz: (timestamps, quats wxyz, xyz)."""
    p = _read_feather(str(log_dir / "city_SE3_egovehicle.feather")).sort_values("timestamp_ns")
    return (p["timestamp_ns"].to_numpy(),
            p[["qw", "qx", "qy", "qz"]].to_numpy(),
            p[["tx_m", "ty_m", "tz_m"]].to_numpy())


def pose_at(ts: int, pose_ts: np.ndarray, quats: np.ndarray, xyz: np.ndarray) -> SE3:
    """Nearest-neighbour ego pose. The pose stream runs ~170 Hz, so worst-case
    error is under 3 ms of ego motion (<4 cm at 50 km/h) -- below the width of a
    rendered line, hence no interpolation."""
    i = int(np.argmin(np.abs(pose_ts - ts)))
    return SE3(rotation=quat_to_mat(quats[i]), translation=xyz[i])


def yaw_of(rot: np.ndarray) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------
def annotation_timestamps(log_dir: Path) -> np.ndarray:
    return np.sort(load_annotations(log_dir)["timestamp_ns"].unique())


def n_windows(log_dir: Path) -> int:
    """How many 110-step windows a log supports.

    This is the hard limit on clip length: ~157 annotated frames - 110 + 1 = 48
    windows = 4.8 s of ROLLING-PREDICTION video per log, whatever the camera
    frame rate. Owner decision (D-N1-15): accept ~5 s clips, do not engineer
    around it.
    """
    return max(0, len(annotation_timestamps(log_dir)) - WINDOW + 1)


def window_timestamps(log_dir: Path, start: int) -> np.ndarray:
    all_ts = annotation_timestamps(log_dir)
    if start + WINDOW > len(all_ts):
        raise ValueError(f"log has {len(all_ts)} annotated frames, need "
                         f"{WINDOW} from index {start}")
    return all_ts[start: start + WINDOW]


def focal_windows(log_dir: Path, focal_uuid: str) -> int:
    """How many LEADING windows `focal_uuid` is fully observed in.

    A track that drops even one annotated frame breaks the window containing it,
    so a rolling clip can only run up to that point. Callers should clip to this
    rather than crash mid-render: a focal vanishing partway through a log is
    ordinary data (occlusion, exit from range), not an error.

    Windows overlap, so window k needs the focal present across the contiguous
    span all_ts[k : k+WINDOW]; the answer is the length of the focal's
    uninterrupted run from index 0, minus WINDOW, plus 1.
    """
    all_ts = annotation_timestamps(log_dir)
    ann = load_annotations(log_dir)
    seen = set(ann.loc[ann["track_uuid"] == focal_uuid, "timestamp_ns"]
               .astype("int64").tolist())
    run = 0
    for t in all_ts:
        if int(t) not in seen:
            break
        run += 1
    return max(0, run - WINDOW + 1)


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def _states(steps: list[int], pos: np.ndarray, head: list[float]) -> list[ObjectState]:
    # shift #1: velocity is finite-differenced, it does not exist in the source
    vel = np.gradient(pos, DT, axis=0) if len(pos) > 1 else np.zeros_like(pos)
    return [ObjectState(observed=s < OBS, timestep=s,
                        position=(float(pos[i, 0]), float(pos[i, 1])),
                        heading=float(head[i]),
                        velocity=(float(vel[i, 0]), float(vel[i, 1])))
            for i, s in enumerate(steps)]


def city_name_of(log_dir: Path) -> str:
    """Sensor map filenames carry the city: log_map_archive_<id>____PIT_city_*.json"""
    f = next((log_dir / "map").glob("log_map_archive_*.json")).stem
    return f.split("____")[1].split("_city")[0]


def build_scenario(log_dir: Path, start: int, focal_uuid: str,
                   ) -> tuple[ArgoverseScenario, np.ndarray]:
    """Carve a 110-step (11 s) MF-shaped scenario out of a sensor log.

    `focal_uuid` is required: focal choice is a demo-curation decision made once
    per clip (shift #5), not something to re-decide per frame -- a rolling render
    must keep the same focal agent across every window or the clip is incoherent.

    Returns the scenario and the window's annotation timestamps.
    """
    ann = load_annotations(log_dir)
    pose_ts, quats, xyz = load_poses(log_dir)

    win_ts = window_timestamps(log_dir, start)
    ts_to_step = {int(t): i for i, t in enumerate(win_ts)}
    ann = ann[ann["timestamp_ns"].isin(win_ts)]
    city_SE3_ego = {int(t): pose_at(int(t), pose_ts, quats, xyz) for t in win_ts}

    tracks: list[Track] = []
    for uuid, grp in ann.groupby("track_uuid", sort=True):
        grp = grp.sort_values("timestamp_ns")
        steps, pos, head = [], [], []
        for row in grp.itertuples():
            se3 = city_SE3_ego[int(row.timestamp_ns)]
            # shift #5's mechanics: cuboids are ego-frame, MF tracks are city-frame
            pt = se3.transform_point_cloud(np.array([[row.tx_m, row.ty_m, 0.0]]))[0]
            yaw_ego = yaw_of(quat_to_mat(np.array([row.qw, row.qx, row.qy, row.qz])))
            steps.append(ts_to_step[int(row.timestamp_ns)])
            pos.append(pt[:2])
            head.append(yaw_ego + yaw_of(se3.rotation))
        tracks.append(Track(
            track_id=str(uuid),
            object_states=_states(steps, np.asarray(pos), head),
            object_type=CATEGORY_TO_OBJECT_TYPE.get(
                str(grp["category"].iloc[0]), ObjectType.UNKNOWN),
            # shift #2: completeness stands in for MF's predictability curation
            category=(TrackCategory.SCORED_TRACK if len(steps) == WINDOW
                      else TrackCategory.UNSCORED_TRACK),
        ))

    # shift #3: the AV is a track in MF (track_id 'AV'); SIMPL asserts on it
    ego_pos = np.stack([city_SE3_ego[int(t)].translation[:2] for t in win_ts])
    ego_yaw = [yaw_of(city_SE3_ego[int(t)].rotation) for t in win_ts]
    tracks.append(Track(track_id="AV",
                        object_states=_states(list(range(len(win_ts))), ego_pos, ego_yaw),
                        object_type=ObjectType.VEHICLE,
                        category=TrackCategory.UNSCORED_TRACK))

    hits = [t for t in tracks if t.track_id == focal_uuid]
    if not hits:
        raise ValueError(f"focal {focal_uuid} absent from window start={start}")
    if len(hits[0].object_states) != WINDOW:
        raise ValueError(f"focal {focal_uuid} not fully observed in window "
                         f"start={start} ({len(hits[0].object_states)}/{WINDOW})")
    hits[0].category = TrackCategory.FOCAL_TRACK

    scenario = ArgoverseScenario(
        scenario_id=f"{log_dir.name[:8]}-w{start:03d}",
        timestamps_ns=win_ts,
        tracks=tracks,
        focal_track_id=focal_uuid,
        city_name=city_name_of(log_dir),
        map_id=None,
        slice_id=None,
    )
    return scenario, win_ts
