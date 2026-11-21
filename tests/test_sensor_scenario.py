"""Tests for the AV2-sensor -> ArgoverseScenario synthesis (D-N1-15).

The synthesis is the only new load-bearing code in the RGB demo: everything
downstream (ArgoPreprocAV2 -> SIMPL -> masking -> metrics) is unmodified vendored
code. So the contract worth pinning is "does the object we hand SIMPL look
exactly like the one the motion-forecasting loader would have handed it".

Tests that need a real sensor log skip cleanly when data/av2_sensor/ is absent --
AV2 data is never committed (tests/test_license_guard.py), so a fresh clone has
none.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from av2.datasets.motion_forecasting.data_schema import (  # noqa: E402
    ObjectType,
    TrackCategory,
)
from av2.datasets.sensor.constants import AnnotationCategories  # noqa: E402
from av2.geometry.geometry import quat_to_mat  # noqa: E402
from av2.geometry.se3 import SE3  # noqa: E402

from otraj.sensor.scenario import (  # noqa: E402
    CATEGORY_TO_OBJECT_TYPE,
    DT,
    OBS,
    VEHICLE_CATEGORIES,
    WINDOW,
    _states,
    annotation_timestamps,
    build_scenario,
    focal_windows,
    n_windows,
    window_timestamps,
    yaw_of,
)

SENSOR_ROOT = ROOT / "data" / "av2_sensor" / "val"


def _a_log() -> Path:
    if not SENSOR_ROOT.exists():
        pytest.skip("no data/av2_sensor/val — AV2 data is never committed")
    logs = sorted(p for p in SENSOR_ROOT.iterdir()
                  if (p / "annotations.feather").exists())
    if not logs:
        pytest.skip("no sensor logs fetched")
    return logs[0]


def _a_log_with_pixels() -> Path:
    if not SENSOR_ROOT.exists():
        pytest.skip("no data/av2_sensor/val — AV2 data is never committed")
    for p in sorted(SENSOR_ROOT.iterdir()):
        if any((p / "sensors" / "cameras").glob("*/*.jpg")) if (
                p / "sensors" / "cameras").exists() else False:
            return p
    pytest.skip("no sensor log with camera pixels fetched")


# --------------------------------------------------------------------------
# shift #4 — the taxonomy map must be total, not best-effort
# --------------------------------------------------------------------------
def test_type_map_covers_every_sensor_category() -> None:
    """A missing class silently becomes UNKNOWN, which SIMPL one-hots into a
    different slot -- a real distribution shift hiding as a dict lookup."""
    missing = sorted({e.value for e in AnnotationCategories}
                     - set(CATEGORY_TO_OBJECT_TYPE))
    assert not missing, f"sensor categories with no MF ObjectType: {missing}"


def test_type_map_targets_are_real_object_types() -> None:
    for cat, t in CATEGORY_TO_OBJECT_TYPE.items():
        assert isinstance(t, ObjectType), f"{cat} -> {t!r}"


def test_vehicle_categories_are_the_vehicle_and_bus_slots() -> None:
    assert "REGULAR_VEHICLE" in VEHICLE_CATEGORIES
    assert "SCHOOL_BUS" in VEHICLE_CATEGORIES
    assert "PEDESTRIAN" not in VEHICLE_CATEGORIES
    assert "BICYCLE" not in VEHICLE_CATEGORIES


# --------------------------------------------------------------------------
# shift #1 — velocity is finite-differenced, so the differencing must be right
# --------------------------------------------------------------------------
def test_velocity_is_finite_differenced_at_the_annotation_rate() -> None:
    """Constant 5 m/s along +x must come back as (5, 0) everywhere, including
    the endpoints (np.gradient uses one-sided differences there)."""
    steps = list(range(WINDOW))
    pos = np.stack([np.arange(WINDOW) * 5.0 * DT, np.zeros(WINDOW)], axis=1)
    states = _states(steps, pos, [0.0] * WINDOW)
    vel = np.array([s.velocity for s in states])
    assert np.allclose(vel[:, 0], 5.0), vel[:5, 0]
    assert np.allclose(vel[:, 1], 0.0)


def test_velocity_of_a_single_state_is_zero_not_a_crash() -> None:
    states = _states([0], np.array([[1.0, 2.0]]), [0.0])
    assert states[0].velocity == (0.0, 0.0)


def test_observed_flag_splits_at_the_history_boundary() -> None:
    states = _states(list(range(WINDOW)), np.zeros((WINDOW, 2)), [0.0] * WINDOW)
    assert all(s.observed for s in states[:OBS])
    assert not any(s.observed for s in states[OBS:])


# --------------------------------------------------------------------------
# frame handling
# --------------------------------------------------------------------------
def test_yaw_of_recovers_the_rotation_angle() -> None:
    for theta in (-2.5, -0.3, 0.0, 1.1, 3.0):
        rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
        assert np.isclose(yaw_of(rot), theta)


def test_ego_city_round_trip_is_identity() -> None:
    """The synthesis pushes cuboids ego->city and the renderer pulls predictions
    city->ego. If those two disagree the overlay is silently wrong."""
    rng = np.random.default_rng(0)
    quat = np.array([0.6, 0.1, -0.2, 0.7])
    quat /= np.linalg.norm(quat)
    se3 = SE3(rotation=quat_to_mat(quat), translation=np.array([10.0, -4.0, 1.5]))
    pts = rng.normal(scale=25.0, size=(64, 3))
    back = se3.inverse().transform_point_cloud(se3.transform_point_cloud(pts))
    assert np.allclose(pts, back, atol=1e-9)


# --------------------------------------------------------------------------
# windowing — this is what bounds clip length
# --------------------------------------------------------------------------
def test_window_arithmetic_matches_the_clip_length_claim() -> None:
    log = _a_log()
    n_ann = len(annotation_timestamps(log))
    assert n_windows(log) == max(0, n_ann - WINDOW + 1)
    # the D-N1-15 claim: a ~15 s log yields ~48 windows == ~4.8 s of rolling video
    assert 30 <= n_windows(log) <= 70, n_windows(log)


def test_window_timestamps_are_contiguous_and_sized() -> None:
    log = _a_log()
    ts = window_timestamps(log, 0)
    assert len(ts) == WINDOW
    assert np.all(np.diff(ts) > 0)


def test_window_past_the_end_raises() -> None:
    log = _a_log()
    with pytest.raises(ValueError):
        window_timestamps(log, n_windows(log) + 5)


def test_focal_windows_never_exceeds_the_log_window_count() -> None:
    log = _a_log()
    focal = _first_full_vehicle(log)
    n = focal_windows(log, focal)
    assert 1 <= n <= n_windows(log)


def test_focal_windows_is_zero_for_an_unknown_track() -> None:
    """The renderer clips the clip to this number, so an unknown focal must give
    0 windows (caught up front) rather than a plausible-looking count."""
    assert focal_windows(_a_log(), "not-a-real-track-uuid") == 0


def test_focal_windows_bounds_what_build_scenario_accepts() -> None:
    """The contract the rolling render depends on: every window BELOW the count
    builds, and the first window AT the count does not. A mismatch here is what
    killed a 44-window render mid-flight."""
    log = _a_log()
    focal = _first_full_vehicle(log)
    n = focal_windows(log, focal)
    build_scenario(log, n - 1, focal)                      # last good window
    if n < n_windows(log):                                 # focal drops out early
        with pytest.raises(ValueError, match="fully observed|absent"):
            build_scenario(log, n, focal)


# --------------------------------------------------------------------------
# the synthesized scenario itself
# --------------------------------------------------------------------------
def _first_full_vehicle(log: Path) -> str:
    import pandas as pd
    ts = set(int(t) for t in window_timestamps(log, 0))
    ann = pd.read_feather(log / "annotations.feather")
    ann = ann[ann["timestamp_ns"].isin(ts)]
    for uuid, grp in ann.groupby("track_uuid", sort=True):
        if len(grp) == WINDOW and str(grp["category"].iloc[0]) in VEHICLE_CATEGORIES:
            return str(uuid)
    pytest.skip("no fully-observed vehicle in window 0 of this log")


def test_scenario_shape_matches_the_motion_forecasting_contract() -> None:
    log = _a_log()
    focal = _first_full_vehicle(log)
    scen, win_ts = build_scenario(log, 0, focal)

    assert len(win_ts) == WINDOW
    assert scen.focal_track_id == focal
    assert scen.city_name in ("ATX", "DTW", "MIA", "PAO", "PIT", "WDC")

    ids = [t.track_id for t in scen.tracks]
    assert len(ids) == len(set(ids)), "duplicate track ids"
    # SIMPL's get_trajectories asserts on both of these
    assert "AV" in ids
    focals = [t for t in scen.tracks if t.category == TrackCategory.FOCAL_TRACK]
    assert len(focals) == 1 and focals[0].track_id == focal

    for t in scen.tracks:
        steps = [s.timestep for s in t.object_states]
        assert steps == sorted(steps), f"{t.track_id} states unsorted"
        assert len(steps) == len(set(steps)), f"{t.track_id} duplicate timesteps"
        assert min(steps) >= 0 and max(steps) < WINDOW
        assert all(s.observed == (s.timestep < OBS) for s in t.object_states)


def test_focal_must_be_fully_observed() -> None:
    log = _a_log()
    with pytest.raises(ValueError, match="absent|fully observed"):
        build_scenario(log, 0, "not-a-real-track-uuid")


def test_av_track_follows_the_ego_pose() -> None:
    log = _a_log()
    scen, win_ts = build_scenario(log, 0, _first_full_vehicle(log))
    av = next(t for t in scen.tracks if t.track_id == "AV")
    assert len(av.object_states) == WINDOW
    pos = np.array([s.position for s in av.object_states])
    # ego motion over 11 s must be finite and physically plausible (<= 110 m/s)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert np.all(np.isfinite(pos))
    assert step.max() < 11.0, f"implausible ego step {step.max():.1f} m per 0.1 s"


def test_scenario_is_accepted_by_the_unmodified_simpl_preprocessor() -> None:
    """The real contract: D-N1-5 forbids editing vendored code, so the synthesized
    scenario has to satisfy ArgoPreprocAV2 as-is."""
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "third_party" / "SIMPL"))
    sys.path.insert(0, str(ROOT / "third_party" / "SIMPL" / "data_av2"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import _simpl_compat  # noqa: F401
    from av2.map.map_api import ArgoverseStaticMap
    from av2_preprocess import ArgoPreprocAV2

    log = _a_log()
    focal = _first_full_vehicle(log)
    scen, _ = build_scenario(log, 0, focal)
    static_map = ArgoverseStaticMap.from_map_dir(log / "map", build_raster=False)
    pre = ArgoPreprocAV2(
        SimpleNamespace(obs_len=OBS, pred_len=60, debug=False, viz=False, mode="val"),
        verbose=False)
    data, headers = pre.process(scen.scenario_id, scen, static_map)

    assert headers == ["SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TRAJS", "LANE_GRAPH"]
    trajs = data[0][headers.index("TRAJS")]
    assert trajs["trajs_pos"].shape[1:] == (WINDOW, 2)
    assert trajs["trajs_pos"].shape[0] >= 2, "focal + AV at minimum"
    assert np.isfinite(trajs["trajs_pos"]).all()
    assert trajs["has_flags"][0].sum() == WINDOW, "focal must be fully flagged"


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def test_projection_puts_a_point_ahead_of_the_camera_inside_the_image() -> None:
    from av2.geometry.camera.pinhole_camera import PinholeCamera

    from otraj.sensor import load_poses, pose_at, project_city_to_img

    log = _a_log_with_pixels()
    cam_name = next((log / "sensors" / "cameras").glob("*/*.jpg")).parent.name
    cam = PinholeCamera.from_feather(log, cam_name)
    pose_ts, quats, xyz = load_poses(log)
    ts = int(pose_ts[len(pose_ts) // 2])
    ego = pose_at(ts, pose_ts, quats, xyz)

    # 12 m straight down the camera's own optical axis, expressed in the city frame
    fwd_cam = np.array([[0.0, 0.0, 12.0]])
    fwd_ego = cam.extrinsics[:3, :3].T @ (fwd_cam[0] - cam.extrinsics[:3, 3])
    pt_city = ego.transform_point_cloud(fwd_ego[None])
    uv, valid = project_city_to_img(pt_city[:, :2], pt_city[:, 2], cam, ego)

    assert valid[0], "a point on the optical axis must survive the frustum cull"
    assert abs(uv[0, 0] - cam.width_px / 2) < 2.0
    assert abs(uv[0, 1] - cam.height_px / 2) < 2.0
