"""City-frame -> ring-camera image projection for the RGB demo (D-N1-15).

Chain: city -> ego @ CAMERA timestamp -> camera -> image, via the log's own
`city_SE3_egovehicle` and `calibration/`. Ground height comes from the sensor
split's raster layer rather than a flat-earth assumption (the motion-forecasting
split does not ship one).

The chain was verified against ground truth by projecting annotated 3D cuboids
through the identical path and checking they land on the vehicles -- see
`cuboid_corners_city` + `scripts/rgb_overlay_spike.py --boxes`, and
reports/rgb_demo_feasibility.md §3. That check is independent of both the model
and the trajectory code, which is the point: a trajectory drawn from the same
annotations that produced it looks plausible even when the projection is wrong.
"""
from __future__ import annotations

import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap


def ground_height(static_map: ArgoverseStaticMap, xy: np.ndarray) -> np.ndarray:
    """Ground-surface z (city frame) under each xy. Requires build_raster=True."""
    return static_map.raster_ground_height_layer.get_ground_height_at_xy(xy)


def project_city_to_img(xy: np.ndarray, z: np.ndarray, cam: PinholeCamera,
                        city_SE3_ego: SE3) -> tuple[np.ndarray, np.ndarray]:
    """(uv, is_valid) for city-frame points, through the ego pose at the camera's
    own timestamp. `is_valid` is av2's frustum cull (cheirality + image bounds)."""
    pts_city = np.concatenate([np.asarray(xy, dtype=float),
                               np.asarray(z, dtype=float)[:, None]], axis=1)
    pts_ego = city_SE3_ego.inverse().transform_point_cloud(pts_city)
    uv, _, valid = cam.project_ego_to_img(pts_ego)
    return uv, valid


def cuboid_corners_city(row, city_SE3_ego_ann: SE3) -> np.ndarray:
    """8 corners of an annotated cuboid, ego frame -> city frame."""
    length, width, height = row.length_m, row.width_m, row.height_m
    unit = np.array([[sx * length / 2, sy * width / 2, sz * height / 2]
                     for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    rot = quat_to_mat(np.array([row.qw, row.qx, row.qy, row.qz]))
    ego = unit @ rot.T + np.array([row.tx_m, row.ty_m, row.tz_m])
    return city_SE3_ego_ann.transform_point_cloud(ego)


# 12 edges of the corner ordering produced above (sx, sy, sz in that nesting)
BOX_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def nearest_index(stamps: np.ndarray, ts: int) -> int:
    return int(np.argmin(np.abs(np.asarray(stamps) - ts)))
