"""AV2 Sensor -> Motion-Forecasting scenario synthesis (D-N1-15).

The AV2 motion-forecasting and sensor datasets are mined from disjoint corpora
and share no scenario<->log mapping, so a sensor log cannot be *looked up* as a
forecasting scenario -- it has to be SYNTHESIZED into one. This package does
exactly that and nothing else, so the whole downstream stack (ArgoPreprocAV2 ->
SIMPL -> masking -> metrics) runs unmodified (D-N1-5).

See reports/rgb_demo_feasibility.md for the five named distribution shifts the
synthesis introduces.
"""
from otraj.sensor.projection import (
    cuboid_corners_city,
    ground_height,
    project_city_to_img,
)
from otraj.sensor.scenario import (
    CATEGORY_TO_OBJECT_TYPE,
    FUT,
    OBS,
    VEHICLE_CATEGORIES,
    build_scenario,
    focal_windows,
    load_annotations,
    load_poses,
    n_windows,
    pose_at,
    window_timestamps,
    yaw_of,
)

__all__ = [
    "CATEGORY_TO_OBJECT_TYPE",
    "FUT",
    "OBS",
    "VEHICLE_CATEGORIES",
    "build_scenario",
    "cuboid_corners_city",
    "focal_windows",
    "load_annotations",
    "ground_height",
    "load_poses",
    "n_windows",
    "pose_at",
    "project_city_to_img",
    "window_timestamps",
    "yaw_of",
]
