"""Apply occlusion masks to SIMPL-preprocessed AV2 samples (D-N1-14).

SIMPL's NATIVE missing-step convention (third_party/SIMPL/data_av2/
av2_preprocess.py, get_trajectories + padding_traj_nn) is, per timestep with
validity flag 0:

  * has_flags -> 0 (surfaces as the PAD_OBS actor-feature channel),
  * position/heading -> padding_traj_nn = FORWARD-fill then BACKWARD-fill
    (interior and trailing gaps hold the last pre-gap value; leading gaps
    take the first valid value; it is NOT distance-nearest-neighbor),
  * velocity -> 0,
  * one-hot object type -> all-zero row.

Occluding a step == making the model see exactly what it would have seen had
the tracker never emitted that step, so this module reproduces that convention
bit-for-bit on the PREPROCESSED (per-actor normalized) tensors. Copying values
between timesteps commutes with SIMPL's per-actor affine normalization, so
post-normalization application is exact, with one deliberate deviation:
fills only ever source from OBSERVED-WINDOW steps (indices 0..obs_len-1).
Native preprocessing pads over the full 110-step sequence and could pull
future (t>0) values into a trailing gap; for occlusion eval that would be
ground-truth leakage. Forward-fill makes this moot for interior/trailing
gaps (they only look backward); the constraint bites only for leading gaps,
which backfill from the first VISIBLE obs step, never from the future.

The anchor fields (TRAJS_CTRS/TRAJS_VECS, i.e. per-actor origin/rotation at
index obs_len-1) are baked in at preprocess time from the true t=0 pose. Under
R-A ("re-emerged", last 5 steps forced visible) that pose is genuinely
observed, so anchors are legitimate. Under R-B the true-t=0 anchor is NOT
observable -- R-B rendering/eval through this module carries that caveat
(documented wherever R-B output is shown; the D-N1-14 headline curve is R-A
only). SIMPL has no native R-B precedent at all: agents invisible at t=0 are
dropped entirely by its preprocessing.

C2 "cheap-fix null" imputation (context.md D-N1-2): instead of the native
convention, masked steps are FILLED with plausible values and flagged VALID
(the model is deliberately lied to -- that is the point of the null):
  * interior gaps: linear interpolation between the bounding visible steps
    (position, velocity; heading via shortest-arc interpolation),
  * leading gaps: constant-velocity backward extrapolation from the first
    visible step (its recorded velocity),
  * trailing gaps: constant-velocity forward extrapolation from the last
    visible step,
  * has_flags -> 1, object type -> restored (copied from a visible step).

Everything here is pure numpy on the unpickled sample dict -- no torch, no
SIMPL imports -- so it is unit-testable in the torch-free CI guard subset.
"""
from __future__ import annotations

import numpy as np

OBS_LEN = 50
FOCAL_ROW = 0  # SIMPL preprocessing writes the focal track first (sorted_idcs)


def _ffill_bfill(values: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """SIMPL padding_traj_nn semantics restricted to the given window:
    forward-fill from the last visible step, then backward-fill leading gaps.
    `values` is (T,) or (T, D); `visible` is a (T,) bool mask."""
    out = values.copy()
    vis_idx = np.flatnonzero(visible)
    if vis_idx.size == 0:
        raise ValueError("cannot pad a fully-occluded window (no visible step)")
    # forward fill
    last = None
    for t in range(out.shape[0]):
        if visible[t]:
            last = t
        elif last is not None:
            out[t] = out[last]
    # backward fill the leading gap
    first_vis = vis_idx[0]
    out[:first_vis] = out[first_vis]
    return out


def apply_native_mask(
    trajs: dict, mask: np.ndarray, row: int = FOCAL_ROW, obs_len: int = OBS_LEN,
) -> dict:
    """Occlude `row`'s masked obs steps in a preprocessed TRAJS dict, IN PLACE,
    per the native SIMPL convention described in the module docstring.
    `mask` is bool (obs_len,), True == occluded. Returns `trajs`."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (obs_len,):
        raise ValueError(f"mask shape {mask.shape} != ({obs_len},)")
    if not mask.any():
        return trajs

    flags = trajs["has_flags"][row]
    visible = (flags[:obs_len] > 0) & ~mask
    if not visible.any():
        raise ValueError("mask occludes every valid obs step of the target row")

    pos, ang = trajs["trajs_pos"][row], trajs["trajs_ang"][row]
    pos[:obs_len] = _ffill_bfill(pos[:obs_len], visible)
    ang[:obs_len] = _ffill_bfill(ang[:obs_len], visible)
    trajs["trajs_vel"][row, :obs_len][mask] = 0.0
    trajs["trajs_type"][row, :obs_len][mask] = 0
    flags[:obs_len][mask] = 0
    return trajs


def _interp_angle(a0: float, a1: float, w: np.ndarray) -> np.ndarray:
    """Shortest-arc linear interpolation between two angles; w in (0, 1)."""
    delta = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return a0 + w * delta


def apply_cv_imputation(
    trajs: dict, mask: np.ndarray, row: int = FOCAL_ROW, obs_len: int = OBS_LEN,
    hz: float = 10.0,
) -> dict:
    """C2 cheap-fix null: fill `row`'s masked obs steps with linear/constant-
    velocity imputation and mark them VALID, in place. See module docstring."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (obs_len,):
        raise ValueError(f"mask shape {mask.shape} != ({obs_len},)")
    if not mask.any():
        return trajs

    flags = trajs["has_flags"][row]
    visible = (flags[:obs_len] > 0) & ~mask
    vis_idx = np.flatnonzero(visible)
    if vis_idx.size == 0:
        raise ValueError("mask occludes every valid obs step of the target row")

    pos = trajs["trajs_pos"][row]
    ang = trajs["trajs_ang"][row]
    vel = trajs["trajs_vel"][row]
    typ = trajs["trajs_type"][row]
    dt = 1.0 / hz
    type_row = typ[vis_idx[0]].copy()  # object type is constant while observed

    # walk contiguous masked runs inside the obs window
    t = 0
    while t < obs_len:
        if not mask[t]:
            t += 1
            continue
        run_start = t
        while t < obs_len and mask[t]:
            t += 1
        run_end = t - 1  # inclusive
        before = vis_idx[vis_idx < run_start]
        after = vis_idx[vis_idx > run_end]
        steps = np.arange(run_start, run_end + 1)

        if before.size and after.size:  # interior gap: linear interpolation
            b, a = before[-1], after[0]
            w = (steps - b) / (a - b)
            pos[steps] = pos[b] + w[:, None] * (pos[a] - pos[b])
            vel[steps] = vel[b] + w[:, None] * (vel[a] - vel[b])
            ang[steps] = _interp_angle(float(ang[b]), float(ang[a]), w)
        elif after.size:  # leading gap: CV backward extrapolation
            a = after[0]
            back = (steps - a) * dt  # negative
            pos[steps] = pos[a] + back[:, None] * vel[a]
            vel[steps] = vel[a]
            ang[steps] = ang[a]
        else:  # trailing gap: CV forward extrapolation
            b = before[-1]
            fwd = (steps - b) * dt
            pos[steps] = pos[b] + fwd[:, None] * vel[b]
            vel[steps] = vel[b]
            ang[steps] = ang[b]

        typ[steps] = type_row
        flags[steps] = 1
    return trajs
