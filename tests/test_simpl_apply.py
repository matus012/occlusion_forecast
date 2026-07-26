"""Unit tests for otraj.masking.simpl_apply (D-N1-14 SIMPL occlusion adapter).

Pure numpy — runs in the torch-free CI guard subset.
"""
from __future__ import annotations

import numpy as np
import pytest

from otraj.masking.generator import N_STEPS, generate_mask
from otraj.masking.simpl_apply import (
    OBS_LEN,
    apply_cv_imputation,
    apply_native_mask,
)

SEQ = 110


def _sample(n_actors: int = 3) -> dict:
    """Synthetic TRAJS dict shaped like SIMPL av2 preprocessing output:
    constant-velocity straight lines, fully observed."""
    rng = np.random.default_rng(7)
    t = np.arange(SEQ, dtype=np.float32)
    trajs_pos = np.stack(
        [np.stack([t * (i + 1) * 0.1, t * 0.05 * i], axis=-1) for i in range(n_actors)]
    ).astype(np.float32)
    return {
        "trajs_pos": trajs_pos,
        "trajs_ang": np.tile(t * 0.01, (n_actors, 1)).astype(np.float32),
        "trajs_vel": rng.normal(size=(n_actors, SEQ, 2)).astype(np.float32),
        "trajs_type": np.tile(
            np.eye(7, dtype=np.int16)[0], (n_actors, SEQ, 1)),
        "has_flags": np.ones((n_actors, SEQ), dtype=np.int16),
    }


def _block_mask(start: int, end: int) -> np.ndarray:
    m = np.zeros(OBS_LEN, dtype=bool)
    m[start:end + 1] = True
    return m


class TestNativeMask:
    def test_flags_vel_type_zeroed_only_on_masked_steps(self):
        trajs = _sample()
        mask = _block_mask(10, 19)
        apply_native_mask(trajs, mask)
        assert (trajs["has_flags"][0, 10:20] == 0).all()
        assert (trajs["trajs_vel"][0, 10:20] == 0).all()
        assert (trajs["trajs_type"][0, 10:20] == 0).all()
        # untouched: before/after the block, the future, and other actors
        assert (trajs["has_flags"][0, :10] == 1).all()
        assert (trajs["has_flags"][0, 20:] == 1).all()
        assert (trajs["has_flags"][1:] == 1).all()

    def test_interior_gap_forward_fills_last_visible(self):
        trajs = _sample()
        pre = trajs["trajs_pos"][0].copy()
        apply_native_mask(trajs, _block_mask(10, 19))
        # native padding_traj_nn is ffill-then-bfill: the whole gap holds
        # the last pre-gap value, NOT distance-nearest or interpolated
        assert np.allclose(trajs["trajs_pos"][0, 10:20], pre[9])
        assert np.allclose(trajs["trajs_pos"][0, 20:], pre[20:])

    def test_leading_gap_backfills_first_visible(self):
        trajs = _sample()
        pre = trajs["trajs_pos"][0].copy()
        apply_native_mask(trajs, _block_mask(0, 14))  # M2-style prefix
        assert np.allclose(trajs["trajs_pos"][0, :15], pre[15])

    def test_no_future_leakage_for_trailing_gap(self):
        trajs = _sample()
        pre = trajs["trajs_pos"][0].copy()
        mask = _block_mask(40, 49)  # R-B-style: extends through t=0
        apply_native_mask(trajs, mask)
        # fill must come from index 39 (last visible obs), never from t>0
        assert np.allclose(trajs["trajs_pos"][0, 40:50], pre[39])
        assert np.allclose(trajs["trajs_pos"][0, 50:], pre[50:])

    def test_empty_mask_is_identity(self):
        trajs = _sample()
        pre = {k: v.copy() for k, v in trajs.items()}
        apply_native_mask(trajs, np.zeros(OBS_LEN, dtype=bool))
        for k in pre:
            assert np.array_equal(trajs[k], pre[k])

    def test_fully_masked_window_raises(self):
        trajs = _sample()
        with pytest.raises(ValueError):
            apply_native_mask(trajs, np.ones(OBS_LEN, dtype=bool))

    def test_commutes_with_affine_normalization(self):
        """Copying values between timesteps commutes with a per-actor affine
        transform -- the property that justifies post-normalization masking."""
        trajs_a = _sample()
        trajs_b = {k: v.copy() for k, v in trajs_a.items()}
        rot = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=np.float32)
        orig = np.array([3.0, -2.0], dtype=np.float32)
        trajs_b["trajs_pos"][0] = (trajs_b["trajs_pos"][0] - orig) @ rot
        mask = generate_mask("commute-test", "S3", "R-A").mask
        apply_native_mask(trajs_a, mask)
        apply_native_mask(trajs_b, mask)
        expected = (trajs_a["trajs_pos"][0] - orig) @ rot
        assert np.allclose(trajs_b["trajs_pos"][0], expected, atol=1e-5)

    def test_generator_masks_satisfy_ra_invariant_after_application(self):
        trajs = _sample()
        mask = generate_mask("some-scenario", "S4", "R-A").mask
        apply_native_mask(trajs, mask)
        # R-A: last 5 obs steps stay visible
        assert (trajs["has_flags"][0, N_STEPS - 5:N_STEPS] == 1).all()


class TestCVImputation:
    def test_masked_steps_flagged_valid_and_type_restored(self):
        trajs = _sample()
        apply_cv_imputation(trajs, _block_mask(10, 19))
        assert (trajs["has_flags"][0, :OBS_LEN] == 1).all()
        assert (trajs["trajs_type"][0, 10:20, 0] == 1).all()

    def test_interior_gap_is_linear_interpolation(self):
        trajs = _sample()
        pre = trajs["trajs_pos"][0].copy()
        apply_cv_imputation(trajs, _block_mask(10, 19))
        w = (np.arange(10, 20) - 9) / (20 - 9)
        expected = pre[9] + w[:, None] * (pre[20] - pre[9])
        assert np.allclose(trajs["trajs_pos"][0, 10:20], expected, atol=1e-5)

    def test_leading_gap_cv_backward_extrapolation(self):
        trajs = _sample()
        trajs["trajs_vel"][0, 15] = np.array([2.0, 0.0])
        pre_pos = trajs["trajs_pos"][0, 15].copy()
        apply_cv_imputation(trajs, _block_mask(0, 14))
        expected_t0 = pre_pos + (0 - 15) * 0.1 * np.array([2.0, 0.0])
        assert np.allclose(trajs["trajs_pos"][0, 0], expected_t0, atol=1e-5)

    def test_trailing_gap_cv_forward_extrapolation(self):
        trajs = _sample()
        trajs["trajs_vel"][0, 39] = np.array([1.0, 1.0])
        pre_pos = trajs["trajs_pos"][0, 39].copy()
        apply_cv_imputation(trajs, _block_mask(40, 49))
        expected_t49 = pre_pos + 10 * 0.1 * np.array([1.0, 1.0])
        assert np.allclose(trajs["trajs_pos"][0, 49], expected_t49, atol=1e-5)

    def test_flicker_mask_all_runs_filled(self):
        trajs = _sample()
        mask = generate_mask("flicker-heavy-id-M3", "S3", "R-A").mask
        apply_cv_imputation(trajs, mask)
        assert (trajs["has_flags"][0, :OBS_LEN] == 1).all()
