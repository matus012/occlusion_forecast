"""Unit tests for otraj.masking.p2_stats (Phase 3 prep, D-N1-2).

Uses SMALL synthetic fixtures only -- no files from the sibling P2 project
(ws/100_occlusion_mot) are read or committed here. Note on fixture shape: the
task brief anticipated "scenario JSON" fixtures, but exploring the real P2
data (see p2_stats.py module docstring) showed the actual per-frame
visibility ground truth lives in MOT17-format CSV (``gt/gt.txt``) plus a
``seqinfo.ini`` for fps/seq_length -- scenario.json only holds waypoints/
occluder geometry with no visibility column. These fixtures mirror the real
on-disk schema (built fresh per test via ``tmp_path``, nothing copied in).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from otraj.masking.p2_stats import (
    AgentTrace,
    OcclusionRun,
    build_p2_stats,
    classify_agent_pattern,
    extract_agent_runs,
    fit_lognormal,
    load_scenario_visibility,
)

FPS = 20  # deliberately NOT the same as the real P2 data's 20fps coincidence--
# see test_derives_fps_from_data_not_hardcoded, which uses a different value.


def _make_trace(vis: list[float], fps: int = FPS, agent_id: int = 1,
                 scenario_id: str = "synthetic_scn") -> AgentTrace:
    n = len(vis)
    return AgentTrace(
        scenario_id=scenario_id,
        agent_id=agent_id,
        frames=np.arange(1, n + 1, dtype=np.int64),
        vis=np.array(vis, dtype=np.float64),
        fps=fps,
    )


def _write_synthetic_scenario(
    root: Path,
    name: str,
    agents_vis: dict[int, list[float]],
    fps: int = FPS,
) -> Path:
    """Write a minimal MOT17-format scenario dir: gt/gt.txt + seqinfo.ini."""
    scen_dir = root / name
    gt_dir = scen_dir / "gt"
    gt_dir.mkdir(parents=True)
    seq_length = max(len(v) for v in agents_vis.values())
    lines = []
    for agent_id, vis in agents_vis.items():
        for i, v in enumerate(vis):
            frame = i + 1
            # frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,visibility
            lines.append(f"{frame},{agent_id},0.0,0.0,10.0,10.0,1.0000,1,{v:.4f}")
    (gt_dir / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (scen_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\n"
        f"seqLength={seq_length}\nimWidth=100\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )
    return scen_dir


# --------------------------------------------------------------------------
# extract_agent_runs: edge cases
# --------------------------------------------------------------------------


def test_interior_block_both_anchored() -> None:
    """M1 archetype: visible - occluded (dips <0.25) - visible."""
    vis = [1.0] * 5 + [0.1] * 7 + [1.0] * 5
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace)
    assert not is_artifact
    assert len(runs) == 1
    r = runs[0]
    assert r.left_anchored and r.right_anchored
    assert r.duration_frames == 7
    assert r.start_frame == 6 and r.end_frame == 12


def test_starts_occluded_leading_run_unanchored() -> None:
    """M2 archetype: occluded from frame 1 (no left anchor), then visible."""
    vis = [0.1] * 6 + [1.0] * 10
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace)
    assert not is_artifact
    assert len(runs) == 1
    r = runs[0]
    assert r.left_anchored is False
    assert r.right_anchored is True
    assert r.start_frame == 1
    assert r.duration_frames == 6


def test_ends_occluded_trailing_run_unanchored() -> None:
    """Occlusion never recovers before the trace ends (no right anchor)."""
    vis = [1.0] * 10 + [0.1] * 6
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace)
    assert not is_artifact
    assert len(runs) == 1
    r = runs[0]
    assert r.left_anchored is True
    assert r.right_anchored is False
    assert r.end_frame == 16


def test_single_frame_flicker_filtered_by_min_len() -> None:
    """A 1-frame dip is shorter than min_len=5 -> not a segment."""
    vis = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace, min_len=5)
    assert not is_artifact
    assert runs == []


def test_partial_occlusion_never_dipping_below_vis_lo_is_excluded() -> None:
    """Long run that stays in [vis_lo, vis_hi) the whole time is NOT a segment
    (contamination rule ported from P1 D14)."""
    vis = [1.0] * 5 + [0.3] * 10 + [1.0] * 5
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace, vis_lo=0.25, vis_hi=0.5, min_len=5)
    assert not is_artifact
    assert runs == []


def test_min_len_boundary_exact() -> None:
    """A run of exactly min_len frames counts; min_len - 1 does not."""
    vis_ok = [1.0] * 5 + [0.1] * 5 + [1.0] * 5
    runs_ok, _ = extract_agent_runs(_make_trace(vis_ok), min_len=5)
    assert len(runs_ok) == 1 and runs_ok[0].duration_frames == 5

    vis_short = [1.0] * 5 + [0.1] * 4 + [1.0] * 5
    runs_short, _ = extract_agent_runs(_make_trace(vis_short), min_len=5)
    assert runs_short == []


def test_fully_occluded_agent_flagged_as_artifact_not_a_segment() -> None:
    """P1 D23 'walker-6' render-artifact pattern: vis==0.0 for the ENTIRE
    trace -- never anchored, must not be reported as one giant segment."""
    vis = [0.0] * 30
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace)
    assert is_artifact is True
    assert runs == []


def test_multiple_interior_runs_flicker_shape() -> None:
    """M3 archetype: several short interior occlusion runs."""
    vis = (
        [1.0] * 10
        + [0.1] * 6
        + [1.0] * 10
        + [0.1] * 8
        + [1.0] * 10
        + [0.1] * 6
        + [1.0] * 10
    )
    trace = _make_trace(vis)
    runs, is_artifact = extract_agent_runs(trace)
    assert not is_artifact
    assert len(runs) == 3
    assert all(r.left_anchored and r.right_anchored for r in runs)


# --------------------------------------------------------------------------
# duration-seconds conversion
# --------------------------------------------------------------------------


def test_duration_seconds_conversion() -> None:
    run = OcclusionRun(
        scenario_id="s", agent_id=1, start_frame=10, end_frame=29, fps=20,
        left_anchored=True, right_anchored=True,
    )
    assert run.duration_frames == 20
    assert run.duration_s == pytest.approx(1.0)

    run_odd_fps = OcclusionRun(
        scenario_id="s", agent_id=1, start_frame=1, end_frame=5, fps=25,
        left_anchored=True, right_anchored=True,
    )
    assert run_odd_fps.duration_frames == 5
    assert run_odd_fps.duration_s == pytest.approx(0.2)


# --------------------------------------------------------------------------
# pattern classification: M1 / M2 / M3 archetypes
# --------------------------------------------------------------------------


def _run(left_anchored: bool, right_anchored: bool, duration_frames: int = 6,
          fps: int = 20, agent_id: int = 1) -> OcclusionRun:
    return OcclusionRun(
        scenario_id="s", agent_id=agent_id, start_frame=1,
        end_frame=duration_frames, fps=fps,
        left_anchored=left_anchored, right_anchored=right_anchored,
    )


def test_classify_none_when_no_runs() -> None:
    assert classify_agent_pattern([]) == "none"


def test_classify_m1_single_interior_block() -> None:
    assert classify_agent_pattern([_run(True, True)]) == "M1"


def test_classify_m2_single_leading_prefix() -> None:
    assert classify_agent_pattern([_run(False, True)]) == "M2"


def test_classify_other_trailing_single_unrecovered_run() -> None:
    assert classify_agent_pattern([_run(True, False)]) == "other_trailing"


@pytest.mark.parametrize("n_runs", [2, 3, 4])
def test_classify_m3_flicker_counts(n_runs: int) -> None:
    runs = [_run(True, True, duration_frames=6) for _ in range(n_runs)]
    assert classify_agent_pattern(runs) == "M3"


def test_classify_other_many_above_four_runs() -> None:
    runs = [_run(True, True) for _ in range(5)]
    assert classify_agent_pattern(runs) == "other_many"


# --------------------------------------------------------------------------
# loader: schema + fail-fast behavior
# --------------------------------------------------------------------------


def test_load_scenario_visibility_reads_dense_traces(tmp_path: Path) -> None:
    scen_dir = _write_synthetic_scenario(
        tmp_path, "scn_a", {1: [1.0] * 10, 2: [1.0] * 5 + [0.1] * 6 + [1.0] * 5}, fps=15,
    )
    traces = load_scenario_visibility(scen_dir)
    assert set(traces) == {1, 2}
    assert traces[1].fps == 15
    assert len(traces[2].frames) == 16
    assert traces[2].frames[0] == 1 and traces[2].frames[-1] == 16


def test_load_scenario_visibility_fails_fast_on_frame_gap(tmp_path: Path) -> None:
    scen_dir = tmp_path / "broken"
    gt_dir = scen_dir / "gt"
    gt_dir.mkdir(parents=True)
    # agent 1 has a missing frame 3 -> not dense/contiguous
    (gt_dir / "gt.txt").write_text(
        "1,1,0,0,1,1,1.0000,1,1.0000\n"
        "2,1,0,0,1,1,1.0000,1,1.0000\n"
        "4,1,0,0,1,1,1.0000,1,1.0000\n",
        encoding="utf-8",
    )
    (scen_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=broken\nimDir=img1\nframeRate=20\nseqLength=4\n"
        "imWidth=100\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dense/contiguous"):
        load_scenario_visibility(scen_dir)


def test_load_scenario_visibility_missing_gt_file_fails_fast(tmp_path: Path) -> None:
    scen_dir = tmp_path / "no_gt"
    scen_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_scenario_visibility(scen_dir)


# --------------------------------------------------------------------------
# log-normal duration fit
# --------------------------------------------------------------------------


def test_fit_lognormal_recovers_plausible_params() -> None:
    rng = np.random.default_rng(0)
    true_shape, true_scale = 0.8, 1.2
    durations = rng.lognormal(mean=np.log(true_scale), sigma=true_shape, size=5000)
    fit = fit_lognormal(durations)
    assert fit.n == 5000
    assert fit.shape == pytest.approx(true_shape, rel=0.15)
    assert fit.scale == pytest.approx(true_scale, rel=0.15)
    assert fit.median_s < fit.mean_s  # right-skewed, as expected for log-normal


def test_fit_lognormal_rejects_empty() -> None:
    with pytest.raises(ValueError):
        fit_lognormal(np.array([]))


def test_fit_lognormal_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        fit_lognormal(np.array([1.0, 0.0, 2.0]))


# --------------------------------------------------------------------------
# end-to-end: build_p2_stats on a tiny synthetic scenario set
# --------------------------------------------------------------------------


def _build_tiny_dataset(root: Path, fps: int = 12) -> None:
    """Two scenarios, one agent of each archetype, plus one artifact agent --
    fps deliberately != the real P2 data's 20fps to prove fps is derived, not
    assumed."""
    _write_synthetic_scenario(
        root,
        "scn_one",
        {
            1: [1.0] * 5 + [0.1] * 7 + [1.0] * 5,  # M1
            2: [0.1] * 6 + [1.0] * 14,  # M2
            3: [0.0] * 20,  # artifact: fully occluded, excluded
        },
        fps=fps,
    )
    _write_synthetic_scenario(
        root,
        "scn_two",
        {
            # M3 (2 interior runs); total length matches scn_one's seqLength
            # (20 frames) so the uniform-nominal-duration invariant holds.
            4: [1.0] * 2 + [0.1] * 5 + [1.0] * 2 + [0.1] * 5 + [1.0] * 6,
            5: [1.0] * 20,  # none
        },
        fps=fps,
    )


def test_build_p2_stats_end_to_end(tmp_path: Path) -> None:
    _build_tiny_dataset(tmp_path)
    result = build_p2_stats(tmp_path)

    assert result.n_scenarios == 2
    assert result.n_agents_total == 5
    assert len(result.excluded_artifact_agents) == 1
    assert result.excluded_artifact_agents[0] == ("scn_one", 3)
    assert result.n_agents_analyzed == 4
    assert result.fps == 12  # derived from seqinfo.ini, not hardcoded to 20

    assert result.pattern_counts["M1"] == 1
    assert result.pattern_counts["M2"] == 1
    assert result.pattern_counts["M3"] == 1
    assert result.pattern_counts["none"] == 1
    assert result.duration_fit.n == len(result.runs)

    d = result.to_dict()
    assert d["fps"] == 12
    assert d["pattern_mix"]["fractions_of_M1_M2_M3_only"]["M1"] == pytest.approx(1 / 3)
    assert d["mot17_sanity_anchor"]["median_gap_frames"] == 37


def test_derives_fps_from_data_not_hardcoded(tmp_path: Path) -> None:
    _write_synthetic_scenario(
        tmp_path, "scn_odd_fps",
        {1: [1.0] * 5 + [0.1] * 6 + [1.0] * 5}, fps=7,
    )
    result = build_p2_stats(tmp_path)
    assert result.fps == 7
    assert result.runs[0].duration_s == pytest.approx(6 / 7)


def test_build_p2_stats_determinism(tmp_path: Path) -> None:
    _build_tiny_dataset(tmp_path)
    r1 = build_p2_stats(tmp_path).to_dict()
    r2 = build_p2_stats(tmp_path).to_dict()
    assert r1 == r2


def test_build_p2_stats_rejects_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_p2_stats(tmp_path)
