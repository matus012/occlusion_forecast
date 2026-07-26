"""P2 CARLA occlusion-segment statistics extraction (N1 Phase 3 prep, D-N1-2).

Parameterizes the N1 occlusion mask generator from real per-agent visibility
recorded by the sibling P2 project (ws/100_occlusion_mot, read-only source,
never modified here).

Schema note (learned by exploring the source data, NOT assumed up front):
per-frame ground-truth visibility for each CARLA scenario lives in
``<scenario_dir>/gt/gt.txt`` in MOT17-challenge CSV format::

    frame, agent_id, bb_left, bb_top, bb_w, bb_h, conf, class, visibility

with frame rate in ``<scenario_dir>/seqinfo.ini`` (``frameRate=``). The sibling
``scenario.json`` / ``specs/<name>.json`` files hold only the CARLA scene spec
(walker waypoints, occluder boxes, camera pose) -- they carry NO per-frame
visibility column and are not used here. All 24 scenarios observed at fps=20,
seqLength=400 (20.0s), pedestrian-only (class==1, conf==1.0 throughout) -- read
from the data per scenario regardless, never hardcoded.

Segment definition ported from P1 D14 (``omot/eval/occlusion.py``): an
occlusion run requires visibility to dip below ``vis_lo`` (0.25) at least once
and to stay below ``vis_hi`` (0.5) for >= ``min_len`` (5) consecutive frames.
P1's original algorithm only detects runs strictly BETWEEN two visible anchors
(visibility >= vis_hi on both sides). That misses two patterns this project
needs: M2 "prefix" (occluded from the start of the trace, no left anchor) and
occlusion that never recovers before the trace ends (no right anchor). This
module generalizes the same threshold semantics to ALL maximal low-visibility
runs, tagging each with ``left_anchored`` / ``right_anchored`` so the P1-exact
(both-anchored) subset remains recoverable for comparison.

Known data artifact (P1 D23, "walker-6 case"): a small number of walkers in the
``behind_static`` template never become properly visible -- a render/shadow bug
(occluder prop geometry vs. spec slabs mismatch), not real transient occlusion.
Three of the four excluded agents read visibility == 0.0 throughout; one shows a
brief sub-threshold noise bump (peak 0.21) before reverting to 0.0. The
exclusion criterion is data-driven, not a hardcoded list: any agent that never
once reaches ``vis_hi`` is excluded from all statistics below and reported
separately as ``excluded_artifact_agents``.
"""
from __future__ import annotations

import configparser
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# MOT17-format gt.txt column indices.
COL_FRAME, COL_ID, COL_X, COL_Y, COL_W, COL_H, COL_CONF, COL_CLS, COL_VIS = range(9)

DEFAULT_VIS_LO = 0.25
DEFAULT_VIS_HI = 0.5
DEFAULT_MIN_LEN = 5  # frames

# M3 flicker duration band per context.md D-N1-2 (used only for the secondary
# "strict band" diagnostic -- classification itself is count-based, see
# classify_agent_pattern).
FLICKER_LO_S = 0.2
FLICKER_HI_S = 0.6

# MOT17 sanity anchor, as pre-declared in context.md D-N1-2 (P1 D14,
# MOT17-02 single-sequence figure) -- cited, NEVER fit against.
MOT17_ANCHOR_MEDIAN_GAP_FRAMES = 37
MOT17_ANCHOR_MEDIAN_GAP_S = 1.5


@dataclass(frozen=True)
class OcclusionRun:
    """One maximal contiguous low-visibility run for one agent."""

    scenario_id: str
    agent_id: int
    start_frame: int
    end_frame: int  # inclusive
    fps: int
    left_anchored: bool  # a visible (vis >= vis_hi) frame immediately precedes this run
    right_anchored: bool  # a visible frame immediately follows this run

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_s(self) -> float:
        return self.duration_frames / self.fps


@dataclass(frozen=True)
class AgentTrace:
    scenario_id: str
    agent_id: int
    frames: np.ndarray  # (n,) int, dense & sorted
    vis: np.ndarray  # (n,) float in [0, 1]
    fps: int


@dataclass(frozen=True)
class LogNormalFit:
    shape: float
    loc: float
    scale: float
    median_s: float
    mean_s: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "shape": self.shape,
            "loc": self.loc,
            "scale": self.scale,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "n": self.n,
        }


@dataclass
class ExtractionResult:
    n_scenarios: int
    n_agents_total: int
    excluded_artifact_agents: list[tuple[str, int]]  # (scenario_id, agent_id)
    n_agents_analyzed: int
    fps: int
    scenario_duration_s: float
    runs: list[OcclusionRun]
    duration_fit: LogNormalFit
    segments_per_agent_mean: float
    segments_per_agent_second: float
    spacing_median_s: float
    spacing_mean_s: float
    n_spacing_samples: int
    pattern_counts: dict[str, int]
    pattern_fractions: dict[str, float]
    m1m2m3_fractions: dict[str, float]
    n_agents_with_runs: int
    m3_strict_band_fraction: float
    both_anchored_segment_count: int  # P1-D14-exact subset, for direct comparison
    both_anchored_gap_median_frames: float
    mot17_anchor: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_scenarios": self.n_scenarios,
            "n_agents_total": self.n_agents_total,
            "excluded_artifact_agents": [
                {"scenario_id": s, "agent_id": a} for s, a in self.excluded_artifact_agents
            ],
            "n_agents_analyzed": self.n_agents_analyzed,
            "fps": self.fps,
            "scenario_duration_s": self.scenario_duration_s,
            "n_segments": len(self.runs),
            "duration_fit_lognormal": self.duration_fit.to_dict(),
            "segments_per_agent_mean": self.segments_per_agent_mean,
            "segments_per_agent_second": self.segments_per_agent_second,
            "spacing_s": {
                "median": self.spacing_median_s,
                "mean": self.spacing_mean_s,
                "n_samples": self.n_spacing_samples,
            },
            "pattern_mix": {
                "counts": self.pattern_counts,
                "fractions_of_all_analyzed_agents": self.pattern_fractions,
                "fractions_of_M1_M2_M3_only": self.m1m2m3_fractions,
                "n_agents_with_runs": self.n_agents_with_runs,
                "m3_strict_duration_band_fraction": self.m3_strict_band_fraction,
                "v1_prior": {"M1": 0.6, "M2": 0.25, "M3": 0.15},
                "note": (
                    "fractions_of_M1_M2_M3_only is the apples-to-apples comparison "
                    "against the v1 prior (which only defines M1/M2/M3, summing to "
                    "1.0); fractions_of_all_analyzed_agents also includes agents "
                    "with zero qualifying segments ('none') and two taxonomy "
                    "leftovers outside M1/M2/M3 ('other_trailing': occlusion never "
                    "recovers before the trace ends; 'other_many': >4 segments)."
                ),
            },
            "p1_d14_exact_comparison": {
                "both_anchored_segment_count": self.both_anchored_segment_count,
                "both_anchored_gap_median_frames": self.both_anchored_gap_median_frames,
                "cited_p2_reference_n_segments": 223,
                "cited_p2_reference_n_scenarios": 24,
                "note": (
                    "both-anchored count uses the identical P1 D14 thresholds/logic "
                    "(vis_hi=0.5, vis_lo=0.25, min_len=5) as a direct sanity check "
                    "against the ~223-segment figure cited in context.md D-N1-2 / P1 "
                    "D23; it excludes the unanchored leading/trailing runs this module "
                    "adds to detect M2/trailing patterns."
                ),
            },
            "mot17_sanity_anchor": self.mot17_anchor,
        }


def _read_seqinfo(scenario_dir: Path) -> tuple[int, int]:
    """Return (fps, seq_length_frames) read from seqinfo.ini. Never assumed."""
    ini_path = scenario_dir / "seqinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini missing under {scenario_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    if "Sequence" not in cfg or "frameRate" not in cfg["Sequence"]:
        raise ValueError(f"{ini_path}: missing [Sequence] frameRate")
    if "seqLength" not in cfg["Sequence"]:
        raise ValueError(f"{ini_path}: missing [Sequence] seqLength")
    rate = float(cfg["Sequence"]["frameRate"])
    if not rate.is_integer():
        raise ValueError(f"{ini_path}: non-integer frameRate {rate} unsupported")
    return int(rate), int(cfg["Sequence"]["seqLength"])


def load_scenario_visibility(scenario_dir: Path) -> dict[int, AgentTrace]:
    """Parse one scenario's gt.txt into a dense per-agent visibility trace.

    Fails fast if an agent's frames are not a dense contiguous range (this
    project relies on that to define runs by simple frame arithmetic).
    """
    gt_path = scenario_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt missing under {scenario_dir}")
    fps, _seq_length = _read_seqinfo(scenario_dir)
    scenario_id = scenario_dir.name

    rows = gt_path.read_text(encoding="utf-8").strip().splitlines()
    if not rows:
        raise ValueError(f"{gt_path}: empty gt.txt")
    arr = np.array([[float(x) for x in r.split(",")] for r in rows], dtype=np.float64)

    out: dict[int, AgentTrace] = {}
    for agent_id in np.unique(arr[:, COL_ID]).astype(np.int64):
        sub = arr[arr[:, COL_ID] == agent_id]
        order = np.argsort(sub[:, COL_FRAME])
        sub = sub[order]
        frames = sub[:, COL_FRAME].astype(np.int64)
        expected = np.arange(frames[0], frames[0] + len(frames))
        if not np.array_equal(frames, expected):
            raise ValueError(
                f"{gt_path}: agent {agent_id} frames are not dense/contiguous "
                f"(got {frames[0]}..{frames[-1]}, {len(frames)} rows)"
            )
        out[int(agent_id)] = AgentTrace(
            scenario_id=scenario_id,
            agent_id=int(agent_id),
            frames=frames,
            vis=sub[:, COL_VIS].astype(np.float64),
            fps=fps,
        )
    return out


def extract_agent_runs(
    trace: AgentTrace,
    vis_lo: float = DEFAULT_VIS_LO,
    vis_hi: float = DEFAULT_VIS_HI,
    min_len: int = DEFAULT_MIN_LEN,
) -> tuple[list[OcclusionRun], bool]:
    """Extract all maximal low-visibility runs for one agent's trace.

    Returns ``(runs, is_fully_occluded)`` where ``is_fully_occluded`` flags an
    agent that never once reaches ``vis_hi`` anywhere in its trace (the P1 D23
    "walker-6" render-artifact pattern) -- callers should exclude such agents
    from aggregate statistics rather than counting the whole trace as one
    segment, since it is not evidence of a real transient occlusion.
    """
    if vis_lo > vis_hi:
        raise ValueError("vis_lo must not exceed vis_hi")
    frames, vis = trace.frames, trace.vis
    visible_idx = np.flatnonzero(vis >= vis_hi)
    if visible_idx.size == 0:
        return [], True

    runs: list[OcclusionRun] = []

    def _maybe_add(lo_i: int, hi_i: int, left_anchored: bool, right_anchored: bool) -> None:
        # low-visibility span is frames[lo_i .. hi_i] inclusive (exclusive of anchors)
        length = hi_i - lo_i + 1
        if length < min_len:
            return
        span = vis[lo_i : hi_i + 1]
        if span.min() >= vis_lo:
            return  # never dips below vis_lo -> partial occlusion, not a segment
        runs.append(
            OcclusionRun(
                scenario_id=trace.scenario_id,
                agent_id=trace.agent_id,
                start_frame=int(frames[lo_i]),
                end_frame=int(frames[hi_i]),
                fps=trace.fps,
                left_anchored=left_anchored,
                right_anchored=right_anchored,
            )
        )

    first_anchor, last_anchor = int(visible_idx[0]), int(visible_idx[-1])

    # Leading unanchored run (before the first visible anchor) -- M2 candidate.
    if first_anchor > 0:
        _maybe_add(0, first_anchor - 1, left_anchored=False, right_anchored=True)

    # Interior runs, strictly between consecutive anchors -- P1-D14-exact.
    for a, b in zip(visible_idx[:-1], visible_idx[1:], strict=True):
        if b - a - 1 <= 0:
            continue
        _maybe_add(a + 1, b - 1, left_anchored=True, right_anchored=True)

    # Trailing unanchored run (after the last visible anchor) -- never reappears.
    if last_anchor < len(vis) - 1:
        _maybe_add(last_anchor + 1, len(vis) - 1, left_anchored=True, right_anchored=False)

    return runs, False


def classify_agent_pattern(
    runs: list[OcclusionRun],
    flicker_lo_s: float = FLICKER_LO_S,
    flicker_hi_s: float = FLICKER_HI_S,
) -> str:
    """Classify one agent's occlusion pattern per context.md D-N1-2.

    - "none": zero qualifying runs.
    - "M1" block: exactly one run, anchored on both sides (visible before/after).
    - "M2" prefix: exactly one run, unanchored on the left (occluded since trace
      start) and anchored on the right (reappears and stays visible).
    - "M3" flicker: 2-4 runs total (context.md's primary criterion is the count;
      the fraction whose runs ALL sit within the strict [0.2, 0.6]s spec band is
      reported separately as a diagnostic, not used to gate classification).
    - "other_trailing": exactly one run, unanchored on the right (never
      reappears before the trace ends) -- relevant to the R-B "still-occluded"
      regime, but outside the M1/M2/M3 taxonomy.
    - "other_many": more than 4 runs.
    """
    n = len(runs)
    if n == 0:
        return "none"
    if n == 1:
        r = runs[0]
        if r.left_anchored and r.right_anchored:
            return "M1"
        if not r.left_anchored and r.right_anchored:
            return "M2"
        return "other_trailing"
    if 2 <= n <= 4:
        return "M3"
    return "other_many"


def _in_strict_flicker_band(runs: list[OcclusionRun]) -> bool:
    return all(FLICKER_LO_S <= r.duration_s <= FLICKER_HI_S for r in runs)


def fit_lognormal(durations_s: np.ndarray) -> LogNormalFit:
    """Fit scipy.stats.lognorm to segment durations (seconds), loc pinned at 0
    (durations are strictly positive by construction)."""
    if durations_s.size == 0:
        raise ValueError("cannot fit log-normal to an empty duration array")
    if np.any(durations_s <= 0):
        raise ValueError("durations must be strictly positive")
    shape, loc, scale = scipy_stats.lognorm.fit(durations_s, floc=0)
    dist = scipy_stats.lognorm(shape, loc=loc, scale=scale)
    return LogNormalFit(
        shape=float(shape),
        loc=float(loc),
        scale=float(scale),
        median_s=float(dist.median()),
        mean_s=float(dist.mean()),
        n=int(durations_s.size),
    )


def _iter_scenario_dirs(scenarios_root: Path) -> list[Path]:
    if not scenarios_root.exists():
        raise FileNotFoundError(f"scenarios root not found: {scenarios_root}")
    dirs = sorted(
        p for p in scenarios_root.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists()
    )
    if not dirs:
        raise ValueError(f"no scenario directories with gt/gt.txt found under {scenarios_root}")
    return dirs


def build_p2_stats(
    scenarios_root: Path,
    vis_lo: float = DEFAULT_VIS_LO,
    vis_hi: float = DEFAULT_VIS_HI,
    min_len: int = DEFAULT_MIN_LEN,
) -> ExtractionResult:
    """Top-level orchestrator: parse every scenario, extract occlusion runs,
    fit the duration distribution, compute rate/spacing, and classify the
    M1/M2/M3 pattern mix."""
    scenario_dirs = _iter_scenario_dirs(scenarios_root)

    all_runs: list[OcclusionRun] = []
    excluded_artifact_agents: list[tuple[str, int]] = []
    pattern_counts: dict[str, int] = {}
    n_agents_total = 0
    n_agents_with_runs = 0
    spacing_samples: list[float] = []
    fps_seen: set[int] = set()
    nominal_duration_seen: set[float] = set()
    total_exposure_s = 0.0
    m3_strict_flags: list[bool] = []

    for sd in scenario_dirs:
        scen_fps, seq_length = _read_seqinfo(sd)
        nominal_duration_seen.add(seq_length / scen_fps)
        traces = load_scenario_visibility(sd)
        for agent_id, trace in traces.items():
            n_agents_total += 1
            fps_seen.add(trace.fps)
            runs, is_artifact = extract_agent_runs(trace, vis_lo, vis_hi, min_len)
            if is_artifact:
                excluded_artifact_agents.append((sd.name, agent_id))
                continue
            # exposure = this agent's own observed presence window, NOT the
            # nominal scenario length -- some walkers exit/enter mid-scenario
            # (e.g. crossing_paths_0007 agent 4 spans frames 11-339 only).
            total_exposure_s += len(trace.frames) / trace.fps
            if runs:
                n_agents_with_runs += 1
            all_runs.extend(runs)
            pattern = classify_agent_pattern(runs)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            if pattern == "M3":
                m3_strict_flags.append(_in_strict_flicker_band(runs))
            # spacing: gap between consecutive runs (visible time between them)
            ordered = sorted(runs, key=lambda r: r.start_frame)
            for prev, nxt in zip(ordered[:-1], ordered[1:], strict=True):
                gap_frames = nxt.start_frame - prev.end_frame - 1
                spacing_samples.append(gap_frames / trace.fps)

    if not all_runs:
        raise ValueError("no occlusion segments extracted from any scenario")
    if len(fps_seen) != 1:
        raise ValueError(f"non-uniform fps across scenarios, cannot assume one rate: {fps_seen}")
    fps = fps_seen.pop()
    if len(nominal_duration_seen) != 1:
        raise ValueError(
            f"non-uniform nominal scenario duration across scenarios: {nominal_duration_seen}"
        )
    scenario_duration_s = nominal_duration_seen.pop()

    durations_s = np.array([r.duration_s for r in all_runs], dtype=np.float64)
    duration_fit = fit_lognormal(durations_s)

    n_agents_analyzed = n_agents_total - len(excluded_artifact_agents)
    segments_per_agent_mean = len(all_runs) / n_agents_analyzed
    # rate normalized by each agent's OWN observed exposure time (not the
    # nominal scenario length), since not every agent is present the full
    # scenario duration.
    segments_per_agent_second = len(all_runs) / total_exposure_s

    spacing_arr = np.array(spacing_samples, dtype=np.float64)
    spacing_median_s = float(np.median(spacing_arr)) if spacing_arr.size else float("nan")
    spacing_mean_s = float(np.mean(spacing_arr)) if spacing_arr.size else float("nan")

    total_classified = sum(pattern_counts.values())
    pattern_fractions = {
        k: v / total_classified for k, v in pattern_counts.items()
    } if total_classified else {}

    # Direct comparison group for the v1 prior (0.6/0.25/0.15 sums to 1.0 over
    # M1/M2/M3 ONLY) -- excludes "none" (not occluded) and the two taxonomy
    # leftovers (other_trailing, other_many) that the v1 prior has no slot for.
    m1m2m3_total = sum(pattern_counts.get(k, 0) for k in ("M1", "M2", "M3"))
    m1m2m3_fractions = (
        {k: pattern_counts.get(k, 0) / m1m2m3_total for k in ("M1", "M2", "M3")}
        if m1m2m3_total
        else {}
    )

    m3_strict_band_fraction = (
        float(np.mean(m3_strict_flags)) if m3_strict_flags else float("nan")
    )

    both_anchored = [r for r in all_runs if r.left_anchored and r.right_anchored]
    both_anchored_gap_median_frames = (
        float(np.median([r.duration_frames for r in both_anchored]))
        if both_anchored
        else float("nan")
    )

    return ExtractionResult(
        n_scenarios=len(scenario_dirs),
        n_agents_total=n_agents_total,
        excluded_artifact_agents=excluded_artifact_agents,
        n_agents_analyzed=n_agents_analyzed,
        fps=fps,
        scenario_duration_s=scenario_duration_s,
        runs=all_runs,
        duration_fit=duration_fit,
        segments_per_agent_mean=segments_per_agent_mean,
        segments_per_agent_second=segments_per_agent_second,
        spacing_median_s=spacing_median_s,
        spacing_mean_s=spacing_mean_s,
        n_spacing_samples=int(spacing_arr.size),
        pattern_counts=pattern_counts,
        pattern_fractions=pattern_fractions,
        m1m2m3_fractions=m1m2m3_fractions,
        n_agents_with_runs=n_agents_with_runs,
        m3_strict_band_fraction=m3_strict_band_fraction,
        both_anchored_segment_count=len(both_anchored),
        both_anchored_gap_median_frames=both_anchored_gap_median_frames,
        mot17_anchor={
            "median_gap_frames": MOT17_ANCHOR_MEDIAN_GAP_FRAMES,
            "median_gap_s": MOT17_ANCHOR_MEDIAN_GAP_S,
            "source": (
                "context.md D-N1-2, citing P1 D14 (MOT17-02 single-sequence figure); "
                "NOT used to fit the P2 duration distribution, sanity anchor only"
            ),
        },
    )


def dump_json(result: ExtractionResult, dest: Path, source_descriptor: str,
              extracted_date: str) -> None:
    """Write the committable stats-only JSON artifact (no raw trajectories/frames).

    ``source_descriptor`` must be a portable, machine-independent description of
    the source (no absolute local paths — this JSON enters public git history).
    """
    payload = result.to_dict()
    payload["source"] = source_descriptor
    payload["extracted_date"] = extracted_date
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
