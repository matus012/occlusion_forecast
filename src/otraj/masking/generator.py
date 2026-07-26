"""Phase 3 occlusion mask generator (D-N1-2 as amended by D-N1-9 / D-N1-10).

Domain: AV2 motion-forecasting observed history = 50 steps @ 10Hz (5s), index
0..49, where index 49 == t=0 (most recent observed step; index 0 == t=-4.9s).
A mask is a boolean array of shape (50,): True == OCCLUDED (the per-timestep
validity flag is flipped to 0 downstream). This module ONLY emits index masks
-- there is no imputation anywhere here; each baseline zeroes/pads masked
positions per its OWN native convention (D-N1-7a: FMAE flag+zero vs SIMPL
flag+nn-pad) at the point of consumption, not here.

Patterns (D-N1-2, D-N1-9 amendment):
  M1 block   -- one contiguous occlusion run. Its FINAL length is DETERMINED
                by the severity target, not by the lognormal draw: M1 has
                exactly one block, so "scale to hit the target" collapses to
                length = target_steps exactly, regardless of what was drawn.
                A raw duration is still sampled from the P2 fit (clipped to
                [0.3, 4.0]s) to keep the code path uniform with M3, but it is
                STRUCTURALLY INERT for M1 over S1-S4 (targets 1.0/2.0/3.0/4.0s
                all already sit inside the clip) -- do not claim elsewhere
                that the fit parameterizes M1's duration; it doesn't. P2
                stats enter the mask DISTRIBUTION via the pattern MIX and
                M3's relative block-size ratios only (see M3 below).
  M2 prefix  -- occluded from index 0 until an onset index (late appearance).
                Length is directly the severity target; no duration is drawn
                at all (a prefix is structurally just "target_steps of
                leading occlusion").
  M3 flicker -- 2-4 short runs. Each run's RAW size is drawn from the fitted
                log-normal (clipped to [0.2, 2.0]s -- D-N1-9 amendment: the
                v1 [0.2, 0.6]s band fit only 9% of real flicker runs), then
                ALL runs are scaled together, preserving those RELATIVE
                proportions, to hit the severity target exactly. This is the
                one pattern where the P2 fit's shape genuinely influences the
                output (via the relative sizing of the 2-4 blocks).

Severity buckets target a masked fraction of the 50 steps (S0=0, S1=0.2,
S2=0.4, S3=0.6, S4=0.8), ACHIEVED EXACTLY in practice (well inside the
+/-0.05 hard, tested tolerance): each bucket's target_steps is already an
exact integer (10/20/30/40 of 50), and the scale-then-integerize step always
redistributes rounding remainder to hit that exact sum. PRECEDENCE: the
fraction guarantee always wins over the per-pattern duration clip -- scaling
small targets across several M3 blocks can push an individual block's
duration below/above its [0.2, 2.0]s clip band; this is intentional and
covered by tests, never silently "fixed" by breaking the fraction guarantee
instead.

Regimes:
  R-A "re-emerged"     -- indices 45..49 (last 5 steps) are FORCED VISIBLE;
                          the mask may never touch them. Headline regime.
  R-B "still-occluded" -- the mask MUST cover index 49 (extends through t=0).
                          This guarantee only binds when there IS a mask
                          (severities S1-S4); S0 stays the universal empty
                          mask in every regime (see generate_mask()).

R-B x M2 resolution (read this before touching REGIME/PATTERN logic):
A prefix (M2) starts at index 0 by definition. For R-B to also require index
49 masked, the prefix would have to run [0, 49] -- i.e. the ENTIRE 50-step
history occluded. That is not "occluded, still occluded at t=0" as a
DEGREE-of-occlusion scenario; it is "the agent was never observed at all",
which breaks the fraction guarantee at every bucket below S4 and is not a
meaningful masked-history training/eval condition (it mirrors physical
reality: appeared-late AND still-occluded-at-t=0 == never seen == no
scenario). RESOLUTION: R-B pattern support is limited to M1 and M3 (the
covering block, or the last flicker block in temporal order, is anchored to
end exactly at index 49 -- the only way to "cover 49" without exceeding the
history bound, since 49 is also the last valid index). M2 is EXCLUDED from
the R-B pattern mix; M1/M3 are renormalized to sum to 1, computed from
MIX_RA (never a hardcoded rounded copy) -- see MIX_RB and RB_MIX_NOTE below.

Determinism (core requirement): every scenario's eval mask is a pure function
of (scenario_id, severity, regime, spec_version) -- see generate_mask(). It is
built from TWO independent sha256-derived streams (see SEED_DERIVATION_NOTE):
  - pattern_seed(scenario_id, regime, spec_version) -- SEVERITY-INDEPENDENT.
    A scenario draws exactly ONE pattern per regime, shared across every
    severity bucket. This is load-bearing for D-N1-10 per-pattern slicing:
    the same scenario must land in the same pattern cohort at S1, S2, S3, S4
    so per-pattern paired statistical tests are comparing the same
    population across the severity curve. (Folding severity into the pattern
    draw was a real bug found in review: it fragmented per-pattern cohorts
    across severities down to near-zero overlap.)
  - placement_seed(scenario_id, severity, regime, spec_version) --
    severity-DEPENDENT, used only for duration sampling + block placement
    once the pattern is already fixed.
No global RNG state, no wall-clock/date input, identical across processes
and platforms.

Train-time API (data-aug semantics, Phase 4): draw_train_mask() uses the same
sha256-seeding technique with "train|{epoch}" folded into the seed string, and
draws (p_occ gate, severity ~ uniform S1..S4, pattern ~ mix) FROM that single
seeded generator so the epoch genuinely changes the draw (not just a label).
The severity-independent-pattern requirement above is an EVAL-manifest
property (cross-severity cohort stability for paired tests); training draws a
fresh (severity, pattern) pair every epoch by design and does not need it.
Training only ever uses regime R-A (context.md D-N1-2).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------

N_STEPS = 50  # AV2 observed history length
HZ = 10.0  # AV2 observation rate
FORCED_VISIBLE_TAIL = 5  # R-A: indices N_STEPS - FORCED_VISIBLE_TAIL .. N_STEPS-1 never masked

SPEC_VERSION = "N1-mask-v2"  # bumped: v2 fixes severity-independent pattern draw (B1) + the
# stars-and-bars placement bias (M4) -- both change the mask distribution, so any manifest
# generated under "N1-mask-v1" is stale and must be regenerated.

PATTERNS: tuple[str, ...] = ("M1", "M2", "M3")
SEVERITIES: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")
REGIMES: tuple[str, ...] = ("R-A", "R-B")

SEVERITY_TARGETS: dict[str, float] = {
    "S0": 0.0,
    "S1": 0.2,
    "S2": 0.4,
    "S3": 0.6,
    "S4": 0.8,
}
SEVERITY_TOLERANCE = 0.05

# --------------------------------------------------------------------------
# Pattern mix (D-N1-9: empirical P2 mix, wins over the v1 prior)
# --------------------------------------------------------------------------

MIX_RA: dict[str, float] = {"M1": 0.40, "M2": 0.05, "M3": 0.55}

# R-B renormalization -- COMPUTED from MIX_RA (never a hardcoded rounded
# copy), M2 excluded (see module docstring "R-B x M2 resolution").
_RB_DENOM = MIX_RA["M1"] + MIX_RA["M3"]
MIX_RB: dict[str, float] = {
    "M1": MIX_RA["M1"] / _RB_DENOM,
    "M3": MIX_RA["M3"] / _RB_DENOM,
}

RB_MIX_NOTE = (
    "R-B ('still-occluded' at t=0) excludes M2 (prefix/late-appearance) from its "
    "pattern mix: a prefix pattern that also covers index 49 would require masking "
    "the ENTIRE 50-step history (a prefix starts at index 0 by definition; R-B "
    "requires index 49 masked), i.e. the agent was never observed at all -- not a "
    "graded masked-history scenario. Physically: 'appeared late AND still occluded "
    "at t=0' == never seen == no scenario. M1/M3 are renormalized to sum to 1.0, "
    f"computed from MIX_RA (M1={MIX_RA['M1']}, M3={MIX_RA['M3']}), not a hardcoded "
    f"rounded copy: M1={MIX_RB['M1']:.6f}, M3={MIX_RB['M3']:.6f}."
)

# --------------------------------------------------------------------------
# Duration clips (seconds) -- D-N1-9 amendment for M3
# --------------------------------------------------------------------------

M1_CLIP_S: tuple[float, float] = (0.3, 4.0)
M3_CLIP_S: tuple[float, float] = (0.2, 2.0)
M3_MIN_BLOCKS = 2
M3_MAX_BLOCKS = 4

# --------------------------------------------------------------------------
# Duration fit (P2 CARLA occlusion stats, results/p2_occlusion_stats.json,
# D-N1-9). results/p2_occlusion_stats.json is a TRACKED repo artifact, not an
# optional/external input -- its absence or corruption is a real
# configuration error and must fail fast, not be papered over with a
# fallback default (a stale silent fallback would let the generator drift
# from the single source of truth without any signal).
# --------------------------------------------------------------------------

_P2_STATS_PATH = Path(__file__).resolve().parents[3] / "results" / "p2_occlusion_stats.json"


def _load_duration_fit(path: Path = _P2_STATS_PATH) -> dict[str, float]:
    """Fail fast: no except clause. A missing file raises FileNotFoundError
    (via Path.read_text), a malformed/renamed schema raises KeyError, and
    corrupt JSON raises json.JSONDecodeError -- all real, actionable errors."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload["duration_fit_lognormal"]
    return {
        "shape": float(fit["shape"]),
        "loc": float(fit["loc"]),
        "scale": float(fit["scale"]),
    }


_FIT = _load_duration_fit()
FIT_SHAPE: float = _FIT["shape"]
FIT_LOC: float = _FIT["loc"]
FIT_SCALE: float = _FIT["scale"]

# --------------------------------------------------------------------------
# Train-time API constants
# --------------------------------------------------------------------------

TRAIN_P_OCC = 0.5
TRAIN_SEVERITIES: tuple[str, ...] = ("S1", "S2", "S3", "S4")


@dataclass(frozen=True)
class MaskResult:
    """One generated mask plus its provenance (D-N1-10: pattern label kept
    alongside the mask so eval manifests can slice per-pattern without
    regenerating masks)."""

    mask: np.ndarray  # bool, shape (N_STEPS,)
    pattern: str  # "M1" | "M2" | "M3" | "none" (S0, or a gated-off train draw)
    severity: str
    regime: str
    achieved_fraction: float
    n_masked: int
    seed: int


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

SEED_DERIVATION_NOTE = (
    "Two independent sha256-derived streams per scenario, each: sha256(seed_string) "
    "-> first 8 bytes (big-endian) -> non-negative int -> numpy.random.default_rng. "
    "No global RNG state, no wall-clock/date input. "
    "pattern_seed_string = f'{spec_version}|{scenario_id}|{regime}|pattern' "
    "(SEVERITY-INDEPENDENT -- one pattern per scenario per regime, shared across "
    "S1-S4; required for D-N1-10 per-pattern paired tests). "
    "placement_seed_string = f'{spec_version}|{scenario_id}|{severity}|{regime}' "
    "(severity-dependent; used only for duration sampling + block placement given "
    "the already-drawn pattern)."
)


def _seed_from_string(s: str) -> int:
    """sha256(s) -> first 8 bytes (big-endian) -> non-negative int. Pure,
    deterministic, no global state, no wall-clock/date input."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def pattern_seed(scenario_id: str, regime: str, spec_version: str = SPEC_VERSION) -> int:
    """Severity-INDEPENDENT seed: a scenario's pattern is fixed per regime
    across all severities (D-N1-10 per-pattern cohort stability)."""
    return _seed_from_string(f"{spec_version}|{scenario_id}|{regime}|pattern")


def placement_seed(
    scenario_id: str, severity: str, regime: str, spec_version: str = SPEC_VERSION,
) -> int:
    """Severity-dependent seed used for duration sampling + block placement,
    given a pattern already drawn from pattern_seed()."""
    return _seed_from_string(f"{spec_version}|{scenario_id}|{severity}|{regime}")


def mix_for_regime(regime: str) -> dict[str, float]:
    if regime == "R-A":
        return dict(MIX_RA)
    if regime == "R-B":
        return dict(MIX_RB)
    raise ValueError(f"unknown regime {regime!r}")


def target_steps_for_severity(severity: str) -> int:
    if severity not in SEVERITY_TARGETS:
        raise ValueError(f"unknown severity {severity!r}")
    return int(round(SEVERITY_TARGETS[severity] * N_STEPS))


def _draw_pattern(rng: np.random.Generator, mix: dict[str, float]) -> str:
    keys = [p for p in PATTERNS if p in mix]
    probs = np.array([mix[k] for k in keys], dtype=np.float64)
    cum = np.cumsum(probs)
    u = rng.random()
    idx = int(np.searchsorted(cum, u, side="right"))
    idx = min(idx, len(keys) - 1)
    return keys[idx]


# --------------------------------------------------------------------------
# Duration sampling + fraction-guarantee scaling
# --------------------------------------------------------------------------


def _sample_duration_steps(rng: np.random.Generator, clip_lo_s: float, clip_hi_s: float) -> int:
    """One duration draw from the fitted log-normal (seconds), clipped, then
    converted to steps @ HZ. Always >= 1 step. For M1 (n_blocks=1) this draw
    is scaled away to exactly target_steps regardless of its value -- see the
    module docstring's M1 note; it is NOT inert for M3, where it sets the
    RELATIVE size of each of the 2-4 blocks before scaling."""
    raw_s = float(rng.lognormal(mean=np.log(FIT_SCALE), sigma=FIT_SHAPE)) + FIT_LOC
    clipped_s = min(max(raw_s, clip_lo_s), clip_hi_s)
    steps = int(round(clipped_s * HZ))
    return max(steps, 1)


def _integerize_to_total(scaled: np.ndarray, target_total: int, min_each: int = 1) -> np.ndarray:
    """Round `scaled` (float lengths) to integers summing EXACTLY to
    `target_total`, each element >= min_each. This is the mechanism by which
    the fraction guarantee wins over per-block duration clips: `scaled` may
    already be outside a pattern's clip band in steps, and this function does
    not restore it -- it only enforces the exact-sum + minimum-length
    invariants needed for a valid, non-degenerate placement."""
    n = scaled.shape[0]
    if target_total < n * min_each:
        raise ValueError(
            f"target_total={target_total} cannot be split into {n} blocks with "
            f"min_each={min_each} steps"
        )
    base = np.maximum(np.floor(scaled).astype(np.int64), min_each)
    diff = int(target_total - base.sum())
    if diff > 0:
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac, kind="stable")
        i = 0
        while diff > 0:
            base[order[i % n]] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        order = np.argsort(-base, kind="stable")
        i = 0
        guard = 0
        while diff < 0:
            idx = order[i % n]
            if base[idx] > min_each:
                base[idx] -= 1
                diff += 1
            i += 1
            guard += 1
            if guard > 1_000_000:
                raise RuntimeError("failed to integerize lengths to target_total")
    return base


def _draw_and_scale_lengths(
    rng: np.random.Generator, n_blocks: int, clip_lo_s: float, clip_hi_s: float,
    target_steps: int,
) -> list[int]:
    """Draw `n_blocks` raw durations from the fitted distribution (clipped),
    then scale them (preserving relative proportions) so their sum is exactly
    `target_steps` -- the fraction-guarantee mechanism. For n_blocks=1 (M1)
    this always collapses to [target_steps] regardless of the raw draw."""
    raw = np.array(
        [_sample_duration_steps(rng, clip_lo_s, clip_hi_s) for _ in range(n_blocks)],
        dtype=np.float64,
    )
    scaled = raw * (target_steps / raw.sum())
    return _integerize_to_total(scaled, target_steps, min_each=1).tolist()


# --------------------------------------------------------------------------
# Block placement
# --------------------------------------------------------------------------


def _place_blocks(
    rng: np.random.Generator, lengths: list[int], domain_size: int,
    anchor_last_to_end: bool,
) -> list[tuple[int, int]]:
    """Place non-overlapping blocks of the given lengths, in temporal order,
    UNIFORMLY at random within [0, domain_size), via proper stars-and-bars.

    If `anchor_last_to_end`, the slack is distributed only BEFORE and BETWEEN
    blocks (no gap after the last one) so the last block's end is pinned to
    `domain_size - 1` by construction -- this is how R-B guarantees coverage
    of index 49 (the last valid index; a block cannot extend past it, so
    "covers 49" and "ends at 49" coincide).

    INTERNAL gaps (between two consecutive blocks, k > 1 i.e. M3) are forced
    to be >= 1 step: without this, two adjacent blocks with zero visible
    space between them would render as a single merged contiguous run,
    silently turning e.g. a drawn 4-block M3 into fewer than 2-4 observed
    runs. Leading/trailing gaps carry no such minimum.

    Uniformity: after subtracting the mandatory internal-gap minimums, the
    remaining `free_slack` is split into `n_gaps` non-negative integer parts
    via the standard stars-and-bars bijection -- choose (n_gaps - 1) DISTINCT
    "bar" positions without replacement from a pool of (free_slack + n_gaps -
    1) slots, sort them, and decode consecutive differences as gap sizes.
    (An earlier version drew bar positions WITH replacement via sorted
    `rng.integers`, which is NOT uniform over compositions -- it measurably
    under-weighted leading/trailing gaps by up to ~17%. `rng.choice(...,
    replace=False)` is the fix.)
    """
    k = len(lengths)
    total = sum(lengths)
    slack = domain_size - total
    if slack < 0:
        raise ValueError(f"blocks of total length {total} do not fit in domain {domain_size}")

    n_gaps = k if anchor_last_to_end else k + 1
    if k > 1:
        mins = [0] + [1] * (k - 1) + ([] if anchor_last_to_end else [0])
    else:
        mins = [0] * n_gaps
    min_total = sum(mins)
    free_slack = slack - min_total
    if free_slack < 0:
        raise ValueError(
            f"blocks of total length {total} plus the {min_total} mandatory "
            f"inter-block visible steps do not fit in domain {domain_size}"
        )

    m = n_gaps - 1  # number of stars-and-bars "bar" positions needed
    if m == 0:
        extra = [free_slack]
    else:
        total_slots = free_slack + m
        bars = np.sort(rng.choice(total_slots, size=m, replace=False))
        extra = []
        prev = -1
        for b in bars:
            extra.append(int(b) - prev - 1)
            prev = int(b)
        extra.append(total_slots - 1 - prev)
    gaps = [mn + ex for mn, ex in zip(mins, extra, strict=True)]

    spans: list[tuple[int, int]] = []
    cursor = gaps[0]
    for i, length in enumerate(lengths):
        start = cursor
        end = start + length - 1
        spans.append((start, end))
        cursor = end + 1
        if i + 1 < k:
            cursor += gaps[i + 1]
    return spans


# --------------------------------------------------------------------------
# Core mask construction (shared by eval + train entry points)
# --------------------------------------------------------------------------


def _build_pattern_mask(
    rng: np.random.Generator, pattern: str, severity: str, regime: str,
) -> tuple[np.ndarray, float, int]:
    mask = np.zeros(N_STEPS, dtype=bool)
    target_steps = target_steps_for_severity(severity)
    if target_steps == 0:
        # Mirrors the S0 short-circuit in generate_mask(): no blocks, no
        # placement, and (deliberately) no R-A/R-B invariant check below --
        # the empty mask trivially satisfies both regimes.
        return mask, 0.0, 0

    domain_size = (N_STEPS - FORCED_VISIBLE_TAIL) if regime == "R-A" else N_STEPS
    anchor = regime == "R-B"
    if target_steps > domain_size:
        raise ValueError(
            f"severity target {target_steps} steps exceeds maskable domain "
            f"{domain_size} for regime {regime}"
        )

    if pattern == "M1":
        lengths = _draw_and_scale_lengths(rng, 1, *M1_CLIP_S, target_steps)
        spans = _place_blocks(rng, lengths, domain_size, anchor)
    elif pattern == "M2":
        if regime == "R-B":
            raise ValueError("M2 is excluded from the R-B pattern mix (see RB_MIX_NOTE)")
        spans = [(0, target_steps - 1)]
    elif pattern == "M3":
        n_blocks = int(rng.integers(M3_MIN_BLOCKS, M3_MAX_BLOCKS + 1))
        lengths = _draw_and_scale_lengths(rng, n_blocks, *M3_CLIP_S, target_steps)
        spans = _place_blocks(rng, lengths, domain_size, anchor)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    for start, end in spans:
        mask[start:end + 1] = True

    n_masked = int(mask.sum())
    achieved_fraction = n_masked / N_STEPS

    # Load-bearing invariants: raised explicitly (NOT `assert`) so they
    # survive `python -O`, which strips assert statements.
    if regime == "R-A" and mask[N_STEPS - FORCED_VISIBLE_TAIL:].any():
        raise RuntimeError(
            "internal invariant violated: R-A forced-visible tail (indices "
            f"{N_STEPS - FORCED_VISIBLE_TAIL}..{N_STEPS - 1}) was masked -- "
            "this must never happen; _place_blocks has a bug if it does"
        )
    if regime == "R-B" and not bool(mask[N_STEPS - 1]):
        raise RuntimeError(
            f"internal invariant violated: R-B did not mask index {N_STEPS - 1} "
            "-- this must never happen; _place_blocks has a bug if it does"
        )

    return mask, achieved_fraction, n_masked


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_mask(
    scenario_id: str, severity: str, regime: str, spec_version: str = SPEC_VERSION,
) -> MaskResult:
    """Deterministic eval mask: a pure function of (scenario_id, severity,
    regime, spec_version). Identical across processes/platforms.

    The pattern is drawn from a SEVERITY-INDEPENDENT stream (pattern_seed):
    the same scenario_id + regime always yields the same pattern regardless
    of severity, so per-pattern cohorts (M1-only / M2-only / M3-only) are the
    SAME set of scenarios across S1-S4 -- required for D-N1-10 per-pattern
    paired statistical tests across the severity curve.
    """
    if severity not in SEVERITY_TARGETS:
        raise ValueError(f"unknown severity {severity!r}")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")

    if severity == "S0":
        # S0 = no mask, by spec definition. No pattern is drawn (there is
        # nothing to label) and the R-B "must cover index 49" invariant is
        # deliberately NOT enforced here -- it only binds when there IS a
        # mask (S1-S4); S0 stays the universal empty mask in every regime.
        seed = placement_seed(scenario_id, severity, regime, spec_version)
        return MaskResult(
            mask=np.zeros(N_STEPS, dtype=bool), pattern="none", severity=severity,
            regime=regime, achieved_fraction=0.0, n_masked=0, seed=seed,
        )

    p_seed = pattern_seed(scenario_id, regime, spec_version)
    pattern = _draw_pattern(np.random.default_rng(p_seed), mix_for_regime(regime))

    seed = placement_seed(scenario_id, severity, regime, spec_version)
    rng = np.random.default_rng(seed)
    mask, achieved_fraction, n_masked = _build_pattern_mask(rng, pattern, severity, regime)
    return MaskResult(
        mask=mask, pattern=pattern, severity=severity, regime=regime,
        achieved_fraction=achieved_fraction, n_masked=n_masked, seed=seed,
    )


def draw_train_mask(
    scenario_id: str, epoch: int, spec_version: str = SPEC_VERSION,
    p_occ: float = TRAIN_P_OCC,
) -> MaskResult:
    """Train-time data-aug draw: same sha256-seeding technique, but the seed
    string folds in "train|{epoch}" (not severity/regime) and (gate,
    severity, pattern) are ALL drawn from that single seeded generator, so
    varying the epoch genuinely changes the draw rather than just a label.
    Training only ever uses regime R-A (context.md D-N1-2). Unlike
    generate_mask(), there is no severity-independent-pattern requirement
    here -- a fresh (severity, pattern) pair every epoch is the intended
    data-aug semantics, not a bug."""
    seed = _seed_from_string(f"{spec_version}|{scenario_id}|train|{epoch}")
    rng = np.random.default_rng(seed)

    if rng.random() >= p_occ:
        return MaskResult(
            mask=np.zeros(N_STEPS, dtype=bool), pattern="none", severity="S0", regime="R-A",
            achieved_fraction=0.0, n_masked=0, seed=seed,
        )

    severity = str(rng.choice(np.array(TRAIN_SEVERITIES)))
    pattern = _draw_pattern(rng, MIX_RA)
    mask, achieved_fraction, n_masked = _build_pattern_mask(rng, pattern, severity, "R-A")
    return MaskResult(
        mask=mask, pattern=pattern, severity=severity, regime="R-A",
        achieved_fraction=achieved_fraction, n_masked=n_masked, seed=seed,
    )
