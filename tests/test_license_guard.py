"""AV2 license-hygiene guard: no dataset-derived content may be TRACKED by git.

Binding rule (ported from P1 D37/D41; public repo from day 1 makes this
load-bearing — there is no retroactive-purge budget on public history):
Argoverse 2 raw data, derived per-scenario tensors, map/trajectory dumps, and
rendered viz pixels must never enter the repo or any remote. Committable classes
are ONLY: loaders, manifests, SHA256 records, download docs, metric JSONs, and
explicitly allowlisted human-classified visuals.

Convention for visuals: every tracked image/video/animation must be explicitly
classified in ALLOWED_TRACKED_VISUALS below as "plot" (pure matplotlib, no
dataset pixels) or "schematic" (synthetic illustration). AV2-derived pixels are
never allowlistable without a documented license check (would require a new
class + decision-log entry).

Runs under pytest, so scripts/check_gates.py G_quality enforces it on every
gate check and in CI.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGE_EXTS = (".jpg", ".jpeg", ".bmp", ".webp", ".ppm", ".tif", ".tiff")
# Any serialized container that could hold dataset scenarios/tensors/embeddings.
# AV2 ships scenarios and maps as parquet — explicitly included.
BLOB_EXTS = (".pt", ".pth", ".ckpt", ".npz", ".npy", ".onnx", ".engine", ".tar",
             ".pkl", ".pickle", ".h5", ".hdf5", ".lmdb", ".mdb", ".arrow",
             ".parquet", ".feather", ".msgpack")
# Default-deny size ceiling: big binaries cannot slip in under an unlisted
# extension; anything over the ceiling must be an allowlisted visual.
SIZE_CEILING_BYTES = 2 * 1024 * 1024


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True,
        timeout=60,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def test_no_tracked_files_under_data_or_cache() -> None:
    offenders = [
        f for f in tracked_files()
        if f.startswith(("data/", "cache/", "checkpoints/", "third_party/"))
    ]
    assert not offenders, f"dataset/vendored content tracked: {offenders}"


def test_no_tracked_raw_images_anywhere() -> None:
    offenders = [f for f in tracked_files() if f.lower().endswith(RAW_IMAGE_EXTS)]
    assert not offenders, f"raw image files tracked (license risk): {offenders}"


def test_no_tracked_binary_blobs_outside_fixtures() -> None:
    offenders = [
        f for f in tracked_files()
        if f.lower().endswith(BLOB_EXTS) and not f.startswith("tests/fixtures/")
    ]
    assert not offenders, f"serialized tensors/scenario dumps tracked: {offenders}"


# Every tracked visual must be EXPLICITLY classified here — a deliberate human
# classification step (P1 D41 pattern). Classes:
#   "plot"      — pure matplotlib/vector output, zero dataset pixels
#   "schematic" — synthetic illustration/animation, zero dataset pixels
# AV2-derived pixels have NO valid class; adding one requires a license check
# and a D-N1-x decision-log entry first.
VISUAL_EXTS = (".png", ".gif", ".mp4", ".avi", ".webm", ".svg")
ALLOWED_TRACKED_VISUALS: dict[str, str] = {}


def test_tracked_visuals_are_allowlisted() -> None:
    """Every tracked image/video must be explicitly classified."""
    visuals = [f for f in tracked_files() if f.lower().endswith(VISUAL_EXTS)]
    unclassified = [f for f in visuals if f not in ALLOWED_TRACKED_VISUALS]
    assert not unclassified, (
        f"tracked visuals not in the allowlist (classify as 'plot' or "
        f"'schematic' in tests/test_license_guard.py, or keep them local): "
        f"{unclassified}"
    )


def test_no_large_tracked_files_outside_visual_allowlist() -> None:
    """Default-deny for big binaries: a scenario cache renamed to an unlisted
    extension still fails here."""
    offenders = []
    for f in tracked_files():
        p = ROOT / f
        if p.exists() and p.stat().st_size > SIZE_CEILING_BYTES:
            if f not in ALLOWED_TRACKED_VISUALS:
                offenders.append(f"{f} ({p.stat().st_size / 1e6:.1f} MB)")
    assert not offenders, f"large tracked files outside the visual allowlist: {offenders}"


def test_allowlist_itself_is_clean() -> None:
    """The allowlist may never contain dataset-pixel content by construction."""
    for path, cls in ALLOWED_TRACKED_VISUALS.items():
        assert cls in ("plot", "schematic"), f"{path}: unknown class {cls!r}"
