# Third-party notices

This project's own code (`src/otraj/`, `scripts/`, `tests/`) is Apache-2.0
(see `LICENSE`). This file lists everything it depends on or uses: Python
package dependencies, vendored third-party repositories, the dataset, and the
compute provider.

Verified 2026-09-26 by reading installed package metadata
(`importlib.metadata`) inside `.venv` and `.venv-qcnet`, and by inspecting
`third_party/*/LICENSE` and this repo's own `results/*_vendor_manifest.json`
and `results/*_ckpt_manifest.json` files. "see upstream" means the installed
package metadata carries no machine-readable license field; check the
package's own repository/PyPI page.

## Vendored third-party code (`third_party/`, gitignored — never tracked)

| Package | Repo | Pinned commit | License | Notes |
|---|---|---|---|---|
| Forecast-MAE | jchengai/forecast-mae | `cb86ea92601d23a8af7713389e9fb78d7e546a65` | **NONE declared** (no LICENSE file, GitHub metadata null, verified 2026-07-26) | Not redistributable. Wrap-and-import only, no file ever copied into `src/`. Demoted from training baseline (D-N1-13); its checkpoint links are dead (D-N1-6). `results/fmae_vendor_manifest.json`. |
| QCNet | ZikangZhou/QCNet | `55cacb418cbbce3753119c1f157360e66993d0d0` | Apache-2.0 | Used only as the G-N1-0 eval-kit sanity anchor (its released checkpoint), never the training baseline. `results/qcnet_vendor_manifest.json`. |
| SIMPL | HKUST-Aerial-Robotics/SIMPL | `2a33314c92f42feee2b1b1b863f394672cd23eca` | MIT (LICENSE file present, verified 2026-07-26) | **Training baseline** (D-N1-13). AV2 checkpoint ships inside the SIMPL repo itself at `third_party/SIMPL/saved_models/simpl_av2_bezier_ckpt-fix.tar` (author retrain; SHA256 `689f1e7c71129a9745b17dba998baf91120b578fe749cd413bf6564f6c911337`, 30.6 MB) — this is upstream's own file, obtained by cloning SIMPL at the pinned commit, not something this project trained or hosts. `results/simpl_vendor_manifest.json`. |

None of the three vendored repos or their checkpoints are tracked by git in
this repo (`third_party/` is in `.gitignore`; `git ls-files | grep third_party`
returns nothing). Rebuild by cloning each repo and checking out the pinned
commit — see `RESUME.md`.

## Dataset — Argoverse 2

- **License: CC BY-NC-SA 4.0** (Argoverse 2's published terms of use —
  non-commercial, share-alike, attribution required). This project's own
  local docs/manifests (`results/av2_verification.json`, `results/av2_manifests/`)
  record structure/hash verification but do not themselves restate the
  license text; the CC BY-NC-SA 4.0 designation is Argoverse 2's standing
  public license and was not re-verified against a bundled license file in
  this repo's data.
- **Used for**: (a) evaluation (official `av2` eval kit, minADE₆/minFDE₆/MR₆)
  and (b) training the SIMPL experimental arms (C1/C2/C3-local pilot; the
  gated full-scale matrix would also train on it).
- **No AV2-trained weights are released.** `checkpoints/local/*.tar` (SIMPL
  arms trained on AV2, this project) are gitignored and were never committed
  or pushed — verified via `git ls-files` (no `.ckpt/.pt/.pth/.tar/.safetensors/.zip`
  tracked anywhere in this repo) and `gh release list -R matus012/occlusion_forecast`
  (no releases exist on the repo).
- No raw AV2 data, derived tensors, or dataset-derived pixels are ever
  tracked in git — enforced by `tests/test_license_guard.py` (`test_no_tracked_files_under_data_or_cache`,
  `test_no_tracked_binary_blobs_outside_fixtures`, `test_no_large_tracked_files_outside_visual_allowlist`),
  which passes as of 2026-09-26.

## Compute — PERUN (TUKE HPC)

TUKE's PERUN cluster (208× NVIDIA H200), available to this project by request.
The full-scale experiment was sized at about 260 H200 GPU-hours (from measured
local throughput) but never run. No PERUN compute has been consumed by this
project.

## Python dependencies

Two separate `uv` venvs (kept apart so the QCNet eval-only environment's
newer torch/PyG stack never touches the pinned SIMPL/FMAE env):

- `.venv` — main env, `requirements.txt`, Python 3.11.9, `torch==2.13.0+cu126`.
- `.venv-qcnet` — QCNet eval-kit-sanity env only, `requirements-qcnet.txt`,
  Python 3.11.9, `torch==2.7.0+cu126` + PyTorch Geometric.

License column: value from the installed distribution's `License` metadata
field where present, else the OSI `Classifier` license, else "see upstream"
(package carries no machine-readable license — check its PyPI/repo page).

### `.venv` (main — `requirements.txt`)

| Package | Version | License |
|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF-2.0 |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache-2.0 |
| argcomplete | 3.7.0 | Apache-2.0 |
| attrs | 26.1.0 | see upstream (MIT) |
| av | 18.0.0 | see upstream (BSD-3-Clause) |
| av2 | 0.3.6 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| click | 8.4.2 | see upstream (BSD-3-Clause) |
| colorama | 0.4.6 | BSD |
| colorlog | 6.12.0 | MIT |
| contourpy | 1.3.3 | BSD-3-Clause |
| coverage | 7.15.2 | Apache-2.0 |
| cycler | 0.12.1 | BSD |
| dependency-groups | 1.3.1 | see upstream |
| distlib | 0.4.3 | PSF-2.0 |
| filelock | 3.29.0 | MIT |
| fonttools | 4.63.0 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2026.4.0 | see upstream (BSD-3-Clause) |
| humanize | 4.16.0 | see upstream (MIT) |
| idna | 3.18 | see upstream (BSD-3-Clause) |
| imageio | 2.37.4 | see upstream (BSD-2-Clause) |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause |
| iniconfig | 2.3.0 | see upstream (MIT) |
| jinja2 | 3.1.6 | BSD |
| joblib | 1.5.3 | see upstream (BSD-3-Clause) |
| kiwisolver | 1.5.0 | BSD |
| kornia | 0.8.3 | Apache-2.0 |
| kornia-rs | 0.1.14 | Apache-2.0 |
| lightning-utilities | 0.15.3 | Apache-2.0 |
| llvmlite | 0.48.0 | see upstream (BSD-2-Clause) |
| markdown | 3.10.2 | see upstream (BSD-3-Clause) |
| markdown-it-py | 4.2.0 | MIT |
| markupsafe | 3.0.3 | see upstream (BSD-3-Clause) |
| matplotlib | 3.11.1 | PSF-based (matplotlib license) |
| mdurl | 0.1.2 | MIT |
| mpmath | 1.3.0 | BSD |
| multidict | 6.7.1 | Apache-2.0 |
| networkx | 3.6.1 | see upstream (BSD-3-Clause) |
| nox | 2026.7.11 | see upstream (Apache-2.0) |
| numba | 0.66.0 | BSD |
| numpy | 2.4.4 | see upstream (BSD-3-Clause) |
| opencv-python | 5.0.0.93 | Apache-2.0 |
| pandas | 3.0.5 | BSD-3-Clause |
| pathlib-abc | 0.5.2 | PSF-2.0 |
| pillow | 12.2.0 | see upstream (MIT-CMU) |
| platformdirs | 4.11.0 | MIT |
| pluggy | 1.6.0 | MIT |
| polars | 1.43.0 | MIT |
| polars-runtime-32 | 1.43.0 | MIT |
| propcache | 0.5.2 | Apache-2.0 |
| pyarrow | 25.0.0 | see upstream (Apache-2.0) |
| pygments | 2.20.0 | see upstream (BSD-2-Clause) |
| pyparsing | 3.3.2 | see upstream (MIT) |
| pyproj | 3.7.2 | see upstream (MIT) |
| pytest | 9.1.1 | see upstream (MIT) |
| pytest-cov | 7.1.0 | MIT |
| pytest-timeout | 2.4.0 | MIT |
| python-dateutil | 2.9.0.post0 | Dual: BSD / Apache-2.0 |
| python-discovery | 1.5.0 | MIT |
| pytorch-lightning | 2.6.5 | Apache-2.0 |
| pyyaml | 6.0.3 | MIT |
| rich | 15.0.0 | MIT |
| ruff | 0.16.0 | see upstream (MIT) |
| scipy | 1.17.1 | BSD |
| setuptools | 78.1.0 | MIT |
| shapely | 2.1.2 | BSD-3-Clause |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD |
| torch | 2.13.0+cu126 | see upstream (BSD-3-Clause) |
| torchmetrics | 1.9.0 | Apache-2.0 |
| torchvision | 0.28.0+cu126 | BSD |
| tqdm | 4.69.1 | MPL-2.0 AND MIT |
| typing-extensions | 4.15.0 | see upstream (PSF-2.0) |
| tzdata | 2026.3 | Apache-2.0 |
| universal-pathlib | 0.3.10 | see upstream (MIT) |
| virtualenv | 21.7.0 | MIT |
| yarl | 1.24.5 | Apache-2.0 |

### `.venv-qcnet` (QCNet eval-only — `requirements-qcnet.txt`)

Same core set as above with these QCNet-specific differences (packages
identical to the `.venv` table above are not repeated):

| Package | Version | License |
|---|---|---|
| charset-normalizer | 3.4.9 | MIT |
| numpy | 1.26.4 | see upstream (BSD-3-Clause) — pinned lower than main env for PyG compat |
| opencv-python | 4.11.0.86 | Apache-2.0 |
| pillow | 12.3.0 | see upstream (MIT-CMU) |
| psutil | 7.2.2 | BSD-3-Clause |
| requests | 2.34.2 | Apache-2.0 |
| torch | 2.7.0+cu126 | BSD-3-Clause |
| torch-cluster | 1.6.3+pt27cu126 | see upstream (MIT) |
| torch-geometric | 2.8.0.post1 | see upstream (MIT) |
| torch-scatter | 2.1.2+pt27cu126 | see upstream (MIT) |
| torchvision | 0.22.0+cu126 | BSD |
| urllib3 | 2.7.0 | see upstream (MIT) |
| xxhash | 3.8.1 | BSD-2-Clause |

## Checkpoints — source and re-downloadability

See `RESUME.md` §"Checkpoints" for the file-by-file manifest coverage
verdict; summary: the QCNet eval-anchor checkpoint is externally sourced and
manifest-pinned (re-downloadable), the four local SIMPL training checkpoints
under `checkpoints/local/` are **not** externally hosted anywhere and are
regenerable only by re-running this project's own training script.
