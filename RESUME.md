# RESUME.md — rebuilding the environment, data and checkpoints

This project is paused (the full-scale experiment needs cluster time, see
`README.md`). `data/`, `checkpoints/`, `.venv/` and `.venv-qcnet/` are not in
git. Everything below is written so those four things can be rebuilt from what
git tracks plus the manifests recorded here.

Sizes recorded 2026-09-26, right before deletion (so you know what you're
rebuilding and roughly how long it will take):

| Path | Size | Source of the number |
|---|---|---|
| `data/av2/train` | 50,464,592,446 B (≈47.0 GB), 199,908 scenarios, 399,816 files | `results/av2_verification.json` (measured 2026-07-27, aggregate SHA256 pinned) |
| `data/av2/val` | 6,313,637,276 B (≈5.9 GB), 24,988 scenarios, 49,976 files | `results/av2_verification.json` (measured 2026-07-26) |
| `data/av2_sensor` | ≈1.38 GB, 3,791 files | measured 2026-09-26 (`os.walk` sum); matches D-N1-15(5)'s "1.38 GB of a 2 GB budget" |
| `data/qcnet_root` | ≈11.36 GB, 74,964 files | measured 2026-09-26; QCNet's own preprocessing cache (`val/raw`, `val/processed`) for G-N1-0 |
| `data/simpl_features` | ≈3.34 GB, 23,500 files | measured 2026-09-26; SIMPL-preprocessed pickles for the `*-local` pilot (`data/simpl_features/local/{train,val}`) |
| `checkpoints/qcnet_av2.ckpt` | 89 MB | `results/qcnet_ckpt_manifest.json` |
| `checkpoints/local/*.tar` (4 files) | 31 MB each, ≈124 MB total | measured 2026-09-26 — **not manifest-covered, see "Checkpoints" below** |

## 0. Environment (`.venv`, `.venv-qcnet`)

Two venvs are used, deliberately kept apart (the QCNet eval-anchor needs a
newer torch/PyG stack than the pinned SIMPL/FMAE env). Python 3.11.9 both.

```powershell
# main env
uv venv .venv --python 3.11
.venv\Scripts\python.exe -m pip install --upgrade pip
# torch/torchvision come from the CUDA 12.6 wheel index; everything else from PyPI
uv pip install --python .venv\Scripts\python.exe torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# QCNet eval-only env
uv venv .venv-qcnet --python 3.11
uv pip install --python .venv-qcnet\Scripts\python.exe torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-qcnet\Scripts\python.exe -r requirements-qcnet.txt
# PyG wheels (torch-scatter/torch-cluster) come from the PyG wheel index, not PyPI:
uv pip install --python .venv-qcnet\Scripts\python.exe torch-geometric==2.8.0.post1 torch-scatter==2.1.2+pt27cu126 torch-cluster==1.6.3+pt27cu126 -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

Verify: `.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"` must print `True`. If it prints `False`, this is a GPU-intended project — stop and diagnose the environment (driver/CUDA mismatch) before running anything, per the environment-probe rule; do not train on CPU.

## 1. Vendored third-party code (`third_party/`)

Untracked by design (D-N1-5 — none of the three upstream repos are
redistributable/relicensable by this project). Re-clone at the pinned
commits (full detail + licenses in `THIRD_PARTY.md`):

```powershell
git clone https://github.com/jchengai/forecast-mae third_party/forecast-mae
git -C third_party/forecast-mae checkout cb86ea92601d23a8af7713389e9fb78d7e546a65

git clone https://github.com/ZikangZhou/QCNet third_party/QCNet
git -C third_party/QCNet checkout 55cacb418cbbce3753119c1f157360e66993d0d0

git clone https://github.com/HKUST-Aerial-Robotics/SIMPL third_party/SIMPL
git -C third_party/SIMPL checkout 2a33314c92f42feee2b1b1b863f394672cd23eca
```

The SIMPL clone brings its own author-shipped checkpoints in
`third_party/SIMPL/saved_models/*.tar` (including
`simpl_av2_bezier_ckpt-fix.tar`, sha256 `689f1e7c71129a9745b17dba998baf91120b578fe749cd413bf6564f6c911337`
— verify with `certutil -hashfile third_party\SIMPL\saved_models\simpl_av2_bezier_ckpt-fix.tar SHA256` on Windows). Nothing is downloaded separately for SIMPL.

## 2. Data (`data/`)

### 2a. AV2 Motion Forecasting (`data/av2/{train,val}`)

Public S3 bucket, anonymous access:

```powershell
aws s3 sync s3://argoverse/datasets/av2/motion-forecasting/train data/av2/train --no-sign-request
aws s3 sync s3://argoverse/datasets/av2/motion-forecasting/val data/av2/val --no-sign-request
```

Verify against the pinned structure/hash manifest (this is what produced
`results/av2_verification.json`):

```powershell
.venv\Scripts\python.exe scripts\verify_av2.py --split train
.venv\Scripts\python.exe scripts\verify_av2.py --split val
```

Expect `n_scenario_dirs` 199908 (train) / 24988 (val) and the aggregate
SHA256 values already recorded in `results/av2_verification.json` to match
(per-file manifests are gitignored under `results/av2_manifests/` — the
aggregate hash in the committed JSON is the check).

### 2b. AV2 Sensor (RGB demo inputs, `data/av2_sensor`)

**Do not bulk-download** — the full sensor split is ~1.06 TB (rejected,
D-N1-15(5)). Selective per-object fetch only, via
`scripts/fetch_sensor_log.py`:

```powershell
# metadata-only scout pool (annotations + ego pose + calibration + map), ~2.3 MB/log
.venv\Scripts\python.exe scripts\fetch_sensor_log.py val --n-logs 40
# pixels for one chosen (log, camera) pair once scene selection has picked it
.venv\Scripts\python.exe scripts\pick_rgb_demo_scene.py   # reproduces results/rgb_demo_scenes.json / rgb_demo_takes.json
.venv\Scripts\python.exe scripts\fetch_sensor_log.py val --log-id <log-id> --cam <ring_camera_name> --images
```

Lidar is never fetched. A clean re-run of the demo build needs ~0.52 GB per
D-N1-15's own note (the 1.38 GB actually used included exploratory fetches).

### 2c. QCNet preprocessing cache (`data/qcnet_root`)

Regenerated by QCNet's own preprocessing over `data/av2/val`, run inside
`.venv-qcnet` (this is what G-N1-0 consumes):

```powershell
.venv-qcnet\Scripts\python.exe third_party\QCNet\val.py --data-root data\qcnet_root --root data\av2\val ...
```

(Exact QCNet CLI flags are documented in `third_party/QCNet/README.md` once
re-cloned; D-N1-8 notes the upstream path-join bug requires an **absolute**
`--root`, not relative.) Then re-run the G-N1-0 sanity check:

```powershell
.venv-qcnet\Scripts\python.exe scripts\eval_checkpoint_g_n1_0.py --ckpt checkpoints\qcnet_av2.ckpt --data-root data\qcnet_root
```

Expect minADE₆/minFDE₆/MR₆ within 0.01 of 0.72/1.25/0.16 (frozen tolerance,
D-N1-8; last measured 0.7201/1.2527/0.1574).

### 2d. SIMPL-preprocessed local features (`data/simpl_features/local`)

Depends on 2a (`data/av2/{train,val}`) and the deterministic subset orderings:

```powershell
.venv\Scripts\python.exe scripts\select_local_subsets.py
.venv\Scripts\python.exe scripts\preprocess_local.py --split train --count 20000 --out-root data\simpl_features\local
.venv\Scripts\python.exe scripts\preprocess_local.py --split val   --count 1500  --out-root data\simpl_features\local
```

(`--count` values are the frozen D-N1-14a/b budget: 20,000-scenario
city-stratified train prefix, 1,500-scenario val prefix — see
`results/local_subsets_manifest.json` and `results/local_budget.json` for
the frozen numbers; do not re-derive them, they were fixed by measurement.)
Preprocessing and training must **not** run concurrently on this laptop (16
GB RAM; D-N1-14a measured host-side OOM when overlapped).

## 3. Checkpoints (`checkpoints/`)

**Manifest coverage verdict — not all files are covered, and not all files
are re-downloadable:**

| File | Size | Manifest | Re-downloadable? |
|---|---|---|---|
| `checkpoints/qcnet_av2.ckpt` | 89 MB | `results/qcnet_ckpt_manifest.json` (Google Drive link + sha256 `b9f852ec888d6fb966a38e3e307af231145b8e1e69cf7b0b9dddd1ae33120f21`, 93,083,579 bytes) | **Yes** — external Google Drive link, last verified live 2026-07-26 (D-N1-6). Re-verify the link is still live before relying on it; it was not re-checked today. |
| `checkpoints/local/c1_seed42.tar` | 31 MB | **none** | **No.** Locally trained (D-N1-14, SIMPL C1-local, seed 42, epoch 5 snapshot at the point this checkpoint was saved). |
| `checkpoints/local/c1_seed42_e5.tar` | 31 MB | **none** | **No.** Canonicalized epoch-5 snapshot of the C1-local run (D-N1-14b truncation). Same run as above. |
| `checkpoints/local/c3_seed42.tar` | 31 MB | **none** | **No.** Locally trained C3-local (occlusion-aug arm) — **collapsed** during this run (D-N1-14c); kept only as the collapse-verification artifact, not a usable predictor. |
| `checkpoints/local/c3_seed42_e5.tar` | 31 MB | **none** | **No.** Epoch-5 snapshot of the same collapsed C3-local run. |

The four `checkpoints/local/*.tar` files have **no external source** — they
are this project's own training output. Regenerate them only by re-running
the training script against the frozen local budget (needs `data/av2/train`
+ `data/simpl_features/local/train` from step 2 above; expect the same
collapse in C3-local unless the curriculum mitigation from D-N1-14d(4) is
implemented first — it was designed but never coded):

```powershell
.venv\Scripts\python.exe scripts\train_local.py --arm c1 --train-count 20000 --epochs 20 --stop-after-epoch 5 --seed 42
.venv\Scripts\python.exe scripts\train_local.py --arm c3 --train-count 20000 --epochs 20 --stop-after-epoch 5 --seed 42
```

`third_party/SIMPL/saved_models/*.tar` are separate — those are upstream's
own author-shipped checkpoints, restored by the `git clone` in step 1, not
part of `checkpoints/`.

## 4. After rebuilding: tests, gates, and the full-scale run

Run in this order:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest -q                       # full suite, incl. tests/test_license_guard.py
.venv\Scripts\python.exe scripts\check_gates.py              # G_quality, G-N1-0..5 per gates.yaml
```

`tests/test_license_guard.py` must pass before any commit — it is what makes
public-repo safety load-bearing on this project (verified 2026-09-26: 6/6
green, `ruff check .` clean).

Then, the next step is cluster time (sized at ~260 H200 GPU-hours from
measured local throughput, `results/local_budget.json`). With it, run the full
matrix (3 arms × ≥3 seeds × S0–S4 × R-A on the full split) and gate G-N1-2
(pre-registered paired Wilcoxon test, `scripts/paired_test.py`) becomes
checkable for the first time.

To re-render the RGB demo against a post-HPC checkpoint, it's one command
(checkpoint-agnostic by construction, D-N1-15(8)):

```powershell
.venv\Scripts\python.exe scripts\render_rgb_demo.py --takes results\rgb_demo_takes.json --ckpt <new-ckpt-path> --ckpt-label <label>
```
