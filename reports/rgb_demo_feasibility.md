# RGB demo feasibility spike — occlusion_forecast (N1)

**Date:** 2026-08-11 · **Time-box:** 4 h hard · **Status:** complete, inside box
**Question:** can we show N1's predictions overlaid on real camera footage instead of BEV renders?

**Verdict: Path A = GO (proven end-to-end, not just analysed). Path B = NO-GO, not needed.**

---

## 0. Premise correction (read this first)

The spike brief says *"model = Forecast-MAE-based (C1-local ckpt)"*. That is stale.
**D-N1-13 (2026-07-27) superseded D-N1-1: the trained baseline is SIMPL**, not
Forecast-MAE — `third_party/SIMPL`, `simpl.simpl.Simpl`, checkpoints
`checkpoints/local/c1_seed42.tar` / `c3_seed42.tar`. Forecast-MAE was dropped over the
missing-license / dead-checkpoint issue (D-N1-5, D-N1-7).

This is not cosmetic. It sets the entire conversion contract, and it makes the job
**easier** than the brief assumed: SIMPL's preprocessor takes the Motion Forecasting
dataclass directly —

```python
ArgoPreprocAV2.process(seq_id, scenario: ArgoverseScenario, static_map: ArgoverseStaticMap)
```

— so the only thing Path A actually has to build is an `ArgoverseScenario`. Everything
downstream (features, dataset, net, masking, metrics) runs unmodified, preserving the
D-N1-5 no-vendored-edits discipline.

---

## 1. AV2 Sensor availability and size — bulk download REJECTED, selective fetch VERIFIED

**Locally present before the spike:** motion-forecasting split only —
`data/av2/train` (199,908 scenarios) + `data/av2/val` (24,988). Zero sensor data.

**Measured from the bucket** (anonymous HTTPS listing, no AWS credentials, no `aws`/`s5cmd`
— none installed and none needed):

| Split | Logs |
|---|---|
| train | 700 |
| val | 150 |
| test | 150 |

One val log fully enumerated (`02678d04-…`, 3,036 objects): **1.06 GB**.

| Component | Size / log | Needed? |
|---|---|---|
| 9 cameras (7 ring + 2 stereo) | 865 MB | one ring camera only |
| ↳ one ring camera, 319 JPEGs @ 20 Hz | 89–129 MB | **yes** |
| lidar (157 sweeps) | 145 MB | **no** — never fetched |
| `annotations.feather` | 279 KB | yes |
| `city_SE3_egovehicle.feather` | 161 KB | yes |
| `calibration/` (intrinsics + extrinsics) | 10 KB | yes |
| `map/` (HD map + ground-height raster) | 1.7 MB | yes |

> **Full sensor dataset ≈ 1.06 TB; val split alone ≈ 159 GB. Both blow through the 30 GB
> flag by 5×–35×, so bulk download is rejected outright.** It is also unnecessary: the
> bucket is a plain public S3 prefix, per-object gettable, so we fetch a **~2.3 MB
> metadata slice per log** and pull pixels only for a (log, camera) pair already chosen.

**Measured fetch rates** (`scripts/fetch_sensor_log.py`, 16 threads):

- metadata slice: 1.61–3.19 MB, ~6–16 s per log serial, ~0 s warm (idempotent, resumable per object)
- one full ring camera: 95.7–128.9 MB in 45–62 s → **1.96–2.40 MB/s**

**Spike actually consumed 0.49 GB** — 10 logs of metadata + 4 camera-logs. A full demo
build needs **≤ 2 GB** (5 logs × up to 3 cameras). No approval threshold is anywhere near.

---

## 2. What sensor logs carry — all four requirements confirmed

Verified by loading real files, not from documentation:

| Requirement | Status |
|---|---|
| Ring camera RGB | ✅ 7 ring @ 20 Hz, `ring_front_center` 1550×2048, fx = 1781.5 px |
| Ego pose | ✅ `city_SE3_egovehicle`, **2,671 poses / 15.7 s ≈ 170 Hz** — far denser than needed |
| Intrinsics / extrinsics | ✅ `PinholeCamera.from_feather(log_dir, cam)` reads both; `project_ego_to_img` ready-made |
| Annotations → MF track format | ⚠️ **convertible, but not a mapping — a synthesis.** See below. |
| HD map | ✅ `log_map_archive_*.json`, **identical format to motion forecasting**; `ArgoverseStaticMap.from_map_dir` also gives a ground-height raster the MF split lacks |

### There is no scenario ↔ log mapping, and we do not need one

The `av2` API (0.3.6) exposes `datasets.motion_forecasting`, `datasets.sensor`,
`datasets.lidar` — **no cross-reference module, no mapping utility**. The two datasets are
mined from disjoint corpora with separate ID spaces. Waiting for a mapping is not a plan.

Instead the spike **synthesizes** an `ArgoverseScenario` from sensor annotations. The
target dataclass is small and fully specified, which is why this works:

```
ArgoverseScenario(scenario_id, timestamps_ns, tracks, focal_track_id, city_name, map_id, slice_id)
  Track(track_id, object_states, object_type, category)
    ObjectState(observed, timestep, position, heading, velocity)
```

Sensor annotations are **10 Hz — exactly the MF rate** (they ride the lidar clock), and a
log carries ~157 annotated frames vs the 110 (5 s history + 6 s future) a scenario needs.

**Five gaps had to be filled. Each is a design decision and a named distribution-shift risk:**

1. **`velocity` does not exist in sensor annotations.** SIMPL consumes `ObjectState.velocity`
   directly (`av2_preprocess.py:191`). Finite-differenced here (`np.gradient`, dt = 0.1 s).
   Noisier than MF's dataset-provided field.
2. **`category` (FOCAL / SCORED / UNSCORED / FRAGMENT) does not exist.** Assigned by
   observation completeness. MF's SCORED tracks are *curated for predictability*; ours are
   merely fully-observed. This is the largest single shift.
3. **The `AV` track does not exist** as an annotation (SIMPL asserts on it). Synthesized
   from ego pose.
4. **30 sensor classes → 10 MF `ObjectType` → 7 SIMPL one-hot slots.** Hand-authored map in
   `CATEGORY_TO_OBJECT_TYPE`; no published correspondence exists.
5. **Frames differ.** Sensor cuboids are ego-frame; MF tracks are city-frame. Converted via
   `city_SE3_egovehicle` per timestamp.

Plus the one that shapes the deliverable: **MF scenarios are mined for interesting focal
agents; a sensor window is not.** Focal selection is ours to make (§3).

---

## 3. Projection prototype — proven, and the transform chain is proven separately

`scripts/rgb_overlay_spike.py` runs the whole chain on real data:

```
sensor annotations (ego, 10 Hz)
  → ArgoverseScenario        [synthesized]
  → ArgoPreprocAV2.process   [third_party/SIMPL, unmodified]
  → Simpl.forward            [C1-local, epoch 5]
  → traj_pred [6, 60, 2]     focal-actor frame
  → scene frame → city frame → ego frame @ camera timestamp
  → PinholeCamera.project_ego_to_img
  → drawn on the ring-camera JPEG
```

Brief asked for one frame, one agent. Delivered, on GPU (`device=cuda:0`), with all
6 modes in frustum:

- `reports/rgb_spike/cand_b.png` — `ring_front_left`, focal travels 21.3 m, 95 % of its
  future inside the frustum, 1536 px span. Prediction fan and GT track down the lane.
- `reports/rgb_spike/cand_c.png` — `ring_rear_left`, focal travels 99.6 m, 100 % visible.

**Camera/annotation time alignment is a non-issue**: cameras run 20 Hz against 10 Hz
annotations, and the nearest ring frame landed **2.3–22.3 ms** away. Ego pose at ~170 Hz
means pose interpolation error is under 4 cm at 50 km/h. Both are below the width of a
rendered line.

**Trajectories are projected onto the map's ground-height raster**
(`raster_ground_height_layer.get_ground_height_at_xy`), not a flat-earth assumption — the
sensor split ships this layer, the MF split does not.

### Transform proof (the part that makes this a GO rather than a "looks right")

A trajectory drawn from the same annotations that produced it will look plausible even if
the projection is wrong. So the chain is checked against something independent of both the
model and the trajectory code: `--boxes` projects the **annotated 3D cuboids** through the
identical `ego → city → ego@cam → image` path.

`reports/rgb_spike/proof_boxes.png` / `proof_zoom.png`: **15 fully-visible cuboids, all
landing on the actual vehicles** — the box truck's cab and cargo body framed correctly,
each queued car boxed, a pedestrian boxed. Calibration, SE3 round-trip and frustum culling
are all confirmed correct. Only fully-visible boxes are drawn, so partial-frustum clipping
cannot flatter the result.

### Scene selection is a metadata-only operation — the cost lever

`scripts/pick_rgb_demo_scene.py` scores every (log, camera, window, agent) candidate on
`fraction of future inside frustum × pixel span`, using **only the ~2.3 MB metadata slice**
(calibration is in there, so visibility is answerable before any pixel is fetched).

**605 candidates across 10 logs from ~23 MB of downloads.** Best scored 1732 (141.8 m
travel, 97 % visible, 1792 px span). This matters: the naive pick — fastest agent in the
log — is routinely behind the ego and yields a **4 px smudge** at the vanishing point,
which is exactly what the first run produced.

---

## 4. License guard — unchanged policy, no new exposure

Camera pixels are AV2-derived, so every rendered frame falls under the same rule as the
D-N1-14d PDF stills: **local, untracked, never allowlisted.**

Concretely, no change to `tests/test_license_guard.py` is required or wanted:

- `data/` and `reports/*` are already gitignored; `data/av2_sensor/` and
  `reports/rgb_spike/` land inside them by construction.
- `ALLOWED_TRACKED_VISUALS` stays empty. AV2-derived pixels have **no valid class** —
  admitting one would need a documented license check plus its own decision-log entry.
- The guard's size ceiling and blob-extension deny-list catch a stray JPEG or feather file
  regardless.

**Verified: full suite green after the spike — 146 passed**, with 0.49 GB of sensor data
and four rendered overlays sitting on disk. `git status` shows only the three new scripts.

The three scripts are committable classes (loaders / download docs). **This report is
text-only and carries no AV2-derived pixels**, but `reports/*` is gitignored with only
`blockers.md` and `perun_request/*.md` whitelisted — so it stays local unless you want an
exact-filename exception added, following the `blockers.md` precedent. Flagging rather than
deciding, since it touches the guard's surface.

---

## 5. Path B (dashcam + YOLO + homography) — desk analysis, NO-GO

Path B was stretch-only, gated on Path A being blocked. **Path A is not blocked**, so this
is recorded and dropped. Two findings are worth keeping anyway.

### (a) Can the C1 ckpt run with null/degraded map input? — measured, not assumed

| Lane segments kept | Result |
|---|---|
| 0 | **FAILS** — `ValueError: need at least one array to concatenate` (preprocessing, `get_trajectories`) |
| 1 | **FAILS** — `RuntimeError: Tensors must have same number of dimensions: got 2 and 1` (net, graph gather / RPE) |
| 5 (of 92) | **RUNS** — mean 6 s endpoint shift vs the full map: **0.82 m** |

So map-free inference is impossible as built — the lane graph is a structural input
(`in_lane=16`, `d_lane=128`, plus agent↔lane relative positional encodings), not an
optional branch. A **stub** lane graph of ≥ ~5 segments does run.

The 0.82 m number is interesting and should not be over-read: it says the C1-local
checkpoint is only weakly map-conditioned — consistent with the already-documented
under-trained regime (6 epochs, C3 collapse, D-N1-14c). **One scenario, one reduced-scale
checkpoint. Indicative, not a gate.** A properly trained model would very likely be far
more map-sensitive, so Path B's premise gets *worse* as the project succeeds.

### (b) Expected distribution shift

Stacked on top of §2's five conversion gaps, Path B would add: mono-homography pseudo-BEV
depth error growing quadratically with range, YOLO+ByteTrack ID switches injecting
discontinuities the model reads as physical motion, no ego pose (so city frame must be
integrated from visual odometry and drifts), and a fabricated lane graph. **Four
uncontrolled error sources between the pixels and a metric.** Nothing survives that a
committee should be shown as evidence of model quality.

Path A costs less, is already proven, and keeps AV2 as the single data authority.

---

## 6. GO / NO-GO

| Path | Verdict | Basis |
|---|---|---|
| **A — AV2 sensor overlay** | **GO** | Full chain executed on real data; transform independently verified; ≤ 2 GB download; no vendored-code edits; license posture unchanged |
| **B — dashcam + YOLO + homography** | **NO-GO** | Not needed (A unblocked); map-free inference impossible as built; four extra uncontrolled error sources |

### Build plan — 9 h, ≤ 1 day

| # | Task | h |
|---|---|---|
| 1 | Promote synthesis to `src/otraj/sensor/scenario.py` + unit tests (frame round-trip, velocity FD, type map coverage over all 30 classes) | 2.0 |
| 2 | Multi-agent overlay: all scored agents, per-mode fans weighted by probability, agent boxes | 1.5 |
| 3 | Rolling-window video: re-predict per annotation step, render at 20 Hz camera rate | 2.0 |
| 4 | Occlusion conditions: apply the deterministic N1-mask-v2 masks (S0 vs S3, R-A) so the RGB demo shows *the project's actual finding*, matching the BEV video | 1.5 |
| 5 | Scene selection over a 40-log scout pool + fetch winning cameras | 0.5 |
| 6 | BEV | RGB side-by-side, HUD, illustrative/truncation labels per D-N1-14d | 1.0 |
| 7 | Guard + suite + report | 0.5 |

**Known constraint to design around (found during the spike, not obvious up front):** a
15 s log yields 157 annotated frames, and each prediction needs a 110-frame window →
**48 windows = 4.8 s of rolling-prediction video per log.** Either accept ~5 s clips,
concatenate several logs, or hold the prediction fixed and let only the camera advance for
the full 15 s. Decide at task 3.

**Reuse:** styling, mask application, HUD and labelling all lift from
`scripts/inference_video.py`. Metric authority and mask determinism are untouched.

### Single next command to start the build

```bash
.venv/Scripts/python.exe scripts/fetch_sensor_log.py val --n-logs 40
```

~92 MB, metadata only, ~4 min. Then `scripts/pick_rgb_demo_scene.py --root
data/av2_sensor/val --top 20` picks the demo scenes, and only their cameras get fetched.

---

## 7. D-N1-15 (DRAFT — not yet appended to context.md)

> **D-N1-15 (2026-08-11): RGB camera demo = AV2 Sensor overlay (Path A). GO.**
> No scenario↔log mapping exists between the AV2 motion-forecasting and sensor datasets,
> and none is needed: an `ArgoverseScenario` is **synthesized** from sensor annotations
> (10 Hz, ego→city via `city_SE3_egovehicle`), then fed to the unmodified
> `ArgoPreprocAV2` → SIMPL path, preserving D-N1-5. Predictions go focal-actor → scene →
> city → ego@camera-timestamp → `PinholeCamera.project_ego_to_img`, drawn on the map's
> ground-height raster. Chain verified independently of the model by projecting annotated
> 3D cuboids onto the vehicles (15/15 correct).
> **Bulk download rejected** — sensor split is ~1.06 TB (1000 logs × ~1.06 GB), val alone
> ~159 GB. Selective anonymous per-object fetch instead: ~2.3 MB metadata/log for scene
> selection, ~96–129 MB for one ring camera of a chosen log; lidar never fetched. Demo
> build budget ≤ 2 GB.
> **Five synthesis gaps are carried as named distribution shifts** — velocity
> finite-differenced (absent from sensor annotations), track category assigned by
> observation completeness rather than MF's predictability curation, `AV` track built from
> ego pose, 30 sensor classes hand-mapped to 7 SIMPL slots, focal agent chosen by camera
> visibility rather than MF's mining. Every RGB surface carries the same illustrative /
> reduced-scale labelling as D-N1-14d.
> **License posture unchanged (D-N1-14d precedent):** camera pixels are AV2-derived →
> outputs stay local and untracked, `ALLOWED_TRACKED_VISUALS` stays empty; only loaders and
> download docs are committed. Guard green (146 tests) with sensor data on disk.
> **Path B (dashcam + YOLOv8/11 + ByteTrack + mono homography) rejected.** Map-free
> inference is impossible as built — measured: 0 lane segments crashes preprocessing, 1
> crashes the net, 5 runs. Predictions moved only 0.82 m when 87 of 92 lanes were stripped,
> which reflects the under-trained C1-local checkpoint (single scenario, indicative only),
> not map-independence; a trained model gets *more* map-dependent, so Path B degrades as
> the project improves.

---

## Artifacts

| Path | What |
|---|---|
| `scripts/fetch_sensor_log.py` | selective anonymous AV2-sensor fetch (metadata / single camera, resumable) |
| `scripts/pick_rgb_demo_scene.py` | metadata-only scene ranking by frustum coverage × pixel span |
| `scripts/rgb_overlay_spike.py` | scenario synthesis → SIMPL → camera projection → overlay (`--boxes` = transform proof) |
| `reports/rgb_spike/cand_b.png`, `cand_c.png` | demo-grade overlays, 6/6 modes in frustum *(local, untracked)* |
| `reports/rgb_spike/proof_boxes.png`, `proof_zoom.png` | 15 annotated cuboids on vehicles — transform proof *(local, untracked)* |
| `data/av2_sensor/` | 0.49 GB, 10 logs metadata + 4 camera-logs *(local, untracked)* |
