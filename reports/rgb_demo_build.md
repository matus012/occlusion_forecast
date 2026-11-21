# RGB demo build — occlusion_forecast (N1), D-N1-15

**Date:** 2026-08-11 · **Follows:** [rgb_demo_feasibility.md](rgb_demo_feasibility.md) (Path A, GO)
**Checkpoint:** `checkpoints/local/c1_seed42.tar` (C1-local, epoch 5)

> Every rendered frame carries: **"under-trained local checkpoint — pipeline
> demonstration; full-scale results pending HPC"**. This build proves the
> *pipeline*, not the model. Same honesty-label discipline as D-N1-14c.

---

## 1. What was built

```
AV2 sensor log (annotations 10 Hz, ego pose ~170 Hz, calibration, HD map)
  └─ src/otraj/sensor/scenario.py    synthesize ArgoverseScenario per 110-step window
       └─ ArgoPreprocAV2.process()   third_party/SIMPL, UNMODIFIED (D-N1-5)
            └─ Simpl.forward()       --ckpt, any arm
                 └─ focal → scene → city frame
                      └─ src/otraj/sensor/projection.py   city → ego@cam → image
                           └─ ring-camera JPEG + fan/GT/history overlay
```

Re-predicts from a fresh window at **every annotation step**, so the fan tracks
the scene. Camera runs 20 Hz against 10 Hz annotations; each prediction is held
across the ~2 camera frames it covers.

| File | Role |
|---|---|
| `src/otraj/sensor/scenario.py` | scenario synthesis, windowing, focal survival |
| `src/otraj/sensor/projection.py` | city→image projection, ground raster, cuboid proof |
| `scripts/fetch_sensor_log.py` | selective anonymous S3 fetch (metadata / one camera) |
| `scripts/pick_rgb_demo_scene.py` | metadata-only scene ranking |
| `scripts/render_rgb_demo.py` | rolling-prediction clip renderer |
| `scripts/rgb_overlay_spike.py` | single frame + `--boxes` transform proof |
| `tests/test_sensor_scenario.py` | 19 tests on the synthesis contract |
| `results/rgb_demo_takes.json` | the four takes, with selection rationale |
| `results/rgb_demo_scenes.json` | all 480 scored candidates (provenance) |

---

## 2. Clips

Outputs in `reports/videos/rgb/` — **AV2-derived pixels, local and untracked.**
`ALLOWED_TRACKED_VISUALS` stays empty.

| Take | Log (val) | Camera | Focal | Windows | Frames | Length | Modes in frustum | History visible | minFDE6 mean | first→last | mp4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | `0b86f508` | ring_rear_left | `c14c6c6a` | 49/49 | 96 | 4.8 s | 6.0/6 | 96/96 | 1.472 m | 0.52 → 1.30 | 12.8 MB |
| **B** | `2c652f9e` | ring_front_left | `f447e33b` | 44/48 | 86 | 4.3 s | 5.87/6 | 64/86 | 13.149 m | 3.71 → 26.42 | 9.2 MB |
| **C** | `19f53e16` | ring_rear_left | `08e86edf` | 48/48 | 94 | 4.7 s | 6.0/6 | 94/94 | 1.510 m | 2.89 → 1.95 | 10.7 MB |
| **D** | `25e5c600` | ring_rear_right | `63777d1b` | 48/48 | 94 | 4.7 s | 6.0/6 | 94/94 | 3.376 m | 3.43 → 2.84 | 7.5 MB |

All clips 1024×830, h264, 20 fps. Each has a `stills/*_mid.png` and a sidecar
`.json` carrying checkpoint path, epoch, per-frame visibility counts and minFDE6.
Total render cost 772 s predict + 275 s render for 370 frames.

**minFDE6 here is not a result** — it is a per-clip sanity number on an
under-trained checkpoint running slightly off-distribution (§6.3, §6.4).

**Take B is the honest outlier at 13.1 m**, degrading 3.71 → 26.42 m across the
clip. That is the *model*, not the projection: its history and GT are drawn on
the correct road geometry and the fan diverges smoothly away from them (visible
in `stills/2c652f9e_..._mid.png`). Its focal is a slow vehicle (42.5 m over 11 s)
decelerating at an intersection, and the 6-epoch checkpoint keeps predicting
forward motion. Kept in the set rather than dropped — a demo reel of only the
takes that flatter the model would misrepresent it.

### Flagship

**Take C — `19f53e16` / ring_rear_left / focal `08e86edf`.**

Chosen because it is the only take that is simultaneously *busy* and *clean*:

- **Densest traffic in the entire 40-log pool by 2×** — 43 fully-observed moving
  vehicles, against 16–21 for the others.
- **Every overlay element visible in every frame**: 6.0/6 modes, 94/94 history,
  94/94 GT. Nothing in the legend is ever absent.
- **Occlusion-relevant geometry**: parked row along the kerb plus an intersection
  on the focal path — the scene type the project is actually about.
- **Representative accuracy**, minFDE6 1.510 m, and it *improves* across the clip
  (2.89 → 1.95 m) as history accumulates.

Take A is the runner-up — marginally better minFDE6 (1.472 m) and the longest
clip (49 windows) — but its scene carries 16 moving vehicles to C's 43, so it
demonstrates the same pipeline against a thinner scene.

---

## 3. Scene selection

Scouted **40 val logs on metadata alone** (~2.3 MB each, ~92 MB total) —
calibration lives in the metadata, so "is this agent big and visible in camera X"
is answerable before a single JPEG is fetched. **480 candidates** scored on:

```
score = vis_span                                   frustum coverage × pixel span
      × (1 + 0.04·min(n_moving, 15))               traffic density
      × (1 + 0.03·min(n_parked, 15))               parked rows near the focal path
      × (1.25 if intersection)                     occlusion-relevant geometry
      × (0.4 + 0.6·frac_visible_hist)              history actually reaches the image
```

Components are reported separately in `results/rgb_demo_scenes.json` rather than
collapsed into one opaque number — the shortlist is for a human to pick from.

**One selection bug was caught before it shipped.** History visibility must be
scored *separately* from future visibility: an agent driving away from the ego has
a fully visible future and a history entirely behind the camera. Scoring only the
future selected four scenes whose history track reached the image in **0 of 14
frames** — the clips would have shipped with a legend entry that never appears.
The originally top-ranked take (`070bbf42`, score 4188) fell to **#15** once
history was scored. The failure was invisible in the metric that selected it.

The quiet spike log (`02678d04`, 3 candidates, 4 px spans) is exactly the kind of
scene this rejects.

---

## 4. Fetch sizes

Bulk download stays rejected: the sensor split is **~1.06 TB** (1000 logs ×
~1.06 GB; 700/150/150), val alone ~159 GB. Selective anonymous per-object fetch
over plain HTTPS instead — no AWS credentials, no `aws`/`s5cmd`.

| Item | Size |
|---|---|
| Metadata, 40 logs (annotations + pose + calibration + map) | 94 MB |
| Pixels, 11 (log, camera) pairs — 3,511 JPEGs | 1,283 MB |
| ↳ of which the four shipped takes use | 422 MB |
| ↳ spike-era, 4 pairs (feasibility proof) | 463 MB |
| ↳ superseded by the re-selection, 3 pairs | 398 MB |
| **Total `data/av2_sensor/`** | **1.38 GB** (budget 2 GB, not exceeded) |
| Lidar | **0 B — never fetched** |

Fetch rate 1.66–2.40 MB/s with 16 threads; ~45–90 s per (log, camera).
`scripts/fetch_sensor_log.py` is idempotent per object, so an interrupted fetch
resumes rather than restarting.

**398 MB was fetched and then not used**: the first scene selection picked three
cameras before history visibility was scored (§3), and the corrected ranking moved
elsewhere. Stated rather than netted out — a re-run from a clean checkout, using
the corrected scout, needs **94 MB metadata + 422 MB pixels ≈ 0.52 GB**.

---

## 5. Re-render against a post-HPC checkpoint

Checkpoint-agnostic by construction — `--ckpt` / `--ckpt-label` are plain
arguments and nothing downstream knows which arm produced them. **One command
re-renders every take:**

```bash
.venv/Scripts/python.exe scripts/render_rgb_demo.py --takes results/rgb_demo_takes.json --ckpt <new-ckpt> --ckpt-label C1-hpc
```

Output filenames carry the label, so arms never overwrite each other. Each clip
writes a sidecar JSON with the checkpoint path, epoch, per-frame visibility
counts and minFDE6 — enough to compare arms without re-watching anything.

Occlusion severities are already wired (`--severity S0..S4 --regime R-A`, the
deterministic N1-mask-v2 masks); masked history steps are dropped from the drawn
history rather than shown as observed. Not exercised in this build — the owner's
minimal cut was fan + GT + history only.

---

## 6. Known limitations

1. **~5 s clips, by data not choice.** A ~15 s log has ~157 annotated frames and
   each prediction needs 110 → ~48 rolling windows. Owner decision: accept it, no
   concatenation, no frozen-prediction pan.
2. **A focal can drop out mid-log.** Take B's focal is fully observed for only
   44 of 48 windows, so its clip is truncated to 44 and says so. This crashed a
   render before it was handled; a track ending partway through a log is ordinary
   data, not an error.
3. **The five named distribution shifts** (feasibility §2, D-N1-15(3)) all apply:
   finite-differenced velocity, category by observation-completeness rather than
   MF's predictability curation, synthesized `AV` track, hand-authored 30→7 class
   map, curated focal. **The model is running slightly off-distribution and the
   numbers here should not be compared to val metrics.**
4. **Under-trained checkpoint.** C1-local is 6 epochs (D-N1-14b truncation).
   minFDE6 on these clips is not a result.
5. **Single focal per clip.** Other agents are in the scenario and inform the
   prediction, but only the focal's trajectory is drawn.
6. **No depth test on the overlay.** Trajectories are drawn on the ground plane
   with painter's-algorithm ordering, so a line passes *over* a foreground
   vehicle instead of behind it — clearly visible in take B, where the fan
   crosses a semi-trailer. Fixing this needs a per-pixel depth buffer from the
   lidar sweep or the cuboids; deliberately not built (polish, out of scope).
   Worth naming because on an occlusion project a viewer may read the artifact
   as a claim about visibility. It is not.
7. **Rear-facing cameras on 3 of 4 takes.** The corrected ranking favoured them;
   front-camera takes with fully visible history were scarcer in a 40-log pool.
8. **Throughput ~0.3 windows/s** (~4.8 s of preprocessing + inference per
   window), so a full clip is ~4–5 min. Dominated by SIMPL's lane-graph
   construction and RPE, not by the synthesis or the rendering.

---

## 7. Gate status

| Check | Result |
|---|---|
| `tests/test_license_guard.py` | **PASS** — `ALLOWED_TRACKED_VISUALS` empty |
| Guard is *load-bearing*, not vacuous | **VERIFIED** — see below |
| Full suite | **183 passed** |
| ruff | clean |
| coverage | **92.6%** (min 80%) |
| Dataset-derived bytes tracked | **0** |

Tracked from this build: loaders/scripts, the scene manifests, tests, and the two
text-only reports (whitelisted by exact filename, `blockers.md` precedent).
`data/av2_sensor/`, `reports/videos/rgb/` and `reports/rgb_spike/` are all
gitignored.

**A guard that passes because nothing reached it is not evidence.** So it was
tested against the failure it exists to prevent: `git add -f` on a rendered
AV2-derived still, then run the guard.

```
E  assert not ['reports/videos/rgb/stills/0b86f508_..._C1local_S0_mid.png']
```

`test_tracked_visuals_are_allowlisted` fails as designed; after unstaging, 6
passed. The guard is doing work, not sitting idle behind a `.gitignore`.
