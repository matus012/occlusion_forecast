# PERUN (TUKE HPC) Allocation Request — Project N1: Occlusion-Aware Trajectory Forecasting on Argoverse 2

**Applicant:** Matúš Filo, 3rd-year BSc "Intelligent Systems", TUKE FEI
**Repository:** github.com/matus012/occlusion_forecast (public, Apache-2.0)
**Request date:** 2026-07-27
**Requested resources (summary):** 260 H200-hours base + itemized contingency to a 350 H200-hour band, ~2 CPU-node-hours (preprocessing), 100 GB scratch storage (details in §5)

> Every table and every derived hour/GB figure in §3–§5 is generated from
> committed measurement artifacts (`results/*.json`) by
> `scripts/build_perun_report.py`, with sources cited at each table; §3.2's
> review-record quantities cite the decision log (`context.md`, D-N1-11).

---

## 1. Project scope

Trajectory forecasting for autonomous driving degrades when the observation
history of an agent is partially hidden — occlusion by other vehicles,
infrastructure, or sensor range. Public leaderboards evaluate only the
fully-observed case. This project quantifies, on the Argoverse 2 Motion
Forecasting benchmark (~250k real scenarios), how much a modern forecaster
degrades under realistically structured occlusion of the 5-second observation
history, and whether occlusion-aware training buys that robustness back.

The headline deliverable is a **degradation curve**: forecast error (official
AV2 metrics: minADE₆ / minFDE₆ / MR₆) versus occlusion severity (0–80% of the
history masked), for three arms trained on identical data:

* **C1** — clean-trained baseline (the robustness gap, i.e. the motivation),
* **C2** — clean-trained + constant-velocity imputation (the "cheap fix" null),
* **C3** — occlusion-aware trained (ours: occlusion applied as data
  augmentation during training).

Occlusion masks are not synthetic noise: their pattern mix, duration
distribution and structure are fitted to per-agent visibility traces extracted
from a prior CARLA multi-object-tracking project (205 occlusion segments,
`results/p2_occlusion_stats.json`), and
applied to AV2 as deterministic, manifest-pinned per-timestep validity masks.
Baseline model: **SIMPL** (MIT license, ~2.6M parameters, checkpoint shipped
in-repo by its authors) — deliberately compact so the scientific claim rests
on a controlled comparison rather than leaderboard rank. The work feeds a
bachelor-thesis chapter on hidden-state estimation; target completion
mid-September 2026.

## 2. Methodology and quality gates

The experimental protocol was frozen **before** any training run:

* **Deterministic eval masks** — every (scenario, severity) pair maps to a
  unique mask via SHA-256 seeding (spec "N1-mask-v2"); all arms are evaluated
  on bit-identical inputs. Committed manifests pin the spec; masks are
  regenerable, never stored.
* **Pattern taxonomy** — M1 contiguous block / M2 late-appearance prefix /
  M3 flicker, mixed per the empirical CARLA-derived ratio (M1 0.40 / M2 0.05 /
  M3 0.55); severities S0–S4 = 0/20/40/60/80% of the 50-step history;
  headline regime R-A ("re-emerged": the last 0.5 s is always visible); a
  stretch regime R-B ("still-occluded" through t=0) doubles the eval
  conditions in the full matrix.
* **Pre-registered statistics (gate G-N1-2)** — one-sided Wilcoxon signed-rank
  on per-scenario paired minFDE₆ deltas (C1 − C3), ≥3 training seeds per arm,
  Holm-Bonferroni over severities S2/S3/S4, α = 0.05; registered in the
  repository (`gates.yaml`, decision D-N1-12) before any masked training
  existed.
* **No-clean-tax gate (G-N1-3)** — C3 must match C1 on unmasked data within a
  pre-frozen tolerance.
* **Collapse monitor (gate G-N1-5, added after the local finding in §3.3)** —
  a history-sensitivity probe (replace all agent-history features with noise;
  measure prediction displacement; `scripts/history_sensitivity_probe.py`)
  runs at every training snapshot of every arm; an arm that fails it is
  reported as collapsed and excluded from gates regardless of its metrics.
  Local reference: healthy arm 52.5 m median displacement, collapsed arm
  8e-6 m (`results/local/probe_c1_seed42.json`, `probe_c3_seed42.json`).
* **C3 design response to the collapse (D-N1-14d)** — the full-matrix C3 arm
  uses a curriculum p_occ ramp 0 → 0.5 over the first 20% of epochs as the
  PRIMARY mitigation (final augmentation distribution unchanged); a fixed
  p_occ = 0.25 arm is the pre-declared CONTINGENCY, costed as an itemized
  line in §5 and spent only if the curriculum arm also collapses.
* **Metric authority** — only the official `av2` evaluation kit computes
  reported metrics.
* **License hygiene** — AV2 data and anything derived from it never enter the
  public repository (guard-enforced: loaders, manifests and hashes only).

## 3. Evidence of readiness (all local, RTX 4060 Laptop 8 GB)

The pipeline is complete end-to-end at reduced scale; HPC is requested for
scale, not development.

**3.1 Eval-kit validation (gate G-N1-0, frozen).** A released QCNet
checkpoint scored through the official av2 kit on all 24,988 validation
scenarios reproduced the published values within 0.003 on every metric
(minADE₆ 0.7201 vs 0.72, minFDE₆ 1.2527 vs 1.25, MR₆
0.1574 vs 0.16; `results/g_n1_0_checkpoint_eval.json`). Our usage of the
metric authority is validated.

**3.2 Mask engine, adversarially reviewed.** The mask generator was reviewed
twice (first implementation rejected; fixes verified in re-review): 100%
pattern-cohort stability across severities on the full validation split,
collision-free seeding, placement distributional exactness (KS p = 1.000),
property tests in CI (decision record D-N1-11, `context.md`).

**3.3 Reduced-scale degradation curve (the headline figure, locally).**
Both training arms were run at reduced scale on the 4060 — labeled
**\*-local** throughout and never mixed with the full-scale gate arms:
20,000 city-stratified scenarios × 6 epochs (6 of a 20-epoch schedule — time-boxed truncation D-N1-14b, identical for both arms, LR not yet annealed), batch 8, 1 seed, wall-clock 5.6 h (C1-local) / 7.7 h (C3-local, p_occ=0.5, empirical mix) — `results/local/train_c*_seed42.json`. Evaluated on 1500 fixed validation scenarios under
deterministic masks, S0–S4, regime R-A (`results/local/eval_C1-local_native.json`, `eval_C2-local_impute.json`, `eval_C3-local_native.json`). C2-local is not a
third training run: it is the C1-local checkpoint evaluated with masked
steps imputed and flagged valid (the "cheap fix" null), which is why its
S0 row equals C1-local's by construction:

| Severity | C1-local minADE₆/minFDE₆/MR₆ | C2-local (imputation) | C3-local (occl-aug) |
|---|---|---|---|
| S0 | 2.472 / 5.481 / 0.604 | 2.472 / 5.481 / 0.604 | 4.164 / 7.833 / 0.854 |
| S1 | 2.701 / 5.780 / 0.666 | 2.471 / 5.482 / 0.605 | 4.164 / 7.833 / 0.854 |
| S2 | 2.915 / 6.014 / 0.700 | 2.472 / 5.481 / 0.599 | 4.164 / 7.833 / 0.854 |
| S3 | 3.173 / 6.395 / 0.745 | 2.472 / 5.481 / 0.601 | 4.164 / 7.833 / 0.854 |
| S4 | 3.707 / 6.970 / 0.765 | 2.470 / 5.488 / 0.599 | 4.164 / 7.833 / 0.854 |

Reading (single seed, descriptive only — the pre-registered G-N1-2 test requires ≥3 seeds/arm). Three results: **(i)** the clean-trained arm degrades monotonically under occlusion, minFDE₆ 5.481 m at S0 → 6.970 m at S4 (+27%) — the robustness gap this project quantifies exists already at 6-epoch reduced scale. **(ii)** Constant-velocity imputation (C2) restores the input statistics almost exactly for this under-trained model (max deviation from clean 0.007 m across severities). **(iii)** The occlusion-aug arm COLLAPSED to a history-invariant predictor (flat 7.833 m at every severity; verified: randomizing ALL actor-history features moves its output 1e-05 m vs 65.7 m for C1-local — `results/local/c3_collapse_verification.json`). C3-local is therefore worse than C1-local everywhere (paired ΔminFDE₆ -2.352 m at S0, narrowing to -0.863 m at S4 as C1 degrades toward C3's flat line). We report the collapse rather than hiding it: it shows the occlusion-aware question CANNOT be answered at laptop scale — 6 of 20 epochs, 10% data, LR never annealed — and adds a concrete requirement (collapse monitoring, ≥3 seeds) to the full-scale runs this request funds.

**3.4 Inference visualizations.** `scripts/inference_video.py` renders
BEV-styled side-by-side videos (C1-local vs C3-local, identical masks;
illustrative scenarios, hand-picked for visual legibility and labeled as
such). The stills attached to this PDF show C1-local on the SAME scenario
with and without occlusion — the degradation effect isolated. The C3-local
panel is deliberately omitted from the attached stills: with the collapsed
arm (§3.3) a fans frame visually misreads as "two similar methods"; the
full videos, where the C3 panel is captioned as the collapse illustration,
are available on request (AV2 license keeps them out of the public repo).
Local paths: `reports/videos/09669770_S3_c1_vs_c3.mp4`, `reports/videos/a6146f53_S3_c1_vs_c3.mp4`.

## 4. Why the full matrix cannot run on the available 8 GB GPU (measured)

SIMPL's authors train with per-GPU batch 16 on RTX 3090 (24 GB). Measured on
the RTX 4060 Laptop (8 GB, TF32 on), `results/local_budget.json`:

| Batch | samples/s (probe, incl. warmup) | steady | peak VRAM | verdict |
|---|---|---|---|---|
| 16 (authors' per-GPU) | 3.09 | — | 8661 MB | VRAM spill past 8GB physical -> shared-memory thrash |
| 12 | 13.12 | — | 8500 MB | VRAM spill on large-scene batches |
| 8 | 12.33 | 21.0 | 5041 MB | chosen operating point |

Batch 16 and even batch 12 exceed physical VRAM → shared-memory thrash
(3–13 samples/s). The workable operating point is batch 8 at
21 samples/s steady state. At that rate one full-scale training
run (199,908 scenarios × 50 epochs ≈ 10.0M samples) costs
**5.5 days**, and the pre-registered 9-run matrix
(3 arms × 3 seeds) **50 days** of continuous single-GPU
compute — before evaluation, re-runs, or the R4 sensitivity sweep. The laptop
is also not a stable platform for multi-hour runs: during the two local
training runs we measured recurring host-side throttle episodes that cut
throughput to 2–3 samples/s for ~2 h stretches (epoch wall-clock varied
12–130 min for C1-local and 11–158 min for C3-local, identical work per
epoch; `results/local/train_c*_seed42.json`), which
is one of the two reasons the local arms had to be time-box truncated to 6
epochs in the first place. That is not a feasible thesis timeline; the local
runs above are the honest maximum this hardware supports.

## 5. Requested allocation (H200)

Basis: measured per-sample training cost on the 4060
(21 samples/s), scaled to the full matrix, converted to H200
throughput with a stated assumption band of 6.0–10.0×
per-GPU speedup over the laptop 4060 (small model, partially dataloader-bound;
the band is deliberately conservative and will be calibrated by the pilot run
below — we commit to reporting the measured factor back).

| Cost item | Basis (measured) | 4060 hours | H200 hours (÷6.0–10.0) |
|---|---|---|---|
| 1 training run (199,908 × 50 ep) | 21.0 samples/s (results/local_budget.json) | 132.2 h | 13–22 h |
| 9-run matrix (3 arms × 3 seeds) | — | 1190 h | 119–198 h |
| Full masked-eval matrix (9 ckpts × 5 severities × 2 regimes × 24,988) | 20.6 ms/scenario (derived: Σ wall_seconds / Σ scenario-evals over the three local eval JSONs) | 13 h | 1–2 h |
| R4 sensitivity sweep (1 arm, v1-prior mix) | = 1 run | 132.2 h | 13–22 h |
| Preprocessing (CPU-side, 224,896 scenarios) | 25.268 scen/s at scale (results/local/preproc_rate.json) | 2.5 h | CPU nodes |
| Pilot calibration | 1 epoch | — | 5 h |
| General contingency (15%) → base total | — | — | **260 h** (upper band) |
| Collapse-mitigation contingency arm (p_occ=0.25, 3 seeds — used ONLY if the primary curriculum-C3 collapses; §2) | = 3 runs | 397 h | 40–66 h |
| Repeat of one failed/collapsed run | = 1 run | 132.2 h | 13–22 h |
| **Total requested band (base + itemized contingency)** | — | — | **350 h** |

| Item | Value |
| --- | --- |
| Total requested | **260 H200-hours base; 350 H200-hours including the itemized collapse-mitigation contingency** (each line justified in the table above — no unitemized padding) |
| Scratch storage | **100 GB** (raw AV2 ≈60 GB + SIMPL features ≈31 GB (137 KB/scenario measured) + checkpoints/logs). The dataset is re-synced cluster-side from the public AV2 bucket (no upload from the applicant); all scratch content is reproducible and deletable at any time. |
| Walltime per training run | 22.0 h (single H200; conversion band applied to the measured batch-8 throughput — the pilot fixes the real batch/rate) |
| Job shape | single-GPU jobs, 9 training runs + eval jobs; no multi-node requirement |
| Pilot/calibration | 5.0 H200-h included: 1-epoch calibration run to fix the conversion factor before the matrix launches |

Deliverables back to PERUN: measured H200 throughput report, the published
degradation-curve study, and public repository artifacts (code, manifests,
gate records — no AV2 data).

---

*Prepared by the applicant with local measurements; every table above is
regenerated by `scripts/build_perun_report.py` from the cited
`results/*.json` files.*
