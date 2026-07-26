# context.md — N1: occlusion-aware trajectory prediction (AV2)
Created 2026-07-26 (ideation, pre-code). Owner: Matúš Filo. Dir: ws/101_occl_traj/.
Repo: github.com/matus012/occlusion_forecast — PUBLIC from day 1 [user-approved
2026-07-26; user creates the repo, Claude Code links remote + pushes]. CONSEQUENCE:
license guard + zero-dataset-content invariant must be green BEFORE the first push;
no D41-style retroactive purge budget exists on a public history.
Strategy: scope.md §4 N1. Sibling: P1/P2 context.md (occlusion-mot) — reuse its discipline
(gates.yaml, frozen val protocol, license guard, paired tests, multi-seed rule D36).

## Mission
Trajectory forecasting on Argoverse 2 motion forecasting split (~250k scenarios) under
PARTIALLY HIDDEN observation history. Differentiation = forecasting degradation + robustness
under occlusion-masked history (masks sourced from P2 CARLA occlusion statistics), NOT
leaderboard rank. Headline result: degradation curve (metric vs occlusion severity) for
clean-trained baseline vs occlusion-aware-trained model, plus one honest clean-data
baseline comparison via the official av2 eval kit.
Feeds thesis: hidden-state estimation on public data. Ship ~Sept 12.

## Baseline decision — D-N1-1 (2026-07-26)
**Chosen: Forecast-MAE** (Cheng et al., ICCV 2023; repo jchengai/forecast-mae).
Runner-up: QCNet (Zhou et al., CVPR 2023; repo ZikangZhou/QCNet).
- Forecast-MAE: AV2-native, released checkpoints (validate eval kit BEFORE training),
  PyTorch Lightning (SLURM/PERUN multi-GPU trivial), training measured in hours-not-days,
  compact codebase, ingests per-timestep validity masks natively → our occlusion masking
  plugs into the existing padding path. Repro-in-≤1wk realistic solo.
- QCNet: stronger absolute numbers, but multi-day multi-GPU training, heavier codebase,
  higher repro risk inside the wk1 box; retraining cost multiplies across our
  masked-training arms. Rejected as primary; documented fallback if FMAE repro fails wk1.
- Contingency #2: SIMPL (lightweight, AV2, open code) if both stall.
- CONFOUND (flag in all writing): Forecast-MAE's PRETEXT masking (MAE pretraining) ≠ our
  OCCLUSION masking (eval/train-time hidden history). Terminology kept disjoint everywhere.
  Bonus experiment: does MAE pretraining buy occlusion robustness for free? (ablation:
  pretrained vs from-scratch under masked eval — cheap, novel-ish, honest.)
- Reference numbers (AV2 val, b=6, FROM MEMORY — verify wk1 against paper/repo before
  freezing G-N1-1): FMAE minADE ≈0.7, minFDE ≈1.4, MR ≈0.17; QCNet ≈0.62/1.19/0.14.
  Gate tolerance set AFTER verification, not from these.

## Data & protocol
- AV2 Motion Forecasting: 11s @ 10Hz = 50 obs steps (5s) + 60 pred steps (6s); focal agent
  scored; official av2-api eval kit = ONLY metric authority (minADE_k, minFDE_k, MR_k, k=6).
- Splits: train/val official. Public leaderboard/test NOT a goal. Val used for reported
  numbers; model selection on a carved-out dev subset of train (seeded, manifest-pinned)
  — D18-style discipline: val touched per pre-registered manifest only.
- Masked-eval protocol is CUSTOM (leaderboard doesn't measure it) — stated honestly in
  README; metric CODE still the official kit, run on masked-input predictions.

## Occlusion-masking spec — D-N1-2 (v1, refine wk1 against real P2 stats)
Source: P2 CARLA occlusion segments (223 segments / 24 scenarios, exact per-agent
visibility; see P1 D23). Extract: duration distribution (fit log-normal in SECONDS),
segments-per-agent rate, multi-segment spacing. MOT17 cross-check: median gap ≈37 frames
(~1.5s) — sanity anchor for the duration fit.
Masking = flip per-timestep validity to 0 on the 50-step history. NO input imputation
(model's job); position values zeroed where invalid (match FMAE padding convention).
Patterns:
- M1 block: single contiguous occlusion, duration ~ P2 fit (clipped 0.3–4.0s), onset
  uniform s.t. block fits in history.
- M2 prefix (late appearance): agent invisible from t=-50 until onset — models occluded
  entry/sensor range.
- M3 flicker: 2–4 short blocks (0.2–0.6s) — partial/marginal visibility tail of P2 dist.
Pattern mix per masked scenario: M1 0.6 / M2 0.25 / M3 0.15 (v1 prior; re-derive from P2
segment stats wk1 — if P2 supports different mix, P2 wins, log the change).
Severity buckets (masked fraction of the focal agent's 50 obs steps):
S0=0 · S1=0.2 · S2=0.4 · S3=0.6 · S4=0.8 (target ±0.05, achieved by scaling block lengths).
Visibility regimes at prediction time t=0:
- R-A "re-emerged": last 5 obs steps forced VISIBLE (occlusion in the past). HEADLINE curve.
- R-B "still-occluded": mask extends through t=0 (predict from stale observation). Stretch
  chapter; hardest, closest to the ghost-car demo.
Who gets masked: v1 = focal agent only. Variant (report-only): focal + nearest-k neighbors
masked with independent draws (scene-level occlusion).
Train application: each train scenario masked with prob p_occ=0.5; severity ~ uniform
{S1..S4}; pattern ~ mix; fresh draw per epoch (data-aug semantics). Clean-trained baseline
= p_occ=0.
Eval application: DETERMINISTIC — mask seeded by (scenario_id, severity, pattern set fixed
per bucket); same masks for every model compared. Mask manifests (per-bucket seed + spec
hash) committed; masks themselves regenerable, never stored per-scenario in repo.
Comparisons (all on identical masked inputs):
  C1 clean-trained baseline (robustness gap — the motivation number)
  C2 clean-trained + constant-velocity/linear imputation of masked steps (cheap-fix null)
  C3 occlusion-aware trained (ours)
Claim shape: C3 degrades slower than C1/C2 across S1–S4 with C3 ≈ C1 at S0 (no clean-data
tax beyond a stated tolerance). Multi-seed: ≥3 training seeds per arm for any claimed
delta (P1 D36 lesson: single-seed deltas are noise).

## Gates (freeze numerics wk1; structure fixed now)
- G-N1-0 eval-kit sanity: released FMAE checkpoint scored through av2 kit reproduces
  published val numbers within tolerance set at verification. Blocks everything.
- G-N1-1 baseline repro: our retrained FMAE within tolerance of published val metrics
  (tolerance frozen after G-N1-0, e.g. +0.05 minADE — decided then, logged).
- G-N1-2 headline: degradation curve, 3 arms × S0–S4 × ≥3 seeds, R-A regime; C3 vs C1
  improvement at S2+ significant (paired per-scenario minFDE, Wilcoxon or bootstrap CI —
  pre-register test wk2 before any masked training completes).
- G-N1-3 no-clean-tax: C3 at S0 within tolerance of C1 at S0.
- G-N1-4 repo/demo: public repo, runnable demo, README GIF (ghost car under bridge →
  prediction fan → confirmed exit), license guard ported from P1 (AV2 license: no raw
  data/derived tensors in repo — loaders + manifests + hashes only).
Timeline tripwire: baseline repro dragging past wk2 → cut scenario/pattern variants
(drop M3 + R-B + neighbor-masking), keep M1/M2 × S0–S4 × R-A core result.

## Compute
Local 4060 8GB: loaders, mask unit tests, overfit-100-scenarios smoke, tiny-config train.
PERUN H200: all real training (arms × seeds), request filed with N1 estimate.
Budget sketch (verify wk1): FMAE full train ~O(10) H200h/run → 3 arms × 3 seeds ≈ O(100)
H200h ceiling; if measured throughput blows this, cut seeds to 2 + report honestly
(P1 D43 lesson: no silent budget drops).
N5 add-on: TensorRT export of trained predictor, 30 FPS on 4060, runs DURING PERUN waits.
Warehouse re-skin: +1wk max, after core ships.

## Risks
- R1 FMAE repro friction (env/version rot) → pinned env day1, checkpoint-eval before any
  training; fallback QCNet→SIMPL pre-decided.
- R2 masking too easy (metrics barely degrade) → severity S4 + R-B regime guarantee
  measurable degradation; if C1 barely degrades even at S4/R-B that is itself a
  publishable robustness finding — report, don't hide.
- R3 masking too hard/underdetermined (multimodal metrics saturate) → MR and per-scenario
  paired deltas carry the signal; report full curves.
- R4 P2 stats domain gap (CARLA urban vs AV2) → masks parameterized by P2 but sensitivity
  check with ±50% duration scaling (one report-only sweep).
- R5 PERUN queue latency → local tiny-config results first; PERUN runs batched.
- R6 pretext-mask confound (see D-N1-1) → disjoint terminology + pretrained-vs-scratch
  ablation.

## Conventions (inherited from P1 unless overridden)
uv venv py3.11, torch cu126 locally / cu-whatever PERUN provides; seeds + device injection
everywhere; results/*.json consumed by check_gates.py; gates.yaml; context.md decision log
D-N1-x append-only; status.txt per session; repo PUBLIC by standing approval (above); AGPL not
required here — license TBD wk1 (FMAE is Apache-2.0 → ours MIT/Apache unless a dep forces
otherwise; log as D-N1-3). [Resolved same day: Apache-2.0, see D-N1-3 below.]

## Decision log (append-only)
- D-N1-1 (2026-07-26): baseline = Forecast-MAE; fallbacks QCNet → SIMPL. (See section above.)
- D-N1-2 (2026-07-26): occlusion-masking spec v1 — M1/M2/M3, S0–S4, R-A/R-B, C1/C2/C3. (Above.)
- D-N1-3 (2026-07-26): license = Apache-2.0 — matches Forecast-MAE upstream, permissive with
  patent grant, no AGPL-forcing dep in the pinned env (torch/av2/lightning all permissive).
- D-N1-4 (2026-07-26): Phase 0 scaffold conventions — package name `otraj`; P1 gate schema
  ported with a `live: quality` flag replacing the hardcoded G3 special case; gates G-N1-0..4
  all UNFROZEN (numerics freeze by measurement only); .gitignore is default-deny for
  data/parquet/visuals with the D41 per-file visual allowlist in tests/test_license_guard.py;
  env pinned day 1 (py3.11, torch 2.13.0+cu126, av2 0.x, lightning — requirements.txt);
  CI = torch-free guard subset (ruff + license guard + gates-config tests) on ubuntu.
- D-N1-5 (2026-07-26): CORRECTION to D-N1-3 premise — forecast-mae has NO declared license
  (no LICENSE file; GitHub license metadata null; checked at pin commit cb86ea9). Our
  Apache-2.0 stands for OUR code, but FMAE is not redistributable: third_party/ stays
  untracked (guard-enforced), pinned via results/fmae_vendor_manifest.json, integration is
  wrap-and-import only — no FMAE file is ever copied into src/ or committed. Published-val
  anchor numbers for G-N1-0 read from the pinned README (fine-tune: minADE6 0.7117,
  minFDE6 1.408, MR6 0.178; scratch: 0.7214/1.430/0.187) → results/fmae_published_val.json.
- D-N1-6 (2026-07-26): ALL THREE FMAE checkpoint links are dead (HKUST OneDrive 404 —
  verified curl + real browser; no mirrors on HF/forks/web). Adaptation, NOT full fallback:
  G-N1-0 eval-kit sanity re-anchored to the QCNet released checkpoint (live Google Drive
  link, repo Apache-2.0, published AV2 val: minADE6 0.72 / minFDE6 1.25 / MR6 0.16) —
  the gate's purpose is validating OUR av2-kit usage against SOMEONE'S released model,
  not FMAE specifically. FMAE remains the training baseline (D-N1-1 stands); G-N1-1
  anchors to FMAE published SCRATCH numbers. QCNet eval runs in its own venv
  (.venv-qcnet) to keep the pinned FMAE env clean. Residual risk: FMAE published numbers
  now unverified by checkpoint — mitigation: G-N1-1 tolerance frozen only after our own
  retrain lands within family range. OPEN (needs user, outward-facing): file a GitHub
  issue on jchengai/forecast-mae asking for checkpoint re-upload.
- D-N1-7 (2026-07-26, user directive): BASELINE LICENSE TRIPWIRE — D-N1-1 (FMAE as
  training baseline) is PROVISIONAL pending license clarification. Deadline end of wk1:
  (a) FMAE issue yields a license grant → keep FMAE, log it; (b) otherwise cost out and
  report in status.txt with a recommendation (user decides):
    (1) switch training baseline to QCNet — Apache-2.0, checkpoint-anchored G-N1-1, but
        honest H200h estimate for retraining 3 arms × 3 seeds against the PERUN budget;
    (2) own-code reimplementation of the FMAE finetune-only variant from the paper — no
        license dependency; viable ONLY if it fits inside the wk2 tripwire box.
  Until resolved: no FMAE-derived code beyond wrap-and-import experiments; masking engine
  (Phase 3) stays baseline-agnostic by design (operates on validity masks, not model
  internals) so it survives either outcome.