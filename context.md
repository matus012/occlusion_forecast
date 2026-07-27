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
- D-N1-7a (2026-07-26, user amendment): option (3) added to the wk1 comparison — SIMPL as
  training baseline (pre-decided contingency #2). VERIFIED same day: MIT license; AV2
  checkpoint SHIPPED IN-REPO (saved_models/simpl_av2_bezier_ckpt-fix.tar, SHA256 in
  results/simpl_vendor_manifest.json — no link rot possible); checkpoint-matched AV2 val
  numbers author-posted in issue #15 (minADE6 0.777 / minFDE6 1.452 / MR6 0.196 /
  b-minFDE6 2.069; caveat: '-fix' ckpt is an author RETRAIN, original paper ckpt lost —
  paper Table IV row slightly better, anchor to the author-posted ckpt-matched numbers);
  per-timestep validity ingestion CONFIRMED (has_flags → PAD_OBS channel in the actor
  feature tensor, simpl/av2_dataset.py:252) — hard requirement SATISFIED, with the honest
  caveat that SIMPL's native convention nn-pads invalid positions at preprocess time
  (flag+nn-pad, vs FMAE's flag+zero) — occlusion masking must follow the native convention
  per-baseline, flag-flip is the cross-baseline invariant; training cost: plain PyTorch
  (NO PyG compiled deps), 4-GPU DDP, 50 epochs, small model (~30MB ckpt incl. optimizer)
  → same order as FMAE, O(10-30) H200h/run, 9-run matrix ≈ 100-270 H200h (fits/near-fits
  ceiling). SIMPL vendored untracked @ 2a33314.
- D-N1-8 (2026-07-26): G-N1-0 PASSED and FROZEN. QCNet released ckpt scored through the
  official av2 kit on ALL 24988 val scenarios (4060, 103 min): minADE6 0.7201 / minFDE6
  1.2527 / MR6 0.1574 vs published 0.72/1.25/0.16 — deltas 0.0001/0.0027/0.0026, all
  within the frozen 0.01 tolerance (2dp rounding band + margin). Our official-kit eval
  path is validated. G-N1-1 tolerance freeze DEFERRED to the D-N1-7 baseline decision
  (anchor model unknown until then); its provisional +0.05/+0.10 offsets stand. Eval env
  notes: PyG 2.8 needed a module-level TargetBuilder forward-shim (no third_party edits);
  QCNet preprocessing requires an ABSOLUTE dataset root (upstream path-join bug with
  relative roots).
- D-N1-9 (2026-07-26): P2 stats extracted (results/p2_occlusion_stats.json; 24 CARLA
  scenarios, 157/161 agents analyzed — 4 walker-6 render artifacts excluded by the
  data-driven criterion "never reaches vis_hi=0.5" (3 all-zero traces + 1 with a
  sub-threshold noise bump peaking 0.21), per P1 D23 — 205 segments @ fps=20). Duration fit: lognormal(shape 1.168, loc 0, scale 1.093) →
  median 1.09s, mean 2.16s; MOT17 anchor (~1.5s median) same order — sane. MIX RE-DERIVED,
  P2 WINS per D-N1-2 pre-commitment: empirical M1 0.396 / M2 0.059 / M3 0.545 (of
  M1M2M3-classified agents) vs v1 prior 0.6/0.25/0.15 — real occlusion is FLICKER-DOMINATED.
  The Phase 3 mask generator (not yet implemented) WILL use mix M1 0.40 / M2 0.05 /
  M3 0.55. SPEC AMENDMENT: the v1 M3 band
  (0.2-0.6s per block) fits only 9% of real flicker runs — M3 block durations now sample
  the fitted lognormal clipped to 0.2-2.0s (2-4 blocks unchanged); M1 clip 0.3-4.0s
  unchanged. Six agents outside the taxonomy (4 trailing-never-recovers, 2 with >4
  segments) reported in the JSON, not force-classified.
- D-N1-10 (2026-07-26, user directive): Phase 3 EVAL-PROTOCOL ADDENDUM. (a) P2-derived
  empirical mix stays the PRIMARY eval condition, but per-pattern degradation curves
  (M1-only / M2-only / M3-only at matched masked fractions) are emitted alongside the
  mixed curve — pattern difficulty differs and aggregate-only reporting could mask it.
  (b) The deterministic eval mask manifest stores per-scenario pattern labels so
  per-pattern slicing needs no mask regeneration. (c) Pre-registered honest reframing:
  if mixed-curve degradation is weak but M1/M2-only curves are strong, the headline
  becomes "degradation depends on occlusion structure, not just masked fraction" —
  a finding, not a fallback.
- D-N1-11 (2026-07-26): mask-generator design facts from the adversarial review (opus),
  binding on all reporting:
  (a) R-B x M2 EXCLUDED — "appeared late AND still occluded at t=0" = never observed =
      no scenario; R-B mix renormalizes to M1 0.421 / M3 0.579 (computed, not hardcoded).
  (b) M1 block duration is DETERMINED by the severity target (fraction guarantee wins);
      the P2 lognormal is inert for M1. P2 statistics enter ONLY via (i) the pattern mix
      and (ii) M3's relative block-size ratios. Docs/README must never claim per-block
      durations follow the P2 fit.
  (c) R4 sensitivity sweep REDEFINED: global ±50% duration scaling cancels against the
      fraction guarantee (measured: M1 100% invariant, M3 mean block length invariant to
      4 s.f. — though ~80% of individual M3 masks still change, and the fit SHAPE
      parameter remains an executable axis; scaling cannot move aggregate duration
      statistics). The executable sensitivity axes are: mix perturbation (empirical
      D-N1-9 mix vs v1 prior 0.6/0.25/0.15 as the alternative condition) and M3
      block-count range. R4 sweep = re-eval one arm under the v1-prior mix, report-only.
  (d) Reporting requirements: R-A severity label is not proportional to forecast-relevant
      information loss (recent-window masking probability is ~2.5-4x lower than nominal —
      recency discount; curve captions must say so); R-A and R-B severity labels are NOT
      difficulty-comparable (S1 R-B can remove more recent information than S4 R-A);
      degenerate-entropy arms exist by construction (M2/R-A and M1/R-B collapse to 1
      distinct mask per severity — the M1/R-B arm measures a single realization shared
      by ~42% of the R-B bucket, 10561/24988) — state this wherever those slices are
      plotted. [34%→42% corrected 2026-07-27 per reviewer's own re-measurement.]
  (e) Review outcome: first implementation REJECTED (severity-coupled pattern redraw
      destroyed per-pattern pairing; license-guard ceiling; missing property tests;
      edge-gap placement bias up to 17% from sorted sampling-with-replacement — fixed
      by a proper stars-and-bars sampler). Fixes + SPEC_VERSION bump ("N1-mask-v2") +
      manifest regeneration verified in re-review (APPROVE): 100% pattern-cohort
      stability across severities on full val, seeds collision-free, mix z<1, placement
      distributionally exact (KS p=1.000). Committed manifests are summary-only
      (av2_manifests precedent), per-scenario labels local with sha256 pinned.
- D-N1-12 (2026-07-27): G-N1-2 statistical test PRE-REGISTERED before any masked training
  exists (protocol requirement met with margin — no arm has trained a single step).
  Test: one-sided Wilcoxon signed-rank on per-scenario paired minFDE6 deltas (C1 - C3,
  H1: C3 < C1), per-scenario values = mean over >=3 training seeds per arm, regime R-A,
  mixed-pattern condition, identical deterministic N1-mask-v2 masks for all arms.
  Severities tested: S2, S3, S4; alpha 0.05, Holm-Bonferroni across the three.
  Gate key paired_p_s2 = Holm-adjusted p at S2; direction criterion = C3 seed-mean beats
  C1 at all of S2/S3/S4. Effect size: seeded bootstrap (10k resamples) 95% CI of mean
  delta, reported alongside. Per-pattern cohort tests (D-N1-10) use the same procedure
  but are report-only, never gate-blocking (degenerate-entropy caveats per D-N1-11d).
  Implementation lands as scripts/paired_test.py (P1 pattern) in Phase 4; the spec text
  in gates.yaml is immutable from this commit.
- D-N1-13 (2026-07-27, owner decision): D-N1-7 RESOLVED — training baseline = SIMPL
  (MIT, checkpoint shipped in-repo; verification facts in D-N1-7a). FMAE demoted:
  revisit ONLY if upstream grants a license AND republishes checkpoints (issue #25
  stays open as a tracker, not a blocker). G-N1-1 re-anchors to the SIMPL author-posted
  ckpt-matched AV2 val numbers (minADE6 0.777 / minFDE6 1.452 / MR6 0.196, D-N1-7a);
  tolerance freeze when the first SIMPL retrain lands. D-N1-1's FMAE choice superseded;
  masking engine unaffected (baseline-agnostic by design, D-N1-7). Occlusion application
  for SIMPL follows the native convention: validity flag→0 + nn-pad pos/ang + zero vel
  (flag-flip is the cross-baseline invariant).
- D-N1-14 (2026-07-27, user directive): HPC COMMITTEE PACKAGE — reduced-scale local
  proof on the 4060 feeding the PERUN request (reports/perun_request/):
  (a) SIMPL arms C1-local (clean) + C3-local (occlusion-aug per spec, p_occ=0.5,
      empirical D-N1-9 mix, fresh draw per epoch), 1 seed each, stratified train subset
      sized by measurement to an overnight-or-less budget; ALL outputs labeled *-local
      and never mixed with the real gate arms (G-N1-1..3 untouched).
  (b) Mini degradation curve: C1/C2/C3-local × S0–S4 × R-A on a fixed val subset,
      deterministic N1-mask-v2 masks, official av2 kit as metric authority; figure
      watermarked "reduced-scale local proof — full matrix pending HPC".
  (c) Side-by-side inference video (C1-local vs C3-local, identical masks, labeled
      "illustrative scenarios"); mp4/gif local-only, script committed.
  (d) PERUN request report: scope + methodology/gates + evidence (G-N1-0 freeze,
      mask-engine review record, mini curve, video stills) + compute justification
      from MEASURED local throughput extrapolated to 3 arms × 3 seeds × full data on
      H200s, with measured 8GB-VRAM infeasibility evidence. Factual tone; user edits
      and submits. [Measurement freezes appended below as sub-entries.]
- D-N1-14a (2026-07-27, frozen by measurement — results/local_budget.json):
  local budget = train prefix 20,000 (city-stratified) + dev slice 1,000 (disjoint
  monitoring only, no model selection — FINAL-epoch ckpt both arms) + val prefix
  2,500 (pattern-cohort-stratified), 20 epochs, batch 8, seed 42, TF32.
  Measured basis: SIMPL bs16 (authors' per-GPU setting) allocates 8.66GB > 8GB
  physical → shared-memory thrash (3 samp/s); bs12 spills too (8.5GB); bs8 = 5.0GB
  peak, ~21 samp/s steady → ~5.5h/arm. Preprocessing 25-27 scen/s (10 workers),
  137KB/scenario. LR polyline milestones scaled 50→20 epochs ([0,2,14,16]).
  Op note: training and preprocessing NEVER run concurrently (16GB RAM; measured
  CUDA host-side OOM + 17 preprocess MemoryErrors when overlapped, both recovered).