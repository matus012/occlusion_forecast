# occlusion_forecast

Occlusion-aware trajectory prediction on Argoverse 2: how much does motion
forecasting degrade when the observation history is partially hidden — and how
much of that degradation can occlusion-aware training buy back?

**Status: scaffold (Phase 0).** Gates, guards, and environment are in place;
baseline evaluation is next. See `gates.yaml` for the quality-gate ladder
(G-N1-0 … G-N1-4) and `context.md` for the full spec and decision log.

## Protocol honesty (read first)

- The occlusion-masked evaluation protocol is **custom** — the AV2 leaderboard
  does not measure it. Metric *code* is still the official `av2` eval kit
  (minADE_6 / minFDE_6 / MR_6), run on masked-input predictions.
- **OCCLUSION masking** (this project: eval/train-time hidden history) is kept
  terminologically disjoint from Forecast-MAE's **PRETEXT masking** (MAE
  pretraining objective). They are different mechanisms.
- No Argoverse 2 data, derived tensors, or dataset-derived pixels are tracked in
  this repository — loaders, manifests, and SHA256 records only, enforced by
  `tests/test_license_guard.py` on every CI run and gate check.
- Any claimed metric delta carries ≥3 training seeds; single-seed deltas are
  reported as noise-level, never as findings.

## Layout

- `src/otraj/` — package (masking engine, eval runners, training arms)
- `gates.yaml` + `scripts/check_gates.py` — executable quality gates
- `tests/` — unit tests incl. the license guard
- `results/` — metric JSONs consumed by the gate checker (no raw dumps)
- `context.md` — authoritative spec + append-only decision log (D-N1-x)

## License

Apache-2.0 (matches the Forecast-MAE baseline this work builds on).
