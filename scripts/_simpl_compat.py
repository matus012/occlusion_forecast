"""Compat shims for vendored SIMPL under the pinned py3.11 env.

Import this module BEFORE any `simpl.*` import. No third_party file is
edited (D-N1-8 precedent: shim at the wrapper layer, never in the vendor
tree).

Shim 1: simpl/simpl.py does `from fractions import gcd`; fractions.gcd was
removed in py3.9 (it is math.gcd). Restoring the attribute pre-import keeps
the vendored file byte-identical to the pinned commit.

Shim 2: utils/evaluator.py imports get_displacement_errors_and_miss_rate from
the LEGACY av1 `argoverse` package (not installable in the pinned env, py3.11
+ modern deps). A faithful numpy reimplementation is registered under that
module path instead. It feeds TRAIN-TIME MONITORING ONLY -- every reported
number in results/ comes from the official av2 kit (scripts/eval_local.py),
never from this shim.
"""
from __future__ import annotations

import fractions
import math
import sys
import types

import numpy as np

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd  # type: ignore[attr-defined]


def _patch_lr_scheduler_verbose() -> None:
    """Shim 3: torch 2.13 removed the `verbose` positional from
    LRScheduler.__init__; SIMPL's PolylineLR still passes it. Re-accept and
    drop it (it only ever controlled printing)."""
    import inspect

    from torch.optim import lr_scheduler as lrs

    orig = lrs.LRScheduler.__init__
    if "verbose" in inspect.signature(orig).parameters:
        return

    def patched(self, optimizer, last_epoch=-1, verbose=False):  # noqa: ANN001
        orig(self, optimizer, last_epoch)

    lrs.LRScheduler.__init__ = patched


_patch_lr_scheduler_verbose()


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: dict,
    gt_trajectories: dict,
    max_guesses: int,
    horizon: int,
    miss_threshold: float = 2.0,
    forecasted_probabilities: dict | None = None,
) -> dict:
    """Reimplementation of argoverse.evaluation.eval_forecasting.
    get_displacement_errors_and_miss_rate (av1 API semantics): candidates are
    pruned to the top-`max_guesses` by forecast probability, best candidate =
    argmin FDE, minADE = ADE of that candidate, probabilities renormalized
    over the pruned set for the brier/p- variants."""
    min_ade, min_fde, n_misses = [], [], []
    prob_min_ade, prob_min_fde, prob_n_misses = [], [], []
    brier_min_ade, brier_min_fde = [], []

    for seq_id, gt in gt_trajectories.items():
        preds = np.asarray(forecasted_trajectories[seq_id], dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)[:horizon]
        n_cand = min(max_guesses, len(preds))
        if forecasted_probabilities is not None:
            probs = np.asarray(forecasted_probabilities[seq_id], dtype=np.float64)
            order = np.argsort(-probs, kind="stable")[:n_cand]
            pruned_probs = probs[order]
            pruned_probs = pruned_probs / pruned_probs.sum()
        else:
            order = np.arange(n_cand)
            pruned_probs = None
        pruned = preds[order][:, :horizon]

        fdes = np.linalg.norm(pruned[:, -1] - gt[-1], axis=-1)
        best = int(np.argmin(fdes))
        best_fde = float(fdes[best])
        best_ade = float(np.linalg.norm(pruned[best] - gt, axis=-1).mean())
        min_ade.append(best_ade)
        min_fde.append(best_fde)
        n_misses.append(best_fde > miss_threshold)

        if pruned_probs is not None:
            p = float(pruned_probs[best])
            neg_log = min(-np.log(p), -np.log(0.05))
            prob_n_misses.append(1.0 if best_fde > miss_threshold else (1.0 - p))
            prob_min_ade.append(neg_log + best_ade)
            prob_min_fde.append(neg_log + best_fde)
            brier_min_ade.append((1.0 - p) ** 2 + best_ade)
            brier_min_fde.append((1.0 - p) ** 2 + best_fde)

    out = {
        "minADE": float(np.mean(min_ade)),
        "minFDE": float(np.mean(min_fde)),
        "MR": float(np.mean(n_misses)),
    }
    if forecasted_probabilities is not None:
        out["p-minADE"] = float(np.mean(prob_min_ade))
        out["p-minFDE"] = float(np.mean(prob_min_fde))
        out["p-MR"] = float(np.mean(prob_n_misses))
        out["brier-minADE"] = float(np.mean(brier_min_ade))
        out["brier-minFDE"] = float(np.mean(brier_min_fde))
    return out


def _register_argoverse_shim() -> None:
    if "argoverse.evaluation.eval_forecasting" in sys.modules:
        return
    pkg = types.ModuleType("argoverse")
    evaluation = types.ModuleType("argoverse.evaluation")
    eval_forecasting = types.ModuleType("argoverse.evaluation.eval_forecasting")
    eval_forecasting.get_displacement_errors_and_miss_rate = (
        get_displacement_errors_and_miss_rate)
    pkg.evaluation = evaluation
    evaluation.eval_forecasting = eval_forecasting
    sys.modules["argoverse"] = pkg
    sys.modules["argoverse.evaluation"] = evaluation
    sys.modules["argoverse.evaluation.eval_forecasting"] = eval_forecasting


_register_argoverse_shim()
