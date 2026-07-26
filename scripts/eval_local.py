"""Masked-eval of a *-local SIMPL checkpoint (D-N1-14b mini degradation curve).

Evaluates one checkpoint on the FIXED val subset (prefix of the deterministic
pattern-cohort-stratified ordering, results/local/val_order.json) across
severities S0-S4, regime R-A, deterministic N1-mask-v2 masks -- identical
masks for every arm by construction (pure function of scenario_id).

Arms map onto (checkpoint, mode):
  C1-local: clean ckpt,   mode=native  (occlusion per native SIMPL convention)
  C2-local: clean ckpt,   mode=impute  (constant-velocity/linear cheap-fix null)
  C3-local: aug ckpt,     mode=native

METRIC AUTHORITY: av2.datasets.motion_forecasting.eval.metrics (official kit),
same convention as the frozen G-N1-0 runner: best trajectory = argmin FDE over
K=6, minADE6 = ADE of that trajectory, MR6 = best-FDE > 2 m. ADE/FDE are
rigid-transform invariant, so they are computed in the focal-actor frame that
SIMPL predicts in (no world-frame transform needed for metrics).

Per-scenario records (sid, pattern cohort, minADE/minFDE/miss) are kept in the
output JSON so paired per-scenario deltas and D-N1-10 per-pattern slices need
no re-run.

Output: results/local/eval_{label}_{mode}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIMPL_ROOT = REPO / "third_party" / "SIMPL"
sys.path.insert(0, str(SIMPL_ROOT))
sys.path.insert(0, str(REPO / "src"))

import _simpl_compat  # noqa: E402, F401  (py3.11 shims; must precede simpl imports)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from av2.datasets.motion_forecasting.eval.metrics import (  # noqa: E402
    compute_ade,
    compute_fde,
)
from config.simpl_av2_cfg import AdvCfg  # noqa: E402
from simpl.av2_dataset import AV2Dataset  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.utils import set_seed  # noqa: E402

from otraj.masking.generator import SPEC_VERSION, generate_mask  # noqa: E402
from otraj.masking.simpl_apply import (  # noqa: E402
    apply_cv_imputation,
    apply_native_mask,
)

MISS_THRESHOLD_M = 2.0
SEVERITIES = ("S0", "S1", "S2", "S3", "S4")


class MaskedEvalDataset(AV2Dataset):
    """Val-subset dataset applying the deterministic eval mask for one
    (severity, regime) condition, in native or impute mode."""

    def __init__(self, files: list[str], severity: str, regime: str, mode: str):
        self._files_override = list(files)
        self.severity = severity
        self.regime = regime
        self.apply_mode = mode
        super().__init__(str(Path(files[0]).parent), mode="val", obs_len=50,
                         pred_len=60, aug=False, verbose=False)
        self.dataset_files = self._files_override
        self.dataset_len = len(self._files_override)

    def data_augmentation(self, df):
        data = super().data_augmentation(df)
        if self.severity != "S0":
            res = generate_mask(data["SEQ_ID"], self.severity, self.regime)
            if self.apply_mode == "native":
                apply_native_mask(data["TRAJS"], res.mask)
            elif self.apply_mode == "impute":
                apply_cv_imputation(data["TRAJS"], res.mask)
            else:
                raise ValueError(f"unknown mode {self.apply_mode!r}")
        return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", required=True, help="e.g. C1-local / C2-local / C3-local")
    ap.add_argument("--mode", choices=["native", "impute"], required=True)
    ap.add_argument("--val-count", type=int, required=True)
    ap.add_argument("--regime", default="R-A")
    ap.add_argument("--severities", default="S0,S1,S2,S3,S4")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "simpl_features" / "local" / "val")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_seed(0)  # eval is deterministic anyway; belt-and-braces
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    order = json.loads(
        (REPO / "results" / "local" / "val_order.json").read_text(encoding="utf-8")
    )["order"]
    subset = order[: args.val_count]
    files = [str(args.features_dir / f"{sid}.pkl") for sid, _ in subset]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} val features missing (first: {missing[0]})")
    pattern_of = dict(subset)

    cfg = AdvCfg()
    net = Simpl(cfg.get_net_cfg(), device).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    result: dict = {
        "label": args.label, "mode": args.mode, "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt.get("epoch"), "regime": args.regime,
        "mask_spec": SPEC_VERSION, "val_count": len(files),
        "metric_authority": "av2.datasets.motion_forecasting.eval.metrics (official kit)",
        "convention": "best = argmin FDE over K=6; MR threshold 2.0 m (G-N1-0 parity)",
        "watermark": "reduced-scale local proof -- full matrix pending HPC",
        "severities": {},
    }

    t0 = time.time()
    for sev in args.severities.split(","):
        assert sev in SEVERITIES, sev
        ds = MaskedEvalDataset(files, sev, args.regime, args.mode)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=ds.collate_fn,
                        drop_last=False, pin_memory=True)
        per_scen: list[dict] = []
        with torch.no_grad():
            for data in tqdm(dl, ncols=80, desc=f"{args.label}/{sev}", mininterval=5.0):
                out = net(net.pre_process(data))
                post = net.post_process(out)
                traj_pred = post["traj_pred"].cpu().numpy()  # [B, K, 60, 2]
                for b, sid in enumerate(data["SEQ_ID"]):
                    gt = data["TRAJS"][b]["TRAJS_POS_FUT"][0].cpu().numpy()  # [60, 2]
                    fde = compute_fde(traj_pred[b], gt)
                    best = int(np.argmin(fde))
                    ade = compute_ade(traj_pred[b], gt)
                    per_scen.append({
                        "sid": sid,
                        "pattern": "none" if sev == "S0" else pattern_of[sid],
                        "minade6": float(ade[best]),
                        "minfde6": float(fde[best]),
                        "miss": bool(fde[best] > MISS_THRESHOLD_M),
                    })
        agg = {
            "n": len(per_scen),
            "minade6": float(np.mean([r["minade6"] for r in per_scen])),
            "minfde6": float(np.mean([r["minfde6"] for r in per_scen])),
            "mr6": float(np.mean([r["miss"] for r in per_scen])),
        }
        for pat in ("M1", "M2", "M3"):
            rows = [r for r in per_scen if r["pattern"] == pat]
            if rows:
                agg[f"minfde6_{pat}"] = float(np.mean([r["minfde6"] for r in rows]))
                agg[f"n_{pat}"] = len(rows)
        result["severities"][sev] = {"aggregate": agg, "per_scenario": per_scen}
        print(f"[{args.label} {sev}] minADE6={agg['minade6']:.4f} "
              f"minFDE6={agg['minfde6']:.4f} MR6={agg['mr6']:.4f} n={agg['n']}",
              flush=True)

    result["wall_seconds"] = round(time.time() - t0, 1)
    out = args.out or (REPO / "results" / "local" /
                       f"eval_{args.label}_{args.mode}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
