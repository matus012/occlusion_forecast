"""G-N1-0 runner: score a released QCNet checkpoint through the OFFICIAL av2 eval kit.

Runs inside .venv-qcnet (torch 2.7.0+cu126 + PyG). Metric AUTHORITY is
av2.datasets.motion_forecasting.eval.metrics (official kit). QCNet's own
torchmetrics are NOT run here; a separate val.py run can serve as cross-check.

Frame note: QCNet predictions and targets are agent-centric. Rigid transforms
preserve L2 distances, so ADE/FDE/MR computed in the agent frame are identical
to world-frame values.

Official AV2 challenge definitions used:
  best mode  = argmin over K of FDE
  minFDE_k   = FDE of best mode
  minADE_k   = ADE of best mode
  MR_k       = fraction of scenarios with minFDE > 2.0 m

Usage (from repo root):
  .venv-qcnet/Scripts/python.exe scripts/eval_checkpoint_g_n1_0.py \
      --root data/qcnet_root --ckpt checkpoints/qcnet_av2.ckpt \
      --out results/g_n1_0_checkpoint_eval.json [--device cuda] [--batch-size 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "QCNet"))

from av2.datasets.motion_forecasting.eval.metrics import (  # noqa: E402
    compute_ade,
    compute_fde,
)
from transforms import TargetBuilder  # noqa: E402  (third_party/QCNet)


class TargetBuilderCompat(TargetBuilder):
    """PyG >= 2.4 makes BaseTransform.forward abstract; QCNet's TargetBuilder
    (PyG 2.3 era) implements __call__ directly and never dispatches to forward,
    so this delegation satisfies the ABC without recursion. Module-level so
    Windows spawn workers can pickle it. third_party/ stays unmodified (D-N1-5).
    """

    def forward(self, data):
        return TargetBuilder.__call__(self, data)

SEED = 2023  # match QCNet's own val seed for the cross-check run
MISS_THRESHOLD_M = 2.0
FOCAL_CATEGORY = 3  # AV2 'focal' track category id in QCNet's encoding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="QCNet dataset root (val/raw junction)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None, help="injected device; default auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="debug: stop after N batches")
    args = ap.parse_args()

    import pytorch_lightning as pl
    from datasets import ArgoverseV2Dataset
    from predictors import QCNet
    from torch_geometric.loader import DataLoader

    pl.seed_everything(SEED, workers=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = QCNet.load_from_checkpoint(checkpoint_path=str(args.ckpt), map_location=device)
    model.eval().to(device)
    dataset = ArgoverseV2Dataset(
        root=str(args.root), split="val",
        transform=TargetBuilderCompat(model.num_historical_steps, model.num_future_steps),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    minades: list[float] = []
    minfdes: list[float] = []
    misses: list[bool] = []
    t0 = time.time()

    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            data = data.to(device)
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
            pred = model(data)
            traj_refine = pred["loc_refine_pos"][..., : model.output_dim]  # [N, K, T, 2]
            eval_mask = data["agent"]["category"] == FOCAL_CATEGORY
            reg_mask = data["agent"]["predict_mask"][:, model.num_historical_steps:]
            traj_eval = traj_refine[eval_mask].cpu().numpy()  # [F, K, T, 2]
            gt_eval = data["agent"]["target"][eval_mask][..., : model.output_dim].cpu().numpy()
            valid_eval = reg_mask[eval_mask].cpu().numpy()  # [F, T]

            for f in range(traj_eval.shape[0]):
                if not valid_eval[f].all():
                    # focal agents in val have full 60-step futures; skip and count
                    # any exception loudly rather than silently averaging over gaps
                    print(f"WARN batch {i} focal {f}: partial future, skipped", flush=True)
                    continue
                fde = compute_fde(traj_eval[f], gt_eval[f])  # (K,)
                best = int(np.argmin(fde))
                ade = compute_ade(traj_eval[f], gt_eval[f])  # (K,)
                minfdes.append(float(fde[best]))
                minades.append(float(ade[best]))
                misses.append(bool(fde[best] > MISS_THRESHOLD_M))
            if i % 50 == 0:
                print(f"batch {i}, focal so far {len(minades)}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    n = len(minades)
    result = {
        "model": "QCNet released checkpoint (results/qcnet_ckpt_manifest.json)",
        "metric_authority": "av2.datasets.motion_forecasting.eval.metrics (official kit)",
        "protocol": "clean val, no occlusion masking (G-N1-0 eval-kit sanity)",
        "n_scenarios": n,
        "minade6": float(np.mean(minades)),
        "minfde6": float(np.mean(minfdes)),
        "mr6": float(np.mean(misses)),
        "miss_threshold_m": MISS_THRESHOLD_M,
        "seed": SEED,
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "date": time.strftime("%Y-%m-%d"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
