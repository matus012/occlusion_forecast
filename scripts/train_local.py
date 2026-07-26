"""Local reduced-scale SIMPL training (D-N1-14a): arms C1-local / C3-local.

Wrap-and-import over third_party/SIMPL (no vendored file modified). Single
4060 GPU, single seed, FIXED epoch count, FINAL-epoch checkpoint for both arms
(no model selection at all -- symmetric across arms, and the val subset is
never touched during training; the dev slice below is disjoint monitoring
only). Everything this script produces is labeled *-local and never mixes
with the real gate arms (G-N1-1..3).

Arms:
  c1  -- clean training (p_occ = 0).
  c3  -- occlusion-aug per spec (D-N1-2 + D-N1-9): p_occ = 0.5, severity ~
         uniform {S1..S4}, pattern ~ empirical mix, regime R-A, fresh draw
         per (scenario, epoch) via otraj.masking.generator.draw_train_mask,
         applied per the NATIVE SIMPL convention (otraj.masking.simpl_apply).

Data: prefixes of the deterministic city-stratified train pool
(results/local/train_pool.json); dev slice = the pool entries immediately
AFTER the training prefix (disjoint by construction).

LR schedule: SIMPL's polyline schedule is defined for 50 epochs
(milestones [0, 5, 35, 40]); for E epochs the milestones scale by E/50 --
same shape, compressed. Values unchanged.

Outputs:
  checkpoints/local/{arm}_seed{seed}.tar            (final; gitignored)
  results/local/train_{arm}_seed{seed}.json          (per-epoch metrics,
      throughput, peak VRAM -- feeds the D-N1-14d HPC extrapolation)
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
import torch  # noqa: E402
from config.simpl_av2_cfg import AdvCfg  # noqa: E402  (SIMPL, wrap-and-import)
from simpl.av2_dataset import AV2Dataset  # noqa: E402
from simpl.av2_loss_fn import LossFunc  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.evaluator import TrajPredictionEvaluator  # noqa: E402
from utils.optimizer import Optimizer  # noqa: E402
from utils.utils import AverageMeterForDict, save_ckpt, set_seed  # noqa: E402

from otraj.masking.generator import draw_train_mask  # noqa: E402
from otraj.masking.simpl_apply import apply_native_mask  # noqa: E402


class LocalAV2Dataset(AV2Dataset):
    """AV2Dataset over an explicit file list, with optional occlusion aug.

    Occlusion is injected in data_augmentation() -- after the parent's flip
    aug, before feature assembly -- so the standard __getitem__/collate path
    (disp/ang/vel/type/PAD_OBS channels) consumes the masked tensors exactly
    as it would consume natively-missing steps.
    """

    def __init__(self, files: list[str], mode: str, aug: bool, occlusion: bool):
        self._files_override = list(files)
        self.occlusion = occlusion
        self.epoch = 0
        # parent scans a directory; hand it the parent dir then override
        super().__init__(str(Path(files[0]).parent), mode=mode, obs_len=50,
                         pred_len=60, aug=aug, verbose=False)
        self.dataset_files = self._files_override
        self.dataset_len = len(self._files_override)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def data_augmentation(self, df):
        data = super().data_augmentation(df)
        if self.occlusion:
            res = draw_train_mask(data["SEQ_ID"], self.epoch)
            if res.mask.any():
                apply_native_mask(data["TRAJS"], res.mask)
        return data


def build_file_lists(features_dir: Path, train_count: int, dev_count: int,
                     ) -> tuple[list[str], list[str]]:
    order = json.loads(
        (REPO / "results" / "local" / "train_pool.json").read_text(encoding="utf-8")
    )["order"]
    ids = [sid for sid, _ in order[: train_count + dev_count]]
    files = [str(features_dir / "train" / f"{sid}.pkl") for sid in ids]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} preprocessed files missing (first: {missing[0]}); "
            "run scripts/preprocess_local.py first")
    return files[:train_count], files[train_count:]


def scaled_opt_cfg(cfg: AdvCfg, epochs: int) -> dict:
    opt_cfg = cfg.get_opt_cfg()
    if opt_cfg["scheduler"] == "polyline":
        opt_cfg["milestones"] = [round(m * epochs / 50) for m in opt_cfg["milestones"]]
    return opt_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["c1", "c3"], required=True)
    ap.add_argument("--train-count", type=int, required=True)
    ap.add_argument("--dev-count", type=int, default=500)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=0,
                    help="probe mode: stop after N iterations, save nothing")
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "simpl_features" / "local")
    args = ap.parse_args()

    set_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA required (this run IS the VRAM evidence)"
    device = torch.device("cuda", 0)
    # TF32 on Ada: standard training precision for this class of model; noted
    # in the throughput record (H200 extrapolation uses the same setting).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_files, dev_files = build_file_lists(
        args.features_dir, args.train_count, args.dev_count)
    train_set = LocalAV2Dataset(train_files, mode="train", aug=True,
                                occlusion=args.arm == "c3")
    dev_set = LocalAV2Dataset(dev_files, mode="val", aug=False, occlusion=False)
    print(f"[data] arm={args.arm} train={len(train_set)} dev={len(dev_set)}")

    cfg = AdvCfg()
    net = Simpl(cfg.get_net_cfg(), device).to(device)
    loss_fn = LossFunc(cfg.get_loss_cfg(), device)
    optimizer = Optimizer(net, scaled_opt_cfg(cfg, args.epochs))
    evaluator = TrajPredictionEvaluator(cfg.get_eval_cfg())
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] SIMPL av2 params={n_params}")

    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=train_set.collate_fn,
                          drop_last=True, pin_memory=True)
    dl_dev = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=dev_set.collate_fn,
                        drop_last=False, pin_memory=True)

    run_name = f"{args.arm}_seed{args.seed}"
    ckpt_dir = REPO / "checkpoints" / "local"
    log_path = REPO / "results" / "local" / f"train_{run_name}.json"
    log: dict = {
        "label": f"{args.arm.upper()}-local",
        "arm": args.arm, "seed": args.seed, "epochs": args.epochs,
        "train_count": args.train_count, "dev_count": args.dev_count,
        "batch_size": args.batch_size, "n_params": n_params,
        "occlusion_aug": args.arm == "c3",
        "p_occ": 0.5 if args.arm == "c3" else 0.0,
        "note": "*-local reduced-scale proof; never mixes with gate arms",
        "device": torch.cuda.get_device_name(0),
        "epochs_log": [],
    }

    t_start = time.time()
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net.train()
        meter, eval_meter = AverageMeterForDict(), AverageMeterForDict()
        t_ep = time.time()
        n_iters = 0
        for data in tqdm(dl_train, ncols=80, desc=f"E{epoch}", mininterval=10.0):
            data_in = net.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data)
            optimizer.zero_grad()
            loss_out["loss"].backward()
            lr = optimizer.step()
            with torch.no_grad():
                eval_meter.update(evaluator.evaluate(net.post_process(out), data))
            meter.update({"loss": loss_out["loss"].detach()})
            n_iters += 1
            if args.max_iters and n_iters >= args.max_iters:
                break
        optimizer.step_scheduler()
        ep_wall = time.time() - t_ep
        peak_mb = torch.cuda.max_memory_allocated(device) // 2**20
        ep_rec = {
            "epoch": epoch, "lr": lr,
            "train_loss": float(meter.metrics["loss"].avg),
            "train_minfde_k": float(eval_meter.metrics["minfde_k"].avg)
            if "minfde_k" in eval_meter.metrics else None,
            "wall_s": round(ep_wall, 1), "iters": n_iters,
            "iters_per_s": round(n_iters / ep_wall, 3),
            "samples_per_s": round(n_iters * args.batch_size / ep_wall, 2),
            "peak_vram_mb": int(peak_mb),
        }

        if args.max_iters:
            print(json.dumps(ep_rec, indent=2))
            return  # probe mode: measurement printed, nothing persisted

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            net.eval()
            dev_meter = AverageMeterForDict()
            with torch.no_grad():
                for data in dl_dev:
                    out = net(net.pre_process(data))
                    dev_meter.update(evaluator.evaluate(net.post_process(out), data))
            ep_rec["dev"] = {k: float(v.avg) for k, v in dev_meter.metrics.items()}
            print(f"[dev E{epoch}] " + dev_meter.get_info())

        log["epochs_log"].append(ep_rec)
        log["total_wall_s"] = round(time.time() - t_start, 1)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"[E{epoch}] loss={ep_rec['train_loss']:.4f} "
              f"{ep_rec['samples_per_s']:.0f} samp/s peak={peak_mb}MB "
              f"wall={ep_wall / 60:.1f}min lr={lr:.2e}", flush=True)

        if epoch in {round(args.epochs * f) for f in (0.25, 0.5, 0.75)}:
            save_ckpt(net, optimizer.opt, epoch, str(ckpt_dir), f"{run_name}_e{epoch}.tar")

    save_ckpt(net, optimizer.opt, args.epochs - 1, str(ckpt_dir), f"{run_name}.tar")
    log["final_ckpt"] = str(ckpt_dir / f"{run_name}.tar")
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[done] {run_name}: {(time.time() - t_start) / 3600:.2f} h total")


if __name__ == "__main__":
    main()
