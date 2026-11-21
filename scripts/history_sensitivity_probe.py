"""History-sensitivity probe (G-N1-5 collapse monitor, D-N1-14d).

Formalizes the D-N1-14c diagnostic that exposed the C3-local collapse: a
checkpoint that ignores agent-history content produces (near-)identical
predictions when the entire ACTORS feature tensor is replaced with noise.

Protocol (spec in gates.yaml G_N1_5_collapse_monitor):
  * probe set: first --n-probe (default 50) scenarios of the fixed val-subset
    ordering (results/local/val_order.json), clean inputs (S0);
  * per scenario: max |delta| between traj_pred on the true input and on the
    same input with ACTORS ~ N(0, 5^2), seeded;
  * probe value = MEDIAN over scenarios (robust to single outliers).

Reference points (results/local/c3_collapse_verification.json):
healthy C1-local ~65.7 m; collapsed C3-local ~1e-5 m. Provisional gate
threshold 1.0 m sits >4 orders of magnitude from both.

Usage:
  python scripts/history_sensitivity_probe.py --ckpt <path> [--label X]
      [--out results/local/probe_<label>.json]
Exit code 1 if the probe value is below --threshold (default 1.0) so training
scripts can call it per snapshot and react.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SIMPL"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _simpl_compat  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from config.simpl_av2_cfg import AdvCfg  # noqa: E402
from eval_local import MaskedEvalDataset  # noqa: E402
from simpl.simpl import Simpl  # noqa: E402

PROBE_SEED = 20260727


def probe_ckpt(ckpt_path: Path, features_dir: Path, n_probe: int) -> dict:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    net = Simpl(AdvCfg().get_net_cfg(), device).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    order = json.loads(
        (REPO / "results" / "local" / "val_order.json").read_text(encoding="utf-8")
    )["order"]
    gen = torch.Generator().manual_seed(PROBE_SEED)
    disps: list[float] = []
    for sid, _ in order[:n_probe]:
        f = [str(features_dir / f"{sid}.pkl")]
        ds = MaskedEvalDataset(f, "S0", "R-A", "native")
        data_a = ds.collate_fn([ds[0]])
        data_b = ds.collate_fn([ds[0]])
        data_b["ACTORS"] = torch.randn(
            data_b["ACTORS"].shape, generator=gen) * 5.0
        with torch.no_grad():
            pa = net.post_process(net(net.pre_process(data_a)))["traj_pred"]
            pb = net.post_process(net(net.pre_process(data_b)))["traj_pred"]
        disps.append(float((pa - pb).abs().max()))

    return {
        "ckpt": str(ckpt_path), "ckpt_epoch": ck.get("epoch"),
        "n_probe": len(disps), "probe_seed": PROBE_SEED,
        "median_disp_m": float(np.median(disps)),
        "min_disp_m": float(np.min(disps)),
        "max_disp_m": float(np.max(disps)),
        "reference": "healthy C1-local ~65.7 m; collapsed C3-local ~1e-5 m "
                     "(results/local/c3_collapse_verification.json)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-probe", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "simpl_features" / "local" / "val")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rec = probe_ckpt(args.ckpt, args.features_dir, args.n_probe)
    rec["threshold_m"] = args.threshold
    rec["passed"] = rec["median_disp_m"] >= args.threshold
    out = args.out or (REPO / "results" / "local" /
                       f"probe_{args.label or args.ckpt.stem}.json")
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in
                      ("ckpt", "median_disp_m", "threshold_m", "passed")}, indent=2))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
