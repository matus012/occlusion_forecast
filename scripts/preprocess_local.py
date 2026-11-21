"""Preprocess AV2 scenarios into SIMPL features for the *-local subsets.

Wrap-and-import over third_party/SIMPL (D-N1-5 discipline: no vendored file is
copied or modified). Consumes a prefix of the deterministic subset orderings
produced by scripts/select_local_subsets.py and writes one pickle per scenario
to data/simpl_features/local/{split}/ (gitignored -- AV2-derived tensors never
enter history).

Resumable: existing output pickles are skipped, so the same command can be
re-run to extend a prefix after the budget freeze.

Throughput is measured and appended to results/local/preproc_rate.json --
this number feeds the D-N1-14d HPC extrapolation (preprocessing cost for the
full split is part of the honest budget table).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SIMPL_ROOT = REPO / "third_party" / "SIMPL"
sys.path.insert(0, str(SIMPL_ROOT / "data_av2"))

_WORKER: dict = {}


def _init_worker(data_dir: str, mode: str) -> None:
    from av2_preprocess import ArgoPreprocAV2  # noqa: PLC0415 (import in worker)

    args = SimpleNamespace(obs_len=50, pred_len=60, debug=False, viz=False, mode=mode)
    _WORKER["preproc"] = ArgoPreprocAV2(args, verbose=False)
    _WORKER["data_dir"] = Path(data_dir)


def _process_one(job: tuple[str, str]) -> tuple[str, bool, str]:
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.map.map_api import ArgoverseStaticMap

    sid, out_path = job
    try:
        seq_dir = _WORKER["data_dir"] / sid
        scenario = scenario_serialization.load_argoverse_scenario_parquet(
            seq_dir / f"scenario_{sid}.parquet")
        static_map = ArgoverseStaticMap.from_json(
            seq_dir / f"log_map_archive_{sid}.json")
        data, headers = _WORKER["preproc"].process(sid, scenario, static_map)
        pd.DataFrame(data, columns=headers).to_pickle(out_path)
        return sid, True, ""
    except Exception as exc:  # worker survives; failure is reported + counted
        return sid, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--count", type=int, required=True,
                    help="prefix length of the subset ordering to ensure on disk")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "av2")
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "data" / "simpl_features" / "local")
    args = ap.parse_args()

    order_file = (REPO / "results" / "local" /
                  ("train_pool.json" if args.split == "train" else "val_order.json"))
    order = json.loads(order_file.read_text(encoding="utf-8"))["order"]
    ids = [sid for sid, _ in order[: args.count]]

    out_dir = args.out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(sid, str(out_dir / f"{sid}.pkl")) for sid in ids
            if not (out_dir / f"{sid}.pkl").exists()]
    print(f"[{args.split}] prefix={args.count}, missing={len(jobs)}, workers={args.workers}")
    if not jobs:
        print("nothing to do")
        return

    t0 = time.time()
    n_ok, n_fail, failures = 0, 0, []
    mode = "train" if args.split == "train" else "val"
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(str(args.data_root / args.split), mode)) as pool:
        for i, (sid, ok, err) in enumerate(
                pool.imap_unordered(_process_one, jobs, chunksize=16), 1):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append({"sid": sid, "error": err})
                print(f"[FAIL] {sid}: {err}")
            if i % 500 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                eta_min = (len(jobs) - i) / rate / 60
                print(f"  {i}/{len(jobs)}  {rate:.1f} scen/s  ETA {eta_min:.1f} min",
                      flush=True)

    elapsed = time.time() - t0
    rate = n_ok / elapsed if elapsed > 0 else 0.0
    print(f"[done] ok={n_ok} fail={n_fail} in {elapsed / 60:.1f} min ({rate:.2f} scen/s)")

    rate_file = REPO / "results" / "local" / "preproc_rate.json"
    records = json.loads(rate_file.read_text(encoding="utf-8")) if rate_file.exists() else []
    records.append({
        "split": args.split, "n_processed": n_ok, "n_failed": n_fail,
        "workers": args.workers, "elapsed_s": round(elapsed, 1),
        "scen_per_s": round(rate, 3), "failures": failures,
    })
    rate_file.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
