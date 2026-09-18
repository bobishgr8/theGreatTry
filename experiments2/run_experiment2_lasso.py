"""Standalone, overnight-runnable script for Experiment 2 (plain numpy MF +
L1/Lasso weight decay, r x lambda - see
week1/experiment2_lasso_mf_sweep.ipynb) across every dataset it's
tractable for. See run_experiment1_plain.py's module docstring for why
ml-25m is excluded and how the resume/output-file conventions work - all
identical here, just a 2D grid instead of 1D.

  python run_experiment2_lasso.py
  python run_experiment2_lasso.py --datasets ml-100k
  python run_experiment2_lasso.py --quick
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import mf_common

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
METHOD = "Lasso MF (numpy, r x lambda)"

R_VALUES = [1, 2, 3, 4, 5, 10, 20, 50, 100]
# sparse near 0, narrowing hard approaching 0.5 - see week1/experiment2's intro cell
LAM_VALUES = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
EPOCHS = 30

QUICK_R_VALUES = [2, 5]
QUICK_LAM_VALUES = [0.0, 0.1]
QUICK_EPOCHS = 2

GRID_FIELDNAMES = ["method", "dataset", "r", "lam", "val_rmse", "seconds"]
BEST_FIELDNAMES = ["method", "dataset", "best_r", "best_lam", "val_rmse", "test_rmse", "seconds"]


def run_dataset(dataset: str, r_values, lam_values, epochs, workers, out_path: Path):
    print(f"\n=== {dataset} ===")
    t0 = time.time()
    data = mf_common.load_dataset(dataset)
    print(f"{dataset}: {data['n_users']:,} users x {data['n_movies']:,} movies | "
          f"train={len(data['train_r']):,} val={len(data['val_r']):,} test={len(data['test_r']):,}")

    existing = mf_common.load_existing_grid(out_path, METHOD, dataset, [("r", int), ("lam", float)])
    configs = [{"r": r, "lam": lam} for r in r_values for lam in lam_values if (r, lam) not in existing]
    print(f"{dataset}: {len(configs)} configs remaining ({len(existing)} already in {out_path.name}) "
          f"of {len(r_values) * len(lam_values)} total ({len(r_values)} r x {len(lam_values)} lambda)")

    def on_result(cfg, val_rmse, secs):
        mf_common.append_results(out_path, [{
            "method": METHOD, "dataset": dataset, "r": cfg["r"], "lam": cfg["lam"],
            "val_rmse": round(val_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)

    results = mf_common.grid_search(configs, data, mf_common._pool_run_lasso, epochs, mf_common.SEED,
                                     workers, desc=f"{dataset} r x lambda sweep", on_result=on_result)

    best_cfg, best_val_rmse = None, float("inf")
    for (r, lam), val_rmse in existing.items():
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = {"r": r, "lam": lam}, val_rmse
    for cfg, val_rmse, secs in results:
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = cfg, val_rmse

    # Retrain just the winner, alone - see mf_common's _pool_run_lasso docstring.
    A, B, mean_rating, val_rmse = mf_common.train_lasso_mf(
        best_cfg["r"], best_cfg["lam"], data["n_users"], data["n_movies"],
        data["train_u"], data["train_m"], data["train_r"], data["val_u"], data["val_m"], data["val_r"],
        epochs=epochs, seed=mf_common.SEED,
    )
    test_rmse = mf_common.rmse(data["test_u"], data["test_m"], data["test_r"], A, B, mean_rating)
    best_row = {"method": METHOD, "dataset": dataset, "best_r": best_cfg["r"], "best_lam": best_cfg["lam"],
                "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
                "seconds": round(time.time() - t0, 1)}
    print(f"{dataset}: best (r={best_cfg['r']}, lam={best_cfg['lam']}) val_rmse={val_rmse:.4f} "
          f"test_rmse={test_rmse:.4f} ({time.time() - t0:.1f}s total)")
    return best_row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=mf_common.DATASETS, default=mf_common.DATASETS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments2" / "results" / "experiment2_lasso_grid.csv")
    parser.add_argument("--best-out", type=Path, default=ROOT / "experiments2" / "results" / "experiment2_lasso_best.csv")
    parser.add_argument("--workers", type=int, default=mf_common.N_THREADS)
    args = parser.parse_args()

    r_values = QUICK_R_VALUES if args.quick else R_VALUES
    lam_values = QUICK_LAM_VALUES if args.quick else LAM_VALUES
    epochs = QUICK_EPOCHS if args.quick else EPOCHS

    print(f"CPU threads: {mf_common.N_THREADS} | workers: {args.workers}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and mf_common.already_done(args.best_out, METHOD, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name}, skipping (--force to rerun) ===")
            continue
        best_row = run_dataset(dataset, r_values, lam_values, epochs, args.workers, args.out)
        mf_common.append_results(args.best_out, [best_row], BEST_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
