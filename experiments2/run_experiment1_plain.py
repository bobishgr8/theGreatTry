"""Standalone, overnight-runnable script for Experiment 1 (plain numpy MF,
r as the only hyperparameter - see week1/experiment1_plain_mf_sweep.ipynb)
across every dataset it's tractable for.

  python run_experiment1_plain.py                    # every dataset in mf_common.DATASETS
  python run_experiment1_plain.py --datasets ml-100k
  python run_experiment1_plain.py --quick             # tiny grid, smoke-test the pipeline

Why ml-25m isn't in mf_common.DATASETS at all (unlike experiments/'s
PyTorch/GPU scripts, which at least attempt it): this is pure per-example
Python - no vectorisation, no GPU - so throughput is roughly fixed at
~124,500 example-epochs/sec/core (measured on ml-100k: 70,585 train rows x
30 epochs in ~17s). At that rate one ml-25m config (17.5M train rows x 30
epochs) alone takes ~70 minutes; a 9-point r-grid would take over 10 hours
on a single core, and Experiments 2-3's much larger (r, lambda) grids would
take days. ml-1m/douban/amazon-games all fit comfortably in an overnight
run (worst case here, douban's full grid, is under 20 minutes).

Resumable exactly like experiments/run_plain_mf.py: a (method, dataset)
pair already in --out is skipped on restart unless --force is passed.

Writes two files: --out (one row per grid point, every r tried) and
--best-out (one row per dataset, the winning r's val/test RMSE) - the
former is what a future plotting script would read to reproduce the
notebook's val-RMSE-by-r curve, the latter mirrors experiments/'s
best-config-per-dataset convention.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import mf_common
from evaluate import sweep, ABSTENTION_RATES

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
METHOD = "Plain MF (numpy, r-only)"

R_VALUES = [1, 2, 3, 4, 5, 10, 20, 50, 100]
EPOCHS = 30

QUICK_R_VALUES = [2, 5]
QUICK_EPOCHS = 2

GRID_FIELDNAMES = ["method", "dataset", "r", "val_rmse", "seconds"]
BEST_FIELDNAMES = ["method", "dataset", "best_r", "val_rmse", "test_rmse", "seconds"]
# Plain MF has no learned uncertainty of its own, so per proposal Sec. 4.3 it
# gets all three naive abstention rules (Algorithms 1-3) applied externally,
# same convention as experiments 6/7/8 (AutoRec/SoftImpute/IGMC).
ABSTENTION_FIELDNAMES = ["method", "dataset", "p", "selective_rmse", "best_r", "val_rmse", "seconds"]


def run_dataset(dataset: str, r_values, epochs, workers, out_path: Path):
    print(f"\n=== {dataset} ===")
    t0 = time.time()
    data = mf_common.load_dataset(dataset)
    print(f"{dataset}: {data['n_users']:,} users x {data['n_movies']:,} movies | "
          f"train={len(data['train_r']):,} val={len(data['val_r']):,} test={len(data['test_r']):,}")

    existing = mf_common.load_existing_grid(out_path, METHOD, dataset, [("r", int)])
    configs = [{"r": r} for r in r_values if (r,) not in existing]
    print(f"{dataset}: {len(configs)} configs remaining ({len(existing)} already in {out_path.name}) "
          f"of {len(r_values)} total")

    def on_result(cfg, val_rmse, secs):
        mf_common.append_results(out_path, [{
            "method": METHOD, "dataset": dataset, "r": cfg["r"],
            "val_rmse": round(val_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)

    results = mf_common.grid_search(configs, data, mf_common._pool_run_plain, epochs, mf_common.SEED,
                                     workers, desc=f"{dataset} r sweep", on_result=on_result)

    best_cfg, best_val_rmse = None, float("inf")
    for (r,), val_rmse in existing.items():
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = {"r": r}, val_rmse
    for cfg, val_rmse, secs in results:
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = cfg, val_rmse

    # Retrain just the winner, alone, to get the A/B this grid search never
    # held onto for every config at once - see mf_common's _pool_run_plain docstring.
    A, B, mean_rating, val_rmse = mf_common.train_plain_mf(
        best_cfg["r"], data["n_users"], data["n_movies"],
        data["train_u"], data["train_m"], data["train_r"], data["val_u"], data["val_m"], data["val_r"],
        epochs=epochs, seed=mf_common.SEED,
    )
    test_rmse = mf_common.rmse(data["test_u"], data["test_m"], data["test_r"], A, B, mean_rating)
    best_row = {"method": METHOD, "dataset": dataset, "best_r": best_cfg["r"],
                "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
                "seconds": round(time.time() - t0, 1)}
    print(f"{dataset}: best r={best_cfg['r']} val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} "
          f"({time.time() - t0:.1f}s total)")

    # Plain MF has no uncertainty of its own (proposal Sec. 4.3): score the
    # same test predictions under all three naive abstention rules so it can
    # be compared fairly against Ours/UAIMC's learned-uncertainty abstention.
    pred = mf_common.predict(data["test_u"], data["test_m"], A, B, mean_rating)
    actual = data["test_r"]
    rng = np.random.default_rng(mf_common.SEED)
    random_score = rng.random(len(actual))
    n_users_total = int(max(data["train_u"].max(), data["test_u"].max())) + 1
    support = np.bincount(data["train_u"], minlength=n_users_total)
    support_score = -support[data["test_u"]].astype(np.float64)

    abstention_rows = []
    for rule_name, score in [("rand", random_score), ("val", -pred), ("supp", support_score)]:
        for p, sel_rmse in sweep(pred, actual, score).items():
            abstention_rows.append({
                "method": f"{METHOD} ({rule_name})", "dataset": dataset, "p": p,
                "selective_rmse": round(sel_rmse, 4), "best_r": best_cfg["r"],
                "val_rmse": round(val_rmse, 4), "seconds": round(time.time() - t0, 1),
            })

    return best_row, abstention_rows


def already_done_abstention(abstention_out: Path, dataset: str) -> bool:
    if not abstention_out.exists():
        return False
    import csv
    seen = set()
    with open(abstention_out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith(METHOD):
                seen.add((row["method"], row["p"]))
    expected = {(f"{METHOD} ({rule})", str(p)) for rule in ("rand", "val", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=mf_common.DATASETS, default=mf_common.DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/epoch budget, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments2" / "results" / "experiment1_plain_grid.csv")
    parser.add_argument("--best-out", type=Path, default=ROOT / "experiments2" / "results" / "experiment1_plain_best.csv")
    parser.add_argument("--abstention-out", type=Path,
                         default=ROOT / "experiments2" / "results" / "experiment1_plain_abstention.csv")
    parser.add_argument("--workers", type=int, default=mf_common.N_THREADS)
    args = parser.parse_args()

    r_values = QUICK_R_VALUES if args.quick else R_VALUES
    epochs = QUICK_EPOCHS if args.quick else EPOCHS

    print(f"CPU threads: {mf_common.N_THREADS} | workers: {args.workers}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}\nabstention results -> {args.abstention_out}")

    t0 = time.time()
    for dataset in args.datasets:
        # A dataset only counts as fully done once BOTH the point-RMSE best
        # row and all three abstention rules are present - so a run that
        # already has the former (e.g. last night's, from before this
        # abstention sweep existed) is correctly detected as needing a rerun,
        # not silently skipped.
        if not args.force and mf_common.already_done(args.best_out, METHOD, dataset) \
                and already_done_abstention(args.abstention_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name} and {args.abstention_out.name}, "
                  f"skipping (--force to rerun) ===")
            continue
        best_row, abstention_rows = run_dataset(dataset, r_values, epochs, args.workers, args.out)
        mf_common.append_results(args.best_out, [best_row], BEST_FIELDNAMES)
        mf_common.append_results(args.abstention_out, abstention_rows, ABSTENTION_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
