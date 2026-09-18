"""Standalone, overnight-runnable script for Experiment 3 (plain numpy
"Ours" - joint rating + uncertainty, proposal eq. 4-5, r x lambda - see
week1/experiment3_ours_sweep.ipynb) across every dataset it's tractable
for. See run_experiment1_plain.py's module docstring for why ml-25m is
excluded and how the resume/output-file conventions work.

  python run_experiment3_ours.py
  python run_experiment3_ours.py --datasets ml-100k
  python run_experiment3_ours.py --quick

lambda here plays eq. (5)'s role (penalising the uncertainty head, not
embedding magnitude) - a much smaller natural scale than Experiment 2's
Lasso lambda, matching what experiments/run_ours.py's PyTorch version
already found (lambda=0.001 won on most datasets there). See
week1/experiment3_ours_sweep.ipynb's intro cell for the same note.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import mf_common
from evaluate import sweep, ABSTENTION_RATES

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
METHOD = "Ours (numpy, r x lambda)"

R_VALUES = [1, 2, 3, 4, 5, 10, 20, 50, 100]
# Second pass (upper bound 0.5) still came back edge-hugging: 3 of 4
# datasets picked the max lambda value (0.5) as the winner, the 4th (0.4)
# right next to it - still no plateau/collapse found. Pushed out to 2.0.
# Keeps every prior value below 0.5 (so this run's results are directly
# comparable/supersede both earlier passes at those points) and adds a
# comparably-spaced set of new points from 0.5 to 2.0.
LAM_VALUES = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045,
              0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
EPOCHS = 30

QUICK_R_VALUES = [2, 5]
QUICK_LAM_VALUES = [0.0, 0.001]
QUICK_EPOCHS = 2

# Per-example SGD at the 0.01 default applies ~10.4M updates on amazon-games
# (348k train rows x 30 epochs) and diverges to nonsense there regardless of
# (r, lam) - confirmed empirically 0.01 and 0.005 both blow up (garbage
# values like 1e146, not even a clean NaN) while 0.002 is stable and lands
# in the expected ~1.0-1.2 range. ml-100k/ml-1m/douban keep the default so
# their already-recorded results stay reproducible. See mf_common's
# _pool_run_ours docstring for the full story.
LEARNING_RATES = {"amazon-games": 0.002}
DEFAULT_LR = 0.01

GRID_FIELDNAMES = ["method", "dataset", "r", "lam", "val_rmse", "seconds"]
BEST_FIELDNAMES = ["method", "dataset", "best_r", "best_lam", "val_rmse", "test_rmse", "seconds"]
# Ours HAS its own learned uncertainty (U = CD^T, proposal eq. 5) - unlike
# experiment1/6/7/8's methods, it does NOT get the three naive abstention
# rules; it's scored by abstaining on the entries its own Uhat flags as least
# reliable, same convention as UAIMC's W in experiment5.
ABSTENTION_FIELDNAMES = ["method", "dataset", "p", "selective_rmse", "best_r", "best_lam", "val_rmse", "seconds"]


def run_dataset(dataset: str, r_values, lam_values, epochs, workers, out_path: Path):
    print(f"\n=== {dataset} ===")
    t0 = time.time()
    data = mf_common.load_dataset(dataset)
    print(f"{dataset}: {data['n_users']:,} users x {data['n_movies']:,} movies | "
          f"train={len(data['train_r']):,} val={len(data['val_r']):,} test={len(data['test_r']):,}")

    lr = LEARNING_RATES.get(dataset, DEFAULT_LR)

    existing = mf_common.load_existing_grid(out_path, METHOD, dataset, [("r", int), ("lam", float)])
    configs = [{"r": r, "lam": lam} for r in r_values for lam in lam_values if (r, lam) not in existing]
    print(f"{dataset}: {len(configs)} configs remaining ({len(existing)} already in {out_path.name}) "
          f"of {len(r_values) * len(lam_values)} total ({len(r_values)} r x {len(lam_values)} lambda), "
          f"learning_rate={lr}")

    def on_result(cfg, val_rmse, secs):
        mf_common.append_results(out_path, [{
            "method": METHOD, "dataset": dataset, "r": cfg["r"], "lam": cfg["lam"],
            "val_rmse": round(val_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)

    results = mf_common.grid_search(configs, data, mf_common._pool_run_ours, epochs, mf_common.SEED,
                                     workers, desc=f"{dataset} r x lambda sweep", on_result=on_result,
                                     learning_rate=lr)

    best_cfg, best_val_rmse = None, float("inf")
    for (r, lam), val_rmse in existing.items():
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = {"r": r, "lam": lam}, val_rmse
    for cfg, val_rmse, secs in results:
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = cfg, val_rmse

    # Retrain just the winner, alone - see mf_common's _pool_run_ours docstring
    # (this experiment holds 4 matrices per config, so this fix mattered most here).
    A, B, C, D, mean_rating, val_rmse = mf_common.train_ours_mf(
        best_cfg["r"], best_cfg["lam"], data["n_users"], data["n_movies"],
        data["train_u"], data["train_m"], data["train_r"], data["val_u"], data["val_m"], data["val_r"],
        epochs=epochs, seed=mf_common.SEED, learning_rate=lr,
    )
    test_rmse = mf_common.rmse(data["test_u"], data["test_m"], data["test_r"], A, B, mean_rating)
    best_row = {"method": METHOD, "dataset": dataset, "best_r": best_cfg["r"], "best_lam": best_cfg["lam"],
                "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
                "seconds": round(time.time() - t0, 1)}
    print(f"{dataset}: best (r={best_cfg['r']}, lam={best_cfg['lam']}) val_rmse={val_rmse:.4f} "
          f"test_rmse={test_rmse:.4f} ({time.time() - t0:.1f}s total)")

    # Score the SAME test predictions under Ours' own learned uncertainty
    # (Uhat = C.D^T, no mean-centering - this is a confidence score, not a
    # rating). Higher Uhat already means "less confident" (it's exactly what
    # train_ours_mf's w=exp(-Uhat) treats as low-weight), matching sweep's
    # "higher unreliability = abstain first" convention with no sign flip.
    pred = mf_common.predict(data["test_u"], data["test_m"], A, B, mean_rating)
    actual = data["test_r"]
    u_hat = mf_common.predict(data["test_u"], data["test_m"], C, D, mean=0.0)
    abstention_rows = [
        {"method": METHOD, "dataset": dataset, "p": p, "selective_rmse": round(sel_rmse, 4),
         "best_r": best_cfg["r"], "best_lam": best_cfg["lam"], "val_rmse": round(val_rmse, 4),
         "seconds": round(time.time() - t0, 1)}
        for p, sel_rmse in sweep(pred, actual, u_hat).items()
    ]

    return best_row, abstention_rows


def already_done_abstention(abstention_out: Path, dataset: str) -> bool:
    if not abstention_out.exists():
        return False
    import csv
    seen = set()
    with open(abstention_out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"] == METHOD:
                seen.add(row["p"])
    expected = {str(p) for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=mf_common.DATASETS, default=mf_common.DATASETS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments2" / "results" / "experiment3_ours_grid.csv")
    parser.add_argument("--best-out", type=Path, default=ROOT / "experiments2" / "results" / "experiment3_ours_best.csv")
    parser.add_argument("--abstention-out", type=Path,
                         default=ROOT / "experiments2" / "results" / "experiment3_ours_abstention.csv")
    parser.add_argument("--workers", type=int, default=mf_common.N_THREADS)
    args = parser.parse_args()

    r_values = QUICK_R_VALUES if args.quick else R_VALUES
    lam_values = QUICK_LAM_VALUES if args.quick else LAM_VALUES
    epochs = QUICK_EPOCHS if args.quick else EPOCHS

    print(f"CPU threads: {mf_common.N_THREADS} | workers: {args.workers}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}\nabstention results -> {args.abstention_out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and mf_common.already_done(args.best_out, METHOD, dataset) \
                and already_done_abstention(args.abstention_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name} and {args.abstention_out.name}, "
                  f"skipping (--force to rerun) ===")
            continue
        best_row, abstention_rows = run_dataset(dataset, r_values, lam_values, epochs, args.workers, args.out)
        mf_common.append_results(args.best_out, [best_row], BEST_FIELDNAMES)
        mf_common.append_results(args.abstention_out, abstention_rows, ABSTENTION_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
