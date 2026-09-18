"""Standalone, overnight-runnable ablation: does the uncertainty matrix
U_hat = C D^T (proposal eq. 4) need the same rank as the rating matrix
S = A B^T, or can it be smaller (a coarse, cheap confidence signal) or does
it actually want to be larger (uncertainty as a harder pattern to capture
than the ratings themselves)?

Nothing in eq. 4-5 requires rank(C,D) = rank(A,B) - that's just the
simplest choice experiment 3 made, not a mathematical necessity. This
script holds the rating rank r FIXED at whatever experiment 3 already
found best for that dataset (so we're isolating exactly one variable) and
sweeps the uncertainty rank r_u both below and above it, plus a small
lambda grid alongside.

  python run_experiment4_rank_ablation.py
  python run_experiment4_rank_ablation.py --datasets ml-100k
  python run_experiment4_rank_ablation.py --quick

Uses mf_common.train_ours_mf_ranks (see its docstring) and the exact same
resumable-grid / ProcessPoolExecutor / per-dataset-learning-rate machinery
as run_experiment3_ours.py - see that file's module docstring and
mf_common.py's _pool_run_ours docstring for why workers>1 and a per-dataset
learning-rate override exist at all (amazon-games-specific numerical
instability at the default 0.01, and this machine's ~16GB RAM ceiling).
amazon-games is excluded from the *default* dataset list (not from
mf_common.DATASETS) purely for overnight time budget - experiment 3's
single-rank sweep there alone took ~10 hours; pass --datasets amazon-games
explicitly if you want it too.
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
METHOD = "Ours rank-ablation (numpy, r_u x lambda)"

# Overnight-budget default: skip amazon-games (see module docstring).
DEFAULT_DATASETS = ["ml-100k", "ml-1m", "douban"]

# The rating rank r is held fixed per dataset at whatever
# experiments2/results/experiment3_ours_best.csv already found best there,
# so this experiment varies exactly one thing (r_u) relative to that
# already-completed run, not two things at once. amazon-games's r=100 is
# included for completeness even though it's not in DEFAULT_DATASETS.
FIXED_RATING_RANK = {"ml-100k": 2, "ml-1m": 5, "douban": 3, "amazon-games": 100}

# Spans both well below and well above every FIXED_RATING_RANK value above,
# so "smaller than the rating matrix" and "larger than the rating matrix"
# are both genuinely represented for every dataset, not just relative to
# one of them.
RANK_U_VALUES = [1, 2, 5, 10, 20, 50]
LAM_VALUES = [0.001, 0.01, 0.1]
EPOCHS = 30

QUICK_RANK_U_VALUES = [1, 10]
QUICK_LAM_VALUES = [0.001]
QUICK_EPOCHS = 2

# Same amazon-games instability as experiment 3 (see mf_common's
# _pool_run_ours docstring) - identical fix, identical reasoning, since the
# update rule itself (train_ours_mf_ranks) is unchanged from train_ours_mf.
LEARNING_RATES = {"amazon-games": 0.002}
DEFAULT_LR = 0.01

GRID_FIELDNAMES = ["method", "dataset", "r", "r_u", "lam", "val_rmse", "seconds"]
BEST_FIELDNAMES = ["method", "dataset", "r", "best_r_u", "best_lam", "val_rmse", "test_rmse", "seconds"]
# Same reasoning as experiment3_ours.py: this variant has its own learned
# Uhat = C D^T too (r_u just varies its rank), so it's scored the same way -
# no external abstention rules, abstain on what its own Uhat flags.
ABSTENTION_FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse", "r", "best_r_u", "best_lam", "val_rmse", "seconds",
]


def run_dataset(dataset: str, rank_u_values, lam_values, epochs, workers, out_path: Path):
    print(f"\n=== {dataset} ===")
    t0 = time.time()
    data = mf_common.load_dataset(dataset)
    print(f"{dataset}: {data['n_users']:,} users x {data['n_movies']:,} movies | "
          f"train={len(data['train_r']):,} val={len(data['val_r']):,} test={len(data['test_r']):,}")

    r = FIXED_RATING_RANK[dataset]
    lr = LEARNING_RATES.get(dataset, DEFAULT_LR)

    existing = mf_common.load_existing_grid(out_path, METHOD, dataset, [("r_u", int), ("lam", float)])
    configs = [{"r": r, "r_u": r_u, "lam": lam} for r_u in rank_u_values for lam in lam_values
               if (r_u, lam) not in existing]
    print(f"{dataset}: rating rank r={r} (fixed, from experiment 3's best), {len(configs)} configs remaining "
          f"({len(existing)} already in {out_path.name}) of {len(rank_u_values) * len(lam_values)} total "
          f"({len(rank_u_values)} r_u x {len(lam_values)} lambda), learning_rate={lr}")

    def on_result(cfg, val_rmse, secs):
        mf_common.append_results(out_path, [{
            "method": METHOD, "dataset": dataset, "r": cfg["r"], "r_u": cfg["r_u"], "lam": cfg["lam"],
            "val_rmse": round(val_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)

    results = mf_common.grid_search(configs, data, mf_common._pool_run_ours_ranks, epochs, mf_common.SEED,
                                     workers, desc=f"{dataset} r_u x lambda sweep (r={r} fixed)",
                                     on_result=on_result, learning_rate=lr)

    best_cfg, best_val_rmse = None, float("inf")
    for (r_u, lam), val_rmse in existing.items():
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = {"r": r, "r_u": r_u, "lam": lam}, val_rmse
    for cfg, val_rmse, secs in results:
        if val_rmse < best_val_rmse:
            best_cfg, best_val_rmse = cfg, val_rmse

    # Retrain just the winner, alone - same memory reasoning as experiment 3
    # (mf_common's _pool_run_ours docstring): holding every config's A/B/C/D
    # alive at once, rather than just the winner's, is what exhausts host RAM.
    A, B, C, D, mean_rating, val_rmse = mf_common.train_ours_mf_ranks(
        best_cfg["r"], best_cfg["r_u"], best_cfg["lam"], data["n_users"], data["n_movies"],
        data["train_u"], data["train_m"], data["train_r"], data["val_u"], data["val_m"], data["val_r"],
        epochs=epochs, seed=mf_common.SEED, learning_rate=lr,
    )
    test_rmse = mf_common.rmse(data["test_u"], data["test_m"], data["test_r"], A, B, mean_rating)
    best_row = {"method": METHOD, "dataset": dataset, "r": r, "best_r_u": best_cfg["r_u"],
                "best_lam": best_cfg["lam"], "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
                "seconds": round(time.time() - t0, 1)}
    print(f"{dataset}: best (r={r} fixed, r_u={best_cfg['r_u']}, lam={best_cfg['lam']}) "
          f"val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} ({time.time() - t0:.1f}s total)")

    # Same Uhat-based abstention sweep as experiment3_ours.py - see its
    # comment for why no sign flip is needed.
    pred = mf_common.predict(data["test_u"], data["test_m"], A, B, mean_rating)
    actual = data["test_r"]
    u_hat = mf_common.predict(data["test_u"], data["test_m"], C, D, mean=0.0)
    abstention_rows = [
        {"method": METHOD, "dataset": dataset, "p": p, "selective_rmse": round(sel_rmse, 4),
         "r": r, "best_r_u": best_cfg["r_u"], "best_lam": best_cfg["lam"], "val_rmse": round(val_rmse, 4),
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
    parser.add_argument("--datasets", nargs="+", choices=mf_common.DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments2" / "results" / "experiment4_rank_ablation_grid.csv")
    parser.add_argument("--best-out", type=Path, default=ROOT / "experiments2" / "results" / "experiment4_rank_ablation_best.csv")
    parser.add_argument("--abstention-out", type=Path,
                         default=ROOT / "experiments2" / "results" / "experiment4_rank_ablation_abstention.csv")
    parser.add_argument("--workers", type=int, default=2)  # see run_plain_mf.py's DEFAULT_WORKERS for why not N_THREADS
    args = parser.parse_args()

    rank_u_values = QUICK_RANK_U_VALUES if args.quick else RANK_U_VALUES
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
        best_row, abstention_rows = run_dataset(dataset, rank_u_values, lam_values, epochs, args.workers, args.out)
        mf_common.append_results(args.best_out, [best_row], BEST_FIELDNAMES)
        mf_common.append_results(args.abstention_out, abstention_rows, ABSTENTION_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
