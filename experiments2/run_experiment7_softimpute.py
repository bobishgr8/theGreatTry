"""SoftImpute (Hastie, Mazumder, Lee & Zadeh 2015) baseline, moved into
experiments2's conventions from experiments/run_softimpute.py (same
algorithm, relocated and switched to the grid+best CSV pair every other
experiment*.py here uses).

  python run_experiment7_softimpute.py                    # every tractable dataset
  python run_experiment7_softimpute.py --datasets ml-100k
  python run_experiment7_softimpute.py --quick             # tiny grid, smoke-test the pipeline

Unlike Plain MF, SoftImpute can't work rating-by-rating: each iteration
fills a dense n_users x n_items matrix with the current low-rank estimate,
takes its SVD, shrinks the singular values by sigma, and reconstructs.
Datasets over SIZE_LIMIT_ENTRIES are skipped with a clear message instead of
attempted (would OOM or hang for a very long time) - see module docstring
math in the original experiments/run_softimpute.py for the exact byte sizes.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent

import mf_common
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games"]
SEED = mf_common.SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE_LIMIT_ENTRIES = 100_000_000  # douban (94.6M) fits, amazon-games (961M) doesn't

METHOD = "SoftImpute"
# Densified around sigma=10 after the original 7-value log grid picked that
# SAME value as the winner on all 3 datasets tonight (an interior optimum,
# not an edge - unlike AutoRec/IGMC - so this adds resolution around the
# known-good region between 3 and 30 rather than extending the range).
SIGMA_GRID = [1e-2, 1e-1, 1.0, 3.0, 5.0, 7.0, 10.0, 13.0, 15.0, 20.0, 30.0, 100.0]
MAX_ITER = 100
TOL = 1e-5

QUICK_SIGMA_GRID = [10.0]
QUICK_MAX_ITER = 3

GRID_FIELDNAMES = ["method", "dataset", "sigma", "val_rmse", "seconds"]
BEST_FIELDNAMES = ["method", "dataset", "p", "selective_rmse", "best_sigma", "val_rmse", "seconds"]


def build_dense(rows, n_users, n_items, device):
    matrix = torch.zeros((n_users, n_items), dtype=torch.float32, device=device)
    mask = torch.zeros((n_users, n_items), dtype=torch.bool, device=device)
    u = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(rows[:, 1], dtype=torch.long, device=device)
    r = torch.as_tensor(rows[:, 2], dtype=torch.float32, device=device)
    matrix[u, i] = r
    mask[u, i] = True
    return matrix, mask


def soft_impute(matrix, mask, sigma, max_iter, tol, heartbeat_u=None, heartbeat_i=None, heartbeat_r=None,
                 heartbeat_every=None, tag=None):
    mean_rating = matrix[mask].mean()
    centered = torch.zeros_like(matrix)
    centered[mask] = matrix[mask] - mean_rating

    estimate = torch.zeros_like(matrix)
    iters_run = 0
    for it in range(max_iter):
        filled = torch.where(mask, centered, estimate)
        U, S, Vh = torch.linalg.svd(filled, full_matrices=False)
        S = torch.clamp(S - sigma, min=0.0)
        new_estimate = (U * S) @ Vh

        old_norm_sq = torch.sum(estimate * estimate)
        change = torch.sum((new_estimate - estimate) ** 2) / torch.clamp(old_norm_sq, min=1e-12)
        estimate = new_estimate
        iters_run = it + 1

        if heartbeat_every and heartbeat_u is not None and iters_run % heartbeat_every == 0:
            preview = estimate[heartbeat_u, heartbeat_i] + mean_rating
            v_rmse = torch.sqrt(torch.mean((preview - heartbeat_r) ** 2)).item()
            print(f"    [{tag}] iter {iters_run}/{max_iter}: change={change.item():.2e} val_rmse~={v_rmse:.4f}")

        if change.item() < tol:
            break

    return estimate, mean_rating.item(), iters_run


def run_dataset(dataset: str, sigma_grid, max_iter, tol, clip: bool, grid_out: Path) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = mf_common.load_dataset(dataset, seed=SEED)
    n_users, n_items = split["n_users"], split["n_movies"]
    n_entries = n_users * n_items
    print(f"{dataset}: {n_users:,} users x {n_items:,} items = {n_entries:,} dense entries | "
          f"train={len(split['train_u']):,} val={len(split['val_u']):,} test={len(split['test_u']):,}")

    if n_entries > SIZE_LIMIT_ENTRIES:
        print(f"{dataset}: SKIPPED - a dense {n_entries:,}-entry matrix is over the "
              f"{SIZE_LIMIT_ENTRIES:,}-entry safety limit (see module docstring).")
        return []

    train_rows = np.column_stack([split["train_u"], split["train_m"], split["train_r"]])
    val_rows = np.column_stack([split["val_u"], split["val_m"], split["val_r"]])
    test_rows = np.column_stack([split["test_u"], split["test_m"], split["test_r"]])

    train_matrix, train_mask = build_dense(train_rows, n_users, n_items, DEVICE)
    val_u = torch.as_tensor(val_rows[:, 0], dtype=torch.long, device=DEVICE)
    val_i = torch.as_tensor(val_rows[:, 1], dtype=torch.long, device=DEVICE)
    val_r = torch.as_tensor(val_rows[:, 2], dtype=torch.float32, device=DEVICE)

    existing = mf_common.load_existing_grid(grid_out, METHOD, dataset, [("sigma", float)])
    todo = [s for s in sigma_grid if (s,) not in existing]
    heartbeat_every = max(1, max_iter // 10)
    print(f"{dataset}: {len(sigma_grid)} sigma values ({len(sigma_grid) - len(todo)} already done), "
          f"max_iter={max_iter}, tol={tol:.0e}")

    best_sigma, best_val_rmse = None, float("inf")
    for (s,), v in existing.items():
        if v < best_val_rmse:
            best_val_rmse, best_sigma = v, s

    for n, sigma in enumerate(todo, 1):
        t0 = time.time()
        estimate, mean_rating, iters_run = soft_impute(
            train_matrix, train_mask, sigma, max_iter, tol,
            heartbeat_u=val_u, heartbeat_i=val_i, heartbeat_r=val_r,
            heartbeat_every=heartbeat_every, tag=f"sigma={sigma:g}",
        )
        pred = estimate[val_u, val_i] + mean_rating
        if clip:
            pred = pred.clamp(1, 5)
        v_rmse = torch.sqrt(torch.mean((pred - val_r) ** 2)).item()
        secs = time.time() - t0

        mf_common.append_results(grid_out, [{
            "method": METHOD, "dataset": dataset, "sigma": sigma, "val_rmse": round(v_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)
        if v_rmse < best_val_rmse:
            best_val_rmse, best_sigma = v_rmse, sigma
        print(f"  [{n}/{len(todo)}] sigma={sigma:g} -> val_rmse={v_rmse:.4f} best={best_val_rmse:.4f} "
              f"({iters_run} iters, {secs:.1f}s)")

    print(f"{dataset}: best sigma={best_sigma:g} (val_rmse={best_val_rmse:.4f})")

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    trainval_matrix, trainval_mask = build_dense(trainval_rows, n_users, n_items, DEVICE)
    test_u = torch.as_tensor(test_rows[:, 0], dtype=torch.long, device=DEVICE)
    test_i = torch.as_tensor(test_rows[:, 1], dtype=torch.long, device=DEVICE)
    test_r = torch.as_tensor(test_rows[:, 2], dtype=torch.float32, device=DEVICE)

    t0 = time.time()
    final_estimate, final_mean, iters_run = soft_impute(
        trainval_matrix, trainval_mask, best_sigma, max_iter, tol,
        heartbeat_u=test_u, heartbeat_i=test_i, heartbeat_r=test_r,
        heartbeat_every=heartbeat_every, tag=f"{dataset} refit",
    )
    print(f"{dataset}: refit on train+val done ({iters_run} iters, {time.time() - t0:.1f}s)")

    pred = final_estimate[test_u, test_i] + final_mean
    if clip:
        pred = pred.clamp(1, 5)
    pred = pred.cpu().numpy()
    actual = test_rows[:, 2]
    test_user_idx = test_rows[:, 0].astype(int)
    trainval_user_idx = trainval_rows[:, 0].astype(int)

    rng = np.random.default_rng(SEED)
    random_score = rng.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_user_idx.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_user_idx].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("val", -pred), ("supp", support_score)]:
        for p, val in sweep(pred, actual, score).items():
            rows_out.append({
                "method": f"SoftImpute ({rule_name})",
                "dataset": dataset,
                "p": p,
                "selective_rmse": round(val, 4),
                "best_sigma": best_sigma,
                "val_rmse": round(best_val_rmse, 4),
                "seconds": round(time.time() - t_dataset0, 1),
            })

    print(f"{dataset}: done in {time.time() - t_dataset0:.1f}s total")
    return rows_out


def already_done_best(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    import csv
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("SoftImpute"):
                seen.add((row["method"], row["p"]))
    expected = {(f"SoftImpute ({rule})", str(p)) for rule in ("rand", "val", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=[d for d in DATASETS if d != "ml-25m"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "experiment7_softimpute_grid.csv")
    parser.add_argument("--best-out", type=Path, default=HERE / "results" / "experiment7_softimpute_best.csv")
    parser.add_argument("--clip", action="store_true", help="clip predictions to [1,5] (off by default)")
    args = parser.parse_args()

    sigma_grid = QUICK_SIGMA_GRID if args.quick else SIGMA_GRID
    max_iter = QUICK_MAX_ITER if args.quick else MAX_ITER

    print(f"device: {DEVICE}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done_best(args.best_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, sigma_grid, max_iter, TOL, args.clip, args.out)
        if rows:
            mf_common.append_results(args.best_out, rows, BEST_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.best_out}")


if __name__ == "__main__":
    main()
