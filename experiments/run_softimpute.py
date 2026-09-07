"""Standalone, overnight-runnable script for SoftImpute (Hastie, Mazumder,
Lee & Zadeh 2015): sigma sweep + refit + abstention scoring, across every
dataset small enough to fit a dense matrix. Produces Table 1's
"SoftImpute (rand)"/"SoftImpute (supp)" rows.

  python run_softimpute.py                    # every tractable dataset
  python run_softimpute.py --datasets ml-100k
  python run_softimpute.py --quick             # tiny grid, smoke-test the pipeline

Unlike Plain MF, SoftImpute can't work rating-by-rating: each iteration
fills a dense n_users x n_items matrix with the current low-rank estimate,
takes its SVD, shrinks the singular values by sigma, and reconstructs. That
caps which datasets this can even attempt:

  ml-100k        5.9M entries   ~47MB  (float32)  - fine
  ml-1m         22.4M entries  ~179MB (float32)  - fine
  douban        94.6M entries  ~757MB (float32)  - slow but doable
  amazon-games 961.2M entries  ~3.8GB (float32)  - SVD cost alone is impractical
  ml-25m         9.6B entries  ~76.8GB(float32)  - doesn't even fit in memory

Datasets over SIZE_LIMIT_ENTRIES are skipped with a clear message instead of
attempted (which would OOM or hang for a very long time). A real fix would
reformulate each SVD as a sparse-residual-plus-low-rank matvec (what the
actual softImpute R package does via PROPACK) so it never touches a dense
matrix - not implemented here.

Resumable exactly like run_plain_mf.py: results append to --out as each
dataset finishes, and a dataset already fully scored is skipped on restart
unless --force is passed.
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE_LIMIT_ENTRIES = 100_000_000  # douban (94.6M) fits, amazon-games (961M) doesn't

SIGMA_GRID = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
MAX_ITER = 100
TOL = 1e-5  # relative squared-norm change between iterations

QUICK_SIGMA_GRID = [10.0]
QUICK_MAX_ITER = 3

FIELDNAMES = ["method", "dataset", "p", "selective_rmse", "best_sigma", "val_rmse", "seconds"]


def build_dense(rows, n_users, n_items, device):
    """A dense (matrix, observed-mask) pair for SoftImpute's fill+SVD step -
    the one place this method fundamentally can't stay sparse."""
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
    """Centers by the observed training mean (same reason as Plain MF's
    centering: a near-zero low-rank estimate should predict "the mean", not
    "zero stars"), then repeats fill -> SVD -> soft-threshold -> reconstruct
    until the relative change between iterations drops below tol.

    heartbeat_* prints a val RMSE preview every heartbeat_every iterations
    purely for overnight-log visibility (see run_plain_mf.py's identical
    reasoning) - it's unclipped and not the metric actually reported."""
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


def already_done(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("SoftImpute"):
                seen.add((row["method"], row["p"]))
    expected = {(f"SoftImpute ({rule})", str(p)) for rule in ("rand", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, sigma_grid, max_iter, tol, clip: bool) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    n_entries = split.n_users * split.n_items
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items = {n_entries:,} dense entries | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    if n_entries > SIZE_LIMIT_ENTRIES:
        print(f"{dataset}: SKIPPED - a dense {n_entries:,}-entry matrix is over the "
              f"{SIZE_LIMIT_ENTRIES:,}-entry safety limit (see module docstring for why "
              f"SoftImpute can't just work rating-by-rating like Plain MF).")
        return []

    train_matrix, train_mask = build_dense(split.train, split.n_users, split.n_items, DEVICE)
    val_u = torch.as_tensor(split.val[:, 0], dtype=torch.long, device=DEVICE)
    val_i = torch.as_tensor(split.val[:, 1], dtype=torch.long, device=DEVICE)
    val_r = torch.as_tensor(split.val[:, 2], dtype=torch.float32, device=DEVICE)

    heartbeat_every = max(1, max_iter // 10)
    print(f"{dataset}: {len(sigma_grid)} sigma values, max_iter={max_iter}, tol={tol:.0e}")

    best_sigma, best_val_rmse = None, float("inf")
    pbar = tqdm(sigma_grid, desc=f"{dataset} sigma sweep")
    for n, sigma in enumerate(pbar, 1):
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

        if v_rmse < best_val_rmse:
            best_val_rmse, best_sigma = v_rmse, sigma
        pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
        pbar.write(f"  [{n}/{len(sigma_grid)}] sigma={sigma:g} -> val_rmse={v_rmse:.4f} "
                   f"({iters_run} iters, {time.time() - t0:.1f}s)")

    print(f"{dataset}: best sigma={best_sigma:g} (val_rmse={best_val_rmse:.4f})")

    trainval_rows = np.concatenate([split.train, split.val], axis=0)
    trainval_matrix, trainval_mask = build_dense(trainval_rows, split.n_users, split.n_items, DEVICE)
    test_u = torch.as_tensor(split.test[:, 0], dtype=torch.long, device=DEVICE)
    test_i = torch.as_tensor(split.test[:, 1], dtype=torch.long, device=DEVICE)
    test_r = torch.as_tensor(split.test[:, 2], dtype=torch.float32, device=DEVICE)

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
    actual = split.test[:, 2]
    test_user_idx = split.test[:, 0].astype(int)
    trainval_user_idx = trainval_rows[:, 0].astype(int)

    rng = np.random.default_rng(SEED)
    random_score = rng.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_user_idx.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_user_idx].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("supp", support_score)]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/iter budget, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "softimpute_results.csv")
    parser.add_argument("--clip", action="store_true", help="clip predictions to [1,5] (off by default, per week2/README.md)")
    args = parser.parse_args()

    sigma_grid = QUICK_SIGMA_GRID if args.quick else SIGMA_GRID
    max_iter = QUICK_MAX_ITER if args.quick else MAX_ITER

    print(f"device: {DEVICE}")
    print(f"results file: {args.out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, sigma_grid, max_iter, TOL, args.clip)
        if rows:
            append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
