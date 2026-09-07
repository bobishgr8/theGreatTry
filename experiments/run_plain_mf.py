"""Standalone, overnight-runnable script for Plain MF (Koren et al.): grid
search + refit + abstention scoring, across every dataset in the IS470
proposal. Produces Table 1's "Plain MF (rand)" / "Plain MF (supp)" rows.

Unlike experiments/mf_vs_ours.ipynb (interactive, ML-100K only, fixed epoch
budget), this is meant to be kicked off once and left running unattended:

  python run_plain_mf.py                       # full run, all 5 datasets
  python run_plain_mf.py --datasets ml-100k    # just one dataset
  python run_plain_mf.py --quick               # tiny grid, smoke-test the pipeline

Resumable: results are appended to --out as each dataset finishes, and a
dataset already fully scored in --out is skipped on the next run unless
--force is passed. Safe to kill and restart.

Early stopping (not a fixed epoch count) is what makes a single overnight
run across ML-100K through ML-25M tractable: ML-25M has ~250x ML-100K's
training rows, so giving every dataset the same fixed epoch budget would
either waste hours on ML-100K or make ML-25M's grid search take far longer
than a night.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

print = functools.partial(print, flush=True)  # every print reaches a redirected logfile immediately
# tqdm auto-detects a non-tty (e.g. output redirected to a logfile) and falls
# back to occasional plain lines instead of \r-spamming it, so the bars below
# are free in an interactive terminal and harmless in an overnight log.

ROOT = Path(__file__).resolve().parent.parent  # theGreatTry/experiments -> theGreatTry
sys.path.append(str(ROOT / "dataset loaders and cleaners"))

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_THREADS = os.cpu_count() or 1
DEFAULT_WORKERS = min(4, N_THREADS)
torch.set_num_threads(N_THREADS)

# Deliberately modest so a full overnight run across all five datasets
# (ML-25M included) finishes in a reasonable window - widen if you have
# more time/GPU budget than one night.
K_GRID = [10, 20, 50]
LR_GRID = [0.003, 0.01]
WD_GRID = [1e-3, 1e-2, 5e-2]
MAX_EPOCHS = 200
PATIENCE = 10  # stop a config once val RMSE hasn't improved for this many epochs
BATCH_SIZE = 4096

QUICK_K_GRID = [10]
QUICK_LR_GRID = [0.01]
QUICK_WD_GRID = [1e-2]
QUICK_MAX_EPOCHS = 3
QUICK_PATIENCE = 3

FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_k", "best_lr", "best_weight_decay", "val_rmse", "seconds",
]


class PlainMF(nn.Module):
    def __init__(self, n_users, n_items, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(n_users, k, generator=g) * 0.1)
        self.B = nn.Parameter(torch.randn(n_items, k, generator=g) * 0.1)

    def forward(self, u, i):
        return (self.A[u] * self.B[i]).sum(dim=1)


def to_tensors(rows, device):
    u = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(rows[:, 1], dtype=torch.long, device=device)
    r = torch.as_tensor(rows[:, 2], dtype=torch.float32, device=device)
    return u, i, r


def minibatches(n, batch_size, device, generator):
    perm = torch.randperm(n, device=device, generator=generator)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def rmse(pred, actual):
    return torch.sqrt(torch.mean((pred - actual) ** 2)).item()


def train_plain_mf(
    tr_u, tr_i, tr_r, va_u, va_i, va_r, n_users, n_items, k, lr, weight_decay, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    """Trains with early stopping, restoring the best-val-RMSE checkpoint
    (not just whatever epoch happened to run last) before returning.

    S = A B^T has no bias/intercept term (proposal eq. 4, literally), but
    ratings average ~3.5 while embeddings start small - so hitting that
    absolute scale requires large embedding norms, which weight_decay
    directly fights (at high weight_decay the model just collapses toward
    predicting 0, i.e. RMSE ~= sqrt(mean(rating^2))). Centering the training
    target by its own mean - exactly what softImpute.ipynb already does
    before its SVD step - means a near-zero embedding predicts "the mean"
    instead of "zero stars", so weight_decay's pull no longer fights the
    model's ability to reach a sane baseline. RMSE is unaffected by this
    shift (it's invariant to subtracting the same constant from both sides),
    only trainability is; the mean is added back wherever this model's
    output is used for an actual rating prediction."""
    device = tr_u.device
    mean_rating = tr_r.mean()
    tr_r_c = tr_r - mean_rating
    va_r_c = va_r - mean_rating

    model = PlainMF(n_users, n_items, k, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=device).manual_seed(seed)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"k={k} lr={lr} wd={weight_decay:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        model.train()
        for idx in minibatches(len(tr_r), batch_size, device, gen):
            opt.zero_grad()
            pred = model(tr_u[idx], tr_i[idx])
            loss = torch.mean((pred - tr_r_c[idx]) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_rmse = rmse(model(va_u, va_i), va_r_c)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"k={k} lr={lr} wd={weight_decay:.0e}"
            print(f"    [{tag}] epoch {epoch + 1}/{max_epochs}: val_rmse={v_rmse:.4f} (best={best_rmse:.4f})")

        if v_rmse < best_rmse - 1e-5:
            best_rmse = v_rmse
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_rmse, epochs_run, mean_rating.item()


_worker_data = {}  # populated once per worker process by _pool_init, not per task


def _pool_init(train_rows, val_rows, n_users, n_items, threads_per_worker):
    """Runs once when each worker process starts (not once per config), so the
    - potentially large, e.g. ~400MB for ML-25M - train/val arrays are sent
    over IPC only len(workers) times per dataset, not once per config. Each
    worker gets its own CUDA context; the OS/driver time-slices the physical
    GPU across them, which is the actual point (a single Plain MF training
    loop is too small to saturate the GPU or need many CPU threads on its
    own, so several running concurrently is what cuts wall-clock time)."""
    global _worker_data
    torch.set_num_threads(max(1, threads_per_worker))
    _worker_data = {
        "train": to_tensors(train_rows, DEVICE),
        "val": to_tensors(val_rows, DEVICE),
        "n_users": n_users,
        "n_items": n_items,
    }


def _pool_train_config(cfg, max_epochs, patience, batch_size, seed):
    train_u, train_i, train_r = _worker_data["train"]
    val_u, val_i, val_r = _worker_data["val"]
    t0 = time.time()
    # ~10 heartbeat lines per config so a slow config (e.g. ML-25M) still
    # shows visible movement, since the outer grid-search bar only advances
    # once a whole config finishes and shows nothing while one is in flight.
    heartbeat_every = max(1, max_epochs // 10)
    _, v_rmse, epochs_run, _ = train_plain_mf(
        train_u, train_i, train_r, val_u, val_i, val_r,
        _worker_data["n_users"], _worker_data["n_items"],
        max_epochs=max_epochs, patience=patience, batch_size=batch_size, seed=seed,
        heartbeat_every=heartbeat_every, **cfg,
    )
    return cfg, v_rmse, epochs_run, time.time() - t0


def already_done(results_path: Path, dataset: str) -> bool:
    """True if results_path already has every (rule, p) row for this dataset."""
    if not results_path.exists():
        return False
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("Plain MF"):
                seen.add((row["method"], row["p"]))
    expected = {(f"Plain MF ({rule})", str(p)) for rule in ("rand", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, grid_k, grid_lr, grid_wd, max_epochs, patience, batch_size, workers) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    test_u, test_i, test_r = to_tensors(split.test, DEVICE)
    trainval_rows = np.concatenate([split.train, split.val], axis=0)
    trainval_u, trainval_i, trainval_r = to_tensors(trainval_rows, DEVICE)

    configs = [{"k": k, "lr": lr, "weight_decay": wd} for k in grid_k for lr in grid_lr for wd in grid_wd]
    print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, patience={patience}, "
          f"batch_size={batch_size}, workers={workers}")

    best_cfg, best_val_rmse = None, float("inf")
    threads_per_worker = max(1, N_THREADS // max(1, workers))

    if workers <= 1:
        # sequential fallback (also just simpler to debug against)
        _pool_init(split.train, split.val, split.n_users, split.n_items, N_THREADS)
        pbar = tqdm(configs, desc=f"{dataset} grid search")
        for n, cfg in enumerate(pbar, 1):
            _, v_rmse, epochs_run, dt = _pool_train_config(cfg, max_epochs, patience, batch_size, SEED)
            if v_rmse < best_val_rmse:
                best_val_rmse, best_cfg = v_rmse, cfg
            pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
            pbar.write(f"  [{n}/{len(configs)}] k={cfg['k']:>3} lr={cfg['lr']} wd={cfg['weight_decay']:.0e} "
                       f"-> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {dt:.1f}s)")
    else:
        # Several configs train concurrently, each its own process/CUDA context,
        # sharing the physical GPU - a single Plain MF training loop is too
        # small to saturate the GPU on its own, so this is what actually cuts
        # wall-clock time (see _pool_init's docstring).
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_pool_init,
            initargs=(split.train, split.val, split.n_users, split.n_items, threads_per_worker),
        ) as executor:
            futures = {
                executor.submit(_pool_train_config, cfg, max_epochs, patience, batch_size, SEED): cfg
                for cfg in configs
            }
            pbar = tqdm(as_completed(futures), total=len(configs), desc=f"{dataset} grid search ({workers} workers)")
            for n, future in enumerate(pbar, 1):
                cfg, v_rmse, epochs_run, dt = future.result()
                if v_rmse < best_val_rmse:
                    best_val_rmse, best_cfg = v_rmse, cfg
                pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
                pbar.write(f"  [{n}/{len(configs)}] k={cfg['k']:>3} lr={cfg['lr']} wd={cfg['weight_decay']:.0e} "
                           f"-> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {dt:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    t0 = time.time()
    final_model, _, epochs_run, final_mean = train_plain_mf(
        trainval_u, trainval_i, trainval_r, test_u, test_i, test_r,
        split.n_users, split.n_items, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, seed=SEED, desc=f"{dataset} refit", show_progress=True, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    with torch.no_grad():
        pred = final_model(test_u, test_i).cpu().numpy() + final_mean
    actual = split.test[:, 2]
    test_user_idx = split.test[:, 0].astype(int)
    trainval_user_idx = trainval_rows[:, 0].astype(int)

    rng = np.random.default_rng(SEED)
    random_score = rng.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_user_idx.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_user_idx].astype(np.float64)  # more support = more reliable

    rows_out = []
    for rule_name, score in [("rand", random_score), ("supp", support_score)]:
        for p, val in sweep(pred, actual, score).items():
            rows_out.append({
                "method": f"Plain MF ({rule_name})",
                "dataset": dataset,
                "p": p,
                "selective_rmse": round(val, 4),
                "best_k": best_cfg["k"],
                "best_lr": best_cfg["lr"],
                "best_weight_decay": best_cfg["weight_decay"],
                "val_rmse": round(best_val_rmse, 4),
                "seconds": round(time.time() - t_dataset0, 1),
            })

    print(f"{dataset}: done in {time.time() - t_dataset0:.1f}s total")
    return rows_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/epoch budget, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "table1_results.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"configs trained concurrently per dataset (default {DEFAULT_WORKERS}; "
                              "1 disables multiprocessing). Plain MF is too small to saturate the GPU "
                              "on its own, so this - not more CPU threads per run - is the real lever "
                              "for wall-clock speed; each worker is its own process/CUDA context "
                              "sharing the physical GPU, so pushing it too high just thrashes VRAM/WDDM "
                              "scheduling instead of helping.")
    args = parser.parse_args()

    grid_k, grid_lr, grid_wd = (QUICK_K_GRID, QUICK_LR_GRID, QUICK_WD_GRID) if args.quick else (K_GRID, LR_GRID, WD_GRID)
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    workers = 1 if args.quick else args.workers  # not worth pool-startup overhead for a smoke test

    print(f"device: {DEVICE} | CPU threads: {N_THREADS} | workers: {workers}")
    print(f"results file: {args.out}")
    if "ml-25m" in args.datasets and not args.quick:
        print("NOTE: ML-25M's grid search is the long pole here even with early stopping; "
              "if you want to bound total runtime, run it separately with --datasets ml-25m "
              "so the smaller datasets aren't stuck waiting behind it.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, grid_k, grid_lr, grid_wd, max_epochs, patience, args.batch_size, workers)
        append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
