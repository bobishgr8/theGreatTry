"""Standalone, overnight-runnable script for "Ours" (proposal eq. 4-5, joint
rating + uncertainty factorisation): grid search + refit + abstention scoring.
Produces Table 1's "Ours (U_hat)" row.

Promotes experiments/mf_vs_ours.ipynb's OursModel/joint_loss to the same
resumable, early-stopping, GPU-first pattern as run_plain_mf.py:

  python run_ours.py                       # every dataset except ml-25m
  python run_ours.py --datasets ml-100k
  python run_ours.py --quick               # tiny grid, smoke-test the pipeline

Unlike Plain MF / IGMC / SoftImpute / AutoRec, "Ours" has its own learned
uncertainty (U = C D^T), so there is exactly one row per (dataset, p) here,
not a (rand, supp) pair - the model abstains on its own U_hat, per eq. (5).

ml-25m is left out of DATASETS below: Plain MF's grid search on ml-25m alone
already needed ~50min/config and repeatedly triggered system-wide low-memory
kills even single-threaded (see run_plain_mf.py's DEFAULT_WORKERS comment).
Ours trains twice as many embedding tables (A,B,C,D vs A,B) per step, so it
would only be worse; revisit once Plain MF's ml-25m run itself is solved.
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

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, see module docstring
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_THREADS = os.cpu_count() or 1
DEFAULT_WORKERS = 1  # see run_plain_mf.py's identical reasoning - shared 8GB GPU, 16GB host RAM
torch.set_num_threads(N_THREADS)

# Same grid shape (3 k x 2 lr x 3 third-axis = 18 configs) as run_plain_mf.py's
# K_GRID/LR_GRID/WD_GRID, so the two methods get a like-for-like search budget
# per proposal Sec. 4.2's fairness note - lam replaces weight_decay as the
# third axis since Ours has no weight_decay term (eq. 5 has none; only lam
# regularises U, see joint_loss below).
K_GRID = [10, 20, 50]
LR_GRID = [0.003, 0.01]
LAM_GRID = [1e-5, 1e-4, 1e-3]
MAX_EPOCHS = 200
PATIENCE = 10
BATCH_SIZE = 4096

QUICK_K_GRID = [10]
QUICK_LR_GRID = [0.01]
QUICK_LAM_GRID = [1e-4]
QUICK_MAX_EPOCHS = 3
QUICK_PATIENCE = 3

FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_k", "best_lr", "best_lam", "val_rmse", "seconds",
]


class OursModel(nn.Module):
    """S = A B^T (rating head), U = C D^T (uncertainty head) - proposal eq. (4)."""

    def __init__(self, n_users, n_items, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(n_users, k, generator=g) * 0.1)
        self.B = nn.Parameter(torch.randn(n_items, k, generator=g) * 0.1)
        self.C = nn.Parameter(torch.randn(n_users, k, generator=g) * 0.1)
        self.D = nn.Parameter(torch.randn(n_items, k, generator=g) * 0.1)

    def forward(self, u, i):
        s = (self.A[u] * self.B[i]).sum(dim=1)
        u_hat = (self.C[u] * self.D[i]).sum(dim=1)
        return s, u_hat


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


def joint_loss(s, u_hat, actual, lam, full_n):
    """Proposal eq. (5): mean((S-R)^2 * exp(-U)) + lam * sum_Omega(U), the
    second term's sum estimated from this minibatch and rescaled to the full
    training set (an unbiased estimator since minibatches partition a random
    permutation) so lam doesn't need re-tuning whenever batch_size changes."""
    mse_term = torch.mean((s - actual) ** 2 * torch.exp(-u_hat))
    reg_term = lam * u_hat.sum() * (full_n / len(u_hat))
    return mse_term + reg_term


def train_ours(
    tr_u, tr_i, tr_r, va_u, va_i, va_r, n_users, n_items, k, lr, lam, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    """Early stopping on validation RMSE of the *rating* head alone (U_hat is
    not scored during selection - the proposal's hypothesis is precisely that
    training with an uncertainty signal should improve the rating head, so
    selecting on rating RMSE is the fair test of that, matching Sec. 4.2's
    "Hyperparameters selected on validation full RMSE").

    Same mean-centering as Plain MF (see its docstring for why): S has no
    intercept, so a near-zero embedding should predict "the mean", not "zero
    stars". U is left uncentered - it has no such degenerate-at-zero problem,
    exp(-U) already saturates smoothly on both sides."""
    device = tr_u.device
    mean_rating = tr_r.mean()
    tr_r_c = tr_r - mean_rating
    va_r_c = va_r - mean_rating

    model = OursModel(n_users, n_items, k, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    full_n = len(tr_r)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"k={k} lr={lr} lam={lam:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        model.train()
        for idx in minibatches(full_n, batch_size, device, gen):
            opt.zero_grad()
            s, u_hat = model(tr_u[idx], tr_i[idx])
            loss = joint_loss(s, u_hat, tr_r_c[idx], lam, full_n)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            s_val, _ = model(va_u, va_i)
            v_rmse = rmse(s_val, va_r_c)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"k={k} lr={lr} lam={lam:.0e}"
            print(f"    [{tag}] epoch {epoch + 1}/{max_epochs}: val_rmse={v_rmse:.4f} (best={best_rmse:.4f})")

        if v_rmse < best_rmse - 1e-5:
            best_rmse = v_rmse
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_rmse, epochs_run, mean_rating.item()


_worker_data = {}


def _pool_init(train_rows, val_rows, n_users, n_items, threads_per_worker):
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
    heartbeat_every = max(1, max_epochs // 10)
    _, v_rmse, epochs_run, _ = train_ours(
        train_u, train_i, train_r, val_u, val_i, val_r,
        _worker_data["n_users"], _worker_data["n_items"],
        max_epochs=max_epochs, patience=patience, batch_size=batch_size, seed=seed,
        heartbeat_every=heartbeat_every, **cfg,
    )
    return cfg, v_rmse, epochs_run, time.time() - t0


def already_done(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"] == "Ours (U_hat)":
                seen.add(row["p"])
    expected = {str(p) for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, grid_k, grid_lr, grid_lam, max_epochs, patience, batch_size, workers) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    test_u, test_i, test_r = to_tensors(split.test, DEVICE)
    trainval_rows = np.concatenate([split.train, split.val], axis=0)
    trainval_u, trainval_i, trainval_r = to_tensors(trainval_rows, DEVICE)

    configs = [{"k": k, "lr": lr, "lam": lam} for k in grid_k for lr in grid_lr for lam in grid_lam]
    print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, patience={patience}, "
          f"batch_size={batch_size}, workers={workers}")

    best_cfg, best_val_rmse = None, float("inf")
    threads_per_worker = max(1, N_THREADS // max(1, workers))

    if workers <= 1:
        _pool_init(split.train, split.val, split.n_users, split.n_items, N_THREADS)
        pbar = tqdm(configs, desc=f"{dataset} grid search")
        for n, cfg in enumerate(pbar, 1):
            _, v_rmse, epochs_run, dt = _pool_train_config(cfg, max_epochs, patience, batch_size, SEED)
            if v_rmse < best_val_rmse:
                best_val_rmse, best_cfg = v_rmse, cfg
            pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
            pbar.write(f"  [{n}/{len(configs)}] k={cfg['k']:>3} lr={cfg['lr']} lam={cfg['lam']:.0e} "
                       f"-> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {dt:.1f}s)")
    else:
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
                pbar.write(f"  [{n}/{len(configs)}] k={cfg['k']:>3} lr={cfg['lr']} lam={cfg['lam']:.0e} "
                           f"-> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {dt:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    t0 = time.time()
    final_model, _, epochs_run, final_mean = train_ours(
        trainval_u, trainval_i, trainval_r, test_u, test_i, test_r,
        split.n_users, split.n_items, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, seed=SEED, desc=f"{dataset} refit", show_progress=True, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    with torch.no_grad():
        pred, u_hat = final_model(test_u, test_i)
        pred = pred.cpu().numpy() + final_mean
        u_hat = u_hat.cpu().numpy()
    actual = split.test[:, 2]

    rows_out = []
    for p, val in sweep(pred, actual, u_hat).items():
        rows_out.append({
            "method": "Ours (U_hat)",
            "dataset": dataset,
            "p": p,
            "selective_rmse": round(val, 4),
            "best_k": best_cfg["k"],
            "best_lr": best_cfg["lr"],
            "best_lam": best_cfg["lam"],
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
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "ours_results.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    grid_k, grid_lr, grid_lam = (
        (QUICK_K_GRID, QUICK_LR_GRID, QUICK_LAM_GRID) if args.quick else (K_GRID, LR_GRID, LAM_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    workers = 1 if args.quick else args.workers

    print(f"device: {DEVICE} | CPU threads: {N_THREADS} | workers: {workers}")
    print(f"results file: {args.out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, grid_k, grid_lr, grid_lam, max_epochs, patience, args.batch_size, workers)
        append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
