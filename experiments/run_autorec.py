"""Standalone, overnight-runnable script for I-AutoRec (Sedhain et al. 2015):
grid search + refit + abstention scoring. Produces Table 1's "AutoRec (rand)"
/ "AutoRec (supp)" rows.

  python run_autorec.py                    # every dataset small enough (see below)
  python run_autorec.py --datasets ml-100k
  python run_autorec.py --quick             # tiny grid, smoke-test the pipeline

Item-based AutoRec: each item is one training example, a length-n_users
vector of that item's ratings (0 where unobserved), autoencoded through one
sigmoid hidden layer and a linear (identity) output layer, with the
reconstruction loss masked to only the observed entries - exactly like
run_softimpute.py, this needs the rating matrix dense at least once, so it
inherits that script's SIZE_LIMIT_ENTRIES cutoff (ml-25m, amazon-games are
skipped; see run_softimpute.py's module docstring for the exact byte math).
Unlike SoftImpute's per-iteration SVD, one autoencoder forward/backward pass
over the whole (small) dense matrix is cheap, so this trains full-batch (one
"batch" = every item) rather than minibatching - simpler, and the datasets
this can even attempt already fit comfortably on an 8GB GPU (douban's dense
matrix is ~378MB in float32).

No weight-decay-vs-batch-size interaction to worry about here, unlike Plain
MF: nn.Linear's weights are dense and touched by every example every step
regardless of anything, so Adam's own weight_decay is fine as-is.

nn.Linear's decoder bias absorbs the rating scale (~3.5 stars) directly, so
- unlike Plain MF/SoftImpute/Ours, whose bilinear S=AB^T forms have no
intercept - AutoRec needs no explicit mean-centering trick.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
import time
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

DATASETS = ["ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_THREADS = os.cpu_count() or 1
torch.set_num_threads(N_THREADS)

SIZE_LIMIT_ENTRIES = 100_000_000  # same cutoff and reasoning as run_softimpute.py

HIDDEN_GRID = [50, 100, 200]
LR_GRID = [0.003, 0.01]
WD_GRID = [1e-4, 1e-3, 1e-2]
MAX_EPOCHS = 200
PATIENCE = 10

QUICK_HIDDEN_GRID = [20]
QUICK_LR_GRID = [0.01]
QUICK_WD_GRID = [1e-3]
QUICK_MAX_EPOCHS = 3
QUICK_PATIENCE = 3

FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_lr", "best_weight_decay", "val_rmse", "seconds",
]


class AutoRec(nn.Module):
    def __init__(self, n_users, hidden, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(n_users, hidden)
        self.decoder = nn.Linear(hidden, n_users)

    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))
        return self.decoder(h)


def build_dense_items(rows, n_users, n_items, device):
    """(n_items, n_users) dense matrix + observed-mask, item-major so each row
    is one AutoRec training example."""
    matrix = torch.zeros((n_items, n_users), dtype=torch.float32, device=device)
    mask = torch.zeros((n_items, n_users), dtype=torch.bool, device=device)
    u = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(rows[:, 1], dtype=torch.long, device=device)
    r = torch.as_tensor(rows[:, 2], dtype=torch.float32, device=device)
    matrix[i, u] = r
    mask[i, u] = True
    return matrix, mask


def masked_rmse(pred, matrix, mask):
    diff = (pred - matrix)[mask]
    return torch.sqrt(torch.mean(diff**2)).item()


def train_autorec(
    train_matrix, train_mask, val_matrix, val_mask, hidden, lr, weight_decay, max_epochs, patience, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    """Full-batch: every item is in every step, so there is no minibatch loop
    - one forward/backward pass over the whole dense matrix per epoch. Trains
    to reconstruct train_matrix (masked to train-observed cells only); val is
    scored by feeding the same train_matrix through the model (val/test cells
    are never part of the input, only of what's compared against on output),
    exactly like run_softimpute.py's heartbeat/val evaluation."""
    n_users = train_matrix.shape[1]
    model = AutoRec(n_users, hidden, seed).to(train_matrix.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"hidden={hidden} lr={lr} wd={weight_decay:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        model.train()
        opt.zero_grad()
        recon = model(train_matrix)
        loss = torch.mean(((recon - train_matrix) ** 2)[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            recon = model(train_matrix)
            v_rmse = masked_rmse(recon, val_matrix, val_mask)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"hidden={hidden} lr={lr} wd={weight_decay:.0e}"
            print(f"    [{tag}] epoch {epoch + 1}/{max_epochs}: val_rmse={v_rmse:.4f} (best={best_rmse:.4f})")

        if v_rmse < best_rmse - 1e-5:
            best_rmse = v_rmse
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_rmse, epochs_run


def already_done(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("AutoRec"):
                seen.add((row["method"], row["p"]))
    expected = {(f"AutoRec ({rule})", str(p)) for rule in ("rand", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, hidden_grid, lr_grid, wd_grid, max_epochs, patience) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    n_entries = split.n_users * split.n_items
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items = {n_entries:,} dense entries | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    if n_entries > SIZE_LIMIT_ENTRIES:
        print(f"{dataset}: SKIPPED - a dense {n_entries:,}-entry matrix is over the "
              f"{SIZE_LIMIT_ENTRIES:,}-entry safety limit (same reasoning as run_softimpute.py).")
        return []

    train_matrix, train_mask = build_dense_items(split.train, split.n_users, split.n_items, DEVICE)
    val_matrix, val_mask = build_dense_items(split.val, split.n_users, split.n_items, DEVICE)

    configs = [{"hidden": h, "lr": lr, "weight_decay": wd} for h in hidden_grid for lr in lr_grid for wd in wd_grid]
    print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, patience={patience}")

    heartbeat_every = max(1, max_epochs // 10)
    best_cfg, best_val_rmse = None, float("inf")
    pbar = tqdm(configs, desc=f"{dataset} grid search")
    for n, cfg in enumerate(pbar, 1):
        t0 = time.time()
        _, v_rmse, epochs_run = train_autorec(
            train_matrix, train_mask, val_matrix, val_mask,
            max_epochs=max_epochs, patience=patience, seed=SEED,
            heartbeat_every=heartbeat_every, **cfg,
        )
        if v_rmse < best_val_rmse:
            best_val_rmse, best_cfg = v_rmse, cfg
        pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
        pbar.write(f"  [{n}/{len(configs)}] hidden={cfg['hidden']:>3} lr={cfg['lr']} wd={cfg['weight_decay']:.0e} "
                   f"-> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    trainval_rows = np.concatenate([split.train, split.val], axis=0)
    trainval_matrix, trainval_mask = build_dense_items(trainval_rows, split.n_users, split.n_items, DEVICE)
    test_matrix, test_mask = build_dense_items(split.test, split.n_users, split.n_items, DEVICE)

    t0 = time.time()
    final_model, _, epochs_run = train_autorec(
        trainval_matrix, trainval_mask, test_matrix, test_mask,
        max_epochs=max_epochs, patience=patience, seed=SEED,
        desc=f"{dataset} refit", show_progress=True, heartbeat_every=heartbeat_every, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    test_u = split.test[:, 0].astype(int)
    test_i = split.test[:, 1].astype(int)
    with torch.no_grad():
        recon = final_model(trainval_matrix)
        pred = recon[test_i, test_u].cpu().numpy()
    actual = split.test[:, 2]
    trainval_user_idx = trainval_rows[:, 0].astype(int)

    rng = np.random.default_rng(SEED)
    random_score = rng.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_u.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_u].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("supp", support_score)]:
        for p, val in sweep(pred, actual, score).items():
            rows_out.append({
                "method": f"AutoRec ({rule_name})",
                "dataset": dataset,
                "p": p,
                "selective_rmse": round(val, 4),
                "best_hidden": best_cfg["hidden"],
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
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "autorec_results.csv")
    args = parser.parse_args()

    hidden_grid, lr_grid, wd_grid = (
        (QUICK_HIDDEN_GRID, QUICK_LR_GRID, QUICK_WD_GRID) if args.quick else (HIDDEN_GRID, LR_GRID, WD_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE

    print(f"device: {DEVICE}")
    print(f"results file: {args.out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, hidden_grid, lr_grid, wd_grid, max_epochs, patience)
        if rows:
            append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
