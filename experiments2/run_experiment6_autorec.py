"""I-AutoRec (Sedhain et al. 2015) baseline, moved into experiments2's
conventions from experiments/run_autorec.py (same model/training code,
relocated and switched to the grid+best CSV pair every other experiment*.py
here uses, for the same crash-resumability reason as experiment3/4/5/8).

  python run_experiment6_autorec.py                    # every dataset small enough (see below)
  python run_experiment6_autorec.py --datasets ml-100k
  python run_experiment6_autorec.py --quick             # tiny grid, smoke-test the pipeline

Item-based AutoRec: each item is one training example, a length-n_users
vector of that item's ratings (0 where unobserved), autoencoded through one
sigmoid hidden layer and a linear (identity) output layer, with the
reconstruction loss masked to only the observed entries. This needs the
rating matrix dense at least once, so it inherits SIZE_LIMIT_ENTRIES from
run_softimpute.py's reasoning (ml-25m, amazon-games are skipped automatically
if too large). Trains full-batch (one "batch" = every item), not minibatched.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent

import mf_common
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games"]
SEED = mf_common.SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE_LIMIT_ENTRIES = 100_000_000  # same cutoff/reasoning as run_experiment7_softimpute.py

METHOD = "AutoRec"
# Expanded from the original {50,100,200}x{0.003,0.01}x{1e-4,1e-3,1e-2}
# (18 configs) after that grid's winner landed on hidden=200 (its max),
# lr=0.01 (its max), and wd=1e-4 (its min) on EVERY dataset - a clean
# edge-hugging signal in all three directions at once, so this doesn't just
# add resolution, it extends past every edge the old grid hit. AutoRec is
# cheap (worst-case config tonight was ~21s), so a full brute grid stays
# affordable even at 5x3x5=75 configs/dataset - no need for a targeted
# search like IGMC/UAIMC get.
HIDDEN_GRID = [50, 100, 200, 400, 800]
LR_GRID = [0.003, 0.01, 0.03]
WD_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
MAX_EPOCHS = 200
PATIENCE = 10

QUICK_HIDDEN_GRID = [20]
QUICK_LR_GRID = [0.01]
QUICK_WD_GRID = [1e-3]
QUICK_MAX_EPOCHS = 3
QUICK_PATIENCE = 3

GRID_FIELDNAMES = ["method", "dataset", "hidden", "lr", "weight_decay", "val_rmse", "seconds"]
BEST_FIELDNAMES = [
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
    n_users = train_matrix.shape[1]
    model = AutoRec(n_users, hidden, seed).to(train_matrix.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    from tqdm import tqdm
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


def run_dataset(dataset: str, hidden_grid, lr_grid, wd_grid, max_epochs, patience, grid_out: Path) -> list[dict]:
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

    train_matrix, train_mask = build_dense_items(train_rows, n_users, n_items, DEVICE)
    val_matrix, val_mask = build_dense_items(val_rows, n_users, n_items, DEVICE)

    configs = [{"hidden": h, "lr": lr, "weight_decay": wd} for h in hidden_grid for lr in lr_grid for wd in wd_grid]
    existing = mf_common.load_existing_grid(grid_out, METHOD, dataset, [("hidden", int), ("lr", float), ("weight_decay", float)])
    todo = [cfg for cfg in configs if (cfg["hidden"], cfg["lr"], cfg["weight_decay"]) not in existing]
    print(f"{dataset}: {len(configs)} configs ({len(configs) - len(todo)} already done), "
          f"max_epochs={max_epochs}, patience={patience}")

    best_cfg, best_val_rmse = None, float("inf")
    for key, v in existing.items():
        if v < best_val_rmse:
            best_val_rmse, best_cfg = v, dict(zip(["hidden", "lr", "weight_decay"], key))

    heartbeat_every = max(1, max_epochs // 10)
    for n, cfg in enumerate(todo, 1):
        t0 = time.time()
        _, v_rmse, epochs_run = train_autorec(
            train_matrix, train_mask, val_matrix, val_mask,
            max_epochs=max_epochs, patience=patience, seed=SEED, heartbeat_every=heartbeat_every, **cfg,
        )
        secs = time.time() - t0
        mf_common.append_results(grid_out, [{
            "method": METHOD, "dataset": dataset, "hidden": cfg["hidden"], "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"], "val_rmse": round(v_rmse, 4), "seconds": round(secs, 1),
        }], GRID_FIELDNAMES)
        if v_rmse < best_val_rmse:
            best_val_rmse, best_cfg = v_rmse, cfg
        print(f"  [{n}/{len(todo)}] hidden={cfg['hidden']:>3} lr={cfg['lr']} wd={cfg['weight_decay']:.0e} "
              f"-> val_rmse={v_rmse:.4f} best={best_val_rmse:.4f} ({epochs_run} epochs, {secs:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    trainval_matrix, trainval_mask = build_dense_items(trainval_rows, n_users, n_items, DEVICE)
    test_matrix, test_mask = build_dense_items(test_rows, n_users, n_items, DEVICE)

    t0 = time.time()
    final_model, _, epochs_run = train_autorec(
        trainval_matrix, trainval_mask, test_matrix, test_mask,
        max_epochs=max_epochs, patience=patience, seed=SEED,
        desc=f"{dataset} refit", show_progress=True, heartbeat_every=heartbeat_every, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    test_u = test_rows[:, 0].astype(int)
    test_i = test_rows[:, 1].astype(int)
    with torch.no_grad():
        recon = final_model(trainval_matrix)
        pred = recon[test_i, test_u].cpu().numpy()
    actual = test_rows[:, 2]
    trainval_user_idx = trainval_rows[:, 0].astype(int)

    rng = np.random.default_rng(SEED)
    random_score = rng.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_u.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_u].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("val", -pred), ("supp", support_score)]:
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


def already_done_best(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    import csv
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("AutoRec"):
                seen.add((row["method"], row["p"]))
    expected = {(f"AutoRec ({rule})", str(p)) for rule in ("rand", "val", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=[d for d in DATASETS if d != "ml-25m"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "experiment6_autorec_grid.csv")
    parser.add_argument("--best-out", type=Path, default=HERE / "results" / "experiment6_autorec_best.csv")
    args = parser.parse_args()

    hidden_grid, lr_grid, wd_grid = (
        (QUICK_HIDDEN_GRID, QUICK_LR_GRID, QUICK_WD_GRID) if args.quick else (HIDDEN_GRID, LR_GRID, WD_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE

    print(f"device: {DEVICE}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done_best(args.best_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, hidden_grid, lr_grid, wd_grid, max_epochs, patience, args.out)
        if rows:
            mf_common.append_results(args.best_out, rows, BEST_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.best_out}")


if __name__ == "__main__":
    main()
