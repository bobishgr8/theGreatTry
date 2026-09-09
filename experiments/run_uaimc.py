"""Standalone, overnight-runnable script for UA-IMC (Kasalicky, Ledent &
Alves, RecSys'23 [1]): grid search + refit + abstention scoring. Produces
Table 1's "UA-IMC (U_hat)" row.

  python run_uaimc.py                     # every dataset except ml-25m
  python run_uaimc.py --datasets ml-100k
  python run_uaimc.py --quick             # tiny grid/slice, smoke-test the pipeline

**This is a deliberate approximation, not a faithful reimplementation of
[1].** The real UA-IMC couples an inductive graph encoder with its own
specific uncertainty parameterisation and loss, detailed across that paper's
full method section - reproducing it exactly is out of scope here. What this
script actually does is the smallest change that turns week2/IGMC.ipynb's
graph encoder into an uncertainty-aware one in the *same spirit* as UA-IMC
(a GNN-based inductive method that outputs both a rating and a confidence),
using this repo's own already-derived joint loss for that (proposal eq. 5,
i.e. exactly run_ours.py's loss/selection logic, but with run_igmc.py's
R-GCN subgraph encoder standing in for Ours' plain bilinear embeddings, and
two MLP heads off the same backbone in place of Ours' separate C,D
matrices). Where IGMC's node_embed + R-GCN layers + subgraph extraction are
identical to run_igmc.py, they're imported from there rather than
re-derived - see that module's docstring for what's simplified about the
graph construction itself (1-hop subgraphs, DRNL substitute, neighbour
sampling). Treat this row in Table 1 as "our joint-uncertainty loss grafted
onto an inductive GNN backbone", not as a validated reproduction of [1]'s
numbers.

[1] Uncertainty-adjusted Inductive Matrix Completion with GNNs, RecSys '23.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import random
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
sys.path.append(str(Path(__file__).resolve().parent))

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES
from run_igmc import (
    DEVICE, N_THREADS, N_RATING_CLASSES,
    build_adjacency, precompute_subgraphs, collate, RGCNLayer,
)

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, same reasoning as run_igmc.py
SEED = 42
torch.set_num_threads(N_THREADS)

# Same grid shape as run_igmc.py (each config here costs about as much as an
# IGMC config - same backbone, twice the output heads).
HIDDEN_GRID = [16, 32]
LAYER_GRID = [2, 3]
LR_GRID = [0.003]
LAM_GRID = [1e-4, 1e-3]  # replaces IGMC's weight_decay axis, see joint_loss below
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 512

QUICK_HIDDEN_GRID = [16]
QUICK_LAYER_GRID = [2]
QUICK_LR_GRID = [0.01]
QUICK_LAM_GRID = [1e-4]
QUICK_MAX_EPOCHS = 2
QUICK_PATIENCE = 2
QUICK_ROW_LIMIT = 800

FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_layers", "best_lr", "best_lam", "val_rmse", "seconds",
]


class UAIMC(nn.Module):
    """Same R-GCN backbone as run_igmc.py's IGMC, but with two MLP heads off
    the concatenated target representation - a rating head (S) and an
    uncertainty head (U), the GNN analogue of Ours' separate A,B / C,D
    matrices (proposal eq. 4)."""

    def __init__(self, hidden_dim=32, n_layers=3, n_relations=N_RATING_CLASSES, n_node_roles=4, dropout=0.2):
        super().__init__()
        self.node_embed = nn.Embedding(n_node_roles, hidden_dim)
        self.layers = nn.ModuleList(
            [RGCNLayer(hidden_dim, hidden_dim, n_relations) for _ in range(n_layers)]
        )
        concat_dim = hidden_dim * n_layers * 2

        def head():
            return nn.Sequential(
                nn.Linear(concat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.rating_head = head()
        self.uncertainty_head = head()

    def forward(self, batch):
        h = self.node_embed(batch["node_role"])
        layer_outs = []
        for layer in self.layers:
            h = layer(h, batch["edge_src"], batch["edge_dst"], batch["edge_rel"], batch["n_nodes"])
            layer_outs.append(h)
        h_all = torch.cat(layer_outs, dim=-1)
        u_repr = h_all[batch["target_u"]]
        i_repr = h_all[batch["target_i"]]
        graph_repr = torch.cat([u_repr, i_repr], dim=-1)
        s = self.rating_head(graph_repr).squeeze(-1)
        u_hat = self.uncertainty_head(graph_repr).squeeze(-1)
        return s, u_hat


def joint_loss(s, u_hat, actual, lam, full_n):
    """Proposal eq. (5), identical to run_ours.py's joint_loss - see there
    for the derivation of the minibatch-rescaled regulariser term."""
    mse_term = torch.mean((s - actual) ** 2 * torch.exp(-u_hat))
    reg_term = lam * u_hat.sum() * (full_n / len(u_hat))
    return mse_term + reg_term


def predict_uaimc(model, graphs, batch_size, device):
    model.eval()
    preds, uhats = [], []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = collate(graphs[start : start + batch_size], device)
            s, u_hat = model(batch)
            preds.append(s.cpu().numpy())
            uhats.append(u_hat.cpu().numpy())
    return np.concatenate(preds), np.concatenate(uhats)


def evaluate_rmse(model, graphs, batch_size, device):
    """Selection metric is rating-head RMSE alone (U_hat unscored during
    selection) - same fairness reasoning as run_ours.py's train_ours."""
    pred, _ = predict_uaimc(model, graphs, batch_size, device)
    actual = np.array([g[2] for g in graphs])
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def train_uaimc(
    train_graphs, val_graphs, hidden_dim, n_layers, lr, lam, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    torch.manual_seed(seed)
    model = UAIMC(hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    shuffle_rng = random.Random(seed)
    idx_all = list(range(len(train_graphs)))
    full_n = len(train_graphs)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"h={hidden_dim} L={n_layers} lr={lr} lam={lam:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        shuffle_rng.shuffle(idx_all)
        model.train()
        for start in range(0, len(idx_all), batch_size):
            batch_idx = idx_all[start : start + batch_size]
            batch = collate([train_graphs[k] for k in batch_idx], DEVICE)
            opt.zero_grad()
            s, u_hat = model(batch)
            loss = joint_loss(s, u_hat, batch["rating"], lam, full_n)
            loss.backward()
            opt.step()

        v_rmse = evaluate_rmse(model, val_graphs, batch_size, DEVICE)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"h={hidden_dim} L={n_layers} lr={lr} lam={lam:.0e}"
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
            if row["dataset"] == dataset and row["method"] == "UA-IMC (U_hat)":
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


def run_dataset(dataset: str, hidden_grid, layer_grid, lr_grid, lam_grid, max_epochs, patience, batch_size,
                 row_limit=None) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    train_rows, val_rows, test_rows = split.train, split.val, split.test
    if row_limit:
        train_rows, val_rows, test_rows = train_rows[:row_limit], val_rows[:row_limit], test_rows[:row_limit]

    user_items, item_users = build_adjacency(train_rows)
    rng = random.Random(SEED)
    sel_train_graphs = precompute_subgraphs(train_rows, user_items, item_users, rng, f"{dataset} train subgraphs")
    sel_val_graphs = precompute_subgraphs(val_rows, user_items, item_users, rng, f"{dataset} val subgraphs")
    print(f"{dataset}: train={len(sel_train_graphs):,} val={len(sel_val_graphs):,} subgraphs cached (model selection)")

    configs = [
        {"hidden_dim": h, "n_layers": l, "lr": lr, "lam": lam}
        for h in hidden_grid for l in layer_grid for lr in lr_grid for lam in lam_grid
    ]
    print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, patience={patience}, batch_size={batch_size}")

    heartbeat_every = max(1, max_epochs // 10)
    best_cfg, best_val_rmse = None, float("inf")
    pbar = tqdm(configs, desc=f"{dataset} grid search")
    for n, cfg in enumerate(pbar, 1):
        t0 = time.time()
        _, v_rmse, epochs_run = train_uaimc(
            sel_train_graphs, sel_val_graphs, max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, seed=SEED, heartbeat_every=heartbeat_every, **cfg,
        )
        if v_rmse < best_val_rmse:
            best_val_rmse, best_cfg = v_rmse, cfg
        pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
        pbar.write(f"  [{n}/{len(configs)}] hidden={cfg['hidden_dim']:>3} layers={cfg['n_layers']} "
                   f"lr={cfg['lr']} lam={cfg['lam']:.0e} -> val_rmse={v_rmse:.4f} "
                   f"({epochs_run} epochs, {time.time() - t0:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    # See run_igmc.py's identical fix - free model-selection subgraphs before
    # phase 2 builds a second full set, so both aren't alive simultaneously.
    del sel_train_graphs, sel_val_graphs, user_items, item_users
    gc.collect()

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    user_items, item_users = build_adjacency(trainval_rows)
    rng2 = random.Random(SEED)
    trainval_graphs = precompute_subgraphs(trainval_rows, user_items, item_users, rng2, f"{dataset} trainval subgraphs")
    final_test_graphs = precompute_subgraphs(test_rows, user_items, item_users, rng2, f"{dataset} test subgraphs")

    t0 = time.time()
    final_model, _, epochs_run = train_uaimc(
        trainval_graphs, final_test_graphs, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, seed=SEED, desc=f"{dataset} refit", show_progress=True,
        heartbeat_every=heartbeat_every, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    actual = np.array([g[2] for g in final_test_graphs])
    pred, u_hat = predict_uaimc(final_model, final_test_graphs, batch_size, DEVICE)

    rows_out = []
    for p, val in sweep(pred, actual, u_hat).items():
        rows_out.append({
            "method": "UA-IMC (U_hat)",
            "dataset": dataset,
            "p": p,
            "selective_rmse": round(val, 4),
            "best_hidden": best_cfg["hidden_dim"],
            "best_layers": best_cfg["n_layers"],
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
    parser.add_argument("--quick", action="store_true", help="tiny grid/slice, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "uaimc_results.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    hidden_grid, layer_grid, lr_grid, lam_grid = (
        (QUICK_HIDDEN_GRID, QUICK_LAYER_GRID, QUICK_LR_GRID, QUICK_LAM_GRID)
        if args.quick else (HIDDEN_GRID, LAYER_GRID, LR_GRID, LAM_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    row_limit = QUICK_ROW_LIMIT if args.quick else None

    print(f"device: {DEVICE} | CPU threads: {N_THREADS}")
    print(f"results file: {args.out}")
    print("NOTE: this is an approximation of UA-IMC, not a faithful reimplementation - see module docstring.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, hidden_grid, layer_grid, lr_grid, lam_grid, max_epochs, patience,
                            args.batch_size, row_limit=row_limit)
        append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
