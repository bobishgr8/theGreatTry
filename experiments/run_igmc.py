"""Standalone, overnight-runnable script for IGMC (Zhang & Chen 2020): grid
search + refit + abstention scoring. Produces Table 1's "IGMC (rand)" /
"IGMC (supp)" rows.

Promotes week2/IGMC.ipynb's 1-hop-enclosing-subgraph R-GCN model to the same
resumable, early-stopping pattern as run_plain_mf.py:

  python run_igmc.py                     # every dataset except ml-25m
  python run_igmc.py --datasets ml-100k
  python run_igmc.py --quick             # tiny grid/slice, smoke-test the pipeline

See week2/IGMC.ipynb's markdown cells for the model itself (1-hop enclosing
subgraphs, 4-way structural node roles as an exact substitute for DRNL at
h=1, R-GCN message passing, target-edge deletion) - that reasoning is not
repeated here. What changes here vs. the notebook:

  - early stopping (patience-based) instead of a fixed epoch count, so a
    slow dataset doesn't need its own hand-tuned EPOCHS;
  - results append to --out per dataset and a finished dataset is skipped on
    restart, like every other run_*.py script;
  - GPU-first (DEVICE), matching the rest of the repo.

Subgraph extraction is still plain-Python dict lookups (see the notebook's
"Notes" section) - the real bottleneck, not the R-GCN forward/backward pass
itself. ml-25m is excluded from DATASETS for this reason (162K users, 9.6B
possible cells - the same class of problem that makes SoftImpute/AutoRec
exclude it, compounded by needing that dict-lookup extraction 60M+ times).
douban/amazon-games are included but may be genuinely slow overnight; that's
expected, not a bug - this is exactly the kind of run this pattern exists
for (kick off, leave running, --force a re-run of just the slow one later if
needed).
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import os
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

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, see module docstring
SEED = 42
N_RATING_CLASSES = 5  # proposal Sec. 3: ratings in {1,...,5}
MAX_NEIGHBORS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_THREADS = os.cpu_count() or 1
torch.set_num_threads(N_THREADS)

# Intentionally much smaller than Plain MF/Ours' 18-config grid: each config
# here costs orders of magnitude more (a full R-GCN forward/backward per
# minibatch of subgraphs, not a plain embedding lookup) - see week2/IGMC.ipynb.
HIDDEN_GRID = [16, 32]
LAYER_GRID = [2, 3]
LR_GRID = [0.003]
WD_GRID = [0.0, 1e-4]
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 512

QUICK_HIDDEN_GRID = [16]
QUICK_LAYER_GRID = [2]
QUICK_LR_GRID = [0.01]
QUICK_WD_GRID = [1e-4]
QUICK_MAX_EPOCHS = 2
QUICK_PATIENCE = 2
QUICK_ROW_LIMIT = 800  # matches the notebook's FAST_SMOKE_TEST slice sizes

FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_layers", "best_lr", "best_weight_decay", "val_rmse", "seconds",
]


def build_adjacency(rows):
    user_items: dict[int, dict[int, float]] = {}
    item_users: dict[int, dict[int, float]] = {}
    for u, i, r in rows:
        u, i = int(u), int(i)
        user_items.setdefault(u, {})[i] = float(r)
        item_users.setdefault(i, {})[u] = float(r)
    return user_items, item_users


def _relation(rating: float) -> int:
    return int(min(max(round(rating), 1), N_RATING_CLASSES)) - 1


def _sample(neighbors: list[int], rng: random.Random) -> list[int]:
    if len(neighbors) <= MAX_NEIGHBORS:
        return neighbors
    return rng.sample(neighbors, MAX_NEIGHBORS)


def extract_subgraph(u: int, i: int, user_items, item_users, rng: random.Random):
    """Returns (node_role, edges) for the 1-hop enclosing subgraph of (u, i):
    node_role a uint8 array (values in {0,1,2,3} = {target_user, target_item,
    other_user, other_item}, local index 0 always u, local index 1 always i),
    edges an (E, 3) int32 array of (src_local, dst_local, relation) rows.

    Built as plain Python lists first (append is cheapest that way) and
    converted to compact numpy arrays only once, at the end - the arrays,
    not the transient build-lists, are what's actually held for the whole
    dataset at once by precompute_subgraphs below. That distinction is the
    difference between ML-1M's subgraph cache fitting in memory and not: a
    Python list of (int,int,int) tuples costs roughly 10x more per edge than
    an int32 array (tuple + 3 boxed-int object overhead vs. 12 packed
    bytes), and with ~700K examples x dozens of edges each, that 10x was
    enough to OOM-kill the host outright (see run_igmc.py's module docstring
    changelog / the run that motivated this)."""
    other_items = _sample([j for j in user_items.get(u, {}) if j != i], rng)
    other_users = _sample([a for a in item_users.get(i, {}) if a != u], rng)

    other_user_base = 2
    other_item_base = other_user_base + len(other_users)
    node_role = [0, 1] + [2] * len(other_users) + [3] * len(other_items)

    edges: list[tuple[int, int, int]] = []

    def add(a_local, b_local, rating):
        rel = _relation(rating)
        edges.append((a_local, b_local, rel))
        edges.append((b_local, a_local, rel))

    for k, a in enumerate(other_users):
        add(other_user_base + k, 1, item_users[i][a])
    for k, j in enumerate(other_items):
        add(0, other_item_base + k, user_items[u][j])
    for ku, a in enumerate(other_users):
        a_items = user_items.get(a, {})
        for kj, j in enumerate(other_items):
            r = a_items.get(j)
            if r is not None:
                add(other_user_base + ku, other_item_base + kj, r)

    node_role_arr = np.array(node_role, dtype=np.uint8)
    edges_arr = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 3), dtype=np.int32)
    return node_role_arr, edges_arr


def precompute_subgraphs(rows, user_items, item_users, rng, desc):
    out = []
    for u, i, r in tqdm(rows, desc=desc, leave=False):
        node_role, edges = extract_subgraph(int(u), int(i), user_items, item_users, rng)
        out.append((node_role, edges, float(r)))
    return out


def collate(graphs_batch, device):
    """node_role/edges arrive as the compact numpy arrays extract_subgraph
    produces; batching concatenates them (edge indices shifted by each
    graph's node offset) rather than looping in Python per edge."""
    node_role_parts = []
    edge_parts = []
    target_u, target_i = [], []
    ratings = []
    offset = 0
    for node_role, edges, rating in graphs_batch:
        node_role_parts.append(node_role)
        if len(edges):
            shifted = edges.copy()
            shifted[:, 0] += offset
            shifted[:, 1] += offset
            edge_parts.append(shifted)
        target_u.append(offset + 0)
        target_i.append(offset + 1)
        ratings.append(rating)
        offset += len(node_role)

    all_edges = np.concatenate(edge_parts, axis=0) if edge_parts else np.zeros((0, 3), dtype=np.int32)

    return {
        "node_role": torch.as_tensor(np.concatenate(node_role_parts), dtype=torch.long, device=device),
        "edge_src": torch.as_tensor(all_edges[:, 0], dtype=torch.long, device=device),
        "edge_dst": torch.as_tensor(all_edges[:, 1], dtype=torch.long, device=device),
        "edge_rel": torch.as_tensor(all_edges[:, 2], dtype=torch.long, device=device),
        "target_u": torch.tensor(target_u, dtype=torch.long, device=device),
        "target_i": torch.tensor(target_i, dtype=torch.long, device=device),
        "rating": torch.tensor(ratings, dtype=torch.float32, device=device),
        "n_nodes": offset,
    }


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_relations):
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        self.rel_weight = nn.Parameter(torch.randn(n_relations, in_dim, out_dim) * (1.0 / in_dim**0.5))
        self.n_relations = n_relations

    def forward(self, h, edge_src, edge_dst, edge_rel, n_nodes):
        out = self.self_loop(h)
        if edge_src.numel() > 0:
            for r in range(self.n_relations):
                mask = edge_rel == r
                if not mask.any():
                    continue
                src_r = edge_src[mask]
                dst_r = edge_dst[mask]
                deg_r = torch.zeros(n_nodes, device=h.device)
                deg_r.index_add_(0, dst_r, torch.ones_like(dst_r, dtype=h.dtype))
                deg_r.clamp_(min=1.0)
                msg = h[src_r] @ self.rel_weight[r]
                msg = msg / deg_r[dst_r].unsqueeze(1)
                out = out.index_add(0, dst_r, msg)
        return torch.relu(out)


class IGMC(nn.Module):
    def __init__(self, hidden_dim=32, n_layers=3, n_relations=N_RATING_CLASSES, n_node_roles=4, dropout=0.2):
        super().__init__()
        self.node_embed = nn.Embedding(n_node_roles, hidden_dim)
        self.layers = nn.ModuleList(
            [RGCNLayer(hidden_dim, hidden_dim, n_relations) for _ in range(n_layers)]
        )
        concat_dim = hidden_dim * n_layers * 2
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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
        return self.mlp(graph_repr).squeeze(-1)


def predict_igmc(model, graphs, batch_size, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = collate(graphs[start : start + batch_size], device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def evaluate_rmse(model, graphs, batch_size, device):
    pred = predict_igmc(model, graphs, batch_size, device)
    actual = np.array([g[2] for g in graphs])
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def train_igmc(
    train_graphs, val_graphs, hidden_dim, n_layers, lr, weight_decay, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    torch.manual_seed(seed)
    model = IGMC(hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffle_rng = random.Random(seed)
    idx_all = list(range(len(train_graphs)))

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"h={hidden_dim} L={n_layers} lr={lr} wd={weight_decay}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        shuffle_rng.shuffle(idx_all)
        model.train()
        for start in range(0, len(idx_all), batch_size):
            batch_idx = idx_all[start : start + batch_size]
            batch = collate([train_graphs[k] for k in batch_idx], DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = torch.mean((pred - batch["rating"]) ** 2)
            loss.backward()
            opt.step()

        v_rmse = evaluate_rmse(model, val_graphs, batch_size, DEVICE)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"h={hidden_dim} L={n_layers} lr={lr} wd={weight_decay}"
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
            if row["dataset"] == dataset and row["method"].startswith("IGMC"):
                seen.add((row["method"], row["p"]))
    expected = {(f"IGMC ({rule})", str(p)) for rule in ("rand", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, hidden_grid, layer_grid, lr_grid, wd_grid, max_epochs, patience, batch_size,
                 row_limit=None) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = load_split(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    train_rows, val_rows, test_rows = split.train, split.val, split.test
    if row_limit:
        train_rows, val_rows, test_rows = train_rows[:row_limit], val_rows[:row_limit], test_rows[:row_limit]

    user_items, item_users = build_adjacency(train_rows)  # phase 1: model-selection adjacency (train only)
    rng = random.Random(SEED)
    sel_train_graphs = precompute_subgraphs(train_rows, user_items, item_users, rng, f"{dataset} train subgraphs")
    sel_val_graphs = precompute_subgraphs(val_rows, user_items, item_users, rng, f"{dataset} val subgraphs")
    print(f"{dataset}: train={len(sel_train_graphs):,} val={len(sel_val_graphs):,} subgraphs cached (model selection)")

    configs = [
        {"hidden_dim": h, "n_layers": l, "lr": lr, "weight_decay": wd}
        for h in hidden_grid for l in layer_grid for lr in lr_grid for wd in wd_grid
    ]
    print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, patience={patience}, batch_size={batch_size}")

    heartbeat_every = max(1, max_epochs // 10)
    best_cfg, best_val_rmse = None, float("inf")
    pbar = tqdm(configs, desc=f"{dataset} grid search")
    for n, cfg in enumerate(pbar, 1):
        t0 = time.time()
        _, v_rmse, epochs_run = train_igmc(
            sel_train_graphs, sel_val_graphs, max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, seed=SEED, heartbeat_every=heartbeat_every, **cfg,
        )
        if v_rmse < best_val_rmse:
            best_val_rmse, best_cfg = v_rmse, cfg
        pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
        pbar.write(f"  [{n}/{len(configs)}] hidden={cfg['hidden_dim']:>3} layers={cfg['n_layers']} "
                   f"lr={cfg['lr']} wd={cfg['weight_decay']:.0e} -> val_rmse={v_rmse:.4f} "
                   f"({epochs_run} epochs, {time.time() - t0:.1f}s)")

    print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")

    # Model-selection subgraphs are never touched again - free them before
    # phase 2 builds a second full set (trainval+test), otherwise both sets
    # are alive simultaneously right when memory pressure is highest (this is
    # what OOM-killed ml-1m: peak usage is ~2x what precomputing train alone
    # already showed was substantial, on a 16GB host shared with a desktop
    # session - see run_plain_mf.py's DEFAULT_WORKERS comment for the same
    # machine constraint).
    del sel_train_graphs, sel_val_graphs, user_items, item_users
    gc.collect()

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    user_items, item_users = build_adjacency(trainval_rows)  # phase 2: refit adjacency (train+val)
    rng2 = random.Random(SEED)
    trainval_graphs = precompute_subgraphs(trainval_rows, user_items, item_users, rng2, f"{dataset} trainval subgraphs")
    final_test_graphs = precompute_subgraphs(test_rows, user_items, item_users, rng2, f"{dataset} test subgraphs")

    t0 = time.time()
    final_model, _, epochs_run = train_igmc(
        trainval_graphs, final_test_graphs, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, seed=SEED, desc=f"{dataset} refit", show_progress=True,
        heartbeat_every=heartbeat_every, **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    actual = np.array([g[2] for g in final_test_graphs])
    test_user_idx = test_rows[:, 0].astype(int)
    trainval_user_idx = trainval_rows[:, 0].astype(int)
    pred = predict_igmc(final_model, final_test_graphs, batch_size, DEVICE)

    rng_score = np.random.default_rng(SEED)
    random_score = rng_score.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_user_idx.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_user_idx].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("supp", support_score)]:
        for p, val in sweep(pred, actual, score).items():
            rows_out.append({
                "method": f"IGMC ({rule_name})",
                "dataset": dataset,
                "p": p,
                "selective_rmse": round(val, 4),
                "best_hidden": best_cfg["hidden_dim"],
                "best_layers": best_cfg["n_layers"],
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
    parser.add_argument("--quick", action="store_true", help="tiny grid/slice, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "igmc_results.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    hidden_grid, layer_grid, lr_grid, wd_grid = (
        (QUICK_HIDDEN_GRID, QUICK_LAYER_GRID, QUICK_LR_GRID, QUICK_WD_GRID)
        if args.quick else (HIDDEN_GRID, LAYER_GRID, LR_GRID, WD_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    row_limit = QUICK_ROW_LIMIT if args.quick else None

    print(f"device: {DEVICE} | CPU threads: {N_THREADS}")
    print(f"results file: {args.out}")
    print("NOTE: subgraph extraction is pure-Python dict lookups (see module docstring) - "
          "douban/amazon-games may take a long while even though the R-GCN itself is small.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, hidden_grid, layer_grid, lr_grid, wd_grid, max_epochs, patience,
                            args.batch_size, row_limit=row_limit)
        append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
