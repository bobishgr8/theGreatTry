"""Shared GNN infrastructure for experiments2's IGMC-family methods (IGMC
itself, and UA-IMC's uncertainty backbone which reuses IGMC's architecture -
paper Sec. 3 explicitly says so). Ported from experiments/run_igmc.py, with
one structural change: subgraphs are cached on DISK (memory-mapped), not as
a Python list held fully in host RAM.

=====================================================================
WHY: the actual root cause of the overnight OOM crashes
=====================================================================
experiments/run_uaimc_v2.py's precompute_subgraphs built every subgraph for
a whole dataset (up to ~900K on ml-1m) as a Python list of (node_role, edges,
rating) tuples and kept the WHOLE list alive in process RAM for the entire
grid search / refit phase it belonged to. Each tuple's arrays are already
compact (see extract_subgraph's docstring below), but "compact per-example"
times "every example, simultaneously, for hours" is still a multi-GB
resident set on top of everything else running on the same 16GB host - and
is exactly what the MAX_NEIGHBORS_OVERRIDE hack in that file fought by
shrinking subgraphs, at a real cost to model quality (fewer neighbours per
graph = less signal for the R-GCN).

DiskSubgraphCache below fixes the actual bottleneck instead of shrinking
around it: subgraphs are extracted once and written straight to two flat
binary files (node roles, edges) plus a small in-RAM offset index (a few
ints per example - e.g. 900K x 4 int64 is ~29MB, nothing). Training then
reads batches back via np.memmap, which is OS-paged: only the pages actually
touched by the current minibatch are resident, and the OS can evict them
under memory pressure since they're backed by the file, not swap. This is
the "smarter architecture, not a smaller model" fix - it does NOT change
what the R-GCN sees per subgraph, so MAX_NEIGHBORS can go back up toward the
paper's default instead of staying artificially small (see run_experiment5
/8's MAX_NEIGHBORS choices).

Trade-off: memmap reads under random-shuffle batch order have less
locality than the old sequential list, so this is somewhat slower per-epoch
on a spinning disk; on an SSD (the common case) the difference is minor and
worth it for not OOM-killing multi-hour runs.
"""

from __future__ import annotations

import functools
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))
from recsys_loader import load_split  # noqa: E402

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, see run_experiment1_plain.py
SEED = 42
N_RATING_CLASSES = 5  # proposal Sec. 3: ratings in {1,...,5}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_THREADS = os.cpu_count() or 1
torch.set_num_threads(N_THREADS)

# Module-level, mutated per-dataset by callers (same convention
# experiments/run_uaimc_v2.py used against run_igmc.MAX_NEIGHBORS) - now a
# throughput/disk-size knob rather than a hard RAM-survival requirement,
# since DiskSubgraphCache removes the "must all fit in RAM at once" ceiling.
MAX_NEIGHBORS = 20


# ---------------------------------------------------------------------------
# Subgraph extraction (unchanged logic from experiments/run_igmc.py)
# ---------------------------------------------------------------------------

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
    """Returns (node_role, edges) for the 1-hop enclosing subgraph of (u, i) -
    see experiments/run_igmc.py's identical function for the full derivation;
    logic here is byte-for-byte the same, only relocated."""
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


# ---------------------------------------------------------------------------
# Disk-backed subgraph cache (the memory fix - see module docstring)
# ---------------------------------------------------------------------------

class DiskSubgraphCache:
    """Read-side handle: two memmapped flat arrays (node roles, edges) plus a
    small in-RAM offset index and per-example ratings. get_batch(indices)
    returns the same list-of-(node_role, edges, rating)-tuples shape the old
    in-RAM list gave collate(), so it's a drop-in replacement at every call
    site - only how the data is STORED changed, not the shape callers see."""

    def __init__(self, node_roles, edges, index, ratings, cache_dir: Path | None):
        self.node_roles = node_roles
        self.edges = edges
        self.index = index  # (n, 4) int64: node_start, node_len, edge_start, edge_len
        self.ratings = ratings
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.index)

    def get(self, k: int):
        ns, nl, es, el = self.index[k]
        node_role = np.asarray(self.node_roles[ns:ns + nl])
        edges = np.asarray(self.edges[es:es + el]) if el else np.zeros((0, 3), dtype=np.int32)
        return node_role, edges, float(self.ratings[k])

    def get_batch(self, indices):
        return [self.get(int(k)) for k in indices]

    def all_ratings(self) -> np.ndarray:
        return np.asarray(self.ratings)

    def close(self):
        """Drop the memmap references so the backing files can be deleted
        (Windows holds an open-file lock on a mapped file; del + gc.collect
        releases it - see cleanup())."""
        self.node_roles = None
        self.edges = None

    def cleanup(self):
        """Close and delete the on-disk cache directory - call once a phase
        that built this cache is fully done with it (mirrors the old code's
        `del sel_train_graphs, ...; gc.collect()` between phases)."""
        import gc
        self.close()
        gc.collect()
        if self.cache_dir is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)


def build_disk_cache(rows, user_items, item_users, rng, cache_dir: Path, desc: str) -> DiskSubgraphCache:
    """Extracts every row's subgraph exactly once, streaming straight to disk
    (never holding more than one subgraph plus the small index arrays in RAM
    at a time), then reopens the written files as memmaps for reading."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    node_role_path = cache_dir / "node_roles.bin"
    edges_path = cache_dir / "edges.bin"

    n = len(rows)
    index = np.empty((n, 4), dtype=np.int64)
    ratings = np.empty(n, dtype=np.float32)
    node_off = 0
    edge_off = 0

    with open(node_role_path, "wb") as nf, open(edges_path, "wb") as ef:
        for k, (u, i, r) in enumerate(tqdm(rows, desc=desc, leave=False)):
            node_role, edges = extract_subgraph(int(u), int(i), user_items, item_users, rng)
            nf.write(node_role.tobytes())
            ef.write(edges.tobytes())
            index[k, 0] = node_off
            index[k, 1] = len(node_role)
            index[k, 2] = edge_off
            index[k, 3] = len(edges)
            ratings[k] = float(r)
            node_off += len(node_role)
            edge_off += len(edges)

    node_roles_mm = (
        np.memmap(node_role_path, dtype=np.uint8, mode="r", shape=(node_off,))
        if node_off else np.zeros((0,), dtype=np.uint8)
    )
    edges_mm = (
        np.memmap(edges_path, dtype=np.int32, mode="r", shape=(edge_off, 3))
        if edge_off else np.zeros((0, 3), dtype=np.int32)
    )
    return DiskSubgraphCache(node_roles_mm, edges_mm, index, ratings, cache_dir)


def collate(graphs_batch, device):
    """node_role/edges arrive as the compact numpy arrays extract_subgraph
    produces (via DiskSubgraphCache.get_batch); batching concatenates them
    (edge indices shifted by each graph's node offset) rather than looping
    in Python per edge."""
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


# ---------------------------------------------------------------------------
# R-GCN / IGMC model (unchanged from experiments/run_igmc.py)
# ---------------------------------------------------------------------------

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


def predict_igmc(model, cache: DiskSubgraphCache, batch_size, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            idx = range(start, min(start + batch_size, len(cache)))
            batch = collate(cache.get_batch(idx), device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def evaluate_rmse_igmc(model, cache: DiskSubgraphCache, batch_size, device):
    pred = predict_igmc(model, cache, batch_size, device)
    actual = cache.all_ratings()
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def train_igmc(
    train_cache: DiskSubgraphCache, val_cache: DiskSubgraphCache, hidden_dim, n_layers, lr, weight_decay,
    max_epochs, patience, batch_size, seed, desc=None, show_progress=False, heartbeat_every=None,
):
    torch.manual_seed(seed)
    model = IGMC(hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffle_rng = random.Random(seed)
    idx_all = list(range(len(train_cache)))

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
            batch_idx = idx_all[start:start + batch_size]
            batch = collate(train_cache.get_batch(batch_idx), DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = torch.mean((pred - batch["rating"]) ** 2)
            loss.backward()
            opt.step()

        v_rmse = evaluate_rmse_igmc(model, val_cache, batch_size, DEVICE)
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


def load_dataset_rows(dataset: str, seed: int = SEED):
    return load_split(dataset, seed=seed)


# ---------------------------------------------------------------------------
# Coordinate-descent local hyperparameter search - shared by any script whose
# grid is too expensive to brute-force (originally written for UAIMC,
# reused by IGMC once its own grid showed the same edge-hugging pattern -
# see run_experiment8_igmc.py's module docstring).
# ---------------------------------------------------------------------------

def local_grid_search(axis_candidates: dict[str, list], start: dict, evaluate_fn, max_passes: int,
                       seed_cache: dict | None = None):
    """Coordinate-descent search: hold every axis but one fixed at the
    current best, sweep that axis's candidates, keep whichever point
    improves val_rmse the most; repeat axis by axis for up to max_passes
    full sweeps, stopping early once a whole pass makes no improvement (a
    local optimum - "the valley"). evaluate_fn(cfg) -> val_rmse; this
    function memoizes every config it evaluates (keyed by axis_candidates'
    own key order, so it works for any axis set - UAIMC's 3, IGMC's 4,
    whatever a future caller needs) and seeds that memo from seed_cache so a
    restart doesn't repeat known points either."""
    axes = list(axis_candidates)

    def key(cfg):
        return tuple(cfg[a] for a in axes)

    seen: dict[tuple, float] = dict(seed_cache or {})

    def ev(cfg):
        k = key(cfg)
        if k in seen:
            return seen[k]
        val = evaluate_fn(cfg)
        seen[k] = val
        return val

    best_cfg = dict(start)
    best_val = ev(best_cfg)

    for pass_n in range(max_passes):
        improved = False
        for axis in axes:
            local_best_val, local_best_cfg = best_val, best_cfg
            for candidate in axis_candidates[axis]:
                cfg = dict(best_cfg)
                cfg[axis] = candidate
                val = ev(cfg)
                if val < local_best_val:
                    local_best_val, local_best_cfg = val, cfg
            if local_best_val < best_val - 1e-6:
                best_val, best_cfg = local_best_val, local_best_cfg
                improved = True
        print(f"  [local search] pass {pass_n + 1}/{max_passes}: best so far {best_cfg} (val_rmse={best_val:.4f})")
        if not improved:
            print(f"  [local search] converged after {pass_n + 1} pass(es) - no axis improved further")
            break

    return best_cfg, best_val
