"""UA-IMC (Kasalicky, Ledent & Alves, RecSys '23), moved into experiments2's
conventions from experiments/run_uaimc_v2.py, on top of gnn_common.py's
disk-backed subgraph cache, with a wider but still TARGETED hyperparameter
search instead of a brute grid (see LocalGridSearch below).

  python run_experiment5_uaimc.py                     # ml-100k and ml-1m
  python run_experiment5_uaimc.py --datasets ml-100k
  python run_experiment5_uaimc.py --quick              # tiny grid/slice, smoke-test the pipeline

See experiments/run_uaimc_v2.py's module docstring for the full paper-
fidelity discussion (nuclear-norm MF + sigmoid-bounded GNN anomaly score +
ARR + IGMC-pretrain-then-joint-finetune, the ml-100k dataset-identity
caveat) - that reasoning is unchanged here and not repeated. What's
different in this version:

1. Subgraphs live in gnn_common.DiskSubgraphCache (memmapped files), not a
   Python list held fully in RAM for the whole grid search - this is the fix
   for the two overnight OOM-driven restarts run_uaimc_v2.py needed on
   ml-1m (see gnn_common.py's module docstring). Because RAM is no longer
   the reason to shrink subgraphs, MAX_NEIGHBORS_OVERRIDE below is far less
   aggressive than run_uaimc_v2.py's (15 vs. that version's 6 for ml-1m) -
   purely a extraction-time/disk-size knob now, not a survival requirement.

2. The hyperparameter search is a coordinate-descent LOCAL search (see
   LocalGridSearch) seeded from the best configs the overnight run already
   found (ml-100k: hidden=32,rank=10,lam=0.01; ml-1m: hidden=16,rank=10,
   lam=0.01), with candidates that deliberately extend PAST the old grid's
   edges - both winners landed on a boundary of the old {16,32}x{10,20}x
   {1e-3,1e-2} grid, which is itself evidence the true optimum might sit
   outside it (rank could want <10, lambda could want >0.01). A full cross
   product of the wider candidate set would be 4x5x4=80 configs/dataset;
   this instead evaluates one axis at a time holding the other two at the
   current best, converging once a full pass improves nothing - "much more
   thorough than 8 configs" without brute-forcing every combination.

3. Every config actually evaluated (from any pass) is appended to
   --out immediately (mf_common's on-disk grid convention), so a restart
   resumes the search using every already-known point instead of redoing
   any of them - LocalGridSearch's own cache is just seeded from --out.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import gnn_common as gnn
import mf_common
from evaluate import sweep, ABSTENTION_RATES

DATASETS = gnn.DATASETS
DEFAULT_DATASETS = ["ml-100k", "ml-1m"]  # same scope as the overnight run - see module docstring
SEED = gnn.SEED
CACHE_ROOT = HERE / "_subgraph_cache" / "experiment5_uaimc"
PRETRAIN_CACHE_DIR = HERE / "_subgraph_cache" / "experiment5_uaimc" / "_pretrain_cache"

# See MAX_NEIGHBORS_OVERRIDE's original comment in experiments/run_uaimc_v2.py
# for the history - this is now purely an extraction-time/disk-size knob
# (gnn_common's disk cache removed the RAM reason to shrink it), so it can
# stay much closer to the paper/IGMC default of 20 than that version's 6.
MAX_NEIGHBORS_OVERRIDE = {"ml-1m": 15, "amazon-games": 15}

# ---- Paper-pinned constants (Sec. 4.1) - not searched, unchanged from run_uaimc_v2.py ----
N_LAYERS = 4
ALPHA = 45.0
BETA = 0.001
MLP_HIDDEN = 64

BATCH_SIZES = {"ml-100k": 200, "douban": 200, "amazon-games": 1000}
DEFAULT_BATCH_SIZE = 200

MAX_EPOCHS = 60
PATIENCE = 8
PRETRAIN_MAX_EPOCHS = 30
PRETRAIN_PATIENCE = 5
PRETRAIN_LR = 0.003
PRETRAIN_WD = 0.0

# ---- The search itself: targeted local search, not a brute grid (see module docstring) ----
# Known overnight winners (both on an edge of the old {16,32}x{10,20}x
# {1e-3,1e-2} grid), used to seed the search and to decide which direction
# to extend candidates in.
KNOWN_BEST = {
    "ml-100k": {"hidden_dim": 32, "rank": 10, "lam": 0.01},
    "ml-1m": {"hidden_dim": 16, "rank": 10, "lam": 0.01},
}
AXIS_CANDIDATES = {
    "hidden_dim": [16, 24, 32, 48],
    "rank": [5, 8, 10, 15, 20],
    "lam": [0.003, 0.01, 0.03, 0.1],
}
MAX_PASSES = 2  # full coordinate sweeps; stops earlier if a pass improves nothing

QUICK_AXIS_CANDIDATES = {"hidden_dim": [16], "rank": [10], "lam": [1e-3]}
QUICK_MAX_EPOCHS = 2
QUICK_PATIENCE = 2
QUICK_PRETRAIN_MAX_EPOCHS = 2
QUICK_PRETRAIN_PATIENCE = 2
QUICK_ROW_LIMIT = 800

METHOD = "UA-IMC (faithful)"
GRID_FIELDNAMES = ["method", "dataset", "hidden_dim", "rank", "lam", "val_rmse", "seconds"]
BEST_FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_rank", "best_lam", "val_rmse", "seconds",
]

CHECKPOINT_PATH = HERE / "results" / "experiment5_uaimc_checkpoint.json"


def load_checkpoint(dataset: str) -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")).get(dataset)


def save_checkpoint(dataset: str, best_cfg: dict, best_val_rmse: float):
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")) if CHECKPOINT_PATH.exists() else {}
    data[dataset] = {"best_cfg": best_cfg, "best_val_rmse": best_val_rmse}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_checkpoint(dataset: str):
    if not CHECKPOINT_PATH.exists():
        return
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    data.pop(dataset, None)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# local_grid_search moved to gnn_common.py (generalized to any axis set) so
# run_experiment8_igmc.py can reuse it too, once its own grid showed the
# same edge-hugging pattern UAIMC's did - see that module's docstring.


# ---------------------------------------------------------------------------
# UA-IMC model (unchanged from experiments/run_uaimc_v2.py, using gnn.RGCNLayer/IGMC)
# ---------------------------------------------------------------------------

class UncertaintyGNN(nn.Module):
    """See experiments/run_uaimc_v2.py's UncertaintyGNN docstring - identical
    architecture and reasoning, just built on gnn_common's RGCNLayer."""

    def __init__(self, hidden_dim, n_layers=N_LAYERS, n_relations=gnn.N_RATING_CLASSES, n_node_roles=4,
                 mlp_hidden=MLP_HIDDEN, dropout=0.2):
        super().__init__()
        self.node_embed = nn.Embedding(n_node_roles, hidden_dim)
        self.layers = nn.ModuleList(
            [gnn.RGCNLayer(hidden_dim, hidden_dim, n_relations) for _ in range(n_layers)]
        )
        concat_dim = hidden_dim * n_layers * 2
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def load_pretrained_backbone(self, igmc_model: "gnn.IGMC"):
        self.node_embed.load_state_dict(igmc_model.node_embed.state_dict())
        self.layers.load_state_dict(igmc_model.layers.state_dict())

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

    def arr_loss(self):
        total = self.node_embed.weight.new_zeros(())
        for layer in self.layers:
            w = layer.rel_weight
            total = total + ((w[1:] - w[:-1]) ** 2).sum()
        return total


class UAIMC(nn.Module):
    def __init__(self, n_users, n_items, rank, hidden_dim, n_layers=N_LAYERS, seed=SEED, mean_rating=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(torch.randn(n_users, rank, generator=g) * 0.1)
        self.V = nn.Parameter(torch.randn(n_items, rank, generator=g) * 0.1)
        self.uncertainty = UncertaintyGNN(hidden_dim, n_layers)
        self.register_buffer("mean_rating", torch.tensor(float(mean_rating)))

    def predict_rating(self, u_idx, i_idx):
        return (self.U[u_idx] * self.V[i_idx]).sum(dim=1) + self.mean_rating

    def forward(self, subgraph_batch, u_idx, i_idx):
        s = self.predict_rating(u_idx, i_idx)
        w = self.uncertainty(subgraph_batch)
        return s, w


def joint_loss(s, w, actual, model, alpha, lam, beta, num_batches_per_epoch):
    """See experiments/run_uaimc_v2.py's joint_loss docstring for the full
    eq. 2 derivation and the minibatch-rescaling reasoning - unchanged."""
    mse_term = 0.5 * torch.mean(torch.exp(-w) * (actual - s) ** 2 + alpha * w)
    reg_term = 0.5 * lam * (model.U.pow(2).sum() + model.V.pow(2).sum()) / num_batches_per_epoch
    arr_term = beta * model.uncertainty.arr_loss() / num_batches_per_epoch
    return mse_term + reg_term + arr_term


def _global_ids(rows, batch_idx, device):
    u = torch.as_tensor(rows[batch_idx, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(rows[batch_idx, 1], dtype=torch.long, device=device)
    return u, i


def predict_uaimc(model, cache: gnn.DiskSubgraphCache, rows, batch_size, device):
    model.eval()
    preds, ws = [], []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            idx = np.arange(start, min(start + batch_size, len(cache)))
            batch = gnn.collate(cache.get_batch(idx), device)
            u_idx, i_idx = _global_ids(rows, idx, device)
            s, w = model(batch, u_idx, i_idx)
            preds.append(s.cpu().numpy())
            ws.append(w.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ws)


def evaluate_rmse_uaimc(model, cache, rows, batch_size, device):
    pred, _ = predict_uaimc(model, cache, rows, batch_size, device)
    actual = rows[:, 2]
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def train_uaimc(
    train_cache, train_rows, val_cache, val_rows, n_users, n_items,
    rank, hidden_dim, lam, pretrained_backbone, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    torch.manual_seed(seed)
    mean_rating = float(train_rows[:, 2].mean())
    model = UAIMC(n_users, n_items, rank, hidden_dim, seed=seed, mean_rating=mean_rating).to(gnn.DEVICE)
    model.uncertainty.load_pretrained_backbone(pretrained_backbone)

    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    shuffle_rng = random.Random(seed)
    idx_all = list(range(len(train_cache)))
    num_batches_per_epoch = -(-len(idx_all) // batch_size)

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    from tqdm import tqdm
    bar = tqdm(range(max_epochs), desc=desc or f"h={hidden_dim} rank={rank} lam={lam:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        shuffle_rng.shuffle(idx_all)
        model.train()
        for start in range(0, len(idx_all), batch_size):
            batch_idx = np.array(idx_all[start:start + batch_size])
            batch = gnn.collate(train_cache.get_batch(batch_idx), gnn.DEVICE)
            u_idx, i_idx = _global_ids(train_rows, batch_idx, gnn.DEVICE)
            actual = torch.as_tensor(train_rows[batch_idx, 2], dtype=torch.float32, device=gnn.DEVICE)

            opt.zero_grad()
            s, w = model(batch, u_idx, i_idx)
            loss = joint_loss(s, w, actual, model, ALPHA, lam, BETA, num_batches_per_epoch)
            loss.backward()
            opt.step()
        scheduler.step()

        v_rmse = evaluate_rmse_uaimc(model, val_cache, val_rows, batch_size, gnn.DEVICE)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"h={hidden_dim} rank={rank} lam={lam:.0e}"
            print(f"    [{tag}] epoch {epoch + 1}/{max_epochs}: val_rmse={v_rmse:.4f} (best={best_rmse:.4f})")

        if v_rmse < best_rmse - 1e-5:
            best_rmse = v_rmse
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_rmse, epochs_run


# ---------------------------------------------------------------------------

def run_dataset(dataset: str, axis_candidates, max_epochs, patience, pretrain_max_epochs, pretrain_patience,
                 grid_out: Path, max_passes: int, row_limit=None) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()
    batch_size = BATCH_SIZES.get(dataset, DEFAULT_BATCH_SIZE)

    gnn.MAX_NEIGHBORS = MAX_NEIGHBORS_OVERRIDE.get(dataset, 20)
    print(f"{dataset}: MAX_NEIGHBORS={gnn.MAX_NEIGHBORS}")

    split = gnn.load_dataset_rows(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,} | batch_size={batch_size}")

    train_rows, val_rows, test_rows = split.train, split.val, split.test
    if row_limit:
        train_rows, val_rows, test_rows = train_rows[:row_limit], val_rows[:row_limit], test_rows[:row_limit]

    heartbeat_every = max(1, max_epochs // 10)
    # ml-100k used to be ml-latest-small under a wrong label (610 users/9,724
    # items); now that datasets/movieLense-100k/ratings.csv is the real 1998
    # u.data benchmark (943/1,682), any checkpoint left over from a search on
    # the old data would silently be reused for the new one - load_checkpoint
    # only keys on the dataset name, not on what's actually on disk. Keying
    # ml-100k's checkpoint as "ml100ku" instead keeps that identity distinct
    # going forward without touching the (unaffected) cache dirs or result
    # CSVs, which already carry "ml-100k" as a column value rather than a key.
    checkpoint_key = "ml100ku" if dataset == "ml-100k" else dataset
    checkpoint = load_checkpoint(checkpoint_key)

    if checkpoint is not None:
        best_cfg, best_val_rmse = checkpoint["best_cfg"], checkpoint["best_val_rmse"]
        print(f"{dataset}: resuming from checkpoint - skipping search, best config already known: "
              f"{best_cfg} (val_rmse={best_val_rmse:.4f})")
    else:
        sel_cache_dir = CACHE_ROOT / f"{dataset}_sel"
        user_items, item_users = gnn.build_adjacency(train_rows)
        rng = random.Random(SEED)
        sel_train_cache = gnn.build_disk_cache(train_rows, user_items, item_users, rng, sel_cache_dir / "train",
                                                f"{dataset} train subgraphs")
        sel_val_cache = gnn.build_disk_cache(val_rows, user_items, item_users, rng, sel_cache_dir / "val",
                                              f"{dataset} val subgraphs")
        print(f"{dataset}: train={len(sel_train_cache):,} val={len(sel_val_cache):,} subgraphs cached on disk "
              f"(model selection)")

        existing = mf_common.load_existing_grid(
            grid_out, METHOD, dataset, [("hidden_dim", int), ("rank", int), ("lam", float)],
        )
        n_evaluated = [len(existing)]

        # One plain-IGMC pretrain per distinct hidden_dim actually visited by
        # the search, trained lazily the first time that hidden_dim is
        # needed and reused for every (rank, lam) tried at that width -
        # unlike the old static grid, the local search doesn't know in
        # advance which hidden_dim values it will touch.
        pretrained_by_hidden: dict[int, "gnn.IGMC"] = {}

        def get_pretrained(h: int):
            if h not in pretrained_by_hidden:
                # This pretrain is the single most expensive, most repeatedly-lost step
                # on ml-1m (~55-60 min) - three consecutive low-memory kills tonight all
                # landed a few minutes into the UAIMC training that follows it, throwing
                # the whole pretrain away each time even though it's independent of the
                # (rank, lam) search and identical on every restart. Caching its weights
                # to disk means a restart pays for it once, not once per kill.
                cache_path = PRETRAIN_CACHE_DIR / f"{dataset}_h{h}.pt"
                if cache_path.exists():
                    igmc_model = gnn.IGMC(hidden_dim=h, n_layers=N_LAYERS).to(gnn.DEVICE)
                    igmc_model.load_state_dict(torch.load(cache_path, map_location=gnn.DEVICE))
                    pretrained_by_hidden[h] = igmc_model
                    print(f"{dataset}: loaded cached IGMC backbone (hidden_dim={h}) from {cache_path.name}")
                else:
                    print(f"{dataset}: pretraining plain IGMC backbone (hidden_dim={h}, n_layers={N_LAYERS}) "
                          f"for uncertainty-network initialization (paper Sec. 4.1)...")
                    t0 = time.time()
                    igmc_model, igmc_val_rmse, igmc_epochs = gnn.train_igmc(
                        sel_train_cache, sel_val_cache, hidden_dim=h, n_layers=N_LAYERS,
                        lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD, max_epochs=pretrain_max_epochs,
                        patience=pretrain_patience, batch_size=batch_size, seed=SEED,
                        desc=f"{dataset} IGMC pretrain h={h}",
                    )
                    pretrained_by_hidden[h] = igmc_model
                    print(f"{dataset}: pretrained IGMC h={h} val_rmse={igmc_val_rmse:.4f} "
                          f"({igmc_epochs} epochs, {time.time() - t0:.1f}s)")
                    PRETRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(igmc_model.state_dict(), cache_path)
            return pretrained_by_hidden[h]

        def evaluate_fn(cfg: dict) -> float:
            t0 = time.time()
            backbone = get_pretrained(cfg["hidden_dim"])
            _, v_rmse, epochs_run = train_uaimc(
                sel_train_cache, train_rows, sel_val_cache, val_rows,
                split.n_users, split.n_items, cfg["rank"], cfg["hidden_dim"], cfg["lam"],
                backbone, max_epochs, patience, batch_size, SEED, heartbeat_every=heartbeat_every,
            )
            secs = time.time() - t0
            mf_common.append_results(grid_out, [{
                "method": METHOD, "dataset": dataset, "hidden_dim": cfg["hidden_dim"], "rank": cfg["rank"],
                "lam": cfg["lam"], "val_rmse": round(v_rmse, 4), "seconds": round(secs, 1),
            }], GRID_FIELDNAMES)
            n_evaluated[0] += 1
            print(f"  [{n_evaluated[0]}] hidden={cfg['hidden_dim']:>3} rank={cfg['rank']:>3} "
                  f"lam={cfg['lam']:.0e} -> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {secs:.1f}s)")
            return v_rmse

        start_cfg = KNOWN_BEST.get(dataset, {
            "hidden_dim": axis_candidates["hidden_dim"][0],
            "rank": axis_candidates["rank"][0],
            "lam": axis_candidates["lam"][0],
        })
        print(f"{dataset}: starting local search from {start_cfg} "
              f"(seeded with {len(existing)} already-known point(s))")
        best_cfg, best_val_rmse = gnn.local_grid_search(axis_candidates, start_cfg, evaluate_fn, max_passes,
                                                      seed_cache=existing)

        print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f}) "
              f"after {n_evaluated[0]} total evaluated configs")
        save_checkpoint(checkpoint_key, best_cfg, best_val_rmse)

        sel_train_cache.cleanup()
        sel_val_cache.cleanup()
        del user_items, item_users, pretrained_by_hidden
        gc.collect()

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    refit_cache_dir = CACHE_ROOT / f"{dataset}_refit"
    user_items, item_users = gnn.build_adjacency(trainval_rows)
    rng2 = random.Random(SEED)
    trainval_cache = gnn.build_disk_cache(trainval_rows, user_items, item_users, rng2, refit_cache_dir / "trainval",
                                           f"{dataset} trainval subgraphs")
    final_test_cache = gnn.build_disk_cache(test_rows, user_items, item_users, rng2, refit_cache_dir / "test",
                                             f"{dataset} test subgraphs")

    # Same caching as get_pretrained() above, keyed separately (refit_ prefix) since
    # this pretrain runs on trainval, not train alone - a different backbone.
    refit_cache_path = PRETRAIN_CACHE_DIR / f"{dataset}_refit_h{best_cfg['hidden_dim']}.pt"
    if refit_cache_path.exists():
        final_igmc = gnn.IGMC(hidden_dim=best_cfg["hidden_dim"], n_layers=N_LAYERS).to(gnn.DEVICE)
        final_igmc.load_state_dict(torch.load(refit_cache_path, map_location=gnn.DEVICE))
        print(f"{dataset}: loaded cached refit IGMC backbone from {refit_cache_path.name}")
    else:
        print(f"{dataset}: re-pretraining IGMC backbone on train+val for the final refit "
              f"(hidden_dim={best_cfg['hidden_dim']})...")
        t0 = time.time()
        final_igmc, _, _ = gnn.train_igmc(
            trainval_cache, final_test_cache, hidden_dim=best_cfg["hidden_dim"], n_layers=N_LAYERS,
            lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD, max_epochs=pretrain_max_epochs,
            patience=pretrain_patience, batch_size=batch_size, seed=SEED, desc=f"{dataset} refit IGMC pretrain",
        )
        print(f"{dataset}: refit pretrain done ({time.time() - t0:.1f}s)")
        PRETRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(final_igmc.state_dict(), refit_cache_path)

    t0 = time.time()
    final_model, _, epochs_run = train_uaimc(
        trainval_cache, trainval_rows, final_test_cache, test_rows,
        split.n_users, split.n_items, best_cfg["rank"], best_cfg["hidden_dim"], best_cfg["lam"],
        final_igmc, max_epochs, patience, batch_size, SEED,
        desc=f"{dataset} refit", show_progress=True, heartbeat_every=heartbeat_every,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    actual = test_rows[:, 2]
    pred, w = predict_uaimc(final_model, final_test_cache, test_rows, batch_size, gnn.DEVICE)

    rows_out = []
    for p, val in sweep(pred, actual, w).items():
        rows_out.append({
            "method": METHOD,
            "dataset": dataset,
            "p": p,
            "selective_rmse": round(val, 4),
            "best_hidden": best_cfg["hidden_dim"],
            "best_rank": best_cfg["rank"],
            "best_lam": best_cfg["lam"],
            "val_rmse": round(best_val_rmse, 4),
            "seconds": round(time.time() - t_dataset0, 1),
        })

    trainval_cache.cleanup()
    final_test_cache.cleanup()
    clear_checkpoint(checkpoint_key)
    print(f"{dataset}: done in {time.time() - t_dataset0:.1f}s total")
    return rows_out


def already_done_best(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    import csv
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"] == METHOD:
                seen.add(row["p"])
    expected = {str(p) for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/slice, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --best-out already has it")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "experiment5_uaimc_grid.csv")
    parser.add_argument("--best-out", type=Path, default=HERE / "results" / "experiment5_uaimc_best.csv")
    parser.add_argument("--passes", type=int, default=MAX_PASSES,
                         help="max coordinate-descent sweeps per dataset (default matches the recommended "
                              "'thorough but targeted' search - see module docstring)")
    args = parser.parse_args()

    axis_candidates = QUICK_AXIS_CANDIDATES if args.quick else AXIS_CANDIDATES
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    pretrain_max_epochs = QUICK_PRETRAIN_MAX_EPOCHS if args.quick else PRETRAIN_MAX_EPOCHS
    pretrain_patience = QUICK_PRETRAIN_PATIENCE if args.quick else PRETRAIN_PATIENCE
    row_limit = QUICK_ROW_LIMIT if args.quick else None
    max_passes = 1 if args.quick else args.passes

    print(f"device: {gnn.DEVICE} | CPU threads: {gnn.N_THREADS}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}")
    print("NOTE: faithful UA-IMC reimplementation on gnn_common's disk-backed subgraph cache, with a "
          "coordinate-descent local hyperparameter search (see module docstring) instead of a brute grid.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done_best(args.best_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, axis_candidates, max_epochs, patience, pretrain_max_epochs, pretrain_patience,
                            args.out, max_passes, row_limit=row_limit)
        mf_common.append_results(args.best_out, rows, BEST_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.best_out}")


if __name__ == "__main__":
    main()
