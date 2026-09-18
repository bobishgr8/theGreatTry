"""Standalone, overnight-runnable IGMC (Zhang & Chen 2020) baseline, moved
into experiments2's conventions and built on gnn_common.py's disk-backed
subgraph cache (see that module's docstring for why - this is the same
model ported from experiments/run_igmc.py, not a re-derivation).

  python run_experiment8_igmc.py                     # every dataset except ml-25m
  python run_experiment8_igmc.py --datasets ml-100k
  python run_experiment8_igmc.py --quick              # tiny grid/slice, smoke-test the pipeline

=====================================================================
Coordinate-descent local search, not the original 8-config grid
=====================================================================
The first overnight run used a brute {16,32}x{2,3}x{0.003}x{0.0,1e-4} grid
(8 configs) - faithful to experiments/run_igmc.py's own budget, but every
one of the 4 datasets it finished tonight picked n_layers=3 (the MAX of the
2 values tried) by a wide margin (~0.06-0.13 RMSE over n_layers=2), and 3/4
picked hidden_dim=16 (the MIN of the 2 values tried). lr was never searched
at all (a single fixed 0.003). That's the same edge-hugging + under-searched
pattern that motivated UAIMC's local search (see run_experiment5_uaimc.py's
module docstring) - so IGMC gets the same treatment now: gnn_common's
generic local_grid_search (moved there for exactly this reuse), seeded from
tonight's per-dataset winners, searching hidden_dim/n_layers/lr/weight_decay
candidates that deliberately extend past both edges the old grid hit
(n_layers up to 5, hidden_dim down to 8) plus a genuine lr sweep.

Two output files, matching experiment3/4's convention (not the single-file
pattern experiments/run_igmc.py used) so a crash mid-search loses at most
the one config in flight, not the whole search - and a checkpoint file
(like experiment5's) so a Phase-2 crash after the search converges doesn't
redo the search itself, only the interrupted refit:

  --out       (experiment8_igmc_grid.csv) - every (dataset, config) tried,
              appended the moment each one finishes (resumable).
  --best-out  (experiment8_igmc_best.csv) - the winning config per dataset,
              plus its abstention-swept selective RMSE (rand/val/supp rules,
              proposal Sec. 4.3) on the held-out test set.
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

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import gnn_common as gnn
import mf_common
from evaluate import sweep, ABSTENTION_RATES

DATASETS = gnn.DATASETS
SEED = gnn.SEED
CACHE_ROOT = HERE / "_subgraph_cache" / "experiment8_igmc"

METHOD = "IGMC"

# Known winners from tonight's original 8-config grid, used to seed the
# local search (and as the fallback default for any dataset never run yet).
KNOWN_BEST = {
    "ml-100k": {"hidden_dim": 16, "n_layers": 3, "lr": 0.003, "weight_decay": 1e-4},
    "ml-1m": {"hidden_dim": 16, "n_layers": 3, "lr": 0.003, "weight_decay": 1e-4},
    "douban": {"hidden_dim": 16, "n_layers": 3, "lr": 0.003, "weight_decay": 0.0},
    "amazon-games": {"hidden_dim": 16, "n_layers": 3, "lr": 0.003, "weight_decay": 1e-4},
}
DEFAULT_START = {"hidden_dim": 16, "n_layers": 3, "lr": 0.003, "weight_decay": 1e-4}
AXIS_CANDIDATES = {
    "hidden_dim": [8, 16, 24, 32],
    "n_layers": [2, 3, 4, 5],
    "lr": [0.001, 0.003, 0.01],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}
MAX_PASSES = 2

QUICK_AXIS_CANDIDATES = {"hidden_dim": [16], "n_layers": [2], "lr": [0.01], "weight_decay": [1e-4]}
QUICK_MAX_EPOCHS = 2
QUICK_PATIENCE = 2
QUICK_ROW_LIMIT = 800

MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 512

GRID_FIELDNAMES = ["method", "dataset", "hidden_dim", "n_layers", "lr", "weight_decay", "val_rmse", "seconds"]
BEST_FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_layers", "best_lr", "best_weight_decay", "val_rmse", "seconds",
]

CHECKPOINT_PATH = HERE / "results" / "experiment8_igmc_checkpoint.json"


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


def run_dataset(dataset: str, axis_candidates, max_epochs, patience, batch_size,
                 grid_out: Path, max_passes: int, row_limit=None) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()

    split = gnn.load_dataset_rows(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")

    train_rows, val_rows, test_rows = split.train, split.val, split.test
    if row_limit:
        train_rows, val_rows, test_rows = train_rows[:row_limit], val_rows[:row_limit], test_rows[:row_limit]

    # Same ml-100k identity fix as experiment5_uaimc.py's checkpoint code: the
    # dataset used to be ml-latest-small under a wrong label, so a checkpoint
    # keyed just "ml-100k" could silently be reused for the now-corrected data.
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
            grid_out, METHOD, dataset,
            [("hidden_dim", int), ("n_layers", int), ("lr", float), ("weight_decay", float)],
        )
        n_evaluated = [len(existing)]

        def evaluate_fn(cfg: dict) -> float:
            t0 = time.time()
            _, v_rmse, epochs_run = gnn.train_igmc(
                sel_train_cache, sel_val_cache, max_epochs=max_epochs, patience=patience,
                batch_size=batch_size, seed=SEED, heartbeat_every=max(1, max_epochs // 10), **cfg,
            )
            secs = time.time() - t0
            mf_common.append_results(grid_out, [{
                "method": METHOD, "dataset": dataset, "hidden_dim": cfg["hidden_dim"], "n_layers": cfg["n_layers"],
                "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "val_rmse": round(v_rmse, 4), "seconds": round(secs, 1),
            }], GRID_FIELDNAMES)
            n_evaluated[0] += 1
            print(f"  [{n_evaluated[0]}] hidden={cfg['hidden_dim']:>3} layers={cfg['n_layers']} "
                  f"lr={cfg['lr']} wd={cfg['weight_decay']:.0e} -> val_rmse={v_rmse:.4f} "
                  f"({epochs_run} epochs, {secs:.1f}s)")
            return v_rmse

        start_cfg = KNOWN_BEST.get(dataset, DEFAULT_START)
        print(f"{dataset}: starting local search from {start_cfg} "
              f"(seeded with {len(existing)} already-known point(s))")
        best_cfg, best_val_rmse = gnn.local_grid_search(axis_candidates, start_cfg, evaluate_fn, max_passes,
                                                          seed_cache=existing)

        print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f}) "
              f"after {n_evaluated[0]} total evaluated configs")
        save_checkpoint(checkpoint_key, best_cfg, best_val_rmse)

        sel_train_cache.cleanup()
        sel_val_cache.cleanup()
        del user_items, item_users
        gc.collect()

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    refit_cache_dir = CACHE_ROOT / f"{dataset}_refit"
    user_items, item_users = gnn.build_adjacency(trainval_rows)
    rng2 = random.Random(SEED)
    trainval_cache = gnn.build_disk_cache(trainval_rows, user_items, item_users, rng2, refit_cache_dir / "trainval",
                                           f"{dataset} trainval subgraphs")
    test_cache = gnn.build_disk_cache(test_rows, user_items, item_users, rng2, refit_cache_dir / "test",
                                       f"{dataset} test subgraphs")

    t0 = time.time()
    final_model, _, epochs_run = gnn.train_igmc(
        trainval_cache, test_cache, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, seed=SEED, desc=f"{dataset} refit", show_progress=True,
        heartbeat_every=max(1, max_epochs // 10), **best_cfg,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    actual = test_cache.all_ratings()
    test_user_idx = test_rows[:, 0].astype(int)
    trainval_user_idx = trainval_rows[:, 0].astype(int)
    pred = gnn.predict_igmc(final_model, test_cache, batch_size, gnn.DEVICE)

    rng_score = np.random.default_rng(SEED)
    random_score = rng_score.random(len(actual))

    n_users_total = int(max(trainval_user_idx.max(), test_user_idx.max())) + 1
    support = np.bincount(trainval_user_idx, minlength=n_users_total)
    support_score = -support[test_user_idx].astype(np.float64)

    rows_out = []
    for rule_name, score in [("rand", random_score), ("val", -pred), ("supp", support_score)]:
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

    trainval_cache.cleanup()
    test_cache.cleanup()
    clear_checkpoint(checkpoint_key)
    print(f"{dataset}: done in {time.time() - t_dataset0:.1f}s total")
    return rows_out


def already_done_best(results_path: Path, dataset: str) -> bool:
    seen_p = set()
    if not results_path.exists():
        return False
    import csv
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"].startswith("IGMC"):
                seen_p.add((row["method"], row["p"]))
    expected = {(f"IGMC ({rule})", str(p)) for rule in ("rand", "val", "supp") for p in ABSTENTION_RATES}
    return expected.issubset(seen_p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/slice, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --best-out already has it")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "experiment8_igmc_grid.csv")
    parser.add_argument("--best-out", type=Path, default=HERE / "results" / "experiment8_igmc_best.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--passes", type=int, default=MAX_PASSES,
                         help="max coordinate-descent sweeps per dataset")
    parser.add_argument("--neighbors", type=int, default=20,
                         help="max sampled cross-neighbors per subgraph (paper/IGMC default 20); the disk "
                              "cache removes the RAM reason to shrink this, but very dense datasets may still "
                              "want it smaller purely to bound extraction time/disk size")
    args = parser.parse_args()

    axis_candidates = QUICK_AXIS_CANDIDATES if args.quick else AXIS_CANDIDATES
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    row_limit = QUICK_ROW_LIMIT if args.quick else None
    max_passes = 1 if args.quick else args.passes
    gnn.MAX_NEIGHBORS = args.neighbors

    print(f"device: {gnn.DEVICE} | CPU threads: {gnn.N_THREADS} | MAX_NEIGHBORS={gnn.MAX_NEIGHBORS}")
    print(f"grid results -> {args.out}\nbest results  -> {args.best_out}")
    print("NOTE: coordinate-descent local search (see module docstring), not a brute grid.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done_best(args.best_out, dataset):
            print(f"\n=== {dataset}: already complete in {args.best_out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, axis_candidates, max_epochs, patience,
                            args.batch_size, args.out, max_passes, row_limit=row_limit)
        mf_common.append_results(args.best_out, rows, BEST_FIELDNAMES)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.best_out}")


if __name__ == "__main__":
    main()
