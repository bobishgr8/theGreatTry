"""Shared training code for experiments2's overnight, all-dataset runs.

This is week1/mf_grid.py's plain-numpy, per-example-SGD matrix
factorisation (Experiments 1-3: plain MF / +Lasso / +joint uncertainty),
generalised from ml-100k-only to any dataset and promoted to the same
resumable, standalone-script pattern as experiments/run_plain_mf.py etc.

Copied rather than imported from week1/mf_grid.py on purpose: that module
is an actively-edited playground (see week1/experiment*.ipynb), and an
overnight multi-hour run has no business depending on code that might
change out from under it mid-run. The training math itself is identical -
see week1/mf_grid.py's docstrings for the derivations (mean-centering,
Lasso's `-lam*sign(...)` term, Ours' eq. 4-5 joint update).

ml-25m is not supported here - see run_experiment1_plain.py's module
docstring for the throughput math showing why (this is pure per-example
Python, no GPU/vectorisation, unlike experiments/run_*.py's PyTorch
versions of the same ideas).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))
from recsys_loader import load_split

SEED = 42
N_THREADS = os.cpu_count() or 1

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, see run_experiment1_plain.py


def load_dataset(name: str, seed: int = SEED):
    split = load_split(name, seed=seed)
    return {
        "train_u": split.train[:, 0].astype(int), "train_m": split.train[:, 1].astype(int), "train_r": split.train[:, 2].astype(float),
        "val_u": split.val[:, 0].astype(int), "val_m": split.val[:, 1].astype(int), "val_r": split.val[:, 2].astype(float),
        "test_u": split.test[:, 0].astype(int), "test_m": split.test[:, 1].astype(int), "test_r": split.test[:, 2].astype(float),
        "n_users": split.n_users, "n_movies": split.n_items,
    }


def predict(user_ids, movie_ids, A, B, mean=0.0):
    return np.sum(A[user_ids] * B[movie_ids], axis=1) + mean


def rmse(user_ids, movie_ids, true_ratings, A, B, mean=0.0):
    preds = predict(user_ids, movie_ids, A, B, mean)
    return np.sqrt(np.mean((true_ratings - preds) ** 2))


def train_plain_mf(r, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                    learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 1: r is the only hyperparameter - no weight decay term."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    order = np.arange(len(train_r))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, m, rating = train_u[idx], train_m[idx], train_r_c[idx]
            pred = np.dot(A[u], B[m])
            error = rating - pred

            A_old = A[u].copy()
            B_old = B[m].copy()

            A[u] += learning_rate * error * B_old
            B[m] += learning_rate * error * A_old

    val_rmse = rmse(val_u, val_m, val_r, A, B, mean_rating)
    return A, B, mean_rating, val_rmse


def train_lasso_mf(r, lam, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                    learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 2: train_plain_mf + an L1/Lasso penalty lambda*(|A_u|+|B_m|)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    order = np.arange(len(train_r))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, m, rating = train_u[idx], train_m[idx], train_r_c[idx]
            pred = np.dot(A[u], B[m])
            error = rating - pred

            A_old = A[u].copy()
            B_old = B[m].copy()

            A[u] += learning_rate * (error * B_old - lam * np.sign(A_old))
            B[m] += learning_rate * (error * A_old - lam * np.sign(B_old))

    val_rmse = rmse(val_u, val_m, val_r, A, B, mean_rating)
    return A, B, mean_rating, val_rmse


def train_ours_mf(r, lam, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                   learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 3 ("Ours", proposal eq. 4-5): S = A B^T, U = C D^T,
    exp(-U)-weighted rating loss plus a lam*U penalty. See
    week1/mf_grid.py's train_ours_mf docstring for the per-example gradient
    derivation."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))
    C = rng.normal(0, 0.1, size=(num_users, r))
    D = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    order = np.arange(len(train_r))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, m, rating = train_u[idx], train_m[idx], train_r_c[idx]

            s = np.dot(A[u], B[m])
            u_hat = np.dot(C[u], D[m])
            # Clipping w (below) alone wasn't enough: A[u] += lr*error*w*B_old
            # has no bound of its own, so even a moderately large w lets one
            # update jump an embedding row far enough that the *next*
            # prediction error is itself huge - cascading to overflow in a
            # handful of iterations regardless of how tightly w is clipped.
            # Centred ratings span roughly +-4, so error has no legitimate
            # reason to exceed a handful of that either; clipping it here
            # only ever bites once a step has already overshot, same
            # no-op-when-healthy reasoning as w's clip below.
            error = np.clip(rating - s, -10, 10)
            # w = exp(-u_hat) feeds back into u_hat's own update
            # (u_grad = error^2 * w - lam), so once w drifts large there is
            # no restoring force strong enough to stop it - confirmed on
            # amazon-games, where every lambda from 0 to 2.0 overflowed to
            # NaN even with error clipped. +-6 bounds w to [~0.0025, ~403],
            # comfortably wide for real confidence differentiation; still a
            # no-op for any config that wasn't already diverging (confirmed
            # no change to ml-100k's already-recorded r=2,lam=0.4 result).
            w = np.exp(-np.clip(u_hat, -6, 6))

            A_old = A[u].copy()
            B_old = B[m].copy()
            C_old = C[u].copy()
            D_old = D[m].copy()

            A[u] += learning_rate * error * w * B_old
            B[m] += learning_rate * error * w * A_old

            u_grad = (error**2) * w - lam
            C[u] += learning_rate * u_grad * D_old
            D[m] += learning_rate * u_grad * C_old

    val_rmse = rmse(val_u, val_m, val_r, A, B, mean_rating)
    return A, B, C, D, mean_rating, val_rmse


def train_ours_mf_ranks(r, r_u, lam, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                         learning_rate=0.01, epochs=30, seed=SEED):
    """Same model and update rule as train_ours_mf above, except the
    uncertainty factors C, D get their own rank r_u instead of reusing the
    rating factors' rank r. Nothing about proposal eq. 4-5 requires U = C D^T
    to have the same rank as S = A B^T - that's just the simplest choice, not
    a mathematical necessity - so this is the natural ablation: does the
    uncertainty signal need as much (or as little) capacity as the rating
    signal to be useful for abstention? A smaller r_u tests whether a coarse,
    heavily-compressed confidence signal is already enough; a larger r_u
    tests whether uncertainty is actually a *harder* pattern to capture than
    the ratings themselves and wants more room.

    Kept as a separate function (duplicated body) rather than adding an
    `r_u=None` default to train_ours_mf so that experiment 3's already-
    validated code path can never be touched by future edits here - see
    that function's docstring for the clipping rationale, which applies
    identically here since the update rule is unchanged."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))
    C = rng.normal(0, 0.1, size=(num_users, r_u))
    D = rng.normal(0, 0.1, size=(num_movies, r_u))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    order = np.arange(len(train_r))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, m, rating = train_u[idx], train_m[idx], train_r_c[idx]

            s = np.dot(A[u], B[m])
            u_hat = np.dot(C[u], D[m])
            error = np.clip(rating - s, -10, 10)
            w = np.exp(-np.clip(u_hat, -6, 6))

            A_old = A[u].copy()
            B_old = B[m].copy()
            C_old = C[u].copy()
            D_old = D[m].copy()

            A[u] += learning_rate * error * w * B_old
            B[m] += learning_rate * error * w * A_old

            u_grad = (error**2) * w - lam
            C[u] += learning_rate * u_grad * D_old
            D[m] += learning_rate * u_grad * C_old

    val_rmse = rmse(val_u, val_m, val_r, A, B, mean_rating)
    return A, B, C, D, mean_rating, val_rmse


# ---- ProcessPoolExecutor plumbing (same pattern as experiments/run_plain_mf.py) ----

_worker_data = {}
_worker_lr = 0.01


def _pool_init(data, learning_rate=0.01):
    global _worker_data, _worker_lr
    _worker_data = data
    _worker_lr = learning_rate


def _pool_run_plain(cfg, epochs, seed):
    """Deliberately returns only the scalar val_rmse, not A/B - grid_search
    below keeps every config's result alive for the whole sweep, and for a
    100+ config grid on a dataset with big embedding matrices (e.g.
    amazon-games), holding every config's trained A/B (or, for
    _pool_run_ours, A/B/C/D) simultaneously - rather than just the winner's
    - is what actually exhausted host memory on this machine (see the
    overnight run that motivated this comment). The winning config gets
    retrained once more, alone, by refit_plain in run_experiment1_plain.py,
    which is the only place that ever needs the real matrices."""
    d = _worker_data
    t0 = time.time()
    _A, _B, _mean, val_rmse = train_plain_mf(
        cfg["r"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed,
    )
    return cfg, val_rmse, time.time() - t0


def _pool_run_lasso(cfg, epochs, seed):
    """See _pool_run_plain's docstring - same reasoning."""
    d = _worker_data
    t0 = time.time()
    _A, _B, _mean, val_rmse = train_lasso_mf(
        cfg["r"], cfg["lam"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed,
    )
    return cfg, val_rmse, time.time() - t0


def _pool_run_ours(cfg, epochs, seed):
    """See _pool_run_plain's docstring - same reasoning (this one would
    have been the worst offender, holding 4 large matrices per config
    instead of 2).

    Uses _worker_lr rather than train_ours_mf's default: amazon-games
    diverges to nonsense (not even NaN - silently wrong numbers like 1e146)
    at the default 0.01 regardless of (r, lam) or the u_hat/error clipping
    above, because per-example SGD applies ~10.4M updates for this dataset
    (348k train rows x 30 epochs) - confirmed empirically that 0.002 is
    stable while 0.01/0.005 aren't. ml-100k/ml-1m/douban keep the default
    (their recorded results were produced at 0.01 and stay reproducible);
    see run_experiment3_ours.py's LEARNING_RATES for the per-dataset map."""
    d = _worker_data
    t0 = time.time()
    _A, _B, _C, _D, _mean, val_rmse = train_ours_mf(
        cfg["r"], cfg["lam"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed, learning_rate=_worker_lr,
    )
    return cfg, val_rmse, time.time() - t0


def _pool_run_ours_ranks(cfg, epochs, seed):
    """Rank-ablation counterpart to _pool_run_ours - same reasoning, same
    _worker_lr override, just calling train_ours_mf_ranks with cfg's r_u."""
    d = _worker_data
    t0 = time.time()
    _A, _B, _C, _D, _mean, val_rmse = train_ours_mf_ranks(
        cfg["r"], cfg["r_u"], cfg["lam"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed, learning_rate=_worker_lr,
    )
    return cfg, val_rmse, time.time() - t0


def grid_search(configs, data, worker_fn, epochs, seed, workers, desc, on_result=None, learning_rate=0.01):
    """Every worker_fn returns just (cfg, val_rmse, seconds) - light enough
    that holding every config's result for the whole sweep is negligible
    (a handful of floats each), unlike the embedding matrices that used to
    live here. tqdm bar instead of one print per config, since a full
    overnight grid here can be 100+ configs per dataset - see
    run_experiment1_plain.py's module docstring for the timing math.

    on_result(cfg, val_rmse, seconds), if given, fires the moment each
    config finishes - callers use it to append that single row to the grid
    CSV immediately (see load_existing_grid below), so a low-memory kill
    mid-sweep loses at most the configs still in flight, not the whole
    dataset's progress so far. Without this, resuming a 200+ config sweep
    interrupted at config 118 silently redid all 118 - exactly what
    happened across several ml-1m restarts before this existed.

    learning_rate is threaded through to each worker via _pool_init purely
    for _pool_run_ours's per-dataset override (see its docstring) -
    _pool_run_plain/_pool_run_lasso ignore it and keep their own defaults."""
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init, initargs=(data, learning_rate)) as ex:
        futures = {ex.submit(worker_fn, cfg, epochs, seed): cfg for cfg in configs}
        best_so_far = float("inf")
        pbar = tqdm(as_completed(futures), total=len(configs), desc=desc)
        for future in pbar:
            cfg, val_rmse, secs = future.result()
            results.append((cfg, val_rmse, secs))
            if on_result:
                on_result(cfg, val_rmse, secs)
            best_so_far = min(best_so_far, val_rmse)
            pbar.set_postfix(val_rmse=f"{val_rmse:.4f}", best=f"{best_so_far:.4f}")
            pbar.write(f"  {cfg} -> val RMSE={val_rmse:.4f} ({secs:.1f}s)")
    return results


def load_existing_grid(path: Path, method: str, dataset: str, key_spec: list) -> dict:
    """{(key values...): val_rmse} for every row already recorded for this
    (method, dataset) in path - key_spec is e.g. [("r", int), ("lam",
    float)] (Experiment 1 has no lambda, so just [("r", int)]). Lets a
    restarted run submit only the configs it doesn't already have to the
    pool, instead of redoing the whole grid."""
    existing = {}
    if not path.exists():
        return existing
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] != method or row["dataset"] != dataset:
                continue
            key = tuple(cast(row[name]) for name, cast in key_spec)
            existing[key] = float(row["val_rmse"])
    return existing


def already_done(results_path: Path, method: str, dataset: str) -> bool:
    if not results_path.exists():
        return False
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method and row["dataset"] == dataset:
                return True
    return False


def append_results(results_path: Path, rows: list[dict], fieldnames: list[str]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
