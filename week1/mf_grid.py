"""Shared training code for week1's matrix-factorisation grid-search
notebooks - not meant to be run directly, only imported.

Lives in a real module rather than notebook cells because
ProcessPoolExecutor workers pickle-import whatever function they run,
which (on Windows' spawn start method) only works for module-level
functions, not ones defined inline in a notebook.

Experiment 1 (train_plain_mf): plain MF, no regularisation term at all -
r is the only hyperparameter. Experiment 2 (train_lasso_mf): the exact
same per-example SGD loop with one line added, an L1/Lasso penalty on A
and B, giving lambda as a second hyperparameter.

Both training loops are deliberately left as simple as the original
week1/MatrixFactorisation.ipynb: a plain Python for-loop over shuffled
examples with a hand-written numpy update, no vectorisation, no
early-stopping - fixed epoch count, easy to keep hand-editing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))
from recsys_loader import load_split

SEED = 42
N_THREADS = os.cpu_count() or 1


def load_ml100k(seed: int = SEED):
    """Same 70/20/10 split every other experiment in this repo uses (via
    recsys_loader), instead of the original notebook's own hand-rolled CSV
    read + shuffle - one fewer place a splitting bug can hide."""
    split = load_split("ml-100k", seed=seed)
    return {
        "train_u": split.train[:, 0].astype(int), "train_m": split.train[:, 1].astype(int), "train_r": split.train[:, 2].astype(float),
        "val_u": split.val[:, 0].astype(int), "val_m": split.val[:, 1].astype(int), "val_r": split.val[:, 2].astype(float),
        "test_u": split.test[:, 0].astype(int), "test_m": split.test[:, 1].astype(int), "test_r": split.test[:, 2].astype(float),
        "n_users": split.n_users, "n_movies": split.n_items,
    }


def predict(user_ids, movie_ids, A, B, mean=0.0):
    """mean defaults to 0.0 so this still works unchanged wherever a caller
    doesn't care about centering; the training functions below always pass
    the real training-set mean."""
    return np.sum(A[user_ids] * B[movie_ids], axis=1) + mean


def rmse(user_ids, movie_ids, true_ratings, A, B, mean=0.0):
    preds = predict(user_ids, movie_ids, A, B, mean)
    return np.sqrt(np.mean((true_ratings - preds) ** 2))


def train_plain_mf(r, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                    learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 1: r is the only hyperparameter - no weight decay term at
    all, unlike the original notebook's fixed `regularization=0.02`.

    Mean-centered: A/B are trained against (rating - mean_rating), so a
    nearly-zero embedding predicts "the average rating" rather than "zero
    stars" - the mean is added back by every rmse() call below and must be
    added back by any caller using A/B to predict a real rating afterwards
    (see the returned mean_rating)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    train_hist, val_hist = [], []
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

        train_hist.append(rmse(train_u, train_m, train_r, A, B, mean_rating))
        val_hist.append(rmse(val_u, val_m, val_r, A, B, mean_rating))

    return A, B, mean_rating, train_hist, val_hist


def train_lasso_mf(r, lam, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                    learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 2: identical to train_plain_mf (mean-centering included),
    with one addition - an L1/Lasso penalty lambda*(|A_u| + |B_m|) added to
    the loss, which turns into a `- lam * sign(...)` term in the
    per-example gradient step."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    train_hist, val_hist = [], []
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

        train_hist.append(rmse(train_u, train_m, train_r, A, B, mean_rating))
        val_hist.append(rmse(val_u, val_m, val_r, A, B, mean_rating))

    return A, B, mean_rating, train_hist, val_hist


def train_ours_mf(r, lam, num_users, num_movies, train_u, train_m, train_r, val_u, val_m, val_r,
                   learning_rate=0.01, epochs=30, seed=SEED):
    """Experiment 3 ("Ours", proposal eq. 4-5): a second embedding pair
    C, D (same rank r as A, B) predicts an uncertainty U = C D^T that
    down-weights each example's rating-loss contribution by exp(-U) while
    paying a `lam * U` penalty for doing so - larger U means "less
    confident". Same model experiments/run_ours.py trains with PyTorch/
    autodiff over minibatches; here it's the plain per-example SGD step
    (batch size 1) worked out by hand instead, mean-centered exactly like
    train_plain_mf/train_lasso_mf above.

    Per-example loss: (S - R_c)^2 * exp(-U) + lam * U, where S = A_u.B_m,
    U = C_u.D_m, R_c = R - mean_rating, error = R_c - S. Differentiating
    w.r.t. S and U and dropping the constant factor of 2 (same convention
    every other training loop in this module already uses) gives the two
    update rules below."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(num_users, r))
    B = rng.normal(0, 0.1, size=(num_movies, r))
    C = rng.normal(0, 0.1, size=(num_users, r))
    D = rng.normal(0, 0.1, size=(num_movies, r))

    mean_rating = train_r.mean()
    train_r_c = train_r - mean_rating

    train_hist, val_hist = [], []
    order = np.arange(len(train_r))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, m, rating = train_u[idx], train_m[idx], train_r_c[idx]

            s = np.dot(A[u], B[m])
            u_hat = np.dot(C[u], D[m])
            error = rating - s
            w = np.exp(-u_hat)

            A_old = A[u].copy()
            B_old = B[m].copy()
            C_old = C[u].copy()
            D_old = D[m].copy()

            A[u] += learning_rate * error * w * B_old
            B[m] += learning_rate * error * w * A_old

            u_grad = (error**2) * w - lam
            C[u] += learning_rate * u_grad * D_old
            D[m] += learning_rate * u_grad * C_old

        train_hist.append(rmse(train_u, train_m, train_r, A, B, mean_rating))
        val_hist.append(rmse(val_u, val_m, val_r, A, B, mean_rating))

    return A, B, C, D, mean_rating, train_hist, val_hist


# ---- ProcessPoolExecutor plumbing (same pattern as experiments/run_plain_mf.py) ----

_worker_data = {}


def _pool_init(data):
    global _worker_data
    _worker_data = data


def _pool_run_plain(cfg, epochs, seed):
    d = _worker_data
    t0 = time.time()
    A, B, mean_rating, train_hist, val_hist = train_plain_mf(
        cfg["r"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed,
    )
    return cfg, A, B, mean_rating, train_hist, val_hist, time.time() - t0


def _pool_run_lasso(cfg, epochs, seed):
    d = _worker_data
    t0 = time.time()
    A, B, mean_rating, train_hist, val_hist = train_lasso_mf(
        cfg["r"], cfg["lam"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed,
    )
    return cfg, A, B, mean_rating, train_hist, val_hist, time.time() - t0


def _pool_run_ours(cfg, epochs, seed):
    d = _worker_data
    t0 = time.time()
    A, B, C, D, mean_rating, train_hist, val_hist = train_ours_mf(
        cfg["r"], cfg["lam"], d["n_users"], d["n_movies"],
        d["train_u"], d["train_m"], d["train_r"], d["val_u"], d["val_m"], d["val_r"],
        epochs=epochs, seed=seed,
    )
    return cfg, A, B, C, D, mean_rating, train_hist, val_hist, time.time() - t0


def grid_search(configs, data, worker_fn, epochs=30, seed=SEED, workers=None):
    """Trains every config in `configs` (each a dict of hyperparameters,
    e.g. {"r": 10} or {"r": 10, "lam": 0.01}) in parallel worker processes,
    one per config, all sharing the same train/val arrays (sent to each
    worker once via the pool initializer, not once per config).

    Returns a list of whatever `worker_fn` returns, one tuple per config,
    in whatever order they finish - unordered on purpose, sort by val RMSE
    yourself in the notebook. Every _pool_run_* worker in this module ends
    its return tuple the same way, `..., val_hist, seconds`, regardless of
    what it puts in the middle (2 embedding matrices for plain/lasso MF, 4
    for Ours) - that's all this function itself relies on, for the printed
    progress line."""
    workers = workers or N_THREADS
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init, initargs=(data,)) as ex:
        futures = {ex.submit(worker_fn, cfg, epochs, seed): cfg for cfg in configs}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            cfg, val_hist, secs = result[0], result[-2], result[-1]
            results.append(result)
            done += 1
            print(f"[{done}/{len(configs)}] {cfg} -> val RMSE={val_hist[-1]:.4f} ({secs:.1f}s)")
    return results
