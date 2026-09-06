"""Canonical loader for the explicit-feedback datasets in the IS470 proposal.

Every dataset is reduced to (user, item, rating) triples with user/item ids
remapped to contiguous 0..n-1 / 0..m-1 indices, then Omega is split once into
train/val/test, per Sec. 3 of the proposal: the id space is shared across all
three splits because S = AB^T and U = CD^T are indexed by the full n x m
matrix - train/val/test only decide which cells of that matrix are visible to
a given step, not what the matrix's shape is.

Two preprocessing modes are supported (see PREPROCESSING_MODES below):
  - "proposal": exactly Sec. 3 of the IS470 proposal - 70/20/10 split, no
    dedup/pruning/rounding.
  - "uaimc": matches the preprocessing in Kasalicky/Ledent/Alves (RecSys'23,
    Sec. 4 "Preprocessing") - at most one (most-recent) rating per user-item
    pair, Amazon Games pruned to users/items with >= 5 ratings, ratings
    rounded up to the nearest integer, 90/5/5 split. Useful for sanity-checking
    numbers directly against that paper's Table 1.

Usage:
    from recsys_loader import load_split

    split = load_split("ml-100k")                       # proposal protocol
    split = load_split("ml-100k", preprocessing="uaimc")  # UAIMC-matched protocol
    split.n_users, split.n_items      # matrix shape for A/B/C/D
    split.train[:, 0]                 # user indices (int)
    split.train[:, 1]                 # item indices (int)
    split.train[:, 2]                 # ratings (float)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "datasets-cleaned"

PREPROCESSING_MODES = ("proposal", "uaimc")

SPLIT_RATIOS: dict[str, tuple[float, float, float]] = {
    "proposal": (0.70, 0.20, 0.10),  # IS470 proposal Sec. 3
    "uaimc": (0.90, 0.05, 0.05),  # Kasalicky/Ledent/Alves RecSys'23 Sec. 4
}

MIN_SUPPORT_DATASETS = {"amazon-games"}  # UAIMC only prunes low-support entries here


@dataclass
class RatingSplit:
    name: str
    preprocessing: str
    n_users: int
    n_items: int
    user_ids: np.ndarray  # original id for each user index, i.e. user_ids[i] -> raw id
    item_ids: np.ndarray
    train: np.ndarray  # (n, 3) columns: [user_idx, item_idx, rating]
    val: np.ndarray
    test: np.ndarray


def _remap(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Factorize raw ids into contiguous 0..k-1 codes, returning (codes, original ids)."""
    codes, uniques = pd.factorize(series, sort=True)
    return codes, uniques.to_numpy()


def _split_indices(n: int, ratios: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _dedupe_keep_latest(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """UAIMC preprocessing: at most one rating per (user, item) pair, keeping the
    most recent by timestamp (ties broken arbitrarily but deterministically)."""
    return df.sort_values(timestamp_col, kind="stable").drop_duplicates(subset=["user_id", "item_id"], keep="last")


def _prune_low_support(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """UAIMC preprocessing (Amazon Games only): iteratively drop users/items with
    fewer than min_count ratings, since removing one side can push the other
    below threshold too."""
    while True:
        before = len(df)
        df = df[df.groupby("user_id")["user_id"].transform("size") >= min_count]
        df = df[df.groupby("item_id")["item_id"].transform("size") >= min_count]
        if len(df) == before:
            return df


def _finalize(df: pd.DataFrame, name: str, preprocessing: str, seed: int) -> RatingSplit:
    if preprocessing == "uaimc":
        df = _dedupe_keep_latest(df, "timestamp")
        if name in MIN_SUPPORT_DATASETS:
            df = _prune_low_support(df, min_count=5)
        df = df.assign(rating=np.ceil(df["rating"].to_numpy(dtype=np.float64)))

    user_codes, user_ids = _remap(df["user_id"])
    item_codes, item_ids = _remap(df["item_id"])
    ratings = df["rating"].to_numpy(dtype=np.float64)

    table = np.column_stack([user_codes, item_codes, ratings])
    train_idx, val_idx, test_idx = _split_indices(len(table), SPLIT_RATIOS[preprocessing], seed)
    return RatingSplit(
        name=name,
        preprocessing=preprocessing,
        n_users=len(user_ids),
        n_items=len(item_ids),
        user_ids=user_ids,
        item_ids=item_ids,
        train=table[train_idx],
        val=table[val_idx],
        test=table[test_idx],
    )


def _load_movielens(dataset_dir: str, id_cols: tuple[str, str, str, str], name: str, preprocessing: str, seed: int) -> RatingSplit:
    path = CLEANED_DIR / dataset_dir / "ratings.csv"
    user_col, item_col, rating_col, ts_col = id_cols
    df = pd.read_csv(path, usecols=[user_col, item_col, rating_col, ts_col])
    df = df.rename(columns={user_col: "user_id", item_col: "item_id", rating_col: "rating", ts_col: "timestamp"})
    return _finalize(df, name, preprocessing, seed)


def _load_douban(preprocessing: str, seed: int) -> RatingSplit:
    path = CLEANED_DIR / "douban" / "moviereviews_cleaned.csv"
    df = pd.read_csv(path, usecols=["user_id", "movie_id", "rating", "time"])
    df = df.rename(columns={"movie_id": "item_id", "time": "timestamp"})  # "time" is an ISO date string, sorts fine lexicographically
    return _finalize(df, "douban", preprocessing, seed)


def _load_amazon_games(preprocessing: str, seed: int) -> RatingSplit:
    path = CLEANED_DIR / "amazongGames" / "Video_Games_5.csv"
    df = pd.read_csv(path, usecols=["reviewerID", "asin", "overall", "unixReviewTime"])
    df = df.rename(columns={"reviewerID": "user_id", "asin": "item_id", "overall": "rating", "unixReviewTime": "timestamp"})
    return _finalize(df, "amazon-games", preprocessing, seed)


LOADERS = {
    "ml-100k": lambda mode, seed: _load_movielens(
        "movieLense-100k", ("userId", "movieId", "rating", "timestamp"), "ml-100k", mode, seed
    ),
    "ml-1m": lambda mode, seed: _load_movielens(
        "movieLense-1M", ("UserID", "MovieID", "Rating", "Timestamp"), "ml-1m", mode, seed
    ),
    "ml-25m": lambda mode, seed: _load_movielens(
        "movieLense-25M", ("userId", "movieId", "rating", "timestamp"), "ml-25m", mode, seed
    ),
    "douban": lambda mode, seed: _load_douban(mode, seed),
    "amazon-games": lambda mode, seed: _load_amazon_games(mode, seed),
}


def load_split(dataset: str, seed: int = 0, preprocessing: str = "proposal") -> RatingSplit:
    """Load one of "ml-100k", "ml-1m", "ml-25m", "douban", "amazon-games".

    preprocessing: "proposal" (default, IS470 Sec. 3 protocol) or "uaimc"
    (matches Kasalicky/Ledent/Alves RecSys'23 preprocessing, see module docstring).
    """
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}, choose from {sorted(LOADERS)}")
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(f"unknown preprocessing {preprocessing!r}, choose from {PREPROCESSING_MODES}")
    return LOADERS[dataset](preprocessing, seed)


if __name__ == "__main__":
    for mode in PREPROCESSING_MODES:
        print(f"--- preprocessing={mode} ---")
        for dataset in LOADERS:
            split = load_split(dataset, preprocessing=mode)
            total = len(split.train) + len(split.val) + len(split.test)
            print(
                f"{split.name:14s} users={split.n_users:>8,} items={split.n_items:>8,} "
                f"train={len(split.train):>9,} val={len(split.val):>9,} test={len(split.test):>9,} "
                f"(total={total:,})"
            )
