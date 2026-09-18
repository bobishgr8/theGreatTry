"""Abstention rules shared by every method in experiments2 (proposal Sec. 3
and Sec. 4.2). Ported from experiments/abstention.py so experiments2's
scripts are self-contained (no sys.path reach into experiments/) - the logic
itself is dataset/framework-agnostic and unchanged.

Every rule reduces to the same primitive: rank test entries by an "unreliability"
score (higher = less confident) and keep the (1-p) fraction with the lowest
score, per eq. (2): |Omega_p(f)| = ceil((1-p) * |Omega_test|). A learned
uncertainty model (Ours' C/D factors, UAIMC's W) just plugs its own score into
retain_indices; the two baselines below are the naive, untrained substitutes
for that score that the proposal compares against.
"""

from __future__ import annotations

import numpy as np


def retain_indices(unreliability: np.ndarray, p: float) -> np.ndarray:
    """Indices to keep at abstention rate p, sorted ascending by unreliability
    (lowest first = kept first). Ties broken by original index for determinism."""
    n = len(unreliability)
    n_keep = int(np.ceil((1 - p) * n))
    order = np.lexsort((np.arange(n), unreliability))  # stable tie-break on index
    return np.sort(order[:n_keep])


def random_abstention(n: int, p: float, seed: int) -> np.ndarray:
    """Baseline: abstain on a uniformly random fraction p, ignoring any model signal."""
    rng = np.random.default_rng(seed)
    return retain_indices(rng.random(n), p)


def support_abstention(train_user_idx: np.ndarray, test_user_idx: np.ndarray, p: float) -> np.ndarray:
    """Baseline: abstain first on test entries whose user has the fewest TRAINING
    ratings ("low support" -> treated as least reliable)."""
    n_users = int(max(train_user_idx.max(), test_user_idx.max())) + 1
    train_support = np.bincount(train_user_idx, minlength=n_users)
    unreliability = -train_support[test_user_idx].astype(np.float64)  # more support = more reliable
    return retain_indices(unreliability, p)


def low_rating_abstention(pred: np.ndarray, p: float) -> np.ndarray:
    """Baseline: abstain first on test entries the model predicts the LOWEST
    rating for (proposal Algorithm 2) - a low predicted rating is treated as
    the least reliable/least worth recommending, independent of any actual
    uncertainty signal. Framed as retain_indices' "higher = less confident"
    convention by negating pred, so the lowest-predicted entries get the
    highest unreliability and are the first to go."""
    return retain_indices(-pred, p)
