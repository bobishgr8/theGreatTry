"""Selective-RMSE evaluation shared by every method's notebook (proposal eq. 2-3).

Every baseline/method notebook ends up calling `sweep(...)` once per abstention
rule it wants to score, and the results (one dict per row) are exactly what
Table 1 in the proposal is assembled from.
"""

from __future__ import annotations

import numpy as np

from abstention import retain_indices

ABSTENTION_RATES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)


def selective_rmse(pred: np.ndarray, actual: np.ndarray, retained: np.ndarray) -> float:
    """RMSE over the retained subset only, eq. (3). At p=0, retained == everything."""
    diff = pred[retained] - actual[retained]
    return float(np.sqrt(np.mean(diff**2)))


def sweep(
    pred: np.ndarray,
    actual: np.ndarray,
    unreliability: np.ndarray,
    rates: tuple[float, ...] = ABSTENTION_RATES,
) -> dict[float, float]:
    """Selective RMSE at each abstention rate, abstaining on the highest-unreliability
    entries first. `unreliability` is whatever ranks entries for a given method:
    a learned Uhat, or the score from a baseline abstention rule."""
    return {p: selective_rmse(pred, actual, retain_indices(unreliability, p)) for p in rates}
