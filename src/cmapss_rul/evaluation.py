"""Evaluation metrics for remaining-useful-life regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return NASA's asymmetric prognostics score.

    Positive errors are late, unsafe predictions and receive the steeper penalty.
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )
    return float(penalties.sum())


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the shared project metrics with negative predictions clipped."""
    true = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "rmse": float(mean_squared_error(true, predicted) ** 0.5),
        "mae": float(mean_absolute_error(true, predicted)),
        "r2": float(r2_score(true, predicted)),
        "nasa_score": nasa_score(true, predicted),
    }
