"""Forecast accuracy metrics: MAE, RMSE, sMAPE, MASE, and skill score vs seasonal-naive.

All functions operate on 1-D numpy arrays (or array-likes) and handle common edge
cases (zero denominators in sMAPE/MASE) by returning NaN for undefined elements
rather than raising or producing inf, with the overall aggregate then computed by
ignoring NaNs.
"""

from __future__ import annotations

import numpy as np


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Array-like of true values.
        y_pred: Array-like of predicted values, same shape as `y_true`.

    Returns:
        MAE as a float.
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Array-like of true values.
        y_pred: Array-like of predicted values, same shape as `y_true`.

    Returns:
        RMSE as a float.
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error, as a percentage (0-200 scale).

    Elements where `|y_true| + |y_pred|` is below `epsilon` (both effectively
    zero) are excluded from the mean rather than causing a division-by-zero /
    inf contribution.

    Args:
        y_true: Array-like of true values.
        y_pred: Array-like of predicted values, same shape as `y_true`.
        epsilon: Threshold below which a denominator is treated as "zero" and the
            corresponding element excluded.

    Returns:
        sMAPE as a float percentage in [0, 200]. Returns NaN if all elements are
        excluded (i.e. every true+pred pair is ~0).
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > epsilon
    if not np.any(valid):
        return float("nan")
    ratio = np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return float(np.mean(ratio) * 200.0)


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error, scaled by the in-sample naive seasonal MAE.

    The scale factor is the mean absolute error of a seasonal-naive forecast
    (`y_train[t] - y_train[t - seasonality]`) computed on the TRAINING series only
    (never on test data, to avoid leakage into the metric itself).

    Args:
        y_true: Array-like of true test-set values.
        y_pred: Array-like of predicted test-set values, same shape as `y_true`.
        y_train: Array-like of the training series, used to compute the naive
            in-sample scale factor.
        seasonality: Seasonal naive lag (e.g. 1 for simple persistence, 24 for
            hourly-daily seasonality).

    Returns:
        MASE as a float. Returns NaN if the in-sample scale factor is zero
        (degenerate constant training series) or if `y_train` is too short for
        the given `seasonality`.
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    y_train = _to_array(y_train)

    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if len(y_train) <= seasonality:
        return float("nan")

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(naive_errors))

    if scale == 0.0 or np.isnan(scale):
        return float("nan")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def skill_score_vs_seasonal_naive(y_true, y_pred, y_naive) -> float:
    """Skill score of a forecast relative to a seasonal-naive benchmark, using MAE.

    Defined as `1 - MAE(y_pred) / MAE(y_naive)`. Positive values indicate the
    forecast beats the seasonal-naive benchmark; 0 means equal performance;
    negative means worse than naive.

    Args:
        y_true: Array-like of true values.
        y_pred: Array-like of model predictions, same shape as `y_true`.
        y_naive: Array-like of seasonal-naive predictions, same shape as `y_true`.

    Returns:
        Skill score as a float. Returns NaN if the naive benchmark's MAE is zero
        (degenerate case where naive is perfect).
    """
    naive_mae = mae(y_true, y_naive)
    if naive_mae == 0.0:
        return float("nan")
    return float(1.0 - mae(y_true, y_pred) / naive_mae)
