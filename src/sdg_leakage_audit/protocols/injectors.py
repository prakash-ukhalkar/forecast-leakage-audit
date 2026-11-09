"""Leakage-injection logic for protocols P2, P3, P4, P5.

Each function pair below implements both a leaky path (the protocol error under
audit) and a corrected path (matching P0 behaviour), sharing the same interface so
notebooks/07 can swap between them and measure the resulting performance inflation.
Every function's docstring states exactly what protocol error it introduces (or
avoids) and why that causes (or prevents) leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# P2: scaler/imputer fit order (full dataset vs train-only)
# ---------------------------------------------------------------------------


def fit_scaler_leaky_full_data(
    full_data: np.ndarray,
) -> StandardScaler:
    """P2 (WRONG, under audit): fit a StandardScaler on the FULL dataset (train+test).

    This leaks test-set distributional statistics (mean, variance) into the
    normalization that will later be applied to the training fold: the model
    effectively "sees" summary statistics of future/test data before training,
    which it would never have access to in a genuine forecasting deployment.

    Args:
        full_data: Array of shape (n_samples, n_features) containing ALL rows,
            i.e. train and test concatenated.

    Returns:
        A `StandardScaler` fit on `full_data` (leaky).
    """
    scaler = StandardScaler()
    scaler.fit(full_data)
    logger.debug("fit_scaler_leaky_full_data (P2): fit on %d rows (train+test)", full_data.shape[0])
    return scaler


def fit_scaler_correct_train_only(
    train_data: np.ndarray,
) -> StandardScaler:
    """P0 (CORRECT): fit a StandardScaler on the TRAIN fold only.

    The scaler never observes test-set values, so normalization statistics
    reflect only information available at training time -- matching a genuine
    deployment where future data is unseen.

    Args:
        train_data: Array of shape (n_train, n_features) containing only the
            training fold.

    Returns:
        A `StandardScaler` fit on `train_data` only.
    """
    scaler = StandardScaler()
    scaler.fit(train_data)
    logger.debug("fit_scaler_correct_train_only (P0): fit on %d rows (train only)", train_data.shape[0])
    return scaler


def fit_imputer_leaky_full_data(full_data: np.ndarray, strategy: str = "mean") -> SimpleImputer:
    """P2 (WRONG, under audit): fit a SimpleImputer on the full dataset (train+test).

    Analogous to `fit_scaler_leaky_full_data`: imputed values for missing entries
    in the train fold are computed using statistics that include test-set rows.

    Args:
        full_data: Array containing all rows (train+test concatenated).
        strategy: Imputation strategy ("mean", "median", "most_frequent").

    Returns:
        A `SimpleImputer` fit on `full_data` (leaky).
    """
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(full_data)
    return imputer


def fit_imputer_correct_train_only(train_data: np.ndarray, strategy: str = "mean") -> SimpleImputer:
    """P0 (CORRECT): fit a SimpleImputer on the train fold only.

    Args:
        train_data: Array containing only the training fold.
        strategy: Imputation strategy ("mean", "median", "most_frequent").

    Returns:
        A `SimpleImputer` fit on `train_data` only.
    """
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(train_data)
    return imputer


# ---------------------------------------------------------------------------
# P3: overlapping windows across the train/test boundary
# ---------------------------------------------------------------------------


def make_windows_leaky_overlapping(
    series: np.ndarray,
    window_size: int,
    boundary: int,
) -> tuple[np.ndarray, np.ndarray]:
    """P3 (WRONG, under audit): build sliding windows with NO purge at the boundary.

    Windows are built at every consecutive start position (stride 1) across the
    entire series, including positions whose window spans across `boundary`
    (i.e. a window that contains timestamps from both the train side and the test
    side). Consecutive overlapping windows are near-duplicates of each other, so
    the "test" windows nearest the boundary are almost identical to the last few
    "train" windows, inflating apparent accuracy. Returns boolean masks marking
    which windows belong to train/test by whether their start position is before
    or at/after `boundary`, WITHOUT excluding windows that straddle the boundary.

    Args:
        series: 1-D array of the time series.
        window_size: Number of consecutive points per window.
        boundary: Position dividing "train side" (< boundary) from "test side"
            (>= boundary) starts.

    Returns:
        Tuple `(train_window_starts, test_window_starts)`: integer start-position
        arrays for windows assigned to train and test respectively. Some windows in
        each set may span across `boundary` (leaky) and the two sets' underlying
        data can overlap row-wise.
    """
    n = len(series)
    all_starts = np.arange(0, n - window_size + 1)
    train_window_starts = all_starts[all_starts < boundary]
    test_window_starts = all_starts[all_starts >= boundary]
    return train_window_starts, test_window_starts


def make_windows_correct_purged(
    series: np.ndarray,
    window_size: int,
    boundary: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """P0 (CORRECT): build sliding windows with a purge gap, no boundary-straddling.

    Only windows fully contained within the train side (window entirely before
    `boundary`) are kept for train, and only windows fully contained within the
    test side (window start at or after `boundary + horizon`, the purge gap) are
    kept for test. No window straddles the boundary, and no row is shared between
    a train window and a test window.

    Args:
        series: 1-D array of the time series.
        window_size: Number of consecutive points per window.
        boundary: Position dividing train side from test side.
        horizon: Forecast horizon; purge gap size enforced after `boundary`.

    Returns:
        Tuple `(train_window_starts, test_window_starts)`, each fully contained in
        their respective side with the purge gap respected.
    """
    n = len(series)
    all_starts = np.arange(0, n - window_size + 1)
    # Train windows must end strictly before boundary.
    train_window_starts = all_starts[all_starts + window_size <= boundary]
    # Test windows must start at/after boundary + horizon (the purge gap).
    test_window_starts = all_starts[all_starts >= boundary + horizon]
    return train_window_starts, test_window_starts


# ---------------------------------------------------------------------------
# P4: future-informed ("centered") rolling features vs causal ("trailing") ones
# ---------------------------------------------------------------------------


def rolling_feature_leaky_centered(series: pd.Series, window: int) -> pd.Series:
    """P4 (WRONG, under audit): centered rolling mean, which uses FUTURE values.

    `pandas.Series.rolling(window, center=True)` computes, at index i, the mean of
    values from roughly i - window//2 .. i + window//2 -- i.e. it uses values at
    timestamps AFTER i. Using this as a feature for forecasting at time i leaks
    information about the future that would not be available at prediction time.

    Args:
        series: Input time series.
        window: Rolling window size.

    Returns:
        Series of centered rolling means, aligned to `series.index`. By
        construction, `result[i]` depends on `series[i+1:]` values.
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()


def rolling_feature_correct_causal(series: pd.Series, window: int) -> pd.Series:
    """P0 (CORRECT): trailing (causal) rolling mean, using only past values.

    At index i, uses only `series[i - window + 1 : i + 1]` (default pandas
    rolling, `center=False`), so the feature is computable using only information
    available up to and including time i -- consistent with a genuine deployment.

    Args:
        series: Input time series.
        window: Rolling window size.

    Returns:
        Series of trailing rolling means, aligned to `series.index`. By
        construction, `result[i]` depends only on `series[:i+1]`.
    """
    return series.rolling(window=window, center=False, min_periods=1).mean()


def lag_feature_leaky_full_series(series: pd.Series, lag: int) -> pd.Series:
    """P4 (WRONG, under audit): lag feature computed on the full series pre-split.

    This itself is a correct causal shift (`series.shift(lag)`); the leak occurs
    when this is computed on the FULL concatenated series (train+test) before any
    split/sort is applied, and the data was not strictly sorted by time beforehand
    -- e.g. after a merge/concat/shuffle, "lag(1)" can silently reference a
    different (non-adjacent-in-time) row. Here we simulate that failure mode
    explicitly: the series is shuffled once, then lagged, so lag values no longer
    correspond to true temporal predecessors.

    Args:
        series: Input time series, assumed already shuffled/out-of-order upstream
            (callers should pass the *unsorted* concatenation to reproduce the bug).
        lag: Number of positions to shift.

    Returns:
        Series shifted by `lag` positions in whatever (possibly incorrect) row
        order `series` currently has.
    """
    return series.shift(lag)


def lag_feature_correct_train_fold_only(series: pd.Series, lag: int) -> pd.Series:
    """P0 (CORRECT): lag feature computed on a single, time-sorted fold.

    The input `series` must already be restricted to one fold (e.g. the train
    fold) and sorted chronologically; the shift then always references the true
    temporal predecessor `lag` steps back within that fold, with no cross-fold
    contamination.

    Args:
        series: Time-sorted series restricted to a single fold.
        lag: Number of positions to shift.

    Returns:
        Series shifted by `lag` positions within the given fold.
    """
    return series.shift(lag)


# ---------------------------------------------------------------------------
# P5: hyperparameter/model selection on test set vs on a preceding validation block
# ---------------------------------------------------------------------------


def select_best_params_leaky_on_test(
    candidate_params: list[dict],
    score_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """P5 (WRONG, under audit): select hyperparameters by scoring directly on the
    test set and keeping whichever candidate scores best there.

    This is a direct leakage of test-set information into model selection: even if
    no single model ever "sees" test data during its own `.fit()`, the selection
    process itself overfits to test-set noise, because the test set is used
    exactly like a validation set, then re-reported as if it were still a fair,
    unseen holdout.

    Args:
        candidate_params: List of hyperparameter dicts to choose among.
        score_fn: Callable `score_fn(params, X, y) -> float` (lower is better)
            that fits/evaluates a model with the given params on the given data.
        X_test: Test-set features (leaked into selection here).
        y_test: Test-set targets (leaked into selection here).

    Returns:
        The element of `candidate_params` that minimizes `score_fn(params, X_test, y_test)`.
    """
    scores = [score_fn(params, X_test, y_test) for params in candidate_params]
    best_idx = int(np.argmin(scores))
    logger.debug("select_best_params_leaky_on_test (P5): chose params=%s", candidate_params[best_idx])
    return candidate_params[best_idx]


def select_best_params_correct_on_validation(
    candidate_params: list[dict],
    score_fn,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """P0 (CORRECT): select hyperparameters by scoring on a validation block that
    strictly precedes the test block, never touching test data.

    Args:
        candidate_params: List of hyperparameter dicts to choose among.
        score_fn: Callable `score_fn(params, X, y) -> float` (lower is better).
        X_val: Validation-set features (precedes test chronologically; test
            remains untouched by this selection step).
        y_val: Validation-set targets.

    Returns:
        The element of `candidate_params` that minimizes `score_fn(params, X_val, y_val)`.
    """
    scores = [score_fn(params, X_val, y_val) for params in candidate_params]
    best_idx = int(np.argmin(scores))
    logger.debug("select_best_params_correct_on_validation (P0): chose params=%s", candidate_params[best_idx])
    return candidate_params[best_idx]
