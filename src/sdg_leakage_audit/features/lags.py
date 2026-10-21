"""Lag-feature construction (causal, split-aware).

See `protocols/injectors.py` for the leaky-vs-correct lag computation used in the
P4 audit specifically; `make_lag_features` here always implements the correct
(causal, single-fold) behavior, equivalent to
`protocols.injectors.lag_feature_correct_train_fold_only` applied per column/lag.
"""

from __future__ import annotations

import pandas as pd


def make_lag_features(df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add causal lag features of `column` to `df` for each lag in `lags`.

    Each lag is a plain `pandas.Series.shift(lag)`, which only ever references
    values strictly before the current row. The caller is responsible for
    restricting `df` to a single fold before calling this, since a lag computed
    across a fold boundary (e.g. on a concatenated train+test series) would let a
    train-fold feature reference a test-fold value or vice versa.

    Args:
        df: Input DataFrame, assumed sorted chronologically and restricted to a
            single fold (to avoid cross-fold leakage; see protocols/injectors.py).
        column: Name of the column to lag.
        lags: List of positive integer lag values.

    Returns:
        A copy of `df` with added columns named f"{column}_lag{lag}" for each lag.
    """
    if any(lag <= 0 for lag in lags):
        raise ValueError(f"All lags must be positive integers, got {lags}")

    out = df.copy()
    for lag in lags:
        out[f"{column}_lag{lag}"] = out[column].shift(lag)
    return out
