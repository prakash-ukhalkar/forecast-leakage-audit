"""Rolling-window statistical features (causal, trailing only).

See `protocols/injectors.py` for the leaky centered-vs-causal rolling feature
comparison used in the P4 audit; `make_rolling_features` here always implements the
correct (trailing, causal) behavior, equivalent to
`protocols.injectors.rolling_feature_correct_causal` applied per column/window/stat.
"""

from __future__ import annotations

import pandas as pd

_SUPPORTED_STATS = {"mean", "std", "min", "max"}


def make_rolling_features(
    df: pd.DataFrame, column: str, windows: list[int], stats: list[str]
) -> pd.DataFrame:
    """Add trailing (causal) rolling statistics of `column` to `df`.

    Each feature uses `pandas.Series.rolling(window, center=False)`, so at row i it
    only aggregates values from `[i - window + 1, i]` -- never future values. A full
    `window` of history is required (`min_periods=window`); rows with insufficient
    history are left as NaN rather than silently computed from a partial window,
    since a partial-window statistic computed consistently for both train and test
    rows would quietly change the feature's effective window size near each fold's
    start.

    Args:
        df: Input DataFrame, sorted chronologically, restricted to a single fold.
        column: Name of the column to compute rolling statistics over.
        windows: List of window sizes.
        stats: List of statistic names to compute, e.g. ["mean", "std", "min", "max"].

    Returns:
        A copy of `df` with added columns named f"{column}_roll{window}_{stat}".
    """
    unsupported = set(stats) - _SUPPORTED_STATS
    if unsupported:
        raise ValueError(f"Unsupported rolling stats {unsupported}; supported: {_SUPPORTED_STATS}")

    out = df.copy()
    for window in windows:
        rolling = out[column].rolling(window=window, center=False, min_periods=window)
        for stat in stats:
            out[f"{column}_roll{window}_{stat}"] = getattr(rolling, stat)()
    return out
