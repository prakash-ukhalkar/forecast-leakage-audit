"""Feature-build pipeline orchestration: combines lags, rolling, and calendar features.

Must respect fold boundaries: this function should be called separately per fold
(train/val/test) under protocol P0, so lag/rolling features never reference a value
from outside the fold passed in. Calendar features carry no such risk (see
`features/calendar.py`) and are safe to compute regardless of fold boundaries.
"""

from __future__ import annotations

import pandas as pd

from sdg_leakage_audit.features.calendar import make_calendar_features
from sdg_leakage_audit.features.lags import make_lag_features
from sdg_leakage_audit.features.rolling import make_rolling_features


def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str,
    lags: list[int],
    rolling_windows: list[int],
    include_calendar: bool = True,
    rolling_stats: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for one fold of data.

    Args:
        df: Input DataFrame for a single fold, sorted chronologically.
        target_column: Name of the target column (used to build lag/rolling features).
        lags: Lag values to include.
        rolling_windows: Rolling window sizes to include.
        include_calendar: Whether to add calendar features.
        rolling_stats: Rolling statistics to compute per window; defaults to
            `["mean", "std"]` if None.

    Returns:
        DataFrame of engineered features, aligned to `df.index`, with rows
        containing insufficient history (e.g. earliest rows for large lags/windows)
        left as NaN per caller's downstream imputation strategy -- this function
        never drops rows, since dropping must happen fold-locally and consistently
        with how the caller handles the target's own NaN/alignment.
    """
    out = make_lag_features(df, column=target_column, lags=lags)
    out = make_rolling_features(
        out, column=target_column, windows=rolling_windows, stats=rolling_stats or ["mean", "std"]
    )
    if include_calendar:
        out = make_calendar_features(out)
    return out
