"""Calendar/time-based features (hour, day-of-week, month, cyclic encodings).

Calendar features are a pure function of the timestamp, so -- unlike lag/rolling
features -- they carry no leakage risk regardless of fold boundaries and can be
computed on the full series before or after splitting with identical results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_calendar_features(df: pd.DataFrame, timestamp_column: str | None = None) -> pd.DataFrame:
    """Add calendar features derived from a DatetimeIndex or timestamp column.

    Args:
        df: Input DataFrame.
        timestamp_column: Name of a timestamp column to use; if None, `df.index`
            is assumed to be a DatetimeIndex.

    Returns:
        A copy of `df` with added columns: `hour`, `day_of_week`, `month`,
        `day_of_year`, and sine/cosine cyclic encodings `hour_sin`/`hour_cos`
        (period 24), `day_of_week_sin`/`day_of_week_cos` (period 7), and
        `month_sin`/`month_cos` (period 12).
    """
    out = df.copy()
    timestamps = pd.DatetimeIndex(out[timestamp_column]) if timestamp_column is not None else out.index
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex when timestamp_column is None")

    out["hour"] = timestamps.hour
    out["day_of_week"] = timestamps.dayofweek
    out["month"] = timestamps.month
    out["day_of_year"] = timestamps.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out
