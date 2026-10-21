"""Data cleaning: frequency enforcement, sentinel handling, duplicate/outlier flags.

These functions are intentionally *detection/standardization* only -- they introduce
NaN rows (frequency gaps) or flag rows (duplicates, outliers) without imputing or
dropping anything by default, so the caller decides what to do and, critically, so
any imputation is fit on a train fold only. See `protocols/injectors.py` for the
leakage-safe (train-fit) vs leaky (full-data-fit) comparison this must respect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


def enforce_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Reindex a DataFrame onto a regular DatetimeIndex at the given frequency.

    Args:
        df: Input DataFrame with a DatetimeIndex.
        freq: Pandas frequency string (e.g. "1h", "10min").

    Returns:
        Reindexed DataFrame spanning `df.index.min()` to `df.index.max()` at `freq`,
        with gaps introduced as NaN rows where observations were missing from the
        original series. Any pre-existing duplicate timestamps are averaged first
        (duplicates must be flagged/reviewed via `flag_duplicates` before calling
        this, since silently averaging can hide a data-quality issue).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("enforce_frequency requires a DatetimeIndex")

    if df.index.has_duplicates:
        logger.warning(
            "enforce_frequency: %d duplicate timestamps found; averaging duplicates "
            "before reindexing",
            df.index.duplicated().sum(),
        )
        df = df.groupby(level=0).mean(numeric_only=True)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    reindexed = df.reindex(full_index)
    reindexed.index.name = "timestamp"

    n_gap_rows = reindexed.isna().all(axis=1).sum()
    logger.info(
        "enforce_frequency: %d -> %d rows at freq=%s (%d fully-missing gap rows introduced)",
        len(df),
        len(reindexed),
        freq,
        n_gap_rows,
    )
    return reindexed


def replace_sentinel_values(
    df: pd.DataFrame, sentinel: float, columns: list[str] | None = None
) -> pd.DataFrame:
    """Replace a known sentinel missing-value code (e.g. -200) with NaN.

    Args:
        df: Input DataFrame.
        sentinel: Sentinel value to replace (e.g. -200 for UCI Air Quality).
        columns: Columns to apply replacement to; defaults to all numeric columns.

    Returns:
        A copy of `df` with sentinel values replaced by NaN in the target columns.
    """
    df = df.copy()
    target_columns = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()

    n_replaced = 0
    for col in target_columns:
        mask = df[col] == sentinel
        n_replaced += int(mask.sum())
        df.loc[mask, col] = np.nan

    logger.info("replace_sentinel_values: replaced %d occurrences of %s with NaN", n_replaced, sentinel)
    return df


def flag_duplicates(df: pd.DataFrame) -> pd.Series:
    """Flag duplicate timestamps in a DataFrame's index.

    Args:
        df: Input DataFrame with a DatetimeIndex.

    Returns:
        Boolean Series aligned to `df.index`, True where the timestamp is a
        duplicate (every occurrence after the first for that timestamp).
    """
    return pd.Series(df.index.duplicated(keep="first"), index=df.index, name="is_duplicate")


def flag_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Flag outliers in a series using the IQR rule.

    Args:
        series: Input numeric series.
        k: IQR multiplier (default 1.5).

    Returns:
        Boolean Series, True where the value falls outside
        `[Q1 - k*IQR, Q3 + k*IQR]`. NaN values are never flagged.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    is_outlier = (series < lower) | (series > upper)
    return is_outlier.fillna(False).rename("is_outlier")


def clean_and_report(
    df: pd.DataFrame,
    freq: str,
    target_column: str,
    sentinel: float | None = None,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the standard cleaning pipeline on a loaded dataset and build a quality report.

    Pipeline: flag duplicate timestamps -> replace sentinel values (if given) ->
    enforce regular frequency (grouped per `group_column`, e.g. per monitoring
    station, if given) -> flag outliers on `target_column`. No imputation is
    performed here; any fitted imputation step belongs in the train-only leg of the
    forecasting pipeline (see `protocols/injectors.py`), never in this shared
    cross-dataset cleaning step.

    Args:
        df: Loaded DataFrame with a DatetimeIndex (from a `loaders.load_*` function).
        freq: Pandas frequency string to enforce (e.g. "1h", "10min").
        target_column: Name of the forecast target column, used for outlier flagging
            and missingness reporting.
        sentinel: Optional sentinel missing-value code to replace with NaN before
            frequency enforcement (e.g. -200 for UCI Air Quality).
        group_column: Optional column identifying independent sub-series (e.g.
            `"station"` for the Beijing multi-site dataset) that frequency
            enforcement must be applied to separately, since each group has its own
            timestamp range/order.

    Returns:
        A tuple `(cleaned_df, report)` where `report` contains: `n_rows_raw`,
        `n_duplicate_timestamps`, `n_sentinel_replaced`, `n_rows_after_freq`,
        `n_gap_rows_introduced`, `pct_missing_target_after_freq`, `n_outliers_target`.
    """
    report: dict[str, Any] = {"n_rows_raw": len(df)}

    if group_column is None:
        n_duplicates = int(flag_duplicates(df).sum())
    else:
        combined_keys = pd.MultiIndex.from_arrays([df[group_column].to_numpy(), df.index])
        n_duplicates = int(combined_keys.duplicated(keep="first").sum())
    report["n_duplicate_timestamps"] = n_duplicates

    working = df
    if sentinel is not None:
        before_na = working[target_column].isna().sum()
        working = replace_sentinel_values(working, sentinel=sentinel, columns=[target_column])
        report["n_sentinel_replaced"] = int(working[target_column].isna().sum() - before_na)
    else:
        report["n_sentinel_replaced"] = 0

    if group_column is not None:
        cleaned_parts = []
        for group_value, group_df in working.groupby(group_column):
            freq_df = enforce_frequency(group_df.drop(columns=[group_column]), freq=freq)
            freq_df[group_column] = group_value
            cleaned_parts.append(freq_df)
        cleaned = pd.concat(cleaned_parts).sort_index()
    else:
        cleaned = enforce_frequency(working, freq=freq)

    report["n_rows_after_freq"] = len(cleaned)
    report["n_gap_rows_introduced"] = report["n_rows_after_freq"] - (
        report["n_rows_raw"] - report["n_duplicate_timestamps"]
    )
    report["pct_missing_target_after_freq"] = float(cleaned[target_column].isna().mean() * 100.0)

    outlier_flags = flag_outliers_iqr(cleaned[target_column])
    report["n_outliers_target"] = int(outlier_flags.sum())

    return cleaned, report
