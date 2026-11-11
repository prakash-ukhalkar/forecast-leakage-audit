"""Tests for src.sdg_leakage_audit.features (lags, rolling, calendar, build)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdg_leakage_audit.features.build import build_feature_matrix
from sdg_leakage_audit.features.calendar import make_calendar_features
from sdg_leakage_audit.features.lags import make_lag_features
from sdg_leakage_audit.features.rolling import make_rolling_features


def _series_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"y": np.arange(n, dtype=float)}, index=idx)


def test_make_lag_features_shifts_causally():
    df = _series_df()
    out = make_lag_features(df, column="y", lags=[1, 3])
    assert out["y_lag1"].iloc[5] == df["y"].iloc[4]
    assert out["y_lag3"].iloc[5] == df["y"].iloc[2]
    assert np.isnan(out["y_lag1"].iloc[0])  # no history for the first row


def test_make_lag_features_does_not_mutate_input():
    df = _series_df()
    make_lag_features(df, column="y", lags=[1])
    assert "y_lag1" not in df.columns


def test_make_lag_features_rejects_non_positive_lag():
    df = _series_df()
    with pytest.raises(ValueError):
        make_lag_features(df, column="y", lags=[0])


def test_make_rolling_features_trailing_window_only_uses_past_values():
    df = _series_df()
    out = make_rolling_features(df, column="y", windows=[3], stats=["mean"])
    # Row 5 (y=5.0): trailing window of 3 is rows [3, 4, 5] -> values [3, 4, 5] -> mean 4.0
    assert out["y_roll3_mean"].iloc[5] == pytest.approx(4.0)
    # Full window unavailable for the first two rows.
    assert np.isnan(out["y_roll3_mean"].iloc[0])
    assert np.isnan(out["y_roll3_mean"].iloc[1])


def test_make_rolling_features_rejects_unsupported_stat():
    df = _series_df()
    with pytest.raises(ValueError):
        make_rolling_features(df, column="y", windows=[3], stats=["median"])


def test_make_calendar_features_from_datetime_index():
    df = _series_df(30)
    out = make_calendar_features(df)
    assert out["hour"].iloc[1] == 1  # 2020-01-01 01:00
    assert set(["hour_sin", "hour_cos", "day_of_week_sin", "month_sin"]).issubset(out.columns)
    # Cyclic encoding sanity: sin^2 + cos^2 == 1.
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)


def test_make_calendar_features_raises_without_datetime_index():
    df = pd.DataFrame({"y": [1, 2, 3]})  # default RangeIndex, not datetime
    with pytest.raises(TypeError):
        make_calendar_features(df)


def test_build_feature_matrix_combines_all_feature_groups():
    df = _series_df(30)
    out = build_feature_matrix(df, target_column="y", lags=[1, 2], rolling_windows=[3], include_calendar=True)
    assert "y_lag1" in out.columns
    assert "y_roll3_mean" in out.columns
    assert "hour" in out.columns


def test_build_feature_matrix_without_calendar():
    df = _series_df(30)
    out = build_feature_matrix(df, target_column="y", lags=[1], rolling_windows=[2], include_calendar=False)
    assert "hour" not in out.columns
