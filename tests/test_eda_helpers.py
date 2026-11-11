"""Tests for stats.tests.stationarity_tests and viz plotting helpers.

Plotting tests only check that the functions run end-to-end and produce a
non-empty file (visual correctness isn't something a unit test can verify), using
Matplotlib's non-interactive Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from sdg_leakage_audit.stats.tests import stationarity_tests
from sdg_leakage_audit.viz.plots import (
    plot_acf_pacf,
    plot_distribution_shift,
    plot_missingness_heatmap,
    plot_time_series,
)
from sdg_leakage_audit.viz.style import apply_plot_style


def test_apply_plot_style_runs_without_error():
    apply_plot_style()  # should not raise


def test_stationarity_tests_detects_stationary_white_noise():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=500))
    result = stationarity_tests(series)
    assert result["adf_rejects_unit_root"] is True
    assert result["kpss_rejects_stationary"] is False
    assert result["conclusion"] == "stationary"


def test_stationarity_tests_detects_non_stationary_random_walk():
    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.normal(size=500)))
    result = stationarity_tests(series)
    assert result["adf_rejects_unit_root"] is False
    assert result["conclusion"] in {"non_stationary", "inconclusive"}


def test_stationarity_tests_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        stationarity_tests(pd.Series([1.0, 2.0, 3.0]))


def test_stationarity_tests_drops_nans_before_testing():
    rng = np.random.default_rng(2)
    values = rng.normal(size=500)
    series = pd.Series(values)
    series.iloc[::10] = np.nan
    result = stationarity_tests(series)
    assert isinstance(result["adf_pvalue"], float)


def _synthetic_series(n: int = 300) -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(rng.normal(size=n), index=idx, name="value")


def test_plot_time_series_saves_file(tmp_path: Path):
    out = tmp_path / "ts.png"
    plot_time_series(_synthetic_series(), title="test", save_path=out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_acf_pacf_saves_file(tmp_path: Path):
    out = tmp_path / "acf.png"
    plot_acf_pacf(_synthetic_series(), lags=20, save_path=out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_missingness_heatmap_saves_file(tmp_path: Path):
    series = _synthetic_series()
    df = pd.DataFrame({"a": series, "b": series.where(series > 0)})
    out = tmp_path / "missingness.png"
    plot_missingness_heatmap(df, save_path=out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_distribution_shift_saves_file(tmp_path: Path):
    out = tmp_path / "shift.png"
    early = _synthetic_series(200)
    late = _synthetic_series(200) + 5.0
    plot_distribution_shift(early, late, title="test shift", save_path=out)
    assert out.is_file() and out.stat().st_size > 0
