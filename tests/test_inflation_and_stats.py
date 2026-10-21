"""Tests for evaluation.inflation, stats.bootstrap, stats.tests (paired/Friedman/Cliff's delta)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sdg_leakage_audit.evaluation.inflation import compute_inflation
from sdg_leakage_audit.stats.bootstrap import moving_block_bootstrap_ci
from sdg_leakage_audit.stats.tests import (
    effect_size_cliffs_delta,
    friedman_nemenyi,
    paired_tests_with_correction,
)
from sdg_leakage_audit.viz.plots import (
    plot_inflation_bars,
    plot_inflation_heatmap,
    save_figure_for_export,
)


def _synthetic_results(n_keys: int = 10) -> pd.DataFrame:
    """P0 has mae=1.0 everywhere; P1 is consistently better (leaky); P2 is the same as P0."""
    rows = []
    for i in range(n_keys):
        key = {"dataset_id": "d", "model_name": "m", "horizon": 1, "seed": i}
        rows.append({**key, "protocol_id": "P0", "mae": 1.0, "skill_score": 0.5})
        rows.append({**key, "protocol_id": "P1", "mae": 0.5, "skill_score": 0.8})
        rows.append({**key, "protocol_id": "P2", "mae": 1.0, "skill_score": 0.5})
    return pd.DataFrame(rows)


def test_compute_inflation_sign_convention_for_lower_is_better_metric():
    df = _synthetic_results()
    out = compute_inflation(df, baseline_protocol="P0")
    p1_rows = out[out["protocol_id"] == "P1"]
    # P1's mae (0.5) is lower (better) than P0's (1.0) -> positive inflation.
    assert np.allclose(p1_rows["mae_inflation_abs"].to_numpy(), 0.5)
    assert np.allclose(p1_rows["mae_inflation_pct"].to_numpy(), 50.0)


def test_compute_inflation_sign_convention_for_higher_is_better_metric():
    df = _synthetic_results()
    out = compute_inflation(df, baseline_protocol="P0")
    p1_rows = out[out["protocol_id"] == "P1"]
    # P1's skill_score (0.8) is higher (better) than P0's (0.5) -> positive inflation.
    assert np.allclose(p1_rows["skill_score_inflation_abs"].to_numpy(), 0.3)


def test_compute_inflation_zero_for_identical_protocol():
    df = _synthetic_results()
    out = compute_inflation(df, baseline_protocol="P0")
    p2_rows = out[out["protocol_id"] == "P2"]
    assert np.allclose(p2_rows["mae_inflation_abs"].to_numpy(), 0.0)


def test_compute_inflation_raises_for_missing_baseline():
    df = _synthetic_results()
    df = df[df["protocol_id"] != "P0"]
    with pytest.raises(ValueError):
        compute_inflation(df, baseline_protocol="P0")


def test_moving_block_bootstrap_ci_point_estimate_is_sample_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    point, lower, upper = moving_block_bootstrap_ci(values, block_length=2, n_resamples=500, seed=0)
    assert point == pytest.approx(3.0)
    assert lower <= point <= upper


def test_moving_block_bootstrap_ci_collapses_when_block_length_equals_n():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    point, lower, upper = moving_block_bootstrap_ci(values, block_length=4, n_resamples=200, seed=0)
    # Only one possible block (the whole series) -> every resample mean is identical.
    assert lower == pytest.approx(point)
    assert upper == pytest.approx(point)


def test_moving_block_bootstrap_ci_rejects_invalid_block_length():
    with pytest.raises(ValueError):
        moving_block_bootstrap_ci(np.array([1.0, 2.0]), block_length=5, n_resamples=10)


def test_paired_tests_with_correction_detects_difference():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        key = {"dataset_id": "d", "model_name": "m", "horizon": 1, "seed": i}
        rows.append({**key, "protocol_id": "P0", "mae": 1.0 + rng.normal(scale=0.01)})
        rows.append({**key, "protocol_id": "P1", "mae": 0.5 + rng.normal(scale=0.01)})
    df = pd.DataFrame(rows)
    result = paired_tests_with_correction(df, metric="mae", method="holm")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["significant"]
    assert row["pvalue_corrected"] < 0.05


def test_paired_tests_with_correction_skips_insufficient_pairs():
    df = _synthetic_results(n_keys=1)  # only 1 matched pair per protocol combo
    result = paired_tests_with_correction(df, metric="mae")
    assert result.empty


def test_friedman_nemenyi_detects_difference_among_protocols():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(15):
        key = {"dataset_id": "d", "model_name": "m", "horizon": 1, "seed": i}
        rows.append({**key, "protocol_id": "P0", "mae": 1.0 + rng.normal(scale=0.01)})
        rows.append({**key, "protocol_id": "P1", "mae": 0.5 + rng.normal(scale=0.01)})
        rows.append({**key, "protocol_id": "P2", "mae": 1.5 + rng.normal(scale=0.01)})
    df = pd.DataFrame(rows)
    result = friedman_nemenyi(df, metric="mae")
    assert result["friedman_pvalue"] < 0.05
    assert result["nemenyi_pvalues"].shape == (3, 3)
    assert list(result["nemenyi_pvalues"].columns) == ["P0", "P1", "P2"]


def test_friedman_nemenyi_raises_with_insufficient_blocks():
    df = _synthetic_results(n_keys=1)
    with pytest.raises(ValueError):
        friedman_nemenyi(df, metric="mae")


def test_effect_size_cliffs_delta_extremes():
    x_always_greater = np.array([10.0, 11.0, 12.0])
    y_always_smaller = np.array([1.0, 2.0, 3.0])
    assert effect_size_cliffs_delta(x_always_greater, y_always_smaller) == pytest.approx(1.0)
    assert effect_size_cliffs_delta(y_always_smaller, x_always_greater) == pytest.approx(-1.0)


def test_effect_size_cliffs_delta_zero_for_identical_distributions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert effect_size_cliffs_delta(x, x) == pytest.approx(0.0)


def test_effect_size_cliffs_delta_rejects_empty_sample():
    with pytest.raises(ValueError):
        effect_size_cliffs_delta(np.array([]), np.array([1.0]))


def test_plot_inflation_bars_saves_file(tmp_path: Path):
    df = _synthetic_results()
    inflation_df = compute_inflation(df, baseline_protocol="P0")
    out = tmp_path / "inflation.png"
    plot_inflation_bars(inflation_df, metric="mae", save_path=out)
    assert out.is_file() and out.stat().st_size > 0


def test_save_figure_for_export_saves_both_png_and_pdf(tmp_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    base = tmp_path / "myfigure.ignored_suffix"
    save_figure_for_export(fig, base)
    assert (tmp_path / "myfigure.png").is_file()
    assert (tmp_path / "myfigure.pdf").is_file()


def test_plot_inflation_heatmap_saves_both_formats(tmp_path: Path):
    pivot = pd.DataFrame({"dataset_a": [1.0, -2.0], "dataset_b": [3.0, 0.5]}, index=["P1", "P2"])
    base = tmp_path / "heatmap"
    plot_inflation_heatmap(pivot, title="test", cbar_label="inflation %", save_path=base)
    assert (tmp_path / "heatmap.png").is_file()
    assert (tmp_path / "heatmap.pdf").is_file()


def test_plot_inflation_heatmap_handles_nan_values():
    pivot = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]}, index=["P1", "P2"])
    fig = plot_inflation_heatmap(pivot, title="test", cbar_label="x")
    assert fig is not None
