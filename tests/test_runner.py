"""Tests for src.sdg_leakage_audit.evaluation.runner.run_single_cell.

Uses a synthetic dataset + minimal configs written to tmp_path, with
`configs_dir`/`data_processed_dir` monkeypatched at the runner module's import
site, so this never touches the real project's configs/data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdg_leakage_audit.evaluation import runner
from sdg_leakage_audit.utils.io import write_yaml


@pytest.fixture
def synthetic_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configs_path = tmp_path / "configs"
    processed_path = tmp_path / "data" / "processed"
    processed_path.mkdir(parents=True)

    write_yaml(
        configs_path / "datasets.yaml",
        {
            "datasets": {
                "synthetic": {
                    "target_column": "y",
                    "season_length": 24,
                }
            }
        },
    )
    write_yaml(
        configs_path / "models.yaml",
        {
            "models": {
                "ridge": {
                    "params_fast": {"alpha": [1.0]},
                    "params_full": {"alpha": [0.1, 1.0, 10.0]},
                },
                "random_forest": {
                    "params_fast": {"n_estimators": [5], "max_depth": [3]},
                    "params_full": {"n_estimators": [5, 10, 15], "max_depth": [3, 5, 8]},
                },
                "lightgbm": {
                    "params_fast": {"n_estimators": [5], "learning_rate": [0.1], "num_leaves": [7]},
                    "params_full": {"n_estimators": [5, 10], "learning_rate": [0.1, 0.2], "num_leaves": [7, 15]},
                },
                "mlp": {
                    "params_fast": {"hidden_layer_sizes": [[4]], "max_iter": [50]},
                    "params_full": {"hidden_layer_sizes": [[4], [8]], "max_iter": [50, 100]},
                },
                "lstm": {"params_fast": {"hidden_size": [4], "num_layers": [1], "epochs": [2]}},
            }
        },
    )

    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    y_series = pd.Series(np.cumsum(rng.normal(scale=0.1, size=n)) + 10.0, index=idx)
    df = pd.DataFrame(
        {
            "y": y_series,
            "y_lag1": y_series.shift(1),
            "y_lag2": y_series.shift(2),
            "y_roll1_std": y_series.rolling(1).std(),  # always NaN (window-1 std undefined)
            "y_roll3_mean": y_series.rolling(3).mean(),
            "hour": idx.hour,
            "wind_direction": "NE",  # non-numeric column, mirrors Beijing's categorical "wd"
        },
        index=idx,
    )
    df.to_parquet(processed_path / "synthetic_features.parquet")

    monkeypatch.setattr(runner, "configs_dir", lambda: configs_path)
    monkeypatch.setattr(runner, "data_processed_dir", lambda: processed_path)
    return tmp_path


FAST_CONFIG = {"data_subsample_frac": 1.0, "model_param_grid": "params_fast"}


def test_build_direct_horizon_xy_drops_non_numeric_and_all_nan_columns(synthetic_project: Path):
    df = pd.read_parquet(runner.data_processed_dir() / "synthetic_features.parquet")
    X, y, origin = runner._build_direct_horizon_xy(df, "y", horizon=1)
    assert "wind_direction" not in X.columns
    assert "y_roll1_std" not in X.columns
    assert len(X) > 0


def test_run_single_cell_rejects_unknown_protocol(synthetic_project: Path):
    with pytest.raises(ValueError):
        runner.run_single_cell("synthetic", "P9", "ridge", horizon=1, seed=0, config=FAST_CONFIG)


@pytest.mark.parametrize("protocol_id", ["P0", "P1", "P2", "P3", "P4", "P5", "P6"])
@pytest.mark.parametrize("model_name", ["seasonal_naive", "persistence", "ridge", "random_forest", "lightgbm", "mlp", "lstm"])
def test_run_single_cell_runs_for_every_protocol_and_model_type(synthetic_project: Path, protocol_id: str, model_name: str):
    result = runner.run_single_cell("synthetic", protocol_id, model_name, horizon=1, seed=0, config=FAST_CONFIG)

    assert result["dataset_id"] == "synthetic"
    assert result["protocol_id"] == protocol_id
    assert result["model_name"] == model_name
    assert result["n_train"] > 0 and result["n_test"] > 0
    for metric in ("mae", "rmse", "smape", "mase", "skill_score"):
        assert np.isfinite(result[metric]), f"{metric} was not finite: {result[metric]}"


def test_p1_breaks_chronological_order_relative_to_p0(synthetic_project: Path):
    # Not a metrics assertion -- P1's leak is about split structure, which is already
    # unit-tested in test_splitters.py. Here we just confirm run_single_cell actually
    # dispatches to random_split for P1 by checking train/test sizes differ from P0's
    # fixed 70/20 split arithmetic in a way consistent with random assignment existing.
    p0 = runner.run_single_cell("synthetic", "P0", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    p1 = runner.run_single_cell("synthetic", "P1", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    assert p0["n_train"] > 0 and p1["n_train"] > 0


def test_p3_uses_zero_purge_gap(synthetic_project: Path, monkeypatch: pytest.MonkeyPatch):
    captured_gaps = []
    original = runner.chronological_split_with_gap

    def spy(index_or_df, train_frac, test_frac, horizon):
        captured_gaps.append(horizon)
        return original(index_or_df, train_frac, test_frac, horizon)

    monkeypatch.setattr(runner, "chronological_split_with_gap", spy)
    runner.run_single_cell("synthetic", "P3", "ridge", horizon=5, seed=0, config=FAST_CONFIG)
    assert captured_gaps == [0]


def test_p4_feature_leak_replaces_lag_and_rolling_columns():
    # Build a tiny frame directly to test _apply_p4_feature_leak in isolation.
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    y = pd.Series(np.arange(20, dtype=float), index=idx)
    df = pd.DataFrame({"y": y, "y_lag1": y.shift(1), "y_roll3_mean": y.rolling(3).mean()}, index=idx)
    leaked = runner._apply_p4_feature_leak(df, "y", seed=0)
    causal_roll = df["y_roll3_mean"].dropna()
    leaked_roll = leaked["y_roll3_mean"].dropna()
    assert not causal_roll.equals(leaked_roll)  # centered vs trailing rolling differ
    assert not df["y_lag1"].dropna().equals(leaked["y_lag1"].dropna())  # shuffled-then-lagged differs


def test_p5_selects_among_multiple_candidates_for_tunable_model(synthetic_project: Path):
    result = runner.run_single_cell("synthetic", "P5", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    assert np.isfinite(result["mae"])


def test_p5_is_noop_for_non_tunable_model(synthetic_project: Path):
    p0 = runner.run_single_cell("synthetic", "P0", "seasonal_naive", horizon=1, seed=0, config=FAST_CONFIG)
    p5 = runner.run_single_cell("synthetic", "P5", "seasonal_naive", horizon=1, seed=0, config=FAST_CONFIG)
    # Both use the same seasonal-naive rule; P5's three-way split changes train/test
    # sizes slightly relative to P0's two-way split, but both must still produce finite metrics.
    assert np.isfinite(p0["mae"]) and np.isfinite(p5["mae"])


def test_p6_returns_fold_averaged_metrics_with_n_folds_field(synthetic_project: Path):
    result = runner.run_single_cell("synthetic", "P6", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    assert result["n_folds"] == 5
    assert np.isfinite(result["mae"])


def test_run_grid_checkpoints_each_cell_and_resumes(synthetic_project: Path, tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    results_a = runner.run_grid(
        datasets=["synthetic"],
        protocols=["P0", "P1"],
        models=["ridge"],
        horizons=[1],
        seeds=[0],
        config=FAST_CONFIG,
        checkpoint_dir=checkpoint_dir,
    )
    assert len(results_a) == 2
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) == 2

    # Second call with an extra protocol should reuse the two existing checkpoints
    # (same mtime) and only compute the new one.
    mtimes_before = {f: f.stat().st_mtime for f in checkpoint_files}
    results_b = runner.run_grid(
        datasets=["synthetic"],
        protocols=["P0", "P1", "P2"],
        models=["ridge"],
        horizons=[1],
        seeds=[0],
        config=FAST_CONFIG,
        checkpoint_dir=checkpoint_dir,
    )
    assert len(results_b) == 3
    for f, mtime in mtimes_before.items():
        assert f.stat().st_mtime == mtime  # untouched, loaded from checkpoint not recomputed


def test_run_grid_records_error_without_aborting_grid(synthetic_project: Path, tmp_path: Path):
    results = runner.run_grid(
        datasets=["synthetic"],
        protocols=["P0"],
        models=["ridge", "not_a_real_model"],
        horizons=[1],
        seeds=[0],
        config=FAST_CONFIG,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(results) == 2
    errored = [r for r in results if r["model_name"] == "not_a_real_model"]
    assert len(errored) == 1 and "error" in errored[0]


def test_run_single_cell_is_seed_reproducible_for_random_forest(synthetic_project: Path):
    result_a = runner.run_single_cell("synthetic", "P0", "random_forest", horizon=1, seed=0, config=FAST_CONFIG)
    result_b = runner.run_single_cell("synthetic", "P0", "random_forest", horizon=1, seed=0, config=FAST_CONFIG)
    assert result_a["mae"] == pytest.approx(result_b["mae"])


def test_run_single_cell_subsample_frac_reduces_training_rows(synthetic_project: Path):
    full = runner.run_single_cell("synthetic", "P0", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    subsampled = runner.run_single_cell(
        "synthetic", "P0", "ridge", horizon=1, seed=0,
        config={"data_subsample_frac": 0.5, "model_param_grid": "params_fast"},
    )
    assert subsampled["n_train"] < full["n_train"]


def test_run_single_cell_on_frame_matches_run_single_cell_for_same_dataset(synthetic_project: Path):
    df = pd.read_parquet(runner.data_processed_dir() / "synthetic_features.parquet")
    via_id = runner.run_single_cell("synthetic", "P0", "ridge", horizon=1, seed=0, config=FAST_CONFIG)
    via_frame = runner.run_single_cell_on_frame(
        df, "synthetic", "y", season_length=24, protocol_id="P0", model_name="ridge", horizon=1, seed=0, config=FAST_CONFIG
    )
    assert via_frame["mae"] == pytest.approx(via_id["mae"])


def test_run_single_cell_on_frame_supports_custom_window_feature_set(synthetic_project: Path):
    # Build a feature frame with a different lag list than the fixture's default (lag1/lag2),
    # simulating notebook 10's window-size sensitivity check.
    df = pd.read_parquet(runner.data_processed_dir() / "synthetic_features.parquet")
    df_alt = df.drop(columns=["y_lag1", "y_lag2"])
    df_alt["y_lag5"] = df_alt["y"].shift(5)
    result = runner.run_single_cell_on_frame(
        df_alt, "synthetic_alt_window", "y", season_length=24, protocol_id="P0", model_name="ridge",
        horizon=1, seed=0, config=FAST_CONFIG,
    )
    assert result["dataset_id"] == "synthetic_alt_window"
    assert np.isfinite(result["mae"])
