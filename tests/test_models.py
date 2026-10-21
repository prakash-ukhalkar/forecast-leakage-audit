"""Tests for src.sdg_leakage_audit.models (baselines, sklearn_models, torch_models, factory)."""

from __future__ import annotations

import numpy as np
import pytest

from sdg_leakage_audit.models.baselines import PersistenceForecaster, SeasonalNaiveForecaster
from sdg_leakage_audit.models.factory import build_model, get_model_family
from sdg_leakage_audit.models.sklearn_models import (
    build_lightgbm,
    build_mlp,
    build_random_forest,
    build_ridge,
)
from sdg_leakage_audit.models.torch_models import SmallGRUForecaster, SmallLSTMForecaster


def test_seasonal_naive_forecaster_uses_value_one_season_before_target():
    history = np.arange(10, dtype=float)  # [0, 1, ..., 9]
    forecaster = SeasonalNaiveForecaster(season_length=4)
    forecast = forecaster.predict(history, horizon=1)
    # forecast[i] should equal history[i - (season_length - horizon)] = history[i - 3]
    assert np.isnan(forecast[:3]).all()
    assert forecast[3] == history[0]
    assert forecast[9] == history[6]


def test_seasonal_naive_forecaster_rejects_horizon_exceeding_season_length():
    forecaster = SeasonalNaiveForecaster(season_length=2)
    with pytest.raises(ValueError):
        forecaster.predict(np.arange(10, dtype=float), horizon=5)


def test_seasonal_naive_forecaster_handles_season_equal_horizon():
    history = np.arange(5, dtype=float)
    forecaster = SeasonalNaiveForecaster(season_length=3)
    forecast = forecaster.predict(history, horizon=3)
    assert np.array_equal(forecast, history)


def test_persistence_forecaster_repeats_origin_value():
    history = np.array([1.0, 2.0, 3.0])
    forecast = PersistenceForecaster().predict(history, horizon=5)
    assert np.array_equal(forecast, history)


def test_build_ridge_returns_fittable_estimator():
    model = build_ridge(alpha=1.0)
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = X @ np.array([1.0, 2.0, -1.0])
    model.fit(X, y)
    assert model.predict(X).shape == (20,)


def test_build_random_forest_defaults_n_jobs():
    model = build_random_forest(n_estimators=5, max_depth=3)
    assert model.n_jobs == -1


def test_build_lightgbm_defaults_silent():
    model = build_lightgbm(n_estimators=5)
    assert model.verbosity == -1


def test_build_mlp_defaults_early_stopping():
    model = build_mlp(hidden_layer_sizes=(4,), max_iter=50)
    assert model.early_stopping is True


def test_small_lstm_forecaster_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3, 1)).astype("float32")
    y = X[:, -1, 0] * 2.0
    model = SmallLSTMForecaster(hidden_size=4, num_layers=1)
    model.fit(X, y, epochs=2)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_small_gru_forecaster_fits_and_predicts():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3, 1)).astype("float32")
    y = X[:, -1, 0] * 2.0
    model = SmallGRUForecaster(hidden_size=4, num_layers=1)
    model.fit(X, y, epochs=2)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_small_lstm_forecaster_predict_before_fit_raises():
    model = SmallLSTMForecaster(hidden_size=4)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 3, 1)))


def test_build_model_dispatches_seasonal_naive():
    model = build_model("seasonal_naive", {"season_length": 24})
    assert isinstance(model, SeasonalNaiveForecaster)


def test_build_model_dispatches_lstm_and_strips_epochs_param():
    model = build_model("lstm", {"hidden_size": 8, "epochs": 5})
    assert isinstance(model, SmallLSTMForecaster)
    assert model.hidden_size == 8


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("not_a_model")


@pytest.mark.parametrize(
    "model_name,expected_family",
    [
        ("seasonal_naive", "baseline"),
        ("persistence", "baseline"),
        ("ridge", "linear"),
        ("random_forest", "tree_ensemble"),
        ("lightgbm", "tree_ensemble"),
        ("mlp", "neural"),
        ("lstm", "neural"),
        ("gru", "neural"),
    ],
)
def test_get_model_family(model_name: str, expected_family: str):
    assert get_model_family(model_name) == expected_family


def test_get_model_family_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_model_family("not_a_model")
