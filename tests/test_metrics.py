"""Tests for src.sdg_leakage_audit.evaluation.metrics."""

from __future__ import annotations

import numpy as np

from sdg_leakage_audit.evaluation.metrics import (
    mae,
    mase,
    rmse,
    skill_score_vs_seasonal_naive,
    smape,
)


def test_mae_hand_computed():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]
    assert np.isclose(mae(y_true, y_pred), (0 + 0 + 2) / 3)


def test_rmse_hand_computed():
    y_true = [0.0, 0.0]
    y_pred = [3.0, 4.0]
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((9 + 16) / 2))


def test_smape_hand_computed():
    y_true = [10.0]
    y_pred = [12.0]
    expected = (abs(10 - 12) / (abs(10) + abs(12))) * 200.0
    assert np.isclose(smape(y_true, y_pred), expected)


def test_smape_handles_zero_denominator():
    y_true = [0.0]
    y_pred = [0.0]
    result = smape(y_true, y_pred)
    assert np.isnan(result)


def test_mase_hand_computed():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # naive diffs all = 1 -> scale = 1
    y_true = [10.0, 10.0]
    y_pred = [11.0, 8.0]
    # MAE(pred) = (1 + 2) / 2 = 1.5; scale = 1.0 -> MASE = 1.5
    assert np.isclose(mase(y_true, y_pred, y_train, seasonality=1), 1.5)


def test_mase_degenerate_constant_train_returns_nan():
    y_train = np.array([5.0, 5.0, 5.0, 5.0])
    result = mase([5.0], [6.0], y_train, seasonality=1)
    assert np.isnan(result)


def test_skill_score_positive_when_better_than_naive():
    y_true = [10.0, 10.0, 10.0]
    y_pred = [10.0, 10.0, 10.0]  # perfect
    y_naive = [12.0, 8.0, 14.0]  # naive has error
    score = skill_score_vs_seasonal_naive(y_true, y_pred, y_naive)
    assert score == 1.0


def test_skill_score_zero_when_equal_to_naive():
    y_true = [10.0, 20.0]
    y_pred = [11.0, 19.0]
    y_naive = [11.0, 19.0]
    score = skill_score_vs_seasonal_naive(y_true, y_pred, y_naive)
    assert np.isclose(score, 0.0)
