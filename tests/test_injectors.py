"""Tests for src.sdg_leakage_audit.protocols.injectors.

Each test confirms the injector actually produces the claimed leakage behavior
(or its corrected counterpart does NOT) on a tiny synthetic DataFrame/array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdg_leakage_audit.protocols.injectors import (
    fit_scaler_correct_train_only,
    fit_scaler_leaky_full_data,
    lag_feature_correct_train_fold_only,
    lag_feature_leaky_full_series,
    make_windows_correct_purged,
    make_windows_leaky_overlapping,
    rolling_feature_correct_causal,
    rolling_feature_leaky_centered,
    select_best_params_correct_on_validation,
    select_best_params_leaky_on_test,
)


def test_p2_leaky_scaler_sees_test_values():
    train = np.array([[1.0], [2.0], [3.0]])
    test = np.array([[100.0], [200.0]])
    full = np.vstack([train, test])

    leaky_scaler = fit_scaler_leaky_full_data(full)
    correct_scaler = fit_scaler_correct_train_only(train)

    # Leaky scaler's mean is influenced by the large test values; correct scaler's is not.
    assert leaky_scaler.mean_[0] > correct_scaler.mean_[0]
    assert np.isclose(correct_scaler.mean_[0], train.mean())
    assert not np.isclose(leaky_scaler.mean_[0], train.mean())


def test_p3_overlapping_windows_straddle_boundary():
    series = np.arange(20, dtype=float)
    window_size = 4
    boundary = 10

    train_starts, test_starts = make_windows_leaky_overlapping(series, window_size, boundary)
    # A window starting just before boundary extends into the test side -> leaky straddling.
    straddling_train_windows = [s for s in train_starts if s + window_size > boundary]
    assert len(straddling_train_windows) > 0

    train_starts_fixed, test_starts_fixed = make_windows_correct_purged(
        series, window_size, boundary, horizon=2
    )
    straddling_fixed = [s for s in train_starts_fixed if s + window_size > boundary]
    assert len(straddling_fixed) == 0
    # Purged test windows must start at/after boundary + horizon.
    assert all(s >= boundary + 2 for s in test_starts_fixed)


def test_p4_centered_rolling_uses_future_values():
    # Construct a series where the "future" half is a constant that would shift
    # the centered rolling mean away from the trailing rolling mean at the boundary.
    series = pd.Series([1.0] * 5 + [100.0] * 5)
    window = 3

    leaky = rolling_feature_leaky_centered(series, window)
    causal = rolling_feature_correct_causal(series, window)

    # At index 4 (last "1.0" before the jump), the centered window includes index 5
    # (value 100.0), pulling the leaky value up; the causal value should not.
    assert leaky.iloc[4] > causal.iloc[4]
    assert causal.iloc[4] == 1.0


def test_p4_leaky_lag_on_shuffled_series_breaks_temporal_meaning():
    # Correctly ordered series: lag(1) at position i should equal value at i-1.
    ordered = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    correct = lag_feature_correct_train_fold_only(ordered, lag=1)
    assert correct.iloc[2] == 20.0

    # Shuffle the same values out of temporal order, then apply the "leaky" lag:
    # it will reference the wrong row's previous-in-ARRAY (not previous-in-TIME) value.
    shuffled = ordered.sample(frac=1.0, random_state=0).reset_index(drop=True)
    leaky = lag_feature_leaky_full_series(shuffled, lag=1)
    # The leaky lag at some position will NOT generally equal (true chronological
    # predecessor), demonstrating the lag no longer tracks real time order.
    mismatches = 0
    for i in range(1, len(shuffled)):
        true_prev_value = shuffled.iloc[i] - 10.0  # since values are 10,20,...,50 spaced by 10
        if not np.isclose(leaky.iloc[i], true_prev_value):
            mismatches += 1
    assert mismatches > 0


def test_p5_leaky_selection_overfits_test_vs_validation():
    # score_fn simulates a model whose "test" score is artificially best for a
    # particular param purely by chance/noise, while true generalization (val) favors another.
    candidates = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]

    def score_on_test(params, X, y):
        # Rigged: alpha=10.0 scores best on test due to "test-set noise" (simulated).
        return {0.1: 5.0, 1.0: 3.0, 10.0: 0.5}[params["alpha"]]

    def score_on_val(params, X, y):
        # True generalization: alpha=1.0 is actually best.
        return {0.1: 4.0, 1.0: 1.0, 10.0: 6.0}[params["alpha"]]

    X_dummy, y_dummy = np.zeros((1, 1)), np.zeros(1)

    leaky_choice = select_best_params_leaky_on_test(candidates, score_on_test, X_dummy, y_dummy)
    correct_choice = select_best_params_correct_on_validation(candidates, score_on_val, X_dummy, y_dummy)

    assert leaky_choice["alpha"] == 10.0
    assert correct_choice["alpha"] == 1.0
    assert leaky_choice != correct_choice
