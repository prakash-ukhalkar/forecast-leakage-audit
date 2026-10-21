"""Automated leakage guard: P0's scaler/imputer must never be fit using test-fold data.

This is the explicit "automated leakage guard" the project brief calls for.
"""

from __future__ import annotations

import numpy as np

from sdg_leakage_audit.protocols.injectors import (
    fit_imputer_correct_train_only,
    fit_scaler_correct_train_only,
)
from sdg_leakage_audit.protocols.splitters import chronological_split_with_gap


def test_p0_scaler_never_sees_test_fold_values():
    n = 200
    horizon = 5
    data = np.arange(n, dtype=float).reshape(-1, 1)

    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.6, test_frac=0.3, horizon=horizon)
    train_data = data[train_idx]
    test_data = data[test_idx]

    scaler = fit_scaler_correct_train_only(train_data)

    # The scaler's fitted mean must equal the train-only mean, and must NOT equal
    # the mean of train+test combined (which would indicate test-fold contamination).
    assert np.isclose(scaler.mean_[0], train_data.mean())
    combined_mean = np.vstack([train_data, test_data]).mean()
    assert not np.isclose(scaler.mean_[0], combined_mean)

    # The fitted scaler object must carry no reference to test_data at all.
    assert scaler.n_samples_seen_ == len(train_data)


def test_p0_imputer_never_sees_test_fold_values():
    n = 200
    horizon = 5
    data = np.arange(n, dtype=float).reshape(-1, 1)
    data[0, 0] = np.nan  # introduce a missing value in train

    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.6, test_frac=0.3, horizon=horizon)
    train_data = data[train_idx]
    test_data = data[test_idx]

    imputer = fit_imputer_correct_train_only(train_data, strategy="mean")

    train_mean_ignoring_nan = np.nanmean(train_data)
    combined_mean_ignoring_nan = np.nanmean(np.vstack([train_data, test_data]))

    assert np.isclose(imputer.statistics_[0], train_mean_ignoring_nan)
    assert not np.isclose(imputer.statistics_[0], combined_mean_ignoring_nan)
