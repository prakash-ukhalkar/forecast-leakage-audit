"""Tests for src.sdg_leakage_audit.protocols.splitters."""

from __future__ import annotations

import numpy as np
import pytest

from sdg_leakage_audit.protocols.splitters import (
    blocked_cv_splits,
    chronological_split_with_gap,
    random_split,
    shuffled_kfold_splits,
)


def test_chronological_split_no_overlap_and_order():
    n = 1000
    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.6, test_frac=0.2, horizon=5)
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert train_idx.max() < test_idx.min()
    assert np.all(np.diff(train_idx) == 1)
    assert np.all(np.diff(test_idx) == 1)


@pytest.mark.parametrize("horizon", [0, 1, 5, 20])
def test_chronological_split_respects_gap(horizon):
    n = 500
    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.5, test_frac=0.3, horizon=horizon)
    actual_gap = test_idx.min() - train_idx.max() - 1
    assert actual_gap >= horizon


def test_chronological_split_invalid_fractions_raise():
    with pytest.raises(ValueError):
        chronological_split_with_gap(100, train_frac=0.7, test_frac=0.5, horizon=1)


def test_random_split_no_overlap():
    n = 200
    train_idx, test_idx = random_split(n, train_frac=0.6, test_frac=0.3, seed=42)
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(train_idx) == int(round(n * 0.6))
    assert len(test_idx) == int(round(n * 0.3))


def test_random_split_is_not_chronologically_contiguous():
    # Demonstrate P1's defining property: train/test positions are interleaved,
    # unlike the chronological split.
    n = 1000
    train_idx, test_idx = random_split(n, train_frac=0.5, test_frac=0.5, seed=0)
    # If it were chronological, train.max() < test.min() or vice versa.
    interleaved = train_idx.max() > test_idx.min() and test_idx.max() > train_idx.min()
    assert interleaved


def test_blocked_cv_no_overlap_and_order():
    n = 500
    folds = blocked_cv_splits(n, n_folds=4, horizon=3, expanding=True)
    assert len(folds) > 0
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert train_idx.max() < test_idx.min()
        gap = test_idx.min() - train_idx.max() - 1
        assert gap >= 3


def test_blocked_cv_expanding_window_grows():
    n = 500
    folds = blocked_cv_splits(n, n_folds=3, horizon=1, expanding=True)
    train_sizes = [len(train) for train, _ in folds]
    assert train_sizes == sorted(train_sizes)  # non-decreasing


def test_shuffled_kfold_no_overlap_per_fold():
    n = 300
    folds = shuffled_kfold_splits(n, n_folds=5, seed=1)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(train_idx) + len(test_idx) == n


def test_shuffled_kfold_differs_from_blocked_cv_in_contiguity():
    n = 300
    kfolds = shuffled_kfold_splits(n, n_folds=5, seed=1)
    # Shuffled-kfold test indices should generally NOT be a contiguous chronological block.
    _, test_idx = kfolds[0]
    is_contiguous = np.all(np.diff(np.sort(test_idx)) == 1)
    assert not is_contiguous
