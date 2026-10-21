"""Train/test (and CV) splitters implementing the P0 corrected baseline, the P1
random-split protocol error, and the P6 shuffled-K-fold vs blocked/expanding-window
CV comparison.

All functions operate on a size `n` (number of rows / windows) and return integer
positional index arrays, so they are agnostic to whether the caller passes a plain
range, a DataFrame's RangeIndex, or a list of window ids -- the caller maps returned
positions back onto its own index/DataFrame.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_n(index_or_df) -> int:
    """Resolve a length from either an integer, a sized object, or a DataFrame."""
    if isinstance(index_or_df, int):
        return index_or_df
    return len(index_or_df)


def chronological_split_with_gap(
    index_or_df,
    train_frac: float,
    test_frac: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """P0: chronological train/test split with a purge gap >= horizon.

    Splits the first `train_frac` fraction of positions into the train set and the
    last `test_frac` fraction into the test set, in original temporal order, with a
    purge gap of exactly `max(horizon, 1)` positions dropped between them so that no
    train-set feature window (which may look `horizon` steps ahead to build a label)
    can overlap with the test set, and vice versa. This is the leakage-safe baseline
    split used by protocol P0.

    Args:
        index_or_df: Either an integer length `n`, or any sized/indexable object
            (e.g. a DataFrame, Index, array) whose length is used as `n`.
        train_frac: Fraction of rows (by position) assigned to train, in (0, 1).
        test_frac: Fraction of rows (by position) assigned to test, in (0, 1).
        horizon: Forecast horizon in steps; the purge gap will be at least this size.

    Returns:
        Tuple `(train_idx, test_idx)` of integer position arrays, strictly
        increasing, non-overlapping, with at least `horizon` positions strictly
        between `train_idx.max()` and `test_idx.min()`.

    Raises:
        ValueError: If `train_frac + test_frac > 1`, fractions are not in (0, 1],
            horizon is negative, or the resulting train/test sets would be empty.
    """
    n = _resolve_n(index_or_df)
    if not (0 < train_frac <= 1) or not (0 < test_frac <= 1):
        raise ValueError("train_frac and test_frac must be in (0, 1]")
    if train_frac + test_frac > 1:
        raise ValueError("train_frac + test_frac must be <= 1")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    gap = max(horizon, 0)
    n_train = int(round(n * train_frac))
    n_test = int(round(n * test_frac))

    train_end = n_train
    test_start = train_end + gap
    test_end = test_start + n_test

    if test_end > n:
        # Shrink test set to fit rather than overflow past n.
        test_end = n
        n_test = test_end - test_start

    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"Resulting split is empty: n_train={n_train}, n_test={n_test} "
            f"(n={n}, gap={gap}); reduce fractions/horizon or increase n"
        )

    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, test_end)

    assert train_idx.max() + gap <= test_idx.min() if gap > 0 else True
    assert len(np.intersect1d(train_idx, test_idx)) == 0

    logger.debug(
        "chronological_split_with_gap: n=%d train=%d test=%d gap=%d", n, len(train_idx), len(test_idx), gap
    )
    return train_idx, test_idx


def random_split(
    index_or_df,
    train_frac: float,
    test_frac: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """P1 (WRONG, under audit): random/shuffled split of rows, ignoring time order.

    This deliberately violates the temporal structure of the data: train and test
    positions are drawn from a random shuffle of all positions, so a test-set row
    can be temporally adjacent to (even immediately next to) a train-set row,
    letting a model exploit short-range autocorrelation it would not have access to
    under a genuine forecasting deployment. Provided here, with this clearly-labeled
    name and docstring, purely so protocol P1's inflation effect can be measured
    against the corrected `chronological_split_with_gap` baseline.

    Args:
        index_or_df: Either an integer length `n`, or a sized/indexable object.
        train_frac: Fraction of rows assigned to train, in (0, 1).
        test_frac: Fraction of rows assigned to test, in (0, 1).
        seed: RNG seed for the shuffle.

    Returns:
        Tuple `(train_idx, test_idx)` of integer position arrays. Non-overlapping,
        but NOT in temporal order and NOT purged -- this is intentional.

    Raises:
        ValueError: If fractions are invalid.
    """
    n = _resolve_n(index_or_df)
    if not (0 < train_frac <= 1) or not (0 < test_frac <= 1) or train_frac + test_frac > 1:
        raise ValueError("Invalid train_frac/test_frac")

    rng = default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * train_frac))
    n_test = int(round(n * test_frac))

    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train : n_train + n_test])

    assert len(np.intersect1d(train_idx, test_idx)) == 0
    logger.debug("random_split (P1): n=%d train=%d test=%d seed=%d", n, len(train_idx), len(test_idx), seed)
    return train_idx, test_idx


def blocked_cv_splits(
    index_or_df,
    n_folds: int,
    horizon: int,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Corrected CV: blocked or expanding-window cross-validation.

    Produces `n_folds` chronologically ordered (train, test) fold pairs. In
    "expanding" mode, each successive fold's train set grows to include all prior
    data (expanding-window CV); in "blocked" mode (`expanding=False`), each fold's
    train set is a fixed-size contiguous block immediately preceding its test block.
    In both modes, a purge gap of `horizon` positions is enforced between each
    fold's train and test blocks, and test blocks never overlap with train blocks.

    Args:
        index_or_df: Either an integer length `n`, or a sized/indexable object.
        n_folds: Number of CV folds to produce.
        horizon: Forecast horizon in steps; purge gap size.
        expanding: If True, use expanding-window train sets; if False, use
            fixed-size blocked train sets.

    Returns:
        List of `(train_idx, test_idx)` tuples, one per fold, in chronological order.

    Raises:
        ValueError: If `n_folds < 1` or the data is too short for the requested
            number of folds.
    """
    n = _resolve_n(index_or_df)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    gap = max(horizon, 0)
    # Reserve n_folds equal-size test blocks at the tail; train grows/blocks precede each.
    test_block_size = n // (n_folds + 1)
    if test_block_size < 1:
        raise ValueError(f"n={n} too small for n_folds={n_folds} with a positive test block size")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_folds):
        test_start = (fold_i + 1) * test_block_size + gap
        test_end = min(test_start + test_block_size, n)
        if test_start >= test_end:
            continue

        train_end = (fold_i + 1) * test_block_size  # ends before the gap, by construction
        if expanding:
            train_start = 0
        else:
            train_start = max(0, train_end - test_block_size)

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        assert len(np.intersect1d(train_idx, test_idx)) == 0
        folds.append((train_idx, test_idx))

    logger.debug("blocked_cv_splits: n=%d n_folds=%d expanding=%s -> %d folds produced", n, n_folds, expanding, len(folds))
    return folds


def shuffled_kfold_splits(
    index_or_df,
    n_folds: int,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """P6 (WRONG, under audit): naive shuffled K-fold CV, ignoring time order.

    Standard scikit-learn-style K-fold: positions are shuffled once, then split into
    `n_folds` equal contiguous chunks of the shuffled order; each fold holds one chunk
    out as "test" and uses the rest as "train". Because the shuffle ignores time
    order, a given fold's train set can contain positions immediately
    before/after its test set in real time, leaking local temporal structure. This
    is provided to measure the CV-estimate inflation of protocol P6 against
    `blocked_cv_splits`.

    Args:
        index_or_df: Either an integer length `n`, or a sized/indexable object.
        n_folds: Number of folds.
        seed: RNG seed for the single shuffle applied before folding.

    Returns:
        List of `(train_idx, test_idx)` tuples, one per fold. Each fold's train/test
        are disjoint, but neither is in temporal order nor purged -- intentional.

    Raises:
        ValueError: If `n_folds < 2`.
    """
    n = _resolve_n(index_or_df)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    rng = default_rng(seed)
    perm = rng.permutation(n)
    fold_assignment = np.array_split(perm, n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_folds):
        test_idx = np.sort(fold_assignment[fold_i])
        train_idx = np.sort(np.concatenate([fold_assignment[j] for j in range(n_folds) if j != fold_i]))
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        folds.append((train_idx, test_idx))

    logger.debug("shuffled_kfold_splits (P6): n=%d n_folds=%d seed=%d", n, n_folds, seed)
    return folds
