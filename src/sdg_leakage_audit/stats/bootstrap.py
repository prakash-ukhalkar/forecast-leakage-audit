"""Moving-block bootstrap for confidence intervals on time-series metrics.

A moving-block bootstrap (rather than an i.i.d. bootstrap) is used because
forecast errors are serially correlated: resampling individual points
independently would understate the true sampling variance of their mean.
"""

from __future__ import annotations

import numpy as np


def moving_block_bootstrap_ci(
    values: np.ndarray,
    block_length: int,
    n_resamples: int,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute a moving-block bootstrap confidence interval for the mean of `values`.

    Each resample is built by drawing contiguous blocks of length `block_length`
    (with replacement, starting positions uniform over `[0, n - block_length]`)
    until the resample reaches at least `len(values)` points, then truncating to
    exactly `len(values)`. This preserves local serial correlation within each
    block, unlike an i.i.d. bootstrap.

    Args:
        values: 1-D array of (serially correlated) per-timestep values, e.g.
            forecast errors.
        block_length: Length of contiguous blocks resampled with replacement.
            Must be in `[1, len(values)]`.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval (e.g. 0.95).
        seed: RNG seed.

    Returns:
        Tuple `(point_estimate, ci_lower, ci_upper)`, where `point_estimate` is
        `values.mean()` and the CI bounds are the appropriate percentiles of the
        bootstrap resample means.

    Raises:
        ValueError: If `block_length` is not in `[1, len(values)]`, or `values`
            is empty.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    if not (1 <= block_length <= n):
        raise ValueError(f"block_length must be in [1, {n}], got {block_length}")

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_length))
    max_start = n - block_length

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        resample = np.concatenate([values[s : s + block_length] for s in starts])[:n]
        boot_means[i] = resample.mean()

    alpha = 1.0 - confidence_level
    ci_lower = float(np.quantile(boot_means, alpha / 2.0))
    ci_upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return float(values.mean()), ci_lower, ci_upper
