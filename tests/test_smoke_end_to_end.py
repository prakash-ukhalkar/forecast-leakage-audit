"""Fast end-to-end smoke test: tiny synthetic series -> P0 split -> persistence -> metrics.

Confirms the pipeline wires together without exceptions. Must run in <5s.
"""

from __future__ import annotations

import numpy as np

from sdg_leakage_audit.evaluation.metrics import mae, rmse
from sdg_leakage_audit.protocols.splitters import chronological_split_with_gap


def test_smoke_end_to_end_pipeline():
    rng = np.random.default_rng(0)
    n = 100
    horizon = 3
    series = np.cumsum(rng.normal(size=n)) + 50.0

    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.7, test_frac=0.2, horizon=horizon)
    assert len(np.intersect1d(train_idx, test_idx)) == 0

    train_series = series[train_idx]
    test_series = series[test_idx]

    # Trivial "persistence" model: forecast = last train value, repeated.
    naive_forecast = np.full(shape=test_series.shape, fill_value=train_series[-1])

    pipeline_mae = mae(test_series, naive_forecast)
    pipeline_rmse = rmse(test_series, naive_forecast)

    assert np.isfinite(pipeline_mae)
    assert np.isfinite(pipeline_rmse)
    assert pipeline_mae >= 0
    assert pipeline_rmse >= 0
