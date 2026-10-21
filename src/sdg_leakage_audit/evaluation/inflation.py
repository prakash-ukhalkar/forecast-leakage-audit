"""Inflation-effect computation: compare leaky-protocol metrics against P0 baseline.

"Inflation" is always signed so that positive = apparent improvement from the
leak (the leaky protocol looks better than the honest P0 baseline), regardless of
whether the underlying metric is lower-is-better (mae/rmse/smape/mase) or
higher-is-better (skill_score).
"""

from __future__ import annotations

import pandas as pd

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)

_KEY_COLS = ["dataset_id", "model_name", "horizon", "seed"]
_LOWER_IS_BETTER_METRICS = ("mae", "rmse", "smape", "mase")
_HIGHER_IS_BETTER_METRICS = ("skill_score",)
_ALL_METRICS = _LOWER_IS_BETTER_METRICS + _HIGHER_IS_BETTER_METRICS


def compute_inflation(results_df: pd.DataFrame, baseline_protocol: str = "P0") -> pd.DataFrame:
    """Compute absolute and percentage inflation of each protocol vs the baseline.

    Each row is matched to its corresponding `baseline_protocol` row by
    (dataset_id, model_name, horizon, seed). For lower-is-better metrics
    (mae, rmse, smape, mase), `inflation_abs = baseline_value - protocol_value`,
    so a leaky protocol that *reduces* error relative to P0 gets positive
    inflation. For skill_score (higher is better), the sign is flipped:
    `inflation_abs = protocol_value - baseline_value`. In both cases, positive
    inflation means "this protocol's error looks more flattering than it should."

    Args:
        results_df: Long-format DataFrame of per-cell results (as produced by
            `evaluation.runner.run_grid`), with at least columns: dataset_id,
            protocol_id, model_name, horizon, seed, and metric columns (mae, rmse,
            smape, mase, skill_score).
        baseline_protocol: Protocol id to treat as the ground-truth baseline
            (default "P0").

    Returns:
        A copy of `results_df` (restricted to rows with a matching baseline row)
        with added columns `{metric}_inflation_abs` and `{metric}_inflation_pct`
        for each metric present in both `results_df` and the known metric set.

    Raises:
        ValueError: If `baseline_protocol` has no rows in `results_df`.
    """
    baseline_df = results_df[results_df["protocol_id"] == baseline_protocol]
    if baseline_df.empty:
        raise ValueError(f"No rows found for baseline_protocol={baseline_protocol!r}")

    metrics = [m for m in _ALL_METRICS if m in results_df.columns]
    baseline_renamed = baseline_df[[*_KEY_COLS, *metrics]].rename(columns={m: f"{m}_baseline" for m in metrics})

    merged = results_df.merge(baseline_renamed, on=_KEY_COLS, how="inner")
    n_dropped = len(results_df) - len(merged)
    if n_dropped > 0:
        logger.warning("compute_inflation: %d row(s) had no matching %s baseline and were dropped", n_dropped, baseline_protocol)

    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        if metric in _LOWER_IS_BETTER_METRICS:
            merged[f"{metric}_inflation_abs"] = merged[baseline_col] - merged[metric]
        else:
            merged[f"{metric}_inflation_abs"] = merged[metric] - merged[baseline_col]
        merged[f"{metric}_inflation_pct"] = merged[f"{metric}_inflation_abs"] / merged[baseline_col].abs() * 100.0

    return merged
