"""Translate forecast-error inflation into SDG decision-relevant terms.

Shared assumption across all three decision contexts below: a safety buffer is
sized proportional to forecast RMSE (`buffer = factor * rmse`), a common pattern
in operational forecasting (e.g. load-reserve margins, alert thresholds, capacity
headroom). The `factor` constants used here are illustrative, not calibrated
real-world standards -- they are explicitly documented as assumptions on each
function and restated in notebook 09's "Stated assumptions" section, per the
project's "state assumptions explicitly; do not overclaim" requirement.

For each decision context, "true" buffer uses the P0 (corrected) RMSE and
"apparent" buffer uses the leaky protocol's RMSE. Since a leaky protocol's RMSE
is, on average, lower (looks better) than P0's true RMSE whenever leakage
inflates apparent performance, the apparent buffer is undersized relative to the
true buffer -- `under_provisioning_pct` quantifies that gap.
"""

from __future__ import annotations

import pandas as pd

_KEY_COLS = ["dataset_id", "protocol_id"]


def _estimate_safety_buffer_impact(
    inflation_df: pd.DataFrame,
    dataset_id: str,
    buffer_factor: float,
    decision_label: str,
) -> pd.DataFrame:
    """Shared helper: buffer = buffer_factor * rmse, compared true (P0) vs apparent.

    Args:
        inflation_df: Output of `evaluation.inflation.compute_inflation` (must
            contain `rmse` and `rmse_baseline` columns).
        dataset_id: Dataset id to filter to.
        buffer_factor: Multiplier applied to RMSE to obtain the safety buffer.
        decision_label: Human-readable label for the decision context, stored in
            the output (e.g. "reserve_margin").

    Returns:
        DataFrame indexed by protocol_id with columns: decision_label,
        rmse_true, rmse_apparent, buffer_true, buffer_apparent,
        under_provisioning_abs, under_provisioning_pct (mean across model/horizon/seed
        cells matching `dataset_id`).
    """
    subset = inflation_df[inflation_df["dataset_id"] == dataset_id]
    if subset.empty:
        raise ValueError(f"No inflation rows found for dataset_id={dataset_id!r}")

    grouped = subset.groupby("protocol_id")[["rmse", "rmse_baseline"]].mean()
    grouped = grouped.rename(columns={"rmse": "rmse_apparent", "rmse_baseline": "rmse_true"})

    grouped["buffer_true"] = buffer_factor * grouped["rmse_true"]
    grouped["buffer_apparent"] = buffer_factor * grouped["rmse_apparent"]
    grouped["under_provisioning_abs"] = grouped["buffer_true"] - grouped["buffer_apparent"]
    grouped["under_provisioning_pct"] = grouped["under_provisioning_abs"] / grouped["buffer_true"] * 100.0
    grouped.insert(0, "decision_label", decision_label)

    return grouped


def estimate_reserve_margin_impact(
    inflation_df: pd.DataFrame, dataset_id: str, reserve_margin_factor: float = 2.0
) -> pd.DataFrame:
    """Estimate the effect of metric inflation on energy reserve-margin decisions.

    Assumption: grid operators size the reserve margin (spare generation/demand-
    response capacity held to cover forecast error) at `reserve_margin_factor`
    times the load-forecast RMSE. A leaky protocol's smaller apparent RMSE would
    lead to an undersized reserve margin relative to the margin P0's true RMSE
    implies -- `under_provisioning_pct` is the resulting capacity shortfall.

    Args:
        inflation_df: Output of `evaluation.inflation.compute_inflation`.
        dataset_id: An SDG7 energy dataset id (e.g. "household_power_consumption").
        reserve_margin_factor: Multiplier applied to RMSE (default 2.0, illustrative).

    Returns:
        DataFrame summarizing the decision-relevant impact (see `_estimate_safety_buffer_impact`).
    """
    return _estimate_safety_buffer_impact(inflation_df, dataset_id, reserve_margin_factor, "reserve_margin")


def estimate_exceedance_alert_impact(
    inflation_df: pd.DataFrame, dataset_id: str, alert_buffer_factor: float = 1.645
) -> pd.DataFrame:
    """Estimate the effect of metric inflation on air-quality exceedance alerting.

    Assumption: an exceedance-alert system adds a safety buffer of
    `alert_buffer_factor` times the pollutant-forecast RMSE above a health
    threshold before triggering an alert (1.645 approximates a one-sided 95%
    normal-error margin, a common convention -- illustrative here, not a
    calibrated regulatory standard). A leaky protocol's smaller apparent RMSE
    understates the buffer actually needed, raising missed-alert (false
    negative) risk when the true error (P0's RMSE) is larger.

    Args:
        inflation_df: Output of `evaluation.inflation.compute_inflation`.
        dataset_id: An SDG3/11 air-quality dataset id (e.g. "air_quality_uci").
        alert_buffer_factor: Multiplier applied to RMSE (default 1.645, illustrative).

    Returns:
        DataFrame summarizing the decision-relevant impact (see `_estimate_safety_buffer_impact`).
    """
    return _estimate_safety_buffer_impact(inflation_df, dataset_id, alert_buffer_factor, "exceedance_alert_buffer")


def estimate_storage_sizing_impact(
    inflation_df: pd.DataFrame, dataset_id: str, storage_buffer_factor: float = 2.0
) -> pd.DataFrame:
    """Estimate the effect of metric inflation on thermal/storage capacity sizing.

    Assumption: capacity planning (e.g. thermal storage, HVAC headroom, battery
    sizing) adds a buffer of `storage_buffer_factor` times the climate/load
    forecast RMSE to cover forecast error. A leaky protocol's smaller apparent
    RMSE leads to undersized capacity relative to what P0's true RMSE implies.

    Args:
        inflation_df: Output of `evaluation.inflation.compute_inflation`.
        dataset_id: A climate/energy dataset id (e.g. "jena_climate").
        storage_buffer_factor: Multiplier applied to RMSE (default 2.0, illustrative).

    Returns:
        DataFrame summarizing the decision-relevant impact (see `_estimate_safety_buffer_impact`).
    """
    return _estimate_safety_buffer_impact(inflation_df, dataset_id, storage_buffer_factor, "storage_capacity_buffer")
