"""Tests for src.sdg_leakage_audit.sdg (mapping, impact)."""

from __future__ import annotations

import pandas as pd
import pytest

from sdg_leakage_audit.sdg.impact import (
    estimate_exceedance_alert_impact,
    estimate_reserve_margin_impact,
    estimate_storage_sizing_impact,
)
from sdg_leakage_audit.sdg.mapping import get_sdg_tags


def test_get_sdg_tags_single_tag():
    assert get_sdg_tags("household_power_consumption") == ["SDG7"]


def test_get_sdg_tags_multi_tag():
    assert get_sdg_tags("air_quality_uci") == ["SDG3", "SDG11"]


def test_get_sdg_tags_raises_for_unknown_dataset():
    with pytest.raises(KeyError):
        get_sdg_tags("not_a_real_dataset")


def _synthetic_inflation_df() -> pd.DataFrame:
    # P0 (true) rmse=10.0; P1 (leaky) apparent rmse=8.0 -> apparent error looks smaller.
    rows = []
    for i in range(5):
        rows.append({"dataset_id": "d", "protocol_id": "P0", "rmse": 10.0, "rmse_baseline": 10.0, "seed": i})
        rows.append({"dataset_id": "d", "protocol_id": "P1", "rmse": 8.0, "rmse_baseline": 10.0, "seed": i})
    return pd.DataFrame(rows)


def test_estimate_reserve_margin_impact_quantifies_under_provisioning():
    df = _synthetic_inflation_df()
    result = estimate_reserve_margin_impact(df, "d", reserve_margin_factor=2.0)
    assert result.loc["P1", "buffer_true"] == pytest.approx(20.0)
    assert result.loc["P1", "buffer_apparent"] == pytest.approx(16.0)
    assert result.loc["P1", "under_provisioning_abs"] == pytest.approx(4.0)
    assert result.loc["P1", "under_provisioning_pct"] == pytest.approx(20.0)
    assert result.loc["P0", "under_provisioning_abs"] == pytest.approx(0.0)
    assert (result["decision_label"] == "reserve_margin").all()


def test_estimate_exceedance_alert_impact_uses_different_factor():
    df = _synthetic_inflation_df()
    result = estimate_exceedance_alert_impact(df, "d", alert_buffer_factor=1.0)
    assert result.loc["P1", "buffer_true"] == pytest.approx(10.0)
    assert (result["decision_label"] == "exceedance_alert_buffer").all()


def test_estimate_storage_sizing_impact_uses_different_label():
    df = _synthetic_inflation_df()
    result = estimate_storage_sizing_impact(df, "d", storage_buffer_factor=1.5)
    assert result.loc["P1", "buffer_true"] == pytest.approx(15.0)
    assert (result["decision_label"] == "storage_capacity_buffer").all()


def test_estimate_reserve_margin_impact_raises_for_missing_dataset():
    df = _synthetic_inflation_df()
    with pytest.raises(ValueError):
        estimate_reserve_margin_impact(df, "nonexistent_dataset")
