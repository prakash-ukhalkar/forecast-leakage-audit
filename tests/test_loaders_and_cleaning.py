"""Tests for src.sdg_leakage_audit.data.loaders and cleaning.

Loader tests use tiny synthetic files mimicking each dataset's known raw-format
quirks (split Date/Time columns, "?" sentinels, missing-space date bug, nested
station zips) rather than the real downloaded files, so they run fast and don't
depend on network access.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdg_leakage_audit.data.cleaning import (
    clean_and_report,
    enforce_frequency,
    flag_duplicates,
    flag_outliers_iqr,
    replace_sentinel_values,
)
from sdg_leakage_audit.data.loaders import (
    load_air_quality_uci,
    load_appliances_energy_prediction,
    load_beijing_multisite_air_quality,
    load_household_power_consumption,
    load_jena_climate,
)


def test_load_household_power_consumption_parses_date_time_and_sentinel(tmp_path: Path):
    csv_path = tmp_path / "household.csv"
    csv_path.write_text(
        "Date,Time,Global_active_power,Global_reactive_power,Voltage,Global_intensity,"
        "Sub_metering_1,Sub_metering_2,Sub_metering_3\n"
        "16/12/2006,17:24:00,4.216,0.418,234.84,18.4,0.0,1.0,17.0\n"
        "16/12/2006,17:25:00,?,?,233.63,23.0,0.0,1.0,16.0\n"
    )
    df = load_household_power_consumption(csv_path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2006-12-16 17:24:00")
    assert np.isnan(df["Global_active_power"].iloc[1])
    assert df["Global_active_power"].iloc[0] == pytest.approx(4.216)


def test_load_appliances_energy_prediction_fixes_missing_space(tmp_path: Path):
    csv_path = tmp_path / "appliances.csv"
    csv_path.write_text("date,lights,Appliances\n2016-01-1117:00:00,30,60\n2016-01-1117:10:00,30,70\n")
    df = load_appliances_energy_prediction(csv_path)
    assert df.index[0] == pd.Timestamp("2016-01-11 17:00:00")
    assert df.index[1] == pd.Timestamp("2016-01-11 17:10:00")
    assert df["Appliances"].tolist() == [60, 70]


def test_load_air_quality_uci_combines_date_time(tmp_path: Path):
    csv_path = tmp_path / "air_quality.csv"
    csv_path.write_text("Date,Time,CO(GT),T\n3/10/2004,18:00:00,2.6,13.6\n3/10/2004,19:00:00,-200,13.3\n")
    df = load_air_quality_uci(csv_path)
    assert df.index[0] == pd.Timestamp("2004-03-10 18:00:00")
    # Sentinel replacement is a cleaning concern, not a loader concern.
    assert df["CO(GT)"].iloc[1] == -200


def test_load_beijing_multisite_air_quality_unpacks_nested_zip_and_stations(tmp_path: Path):
    station_a = (
        "No,year,month,day,hour,PM2.5,station\n"
        "1,2013,3,1,0,4,StationA\n"
        "2,2013,3,1,1,8,StationA\n"
    )
    station_b = (
        "No,year,month,day,hour,PM2.5,station\n"
        "1,2013,3,1,0,12,StationB\n"
        "2,2013,3,1,1,16,StationB\n"
    )
    inner_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_zip_bytes, "w") as inner_zf:
        inner_zf.writestr("PRSA_Data_20130301-20170228/PRSA_Data_StationA.csv", station_a)
        inner_zf.writestr("PRSA_Data_20130301-20170228/PRSA_Data_StationB.csv", station_b)

    outer_zip_path = tmp_path / "uci_501.zip"
    with zipfile.ZipFile(outer_zip_path, "w") as outer_zf:
        outer_zf.writestr("PRSA2017_Data_20130301-20170228.zip", inner_zip_bytes.getvalue())
        outer_zf.writestr("data.csv", "Date,Open\n2018-07-02,183.82\n")  # unrelated sample file

    df = load_beijing_multisite_air_quality(outer_zip_path)
    assert len(df) == 4
    assert set(df["station"].unique()) == {"StationA", "StationB"}
    assert df.index[0] == pd.Timestamp("2013-03-01 00:00:00")


def test_load_jena_climate_parses_dotted_date(tmp_path: Path):
    csv_content = (
        '"Date Time","T (degC)"\n'
        "01.01.2009 00:10:00,-8.02\n"
        "01.01.2009 00:20:00,-8.41\n"
    )
    zip_path = tmp_path / "jena.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("jena_climate_2009_2016.csv", csv_content)

    df = load_jena_climate(zip_path)
    assert df.index[0] == pd.Timestamp("2009-01-01 00:10:00")
    assert df["T (degC)"].iloc[1] == pytest.approx(-8.41)


def test_enforce_frequency_introduces_gap_rows_as_nan():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:02"])  # missing the 00:01 minute
    df = pd.DataFrame({"x": [1.0, 3.0]}, index=idx)
    out = enforce_frequency(df, freq="1min")
    assert len(out) == 3
    assert np.isnan(out["x"].iloc[1])
    assert out["x"].iloc[0] == 1.0 and out["x"].iloc[2] == 3.0


def test_enforce_frequency_averages_duplicate_timestamps():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:01"])
    df = pd.DataFrame({"x": [1.0, 3.0, 5.0]}, index=idx)
    out = enforce_frequency(df, freq="1min")
    assert len(out) == 2
    assert out["x"].iloc[0] == pytest.approx(2.0)  # mean of 1.0 and 3.0


def test_replace_sentinel_values_converts_to_nan():
    df = pd.DataFrame({"a": [-200, 1.0, -200], "b": [1.0, 2.0, 3.0]})
    out = replace_sentinel_values(df, sentinel=-200)
    assert out["a"].isna().sum() == 2
    assert out["b"].isna().sum() == 0
    # Original untouched (function returns a copy).
    assert df["a"].isna().sum() == 0


def test_flag_duplicates_marks_all_but_first_occurrence():
    idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    flags = flag_duplicates(df)
    assert flags.tolist() == [False, True, False]


def test_flag_outliers_iqr_detects_extreme_value_and_ignores_nan():
    series = pd.Series([10.0, 11.0, 9.0, 10.5, 9.5, 1000.0, np.nan])
    flags = flag_outliers_iqr(series)
    assert flags.iloc[5] is True or bool(flags.iloc[5]) is True
    assert bool(flags.iloc[6]) is False  # NaN never flagged
    assert bool(flags.iloc[0]) is False


def test_clean_and_report_single_series_pipeline():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:03"])
    df = pd.DataFrame({"target": [-200.0, 5.0, 6.0]}, index=idx)
    cleaned, report = clean_and_report(df, freq="1min", target_column="target", sentinel=-200.0)

    assert report["n_rows_raw"] == 3
    assert report["n_sentinel_replaced"] == 1
    assert report["n_rows_after_freq"] == 4  # 00:00, 00:01, 00:02 (gap), 00:03
    assert report["n_gap_rows_introduced"] == 1
    assert np.isnan(cleaned["target"].iloc[0])  # sentinel -> NaN
    assert np.isnan(cleaned["target"].iloc[2])  # frequency gap -> NaN


def test_clean_and_report_grouped_by_station():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"] * 2)
    df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 10.0, 20.0],
            "station": ["A", "A", "B", "B"],
        },
        index=idx,
    )
    cleaned, report = clean_and_report(df, freq="1h", target_column="target", group_column="station")
    assert set(cleaned["station"].unique()) == {"A", "B"}
    assert report["n_rows_after_freq"] == 4
