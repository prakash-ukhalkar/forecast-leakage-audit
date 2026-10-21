"""Dataset loaders: parse raw downloaded files into standardized pandas DataFrames.

Each loader returns a DataFrame with a `DatetimeIndex` named `timestamp`, sorted
ascending, with the dataset's target column present under the name used in
`configs/datasets.yaml`. Loaders only parse/standardize structure (timestamps,
column combination, known raw-format quirks); they do NOT impute missing values,
enforce a regular frequency, or drop outliers -- that happens in `cleaning.py`, with
any fitted step applied train-fold-only to avoid leakage (see `protocols/injectors.py`).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_raw_file_path(dataset_id: str, uci_id: int | None, resolved_route: str, raw_dir: Path) -> Path:
    """Reconstruct the on-disk raw file path for a dataset, given its manifest entry.

    Mirrors the exact naming convention used by `data.download.try_ucimlrepo_then_direct`
    (`uci_<id>_ucimlrepo.csv` or `uci_<id>.zip`) and the direct-download path used for
    non-UCI sources in `notebooks/01_data_acquisition.ipynb` (`<dataset_id>.csv.zip`).

    Args:
        dataset_id: Short identifier matching a key in `configs/datasets.yaml`.
        uci_id: The dataset's UCI id, or None for non-UCI sources (e.g. Jena Climate).
        resolved_route: The route recorded in `data/MANIFEST.json` (`"ucimlrepo"` or
            `"direct_zip_url"`).
        raw_dir: Directory containing raw downloaded files (`data/raw/`).

    Returns:
        Path to the expected raw file.

    Raises:
        FileNotFoundError: If the reconstructed path does not exist.
    """
    if uci_id is not None:
        name = f"uci_{uci_id}_ucimlrepo.csv" if resolved_route == "ucimlrepo" else f"uci_{uci_id}.zip"
    else:
        name = f"{dataset_id}.csv.zip"

    path = raw_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected raw file for {dataset_id!r} not found at {path}. "
            "Re-run notebooks/01_data_acquisition.ipynb."
        )
    return path


def load_household_power_consumption(raw_path: Path) -> pd.DataFrame:
    """Load the UCI Individual Household Electric Power Consumption dataset.

    The `ucimlrepo`-exported CSV stores `Date` (dd/mm/yyyy) and `Time` (HH:MM:SS) as
    separate columns, and encodes missing readings as the literal string `"?"` across
    all numeric columns.

    Args:
        raw_path: Path to the raw downloaded CSV (from the `ucimlrepo` export route).

    Returns:
        DataFrame indexed by a constructed `timestamp`, with `Global_active_power`
        (and the other numeric sensor columns) as float columns; `"?"` sentinels are
        converted to NaN.
    """
    df = pd.read_csv(raw_path, low_memory=False)
    timestamp = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = timestamp
    df.index.name = "timestamp"
    return df.sort_index()


def load_appliances_energy_prediction(raw_path: Path) -> pd.DataFrame:
    """Load the UCI Appliances Energy Prediction dataset.

    The `ucimlrepo`-exported `date` column is missing the space between the date and
    time portions (e.g. `"2016-01-1117:00:00"` instead of `"2016-01-11 17:00:00"`);
    this is a known quirk of this dataset's export and is corrected here by splitting
    at the fixed 10-character `YYYY-MM-DD` boundary.

    Args:
        raw_path: Path to the raw downloaded CSV.

    Returns:
        DataFrame indexed by a constructed `timestamp`, with `Appliances` as the
        target column.
    """
    df = pd.read_csv(raw_path)
    fixed_date = df["date"].str[:10] + " " + df["date"].str[10:]
    timestamp = pd.to_datetime(fixed_date, format="%Y-%m-%d %H:%M:%S")
    df = df.drop(columns=["date"])
    df.index = timestamp
    df.index.name = "timestamp"
    return df.sort_index()


def load_air_quality_uci(raw_path: Path) -> pd.DataFrame:
    """Load the UCI Air Quality dataset.

    `Date` (dd/mm/yyyy) and `Time` (HH.MM.SS, colon-separated in the `ucimlrepo`
    export) are combined into a single timestamp. The documented `-200` missing-value
    sentinel is left untouched here (it is a cleaning concern, see
    `cleaning.replace_sentinel_values`) so this loader stays purely structural.

    Args:
        raw_path: Path to the raw downloaded CSV.

    Returns:
        DataFrame indexed by a constructed `timestamp`, with `CO(GT)` as the target
        column and all other sensor columns as floats.
    """
    df = pd.read_csv(raw_path)
    timestamp = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    df.index = timestamp
    df.index.name = "timestamp"
    return df.sort_index()


def load_beijing_multisite_air_quality(raw_path: Path) -> pd.DataFrame:
    """Load the UCI Beijing Multi-Site Air-Quality dataset (id 501).

    The downloaded archive is a zip-of-a-zip: the outer zip (as served by
    `archive.ics.uci.edu`) contains a nested
    `PRSA2017_Data_20130301-20170228.zip` with one CSV per monitoring station
    (plus two unrelated stock-price sample CSVs at the top level, which are ignored).
    Each station CSV has separate `year`/`month`/`day`/`hour` columns and a `station`
    column identifying the monitoring site; all 12 stations are concatenated into one
    long-format DataFrame.

    Args:
        raw_path: Path to the raw downloaded outer zip.

    Returns:
        DataFrame indexed by a constructed `timestamp` (not unique -- 12 stations
        share the same timestamp range), with `station` retained as a column and
        `PM2.5` as the target column.
    """
    with zipfile.ZipFile(raw_path) as outer_zip:
        inner_zip_names = [n for n in outer_zip.namelist() if n.endswith(".zip")]
        if not inner_zip_names:
            raise ValueError(f"No nested station-data zip found inside {raw_path}")
        inner_bytes = outer_zip.read(inner_zip_names[0])

    frames = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        station_members = [n for n in inner_zip.namelist() if n.endswith(".csv")]
        if not station_members:
            raise ValueError(f"No station CSVs found inside nested zip of {raw_path}")
        for member in sorted(station_members):
            with inner_zip.open(member) as fh:
                frames.append(pd.read_csv(fh))

    df = pd.concat(frames, ignore_index=True)
    timestamp = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    df.index = timestamp
    df.index.name = "timestamp"
    logger.info(
        "Loaded Beijing multi-site air quality: %d stations, %d rows total",
        df["station"].nunique(),
        len(df),
    )
    return df.sort_index()


def load_jena_climate(raw_path: Path) -> pd.DataFrame:
    """Load the Jena Climate 2009-2016 dataset.

    The downloaded file is a zip containing a single CSV with a `"Date Time"` column
    formatted as `dd.mm.yyyy HH:MM:SS`.

    Args:
        raw_path: Path to the raw downloaded zip.

    Returns:
        DataFrame indexed by a constructed `timestamp`, with `T (degC)` as the
        target column.
    """
    with zipfile.ZipFile(raw_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {raw_path}")
        with zf.open(csv_names[0]) as fh:
            df = pd.read_csv(fh)

    timestamp = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=["Date Time"])
    df.index = timestamp
    df.index.name = "timestamp"
    return df.sort_index()


DATASET_LOADERS = {
    "household_power_consumption": load_household_power_consumption,
    "appliances_energy_prediction": load_appliances_energy_prediction,
    "air_quality_uci": load_air_quality_uci,
    "beijing_multisite_air_quality": load_beijing_multisite_air_quality,
    "jena_climate": load_jena_climate,
}
"""Registry mapping `configs/datasets.yaml` dataset ids to their loader function."""
