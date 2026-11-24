"""Download ETTh1 and build its engineered feature table (same recipe as the other datasets).

Usage: python scripts/prepare_extension_data.py
Writes data/raw/ett_h1.csv, data/interim/ett_h1.parquet, data/processed/ett_h1_features.parquet
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sdg_leakage_audit.features.build import build_feature_matrix  # noqa: E402

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
HORIZONS = [1, 6, 24]


def main() -> None:
    raw = ROOT / "data" / "raw" / "ett_h1.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    df = pd.read_csv(raw, parse_dates=["date"]).set_index("date").sort_index()
    (ROOT / "data" / "interim").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "data" / "interim" / "ett_h1.parquet")
    feats = build_feature_matrix(df, target_column="OT", lags=HORIZONS, rolling_windows=HORIZONS)
    out = pd.concat([df, feats.drop(columns=[c for c in feats.columns if c in df.columns])], axis=1)
    (ROOT / "data" / "processed").mkdir(exist_ok=True)
    out.to_parquet(ROOT / "data" / "processed" / "ett_h1_features.parquet")
    print("ETTh1 features:", out.shape, list(out.columns)[:14])


if __name__ == "__main__":
    main()
