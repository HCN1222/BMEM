"""Helpers for loading shared stock OHLCV files, including incremental parts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_stock_files(stock_dir: Path, stock_id: str) -> list[Path]:
    stock_dir = Path(stock_dir)
    stock_id = str(stock_id)
    files = sorted(stock_dir.glob(f"{stock_id}_*.parquet"))
    files += sorted(stock_dir.glob(f"{stock_id}_*.csv"))
    return files


def load_stock_data(stock_dir: Path, stock_id: str) -> pd.DataFrame:
    """Load and deduplicate every historical/incremental file for one stock."""
    parts = []
    for path in find_stock_files(stock_dir, stock_id):
        if path.suffix.lower() == ".parquet":
            part = pd.read_parquet(path)
        else:
            part = pd.read_csv(path, dtype={"stock_id": "string"})
        if "stock_id" not in part.columns:
            part["stock_id"] = str(stock_id)
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.drop_duplicates(subset=["stock_id", "date"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )


def latest_stock_date(stock_dir: Path, stock_id: str) -> pd.Timestamp | None:
    """Return the latest date available across all files for one stock."""
    latest = None
    for path in find_stock_files(stock_dir, stock_id):
        if path.suffix.lower() == ".parquet":
            dates = pd.read_parquet(path, columns=["date"])["date"]
        else:
            dates = pd.read_csv(path, usecols=["date"])["date"]
        candidate = pd.to_datetime(dates).max()
        if pd.notna(candidate) and (latest is None or candidate > latest):
            latest = candidate
    return latest
