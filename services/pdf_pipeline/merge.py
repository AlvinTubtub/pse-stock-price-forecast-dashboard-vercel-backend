"""Merges cleaned long-format quotation rows (Date, Issue Name, Symbol,
Open, High, Low, Close, Volume, Value) into the dashboard's existing
per-symbol OHLCV CSVs at data/raw/<SYMBOL>.csv.

This is the bridge between the PDF pipeline and the rest of the app:
services/data_loader.py and services/data_validator.py already read
data/raw/<SYMBOL>.csv with columns Date, Open, High, Low, Close, Volume —
this module writes exactly that shape, upserting by Date so re-running the
pipeline on overlapping reports never duplicates or loses history.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import RAW_DIR

log = logging.getLogger(__name__)

RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def merge_symbol(symbol: str, new_rows: pd.DataFrame, raw_dir=RAW_DIR) -> dict:
    """Upsert new_rows (already in RAW_COLUMNS shape) into data/raw/<symbol>.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{symbol}.csv"

    new_rows = new_rows[RAW_COLUMNS].dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        before = len(existing)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        existing = None
        before = 0
        combined = new_rows

    combined = (
        combined.drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    combined.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    return {
        "symbol": symbol,
        "path": csv_path,
        "rows_before": before,
        "rows_after": len(combined),
        "rows_added": len(combined) - before,
        "latest_date": combined["Date"].max().strftime("%Y-%m-%d") if len(combined) else None,
    }


def merge_into_raw(cleaned_long_df: pd.DataFrame, raw_dir=RAW_DIR) -> list[dict]:
    """Split the cleaned long-format dataset by Symbol and upsert each one
    into its data/raw/<SYMBOL>.csv. Returns a per-symbol summary list."""
    summaries: list[dict] = []
    if cleaned_long_df.empty:
        return summaries

    for symbol, group in cleaned_long_df.groupby("Symbol"):
        wide = group[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        summary = merge_symbol(symbol, wide, raw_dir)
        summaries.append(summary)
        log.info(
            "Merged %s: +%d rows (now %d total, latest %s)",
            symbol, summary["rows_added"], summary["rows_after"], summary["latest_date"],
        )

    return summaries
