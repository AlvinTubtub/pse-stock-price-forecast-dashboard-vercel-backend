"""Cleans the long-format quotation rows extracted from PDFs, before
per-symbol validation and merge into data/raw/.

Adapted from the standalone pse-pdf-pipeline/cleaner.py: same cleaning
steps, but operating on a DataFrame in memory instead of reading/writing
fixed CSV paths, so it composes directly with parser.extract_reports().
"""
from __future__ import annotations

import pandas as pd

NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Value"]


def clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(how="all")

    string_columns = df.select_dtypes(include="object").columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [c for c in ["Date", "Symbol"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    sort_cols = [c for c in ["Date", "Symbol"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.reset_index(drop=True)
