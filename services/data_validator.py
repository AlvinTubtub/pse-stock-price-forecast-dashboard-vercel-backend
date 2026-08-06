"""OHLCV CSV validation.

No database is used anywhere in this app — every uploaded CSV is validated,
processed, and discarded within a single Streamlit run. See the project
README for the reasoning.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class CSVValidationError(ValueError):
    """Raised when an uploaded OHLCV file fails validation."""


def validate_ohlcv_csv(file) -> pd.DataFrame:
    """Load and validate an uploaded OHLCV CSV file.

    `file` is anything pandas.read_csv accepts, including a Streamlit
    UploadedFile object.
    """
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise CSVValidationError("The uploaded file could not be read as a CSV.") from exc

    df.columns = df.columns.str.strip().str.title()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise CSVValidationError(f"Missing required columns: {missing}.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["Date"].isna().any():
        raise CSVValidationError("Some values in the Date column are invalid.")

    if df[NUMERIC_COLUMNS].isna().any().any():
        raise CSVValidationError("The OHLCV columns contain missing or non-numeric values.")

    if df["Date"].duplicated().any():
        raise CSVValidationError("The CSV contains duplicate trading dates.")

    if (df["Volume"] < 0).any():
        raise CSVValidationError("Volume cannot contain negative values.")

    if (df["High"] < df[["Open", "Low", "Close"]].max(axis=1)).any():
        raise CSVValidationError("Some High values are lower than Open, Low, or Close.")

    if (df["Low"] > df[["Open", "High", "Close"]].min(axis=1)).any():
        raise CSVValidationError("Some Low values are higher than Open, High, or Close.")

    if len(df) < 30:
        raise CSVValidationError(
            "At least 30 trading days of history are required to generate lag features "
            "and a meaningful forecast."
        )

    return df.sort_values("Date").reset_index(drop=True)
