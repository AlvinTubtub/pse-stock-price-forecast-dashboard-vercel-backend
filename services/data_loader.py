"""Loads the bundled historical OHLCV CSVs (data/raw/<SYMBOL>.csv) and
builds the company metadata + real price history the dashboard needs.

Replace the files in data/raw/ with your actual PSE EDGE-sourced CSVs
(2020–2026, one file per ticker, columns: Date, Open, High, Low, Close,
Volume). The three sample files shipped with this project are synthetic
placeholders only, so the app can run end-to-end before you drop in your
real data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_validator import validate_ohlcv_csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SECTORS = {
    "FINANCIALS": "Financials",
    "INDUSTRIAL": "Industrial",
    "PROPERTY": "Property",
    "SERVICES": "Services",
    "MINING_OIL": "Mining and Oil",
}

# Static descriptive metadata (name/sector never changes) — mirrors the
# original dashboard's COMPANIES array. basePrice is kept only as a last-
# resort fallback if a CSV is missing for that symbol.
COMPANY_META = [
    {"symbol": "SECB", "name": "Security Bank Corporation", "sector": SECTORS["FINANCIALS"], "basePrice": 65.50},
    {"symbol": "BPI", "name": "Bank of the Philippine Islands", "sector": SECTORS["FINANCIALS"], "basePrice": 110},
    {"symbol": "MBT", "name": "Metropolitan Bank & Trust Co.", "sector": SECTORS["FINANCIALS"], "basePrice": 65},
    {"symbol": "MER", "name": "Manila Electric Company", "sector": SECTORS["INDUSTRIAL"], "basePrice": 380},
    {"symbol": "JFC", "name": "Jollibee Foods Corporation", "sector": SECTORS["INDUSTRIAL"], "basePrice": 240},
    {"symbol": "SHLPH", "name": "Pilipinas Shell Petroleum Corp.", "sector": SECTORS["INDUSTRIAL"], "basePrice": 12},
    {"symbol": "MEG", "name": "Megaworld Corporation", "sector": SECTORS["PROPERTY"], "basePrice": 2.1},
    {"symbol": "ALI", "name": "Ayala Land, Inc.", "sector": SECTORS["PROPERTY"], "basePrice": 32},
    {"symbol": "SMPH", "name": "SM Prime Holdings, Inc.", "sector": SECTORS["PROPERTY"], "basePrice": 33},
    {"symbol": "GLO", "name": "Globe Telecom, Inc.", "sector": SECTORS["SERVICES"], "basePrice": 1800},
    {"symbol": "PGOLD", "name": "Puregold Price Club, Inc.", "sector": SECTORS["SERVICES"], "basePrice": 28},
    {"symbol": "ICT", "name": "Intl. Container Terminal Services", "sector": SECTORS["SERVICES"], "basePrice": 280},
    {"symbol": "APX", "name": "Apex Mining Co., Inc.", "sector": SECTORS["MINING_OIL"], "basePrice": 3.5},
    {"symbol": "NIKL", "name": "Nickel Asia Corporation", "sector": SECTORS["MINING_OIL"], "basePrice": 5.2},
    {"symbol": "SCC", "name": "Semirara Mining and Power Corp.", "sector": SECTORS["MINING_OIL"], "basePrice": 31},
]


def _ohlcv_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "open": round(float(row.Open), 2),
            "high": round(float(row.High), 2),
            "low": round(float(row.Low), 2),
            "close": round(float(row.Close), 2),
            "volume": int(row.Volume),
        }
        for row in df.itertuples()
    ]


def load_companies() -> tuple[list[dict], dict[str, pd.DataFrame], list[str]]:
    """Returns (companies_for_frontend, dataframes_by_symbol, missing_symbols).

    companies_for_frontend matches what dashboard_template.html expects:
    [{symbol, name, sector, basePrice, latestClose, ohlcv: [...]}, ...]
    """
    companies = []
    dataframes: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for meta in COMPANY_META:
        csv_path = DATA_DIR / f"{meta['symbol']}.csv"
        entry = dict(meta)

        if csv_path.exists():
            try:
                df = validate_ohlcv_csv(csv_path)
                dataframes[meta["symbol"]] = df
                entry["latestClose"] = round(float(df["Close"].iloc[-1]), 2)
                entry["ohlcv"] = _ohlcv_records(df)
            except Exception:
                missing.append(meta["symbol"])
                entry["latestClose"] = meta["basePrice"]
                entry["ohlcv"] = []
        else:
            missing.append(meta["symbol"])
            entry["latestClose"] = meta["basePrice"]
            entry["ohlcv"] = []

        companies.append(entry)

    return companies, dataframes, missing
