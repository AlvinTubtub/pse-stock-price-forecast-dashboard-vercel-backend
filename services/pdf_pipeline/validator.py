"""Validates the cleaned long-format quotation dataset before it's merged
into data/raw/. Same checks as the standalone pse-pdf-pipeline/validator.py,
returning a structured dict instead of only a printed report — the
Streamlit page renders these as pass/fail rows.
"""
from __future__ import annotations

import pandas as pd

from .config import EXPECTED_COMPANY_COUNT

REQUIRED_COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_quotes(df: pd.DataFrame) -> dict:
    """Returns a dict: {"ok": bool, "checks": [{"label", "passed", "detail"}], "row_count": int}."""
    checks: list[dict] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks.append({
        "label": "Required columns present",
        "passed": not missing_cols,
        "detail": f"Missing: {', '.join(missing_cols)}" if missing_cols else "",
    })

    if {"Date", "Symbol"}.issubset(df.columns):
        dup_count = int(df.duplicated(subset=["Date", "Symbol"]).sum())
        checks.append({
            "label": "No duplicate Date+Symbol rows",
            "passed": dup_count == 0,
            "detail": f"{dup_count} duplicate rows" if dup_count else "",
        })

    missing_values = df.isna().sum()
    mv = missing_values[missing_values > 0]
    checks.append({
        "label": "No missing values",
        "passed": len(mv) == 0,
        "detail": ", ".join(f"{c}: {n}" for c, n in mv.items()) if len(mv) else "",
    })

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            invalid = int(df[col].isna().sum())
            checks.append({
                "label": f"{col} is numeric",
                "passed": invalid == 0,
                "detail": f"{invalid} invalid values" if invalid else "",
            })

    if {"High", "Low"}.issubset(df.columns):
        bad_hl = int((df["High"] < df["Low"]).sum())
        checks.append({
            "label": "High >= Low",
            "passed": bad_hl == 0,
            "detail": f"{bad_hl} rows" if bad_hl else "",
        })

    if "Volume" in df.columns:
        bad_vol = int((df["Volume"] < 0).sum())
        checks.append({
            "label": "Volume is non-negative",
            "passed": bad_vol == 0,
            "detail": f"{bad_vol} rows" if bad_vol else "",
        })

    if {"Date", "Symbol"}.issubset(df.columns) and not df.empty:
        grouped = df.groupby("Date")["Symbol"].nunique()
        incomplete_dates = grouped[grouped != EXPECTED_COMPANY_COUNT]
        checks.append({
            "label": f"Every trading day has all {EXPECTED_COMPANY_COUNT} companies",
            "passed": len(incomplete_dates) == 0,
            "detail": ", ".join(f"{d.date()}: {n}/{EXPECTED_COMPANY_COUNT}" for d, n in incomplete_dates.items()) if len(incomplete_dates) else "",
        })

    ok = all(c["passed"] for c in checks)
    return {"ok": ok, "checks": checks, "row_count": len(df)}


def format_report(report: dict) -> str:
    lines = ["=" * 60, "PSE PDF PIPELINE — VALIDATION REPORT", "=" * 60, f"Total rows: {report['row_count']}", ""]
    for check in report["checks"]:
        mark = "✅" if check["passed"] else "❌"
        lines.append(f"{mark} {check['label']}" + (f" — {check['detail']}" if check["detail"] else ""))
    return "\n".join(lines)
