"""Extracts OHLCV quotation rows for the 15 target companies out of PSE EOD
quotation report PDFs.

Consolidated from the standalone pse-pdf-pipeline project, which had this
logic split (and partly duplicated) across parser.py and extract_reports.py.
This module keeps a single implementation:

- page/date handling from the old parser.py (restrict to quotation pages,
  regex date extraction)
- the working row-parsing logic from the old extract_reports.py (token
  splitting, numeric parsing, CSV assembly)
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber

from .config import CSV_COLUMNS, MAX_QUOTATION_PAGE, TARGET_COMPANIES

log = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"([A-Za-z]+ \d{1,2}, \d{4})")


class PdfParseError(ValueError):
    """Raised when a PDF report can't be parsed at all (e.g. no date found)."""


def clean_number(value: str) -> float | None:
    """Convert PSE numeric strings ('1,234.50', '(2,500)', '-') to float."""
    value = value.strip()
    if value == "-":
        return None
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1]
    value = value.replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return None
    return -number if negative else number


def extract_report_date(text: str) -> date | None:
    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%B %d, %Y").date()


def parse_quotation_line(line: str, report_date: date) -> list | None:
    """Parse one line of quotation-table text into a CSV_COLUMNS-ordered row,
    or None if the line isn't a target company's quotation row."""
    tokens = line.split()
    if len(tokens) < 10:
        return None

    symbol_index = next((i for i, t in enumerate(tokens) if t in TARGET_COMPANIES), None)
    if symbol_index is None:
        return None

    issue_name = " ".join(tokens[:symbol_index])
    symbol = tokens[symbol_index]
    remaining = tokens[symbol_index + 1:]

    # Expected remaining tokens: Bid Ask Open High Low Close ... Volume Value
    if len(remaining) < 8:
        return None

    open_price = clean_number(remaining[2])
    high_price = clean_number(remaining[3])
    low_price = clean_number(remaining[4])
    close_price = clean_number(remaining[5])
    volume = clean_number(remaining[7])
    value = clean_number(remaining[8]) if len(remaining) >= 9 else None

    return [report_date, issue_name, symbol, open_price, high_price, low_price, close_price, volume, value]


def parse_pdf(pdf_path: Path) -> list[list]:
    """Parse a single PDF report. Returns a list of CSV_COLUMNS-ordered rows.

    Raises PdfParseError if the file is missing, corrupted/unreadable, or no
    report date can be found anywhere in the quotation pages (this last case
    indicates the PDF isn't a PSE EOD report, or PSE changed the report
    layout).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PdfParseError(f"File not found: {pdf_path}")

    records: list[list] = []
    report_date: date | None = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            quotation_pages = pdf.pages[:MAX_QUOTATION_PAGE]
            for page in quotation_pages:
                text = page.extract_text()
                if not text:
                    continue

                if report_date is None:
                    report_date = extract_report_date(text)

                for line in text.split("\n"):
                    if report_date is None:
                        continue
                    record = parse_quotation_line(line, report_date)
                    if record:
                        records.append(record)
    except PdfParseError:
        raise
    except Exception as exc:  # corrupted/unreadable PDF, unexpected pdfplumber errors
        raise PdfParseError(f"Could not read {pdf_path.name}: {exc}") from exc

    if report_date is None:
        raise PdfParseError(f"Could not find a report date in {pdf_path.name}")

    return records


def extract_reports(pdf_paths: Iterable[Path]) -> tuple[pd.DataFrame, list[tuple[Path, str]], list[str]]:
    """Extract quotation rows from every given PDF.

    Returns (long_format_dataframe, errors, warnings):
      - errors: (path, message) for PDFs that failed to parse entirely —
        the pipeline continues past a single bad file instead of aborting
        the whole batch.
      - warnings: messages for PDFs that parsed successfully (a report date
        was found) but yielded zero quotation rows — often a sign PSE
        changed the report layout, worth a human look even though it's not
        a hard failure.
    """
    all_records: list[list] = []
    errors: list[tuple[Path, str]] = []
    warnings: list[str] = []

    for pdf_path in sorted(pdf_paths):
        pdf_path = Path(pdf_path)
        try:
            records = parse_pdf(pdf_path)
            all_records.extend(records)
            if records:
                log.info("Parsed %s: %d rows", pdf_path.name, len(records))
            else:
                log.warning("Parsed %s but extracted 0 rows", pdf_path.name)
                warnings.append(f"{pdf_path.name}: parsed successfully but extracted 0 quotation rows")
        except Exception as exc:
            log.warning("Skipping %s: %s", pdf_path.name, exc)
            errors.append((pdf_path, str(exc)))

    df = pd.DataFrame(all_records, columns=CSV_COLUMNS)
    if not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["Date", "Symbol"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            log.warning("Dropped %d duplicate Date+Symbol rows across the batch", dropped)
            warnings.append(f"Dropped {dropped} duplicate Date+Symbol rows across the batch")

    return df, errors, warnings
