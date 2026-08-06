"""Central configuration for the PSE PDF ingestion pipeline.

Adapted from the standalone `pse-pdf-pipeline` project's config.py. Paths
are rebased onto this repo's existing `data/` layout instead of a separate
`reports/` / `output/` tree, so the pipeline is a native module here rather
than a bolted-on project:

    data/pdf_reports/   staged PDF reports (uploaded or downloaded)
    data/pdf_pipeline/  intermediate ETL artifacts + logs
    data/raw/           final per-symbol OHLCV CSVs the dashboard reads
                         (services/data_loader.py, services/data_validator.py)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ============================================================
# FOLDERS
# ============================================================

PDF_REPORTS_DIR = BASE_DIR / "data" / "pdf_reports"
PIPELINE_DIR = BASE_DIR / "data" / "pdf_pipeline"
RAW_DIR = BASE_DIR / "data" / "raw"

# Repo-root pipeline-run metadata, read by the Streamlit dashboard to show
# when data was last refreshed — written once per run, regardless of
# outcome (see services/pdf_pipeline/pipeline.py:_write_latest_processed).
LATEST_PROCESSED_PATH = BASE_DIR / "latest_processed.json"

MASTER_CSV = PIPELINE_DIR / "master_quotes.csv"
CLEANED_CSV = PIPELINE_DIR / "cleaned_quotes.csv"
VALIDATION_REPORT = PIPELINE_DIR / "validation_report.txt"
PIPELINE_LOG = PIPELINE_DIR / "pipeline.log"

# ============================================================
# DOWNLOAD SETTINGS (PSE EDGE end-of-day quotation reports)
# ============================================================

BASE_URL = "https://documents.pse.com.ph/market_report"
PDF_SUFFIX = "-EOD.pdf"
REQUEST_TIMEOUT = 30
SKIP_EXISTING_FILES = True
VERIFY_SSL = True

# First report this pipeline knows how to fetch. Callers can override this
# with a later start date (e.g. "the day after our newest local data").
EARLIEST_REPORT_DATE = date(2026, 7, 1)

# ============================================================
# TARGET COMPANIES
#
# Symbols match services/data_loader.py:COMPANY_META exactly — these are
# the 15 companies the dashboard tracks. Names are the exact issuer-name
# tokens as printed in the PSE EOD quotation report tables, used to find
# each company's PDF row and to sanity-check the parsed rows.
# ============================================================

TARGET_COMPANIES: dict[str, str] = {
    "BPI": "BANK PH ISLANDS",
    "MBT": "METROBANK",
    "MER": "MERALCO",
    "JFC": "JOLLIBEE",
    "SHLPH": "SHELL PILIPINAS",
    "MEG": "MEGAWORLD",
    "ALI": "AYALA LAND",
    "SMPH": "SM PRIME HLDG",
    "GLO": "GLOBE TELECOM",
    "PGOLD": "PUREGOLD",
    "ICT": "INTL CONTAINER",
    "APX": "APEX MINING",
    "NIKL": "NICKEL ASIA",
    "SCC": "SEMIRARA MINING",
    "SECB": "SECURITY BANK",
}

EXPECTED_COMPANY_COUNT = len(TARGET_COMPANIES)

# Only the quotation pages contain per-company OHLCV rows; later pages hold
# block sales, summaries, and preferred-share references.
MAX_QUOTATION_PAGE = 11

CSV_COLUMNS = ["Date", "Issue Name", "Symbol", "Open", "High", "Low", "Close", "Volume", "Value"]


def ensure_dirs() -> None:
    for d in (PDF_REPORTS_DIR, PIPELINE_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)
