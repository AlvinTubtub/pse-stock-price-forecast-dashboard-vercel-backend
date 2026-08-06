"""Native PDF ingestion pipeline: PSE EDGE EOD PDF reports -> data/raw/<SYMBOL>.csv.

Public API:
    run_pipeline(pdf_paths=None, download=False, start_date=None, end_date=None) -> dict
    list_staged_pdfs() -> list[Path]
    latest_raw_date() -> date | None
"""
from .pipeline import latest_raw_date, list_staged_pdfs, run_pipeline

__all__ = ["run_pipeline", "list_staged_pdfs", "latest_raw_date"]
