"""Orchestrates the PDF ingestion pipeline (and, optionally, training):

    1. Download latest PSE EOD reports (optional)
    2. Extract quotation rows from staged PDFs
    3. Clean the extracted dataset
    4. Validate the cleaned dataset
    5. Merge into data/raw/<SYMBOL>.csv (upsert by date)
    6. Re-validate each touched CSV with the app's own validator, so a bad
       merge can never silently break the forecasting/dashboard modules
    7. Retrain all forecasting models and select the best one per ticker
       (services/model_selector.py) — only if train_models=True
    8. Write latest_processed.json so the dashboard knows the run happened

This is the orchestration layer for data ingestion, called exclusively
from run_pipeline.py (headless, no Streamlit dependency). In production,
run_pipeline.py is only ever invoked by the Fast Pipeline
(.github/workflows/update_pipeline.yml, Monday-Friday, triggered
externally by Cron-job.org) with train_models=False — step 7 above never
runs there. Model retraining is instead handled weekly, on GitHub
Actions' own cron, by .github/workflows/train_models.yml calling
services/model_selector.py directly (bypassing this module entirely,
since there's nothing to ingest on that schedule). Streamlit never calls
either — the dashboard is a pure read-only presentation layer over
whatever was last committed to the repo (see ui/data.py).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv

from .cleaner import clean_quotes
from .config import (
    CLEANED_CSV,
    EARLIEST_REPORT_DATE,
    LATEST_PROCESSED_PATH,
    MASTER_CSV,
    PDF_REPORTS_DIR,
    PIPELINE_LOG,
    RAW_DIR,
    VALIDATION_REPORT,
    ensure_dirs,
)
from .downloader import DownloadResult, download_reports
from .merge import merge_into_raw
from .parser import extract_reports
from .validator import format_report, validate_quotes

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    ensure_dirs()
    pipeline_logger = logging.getLogger("services.pdf_pipeline")
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(PIPELINE_LOG) for h in pipeline_logger.handlers):
        return
    handler = logging.FileHandler(PIPELINE_LOG, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pipeline_logger.addHandler(handler)
    pipeline_logger.setLevel(logging.INFO)


def list_staged_pdfs(reports_dir: Path = PDF_REPORTS_DIR) -> list[Path]:
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("*.pdf"))


def latest_raw_date(raw_dir: Path = RAW_DIR) -> date | None:
    """Newest date already present across all data/raw/<SYMBOL>.csv files —
    used to default the download window to 'only what's missing'."""
    latest: date | None = None
    for csv_path in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=["Date"], parse_dates=["Date"])
        except Exception:
            continue
        if df.empty:
            continue
        d = df["Date"].max().date()
        if latest is None or d > latest:
            latest = d
    return latest


def _write_latest_processed(result: dict) -> None:
    """Writes repo-root latest_processed.json — the single file the
    read-only Streamlit dashboard checks to know when data was last
    refreshed and by which run. Written on every run, success or not, so
    the dashboard always reflects the true state of the last attempt.
    """
    payload = {
        "last_run_at": (result["finished_at"] or datetime.now()).astimezone(timezone.utc).isoformat(timespec="seconds"),
        "status": result["status"],
        "duration_seconds": result["duration_seconds"],
        "pdf_count": result["pdf_count"],
        "record_count": result["record_count"],
        "symbols_updated": [s["symbol"] for s in result["merge_summaries"]],
        "best_models_updated": len(result["training"]["best_models"]) if result["training"] else 0,
        "triggered_by": os.environ.get("GITHUB_EVENT_NAME", "manual"),
        "error": result["error"],
    }
    try:
        LATEST_PROCESSED_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        log.exception("Failed to write %s", LATEST_PROCESSED_PATH)


def run_pipeline(
    pdf_paths: list[Path] | None = None,
    download: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
    train_models: bool = True,
) -> dict:
    """Runs the full pipeline and returns a structured result dict:

        {
            "status": "ok" | "no_files" | "no_rows" | "merged_with_warnings" | "error",
            "started_at": datetime, "finished_at": datetime, "duration_seconds": float,
            "pdf_count": int, "parsed_count": int, "record_count": int,
            "download": DownloadResult | None,
            "parse_errors": [(path, message), ...],
            "parse_warnings": [str, ...],
            "validation": {...} | None,
            "merge_summaries": [{...}, ...],
            "post_validation_errors": [(symbol, message), ...],
            "training": {"best_models": {...}} | None,
            "error": str | None,
        }

    If pdf_paths is None, every PDF already staged in data/pdf_reports/ is
    used. If download=True, new reports are fetched first (from
    latest_raw_date()+1, or EARLIEST_REPORT_DATE if data/raw/ is empty,
    through end_date/today) and added to the batch.

    Individual bad inputs (a missing/corrupted PDF, a report with zero
    extractable rows, a handful of duplicate records) are logged and
    skipped rather than aborting the run — the pipeline only reports
    "error" for failures that leave it with nothing usable to merge.

    If train_models=True (the default) and at least one symbol was merged
    successfully, the full model-training pipeline (feature engineering ->
    train Lag Regression/ARIMA/LSTM -> evaluate -> pick each ticker's best
    model -> save models + best_models.json) runs automatically at the end
    — see services/model_selector.py. A training failure is logged but
    never turns an otherwise-successful ingestion into an "error" result;
    the freshly merged CSVs are still valid even if retraining had a
    problem, and the dashboard falls back to previously-saved models.

    Every run — successful or not — writes repo-root latest_processed.json
    with the outcome, so the (read-only) Streamlit dashboard can always
    show when data was last refreshed without executing anything itself.
    """
    _configure_logging()
    started_at = datetime.now()
    t0 = time.monotonic()
    log.info("=" * 60)
    log.info("Pipeline started at %s", started_at.isoformat(timespec="seconds"))

    result: dict = {
        "status": "ok",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "pdf_count": 0,
        "parsed_count": 0,
        "record_count": 0,
        "download": None,
        "parse_errors": [],
        "parse_warnings": [],
        "validation": None,
        "merge_summaries": [],
        "post_validation_errors": [],
        "training": None,
        "error": None,
    }

    def _finish(status: str | None = None) -> dict:
        if status:
            result["status"] = status
        result["finished_at"] = datetime.now()
        result["duration_seconds"] = round(time.monotonic() - t0, 2)
        log.info(
            "Pipeline finished with status=%s in %.2fs (pdfs=%d parsed=%d records=%d)",
            result["status"], result["duration_seconds"], result["pdf_count"], result["parsed_count"], result["record_count"],
        )
        log.info("=" * 60)
        _write_latest_processed(result)
        return result

    try:
        if download:
            log.info("Downloading reports...")
            existing_latest = latest_raw_date()
            effective_start = start_date or ((existing_latest + timedelta(days=1)) if existing_latest else EARLIEST_REPORT_DATE)
            dl: DownloadResult = download_reports(effective_start, end_date)
            result["download"] = dl
            log.info(
                "Download complete: %d new, %d already present, %d not published, %d failed",
                len(dl.downloaded), len(dl.skipped), len(dl.not_found), len(dl.errors),
            )
            pdf_paths = dl.all_available

        pdf_paths = list(pdf_paths) if pdf_paths is not None else list_staged_pdfs()
        result["pdf_count"] = len(pdf_paths)
        if not pdf_paths:
            log.warning("No PDF reports available to process")
            return _finish("no_files")

        log.info("Extracting %d PDF report(s)...", len(pdf_paths))
        long_df, parse_errors, parse_warnings = extract_reports(pdf_paths)
        result["parse_errors"] = parse_errors
        result["parse_warnings"] = parse_warnings
        result["parsed_count"] = len(pdf_paths) - len(parse_errors)
        result["record_count"] = len(long_df)
        for path, message in parse_errors:
            log.error("Failed to parse %s: %s", Path(path).name, message)
        for message in parse_warnings:
            log.warning(message)

        if long_df.empty:
            log.warning("No quotation rows could be extracted from any staged PDF")
            return _finish("no_rows")

        log.info("Extracted %d rows from %d/%d PDFs", len(long_df), result["parsed_count"], result["pdf_count"])

        log.info("Cleaning extracted data...")
        MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(MASTER_CSV, index=False)
        cleaned = clean_quotes(long_df)
        cleaned.to_csv(CLEANED_CSV, index=False)
        log.info("Cleaned dataset: %d rows (from %d raw rows)", len(cleaned), len(long_df))

        log.info("Validating cleaned data...")
        validation = validate_quotes(cleaned)
        result["validation"] = validation
        VALIDATION_REPORT.write_text(format_report(validation), encoding="utf-8")
        failed_checks = [c["label"] for c in validation["checks"] if not c["passed"]]
        if failed_checks:
            log.warning("Validation issues: %s", "; ".join(failed_checks))
        else:
            log.info("Validation passed: all checks OK")

        log.info("Merging into data/raw/...")
        merge_summaries = merge_into_raw(cleaned)
        result["merge_summaries"] = merge_summaries
        log.info(
            "Merge complete: %d symbol(s) updated (%s)",
            len(merge_summaries),
            ", ".join(f"{s['symbol']}+{s['rows_added']}" for s in merge_summaries) or "none",
        )

        for summary in merge_summaries:
            try:
                validate_ohlcv_csv(summary["path"])
            except CSVValidationError as exc:
                result["post_validation_errors"].append((summary["symbol"], str(exc)))
                log.error("Post-merge validation failed for %s: %s", summary["symbol"], exc)

        if train_models and merge_summaries:
            log.info("Retraining models for %d updated symbol(s)...", len(merge_summaries))
            try:
                from services.model_selector import train_and_select_all

                best_models = train_and_select_all()
                result["training"] = {"best_models": best_models}
                log.info("Model training complete: %d ticker(s) now have saved models.", len(best_models))
            except Exception:
                log.exception("Model training failed after a successful ingestion — dashboard will keep using previously-saved models.")

        if result["post_validation_errors"]:
            return _finish("merged_with_warnings")

        return _finish("ok")

    except Exception as exc:
        log.exception("Pipeline failed with an unexpected error")
        result["error"] = str(exc)
        return _finish("error")
