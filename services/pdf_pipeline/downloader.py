"""Downloads PSE End-of-Day quotation PDF reports from PSE EDGE.

Refactored from the standalone pse-pdf-pipeline/download_reports.py into a
reusable function: the original was a top-level script. Behavior (URL
format, filename format, skip-existing, timeout/SSL settings) is preserved
exactly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import requests

from .config import (
    BASE_URL,
    PDF_REPORTS_DIR,
    PDF_SUFFIX,
    REQUEST_TIMEOUT,
    SKIP_EXISTING_FILES,
    VERIFY_SSL,
)

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2  # attempt 1 -> wait 2s, attempt 2 -> wait 4s, attempt 3 -> wait 8s


@dataclass
class DownloadResult:
    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    not_found: list[date] = field(default_factory=list)
    errors: list[tuple[date, str]] = field(default_factory=list)

    @property
    def all_available(self) -> list[Path]:
        """Every report now on disk for the requested range (fresh + already there)."""
        return sorted(self.downloaded + self.skipped)


def _generate_url(current_date: date) -> str:
    month = current_date.strftime("%B")
    day = current_date.strftime("%d")
    year = current_date.strftime("%Y")
    filename = f"{month}%20{day},%20{year}{PDF_SUFFIX}"
    return f"{BASE_URL}/{filename}"


def _pdf_filename(current_date: date) -> str:
    return current_date.strftime("%Y-%m-%d") + "-EOD.pdf"


def download_one(
    current_date: date,
    reports_dir: Path = PDF_REPORTS_DIR,
    max_retries: int = MAX_RETRIES,
    sleep_fn=time.sleep,
) -> tuple[str, Path | None, str | None]:
    """Download a single day's report, retrying transient failures with
    exponential backoff (2s, 4s, 8s, ...).

    Retried: network errors/timeouts and HTTP 5xx (server-side, likely
    transient). Not retried: HTTP 404 (no report published that day — normal
    for weekends/holidays) and other 4xx (the request itself is wrong, a
    retry won't help).

    Returns (status, path, message) where status is one of:
    "downloaded", "skipped", "not_found", "error".
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_path = reports_dir / _pdf_filename(current_date)

    if SKIP_EXISTING_FILES and save_path.exists():
        return "skipped", save_path, None

    url = _generate_url(current_date)
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
        except Exception as exc:  # network errors, timeouts, SSL issues
            last_error = str(exc)
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, current_date, exc)
            if attempt < max_retries:
                wait = BACKOFF_BASE_SECONDS ** attempt
                log.info("Retrying %s in %ds...", current_date, wait)
                sleep_fn(wait)
                continue
            log.error("Giving up on %s after %d attempts: %s", current_date, max_retries, exc)
            return "error", None, last_error

        if response.status_code == 200:
            save_path.write_bytes(response.content)
            log.info("Downloaded %s (%.2f MB)", save_path.name, len(response.content) / 1024 / 1024)
            return "downloaded", save_path, None

        if response.status_code == 404:
            return "not_found", None, None

        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            log.warning("Attempt %d/%d: server error %s for %s", attempt, max_retries, last_error, current_date)
            if attempt < max_retries:
                wait = BACKOFF_BASE_SECONDS ** attempt
                log.info("Retrying %s in %ds...", current_date, wait)
                sleep_fn(wait)
                continue
            log.error("Giving up on %s after %d attempts: %s", current_date, max_retries, last_error)
            return "error", None, last_error

        # Other 4xx: the request itself is wrong — retrying won't help.
        return "error", None, f"HTTP {response.status_code}"

    return "error", None, last_error or "unknown error"


def download_reports(start_date: date, end_date: date | None = None, reports_dir: Path = PDF_REPORTS_DIR) -> DownloadResult:
    """Download every EOD report between start_date and end_date (inclusive).

    A 404 is expected and normal for weekends/holidays — it's recorded, not
    raised, so a full date range can be requested without pre-filtering
    trading days. Transient failures are retried (see download_one); a
    single persistently-failing day is logged and skipped so the rest of
    the range still gets processed.
    """
    end_date = end_date or date.today()
    result = DownloadResult()

    current = start_date
    while current <= end_date:
        status, path, message = download_one(current, reports_dir)
        if status == "downloaded":
            result.downloaded.append(path)
        elif status == "skipped":
            result.skipped.append(path)
        elif status == "not_found":
            result.not_found.append(current)
        else:
            result.errors.append((current, message or "unknown error"))
        current += timedelta(days=1)

    return result
