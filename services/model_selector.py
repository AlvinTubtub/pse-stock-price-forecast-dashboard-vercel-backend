"""Training orchestration + best-model selection.

This is the one place that trains all three forecasting models for every
ticker, evaluates them, saves each trained model to disk, caches the full
results (metrics/predictions/backtests) for the dashboard to load without
retraining, writes ``best_models.json`` mapping each ticker to whichever
model had the lowest test-set RMSE, and — after every ticker has been
trained — runs the capstone paper's cross-model statistical-significance
suite (Diebold-Mariano/HLN within each company, Friedman + Holm-adjusted
Wilcoxon across companies, and the best-model consistency check) and
writes ``statistical_tests.json``.

Called exclusively by services/pdf_pipeline/pipeline.py after every
successful merge, and by the standalone ``python -m services.model_selector``
entrypoint for local/manual runs — never from inside the Streamlit
dashboard, which only ever reads what this module wrote.

Directory layout produced:

    models/
        lag_regression/<TICKER>.pkl
        arima/<TICKER>.pkl
        lstm/<TICKER>.pth
    prediction_cache/<TICKER>.json   # cached metrics + predictions for ui/data.py
    best_models.json                 # {"BDO": "LSTM", "MER": "ARIMA", ...}
    statistical_tests.json           # cross-model significance test results
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.evaluation import evaluate_naive, run_cross_model_statistical_tests, select_best_model
from services.forecasting import MODEL_LABELS, arima_model, lag_regression, lstm_model

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
LAG_MODELS_DIR = MODELS_DIR / "lag_regression"
ARIMA_MODELS_DIR = MODELS_DIR / "arima"
LSTM_MODELS_DIR = MODELS_DIR / "lstm"
PREDICTION_CACHE_DIR = BASE_DIR / "prediction_cache"
BEST_MODELS_PATH = BASE_DIR / "best_models.json"
STATISTICAL_TESTS_PATH = BASE_DIR / "statistical_tests.json"

STAT_MODEL_KEYS = ("lag_reg", "arima", "lstm")
MIN_CONSISTENCY_COMPANIES = 8


def _ensure_dirs() -> None:
    for d in (LAG_MODELS_DIR, ARIMA_MODELS_DIR, LSTM_MODELS_DIR, PREDICTION_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _align_test_errors(test_sets: dict[str, tuple[list[float], list[float]]]) -> dict[str, np.ndarray]:
    """Each model's (test_actual, test_pred) pair may have a different
    length — regression/ARIMA split on the raw row count, the LSTM splits
    on however many windows its grid-search-chosen lookback produced.
    Truncates every model to the shortest common length, right-aligned
    (all splits end on the same most-recent calendar day), so pairwise
    forecast errors are comparable date-for-date, as the Diebold-Mariano
    test requires.
    """
    min_len = min(len(actual) for actual, _pred in test_sets.values())
    errors = {}
    for model_key, (actual, pred) in test_sets.items():
        actual_arr = np.asarray(actual[-min_len:], dtype=float)
        pred_arr = np.asarray(pred[-min_len:], dtype=float)
        errors[model_key] = actual_arr - pred_arr
    return errors


def train_symbol(symbol: str, df: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    """Trains + evaluates all three models for one ticker, saves each to
    disk, and returns (result, test_errors).

    ``result`` is cached for the dashboard (same shape as the legacy
    ``run_all_models`` return value, plus whatever additive diagnostic
    fields each model's metrics dict now includes, e.g. ARIMA's
    ``ljung_box_pvalue``). ``test_errors`` is {model_key: np.ndarray of
    (actual - predicted) reconstructed-price errors}, aligned to a common
    test window — used by train_and_select_all for the cross-model
    statistical tests, not persisted to the dashboard cache.
    """
    log.info("Training models for %s (%d rows)...", symbol, len(df))

    lag_artifact, lag_metrics, lag_next, lag_backtest, lag_test_actual, lag_test_pred = lag_regression.train(df)
    lag_regression.save(lag_artifact, LAG_MODELS_DIR / f"{symbol}.pkl")

    arima_fitted, order, arima_metrics, arima_next, arima_backtest, arima_test_actual, arima_test_pred = arima_model.train(df)
    arima_model.save(arima_fitted, ARIMA_MODELS_DIR / f"{symbol}.pkl")
    log.info("%s ARIMA order selected: %s", symbol, order)

    lstm_artifact, lstm_metrics, lstm_next, lstm_backtest, lstm_test_actual, lstm_test_pred = lstm_model.train(df)
    lstm_model.save(lstm_artifact, LSTM_MODELS_DIR / f"{symbol}.pth")

    naive_metrics = evaluate_naive(df)

    result = {
        "metrics": {
            "lag_reg": lag_metrics,
            "arima": arima_metrics,
            "lstm": lstm_metrics,
            "naive": naive_metrics,
        },
        "next_close": {
            "lag": round(lag_next, 2),
            "arima": round(arima_next, 2),
            "lstm": round(lstm_next, 2),
        },
        "backtest30": lag_backtest[-30:] if len(lag_backtest) >= 30 else lag_backtest,
        "backtest_by_model": {
            "Lag-Informed Regression": lag_backtest,
            "ARIMA": arima_backtest,
            "LSTM": lstm_backtest,
        },
    }

    test_errors = _align_test_errors({
        "lag_reg": (lag_test_actual, lag_test_pred),
        "arima": (arima_test_actual, arima_test_pred),
        "lstm": (lstm_test_actual, lstm_test_pred),
    })
    return result, test_errors


def train_and_select_all(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Trains + saves models for every ticker CSV in ``raw_dir``, caches
    each ticker's results for the dashboard, writes best_models.json, and
    runs + saves the cross-model statistical-significance suite.

    Returns the {symbol: best_model_label} mapping. Bad/unreadable CSVs
    are logged and skipped rather than aborting the whole run.
    """
    _ensure_dirs()
    best_models: dict[str, str] = {}
    test_errors_by_symbol: dict[str, dict[str, np.ndarray]] = {}
    rmse_by_symbol: dict[str, dict[str, float]] = {}

    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        log.warning("No CSVs found in %s — nothing to train.", raw_dir)
        return best_models

    for csv_path in csv_paths:
        symbol = csv_path.stem
        try:
            df = validate_ohlcv_csv(csv_path)
        except CSVValidationError as exc:
            log.warning("Skipping %s — failed OHLCV validation: %s", symbol, exc)
            continue
        except Exception:
            log.exception("Skipping %s — unexpected error while loading CSV.", symbol)
            continue

        try:
            result, test_errors = train_symbol(symbol, df)
        except Exception:
            log.exception("Training failed for %s — skipping.", symbol)
            continue

        best_key = select_best_model(result["metrics"], list(STAT_MODEL_KEYS))
        best_models[symbol] = MODEL_LABELS[best_key]

        (PREDICTION_CACHE_DIR / f"{symbol}.json").write_text(json.dumps(result, indent=2))
        log.info("%s best model: %s (RMSE %s)", symbol, MODEL_LABELS[best_key], result["metrics"][best_key]["rmse"])

        test_errors_by_symbol[symbol] = test_errors
        rmse_by_symbol[symbol] = {k: float(result["metrics"][k]["rmse"]) for k in STAT_MODEL_KEYS}

    BEST_MODELS_PATH.write_text(json.dumps(best_models, indent=2, sort_keys=True))
    log.info("Wrote %s (%d tickers)", BEST_MODELS_PATH, len(best_models))

    if test_errors_by_symbol:
        try:
            stats = run_cross_model_statistical_tests(
                test_errors_by_symbol, rmse_by_symbol,
                model_keys=STAT_MODEL_KEYS, min_companies=MIN_CONSISTENCY_COMPANIES,
            )
            STATISTICAL_TESTS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True))
            log.info(
                "Wrote %s (Friedman p=%.4g, best-model consistency: %s on %d/%d companies)",
                STATISTICAL_TESTS_PATH,
                stats["friedman"]["p_value"],
                stats["best_model_consistency"]["dominant_model"],
                stats["best_model_consistency"]["dominant_count"],
                stats["best_model_consistency"]["total_companies"],
            )
        except Exception:
            log.exception("Cross-model statistical tests failed — best_models.json is still valid, but statistical_tests.json was not updated.")

    return best_models


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mapping = train_and_select_all()
    print(json.dumps(mapping, indent=2, sort_keys=True))
