"""Exports static JSON artifacts for the Next.js frontend from the
existing pipeline outputs.

This is the ONLY new piece of Python needed to satisfy the target
architecture: everything upstream (scrape -> clean -> features -> train
-> evaluate -> predict) already exists in services/ and is untouched.
This script just re-shapes what run_pipeline.py already produced
(data/raw/*.csv, prediction_cache/*.json, best_models.json,
latest_processed.json, statistical_tests.json) into the flat JSON
contract the frontend reads at build/runtime from public/forecasts/.

Run at the end of the GitHub Actions job, right before the commit step:

    python scripts/export_forecast_artifacts.py

It writes:
    public/forecasts/dashboard.json
    public/forecasts/latest.json
    public/forecasts/metrics.json
    public/forecasts/companies.json
    public/forecasts/company/<SYMBOL>.json
    public/forecasts/history/<SYMBOL>.json

Nothing here trains, scrapes, or infers anything — read-only reshaping,
same contract as the old ui/data.py, so it can safely run on the same
GitHub Actions runner that just finished the pipeline (no extra deps
beyond the stdlib + pandas, both already installed for the pipeline).
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / "prediction_cache"
BEST_MODELS_PATH = BASE_DIR / "best_models.json"
LATEST_PROCESSED_PATH = BASE_DIR / "latest_processed.json"
STAT_TESTS_PATH = BASE_DIR / "statistical_tests.json"

OUT_DIR = BASE_DIR / "public" / "forecasts"
COMPANY_OUT_DIR = OUT_DIR / "company"
HISTORY_OUT_DIR = OUT_DIR / "history"

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}

SECTORS = {
    "FINANCIALS": "Financials",
    "INDUSTRIAL": "Industrial",
    "PROPERTY": "Property",
    "SERVICES": "Services",
    "MINING_OIL": "Mining and Oil",
}

# Mirrors services/data_loader.py COMPANY_META — kept in sync manually
# since it rarely changes (adding a ticker is a deliberate decision).
COMPANY_META = [
    {"symbol": "SECB", "name": "Security Bank Corporation", "sector": SECTORS["FINANCIALS"]},
    {"symbol": "BPI", "name": "Bank of the Philippine Islands", "sector": SECTORS["FINANCIALS"]},
    {"symbol": "MBT", "name": "Metropolitan Bank & Trust Co.", "sector": SECTORS["FINANCIALS"]},
    {"symbol": "MER", "name": "Manila Electric Company", "sector": SECTORS["INDUSTRIAL"]},
    {"symbol": "JFC", "name": "Jollibee Foods Corporation", "sector": SECTORS["INDUSTRIAL"]},
    {"symbol": "SHLPH", "name": "Pilipinas Shell Petroleum Corp.", "sector": SECTORS["INDUSTRIAL"]},
    {"symbol": "MEG", "name": "Megaworld Corporation", "sector": SECTORS["PROPERTY"]},
    {"symbol": "ALI", "name": "Ayala Land, Inc.", "sector": SECTORS["PROPERTY"]},
    {"symbol": "SMPH", "name": "SM Prime Holdings, Inc.", "sector": SECTORS["PROPERTY"]},
    {"symbol": "GLO", "name": "Globe Telecom, Inc.", "sector": SECTORS["SERVICES"]},
    {"symbol": "PGOLD", "name": "Puregold Price Club, Inc.", "sector": SECTORS["SERVICES"]},
    {"symbol": "ICT", "name": "Intl. Container Terminal Services", "sector": SECTORS["SERVICES"]},
    {"symbol": "APX", "name": "Apex Mining Co., Inc.", "sector": SECTORS["MINING_OIL"]},
    {"symbol": "NIKL", "name": "Nickel Asia Corporation", "sector": SECTORS["MINING_OIL"]},
    {"symbol": "SCC", "name": "Semirara Mining and Power Corp.", "sector": SECTORS["MINING_OIL"]},
]


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def ohlcv_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": row.Date.strftime("%Y-%m-%d") if hasattr(row.Date, "strftime") else str(row.Date),
            "open": round(float(row.Open), 2),
            "high": round(float(row.High), 2),
            "low": round(float(row.Low), 2),
            "close": round(float(row.Close), 2),
            "volume": int(row.Volume),
        }
        for row in df.itertuples()
    ]


def best_model_id(metrics: dict) -> str:
    """Lowest MASE wins (matches services/model_selector.py convention)."""
    candidates = [(mid, float(m["mase"])) for mid, m in metrics.items() if mid != "naive"]
    return min(candidates, key=lambda x: x[1])[0] if candidates else "naive"


def main() -> None:
    COMPANY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_models = load_json(BEST_MODELS_PATH, {})
    latest_processed = load_json(LATEST_PROCESSED_PATH, {})
    statistical_tests = load_json(STAT_TESTS_PATH, {})

    generated_at = datetime.now(timezone.utc).isoformat()
    forecast_date = (latest_processed.get("last_run_at") or generated_at).split("T")[0]

    companies_out = []
    per_company_metrics = {}
    all_model_metrics: dict[str, list[dict]] = {mid: [] for mid in MODEL_LABELS}
    missing: list[str] = []

    for meta in COMPANY_META:
        symbol = meta["symbol"]
        csv_path = DATA_DIR / f"{symbol}.csv"
        cache_path = CACHE_DIR / f"{symbol}.json"

        if not csv_path.exists() or not cache_path.exists():
            missing.append(symbol)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        cache = json.loads(cache_path.read_text())

        metrics = cache["metrics"]
        next_close = cache["next_close"]
        backtest_actual = cache.get("backtest30", [])
        backtest_by_model = cache.get("backtest_by_model", {})

        previous_close = round(float(df["Close"].iloc[-1]), 2)
        winning_model_label = best_models.get(symbol) or MODEL_LABELS[best_model_id(metrics)]
        winning_model_id = next(
            (mid for mid, label in MODEL_LABELS.items() if label == winning_model_label),
            best_model_id(metrics),
        )
        # next_close keys are shortened ("lag" not "lag_reg") in the cache
        next_close_key = {"lag_reg": "lag", "arima": "arima", "lstm": "lstm"}.get(winning_model_id)
        predicted_close = round(float(next_close.get(next_close_key, previous_close)), 2) if next_close_key else previous_close

        peso_change = round(predicted_close - previous_close, 2)
        pct_change = round((peso_change / previous_close) * 100, 2) if previous_close else 0.0
        r2 = float(metrics.get(winning_model_id, {}).get("r2", 0))
        confidence = round(max(0.0, min(1.0, r2)) * 100, 1)

        for mid in MODEL_LABELS:
            if mid in metrics:
                all_model_metrics[mid].append(metrics[mid])

        history = ohlcv_records(df)

        company_detail = {
            "symbol": symbol,
            "name": meta["name"],
            "sector": meta["sector"],
            "previousClose": previous_close,
            "predictedClose": predicted_close,
            "pesoChange": peso_change,
            "pctChange": pct_change,
            "direction": "bullish" if peso_change >= 0 else "bearish",
            "model": winning_model_label,
            "confidence": confidence,
            "metrics": metrics,
            "nextClose": next_close,
            "ohlcv": history[-180:],
            "backtestActual": backtest_actual,
            "backtestByModel": backtest_by_model,
        }
        (COMPANY_OUT_DIR / f"{symbol}.json").write_text(json.dumps(company_detail, indent=2))
        (HISTORY_OUT_DIR / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "ohlcv": history}, indent=2))

        per_company_metrics[symbol] = {"metrics": metrics, "bestModel": winning_model_label}

        companies_out.append({
            "symbol": symbol,
            "name": meta["name"],
            "sector": meta["sector"],
            "latestClose": previous_close,
            "predictedClose": predicted_close,
            "pctChange": pct_change,
            "direction": company_detail["direction"],
            "bestModel": winning_model_label,
            "confidence": confidence,
        })

    (OUT_DIR / "companies.json").write_text(json.dumps(companies_out, indent=2))

    aggregate = {}
    for mid, rows in all_model_metrics.items():
        if rows:
            aggregate[mid] = {
                k: round(statistics.mean(float(r[k]) for r in rows), 4)
                for k in ["rmse", "mae", "mase", "r2"]
            }
        else:
            aggregate[mid] = {"rmse": 0, "mae": 0, "mase": 0, "r2": 0}

    non_naive = {k: v for k, v in aggregate.items() if k != "naive"}
    best_model_overall = min(non_naive, key=lambda k: non_naive[k]["mase"]) if non_naive else "naive"
    worst_model_overall = max(non_naive, key=lambda k: non_naive[k]["mase"]) if non_naive else "naive"

    metrics_json = {
        "generatedAt": generated_at,
        "aggregate": {MODEL_LABELS[k]: v for k, v in aggregate.items()},
        "bestModel": MODEL_LABELS[best_model_overall],
        "worstModel": MODEL_LABELS[worst_model_overall],
        "perCompany": per_company_metrics,
        "statisticalTests": statistical_tests,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    gainers = [c for c in companies_out if c["pctChange"] > 0]
    losers = [c for c in companies_out if c["pctChange"] < 0]
    top_gainer = max(companies_out, key=lambda c: c["pctChange"]) if companies_out else None
    top_loser = min(companies_out, key=lambda c: c["pctChange"]) if companies_out else None
    sector_counts: dict[str, int] = {}
    for c in companies_out:
        sector_counts[c["sector"]] = sector_counts.get(c["sector"], 0) + 1

    dashboard_json = {
        "generatedAt": generated_at,
        "forecastDate": forecast_date,
        "lastRunAt": latest_processed.get("last_run_at"),
        "status": latest_processed.get("status", "unknown"),
        "totalCompanies": len(companies_out),
        "missingCompanies": missing,
        "sectors": [{"name": name, "count": count} for name, count in sorted(sector_counts.items())],
        "marketSummary": {
            "gainers": len(gainers),
            "losers": len(losers),
            "unchanged": len(companies_out) - len(gainers) - len(losers),
        },
        "topGainer": top_gainer,
        "topLoser": top_loser,
    }
    (OUT_DIR / "dashboard.json").write_text(json.dumps(dashboard_json, indent=2))

    latest_json = {
        "generatedAt": generated_at,
        "forecastDate": forecast_date,
        "lastRunAt": latest_processed.get("last_run_at"),
        "status": latest_processed.get("status", "unknown"),
    }
    (OUT_DIR / "latest.json").write_text(json.dumps(latest_json, indent=2))

    print(f"Exported {len(companies_out)} companies to {OUT_DIR} ({len(missing)} missing: {missing})")


if __name__ == "__main__":
    main()
