"""Forecasting models package.

    services/forecasting/
        lag_regression.py    training-only StandardScaler + PACF-assisted
                              lag selection + LASSO (final estimator,
                              lambda via 5-fold expanding-window CV)
        arima_model.py        ADF test + CV-based (p, d, q) selection
                              (p<=3, d<=2, q<=3) + walk-forward forecasting
        lstm_model.py          single-layer LSTM, grid-searched over
                              lookback/hidden units/lr/batch size

All three predict next-day ΔClose = Close(t+1) - Close(t) rather than the
raw Close level, and reconstruct a peso price as Close(t) + predicted
ΔClose(t+1) before any metric is computed — see
services/feature_engineering.py.

Each module exposes the same four functions: ``train(df)``, ``save``,
``load``, and ``predict_next`` — see services/model_selector.py for the
training orchestration that calls all three per ticker, persists
everything under models/ + prediction_cache/, and runs the cross-model
statistical-significance suite (statistical_tests.json).

Training only ever runs inside the automated pipeline (run_pipeline.py,
triggered by .github/workflows/update_pipeline.yml) — never inside
Streamlit, which only loads what model_selector.py already cached.
"""
from __future__ import annotations

from . import arima_model, lag_regression, lstm_model

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}

__all__ = ["MODEL_LABELS", "lag_regression", "arima_model", "lstm_model"]
