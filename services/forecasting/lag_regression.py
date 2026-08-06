"""Lag-Informed Regression: training-only StandardScaler -> PACF-assisted
lag selection -> LASSO (final estimator, lambda selected by 5-fold
expanding-window rolling-origin cross-validation), predicting next-day
ΔClose and reconstructing the peso price from the last known Close.

Per the capstone paper's methodology:

  - The target is ΔClose(t+1) = Close(t+1) - Close(t), not the raw Close
    level. Predicted Close(t+1) = Close(t) + Predicted ΔClose(t+1).
  - PACF (partial autocorrelation) on the training return series decides
    which of the 20 lagged-return candidate features are informative
    before the model ever sees them.
  - LASSO *is* the final estimator — there is no secondary OLS refit on
    the selected features. Regularization strength (lambda / alpha) is
    chosen by 5-fold expanding-window rolling-origin CV
    (services/time_series_cv.py), matching the same validation scheme
    used for the ARIMA order search.
  - All reported metrics are computed on reconstructed peso prices, never
    on the raw ΔClose scale.

Training and inference are deliberately separate (see model_selector.py /
services/pdf_pipeline): ``train()`` fits and evaluates a model, ``save()``
/``load()`` persist it with joblib, and ``predict_next()`` produces a
forecast from an already-trained artifact without any retraining — the
Streamlit dashboard only ever calls ``load`` + ``predict_next``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from services.evaluation import compute_metrics
from services.feature_engineering import (
    FULL_FEATURE_COLUMNS,
    RETURN_LAG_COLUMNS,
    build_full_features,
    reconstruct_price,
)
from services.time_series_cv import expanding_window_splitter

try:
    from statsmodels.tsa.stattools import pacf as _pacf

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

log = logging.getLogger(__name__)

MAX_PACF_LAG = 20
PACF_FALLBACK_LAGS = (1, 2, 3, 5)  # used if PACF is unavailable or finds nothing significant
TEST_FRACTION = 0.15


@dataclass
class LagRegressionArtifact:
    """Everything needed to reproduce a prediction without retraining."""

    scaler: StandardScaler
    model: LassoCV
    candidate_features: list[str] = field(default_factory=list)
    pacf_selected_lags: list[int] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)  # non-zero LASSO coefficients, for reporting


def pacf_select_lags(delta_series: pd.Series, max_lag: int = MAX_PACF_LAG, alpha: float = 0.05) -> list[int]:
    """PACF-assisted lag selection: returns the lag numbers (1..max_lag)
    whose partial autocorrelation with the training ΔClose series is
    statistically significant at ``alpha``, using the same asymptotic
    +/-1.96/sqrt(n) confidence band a PACF plot would draw.

    Falls back to a small fixed lag set if statsmodels is unavailable or
    the series is too short/flat for PACF to find anything significant —
    LASSO will still zero out whatever isn't useful.
    """
    series = pd.Series(delta_series).dropna()
    if not HAS_STATSMODELS or len(series) < max_lag * 3:
        return list(PACF_FALLBACK_LAGS)

    try:
        values = _pacf(series, nlags=max_lag, method="ywm")
    except Exception:  # pragma: no cover - defensive: never let this crash training
        log.warning("PACF computation failed; falling back to default lag set.", exc_info=True)
        return list(PACF_FALLBACK_LAGS)

    threshold = norm.ppf(1 - alpha / 2) / np.sqrt(len(series))
    significant = [lag for lag in range(1, max_lag + 1) if abs(values[lag]) > threshold]
    return significant or list(PACF_FALLBACK_LAGS)


def _candidate_columns(pacf_lags: list[int]) -> list[str]:
    """Full feature set, but restricted to only the PACF-significant
    lagged-return columns (the other 19-or-fewer of the 20 candidates are
    dropped before the model ever sees them)."""
    dropped_lags = set(RETURN_LAG_COLUMNS) - {f"return_lag_{lag}" for lag in pacf_lags}
    return [c for c in FULL_FEATURE_COLUMNS if c not in dropped_lags]


def _fit_lasso(X_scaled: np.ndarray, y: np.ndarray, n_samples: int) -> LassoCV:
    """LASSO with lambda (alpha) selected by 5-fold expanding-window
    rolling-origin CV — this *is* the final estimator, no OLS refit."""
    cv = expanding_window_splitter(n_samples)
    return LassoCV(cv=cv, random_state=42, max_iter=50_000, tol=1e-3).fit(X_scaled, y)


def train(df: pd.DataFrame) -> tuple[LagRegressionArtifact, dict, float, list[float], list[float], list[float]]:
    """Returns (artifact, metrics, next_close, backtest_series, test_actual, test_pred).

    ``test_actual``/``test_pred`` are the held-out reconstructed peso
    prices and predictions, in chronological order — used by
    model_selector.py to run the cross-model statistical tests (Diebold-
    Mariano, Friedman, Wilcoxon) against a common test window.
    """
    features_full = build_full_features(df)  # keeps every row, incl. the final one (unknown target)
    features = features_full.dropna()  # rows with a known ΔClose(t+1) target, for training/testing
    n = len(features)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    train_df = features.iloc[: n - n_test]
    test_df = features.iloc[n - n_test :]

    pacf_lags = pacf_select_lags(train_df["target_delta"])
    x_cols = _candidate_columns(pacf_lags)

    scaler = StandardScaler().fit(train_df[x_cols])
    X_train_scaled = scaler.transform(train_df[x_cols])
    X_test_scaled = scaler.transform(test_df[x_cols])

    model = _fit_lasso(X_train_scaled, train_df["target_delta"].values, len(train_df))
    selected_features = [name for name, coef in zip(x_cols, model.coef_) if coef != 0]

    test_pred_delta = model.predict(X_test_scaled)
    test_pred_close = reconstruct_price(df.loc[test_df.index, "Close"], test_pred_delta)
    # Actual Close(t+1): reconstructed from the *actual* ΔClose(t+1), not
    # the legacy same-day `target` column (which is Close(t), not
    # Close(t+1) — see feature_engineering.py's note on that column).
    test_actual_close = reconstruct_price(df.loc[test_df.index, "Close"], test_df["target_delta"].values)

    train_actual_close = reconstruct_price(df.loc[train_df.index, "Close"], train_df["target_delta"].values)
    metrics = compute_metrics(test_actual_close, test_pred_close, y_train=train_actual_close)

    # Refit on all available rows (fresh CV-selected lambda too) so the
    # persisted model and next-day forecast use the most recent data.
    full_scaler = StandardScaler().fit(features[x_cols])
    X_full_scaled = full_scaler.transform(features[x_cols])
    full_model = _fit_lasso(X_full_scaled, features["target_delta"].values, len(features))

    last_row = features_full.iloc[[-1]][x_cols].ffill()
    last_scaled = full_scaler.transform(last_row)
    next_delta = float(full_model.predict(last_scaled)[0])
    next_close = float(df["Close"].iloc[-1] + next_delta)

    backtest_scaled = full_scaler.transform(features_full[x_cols].bfill())
    backtest_delta = full_model.predict(backtest_scaled)
    backtest = reconstruct_price(df.loc[features_full.index, "Close"], backtest_delta).tolist()

    artifact = LagRegressionArtifact(
        scaler=full_scaler,
        model=full_model,
        candidate_features=x_cols,
        pacf_selected_lags=pacf_lags,
        selected_features=selected_features,
    )
    return artifact, metrics, next_close, backtest, test_actual_close.tolist(), test_pred_close.tolist()


def save(artifact: LagRegressionArtifact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path) -> LagRegressionArtifact:
    return joblib.load(path)


def predict_next(artifact: LagRegressionArtifact, df: pd.DataFrame) -> float:
    """Predict next-day Close from an already-trained artifact — no
    retraining, used by the dashboard."""
    features = build_full_features(df)
    last_row = features.iloc[[-1]][artifact.candidate_features].ffill()
    scaled = artifact.scaler.transform(last_row)
    next_delta = float(artifact.model.predict(scaled)[0])
    return float(df["Close"].iloc[-1] + next_delta)
