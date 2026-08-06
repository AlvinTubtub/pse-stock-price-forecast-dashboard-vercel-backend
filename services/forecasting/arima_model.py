"""ARIMA forecasting with automatic (p, d, q) selection.

Per the capstone paper's methodology:

  1. An Augmented Dickey-Fuller (ADF) test checks stationarity of the
     Close series and informs where the differencing search starts.
  2. Order search is restricted to p <= 3, d <= 2, q <= 3 and is scored by
     expanding-window rolling-origin cross-validation (the same scheme
     used by the Lag-Informed Regression's lambda selection — see
     services/time_series_cv.py), not AIC: for each candidate order, every
     CV fold is scored by *walk-forward one-step* forecasting (fit once on
     the fold's training slice, then forecast one step at a time,
     incorporating each new actual observation via a cheap Kalman-filter
     state update rather than a full MLE refit), and the order with the
     lowest mean validation RMSE wins.
  3. Ljung-Box diagnostics on the final model's residuals check for
     leftover autocorrelation (a well-specified model should show none).
  4. The train/test evaluation split is also scored by walk-forward
     one-step forecasting, for consistency with the CV procedure above.

Differencing is handled by statsmodels' own ``d`` term rather than by
manually differencing the series and re-integrating predictions by hand:
fitting ``ARIMA(close, order=(p, d, q))`` on the *undifferenced* Close
series and calling ``.forecast()``/``.append()`` already returns
reconstructed price-level forecasts (statsmodels un-differences
internally) — mathematically the same "reverse differencing" the paper
describes, without the numerical fragility of doing it by hand.

As with the other models, training and inference are separate: ``train()``
fits + evaluates, ``save``/``load`` persist the fitted statsmodels results
object with joblib, and ``predict_next`` forecasts one step ahead from an
already-fitted model with no retraining.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from services.evaluation import compute_metrics
from services.time_series_cv import expanding_window_splitter

log = logging.getLogger(__name__)

try:
    import joblib
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

DEFAULT_ORDER = (5, 1, 0)  # last-resort fallback if statsmodels is unavailable
MAX_P = 3
MAX_D = 2
MAX_Q = 3
TEST_FRACTION = 0.15
LJUNG_BOX_LAGS = 10


def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test: True if the series is already
    stationary (p-value below alpha), meaning d=0 is a reasonable start."""
    if not HAS_STATSMODELS:
        return False
    try:
        _, pvalue, *_ = adfuller(series.dropna())
        return bool(pvalue < alpha)
    except Exception:  # pragma: no cover - defensive
        log.warning("ADF test failed; assuming non-stationary.", exc_info=True)
        return False


def _candidate_orders(d_guess: int) -> list[tuple[int, int, int]]:
    """All (p, d, q) with p<=MAX_P, d<=MAX_D, q<=MAX_Q, excluding the
    degenerate (0, d, 0) no-op order. ``d_guess`` (from the ADF test) is
    tried first so a good order is usually found before slower ones."""
    d_order = sorted({d_guess, *range(MAX_D + 1)}, key=lambda d: (d != d_guess, d))
    orders = []
    for d in d_order:
        for p in range(MAX_P + 1):
            for q in range(MAX_Q + 1):
                if p == 0 and q == 0:
                    continue
                orders.append((p, d, q))
    return orders


def _walk_forward_forecast(initial_fit, future_actuals: np.ndarray) -> np.ndarray:
    """One-step-ahead walk-forward forecasting: forecast the next point
    from ``initial_fit``, then fold the actual value into the model via a
    cheap state-space update (``append(..., refit=False)`` — no MLE
    re-estimation) before forecasting the point after that, and so on.
    """
    preds = np.empty(len(future_actuals), dtype=float)
    current = initial_fit
    for i, actual in enumerate(future_actuals):
        preds[i] = float(current.forecast(steps=1).iloc[0])
        current = current.append([actual], refit=False)
    return preds


def _cv_score_order(close: pd.Series, order: tuple[int, int, int]) -> float:
    """Mean validation RMSE for ``order`` across expanding-window
    rolling-origin CV folds, each scored by walk-forward one-step
    forecasting. Returns +inf if the order can't be fit on any fold."""
    splitter = expanding_window_splitter(len(close))
    fold_rmses = []
    for train_idx, val_idx in splitter.split(close):
        train_slice = close.iloc[train_idx]
        val_slice = close.iloc[val_idx]
        try:
            fold_fit = ARIMA(train_slice, order=order).fit()
            preds = _walk_forward_forecast(fold_fit, val_slice.values)
        except Exception:
            continue
        fold_rmses.append(float(np.sqrt(np.mean((val_slice.values - preds) ** 2))))
    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


def _select_order(train_series: pd.Series) -> tuple[int, int, int]:
    """(p, d, q) with the lowest mean CV validation RMSE, searched over
    p<=3, d<=2, q<=3."""
    d_guess = 0 if is_stationary(train_series) else 1
    best_order, best_score = DEFAULT_ORDER, float("inf")

    for order in _candidate_orders(d_guess):
        score = _cv_score_order(train_series, order)
        if score < best_score:
            best_score, best_order = score, order

    if best_score == float("inf"):  # pragma: no cover - defensive: nothing fit at all
        log.warning("CV order search found no fittable order; using fallback %s.", DEFAULT_ORDER)
    return best_order


def _ljung_box_pvalue(fitted) -> float:
    """Ljung-Box test on residuals at LJUNG_BOX_LAGS lags — a well-
    specified model should show no significant leftover autocorrelation
    (high p-value)."""
    try:
        result = acorr_ljungbox(fitted.resid, lags=[LJUNG_BOX_LAGS], return_df=True)
        return float(result["lb_pvalue"].iloc[0])
    except Exception:  # pragma: no cover - defensive
        log.warning("Ljung-Box test failed.", exc_info=True)
        return float("nan")


def train(df: pd.DataFrame):
    """Returns (fitted_model, order, metrics, next_close, backtest_series,
    test_actual, test_pred)."""
    close = df["Close"]
    n_test = max(1, int(round(len(close) * TEST_FRACTION)))
    train_series = close.iloc[: len(close) - n_test]
    test_series = close.iloc[len(close) - n_test :]

    if not HAS_STATSMODELS:
        # Fallback: naive-ish drift model, so the app still runs in
        # environments without statsmodels installed.
        y_pred = train_series.iloc[-1] + np.cumsum(np.full(len(test_series), train_series.diff().mean()))
        metrics = compute_metrics(test_series.values, y_pred, y_train=train_series.values)
        next_close = float(close.iloc[-1] + train_series.diff().mean())
        backtest = close.shift(1).bfill().tolist()
        return None, DEFAULT_ORDER, metrics, next_close, backtest, test_series.values.tolist(), list(y_pred)

    order = _select_order(train_series)
    log.info("Selected ARIMA order %s (CV-validated RMSE)", order)

    initial_fit = ARIMA(train_series, order=order).fit()
    test_pred = _walk_forward_forecast(initial_fit, test_series.values)
    metrics = compute_metrics(test_series.values, test_pred, y_train=train_series.values)

    full_model = ARIMA(close, order=order).fit()
    metrics["ljung_box_pvalue"] = f"{_ljung_box_pvalue(full_model):.4f}"
    next_close = float(full_model.forecast(steps=1).iloc[0])

    # In-sample one-step-ahead fitted values, for the backtest chart.
    backtest = full_model.predict(start=1, end=len(close) - 1, typ="levels")
    backtest = pd.concat([pd.Series([close.iloc[0]]), backtest]).reset_index(drop=True).tolist()

    return full_model, order, metrics, next_close, backtest, test_series.values.tolist(), test_pred.tolist()


def save(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        # No-statsmodels fallback path: nothing to persist.
        return
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)


def predict_next(model) -> float:
    """Forecast one step ahead from an already-fitted model — no
    retraining, used by the dashboard."""
    return float(model.forecast(steps=1).iloc[0])
