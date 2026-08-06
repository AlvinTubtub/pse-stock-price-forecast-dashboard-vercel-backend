"""Feature engineering for next-day price forecasting.

Shared by all three forecasting models (Lag-Informed Regression, ARIMA's
stationarity checks, and the LSTM's input windowing) so every model sees
the exact same feature definitions — no drift between what regression
trains on and what the LSTM feeds its sequences.

Per the capstone paper's methodology, the forecast *target* is the
next-day price **change** rather than the next-day price level:

    delta_close(t+1) = Close(t+1) - Close(t)
    Predicted Close(t+1) = Close(t) + Predicted delta_close(t+1)

``build_full_features`` exposes both ``target`` (legacy: same-day Close
level — see the note on ``build_lag_features`` below — kept only for
backward compatibility with anything still reading it directly) and
``target_delta`` (next-day ΔClose — what the regression, ARIMA
differencing, and LSTM models now train against). Use
``reconstruct_price`` to turn a ΔClose prediction back into a peso price.

Tiers exposed:

``build_lag_features``
    The original 7 lag/moving-average columns. Kept unchanged for backward
    compatibility with anything still importing it directly.

``build_technical_indicators``
    The technical-indicator set (EMA, RSI, MACD, Bollinger Bands, returns,
    volatility, spreads).

``build_return_features``
    The expanded return-based feature set added for the paper's
    lag-informed regression methodology: lagged returns (1-20), rolling
    return mean/volatility, high-low range, log volume, and rolling
    volume mean.

``build_full_features``
    All of the above, concatenated, plus both target definitions.

All indicators are computed using only same-day-or-earlier data (no
look-ahead), so every row is a valid predictor for that row's *next* day
close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_COLUMNS = ["lag_1", "lag_2", "lag_3", "lag_5", "ma_5", "ma_10", "volume_ma_5"]

TECHNICAL_COLUMNS = [
    "ema_10",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "daily_return",
    "rolling_volatility",
    "hl_spread",
    "oc_spread",
]

# Lagged-return columns, individually addressable (return_lag_1..return_lag_20)
# so PACF-assisted lag selection (services/forecasting/lag_regression.py) can
# pick a subset of these by name.
RETURN_LAG_COLUMNS = [f"return_lag_{lag}" for lag in range(1, 21)]

RETURN_COLUMNS = RETURN_LAG_COLUMNS + [
    "return_mean_5",
    "return_mean_10",
    "return_mean_20",
    "return_vol_5",
    "return_vol_10",
    "return_vol_20",
    "hl_range_pct",
    "log_volume",
    "volume_mean_10",
    "volume_mean_20",
]

FULL_FEATURE_COLUMNS = LAG_COLUMNS + TECHNICAL_COLUMNS + RETURN_COLUMNS


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a validated OHLCV dataframe (sorted by Date), return a dataframe
    of lag/moving-average features plus ``target`` = that same row's own
    Close level (NOT shifted to the next day, despite lag_1..lag_5/ma_5/
    ma_10 themselves being built from shift(1) and later — this column is
    kept only for backward compatibility with any external code still
    reading it directly; new code should use ``target_delta`` from
    ``build_full_features`` instead, which correctly targets
    Close(t+1) - Close(t)).

    Rows without enough history for the longest lag/window are dropped.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["lag_1"] = close.shift(1)
    out["lag_2"] = close.shift(2)
    out["lag_3"] = close.shift(3)
    out["lag_5"] = close.shift(5)
    out["ma_5"] = close.shift(1).rolling(5).mean()
    out["ma_10"] = close.shift(1).rolling(10).mean()
    out["volume_ma_5"] = df["Volume"].shift(1).rolling(5).mean()
    out["target"] = close

    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral RSI where undefined (e.g. no losses yet)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + num_std * std, mid - num_std * std


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return the expanded technical-indicator columns, aligned to df.index.

    Indicators use each row's own Close/High/Low/Open — legitimate for
    predicting that row's *next* day close since nothing here reaches
    forward in time.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["rsi_14"] = _rsi(close, 14)

    macd_line, macd_signal = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal

    bb_upper, bb_lower = _bollinger(close)
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    out["daily_return"] = close.pct_change()
    out["rolling_volatility"] = out["daily_return"].rolling(10).std()
    out["hl_spread"] = df["High"] - df["Low"]
    out["oc_spread"] = df["Open"] - df["Close"]

    return out


def build_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return-based feature set: lagged daily returns (1-20), rolling
    return mean/volatility (5/10/20), high-low range as a fraction of
    Close, log volume, and rolling volume means.

    All computed from same-day-or-earlier data only.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    daily_return = close.pct_change()

    for lag in range(1, 21):
        out[f"return_lag_{lag}"] = daily_return.shift(lag)

    for window in (5, 10, 20):
        out[f"return_mean_{window}"] = daily_return.shift(1).rolling(window).mean()
        out[f"return_vol_{window}"] = daily_return.shift(1).rolling(window).std()

    out["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    out["log_volume"] = np.log1p(df["Volume"].clip(lower=0))
    out["volume_mean_10"] = df["Volume"].shift(1).rolling(10).mean()
    out["volume_mean_20"] = df["Volume"].shift(1).rolling(20).mean()

    return out


def build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features + technical indicators + return features, plus both
    target definitions:

    - ``target``: next day's Close level (legacy, kept for backward
      compatibility with any existing consumer that reads it directly).
    - ``target_delta``: next day's ΔClose = Close(t+1) - Close(t) — the
      target the forecasting models now train against per the paper's
      methodology. Reconstruct a peso price with ``reconstruct_price``.
    """
    lag = build_lag_features(df)
    technical = build_technical_indicators(df)
    returns = build_return_features(df)
    target_close = lag.pop("target")
    target_delta = df["Close"].diff().shift(-1)
    target_delta.index = df.index

    out = pd.concat([lag, technical, returns], axis=1)
    out["target"] = target_close
    out["target_delta"] = target_delta
    return out


def reconstruct_price(last_close: np.ndarray | pd.Series, predicted_delta: np.ndarray | pd.Series) -> np.ndarray:
    """Predicted Close(t+1) = Close(t) + Predicted ΔClose(t+1).

    ``last_close`` and ``predicted_delta`` must be aligned element-wise
    (same length, same row order).
    """
    return np.asarray(last_close, dtype=float) + np.asarray(predicted_delta, dtype=float)


def train_test_split_frame(features: pd.DataFrame, test_frac: float = 0.15):
    """Chronological split — never shuffles, to avoid look-ahead leakage."""
    features = features.dropna()
    n = len(features)
    n_test = max(1, int(round(n * test_frac)))
    train = features.iloc[: n - n_test]
    test = features.iloc[n - n_test :]
    return train, test
