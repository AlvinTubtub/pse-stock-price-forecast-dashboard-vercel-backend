"""Multi-feature LSTM forecasting of next-day ΔClose.

Per the capstone paper's methodology:

  - Predicts scaled ΔClose(t+1) = Close(t+1) - Close(t), not the raw
    Close level. Reconstructed price = Close(t) + inverse-transformed
    predicted ΔClose(t+1).
  - Min-Max scaling (features and target) is fit on the training split
    only, then applied to validation/test — no leakage from future data
    into the scaler's min/max.
  - Architecture: one LSTM layer + one linear output layer.
  - Hyperparameters are chosen by grid search over lookback (5/10/20/30),
    hidden units (25/50/100), learning rate (0.01/0.001), and batch size
    (16/32) — 48 configurations, each trained with mini-batches for up to
    200 epochs with early stopping (patience 10) on validation loss, with
    randomness seeded at 42. The configuration with the lowest best
    validation loss is kept as the final model — no separate retrain step,
    since its already-trained weights (checkpointed at its best epoch)
    are the artifact that gets persisted.

Training and inference are separate: ``train()`` fits + evaluates,
``save``/``load`` persist the artifact, and ``predict_next`` runs a
forward pass only — no retraining — which is what the dashboard calls.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from services.evaluation import compute_metrics
from services.feature_engineering import build_full_features, reconstruct_price

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

SEED = 42
EPOCHS = 200
PATIENCE = 10

LOOKBACK_GRID = (5, 10, 20, 30)
HIDDEN_UNITS_GRID = (25, 50, 100)
LEARNING_RATE_GRID = (0.01, 0.001)
BATCH_SIZE_GRID = (16, 32)

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi_14", "macd", "ema_10", "ema_20", "ma_5", "ma_10",
]

MIN_ROWS_FOR_TRAINING = max(LOOKBACK_GRID) + 60  # enough rows for train+val+test at the largest lookback


class _LSTMNet(nn.Module if HAS_TORCH else object):
    """One LSTM layer + one linear output layer, per the paper's spec."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_feature_frame(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Combine raw OHLCV with the shared technical-indicator set and the
    ΔClose target, keeping only the columns this model needs.

    ``require_target=True`` (the default, used for supervised training)
    drops warm-up rows with NaN features *and* the final row (whose
    target is unknown by definition). ``require_target=False`` (used for
    next-day inference) only drops warm-up rows with NaN features, so the
    true most-recent calendar day — needed to predict the actual unseen
    next day — is kept even though its target is NaN.
    """
    engineered = build_full_features(df)
    indicator_cols = [c for c in FEATURE_COLUMNS if c not in ("Open", "High", "Low", "Close", "Volume")]
    frame = df[["Open", "High", "Low", "Close", "Volume"]].join(engineered[indicator_cols])
    frame["target_delta"] = engineered["target_delta"]
    if require_target:
        return frame.dropna().reset_index(drop=True)
    return frame.dropna(subset=[c for c in frame.columns if c != "target_delta"]).reset_index(drop=True)


def _make_sequences(X: np.ndarray, y: np.ndarray, last_close: np.ndarray, seq_len: int):
    """Windows of length ``seq_len`` predicting the ΔClose *after* the
    window's last day; ``last_close`` (aligned to X/y) lets callers
    reconstruct a peso price from each window's prediction."""
    xs, ys, closes = [], [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
        closes.append(last_close[i + seq_len - 1])
    return np.array(xs), np.array(ys), np.array(closes)


def _fallback_result(df: pd.DataFrame):
    """Used when torch isn't installed, or there isn't enough history for
    even the smallest grid-search configuration."""
    close = df["Close"].values.astype("float32")
    metrics = compute_metrics(close[1:], close[:-1])
    next_close = float(close[-1])
    backtest = np.concatenate([[close[0]], close[:-1]]).tolist()
    return None, metrics, next_close, backtest, close[1:].tolist(), close[:-1].tolist()


def _train_one_config(
    X_train, y_train, X_val, y_val, input_size: int, hidden_size: int, lr: float, batch_size: int,
):
    """Trains one grid configuration with mini-batches and early stopping.
    Returns (best_val_loss, best_state_dict)."""
    torch.manual_seed(SEED)
    model = _LSTMNet(input_size=input_size, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(-1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(-1)

    n_train = len(X_train_t)
    generator = torch.Generator().manual_seed(SEED)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val_loss, best_state


def train(df: pd.DataFrame):
    """Returns (artifact, metrics, next_close, backtest_series, test_actual, test_pred).

    ``artifact`` is a plain dict (state dict + scaler stats + architecture
    + hyperparameters) rather than the live nn.Module, so it can be
    persisted with ``torch.save`` and reconstructed later without needing
    this exact object's class already instantiated.
    """
    frame = _build_feature_frame(df)

    if not HAS_TORCH or len(frame) < MIN_ROWS_FOR_TRAINING:
        return _fallback_result(df)

    X_raw_full = frame[FEATURE_COLUMNS].values.astype("float32")
    y_raw_full = frame["target_delta"].values.astype("float32")
    close_full = frame["Close"].values.astype("float32")

    best_overall = None  # (val_loss, config_dict, state, scalers, seq metadata)

    for seq_len in LOOKBACK_GRID:
        X_seq, y_seq, close_seq = _make_sequences(X_raw_full, y_raw_full, close_full, seq_len)
        n = len(X_seq)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_train = n - n_test - n_val
        if n_train < 20:
            continue

        # Min-Max scaling fit on the training portion only.
        x_scaler = MinMaxScaler()
        x_scaler.fit(X_seq[:n_train].reshape(-1, X_seq.shape[-1]))
        y_scaler = MinMaxScaler()
        y_scaler.fit(y_seq[:n_train].reshape(-1, 1))

        def _scale_x(block):
            shape = block.shape
            return x_scaler.transform(block.reshape(-1, shape[-1])).reshape(shape).astype("float32")

        def _scale_y(block):
            return y_scaler.transform(block.reshape(-1, 1)).reshape(-1).astype("float32")

        X_scaled = _scale_x(X_seq)
        y_scaled = _scale_y(y_seq)

        X_train, y_train = X_scaled[:n_train], y_scaled[:n_train]
        X_val, y_val = X_scaled[n_train : n_train + n_val], y_scaled[n_train : n_train + n_val]
        X_test, y_test = X_scaled[n_train + n_val :], y_scaled[n_train + n_val :]
        close_test = close_seq[n_train + n_val :]

        for hidden_size, lr, batch_size in itertools.product(HIDDEN_UNITS_GRID, LEARNING_RATE_GRID, BATCH_SIZE_GRID):
            val_loss, state = _train_one_config(
                X_train, y_train, X_val, y_val,
                input_size=len(FEATURE_COLUMNS), hidden_size=hidden_size, lr=lr, batch_size=batch_size,
            )
            if state is None:
                continue
            if best_overall is None or val_loss < best_overall["val_loss"]:
                best_overall = {
                    "val_loss": val_loss,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "lr": lr,
                    "batch_size": batch_size,
                    "state": state,
                    "x_scaler": x_scaler,
                    "y_scaler": y_scaler,
                    "X_test": X_test, "y_test": y_test, "close_test": close_test,
                    "X_full": X_scaled, "close_full_seq": close_seq,
                }

    if best_overall is None:
        return _fallback_result(df)

    log.info(
        "LSTM grid search best config: seq_len=%d hidden=%d lr=%s batch=%d (val_loss=%.6f)",
        best_overall["seq_len"], best_overall["hidden_size"], best_overall["lr"],
        best_overall["batch_size"], best_overall["val_loss"],
    )

    model = _LSTMNet(input_size=len(FEATURE_COLUMNS), hidden_size=best_overall["hidden_size"])
    model.load_state_dict(best_overall["state"])
    model.eval()

    y_scaler = best_overall["y_scaler"]
    with torch.inference_mode():
        test_pred_scaled = model(torch.tensor(best_overall["X_test"])).squeeze(-1).numpy()
        all_pred_scaled = model(torch.tensor(best_overall["X_full"])).squeeze(-1).numpy()

    test_pred_delta = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).reshape(-1)
    test_actual_delta = y_scaler.inverse_transform(best_overall["y_test"].reshape(-1, 1)).reshape(-1)

    test_pred_close = reconstruct_price(best_overall["close_test"], test_pred_delta)
    test_actual_close = best_overall["close_test"] + test_actual_delta
    metrics = compute_metrics(test_actual_close, test_pred_close, y_train=close_full)

    # Next-day forecast: the last available window including the true
    # most-recent calendar day (whose target is, by definition, unknown —
    # so it must come from the require_target=False frame, not the
    # labeled training frame, which drops that row).
    inference_frame = _build_feature_frame(df, require_target=False)
    last_window = inference_frame[FEATURE_COLUMNS].values.astype("float32")[-best_overall["seq_len"] :]
    last_window_scaled = best_overall["x_scaler"].transform(last_window).astype("float32")
    with torch.inference_mode():
        next_delta_scaled = model(torch.tensor(last_window_scaled).unsqueeze(0)).item()
    next_delta = float(y_scaler.inverse_transform([[next_delta_scaled]])[0, 0])
    next_close = float(df["Close"].iloc[-1] + next_delta)

    all_pred_delta = y_scaler.inverse_transform(all_pred_scaled.reshape(-1, 1)).reshape(-1)
    backtest_vals = reconstruct_price(best_overall["close_full_seq"], all_pred_delta).tolist()
    backtest = [float(frame["Close"].iloc[0])] * best_overall["seq_len"] + backtest_vals  # pad head to align lengths

    artifact = {
        "state_dict": best_overall["state"],
        "input_size": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "seq_len": best_overall["seq_len"],
        "hidden_size": best_overall["hidden_size"],
        "x_scaler": best_overall["x_scaler"],
        "y_scaler": y_scaler,
    }
    return artifact, metrics, next_close, backtest, test_actual_close.tolist(), test_pred_close.tolist()


def save(artifact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact is None or not HAS_TORCH:
        # No-torch fallback path: nothing to persist.
        return
    torch.save(artifact, path)


def load(path):
    return torch.load(path, weights_only=False)


def predict_next(artifact, df: pd.DataFrame) -> float:
    """Forward pass only from an already-trained artifact — no
    retraining, used by the dashboard."""
    model = _LSTMNet(input_size=artifact["input_size"], hidden_size=artifact["hidden_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    frame = _build_feature_frame(df, require_target=False)
    seq_len = artifact["seq_len"]
    X_raw = frame[artifact["feature_columns"]].values.astype("float32")[-seq_len:]
    X_scaled = artifact["x_scaler"].transform(X_raw).astype("float32")

    with torch.inference_mode():
        next_delta_scaled = model(torch.tensor(X_scaled).unsqueeze(0)).item()
    next_delta = float(artifact["y_scaler"].inverse_transform([[next_delta_scaled]])[0, 0])

    return float(df["Close"].iloc[-1] + next_delta)
