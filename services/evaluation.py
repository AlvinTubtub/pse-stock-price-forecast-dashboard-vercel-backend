"""Shared model-evaluation utilities.

One metrics implementation used by every forecasting model (Lag-Informed
Regression, ARIMA, LSTM) and by the naive baseline, so RMSE/MAE/MASE/R²
are always computed the same way regardless of which model produced the
predictions. All metrics are computed on reconstructed peso prices, never
on a differenced/scaled internal target.

Also implements the capstone paper's cross-model statistical-significance
suite, run once per pipeline update across every ticker
(services/model_selector.py writes the results to statistical_tests.json):

  - Diebold-Mariano test, Newey-West HAC variance + Harvey-Leybourne-
    Newbold (HLN) small-sample correction — pairwise, *within* each
    company.
  - Holm-Bonferroni correction for the resulting multiple comparisons.
  - Friedman rank test — *across* companies, one observation per model
    per company (its test-set RMSE).
  - Holm-adjusted Wilcoxon signed-rank tests — pairwise post-hoc follow-up
    to a significant Friedman result.
  - Best-model consistency check — whether one model has the lowest RMSE
    on at least a majority (8 of 15, by default) of companies.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t as t_dist, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)


def _naive_mae(y_reference: np.ndarray) -> float:
    """The mean absolute error of a naive one-step (yesterday's value)
    forecast, computed over ``y_reference`` — this is the MASE scaling
    denominator. Per Hyndman & Koehler, this should be the *in-sample*
    (training) series, not the held-out series being scored, so that the
    denominator reflects how hard the series is to forecast naively
    independent of the test window.
    """
    y_reference = np.asarray(y_reference, dtype=float)
    if len(y_reference) < 2:
        return 1e-8
    return float(np.mean(np.abs(np.diff(y_reference)))) or 1e-8


def compute_metrics(y_true, y_pred, y_train=None) -> dict:
    """RMSE, MAE, MASE, R² — the four headline metrics used across the
    project, all as formatted strings for direct display.

    ``y_train`` should be the in-sample (training) target series, used as
    the naive one-step-forecast baseline that scales MASE. When omitted
    (e.g. the naive baseline scoring itself), ``y_true`` is used instead.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    naive_reference = y_true if y_train is None else np.asarray(y_train, dtype=float)
    mase = mae / _naive_mae(naive_reference)

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "rmse": f"{rmse:.4f}",
        "mae": f"{mae:.4f}",
        "mase": f"{mase:.4f}",
        "r2": f"{r2:.4f}",
    }


def evaluate_naive(df: pd.DataFrame) -> dict:
    """Baseline: predict tomorrow's close = today's close."""
    close = df["Close"].values
    y_true = close[1:]
    y_pred = close[:-1]
    return compute_metrics(y_true, y_pred)


def build_comparison_table(metrics_by_model: dict[str, dict], labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Turn {"lag_reg": {...}, "arima": {...}, "lstm": {...}, "naive": {...}}
    into a tidy comparison table — one row per model, ranked by RMSE.
    """
    labels = labels or {}
    rows = []
    for key, metrics in metrics_by_model.items():
        rows.append({
            "Model": labels.get(key, key),
            "RMSE": float(metrics["rmse"]),
            "MAE": float(metrics["mae"]),
            "MASE": float(metrics["mase"]),
            "R2": float(metrics["r2"]),
        })
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table


def select_best_model(metrics_by_model: dict[str, dict], candidate_keys: list[str]) -> str:
    """Returns the key (from candidate_keys) with the lowest RMSE."""
    return min(candidate_keys, key=lambda k: float(metrics_by_model[k]["rmse"]))


# --------------------------------------------------------------------------
# Statistical significance suite
# --------------------------------------------------------------------------

def _newey_west_hac_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC (heteroskedasticity- and autocorrelation-consistent)
    long-run variance estimate of the loss-differential series ``d``,
    using a Bartlett kernel out to ``max_lag`` lags."""
    n = len(d)
    d_bar = d.mean()
    demeaned = d - d_bar
    variance = float(np.mean(demeaned**2))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        autocov = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        variance += 2.0 * weight * autocov
    return max(variance, 1e-12)


def diebold_mariano_test(errors1, errors2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test comparing two models' forecast errors on the
    *same* held-out window, with Newey-West HAC variance (automatic
    bandwidth) and the Harvey-Leybourne-Newbold (HLN) small-sample
    correction. ``errors1``/``errors2`` are actual-minus-predicted arrays,
    paired by forecast date. Returns (dm_statistic, p_value); the p-value
    uses a Student-t reference distribution (T-1 df) per HLN.
    """
    e1 = np.asarray(errors1, dtype=float)
    e2 = np.asarray(errors2, dtype=float)
    n = len(e1)
    if n < 2 or n != len(e2):
        return float("nan"), float("nan")

    loss_diff = np.abs(e1) ** power - np.abs(e2) ** power
    d_bar = float(loss_diff.mean())

    max_lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), h - 1, 0)
    var_d = _newey_west_hac_variance(loss_diff, max_lag)

    dm_stat = d_bar / np.sqrt(var_d / n)

    hln_correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_hln = dm_stat * hln_correction

    p_value = float(2 * (1 - t_dist.cdf(abs(dm_hln), df=max(n - 1, 1))))
    return float(dm_hln), p_value


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values in
    the same order as ``p_values`` (each clipped to [0, 1], monotone
    non-decreasing when sorted by the original p-value)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def friedman_test(rmse_by_model: dict[str, list[float]]) -> dict:
    """Friedman rank test across companies: each model contributes one
    RMSE observation per company (paired by company). Tests whether the
    average ranks of the models differ significantly.

    ``rmse_by_model``: {model_key: [rmse_company_1, rmse_company_2, ...]},
    all lists the same length and in the same company order.
    """
    groups = list(rmse_by_model.values())
    n_companies = len(groups[0]) if groups else 0
    if len(groups) < 3 or n_companies < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_companies": n_companies}
    statistic, p_value = friedmanchisquare(*groups)
    return {"statistic": float(statistic), "p_value": float(p_value), "n_companies": n_companies}


def holm_wilcoxon_posthoc(rmse_by_model: dict[str, list[float]]) -> dict:
    """Pairwise Wilcoxon signed-rank tests (paired by company) between
    every pair of models, Holm-adjusted across all pairs. Post-hoc
    follow-up to a significant Friedman result.

    Returns {"model_a vs model_b": {"statistic":.., "p_value":..,
    "holm_p_value":..}, ...}.
    """
    keys = list(rmse_by_model.keys())
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]

    raw_results = []
    for a, b in pairs:
        try:
            statistic, p_value = wilcoxon(rmse_by_model[a], rmse_by_model[b])
            statistic, p_value = float(statistic), float(p_value)
        except ValueError:  # all differences zero, or too few samples
            statistic, p_value = float("nan"), 1.0
        raw_results.append((statistic, p_value))

    adjusted = holm_correction([p for _, p in raw_results])

    return {
        f"{a} vs {b}": {"statistic": stat, "p_value": p, "holm_p_value": holm_p}
        for (a, b), (stat, p), holm_p in zip(pairs, raw_results, adjusted)
    }


def best_model_consistency_check(rmse_by_model: dict[str, list[float]], min_companies: int = 8) -> dict:
    """Whether one model has the lowest RMSE on at least ``min_companies``
    (out of the total, paired by company) — a consistency check that the
    overall best model isn't a fluke of averaging."""
    keys = list(rmse_by_model.keys())
    n_companies = len(rmse_by_model[keys[0]]) if keys else 0

    counts = {k: 0 for k in keys}
    for company_idx in range(n_companies):
        rmses = {k: rmse_by_model[k][company_idx] for k in keys}
        winner = min(rmses, key=rmses.get)
        counts[winner] += 1

    dominant_model = max(counts, key=counts.get) if counts else None
    dominant_count = counts.get(dominant_model, 0)

    return {
        "counts": counts,
        "total_companies": n_companies,
        "min_required": min_companies,
        "dominant_model": dominant_model,
        "dominant_count": dominant_count,
        "pass": dominant_count >= min_companies,
    }


def run_cross_model_statistical_tests(
    test_errors_by_symbol: dict[str, dict[str, np.ndarray]],
    rmse_by_symbol: dict[str, dict[str, float]],
    model_keys: tuple[str, ...] = ("lag_reg", "arima", "lstm"),
    min_companies: int = 8,
) -> dict:
    """Runs the full statistical-significance suite and returns a single
    JSON-serializable dict, written to disk by model_selector.py.

    ``test_errors_by_symbol[symbol][model_key]`` = 1-D array of
    (actual - predicted) reconstructed-price errors on that model's
    held-out test window for that symbol, already aligned so every model
    covers the *same* dates for that symbol (truncated to the shortest
    common test window, right-aligned on the most recent date).

    ``rmse_by_symbol[symbol][model_key]`` = that model's test-set RMSE for
    that symbol (reconstructed-price scale).
    """
    symbols = sorted(test_errors_by_symbol.keys())

    # --- Diebold-Mariano, within each company, Holm-corrected across its pairs ---
    dm_by_symbol = {}
    for symbol in symbols:
        errors = test_errors_by_symbol[symbol]
        pairs = [(a, b) for i, a in enumerate(model_keys) for b in model_keys[i + 1 :] if a in errors and b in errors]
        raw = []
        for a, b in pairs:
            dm_stat, p_value = diebold_mariano_test(errors[a], errors[b])
            raw.append((a, b, dm_stat, p_value))
        adjusted = holm_correction([p for *_, p in raw])
        dm_by_symbol[symbol] = {
            f"{a} vs {b}": {"dm_statistic": stat, "p_value": p, "holm_p_value": holm_p}
            for (a, b, stat, p), holm_p in zip(raw, adjusted)
        }

    # --- Friedman (across companies) + Wilcoxon-Holm post-hoc ---
    # Only symbols where every model has an RMSE keep the pairing valid.
    complete_symbols = [s for s in symbols if all(m in rmse_by_symbol[s] for m in model_keys)]
    rmse_by_model_paired = {
        model: [rmse_by_symbol[symbol][model] for symbol in complete_symbols] for model in model_keys
    }

    friedman_result = friedman_test(rmse_by_model_paired)
    wilcoxon_result = holm_wilcoxon_posthoc(rmse_by_model_paired) if complete_symbols else {}
    consistency = best_model_consistency_check(rmse_by_model_paired, min_companies=min_companies)

    return {
        "n_companies": len(complete_symbols),
        "companies": complete_symbols,
        "diebold_mariano": dm_by_symbol,
        "friedman": friedman_result,
        "wilcoxon_holm_posthoc": wilcoxon_result,
        "best_model_consistency": consistency,
    }
