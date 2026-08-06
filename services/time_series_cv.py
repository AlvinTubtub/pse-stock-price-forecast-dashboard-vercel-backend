"""Shared expanding-window rolling-origin time-series cross-validation.

Used by both the Lag-Informed Regression model (LASSO lambda selection)
and the ARIMA order search, so both follow the exact same validation
scheme the capstone paper specifies — one implementation, no drift.

``sklearn.model_selection.TimeSeriesSplit`` already *is* expanding-window
rolling-origin CV: each fold's training set grows to include everything
before the validation fold, and folds are strictly chronological (no
shuffling, no look-ahead). This module just centralizes the fold count
policy (5 folds when there's enough history, fewer only when the series
is too short) so every caller stays consistent.
"""
from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit

DEFAULT_N_SPLITS = 5


def n_splits_for(n_samples: int, max_splits: int = DEFAULT_N_SPLITS, min_fold_size: int = 10) -> int:
    """Picks a safe number of expanding-window folds for a series of this
    length, capping at ``max_splits`` (5, per the paper) but shrinking for
    short series so every fold still has at least ``min_fold_size`` rows.
    """
    if n_samples < (min_fold_size * 2):
        return 2
    return max(2, min(max_splits, n_samples // min_fold_size))


def expanding_window_splitter(n_samples: int, max_splits: int = DEFAULT_N_SPLITS, min_fold_size: int = 10) -> TimeSeriesSplit:
    """Returns a ``TimeSeriesSplit`` configured for expanding-window
    rolling-origin cross-validation over ``n_samples`` chronological rows."""
    return TimeSeriesSplit(n_splits=n_splits_for(n_samples, max_splits, min_fold_size))
