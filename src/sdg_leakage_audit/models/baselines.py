"""Non-trainable forecasting baselines: seasonal-naive and persistence.

Both baselines are evaluated vectorized, batch-style, consistent with how the
`evaluation.runner` module frames every model (including the trainable ones) as
"direct" multi-origin regression: `predict(history, horizon)` returns one
`horizon`-steps-ahead forecast per position in `history`, i.e.
`result[i]` forecasts the value at position `i + horizon`, using only
`history[:i+1]` -- never any value at or after position `i + horizon`.
"""

from __future__ import annotations

import numpy as np


class SeasonalNaiveForecaster:
    """Forecasts the value observed exactly one season ago.

    Attributes:
        season_length: Number of steps defining one season (e.g. 24 for hourly
            data with daily seasonality). Must be >= the forecast horizon used in
            `predict`, since the rule can only reference values strictly before
            the forecast origin.
    """

    def __init__(self, season_length: int) -> None:
        self.season_length = season_length

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict, for every position in `history`, the value `horizon` steps ahead.

        Args:
            history: 1-D array of observed values.
            horizon: Number of steps ahead to forecast.

        Returns:
            Array the same length as `history`. `result[i]` equals
            `history[i - (season_length - horizon)]` (the value one season before
            the target point); positions with insufficient history are NaN.

        Raises:
            ValueError: If `season_length < horizon` (the rule would need a future value).
        """
        shift = self.season_length - horizon
        if shift < 0:
            raise ValueError(
                f"season_length ({self.season_length}) must be >= horizon ({horizon}) "
                "for the seasonal-naive rule to use only past information"
            )
        history = np.asarray(history, dtype=float)
        forecast = np.full_like(history, np.nan)
        if shift == 0:
            forecast[:] = history
        else:
            forecast[shift:] = history[:-shift]
        return forecast


class PersistenceForecaster:
    """Forecasts the last observed value for all future steps (random-walk naive)."""

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict, for every position in `history`, the value `horizon` steps ahead.

        Args:
            history: 1-D array of observed values.
            horizon: Number of steps ahead to forecast (unused: persistence always
                repeats the value at the forecast origin itself).

        Returns:
            Array the same length as `history`, equal to `history` itself
            (`result[i] == history[i]`).
        """
        return np.asarray(history, dtype=float).copy()
