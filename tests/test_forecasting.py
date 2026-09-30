"""Тесты базовых моделей прогнозирования."""

import numpy as np
import pandas as pd

from time_series_project.forecasting import (
    drift_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_naive_forecast_repeats_last_value() -> None:
    """Проверяет, что Naive повторяет последнее значение."""
    train = pd.Series([1.0, 2.0, 3.0])
    assert np.array_equal(naive_forecast(train, 3), np.array([3.0, 3.0, 3.0]))


def test_seasonal_naive_repeats_last_season() -> None:
    """Проверяет повторение последнего сезонного шаблона."""
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    prediction = seasonal_naive_forecast(train, horizon=6, season_length=4)
    assert np.array_equal(prediction, np.array([10.0, 20.0, 30.0, 40.0, 10.0, 20.0]))


def test_drift_forecast_continues_linear_trend() -> None:
    """Проверяет продолжение линейного тренда моделью Drift."""
    train = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(drift_forecast(train, 2), np.array([5.0, 6.0]))
