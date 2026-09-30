"""Базовые модели прогнозирования для контрольного сравнения."""

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Повторяет последнее известное значение на весь горизонт прогноза."""
    _validate_forecast_input(train, horizon)
    return np.repeat(float(train.iloc[-1]), horizon)


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    """Повторяет значения последнего доступного сезона."""
    _validate_forecast_input(train, horizon)
    if season_length <= 0:
        raise DataValidationError("season_length должен быть положительным")
    if len(train) < season_length:
        raise DataValidationError("Недостаточно наблюдений для сезонного прогноза")

    # Если горизонт длиннее сезона, np.resize повторяет сезонный шаблон циклически.
    season = train.iloc[-season_length:].to_numpy(dtype=float)
    return np.resize(season, horizon)


def drift_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Продолжает средний линейный тренд между первой и последней точкой train."""
    _validate_forecast_input(train, horizon)
    if len(train) < 2:
        raise DataValidationError("Для Drift требуется минимум два наблюдения")

    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / (len(train) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(train.iloc[-1]) + slope * steps


def run_baseline_forecasts(
    train: pd.Series,
    horizon: int,
    season_length: int,
) -> dict[str, np.ndarray]:
    """Строит все базовые прогнозы, доступные для переданного ряда."""
    forecasts = {
        "Naive": naive_forecast(train, horizon),
        "Drift": drift_forecast(train, horizon),
    }

    if season_length > 1 and len(train) >= season_length:
        forecasts["SeasonalNaive"] = seasonal_naive_forecast(
            train,
            horizon,
            season_length,
        )
    return forecasts


def _validate_forecast_input(train: pd.Series, horizon: int) -> None:
    """Проверяет, что данных достаточно для построения прогноза."""
    if train.empty:
        raise DataValidationError("Обучающая часть временного ряда не должна быть пустой")
    if horizon <= 0:
        raise DataValidationError("Горизонт прогноза должен быть положительным")
