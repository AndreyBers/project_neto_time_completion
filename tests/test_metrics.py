"""Тесты метрик качества прогноза."""

import math

import pytest

from time_series_project import forecast_metrics
from time_series_project.exceptions import DataValidationError


def test_forecast_metrics_are_zero_for_perfect_prediction() -> None:
    """Проверяет нулевую ошибку для идеального прогноза."""
    metrics = forecast_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert metrics == {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "sMAPE": 0.0}


def test_forecast_metrics_handle_zero_actual_values() -> None:
    """Проверяет расчет процентных метрик при нулевом фактическом значении."""
    metrics = forecast_metrics([0.0, 2.0], [1.0, 2.0])

    assert math.isfinite(metrics["MAE"])
    assert math.isfinite(metrics["RMSE"])
    assert math.isfinite(metrics["MAPE"])
    assert math.isfinite(metrics["sMAPE"])


def test_forecast_metrics_reject_different_lengths() -> None:
    """Проверяет ошибку при разной длине факта и прогноза."""
    with pytest.raises(DataValidationError):
        forecast_metrics([1.0, 2.0], [1.0])
