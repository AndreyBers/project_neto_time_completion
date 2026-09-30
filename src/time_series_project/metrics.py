"""Метрики качества прогноза."""

from collections.abc import Sequence

import numpy as np

from .exceptions import DataValidationError


def _as_arrays(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Приводит фактические и прогнозные значения к одномерным массивам."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.size == 0:
        raise DataValidationError("Нельзя рассчитать метрики для пустого массива")
    if actual.shape != predicted.shape:
        raise DataValidationError("Фактические и прогнозные значения должны иметь одинаковую длину")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise DataValidationError("Метрики нельзя рассчитывать при наличии NaN или бесконечностей")
    return actual, predicted


def forecast_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Рассчитывает MAE, RMSE, MAPE и sMAPE для одного прогноза."""
    actual, predicted = _as_arrays(y_true, y_pred)
    error = actual - predicted

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    # В MAPE пропускаем точки с нулевым фактическим значением, чтобы не делить на ноль.
    nonzero_actual = np.abs(actual) > np.finfo(float).eps
    if nonzero_actual.any():
        mape = float(np.mean(np.abs(error[nonzero_actual] / actual[nonzero_actual])) * 100.0)
    else:
        mape = float("nan")

    # В sMAPE пропускаем только пары, где факт и прогноз одновременно равны нулю.
    denominator = np.abs(actual) + np.abs(predicted)
    valid_smape = denominator > np.finfo(float).eps
    if valid_smape.any():
        smape = float(np.mean(2.0 * np.abs(error[valid_smape]) / denominator[valid_smape]) * 100.0)
    else:
        smape = 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}
