"""Тесты методов поиска аномалий."""

import pandas as pd

from time_series_project import (
    isolation_forest_anomalies,
    rolling_zscore_anomalies,
    stl_residual_anomalies,
)


def _assert_result_schema(result: pd.DataFrame, expected_length: int) -> None:
    """Проверяет структуру таблицы, которую возвращает детектор."""
    assert list(result.columns) == ["y", "score", "is_anomaly"]
    assert len(result) == expected_length
    assert result["is_anomaly"].dtype == bool


def test_anomaly_detectors_return_common_schema(monthly_series) -> None:
    """Проверяет одинаковую структуру результата у трех методов."""
    results = [
        rolling_zscore_anomalies(monthly_series, window=12, threshold=2.5),
        stl_residual_anomalies(monthly_series, period=12, threshold=3.0),
        isolation_forest_anomalies(monthly_series, contamination=0.05),
    ]

    for result in results:
        _assert_result_schema(result, len(monthly_series))
