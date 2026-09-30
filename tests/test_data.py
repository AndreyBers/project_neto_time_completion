"""Тесты загрузки и подготовки временного ряда."""

import pandas as pd
import pytest

from time_series_project import infer_frequency, infer_season_length, load_series, split_series
from time_series_project.exceptions import DataValidationError


def test_load_series_detects_columns(tmp_path) -> None:
    """Проверяет автоматическое определение даты и числового значения."""
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "Month": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "Passengers": [100, 110, 120, 130, 140, 150],
        }
    ).to_csv(path, index=False)

    series = load_series(path, name="passengers")

    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.name == "passengers"
    assert len(series) == 6
    assert series.iloc[-1] == 150.0


def test_frequency_and_season_length(monthly_series) -> None:
    """Проверяет месячную частоту и годовой сезонный период."""
    assert infer_frequency(monthly_series) == "MS"
    assert infer_season_length(monthly_series) == 12


def test_split_series_preserves_time_order(monthly_series) -> None:
    """Проверяет временной порядок и отсутствие пересечения train/test."""
    train, test = split_series(monthly_series, test_size=12)

    assert len(train) == 60
    assert len(test) == 12
    assert train.index.max() < test.index.min()


def test_split_series_rejects_too_large_test(monthly_series) -> None:
    """Проверяет ошибку, если test занимает весь ряд."""
    with pytest.raises(DataValidationError):
        split_series(monthly_series, test_size=len(monthly_series))
