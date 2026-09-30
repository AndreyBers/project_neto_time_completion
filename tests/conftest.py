"""Общие тестовые данные для временных рядов."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def monthly_series() -> pd.Series:
    """Возвращает месячный ряд с плавным трендом и сезонностью."""
    index = pd.date_range("2018-01-01", periods=72, freq="MS")
    trend = np.linspace(100.0, 200.0, len(index))
    season = 10.0 * np.sin(2.0 * np.pi * np.arange(len(index)) / 12.0)
    return pd.Series(trend + season, index=index, name="target")
