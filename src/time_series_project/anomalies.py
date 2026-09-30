"""Методы поиска аномалий во временном ряду."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

from .exceptions import DataValidationError


def _result_frame(series: pd.Series, score: pd.Series, is_anomaly: pd.Series) -> pd.DataFrame:
    """Собирает результат детектора аномалий в единый DataFrame."""
    return pd.DataFrame(
        {
            "y": series.astype(float),
            "score": score.reindex(series.index).astype(float),
            "is_anomaly": is_anomaly.reindex(series.index).fillna(False).astype(bool),
        },
        index=series.index,
    )


def rolling_zscore_anomalies(
    series: pd.Series,
    window: int,
    threshold: float = 2.5,
) -> pd.DataFrame:
    """Находит локальные выбросы по z-score относительно предыдущих наблюдений."""
    if window < 2:
        raise DataValidationError("Размер скользящего окна должен быть не меньше двух")
    if threshold <= 0:
        raise DataValidationError("Порог z-score должен быть положительным")

    min_periods = max(2, window // 2)
    # Сдвигаем окно на один шаг, чтобы текущая точка не участвовала в собственной оценке.
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)
    safe_std = rolling_std.replace(0.0, np.nan)
    score = ((series - rolling_mean) / safe_std).abs()
    is_anomaly = score > threshold
    return _result_frame(series, score, is_anomaly)


def stl_residual_anomalies(
    series: pd.Series,
    period: int,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Находит выбросы в остатках после STL-разложения ряда."""
    if period < 2:
        raise DataValidationError("Период STL должен быть не меньше двух")
    if len(series) < period * 2:
        raise DataValidationError("Для STL требуется минимум два полных сезонных периода")
    if threshold <= 0:
        raise DataValidationError("Порог должен быть положительным")

    decomposition = STL(series.astype(float), period=period, robust=True).fit()
    residual = pd.Series(decomposition.resid, index=series.index, dtype=float)
    center = float(residual.median())
    scale = float(residual.std(ddof=0))

    if scale <= np.finfo(float).eps:
        score = pd.Series(0.0, index=series.index, dtype=float)
    else:
        score = ((residual - center) / scale).abs()

    is_anomaly = score > threshold
    return _result_frame(series, score, is_anomaly)


def isolation_forest_anomalies(
    series: pd.Series,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Находит нетипичные точки с помощью IsolationForest."""
    if not 0.0 < contamination <= 0.5:
        raise DataValidationError("contamination должен находиться в диапазоне (0, 0.5]")
    if len(series) < 8:
        raise DataValidationError("Для IsolationForest требуется минимум восемь наблюдений")

    # Используем значение ряда, два лага и короткие скользящие статистики.
    features = pd.DataFrame(index=series.index)
    features["y"] = series.astype(float)
    features["lag_1"] = series.shift(1)
    features["lag_2"] = series.shift(2)
    features["rolling_mean_3"] = series.rolling(3, min_periods=1).mean()
    features["rolling_std_3"] = series.rolling(3, min_periods=2).std(ddof=0)
    features = features.bfill().fillna(0.0)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=300,
    )
    labels = model.fit_predict(features)
    score_values = -model.score_samples(features)

    score = pd.Series(score_values, index=series.index, dtype=float)
    is_anomaly = pd.Series(labels == -1, index=series.index, dtype=bool)
    return _result_frame(series, score, is_anomaly)
