
"""
Utility functions and a lightweight pipeline for time series analysis.

The module is intentionally simple and notebook-friendly. It helps to:
- load CSV files with time series;
- convert data to pandas.Series and Nixtla format;
- split data without shuffling;
- calculate forecast metrics;
- build lag features for ML models;
- detect anomalies by several methods;
- run a compact baseline pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import io
import re
import time

import numpy as np
import pandas as pd


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    """Read regular CSV and CSV files where rows are accidentally stored in one line.

    Some classic time series CSV files in this project are stored as a single line:
    "Month","Count" "1949-01",112 "1949-02",118 ...
    This function inserts line breaks before date-like values and then reads the file.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").strip()

    # Insert a line break before records that start with a date-like quoted value.
    text = re.sub(r"\s+(?=\"\d{4}[-/]\d{2}(?:[-/]\d{2})?\"\s*,)", "\n", text)
    text = re.sub(r"\s+(?=\"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\"\s*,)", "\n", text)

    return pd.read_csv(io.StringIO(text))


def _choose_date_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        name = str(col).lower()
        score = 0
        if any(key in name for key in ["date", "month", "week", "time", "period", "year"]):
            score += 2
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            score += 2
        candidates.append((score, col))
    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])
    if not candidates or candidates[0][0] == 0:
        raise ValueError("Не удалось автоматически определить колонку с датой.")
    return candidates[0][1]


def _choose_value_column(df: pd.DataFrame, date_col: str) -> str:
    candidates = []
    for col in df.columns:
        if col == date_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        score = numeric.notna().mean()
        candidates.append((score, col))
    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])
    if not candidates or candidates[0][0] < 0.5:
        raise ValueError("Не удалось автоматически определить числовую колонку ряда.")
    return candidates[0][1]


def load_series(
    path: str | Path,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    name: Optional[str] = None,
) -> pd.Series:
    """Load a time series from CSV and return a sorted pandas.Series."""
    df = read_csv_flexible(path)
    df = df.dropna(how="all")

    date_col = date_col or _choose_date_column(df)
    value_col = value_col or _choose_value_column(df, date_col)

    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])
    data = data.sort_values(date_col)
    data = data.drop_duplicates(subset=[date_col], keep="last")

    series = pd.Series(data[value_col].to_numpy(dtype=float), index=data[date_col], name=name or value_col)
    series.index.name = "ds"
    return series


def infer_frequency(series: pd.Series) -> str:
    """Infer pandas frequency and use safe fallbacks for common datasets."""
    index = series.index

    if not isinstance(index, pd.DatetimeIndex):
        return "unknown"

    index = index.dropna().sort_values().unique()

    if len(index) < 2:
        return "unknown"

    if len(index) < 3:
        delta = index[1] - index[0]

        if delta.days == 1:
            return "D"
        if 6 <= delta.days <= 8:
            return "W"
        if 28 <= delta.days <= 31:
            return "M"
        if 89 <= delta.days <= 92:
            return "Q"
        if 365 <= delta.days <= 366:
            return "Y"

        return "unknown"

    freq = pd.infer_freq(index)

    if freq is not None:
        return freq

    # fallback через медианный интервал
    diffs = pd.Series(index).diff().dropna()
    median_delta = diffs.median()

    if median_delta.days == 1:
        return "D"
    if 6 <= median_delta.days <= 8:
        return "W"
    if 28 <= median_delta.days <= 31:
        return "M"
    if 89 <= median_delta.days <= 92:
        return "Q"
    if 365 <= median_delta.days <= 366:
        return "Y"

    return "unknown"


def infer_season_length(series: pd.Series) -> int:
    """Infer a practical seasonal period for common daily, weekly and monthly series."""
    freq = infer_frequency(series).upper()
    if freq.startswith("M"):
        return 12
    if freq.startswith("W"):
        return 52
    if freq.startswith("D"):
        return 7
    return 1


def to_nixtla_df(series: pd.Series, unique_id: str = "series") -> pd.DataFrame:
    """Convert pandas.Series to Nixtla long format: unique_id, ds, y."""
    return pd.DataFrame({"unique_id": unique_id, "ds": series.index, "y": series.values})


def split_series(series: pd.Series, test_size: Optional[int] = None, test_fraction: float = 0.2):
    """Temporal train-test split without shuffling."""
    n = len(series)
    if test_size is None:
        test_size = max(6, int(round(n * test_fraction)))
        test_size = min(test_size, max(1, n // 3))
    if test_size <= 0 or test_size >= n:
        raise ValueError("Некорректный размер тестовой выборки.")
    train = series.iloc[:-test_size].copy()
    test = series.iloc[-test_size:].copy()
    return train, test


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def forecast_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Calculate MAE, RMSE, MAPE and sMAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def seasonal_naive_forecast(train: pd.Series, h: int, season_length: Optional[int] = None) -> np.ndarray:
    season_length = season_length or infer_season_length(train)
    if len(train) < season_length:
        return naive_forecast(train, h)
    pattern = train.iloc[-season_length:].to_numpy()
    return np.resize(pattern, h)


def drift_forecast(train: pd.Series, h: int) -> np.ndarray:
    if len(train) < 2:
        return naive_forecast(train, h)
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    steps = np.arange(1, h + 1)
    return train.iloc[-1] + slope * steps


def make_lag_features(
    series: pd.Series,
    lags: Iterable[int] = (1, 2, 3, 6, 12),
    rolling_windows: Iterable[int] = (3, 6, 12),
) -> pd.DataFrame:
    """Build lag and rolling features for one-step-ahead ML forecasting."""
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in rolling_windows:
        shifted = df["y"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    return df.dropna()


def rolling_zscore_anomalies(series: pd.Series, window: Optional[int] = None, threshold: float = 3.0) -> pd.DataFrame:
    """Detect anomalies with rolling z-score."""
    window = window or max(3, infer_season_length(series))
    rolling_mean = series.rolling(window=window, min_periods=max(3, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(3, window // 2)).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    is_anomaly = z.abs() > threshold
    return pd.DataFrame({"y": series, "score": z, "is_anomaly": is_anomaly.fillna(False)})


def iqr_anomalies(series: pd.Series, k: float = 1.5) -> pd.DataFrame:
    """Detect global anomalies by IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    is_anomaly = (series < low) | (series > high)
    score = pd.Series(np.maximum(low - series, series - high), index=series.index)
    score = score.where(is_anomaly, 0)
    return pd.DataFrame({"y": series, "score": score, "is_anomaly": is_anomaly})


def stl_residual_anomalies(series: pd.Series, period: Optional[int] = None, threshold: float = 3.0) -> pd.DataFrame:
    """Detect anomalies in residuals after STL decomposition."""
    from statsmodels.tsa.seasonal import STL

    period = period or infer_season_length(series)
    period = max(2, period)
    result = STL(series, period=period, robust=True).fit()
    resid = result.resid
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad == 0:
        score = (resid - resid.mean()) / resid.std(ddof=0)
    else:
        score = 0.6745 * (resid - np.median(resid)) / mad
    is_anomaly = np.abs(score) > threshold
    return pd.DataFrame({"y": series, "score": score, "is_anomaly": is_anomaly})


def isolation_forest_anomalies(
    series: pd.Series,
    contamination: float = 0.05,
    lags: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Detect anomalies with IsolationForest on lag features."""
    from sklearn.ensemble import IsolationForest

    features = make_lag_features(series, lags=lags, rolling_windows=(3,))
    x = features.drop(columns=["y"])
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(x)
    score = -model.score_samples(x)
    out = pd.DataFrame({"y": features["y"], "score": score, "is_anomaly": pred == -1}, index=features.index)
    return out.reindex(series.index).fillna({"is_anomaly": False})


@dataclass
class PipelineResult:
    metrics: pd.DataFrame
    performance: pd.DataFrame


def run_lightweight_pipeline(series: pd.Series, test_size: Optional[int] = None) -> PipelineResult:
    """Run a lightweight pipeline with baseline models and timing.

    This function is intended for pipeline testing and does not replace the full
    statsforecast, mlforecast and neuralforecast notebooks.
    """
    train, test = split_series(series, test_size=test_size)
    h = len(test)
    season_length = infer_season_length(train)

    rows = []
    perf_rows = []
    models = {
        "Naive": lambda: naive_forecast(train, h),
        "SeasonalNaive": lambda: seasonal_naive_forecast(train, h, season_length),
        "Drift": lambda: drift_forecast(train, h),
    }

    for name, func in models.items():
        start = time.perf_counter()
        pred = func()
        duration = time.perf_counter() - start
        metric = forecast_metrics(test.values, pred)
        metric["model"] = name
        rows.append(metric)
        perf_rows.append({"model": name, "seconds": duration})

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    performance = pd.DataFrame(perf_rows).sort_values("seconds")
    return PipelineResult(metrics=metrics, performance=performance)
