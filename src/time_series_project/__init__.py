"""Основные функции для загрузки, анализа и базового прогнозирования рядов."""

from .anomalies import (
    isolation_forest_anomalies,
    rolling_zscore_anomalies,
    stl_residual_anomalies,
)
from .data import infer_frequency, infer_season_length, load_series, split_series, to_nixtla_df
from .metrics import forecast_metrics
from .pipeline import PipelineResult, run_lightweight_pipeline, save_pipeline_result

__all__ = [
    "PipelineResult",
    "forecast_metrics",
    "infer_frequency",
    "infer_season_length",
    "isolation_forest_anomalies",
    "load_series",
    "rolling_zscore_anomalies",
    "run_lightweight_pipeline",
    "save_pipeline_result",
    "split_series",
    "stl_residual_anomalies",
    "to_nixtla_df",
]
