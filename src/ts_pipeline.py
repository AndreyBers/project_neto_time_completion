"""Совместимость с импортами, которые уже используются в ноутбуках.

Основная логика после рефакторинга находится в пакете ``time_series_project``.
Этот модуль оставлен, чтобы существующие ноутбуки продолжали работать без
массовой замены импортов.
"""

from .time_series_project import (
    PipelineResult,
    forecast_metrics,
    infer_frequency,
    infer_season_length,
    isolation_forest_anomalies,
    load_series,
    rolling_zscore_anomalies,
    run_lightweight_pipeline,
    save_pipeline_result,
    split_series,
    stl_residual_anomalies,
    to_nixtla_df,
)

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
