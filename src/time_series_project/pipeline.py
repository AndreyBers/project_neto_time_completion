"""Базовый пайплайн прогнозирования и сохранение его результатов."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from .data import infer_season_length, split_series
from .forecasting import run_baseline_forecasts
from .metrics import forecast_metrics


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Хранит метрики, время выполнения и прогнозы одного запуска."""

    metrics: pd.DataFrame
    performance: pd.DataFrame
    predictions: pd.DataFrame


def run_lightweight_pipeline(series: pd.Series, test_size: int = 12) -> PipelineResult:
    """Запускает базовые модели и рассчитывает метрики на test-отрезке."""
    total_started = perf_counter()

    # Замеряем этапы отдельно, чтобы видеть, сколько времени занимает каждая часть запуска.
    split_started = perf_counter()
    train, test = split_series(series, test_size=test_size)
    split_seconds = perf_counter() - split_started

    forecast_started = perf_counter()
    season_length = infer_season_length(series)
    forecasts = run_baseline_forecasts(
        train=train,
        horizon=len(test),
        season_length=season_length,
    )
    forecast_seconds = perf_counter() - forecast_started

    metrics_started = perf_counter()
    metric_rows: list[dict[str, float | str]] = []
    prediction_frame = pd.DataFrame({"ds": test.index, "actual": test.to_numpy(dtype=float)})

    # Все модели сравниваем на одном test-отрезке и одним набором метрик.
    for model_name, prediction in forecasts.items():
        row: dict[str, float | str] = dict(forecast_metrics(test.to_numpy(), prediction))
        row["model"] = model_name
        row["group"] = "baseline"
        metric_rows.append(row)
        prediction_frame[model_name] = prediction

    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    metrics_seconds = perf_counter() - metrics_started
    total_seconds = perf_counter() - total_started

    performance = pd.DataFrame(
        [
            {"stage": "split", "seconds": split_seconds},
            {"stage": "forecast", "seconds": forecast_seconds},
            {"stage": "metrics", "seconds": metrics_seconds},
            {"stage": "total", "seconds": total_seconds},
        ]
    )

    return PipelineResult(
        metrics=metrics,
        performance=performance,
        predictions=prediction_frame,
    )


def save_pipeline_result(result: PipelineResult, output_dir: str | Path) -> None:
    """Сохраняет метрики, время выполнения и прогнозы в CSV."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    result.metrics.to_csv(destination / "final_pipeline_results.csv", index=False)
    result.performance.to_csv(destination / "pipeline_performance.csv", index=False)
    result.predictions.to_csv(destination / "baseline_predictions.csv", index=False)
