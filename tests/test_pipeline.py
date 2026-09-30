"""Интеграционные тесты базового пайплайна."""

from time_series_project import run_lightweight_pipeline
from time_series_project.pipeline import save_pipeline_result


def test_pipeline_returns_metrics_performance_and_predictions(monthly_series) -> None:
    """Проверяет основные таблицы, которые возвращает пайплайн."""
    result = run_lightweight_pipeline(monthly_series, test_size=12)

    assert {"Naive", "SeasonalNaive", "Drift"} == set(result.metrics["model"])
    assert {"MAE", "RMSE", "MAPE", "sMAPE"}.issubset(result.metrics.columns)
    assert "total" in set(result.performance["stage"])
    assert len(result.predictions) == 12


def test_pipeline_result_can_be_saved(monthly_series, tmp_path) -> None:
    """Проверяет сохранение всех CSV-результатов."""
    result = run_lightweight_pipeline(monthly_series, test_size=12)
    save_pipeline_result(result, tmp_path)

    assert (tmp_path / "final_pipeline_results.csv").exists()
    assert (tmp_path / "pipeline_performance.csv").exists()
    assert (tmp_path / "baseline_predictions.csv").exists()
