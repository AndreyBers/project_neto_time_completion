"""CLI для запуска базового пайплайна из командной строки."""

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .data import infer_frequency, infer_season_length, load_series
from .logging_config import configure_logging
from .pipeline import run_lightweight_pipeline, save_pipeline_result

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Описывает аргументы командной строки и возвращает готовый парсер."""
    parser = argparse.ArgumentParser(
        prog="ts-pipeline",
        description="Базовый пайплайн прогнозирования временного ряда",
    )
    parser.add_argument("--data", required=True, type=Path, help="Путь к CSV с временным рядом")
    parser.add_argument(
        "--test-size",
        type=int,
        default=12,
        help="Количество последних наблюдений, которые используются как test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Папка для сохранения CSV-результатов",
    )
    parser.add_argument(
        "--series-name",
        default=None,
        help="Имя ряда в итоговых данных, если его нужно задать вручную",
    )
    return parser


def main() -> int:
    """Читает параметры, запускает пайплайн и сохраняет результаты."""
    args = build_parser().parse_args()
    config = PipelineConfig(
        data_path=args.data,
        test_size=args.test_size,
        output_dir=args.output_dir,
        series_name=args.series_name,
    )
    config.validate()
    configure_logging()

    LOGGER.info("Загрузка временного ряда из %s", config.data_path)
    series = load_series(config.data_path, name=config.series_name)
    LOGGER.info(
        "Загружено %s наблюдений, частота %s, сезонный период %s",
        len(series),
        infer_frequency(series),
        infer_season_length(series),
    )

    result = run_lightweight_pipeline(series, test_size=config.test_size)
    save_pipeline_result(result, config.output_dir)
    LOGGER.info("Результаты сохранены в %s", config.output_dir)
    LOGGER.info("Лучшая базовая модель по RMSE: %s", result.metrics.iloc[0]["model"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
