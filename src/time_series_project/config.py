"""Параметры запуска базового пайплайна."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Хранит пути и основные параметры одного запуска пайплайна."""

    data_path: Path
    test_size: int = 12
    output_dir: Path = Path("reports")
    series_name: str | None = None

    def validate(self) -> None:
        """Проверяет параметры до чтения данных и запуска расчетов."""
        if self.test_size <= 0:
            raise ValueError("test_size должен быть положительным целым числом")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
