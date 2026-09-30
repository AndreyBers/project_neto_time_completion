"""Загрузка, проверка и подготовка временных рядов."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

_DATE_HINTS = ("date", "time", "month", "day", "year", "ds", "дата", "время", "месяц")
_VALUE_HINTS = ("value", "passenger", "sales", "temperature", "y", "count", "значение")


def _ordered_candidates(columns: Iterable[str], hints: tuple[str, ...]) -> list[str]:
    """Сначала возвращает колонки с подходящими названиями, затем остальные."""
    names = list(columns)
    preferred = [name for name in names if any(hint in name.lower() for hint in hints)]
    return preferred + [name for name in names if name not in preferred]


def _detect_date_column(frame: pd.DataFrame) -> tuple[str, pd.Series]:
    """Находит колонку, в которой большая часть значений распознается как дата."""
    best_column: str | None = None
    best_values: pd.Series | None = None
    best_score = 0.0

    # Сначала пробуем колонки с характерными названиями, затем проверяем остальные.
    for column in _ordered_candidates(frame.columns, _DATE_HINTS):
        source = frame[column]
        if pd.api.types.is_numeric_dtype(source) and not any(
            hint in column.lower() for hint in ("year", "date", "time")
        ):
            continue

        parsed = pd.to_datetime(source, errors="coerce")
        score = float(parsed.notna().mean())
        if score > best_score:
            best_column = column
            best_values = parsed
            best_score = score

    if best_column is None or best_values is None or best_score < 0.8:
        raise DataValidationError(
            "Не удалось определить колонку даты: "
            "не менее 80% значений должны распознаваться как даты"
        )
    return best_column, best_values


def _detect_value_column(frame: pd.DataFrame, excluded: set[str]) -> tuple[str, pd.Series]:
    """Находит колонку, в которой большая часть значений является числовой."""
    best_column: str | None = None
    best_values: pd.Series | None = None
    best_score = 0.0

    candidates = [column for column in frame.columns if column not in excluded]
    for column in _ordered_candidates(candidates, _VALUE_HINTS):
        parsed = pd.to_numeric(frame[column], errors="coerce")
        score = float(parsed.notna().mean())
        if score > best_score:
            best_column = column
            best_values = parsed
            best_score = score

    if best_column is None or best_values is None or best_score < 0.8:
        raise DataValidationError(
            "Не удалось определить числовую колонку значений: "
            "не менее 80% значений должны быть числами"
        )
    return best_column, best_values


def load_series(path: str | Path, name: str | None = None) -> pd.Series:
    """Загружает CSV и возвращает очищенный ряд с DatetimeIndex."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    try:
        frame = pd.read_csv(data_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Не удалось прочитать CSV: {data_path}") from exc

    if frame.empty:
        raise DataValidationError(f"Файл не содержит наблюдений: {data_path}")
    if frame.shape[1] < 2:
        raise DataValidationError("Для временного ряда нужны минимум две колонки: дата и значение")

    date_column, dates = _detect_date_column(frame)
    value_column, values = _detect_value_column(frame, excluded={date_column})

    clean = pd.DataFrame({"ds": dates, "y": values}).dropna(subset=["ds", "y"])
    if clean.empty:
        raise DataValidationError("После очистки не осталось корректных наблюдений")

    # Если одна дата встречается несколько раз, оставляем последнее значение для этой даты.
    clean = clean.sort_values("ds").drop_duplicates(subset="ds", keep="last")
    series = pd.Series(
        clean["y"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(clean["ds"]),
        name=name or value_column,
        dtype=float,
    )

    if len(series) < 3:
        raise DataValidationError("Для анализа временного ряда требуется минимум три наблюдения")
    return series


def infer_frequency(series: pd.Series) -> str:
    """Определяет календарную частоту ряда в формате pandas."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError("Индекс временного ряда должен иметь тип DatetimeIndex")
    if len(series) < 3:
        raise DataValidationError("Для определения частоты требуется минимум три наблюдения")

    frequency = pd.infer_freq(series.index)
    if frequency:
        return frequency

    # Если pandas не определил частоту напрямую, оцениваем ее по медианному шагу между датами.
    date_steps = series.index.to_series().diff().dropna()
    delta_seconds = date_steps.dt.total_seconds().to_numpy(dtype=float)
    deltas = delta_seconds / 86_400.0
    median_days = float(np.median(deltas))

    if 0.9 <= median_days <= 1.1:
        return "D"
    if 6.0 <= median_days <= 8.0:
        return "W"
    if 27.0 <= median_days <= 32.0:
        return "MS"
    if 80.0 <= median_days <= 100.0:
        return "QS"
    if 360.0 <= median_days <= 370.0:
        return "YS"

    raise DataValidationError(
        "Не удалось надежно определить частоту временного ряда по его временному индексу"
    )


def infer_season_length(series: pd.Series) -> int:
    """Возвращает типичный сезонный период для найденной частоты ряда."""
    frequency = infer_frequency(series)
    normalized = frequency.upper()

    if normalized.startswith(("MS", "ME")) or normalized == "M":
        return 12
    if normalized.startswith("Q"):
        return 4
    if normalized.startswith("W"):
        return 52
    if normalized.startswith("B"):
        return 5
    if normalized.startswith("D"):
        return 7
    if normalized.startswith("H"):
        return 24
    return 1


def split_series(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    """Разбивает ряд на train и test по времени, без перемешивания."""
    if test_size <= 0:
        raise DataValidationError("test_size должен быть положительным")
    if len(series) <= test_size:
        raise DataValidationError("test_size должен быть меньше количества наблюдений")

    train = series.iloc[:-test_size].copy()
    test = series.iloc[-test_size:].copy()

    if train.index.max() >= test.index.min():
        raise DataValidationError("Train и test должны следовать друг за другом во времени")
    return train, test


def to_nixtla_df(series: pd.Series, unique_id: str = "series") -> pd.DataFrame:
    """Преобразует Series в формат ``unique_id``, ``ds``, ``y`` для Nixtla."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError("Индекс временного ряда должен иметь тип DatetimeIndex")

    return pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": series.index,
            "y": series.to_numpy(dtype=float),
        }
    )
