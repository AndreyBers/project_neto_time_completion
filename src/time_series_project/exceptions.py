"""Исключения, используемые при проверке данных и запуске пайплайна."""


class TimeSeriesProjectError(Exception):
    """Базовое исключение проекта временных рядов."""


class DataValidationError(TimeSeriesProjectError, ValueError):
    """Ошибка формата, состава или качества входных данных."""
