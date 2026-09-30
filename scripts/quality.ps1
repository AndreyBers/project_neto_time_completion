$ErrorActionPreference = "Stop"

# Проверяем Python-код линтером Ruff.
poetry run ruff check src tests

# Проверяем, что форматирование соответствует Black.
poetry run black --check src tests

# Проверяем аннотации типов в основном пакете.
poetry run mypy src/time_series_project

# Запускаем автоматические тесты.
poetry run pytest
