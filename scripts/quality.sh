#!/usr/bin/env bash
set -euo pipefail

# Запускаем тот же набор проверок, что используется в CI.
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src/time_series_project
poetry run pytest
