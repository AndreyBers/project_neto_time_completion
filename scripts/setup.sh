#!/usr/bin/env bash
set -euo pipefail

# Сначала проверяем, что Poetry доступен в системе.
if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry не найден. Установите Poetry и повторите запуск." >&2
  exit 1
fi

# Храним виртуальное окружение рядом с проектом в папке .venv.
poetry config virtualenvs.in-project true --local

# Устанавливаем зависимости проекта и инструменты разработки.
poetry install --with dev

# Подключаем автоматические проверки перед git commit.
poetry run pre-commit install

echo "Окружение готово. Интерпретатор находится в .venv."
echo "Для DL-зависимостей выполните: poetry install --with dev,dl"
