$ErrorActionPreference = "Stop"

# Сначала проверяем, что Poetry доступен в системе.
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw "Poetry не найден. Установите Poetry и повторите запуск."
}

# Храним виртуальное окружение рядом с проектом в папке .venv.
poetry config virtualenvs.in-project true --local

# Устанавливаем зависимости проекта и инструменты разработки.
poetry install --with dev

# Подключаем автоматические проверки перед git commit.
poetry run pre-commit install

Write-Host "Окружение готово. Интерпретатор находится в .venv."
Write-Host "Для DL-зависимостей выполните: poetry install --with dev,dl"
