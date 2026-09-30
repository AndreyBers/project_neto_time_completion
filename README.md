# Project Neto Time Completion

[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/dependencies-Poetry-60A5FA)](https://python-poetry.org/)
[![pre-commit](https://img.shields.io/badge/quality-pre--commit-FAB040)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![Black](https://img.shields.io/badge/format-Black-000000)](https://black.readthedocs.io/)
[![Mypy](https://img.shields.io/badge/types-Mypy-2A6DB2)](https://mypy.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/tests-Pytest-0A9EDC)](https://docs.pytest.org/)

Учебный ML-проект по анализу и прогнозированию временных рядов. В репозитории собран полный цикл работы с данными: первичный анализ, проверка стационарности, поиск аномалий, статистические модели, классические ML-модели, нейросетевые модели и отдельный воспроизводимый пайплайн для базового прогноза.

Исследовательская часть сохранена в Jupyter Notebook, а повторно используемый код вынесен в отдельный Python-пакет. Для проекта настроены Poetry, локальное виртуальное окружение `.venv`, pre-commit, Ruff, Black, Mypy, Pytest и автоматическая проверка в GitHub Actions.

## Содержание

- [Задача проекта](#задача-проекта)
- [Данные](#данные)
- [Подходы и модели](#подходы-и-модели)
- [Ключевые результаты](#ключевые-результаты)
- [Структура репозитория](#структура-репозитория)
- [Архитектура Python-кода](#архитектура-python-кода)
- [Установка](#установка)
- [Виртуальное окружение](#виртуальное-окружение)
- [Запуск пайплайна](#запуск-пайплайна)
- [Проверка качества кода](#проверка-качества-кода)
- [Pre-commit](#pre-commit)
- [Тесты](#тесты)
- [GitHub Actions](#github-actions)
- [Работа с ноутбуками](#работа-с-ноутбуками)
- [Что изменено при рефакторинге](#что-изменено-при-рефакторинге)
- [Отчеты проекта](#отчеты-проекта)
- [Соответствие заданию](#соответствие-заданию)

## Задача проекта

Цель проекта - сравнить несколько подходов к прогнозированию одномерного временного ряда и собрать воспроизводимый процесс от загрузки данных до сохранения результатов.

Основная постановка задачи:

- прогнозирование одномерного временного ряда
- разбиение train/test строго по времени без перемешивания
- горизонт итогового сравнения - последние 12 наблюдений основного ряда
- единый набор метрик - MAE, RMSE, MAPE и sMAPE
- сравнение базовых, статистических, ML и DL-моделей
- дополнительный анализ аномалий

Основной ряд для итогового сравнения - `international-airline-passengers.csv`. Он содержит выраженный восходящий тренд, годовую сезонность и изменение амплитуды сезонных колебаний.

## Данные

Исходные данные хранятся в `data/raw`.

| Файл | Частота | Содержание |
|---|---:|---|
| `daily-total-female-births-in-cal.csv` | дневная | количество рождений девочек в Калифорнии |
| `international-airline-passengers.csv` | месячная | количество международных авиапассажиров |
| `mean-monthly-air-temperature-deg.csv` | месячная | средняя месячная температура воздуха |
| `monthly-boston-armed-robberies-j.csv` | месячная | количество вооруженных ограблений в Бостоне |
| `monthly-sales-of-company-x-jan-6.csv` | месячная | месячные продажи компании |
| `weekly-closings-of-the-dowjones-.csv` | недельная | недельные значения закрытия Dow Jones |

Подготовленные данные и промежуточные результаты при необходимости сохраняются в `data/processed` и `reports`.

## Подходы и модели

### Базовые модели

Базовые модели нужны как понятная точка отсчета для более сложных методов:

- `Naive` - повторяет последнее известное значение
- `SeasonalNaive` - повторяет значения последнего сезона
- `Drift` - продолжает средний линейный тренд ряда

### Статистические модели

В статистической части используются модели из `statsforecast`:

- `AutoARIMA`
- `AutoETS`
- `AutoTheta`
- другие статистические эксперименты из ноутбуков

### ML-модели

Через `mlforecast` сравниваются:

- `LinearRegression`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

Для моделей используются лаги и календарные признаки, в том числе месяц и квартал.

### DL-модели

Через `neuralforecast` исследуются:

- `NBEATS`
- `NHITS`
- `LSTM`

DL-зависимости вынесены в отдельную группу Poetry `dl`, поэтому для обычной работы с базовым пайплайном PyTorch устанавливать не требуется.

### Поиск аномалий

В проекте реализованы три подхода:

- `rolling z-score` - сравнение точки с предыдущим скользящим окном
- `STL residual` - поиск выбросов в остатках после выделения тренда и сезонности
- `IsolationForest` - поиск нетипичных наблюдений по значению ряда, лагам и локальным статистикам

Функции поиска аномалий возвращают одинаковую структуру:

```text
y | score | is_anomaly
```

## Ключевые результаты

На сохраненном тестовом разбиении ряда международных авиапассажиров получены следующие результаты:

| Подход | Модель | MAE | RMSE | MAPE | sMAPE |
|---|---|---:|---:|---:|---:|
| Baseline | SeasonalNaive | 47.83 | 50.71 | 9.99% | 10.57% |
| Statistical | AutoARIMA | 18.52 | 23.92 | 4.18% | 4.03% |
| ML | LinearRegression | 17.18 | 19.36 | 3.56% | 3.60% |
| DL | LSTM | 16.03 | 18.01 | 3.37% | 3.37% |

В текущем эксперименте минимальный RMSE показала `LSTM`. При этом тестовая выборка состоит только из одного временного отрезка, поэтому этот результат нельзя считать окончательным выбором модели для эксплуатации. Для более надежного сравнения нужен rolling backtesting по нескольким временным окнам.

Подробное описание экспериментов находится в `REPORT.md` и соответствующих ноутбуках.

## Структура репозитория

```text
project_neto_time_completion/
├── .github/
│   └── workflows/
│       └── quality.yml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── REFACTORING_REPORT.md
├── notebook/
│   ├── stationarize_6_series.ipynb
│   ├── MA_models_experiment.ipynb
│   ├── arima_experiment_mean_monthly_air_temperature_stationary.ipynb
│   ├── ARIMA_GARCH.ipynb
│   ├── 01_eda_preprocessing_report.ipynb
│   ├── 02_statsforecast_stat_models.ipynb
│   ├── 03_mlforecast_ml_models.ipynb
│   ├── 04_neuralforecast_dl_models.ipynb
│   ├── 05_anomaly_detection.ipynb
│   └── 06_final_pipeline.ipynb
├── reports/
│   ├── figures/
│   ├── anomaly_detection_summary.csv
│   ├── baseline_predictions.csv
│   ├── final_pipeline_results.csv
│   ├── mlforecast_comparison.csv
│   ├── neuralforecast_comparison.csv
│   ├── pipeline_performance.csv
│   └── statsforecast_comparison.csv
├── scripts/
│   ├── quality.ps1
│   ├── quality.sh
│   ├── setup.ps1
│   └── setup.sh
├── src/
│   ├── __init__.py
│   ├── ts_pipeline.py
│   └── time_series_project/
│       ├── __init__.py
│       ├── __main__.py
│       ├── anomalies.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── exceptions.py
│       ├── forecasting.py
│       ├── logging_config.py
│       ├── metrics.py
│       └── pipeline.py
├── tests/
│   ├── conftest.py
│   ├── test_anomalies.py
│   ├── test_data.py
│   ├── test_forecasting.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── poetry.toml
├── pyproject.toml
├── requirements.txt
├── REPORT.md
└── README.md
```

## Архитектура Python-кода

Основная логика находится в `src/time_series_project`.

| Модуль | Назначение |
|---|---|
| `data.py` | загрузка CSV, определение колонок, очистка, определение частоты и разбиение ряда |
| `metrics.py` | расчет MAE, RMSE, MAPE и sMAPE |
| `forecasting.py` | базовые прогнозы Naive, SeasonalNaive и Drift |
| `anomalies.py` | три метода поиска аномалий с общей структурой результата |
| `pipeline.py` | запуск базового пайплайна и сохранение результатов |
| `config.py` | параметры запуска пайплайна |
| `logging_config.py` | общая настройка логирования |
| `cli.py` | запуск пайплайна из командной строки |
| `exceptions.py` | собственные исключения проекта |
| `src/ts_pipeline.py` | совместимость с импортами, которые уже используются в ноутбуках |

Такое разделение позволяет менять отдельные части проекта независимо друг от друга и не держать всю рабочую логику в одном файле.

## Установка

### Требования

Рекомендуемая версия:

```text
Python 3.11
```

Поддерживаемый диапазон:

```text
Python >=3.10,<3.13
```

Для управления зависимостями используется Poetry.

### Клонирование репозитория

```powershell
git clone https://github.com/AndreyBers/project_neto_time_completion.git
cd project_neto_time_completion
```

### Установка Poetry

Если Poetry уже установлен, достаточно проверить его версию:

```powershell
poetry --version
```

Если команда не найдена, сначала установите Poetry по официальной инструкции, затем повторите настройку проекта.

### Быстрая настройка на Windows

Из корня репозитория выполните:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Скрипт:

- проверит, что Poetry доступен
- настроит локальное окружение `.venv`
- установит основные зависимости и инструменты разработки
- установит Git hook для pre-commit

### Быстрая настройка на Linux или macOS

```bash
bash ./scripts/setup.sh
```

### Ручная настройка

Если автоматический скрипт не нужен, те же действия можно выполнить вручную:

```powershell
poetry config virtualenvs.in-project true --local
poetry install --with dev
poetry run pre-commit install
```

Для установки зависимостей нейросетевых экспериментов:

```powershell
poetry install --with dev,dl
```

## Виртуальное окружение

Poetry настроен на создание виртуального окружения непосредственно в корне проекта:

```text
.venv/
```

Настройка хранится в `poetry.toml`:

```toml
[virtualenvs]
in-project = true
```

Сам каталог `.venv` в Git не добавляется. Это важно, потому что виртуальное окружение содержит платформозависимые файлы, абсолютные пути и может занимать большой объем.

В репозитории должны храниться файлы, по которым окружение можно воспроизвести:

- `pyproject.toml` - список зависимостей и настройки инструментов
- `poetry.lock` - зафиксированные версии зависимостей после первого успешного `poetry install`
- `poetry.toml` - настройка локального расположения `.venv`

Проверить, какое окружение использует Poetry:

```powershell
poetry env info
```

После первого успешного `poetry install` файл `poetry.lock` следует добавить в Git.

## Запуск пайплайна

### Через CLI

```powershell
poetry run ts-pipeline `
  --data data/raw/international-airline-passengers.csv `
  --test-size 12 `
  --output-dir reports `
  --series-name airline_passengers
```

Однострочный вариант:

```powershell
poetry run ts-pipeline --data data/raw/international-airline-passengers.csv --test-size 12 --output-dir reports --series-name airline_passengers
```

После запуска создаются:

```text
reports/final_pipeline_results.csv
reports/pipeline_performance.csv
reports/baseline_predictions.csv
```

### Через Python-модуль

```powershell
poetry run python -m time_series_project --data data/raw/international-airline-passengers.csv --test-size 12
```

### Из Python-кода

Существующий импорт сохранен, поэтому старые ноутбуки не требуется переписывать:

```python
from src.ts_pipeline import (
    forecast_metrics,
    infer_frequency,
    infer_season_length,
    isolation_forest_anomalies,
    load_series,
    rolling_zscore_anomalies,
    run_lightweight_pipeline,
    split_series,
    stl_residual_anomalies,
    to_nixtla_df,
)
```

## Проверка качества кода

На Windows весь набор проверок запускается одной командой:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\quality.ps1
```

На Linux или macOS:

```bash
bash ./scripts/quality.sh
```

Те же проверки можно запускать отдельно:

```powershell
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src/time_series_project
poetry run pytest
```

Используемые инструменты:

| Инструмент | Что проверяет |
|---|---|
| Ruff | ошибки стиля, импорты и типовые проблемы Python-кода |
| Black | единое форматирование Python-кода |
| Mypy | согласованность аннотаций типов |
| Pytest | автоматические тесты |

## Pre-commit

Pre-commit устанавливается автоматически через `scripts/setup.ps1` или `scripts/setup.sh`.

Если нужно установить hook вручную:

```powershell
poetry run pre-commit install
```

После этого проверки запускаются перед каждым `git commit` для измененных файлов.

Проверить весь репозиторий вручную:

```powershell
poetry run pre-commit run --all-files
```

В конфигурацию включены:

- удаление пробелов в конце строк
- проверка конца файла
- проверка YAML и TOML
- поиск незавершенных merge-конфликтов
- ограничение на случайное добавление слишком больших файлов
- Ruff
- Black
- Mypy

## Тесты

Тесты находятся в `tests` и проверяют основные части повторно используемого кода:

- загрузку и автоматическое определение колонок CSV
- определение частоты и сезонного периода
- временное разбиение без перемешивания
- базовые прогнозы
- расчет метрик
- единый формат результатов поиска аномалий
- полный запуск базового пайплайна
- сохранение CSV-результатов

Запуск:

```powershell
poetry run pytest
```

Более подробный вывод:

```powershell
poetry run pytest -v
```

## GitHub Actions

Workflow `.github/workflows/quality.yml` запускается при `push` и `pull_request`.

Он выполняет тот же базовый набор проверок, что и локальный `quality`-скрипт:

1. устанавливает Python 3.11
2. устанавливает Poetry
3. устанавливает зависимости проекта
4. запускает Ruff
5. проверяет форматирование Black
6. запускает Mypy
7. запускает Pytest

Это позволяет увидеть проблему в коде до объединения изменений в основную ветку.

## Работа с ноутбуками

Исследовательские ноутбуки остаются отдельным слоем проекта. Их удобно использовать для графиков, экспериментов, сравнения моделей и подробных выводов, а повторяющаяся логика вынесена в `src`.

Рекомендуемый порядок основных ноутбуков:

1. `01_eda_preprocessing_report.ipynb`
2. `02_statsforecast_stat_models.ipynb`
3. `03_mlforecast_ml_models.ipynb`
4. `04_neuralforecast_dl_models.ipynb`
5. `05_anomaly_detection.ipynb`
6. `06_final_pipeline.ipynb`

Тяжелые DL-эксперименты при необходимости можно выполнять в Google Colab, а базовый пакет проекта и проверки качества запускать локально.

## Что изменено при рефакторинге

До рефакторинга значительная часть повторно используемой логики находилась в одном `src/ts_pipeline.py`. Теперь рабочий код разделен на небольшие модули по ответственности, а `src/ts_pipeline.py` сохранен для совместимости с существующими ноутбуками.

В проект добавлены Poetry, локальное `.venv`, CLI, логирование, типизация, автоматические тесты, pre-commit, Ruff, Black, Mypy и GitHub Actions. Исследовательские ноутбуки и результаты экспериментов при этом сохранены.

Подробное описание исходного состояния, выполненных изменений и соответствия пунктам задания находится в [`docs/REFACTORING_REPORT.md`](docs/REFACTORING_REPORT.md).

## Соответствие заданию

### 2. Рефакторинг кода ML-проекта под production-стандарты

Реализовано:

- модульная структура Python-кода
- отдельная конфигурация запуска
- CLI
- логирование
- собственные исключения
- аннотации типов
- русские docstring и понятные комментарии
- автоматические тесты
- сохранение обратной совместимости с существующими ноутбуками

### 3. Настройка pre-commit, Poetry и линтеров

Реализовано:

- `pyproject.toml`
- Poetry для управления зависимостями
- `pre-commit`
- Ruff
- Black
- Mypy
- Pytest
- единые настройки форматирования в `.editorconfig`
- автоматическая проверка в GitHub Actions

### 4. Интеграция виртуального окружения в Git-репозиторий ML-проекта

Реализовано:

- Poetry создает окружение в локальной папке `.venv`
- расположение окружения задается через `poetry.toml`
- `.venv` исключено через `.gitignore`
- зависимости описаны в `pyproject.toml`
- после первого разрешения зависимостей версии фиксируются в `poetry.lock`
- скрипты `setup.ps1` и `setup.sh` воспроизводят окружение на новой машине

Сам каталог `.venv` намеренно не хранится в Git. В репозитории хранится конфигурация, необходимая для его воспроизводимого создания.

## Отчеты проекта

В проекте используются два отдельных отчета:

- [`REPORT.md`](REPORT.md) - исследование данных, моделей и результатов экспериментов
- [`docs/REFACTORING_REPORT.md`](docs/REFACTORING_REPORT.md) - технический отчет по рефакторингу, Poetry, pre-commit, линтерам и виртуальному окружению

## Автор

**AndreyBers**

GitHub: [github.com/AndreyBers](https://github.com/AndreyBers)
