# Project Neto Time - анализ и прогнозирование временных рядов

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![statsforecast](https://img.shields.io/badge/statsforecast-statistical%20models-green)](https://nixtlaverse.nixtla.io/statsforecast/)
[![mlforecast](https://img.shields.io/badge/mlforecast-ML%20forecasting-lightgrey)](https://nixtlaverse.nixtla.io/mlforecast/)
[![neuralforecast](https://img.shields.io/badge/neuralforecast-DL%20forecasting-purple)](https://nixtlaverse.nixtla.io/neuralforecast/)

## Краткое резюме

`project_neto_time` - практический проект по анализу и прогнозированию временных рядов.

В проекте реализованы:

- подготовка данных и EDA временных рядов;
- проверка стационарности ADF / KPSS;
- стационаризация временных рядов;
- статистические модели прогнозирования через `statsforecast`;
- ML-подход через `mlforecast`;
- DL-подход через `neuralforecast`;
- анализ аномалий временного ряда;
- итоговый pipeline прогнозирования;
- отчет исследования в формате Markdown.

Главная цель проекта - показать полный цикл работы с временным рядом: от постановки задачи и подготовки данных до сравнения статистических, ML и DL-моделей.

## Структура репозитория

```text
project_neto_time/
|
├── data/
│   ├── raw/
│   └── processed/
|
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
|
├── src/
│   ├── __init__.py
│   └── ts_pipeline.py
|
├── reports/
│   ├── figures/
│   ├── eda_summary.csv
│   ├── stationarity_summary.csv
│   ├── statsforecast_comparison.csv
│   ├── mlforecast_comparison.csv
│   ├── neuralforecast_comparison.csv
│   ├── anomaly_detection_summary.csv
│   ├── final_pipeline_results.csv
│   └── pipeline_performance.csv
|
├── REPORT.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Данные

В проекте используются несколько классических временных рядов из папки `data/raw`:

| Файл | Частота | Описание |
|---|---:|---|
| `daily-total-female-births-in-cal.csv` | дневная | количество рождений девочек в Калифорнии |
| `international-airline-passengers.csv` | месячная | количество международных авиапассажиров |
| `mean-monthly-air-temperature-deg.csv` | месячная | средняя месячная температура воздуха |
| `monthly-boston-armed-robberies-j.csv` | месячная | количество вооруженных ограблений в Бостоне |
| `monthly-sales-of-company-x-jan-6.csv` | месячная | месячные продажи компании |
| `weekly-closings-of-the-dowjones-.csv` | недельная | недельные значения закрытия Dow Jones |

Основной временной ряд для итогового сравнения моделей - `international-airline-passengers.csv`, так как он содержит тренд и выраженную сезонность.

## Постановка задачи

Для выбранного временного ряда требуется построить прогноз будущих значений.

Базовая постановка:

- тип задачи - прогнозирование одномерного временного ряда;
- режим - offline forecast;
- горизонт прогноза - последний отложенный период ряда;
- train/test split - по времени, без перемешивания;
- метрики - MAE, RMSE, MAPE, sMAPE;
- контроль качества - сравнение baseline, статистических моделей, ML и DL-моделей;
- дополнительный анализ - проверка аномалий и анализ остатков.

## Ноутбуки проекта

### 1. `01_eda_preprocessing_report.ipynb`

Закрывает задачу подготовки данных и EDA:

- загрузка всех рядов из `data/raw`;
- приведение к `pandas.Series`;
- проверка пропусков и типов данных;
- определение частоты ряда;
- визуальный анализ;
- ADF / KPSS тесты;
- анализ тренда и сезонности;
- сохранение итоговых EDA-таблиц в `reports/`.

### 2. `02_statsforecast_stat_models.ipynb`

Закрывает блок статистических методов через `statsforecast`.

Сравниваются модели:

- Naive;
- SeasonalNaive;
- RandomWalkWithDrift;
- AutoARIMA;
- AutoETS;
- AutoTheta;
- Prophet - опционально, если пакет установлен.

Результат сохраняется в `reports/statsforecast_comparison.csv`.

### 3. `03_mlforecast_ml_models.ipynb`

Закрывает блок ML-подхода через `mlforecast`.

Сравниваются модели:

- LinearRegression;
- RandomForestRegressor;
- HistGradientBoostingRegressor.

Используются лаги и календарные признаки.

Результат сохраняется в `reports/mlforecast_comparison.csv`.

### 4. `04_neuralforecast_dl_models.ipynb`

Закрывает блок DL-подхода через `neuralforecast`.

Сравниваются модели:

- NBEATS;
- NHITS;
- LSTM.

Результат сохраняется в `reports/neuralforecast_comparison.csv`.

### 5. `05_anomaly_detection.ipynb`

Закрывает блок анализа аномалий.

Сравниваются методы:

- rolling z-score;
- STL residual anomalies;
- IsolationForest.

Результат сохраняется в `reports/anomaly_detection_summary.csv`.

### 6. `06_final_pipeline.ipynb`

Закрывает блок итогового pipeline:

- загрузка данных;
- train/test split;
- запуск baseline-моделей;
- расчет метрик;
- тестирование времени выполнения;
- сохранение результатов.

Результаты сохраняются в:

- `reports/final_pipeline_results.csv`;
- `reports/pipeline_performance.csv`.

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/AndreyBers/project_neto_time.git
cd project_neto_time
```

### 2. Создать виртуальное окружение

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

Если `neuralforecast` или `torch` тяжело ставятся локально, DL-ноутбук можно запускать в Google Colab.

### 4. Запустить Jupyter Notebook

```bash
jupyter notebook
```

## Рекомендуемый порядок запуска

1. `01_eda_preprocessing_report.ipynb`
2. `02_statsforecast_stat_models.ipynb`
3. `03_mlforecast_ml_models.ipynb`
4. `04_neuralforecast_dl_models.ipynb`
5. `05_anomaly_detection.ipynb`
6. `06_final_pipeline.ipynb`

## Соответствие требованиям задания

| Требование | Где реализовано |
|---|---|
| Подготовка данных и EDA | `01_eda_preprocessing_report.ipynb` |
| Проверка стационарности | `01_eda_preprocessing_report.ipynb`|
| Статистические методы | `02_statsforecast_stat_models.ipynb` |
| Не менее 5 статистических методов | Naive, SeasonalNaive, Drift, AutoARIMA, AutoETS, AutoTheta |
| ML-методы | `03_mlforecast_ml_models.ipynb` |
| Не менее 3 ML-методов | LinearRegression, RandomForestRegressor, HistGradientBoostingRegressor |
| DL-методы | `04_neuralforecast_dl_models.ipynb` |
| Не менее 3 DL-методов | NBEATS, NHITS, LSTM |
| Анализ аномалий | `05_anomaly_detection.ipynb` |
| Не менее 3 методов аномалий | rolling z-score, STL residual, IsolationForest |
| Pipeline | `src/ts_pipeline.py`, `06_final_pipeline.ipynb` |
| Отчет исследования | `REPORT.md`, `README.md` |

## Автор проекта

**AndreyBers**  
GitHub: [AndreyBers](https://github.com/AndreyBers)

Проект выполнен в рамках практической работы по анализу временных рядов.
