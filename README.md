# FraudDetector

ML-система для выявления мошеннических транзакций в мобильном банкинге.

## Архитектура

```
Транзакция → [Scorecard] → Score ≤ 1? → AUTO-APPROVE (13.5%)
                  ↓
             Score > 1
                  ↓
            [ML Model] → Probability → Risk Level → Action
```

**Two-Stage Detection:**
1. **Scorecard** (25 правил) — быстрая фильтрация явно легитимных транзакций
2. **CatBoost ML** (34 признака) — детальный анализ подозрительных

## Метрики

| Метрика | Значение |
|---------|----------|
| ROC-AUC | 0.969 |
| Precision | 90.6% |
| Recall | 93.9% |
| F1-Score | 0.92 |
| Throughput | 32K TPS |

## Установка

```bash
pip install -r requirements.txt
```

Зависимости:
- catboost (GPU опционально)
- pandas, numpy
- scikit-learn
- shap

## Использование

### Обучение модели

```bash
python src/train_catboost.py
```

Результаты:
- `models/catboost_fraud_model.cbm` — модель
- `models/feature_names.pkl` — список признаков
- `models/model_metrics.txt` — метрики
- `models/shap_summary.png` — SHAP интерпретация

### Детекция фрода

```bash
python src/two_stage_detector.py
```

Результаты:
- `docs/two_stage_detection_results.csv` — результаты детекции

### FastAPI сервис

**Инициализация базы данных:**
```bash
python src/init_db.py
```
Создаёт SQLite базу `data/transactions.db` с 13,140 транзакциями и всеми признаками.

**Запуск сервера:**
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

При старте автоматически проверяет все транзакции и находит фроды.

**Эндпоинты:**
| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/` | Статус сервиса |
| GET | `/frauds` | Список найденных фродов |
| GET | `/stats` | Статистика проверок |
| POST | `/check_now` | Мгновенная проверка новых транзакций |
| POST | `/reset` | Сброс статуса проверки (для демо) |

**Swagger UI:** http://localhost:8000/docs

**Пример ответа `/stats`:**
```json
{
  "total_transactions": 13140,
  "checked": 13140,
  "fraud_detected": 171,
  "actual_fraud_in_data": 165,
  "threshold": 0.80
}
```

### Дообучение модели

```bash
python src/retrain.py
python src/retrain.py --force  # принудительное сохранение
```

Функции:
- Автоматический бэкап текущей модели
- Сравнение метрик старой и новой модели
- Сохранение только если новая лучше
- Возможность rollback

### Тестирование

```bash
# benchmark скорости
python src/benchmark.py

# cross-validation стабильность
python src/cv_stability.py
```

## Структура проекта

```
FraudDetector/
├── src/
│   ├── train_catboost.py     # обучение модели
│   ├── two_stage_detector.py # детекция
│   ├── preprocessing.py      # подготовка данных
│   ├── config.py             # настройки порогов
│   ├── retrain.py            # дообучение
│   ├── benchmark.py          # тест скорости
│   ├── cv_stability.py       # кросс-валидация
│   ├── api.py                # FastAPI сервис
│   └── init_db.py            # инициализация БД
├── models/
│   ├── catboost_fraud_model.cbm
│   ├── feature_names.pkl
│   └── model_metrics.txt
├── data/
│   └── transactions.db       # SQLite база данных
├── docs/
│   └── two_stage_detection_results.csv
└── notebooks/
    └── FraudDetection_Analysis.ipynb
```

## Настройка порогов

В `src/config.py`:

```python
ML_PREDICTION_THRESHOLD = 0.80  # порог для ML модели
SCORECARD_THRESHOLD = -1        # порог scorecard (-1 = отключен)
```

**Как выбирать:**
- Выше `ML_PREDICTION_THRESHOLD` → меньше FP, больше FN
- Ниже `ML_PREDICTION_THRESHOLD` → больше recall, больше FP

## Признаки модели

**Поведенческие (основные):**
- `logins_last_7_days`, `logins_last_30_days` — активность входов
- `monthly_os_changes`, `monthly_phone_model_changes` — смена устройств
- `avg_login_interval_30d`, `std_login_interval_30d` — паттерны входа
- `burstiness_login_interval`, `fano_factor_login_interval` — аномалии

**Транзакционные:**
- `amount` — сумма перевода
- `direction` — направление (входящий/исходящий)

**Категориальные:**
- `last_phone_model_categorical`, `last_os_categorical`

## Интерпретируемость

SHAP объяснения сохраняются при обучении:
- `models/shap_summary.png` — beeswarm plot
- `models/shap_bar.png` — bar plot
- `models/shap_importance.csv` — важность признаков

## Бизнес-эффект

```
Средняя сумма фрода: 228,500 KZT
Пойманных фродов (TP): 155
Предотвращённые потери: 35,417,500 KZT
```
