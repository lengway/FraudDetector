# Быстрая установка и настройка

## 📦 Шаг 1: Установка зависимостей

### Базовая установка (CPU only)
```bash
pip install -r requirements.txt
```

---

## 🎮 Шаг 2: Настройка GPU (опционально)

### Проверка GPU
```bash
# Проверить наличие NVIDIA GPU
nvidia-smi
```

**Если видишь вывод с GPU** → можно использовать GPU  
**Если ошибка** → нет GPU или драйверов, используй CPU

### Установка CUDA (если нужно)

**Windows:**
1. Скачай [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (версия 11.8 или 12.x)
2. Установи с настройками по умолчанию
3. Перезагрузи компьютер
4. Проверь: `nvcc --version`

**Проверка CatBoost GPU:**
```python
python test_gpu.py
```

Должно показать: `✅ GPU detected and available for training`

---

## ⚡ Шаг 3: Настройка Ray Tune (опционально)

**ВАЖНО:** Требует Python <= 3.12

### Проверка версии Python
```bash
python --version
```

### Если Python 3.12 или ниже:
```bash
# Ray уже в requirements.txt, просто проверь
python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
```

**Если работает** → можешь использовать Ray  
**Если ошибка ImportError** → установи отдельно:
```bash
pip install "ray[default]>=2.8.0"
```

### Если Python 3.13:
Ray пока не поддерживается. Варианты:
1. Использовать grid search (уже работает, просто медленнее)
2. Установить Python 3.12 через pyenv/conda

---

## 🚀 Шаг 4: Настройка config.py

Открой `src/config.py` и настрой:

### Вариант 1: Быстрое обучение (рекомендуется для начала)
```python
USE_GRID_SEARCH = False  # Без перебора параметров (быстро ~5 минут)
USE_GPU = False          # CPU пока (потом включишь)
USE_RAY = False          # Без Ray
```

### Вариант 2: С GPU (если есть)
```python
USE_GRID_SEARCH = False  # Быстро
USE_GPU = True           # Попытка GPU (fallback на CPU)
USE_RAY = False
```

### Вариант 3: Полная оптимизация (долго ~30-60 минут)
```python
USE_GRID_SEARCH = True   # Перебор параметров
USE_GPU = True           # GPU если есть
USE_RAY = False          # Или True если Ray работает
```

### Вариант 4: Максимум (Ray + GPU)
```python
USE_GRID_SEARCH = False  # Ray заменяет grid search
USE_GPU = True
USE_RAY = True
RAY_NUM_WORKERS = 4      # Количество параллельных процессов
```

---

## ✅ Шаг 5: Запуск обучения

```bash
python src/train_catboost.py
```

### Что должно произойти:

**С GPU:**
```
✅ GPU detected and available for training
🎯 Training final model on GPU without grid search...
```

**Без GPU (fallback):**
```
⚠️ GPU requested but not available (CUDA not found), falling back to CPU
🎯 Training final model on CPU without grid search...
```

**С Ray Tune:**
```
🚀 Using Ray Tune for distributed hyperparameter search
Ray initialized successfully...
```

**С grid search:**
```
🔍 Running grid search on CPU...
Testing 81 parameter combinations...
```

---

## 🎯 Результаты

После обучения в папке `models/` появятся:
- `catboost_fraud_model.cbm` - обученная модель
- `feature_names.pkl` - список фичей
- `model_metrics.txt` - детальные метрики

---

## 🐛 Troubleshooting

### GPU не обнаружен
1. Проверь: `nvidia-smi`
2. Установи [NVIDIA драйверы](https://www.nvidia.com/Download/index.aspx)
3. Установи [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
4. Перезагрузи ПК
5. Запусти: `python test_gpu.py`

### Ray не работает
```
⚠️ Ray not installed
```
**Решение:** 
- Проверь версию Python: `python --version` (должно быть <= 3.12)
- Установи: `pip install "ray[default]>=2.8.0"`
- Или используй grid search: `USE_RAY = False`

### Grid search слишком долгий
**Решение:**
- Выключи: `USE_GRID_SEARCH = False`
- Или включи Ray: `USE_RAY = True` (быстрее)
- Или уменьши сетку в `HYPERPARAM_GRID` (меньше значений)

### Ошибка "CUDA out of memory"
**Решение:**
- Уменьши batch size (если используется)
- Или используй CPU: `USE_GPU = False`

---

## 📊 Примерное время обучения

| Режим | CPU | GPU (RTX 4060) |
|-------|-----|----------------|
| Без grid search | ~5 мин | ~2 мин |
| Grid search (81 комб) | ~60 мин | ~15 мин |
| Ray Tune (10 trials) | ~30 мин | ~8 мин |

---

## 🎉 Готово!

Теперь можешь:
1. Обучить модель: `python src/train_catboost.py`
2. Проверить метрики: `cat models/model_metrics.txt`
3. Запустить предсказания: `python src/predict.py`
4. Эксперименты с порогами: `python src/experiment_threshold.py`

Подробности в:
- `GPU_RAY_GUIDE.md` - детали GPU и Ray
- `PROJECT_ANALYSIS.md` - полный анализ проекта
- `CHANGELOG.md` - что было исправлено
