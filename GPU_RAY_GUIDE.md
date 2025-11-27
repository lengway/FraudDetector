# GPU и Ray Tune - Руководство

## ✅ Что теперь работает

### 1. GPU поддержка с автоматическим fallback
**Статус:** ✅ Полностью интегрировано

**Как работает:**
```python
# src/config.py
USE_GPU = True  # Включить попытку использовать GPU
GPU_DEVICE_ID = 0  # ID GPU устройства (обычно 0)
```

**Логика работы:**
1. При загрузке модуля `train_catboost.py` вызывается `get_task_type()`
2. Если `USE_GPU = False` → сразу возвращает `'CPU'`
3. Если `USE_GPU = True`:
   - Пытается создать тестовую модель CatBoost с `task_type='GPU'`
   - Обучает на случайных данных (10 строк)
   - Если успешно → возвращает `'GPU'` ✅
   - Если ошибка (нет GPU, нет драйверов, нет CUDA) → возвращает `'CPU'` ⚠️

**Пример вывода:**
```bash
# С GPU:
✅ GPU detected and available for training
🔍 Running grid search on GPU...

# Без GPU (автоматический fallback):
⚠️ GPU requested but not available (CUDA not found), falling back to CPU
🔍 Running grid search on CPU...
```

---

### 2. Ray Tune интеграция
**Статус:** ✅ Опционально интегрировано

**Установка:**
```bash
pip install ray[tune] optuna
```

**Как включить:**
```python
# src/config.py
USE_RAY = True
RAY_NUM_WORKERS = 4  # Количество параллельных испытаний
```

**Как работает:**
1. Если `USE_RAY = True` и Ray установлен:
   - Использует Ray Tune + Optuna для умного перебора гиперпараметров
   - Параллелизует испытания (быстрее чем grid search)
   - 10 итераций Optuna (можно изменить в коде)
2. Если Ray не установлен:
   - Выводит предупреждение: `⚠️ Ray not installed, falling back to grid search`
   - Автоматически переключается на обычный grid search

**Пример вывода:**
```bash
# С Ray:
🚀 Using Ray Tune for distributed hyperparameter search
[Ray Core] Started local Ray instance...
🔎 Ray Tune best params: {'iterations': 2000, 'learning_rate': 0.05, ...}

# Без Ray:
⚠️ Ray not installed, falling back to grid search
🔍 Running grid search on CPU...
Testing 81 parameter combinations...
```

---

## 🎯 Рекомендуемые сценарии использования

### Сценарий 1: Быстрое обучение на CPU (по умолчанию)
```python
# config.py
USE_GPU = False
USE_RAY = False
USE_GRID_SEARCH = False  # Или True для поиска HP
```
**Результат:** Обычное обучение на CPU, один набор параметров

---

### Сценарий 2: Ускорение на GPU
```python
# config.py
USE_GPU = True  # Попытается использовать GPU
USE_RAY = False
USE_GRID_SEARCH = True
```
**Результат:** Grid search с использованием GPU (если доступен), иначе CPU

---

### Сценарий 3: Продвинутый HP tuning с Ray
```python
# config.py
USE_GPU = True   # GPU для каждого trial (если доступен)
USE_RAY = True   # Distributed tuning
RAY_NUM_WORKERS = 8  # 8 параллельных испытаний
```
**Результат:** Умный перебор параметров с Optuna, параллельно на 8 воркерах

---

### Сценарий 4: Максимальная скорость (GPU + Ray)
```python
# config.py
USE_GPU = True
USE_RAY = True
RAY_NUM_WORKERS = 4  # Меньше воркеров если GPU один
USE_GRID_SEARCH = False  # Ray заменяет grid search
```
**Результат:** Optuna + GPU для каждого trial, быстрый поиск

---

## 🔧 Технические детали

### GPU параметры в CatBoost
Когда `TASK_TYPE = 'GPU'`, модели создаются с:
```python
CatBoostClassifier(
    task_type='GPU',
    devices='0',  # GPU_DEVICE_ID из config
    ...
)
```

### Автоматический fallback
```python
def get_task_type():
    if not config.USE_GPU:
        return 'CPU'
    
    try:
        # Тест GPU
        test_model = CatBoostClassifier(
            iterations=1, 
            task_type='GPU', 
            devices=f'{config.GPU_DEVICE_ID}', 
            verbose=False
        )
        test_model.fit(random_data, random_labels)
        return 'GPU'
    except Exception as e:
        print(f"⚠️ GPU not available ({e}), falling back to CPU")
        return 'CPU'
```

---

## 📊 Производительность

### Grid Search (81 комбинация, 3-fold CV = 243 обучения)

| Режим | Примерное время |
|-------|----------------|
| CPU only | ~60-120 минут |
| GPU (если доступен) | ~10-20 минут |

### Ray Tune (10 trials Optuna)

| Режим | Примерное время |
|-------|----------------|
| CPU, 4 воркера | ~20-30 минут |
| GPU, 1 воркер | ~5-10 минут |
| GPU, 4 воркера (4 GPU) | ~2-3 минуты |

*Время зависит от размера датасета и hardware*

---

## ⚙️ Требования

### Для GPU:
- CUDA Toolkit (11.0+) - [Скачать](https://developer.nvidia.com/cuda-downloads)
- NVIDIA GPU с CUDA support (RTX 3060+, RTX 4060+, etc.)
- CatBoost автоматически поддерживает GPU при наличии CUDA

**Проверка GPU:**
```bash
# Проверить NVIDIA драйвер
nvidia-smi

# Проверить CUDA (если установлен)
nvcc --version
```

**Установка CatBoost (с GPU support):**
```bash
pip install --upgrade catboost
```

### Для Ray:
**ВАЖНО:** Ray требует Python <= 3.12 (не поддерживает Python 3.13+)

```bash
# Установить Ray с зависимостями для Tune
pip install "ray[default]>=2.8.0"

# Или весь набор из requirements.txt
pip install -r requirements.txt
```

**Проверка Ray:**
```bash
python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
```

**Если Python 3.13:**
- Ray пока не поддерживается → используйте grid search
- Либо установите Python 3.12 отдельно через pyenv/conda

---

## 🐛 Troubleshooting

### GPU не обнаружен, но должен быть
1. Проверить драйвера: `nvidia-smi`
2. Проверить CUDA: `nvcc --version`
3. Переустановить CatBoost: `pip install --upgrade catboost`

### Ray падает с ошибкой
1. Увеличить `RAY_NUM_WORKERS` (меньше параллелизма)
2. Добавить `ray.init(num_cpus=2, ignore_reinit_error=True)`
3. Проверить версию: `pip install --upgrade ray[tune]`

### Grid search слишком долгий
1. Уменьшить `HYPERPARAM_GRID` в config.py (меньше значений)
2. Использовать `USE_RAY = True` для умного перебора
3. Отключить `USE_GRID_SEARCH = False` (использовать дефолтные параметры)

---

## 🎯 Итоговые рекомендации

**Для разработки/экспериментов:**
```python
USE_GPU = True   # Попытаться использовать
USE_RAY = False
USE_GRID_SEARCH = False
```

**Для production обучения:**
```python
USE_GPU = True
USE_RAY = True   # Если установлен Ray
USE_GRID_SEARCH = True  # Или USE_RAY для замены
```

**Если нет GPU:**
```python
USE_GPU = False
USE_RAY = True   # Параллельный поиск на CPU
```

---

**Автор:** AI Assistant  
**Дата:** 2025-11-27  
**Версия:** 1.0
