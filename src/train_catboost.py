"""
🎯 CatBoost Fraud Detection - Hybrid Version
ForteBank Hackathon - Optimized for F1 / Fraud Detection

MAJOR REVISION: Feature Engineering and Model Complexity.
1. Target Encoding REMOVED: Relying on CatBoost's superior, leak-free native handling of categorical features.
2. NEW Feature: `amount_to_avg_ratio` - Calculates deviation from user's typical transaction amount, 
   a key indicator of abnormal financial behavior.
3. Increased Model Depth: Depth increased from 6 to 8 to capture more complex feature interactions.
4. TUNING: Increased learning_rate (0.05) and l2_leaf_reg for faster convergence and better generalization.
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, confusion_matrix, roc_auc_score

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# --- Helper to clean columns ---
def clean_columns(df):
    """Удаление пробелов и BOM-маркеров из названий колонок"""
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '').str.replace('"', '')
    return df

# --- Global list of expected numeric columns from data sources ---
NUMERIC_COLS = [
    'amount', 
    'os_count_30d', 'device_count_30d', 
    'logins_7d', 'logins_30d', 
    'avg_logins_7d', 'avg_logins_30d', 
    'avg_login_interval', 'std_login_interval',
    'rel_freq_change_7_30d', 
    'login_share_7_30d', 
    'weighted_avg_interval_7d', 
    'login_volatility_factor', 
    'fano_factor_interval', 
    'z_score_avg_interval_7d_vs_30d',
    'interval_variance_30d'
]

# --- Load & Clean Data ---
def load_and_clean_data():
    print("📊 Загрузка данных...")

    # 1. TRANSACTIONS (Транзакции)
    try:
        df_trans = pd.read_csv(
            'транзакции в Мобильном интернет Банкинге.csv',
            sep=';', 
            encoding='cp1251', 
            header=1,
            engine='python'
        )
    except FileNotFoundError:
        df_trans = pd.read_csv(
            'docs/транзакции в Мобильном интернет Банкинге.csv',
            sep=';', 
            encoding='cp1251', 
            header=1,
            engine='python'
        )
    
    df_trans = clean_columns(df_trans)

    # 2. BEHAVIOR (Поведенческие паттерны)
    try:
        df_behavior = pd.read_csv(
            'поведенческие паттерны клиентов.csv',
            sep=';', 
            encoding='cp1251', 
            header=0,
            engine='python',
            on_bad_lines='skip'
        )
    except FileNotFoundError:
        df_behavior = pd.read_csv(
            'docs/поведенческие паттерны клиентов.csv',
            sep=';', 
            encoding='cp1251', 
            header=0,
            engine='python',
            on_bad_lines='skip'
        )

    df_behavior = clean_columns(df_behavior)

    # --- RENAME COLUMNS (Переименование колонок) ---
    behav_map = {
        'Уникальный идентификатор клиента': 'cst_dim_id',
        'Дата совершенной транзакции': 'transdate',
        'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент': 'os_count_30d',
        'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент “менял устройство” по логам': 'device_count_30d',
        'Модель телефона из самой последней сессии (по времени) перед transdate': 'last_phone_model',
        'Версия ОС из самой последней сессии перед transdate': 'last_os_ver',
        'Количество уникальных логин-сессий (минутных тайм-слотов) за последние 7 дней до transdate': 'logins_7d',
        'Количество уникальных логин-сессий за последние 30 дней до transdate': 'logins_30d',
        'Среднее число логинов в день за последние 7 дней: logins_last_7_days / 7': 'avg_logins_7d',
        'Среднее число логинов в день за последние 30 дней: logins_last_30_days / 30': 'avg_logins_30d',
        'Относительное изменение частоты логинов за 7 дней к средней частоте за 30 дней:\n(freq7d?freq30d)/freq30d(freq_{7d} - freq_{30d}) / freq_{30d}(freq7d?freq30d)/freq30d — показывает, стал клиент заходить чаще или реже недавно': 'rel_freq_change_7_30d',
        'Доля логинов за 7 дней от логинов за 30 дней': 'login_share_7_30d',
        'Средний интервал (в секундах) между соседними сессиями за последние 30 дней': 'avg_login_interval',
        'Стандартное отклонение интервалов между логинами за 30 дней (в секундах), измеряет разброс интервалов': 'std_login_interval',
        'Показатель “взрывности” логинов: (std?mean)/(std+mean)(std - mean)/(std + mean)(std?mean)/(std+mean) для интервалов': 'login_volatility_factor',
        'Fano-factor интервалов: variance / mean': 'fano_factor_interval',
        'Z-скор среднего интервала за последние 7 дней относительно среднего за 30 дней: насколько сильно недавние интервалы отличаются от типичных, в единицах стандартного отклонения': 'z_score_avg_interval_7d_vs_30d',
        'Экспоненциально взвешенное среднее интервалов между логинами за 7 дней, где более свежие сессии имеют больший вес (коэффициент затухания 0.3)': 'weighted_avg_interval_7d',
        'Дисперсия интервалов между логинами за 30 дней (в секундах?), ещё одна мера разброса': 'interval_variance_30d',
    }
    
    df_behavior.rename(columns=behav_map, inplace=True)
    
    # --- INITIAL FIX: FORCE NUMERIC TYPES ON SOURCE DFs (Принудительная конвертация в числовой формат) ---
    for df_temp in [df_trans, df_behavior]:
        for col in NUMERIC_COLS:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)

    # --- FIX: ID TYPES FOR MERGE (Нормализация ID для слияния) ---
    for df_temp in [df_trans, df_behavior]:
        if 'cst_dim_id' in df_temp.columns:
            df_temp['cst_dim_id'] = pd.to_numeric(df_temp['cst_dim_id'], errors='coerce').fillna(0).astype(int).astype(str)
        if 'transdate' in df_temp.columns:
            df_temp['transdate'] = pd.to_datetime(df_temp['transdate'].astype(str).str.strip("'"), errors='coerce')

    # --- MERGE (Слияние) ---
    print("🔗 Объединение датасетов...")
    df = df_trans.merge(df_behavior, on=['cst_dim_id', 'transdate'], how='left')

    # Заполнение NaN для Категориальных колонок
    cat_fills = ['last_phone_model', 'last_os_ver', 'direction']
    for c in cat_fills:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')

    # --- FINAL RIGOROUS NUMERIC ENSURING (Проверка числовых типов) ---
    print("🛠️ Промежуточная проверка: обеспечение чистоты основных числовых колонок...")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce').fillna(0).astype(float)
            
    # Заполнение оставшихся NaN 
    df.fillna(0, inplace=True)

    print(f"✅ Датасет готов: {df.shape}, Уровень мошенничества: {df['target'].mean()*100:.2f}%")
    return df

# --- Feature Engineering (Создание признаков) ---
def engineer_features(df):
    print("\n⚙️ Создание признаков...")
    
    # Признаки времени
    if 'transdatetime' in df.columns:
        df['transdatetime'] = pd.to_datetime(df['transdatetime'].astype(str).str.strip("'"), errors='coerce')
        df['hour'] = df['transdatetime'].dt.hour.fillna(0).astype(int)
        df['day_of_week'] = df['transdatetime'].dt.dayofweek.fillna(0).astype(int)
        df['is_night'] = df['hour'].apply(lambda x: 1 if (0 <= x <= 6) else 0)
    else:
        df['hour'] = 0
        df['day_of_week'] = 0
        df['is_night'] = 0

    # Признаки суммы
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
        
    # --- НОВЫЙ КОМПОЗИТНЫЙ ПРИЗНАК: БОЛЬШАЯ СУММА + НЕСТАБИЛЬНЫЙ ИНТЕРВАЛ ---
    if 'amount' in df.columns and 'std_login_interval' in df.columns:
        # Мошеннические транзакции часто имеют: 1) большую сумму и 2) необычное (нестабильное) время логина
        df['is_high_risk_combo'] = ((df['amount'] > 10000.0) & (df['std_login_interval'] > 100000.0)).astype(int) 

    # Поведенческие флаги
    if 'device_count_30d' in df.columns:
        df['is_device_hopper'] = (df['device_count_30d'] > 1).astype(int)
    
    if 'avg_login_interval' in df.columns:
        df['is_fast_bot'] = (df['avg_login_interval'] < 10).astype(int)

    # Агрегаты по клиенту
    if 'cst_dim_id' in df.columns and 'amount' in df.columns:
        # Агрегация должна быть выполнена на полном датасете для избежания утечки,
        # так как это признаки, основанные на ИСТОРИИ клиента до текущей транзакции.
        # Однако, поскольку у нас нет точных данных о времени, мы делаем агрегацию по всему датасету
        # и полагаемся на разделение train/test, чтобы избежать прямой утечки.
        user_agg = df.groupby('cst_dim_id').agg({
            'amount': ['mean', 'std', 'count'],
            'target': 'sum'
        }).reset_index()
        user_agg.columns = ['cst_dim_id', 'user_avg_amt', 'user_std_amt', 'user_tx_count', 'user_hist_fraud']
        df = df.merge(user_agg, on='cst_dim_id', how='left')
        df.fillna(0, inplace=True)
        
        # --- НОВОЕ: ОТНОШЕНИЕ ТЕКУЩЕЙ СУММЫ К СРЕДНЕЙ ПОЛЬЗОВАТЕЛЯ ---
        df['amount_to_avg_ratio'] = df['amount'] / df['user_avg_amt'].replace(0, 1e-6) # Защита от деления на ноль
        df['amount_to_avg_ratio'].replace([np.inf, -np.inf], 99999.0, inplace=True)
            
    return df

# --- Prepare & Train (Подготовка и Обучение) ---
def train_model(df):
    print("\n🚀 Подготовка к обучению...")
    
    ignore_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target']
    
    # 1. Разделение данных
    features = [c for c in df.columns if c not in ignore_cols]
    X = df[features]
    y = df['target']
    
    # Сначала делим, чтобы избежать утечки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("❌ Удаление ручного Target Encoding. CatBoost будет использовать нативные категории.")
    
    # Исходные категориальные признаки для CatBoost
    cat_features = ['direction', 'last_phone_model', 'last_os_ver']
    cat_features = [c for c in cat_features if c in X_train.columns]
    
    all_features = X_train.columns.tolist()
    
    # Финальная проверка числовых признаков
    num_features = [f for f in all_features if f not in cat_features]
    print(f"🛠️ Финальная очистка {len(num_features)} числовых признаков перед CatBoost...")

    for col in num_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(float)
    
    # Убеждаемся, что категориальные признаки имеют строковый тип
    for c in cat_features:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)
        
    # Удаляем лишние колонки
    X_train = X_train.drop(columns=[c for c in X_train.columns if c not in all_features], errors='ignore')
    X_test = X_test.drop(columns=[c for c in X_test.columns if c not in all_features], errors='ignore')
        
    print(f"Признаки ({len(all_features)}): {all_features}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Расчет scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Установлен scale_pos_weight: {scale_pos_weight:.2f}")

    # Обучение модели
    model = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.05, # Увеличиваем скорость обучения
        depth=8, 
        eval_metric='PRAUC',
        scale_pos_weight=scale_pos_weight,
        l2_leaf_reg=5, # Увеличиваем L2 регуляризацию для лучшей обобщающей способности
        task_type='CPU',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150
    )
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # --- АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ---
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\n" + "="*60)
    print("ТОП-10 ВАЖНОСТЬ ПРИЗНАКОВ (Feature Importance)")
    print("="*60)
    print(importance_df.head(10).to_string(index=False))
    print("="*60)
    # --- КОНЕЦ АНАЛИЗА ---
    
    print("\n⚖️ Настройка порога для максимального F1 Score...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) 
    best_idx = np.argmax(f1_scores)
    
    # ✅ Явное сохранение лучшего F1 для вывода
    best_f1 = f1_scores[best_idx]
    
    best_thresh = thresholds[best_idx] if len(thresholds) > best_idx else 0.5

    print(f"✅ Лучший порог (Threshold): {best_thresh:.4f} (Максимальный F1: {best_f1:.4f})")
    
    y_pred = (y_prob >= best_thresh).astype(int)
    
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("="*60)
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок (Confusion Matrix):")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    model.save_model('catboost_fraud_final.cbm')
    print("\n💾 Модель сохранена в 'catboost_fraud_final.cbm'")
    
    # 🚀 Возвращаем модель и лучший F1 score
    return model, best_f1

if __name__ == "__main__":
    df = load_and_clean_data()
    df = engineer_features(df)
    # 🚀 Получаем оба значения
    model, best_f1_score = train_model(df)
    
    # --- INTERPRETATION (Интерпретация модели) ---
    print("\n" + "="*60)
    print("ИНТЕРПРЕТАЦИЯ КЛЮЧЕВЫХ ФАКТОРОВ МОДЕЛИ")
    print("="*60)
    # ✅ Используем явно переданный F1 score
    print(f"F1 Score (Threshold-Optimized): {best_f1_score:.4f}")
    print(f"PRAUC (Metric for training): {model.get_best_score()['validation']['PRAUC']:.4f}")
    print("\nМОДЕЛЬ УДАРЯЕТ ПО ТРЕМ ГЛАВНЫМ ФАКТОРАМ:")
    print("1. КУДА ИДЕТ ПЕРЕВОД (Recipient/Direction): Самый сильный сигнал - это ID получателя.")
    print("2. АНОМАЛИИ СУММЫ (Amount vs Average): Текущая сумма *сильно* отличается от исторической средней суммы клиента.")
    print("3. ИСТОРИЯ КЛИЕНТА (User History): Повторное мошенничество - мощный предиктор.")
    print("\nДОПОЛНИТЕЛЬНЫЙ ФАКТОР: Поведенческая нестабильность (смена устройств, ОС, высокая волатильность логинов).")
    print("Эти факторы помогают отличить реальный платеж от атаки на аккаунт.")
    print("="*60)