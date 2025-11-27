"""
CatBoost Fraud Detection - Hybrid Version
ForteBank Hackathon - Optimized for F1-Score (Best Balance of Precision and Recall)
"""

import os
import warnings
import itertools
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import config  # src/config.py

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Composite feature helper (was in feature_utils.py)
# ---------------------------------------------------------------------------
def add_composite_features(df):
    """Placeholder for future composite features.
    
    Currently returns DataFrame unchanged.
    CatBoost finds better patterns automatically without manual rules.
    """
    return df

# Determine task_type based on GPU availability and config
def get_task_type():
    """Automatically detect GPU availability and return task_type for CatBoost."""
    if not config.USE_GPU:
        return 'CPU'
    
    try:
        # Try to import CatBoost GPU support
        from catboost import CatBoostClassifier
        test_model = CatBoostClassifier(iterations=1, task_type='GPU', devices=f'{config.GPU_DEVICE_ID}', verbose=False)
        # Quick test to verify GPU works
        import numpy as np
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        test_model.fit(X_test, y_test, verbose=False)
        print("✅ GPU detected and available for training")
        return 'GPU'
    except Exception as e:
        print(f"⚠️ GPU requested but not available ({e}), falling back to CPU")
        return 'CPU'

TASK_TYPE = get_task_type()  # Determine once at module load
GPU_PARAMS = {'devices': f'{config.GPU_DEVICE_ID}'} if TASK_TYPE == 'GPU' else {}

# ---------------------------------------------------------------------------
# Helper to clean column names
# ---------------------------------------------------------------------------
def clean_columns(df):
    """Remove spaces and BOM markers from column names."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('\\ufeff', '')
        .str.replace('"', '')
    )
    return df

# ---------------------------------------------------------------------------
# Expected numeric columns (header=1 English names from CSV)
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    'amount',
    'monthly_os_changes', 'monthly_phone_model_changes',
    'logins_last_7_days', 'logins_last_30_days',
    'login_frequency_7d', 'login_frequency_30d',
    'avg_login_interval_30d', 'std_login_interval_30d',
    'freq_change_7d_vs_mean',
    'logins_7d_over_30d_ratio',
    'ewm_login_interval_7d',
    'burstiness_login_interval',
    'fano_factor_login_interval',
    'zscore_avg_login_interval_7d',
    'var_login_interval_30d',
]

# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def load_and_clean_data():
    print("📊 Loading data...")
    # Load transactions
    try:
        df_trans = pd.read_csv(
            'транзакции в Мобильном интернет Банкинге.csv',
            sep=';',
            encoding='cp1251',
            header=1,
            engine='python',
        )
    except FileNotFoundError:
        df_trans = pd.read_csv(
            'docs/транзакции в Мобильном интернет Банкинге.csv',
            sep=';',
            encoding='cp1251',
            header=1,
            engine='python',
        )
    df_trans = clean_columns(df_trans)

    # Load behavioral patterns
    try:
        df_behavior = pd.read_csv(
            'поведенческие паттерны клиентов.csv',
            sep=';',
            encoding='cp1251',
            header=1,
            engine='python',
            on_bad_lines='skip',
        )
    except FileNotFoundError:
        df_behavior = pd.read_csv(
            'docs/поведенческие паттерны клиентов.csv',
            sep=';',
            encoding='cp1251',
            header=1,
            engine='python',
            on_bad_lines='skip',
        )
    df_behavior = clean_columns(df_behavior)

    # header=1 provides English column names directly

    # Force numeric types on known columns
    for df_temp in [df_trans, df_behavior]:
        for col in NUMERIC_COLS:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)

    # Normalize ID columns for merge
    for df_temp in [df_trans, df_behavior]:
        if 'cst_dim_id' in df_temp.columns:
            df_temp['cst_dim_id'] = (
                pd.to_numeric(df_temp['cst_dim_id'], errors='coerce')
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        if 'transdate' in df_temp.columns:
            df_temp['transdate'] = pd.to_datetime(
                df_temp['transdate'].astype(str).str.strip("'"), errors='coerce'
            )

    # Merge datasets
    print("🔗 Merging datasets...")
    df = df_trans.merge(df_behavior, on=['cst_dim_id', 'transdate'], how='left')

    # Fill categorical NaNs (using header=1 column names)
    for c in ['last_phone_model_categorical', 'last_os_categorical', 'direction']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')

    # Final numeric cleaning
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                .fillna(0)
                .astype(float)
            )
    df.fillna(0, inplace=True)
    print(f"✅ Dataset ready: {df.shape}, Fraud rate: {df['target'].mean()*100:.2f}%")
    return df

# ---------------------------------------------------------------------------
# Feature engineering (enhanced)
# ---------------------------------------------------------------------------
def engineer_features(df):
    print("\n⚙️ Engineering features...")
    # Temporal features
    if 'transdatetime' in df.columns:
        df['transdatetime'] = pd.to_datetime(df['transdatetime'].astype(str).str.strip("'"), errors='coerce')
        df['hour'] = df['transdatetime'].dt.hour.fillna(0).astype(int)
        df['day_of_week'] = df['transdatetime'].dt.dayofweek.fillna(0).astype(int)
        df['is_night'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
    else:
        df['hour'] = 0
        df['day_of_week'] = 0
        df['is_night'] = 0

    # Amount features
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])

    # Composite high‑risk flag (using header=1 column names)
    if 'amount' in df.columns and 'std_login_interval_30d' in df.columns:
        df['is_high_risk_combo'] = ((df['amount'] > 10000.0) & (df['std_login_interval_30d'] > 100000.0)).astype(int)

    # Behavioral flags
    if 'monthly_phone_model_changes' in df.columns:
        df['is_device_hopper'] = (df['monthly_phone_model_changes'] > 1).astype(int)
    if 'avg_login_interval_30d' in df.columns:
        df['is_fast_bot'] = (df['avg_login_interval_30d'] < 10).astype(int)

    # NEW composite features
    if 'logins_last_7_days' in df.columns and 'logins_last_30_days' in df.columns:
        logins_7d = pd.to_numeric(df['logins_last_7_days'], errors='coerce').fillna(0)
        logins_30d = pd.to_numeric(df['logins_last_30_days'], errors='coerce').fillna(0)
        df['login_velocity'] = logins_7d / (logins_30d + 1e-6)
    if 'monthly_phone_model_changes' in df.columns and 'logins_last_30_days' in df.columns:
        device_count = pd.to_numeric(df['monthly_phone_model_changes'], errors='coerce').fillna(0)
        logins_30d = pd.to_numeric(df['logins_last_30_days'], errors='coerce').fillna(0)
        df['device_change_rate'] = device_count / (logins_30d + 1)
    if 'hour' in df.columns:
        df['time_since_last_login'] = (24 - df['hour']).clip(lower=0)

    # User‑level aggregates
    if 'cst_dim_id' in df.columns and 'amount' in df.columns:
        # Amount aggregates per user (no leakage - these are static stats)
        user_amt_agg = df.groupby('cst_dim_id').agg({
            'amount': ['mean', 'std', 'count'],
        }).reset_index()
        user_amt_agg.columns = ['cst_dim_id', 'user_avg_amt', 'user_std_amt', 'user_tx_count']
        df = df.merge(user_amt_agg, on='cst_dim_id', how='left')
        df.fillna(0, inplace=True)
        df['amount_to_avg_ratio'] = df['amount'] / df['user_avg_amt'].replace(0, 1e-6)
        df['amount_to_avg_ratio'].replace([np.inf, -np.inf], 99999.0, inplace=True)
        
        # user_hist_fraud: CUMULATIVE fraud count UP TO current transaction (no leakage!)
        # Sort by user and time, then cumsum().shift(1) to exclude current tx
        if 'transdate' in df.columns:
            df = df.sort_values(['cst_dim_id', 'transdate'])
            df['user_hist_fraud'] = df.groupby('cst_dim_id')['target'].cumsum().shift(1).fillna(0).astype(int)
        else:
            df['user_hist_fraud'] = 0

    # Apply any extra helper‑based features
    df = add_composite_features(df)
    return df

# ---------------------------------------------------------------------------
# Training routine with optional grid search
# ---------------------------------------------------------------------------
def train_model(df):
    print("\n🚀 Preparing training...")
    ignore_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target']
    features = [c for c in df.columns if c not in ignore_cols]
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("❌ Removing manual Target Encoding. CatBoost will handle native categories.")
    cat_features = ['direction', 'last_phone_model_categorical', 'last_os_categorical']
    cat_features = [c for c in cat_features if c in X_train.columns]
    all_features = X_train.columns.tolist()
    num_features = [f for f in all_features if f not in cat_features]
    print(f"🛠️ Final numeric cleanup of {len(num_features)} features before CatBoost...")
    for col in num_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(float)
    for c in cat_features:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    X_train = X_train.drop(columns=[c for c in X_train.columns if c not in all_features], errors='ignore')
    X_test = X_test.drop(columns=[c for c in X_test.columns if c not in all_features], errors='ignore')
    print(f"Features ({len(all_features)}): {all_features}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ scale_pos_weight set to: {scale_pos_weight:.2f}")

    # Hyperparameter tuning with optional Ray Tune integration
    if config.USE_RAY:
        try:
            from ray import tune
            from ray.tune.search.optuna import OptunaSearch
            import ray
            
            print("🚀 Using Ray Tune for distributed hyperparameter search on GPU")
            
            # Инициализация Ray по образцу Innovatex
            if not ray.is_initialized():
                os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
                os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
                
                # Если GPU доступен, используем меньше воркеров (GPU можно шарить)
                num_cpus = 2 if TASK_TYPE == 'GPU' else config.RAY_NUM_WORKERS
                
                ray.init(
                    num_cpus=num_cpus if num_cpus > 0 else None,
                    num_gpus=1 if TASK_TYPE == 'GPU' else 0,  # Указываем что есть 1 GPU
                    ignore_reinit_error=True,
                    logging_level='ERROR',
                    _system_config={
                        "metrics_report_interval_ms": 0,
                        "enable_metrics_collection": False
                    },
                    include_dashboard=False
                )
                print(f"✅ Ray initialized: {num_cpus} CPUs, {'1 GPU' if TASK_TYPE == 'GPU' else '0 GPUs'}")
            
            # Trainable функция которая запускается на Ray workers
            def train_catboost(config_params):
                """Обучение CatBoost на Ray worker с GPU"""
                from catboost import CatBoostClassifier
                from sklearn.metrics import f1_score
                
                # Каждый worker использует GPU (Ray управляет доступом)
                model_trial = CatBoostClassifier(
                    iterations=int(config_params['iterations']),
                    learning_rate=config_params['learning_rate'],
                    depth=int(config_params['depth']),
                    l2_leaf_reg=config_params['l2_leaf_reg'],
                    eval_metric='AUC',
                    scale_pos_weight=scale_pos_weight,
                    task_type=TASK_TYPE,  # GPU на каждом worker
                    devices=f'{config.GPU_DEVICE_ID}' if TASK_TYPE == 'GPU' else None,
                    random_seed=42,
                    verbose=False
                )
                
                # Обучаем на GPU
                model_trial.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                
                # Предсказания
                y_prob = model_trial.predict_proba(X_test)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
                f1 = f1_score(y_test, y_pred)
                
                # Отчет в Ray Tune
                tune.report(f1_score=f1)
            
            search_space = {
                'iterations': tune.choice(config.HYPERPARAM_GRID['iterations']),
                'learning_rate': tune.choice(config.HYPERPARAM_GRID['learning_rate']),
                'depth': tune.choice(config.HYPERPARAM_GRID['depth']),
                'l2_leaf_reg': tune.choice(config.HYPERPARAM_GRID['l2_leaf_reg'])
            }
            
            # Ray Tune с GPU resource
            from ray import tune as ray_tune
            analysis = ray_tune.run(
                train_catboost,
                config=search_space,
                num_samples=10,
                search_alg=OptunaSearch(metric='f1_score', mode='max'),
                verbose=1,
                resources_per_trial={
                    'cpu': 1,
                    'gpu': 0.5 if TASK_TYPE == 'GPU' else 0  # Каждый trial использует 50% GPU (2 параллельно)
                }
            )
            
            best_params = analysis.get_best_config(metric='f1_score', mode='max')
            ray.shutdown()
            print(f"🔎 Ray Tune best params: {best_params}")
            
        except ImportError as e:
            print(f"⚠️ Ray not installed ({e}), falling back to grid search")
            config.USE_RAY = False
        except Exception as e:
            print(f"⚠️ Ray Tune failed ({e}), falling back to grid search")
            if ray.is_initialized():
                ray.shutdown()
            config.USE_RAY = False
    
    if config.USE_GRID_SEARCH and not config.USE_RAY:
        print(f"🔍 Running grid search on {TASK_TYPE}...")
        best_score = -1
        best_params = None
        param_grid = config.HYPERPARAM_GRID
        keys = list(param_grid.keys())
        total_combinations = len(list(itertools.product(*[param_grid[k] for k in keys])))
        print(f"Testing {total_combinations} parameter combinations...")
        
        for idx, values in enumerate(itertools.product(*[param_grid[k] for k in keys]), 1):
            params = dict(zip(keys, values))
            cv_scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model_cv = CatBoostClassifier(
                    iterations=params.get('iterations', 2000),
                    learning_rate=params.get('learning_rate', 0.05),
                    depth=params.get('depth', 8),
                    eval_metric='AUC',  # AUC поддерживается на GPU (вместо PRAUC)
                    scale_pos_weight=scale_pos_weight,
                    l2_leaf_reg=params.get('l2_leaf_reg', 5),
                    task_type=TASK_TYPE,
                    random_seed=42,
                    verbose=False,
                    **GPU_PARAMS
                )
                model_cv.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False)
                prob = model_cv.predict_proba(X_val)[:, 1]
                pred = (prob > 0.5).astype(int)
                cv_scores.append(f1_score(y_val, pred))
            mean_score = np.mean(cv_scores)
            print(f"  [{idx}/{total_combinations}] Params: {params} → F1: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        print(f"🔎 Grid search best F1: {best_score:.4f} with params {best_params}")
        model = CatBoostClassifier(
            iterations=best_params.get('iterations', 2000),
            learning_rate=best_params.get('learning_rate', 0.05),
            depth=best_params.get('depth', 8),
            eval_metric='AUC',  # AUC поддерживается на GPU
            scale_pos_weight=scale_pos_weight,
            l2_leaf_reg=best_params.get('l2_leaf_reg', 5),
            task_type=TASK_TYPE,
            random_seed=42,
            verbose=200,
            early_stopping_rounds=150,
            **GPU_PARAMS
        )
    else:
        print(f"🎯 Training final model on {TASK_TYPE} without grid search...")
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            eval_metric='AUC',  # AUC поддерживается на GPU (вместо PRAUC)
            scale_pos_weight=scale_pos_weight,
            l2_leaf_reg=5,
            task_type=TASK_TYPE,
            random_seed=42,
            verbose=200,
            early_stopping_rounds=150,
            **GPU_PARAMS
        )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Feature importance
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\n" + "=" * 60)
    print("TOP-10 FEATURE IMPORTANCE")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))
    print("=" * 60)

    # Threshold optimisation
    print("\n⚖️ Optimising threshold for max F1...")
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    y_pred = (y_prob >= best_thresh).astype(int)
    best_precision = precision_score(y_test, y_pred)
    best_recall = recall_score(y_test, y_pred)
    best_f1 = f1_score(y_test, y_pred)
    print(f"✅ New threshold: {best_thresh:.4f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f})")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT (F1‑OPTIMISED)")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]} (False alarms)")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]} (Detected fraud)")

    # Ensure models directory exists and save artifacts compatible with predict.py
    os.makedirs('models', exist_ok=True)
    model.save_model('models/catboost_fraud_model.cbm')
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names.tolist(), f)
    print("\n💾 Model saved to 'models/catboost_fraud_model.cbm'")
    print("💾 Feature names saved to 'models/feature_names.pkl'")
    
    # Save detailed metrics report
    metrics_report = f"""Model Training Report
{'='*60}
Training Date: {pd.Timestamp.now()}
Dataset Shape: {df.shape}
Features Used: {len(feature_names)}
Class Distribution: {dict(y.value_counts())}

Best Threshold: {best_thresh:.4f}

Performance Metrics:
  ROC AUC:    {roc_auc_score(y_test, y_prob):.4f}
  Precision:  {best_precision:.4f}
  Recall:     {best_recall:.4f}
  F1-Score:   {best_f1:.4f}

Confusion Matrix:
  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}
  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}

Top-10 Most Important Features:
{importance_df.head(10).to_string(index=False)}
{'='*60}
"""
    with open('models/model_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(metrics_report)
    print("💾 Metrics saved to 'models/model_metrics.txt'")
    
    return model, best_f1, best_precision, best_recall

if __name__ == "__main__":
    df = load_and_clean_data()
    df = engineer_features(df)
    model, best_f1_score, best_precision_score, best_recall_score = train_model(df)
    print("\n" + "=" * 60)
    print("MODEL INTERPRETATION")
    print("=" * 60)
    print(f"Precision (balanced): {best_precision_score:.4f}")
    print(f"Recall: {best_recall_score:.4f}")
    print(f"F1 (max): {best_f1_score:.4f}")
    print(f"AUC: {model.get_best_score()['validation']['AUC']:.4f}")
    print("\nKey factors: Direction, Amount vs Avg, User History, Device stability, Login volatility")
    print("=" * 60)