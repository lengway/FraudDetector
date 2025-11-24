"""
CatBoost Model Configuration
ForteBank Fraud Detection
"""

# Model Hyperparameters
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'use_best_model': True
}

# Feature Configuration
FEATURE_GROUPS = {
    'temporal': [
        'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_night', 'is_business_hours'
    ],
    'amount': [
        'amount', 'amount_log', 'amount_vs_user_avg', 'amount_percentile'
    ],
    'user': [
        'user_avg_amount', 'user_std_amount', 'user_tx_count', 'user_fraud_count'
    ],
    'behavioral': [
        'monthly_os_changes', 'monthly_phone_model_changes',
        'logins_last_7_days', 'logins_last_30_days',
        'login_frequency_7d', 'login_frequency_30d',
        'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
        'avg_login_interval_30d', 'std_login_interval_30d',
        'burstiness_login_interval', 'fano_factor_login_interval',
        'zscore_avg_login_interval_7d'
    ]
}

# Categorical features
CATEGORICAL_FEATURES = [
    'day_of_week', 'month', 'is_weekend', 'is_night', 'is_business_hours'
]

# Data paths
DATA_PATHS = {
    'transactions': 'docs/транзакции в Мобильном интернет Банкинге.csv',
    'behavioral': 'docs/поведенческие паттерны клиентов.csv'
}

# Model paths
MODEL_PATHS = {
    'model': 'models/catboost_fraud_model.cbm',
    'features': 'models/feature_names.pkl',
    'metrics': 'models/model_metrics.txt'
}

# Data loading parameters
DATA_PARAMS = {
    'sep': ';',
    'encoding': 'cp1251'
}

# Train/test split
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Risk level mapping
RISK_LEVELS = {
    'low': 'LOW',
    'medium': 'MEDIUM',
    'high': 'HIGH',
    'critical': 'CRITICAL'
}

# Action recommendations
RECOMMENDATIONS = {
    'low': 'APPROVE',
    'medium': 'REVIEW',
    'high': 'ADDITIONAL_VERIFICATION',
    'critical': 'BLOCK'
}

# Metrics targets
METRIC_TARGETS = {
    'auc_roc': 0.90,
    'precision': 0.85,
    'recall': 0.80,
    'f1_score': 0.82
}

# Business impact
BUSINESS_METRICS = {
    'avg_fraud_amount': 50000,  # KZT
    'false_positive_cost': 100,  # KZT (operational cost)
    'false_negative_multiplier': 500  # multiplier of avg fraud amount
}
