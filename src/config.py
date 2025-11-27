# Risk levels mapping
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

# Risk thresholds for fraud probability
THRESHOLDS = {
    'low': 0.30,      # < 0.30 = LOW risk
    'medium': 0.60,   # 0.30-0.60 = MEDIUM risk
    'high': 0.80      # 0.60-0.80 = HIGH risk, >= 0.80 = CRITICAL
}

# ML model prediction threshold (higher = fewer FP, lower recall)
ML_PREDICTION_THRESHOLD = 0.80  # Optimal balance: 91% precision, 92% recall

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

# Hyperparameter grid for CatBoost tuning
HYPERPARAM_GRID = {
    'depth': [6, 8, 10],
    'learning_rate': [0.03, 0.05, 0.1],
    'l2_leaf_reg': [3, 5, 7],
    'iterations': [1500, 2000, 2500]
}

# Toggle grid search usage
USE_GRID_SEARCH = False # Set to True for hyperparameter tuning (slow!)

# GPU usage flag with automatic fallback
USE_GPU = True       # Set to True to enable GPU acceleration (auto-fallback to CPU if unavailable)
GPU_DEVICE_ID = 0       # GPU device index (usually 0)

# Ray Tune integration flags for distributed hyperparameter search
USE_RAY = False        # Ray создает overhead! Используем только для серьезного HP tuning
RAY_NUM_WORKERS = 2    # Для GPU используем меньше (GPU сам параллелится)
