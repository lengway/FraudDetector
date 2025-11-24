"""
🧪 Experiment 1: Threshold Optimization
Test different decision thresholds for fraud detection
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and prepare features"""
    print("📊 Loading data...")
    
    # Load transactions
    df_trans = pd.read_csv(
        'docs/транзакции в Мобильном интернет Банкинге.csv',
        sep=';', encoding='cp1251', header=1
    )
    
    # Load behavioral
    df_behavior = pd.read_csv(
        'docs/поведенческие паттерны клиентов.csv',
        sep=';', encoding='cp1251', header=1
    )
    
    # Clean column names
    df_trans.columns = df_trans.columns.str.strip()
    df_behavior.columns = df_behavior.columns.str.strip()
    
    # Clean datetime
    df_trans['transdate'] = pd.to_datetime(df_trans['transdate'].str.strip("'"), format='%Y-%m-%d %H:%M:%S.%f')
    df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].str.strip("'"), format='%Y-%m-%d %H:%M:%S.%f')
    df_behavior['transdate'] = pd.to_datetime(df_behavior['transdate'].str.strip("'"), format='%Y-%m-%d %H:%M:%S.%f')
    
    # Merge
    df = df_trans.merge(df_behavior, on=['transdate', 'cst_dim_id'], how='left')
    
    # Convert to numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    
    # Create features
    df['hour'] = df['transdatetime'].dt.hour
    df['day_of_week'] = df['transdatetime'].dt.dayofweek
    df['day_of_month'] = df['transdatetime'].dt.day
    df['month'] = df['transdatetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(0, 6).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    
    # User aggregations
    user_agg = df.groupby('cst_dim_id').agg({
        'amount': ['mean', 'std', 'min', 'max', 'count'],
        'target': 'sum'
    }).reset_index()
    user_agg.columns = ['cst_dim_id', 'user_avg_amount', 'user_std_amount', 
                        'user_min_amount', 'user_max_amount', 'user_tx_count', 'user_fraud_count']
    df = df.merge(user_agg, on='cst_dim_id', how='left')
    
    df['amount_vs_user_avg'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-5)
    df['amount_percentile'] = df['amount'] / (df['user_max_amount'] + 1e-5)
    
    # Convert behavioral to numeric
    behavioral_cols = [
        'monthly_os_changes', 'monthly_phone_model_changes',
        'logins_last_7_days', 'logins_last_30_days',
        'login_frequency_7d', 'login_frequency_30d',
        'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
        'avg_login_interval_30d', 'std_login_interval_30d',
        'burstiness_login_interval', 'fano_factor_login_interval',
        'zscore_avg_login_interval_7d'
    ]
    
    for col in behavioral_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Select features
    feature_cols = [
        'amount', 'amount_log', 'amount_vs_user_avg', 'amount_percentile',
        'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_night', 'is_business_hours',
        'user_avg_amount', 'user_std_amount', 'user_tx_count', 'user_fraud_count',
        'monthly_os_changes', 'monthly_phone_model_changes',
        'logins_last_7_days', 'logins_last_30_days',
        'login_frequency_7d', 'login_frequency_30d',
        'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
        'avg_login_interval_30d', 'std_login_interval_30d',
        'burstiness_login_interval', 'fano_factor_login_interval',
        'zscore_avg_login_interval_7d'
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    print(f"✅ Data loaded: {X.shape}")
    return X, y

def test_threshold(model, X_test, y_test, threshold):
    """Test model with specific threshold"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'threshold': threshold,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def main():
    print("=" * 70)
    print("🧪 EXPERIMENT 1: THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load trained model
    print("\n📦 Loading trained model...")
    model = CatBoostClassifier()
    model.load_model('models/catboost_fraud_model.cbm')
    print("✅ Model loaded")
    
    # Test different thresholds
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    print("\n" + "=" * 70)
    print("📊 TESTING DIFFERENT THRESHOLDS")
    print("=" * 70)
    
    results = []
    for threshold in thresholds:
        result = test_threshold(model, X_test, y_test, threshold)
        results.append(result)
        
        print(f"\n🎯 Threshold: {threshold:.2f}")
        print(f"   AUC:       {result['auc']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall:    {result['recall']:.4f}")
        print(f"   F1-Score:  {result['f1']:.4f}")
        print(f"   Confusion Matrix:")
        print(f"   TN={result['cm'][0,0]}, FP={result['cm'][0,1]}")
        print(f"   FN={result['cm'][1,0]}, TP={result['cm'][1,1]}")
    
    # Find best threshold based on F1-score
    best_result = max(results, key=lambda x: x['f1'])
    
    print("\n" + "=" * 70)
    print("🏆 BEST THRESHOLD")
    print("=" * 70)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"AUC:       {best_result['auc']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-Score:  {best_result['f1']:.4f}")
    
    # Compare with baseline (0.5)
    baseline = [r for r in results if r['threshold'] == 0.5][0]
    print("\n📈 Improvement vs Baseline (0.5):")
    print(f"   Precision: {baseline['precision']:.4f} → {best_result['precision']:.4f} ({(best_result['precision']/baseline['precision']-1)*100:+.1f}%)")
    print(f"   Recall:    {baseline['recall']:.4f} → {best_result['recall']:.4f} ({(best_result['recall']/baseline['recall']-1)*100:+.1f}%)")
    print(f"   F1-Score:  {baseline['f1']:.4f} → {best_result['f1']:.4f} ({(best_result['f1']/baseline['f1']-1)*100:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT 1 COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
