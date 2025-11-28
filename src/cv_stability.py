"""
cross-validation stability test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, fbeta_score, confusion_matrix, precision_recall_curve
)
import warnings

from train_catboost import load_and_clean_data, engineer_features

warnings.filterwarnings('ignore')

N_FOLDS = 5
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0,
    'early_stopping_rounds': 50,
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',
}


def run_cross_validation():
    print("=" * 60)
    print("cross-validation")
    print("=" * 60)
    
    print("\nloading data...")
    df = load_and_clean_data()
    df = engineer_features(df)
    print(f"records: {len(df)}, fraud rate: {df['target'].mean()*100:.2f}%")
    
    ignore_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target']
    features = [c for c in df.columns if c not in ignore_cols]
    X = df[features]
    y = df['target']
    
    cat_features = ['direction', 'last_phone_model_categorical', 'last_os_categorical']
    cat_features = [c for c in cat_features if c in X.columns]
    
    num_features = [f for f in X.columns if f not in cat_features]
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    for c in cat_features:
        X[c] = X[c].astype(str)
    
    print(f"features: {len(features)}, categorical: {cat_features}")
    print(f"\nrunning {N_FOLDS}-fold cv...")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, cat_features=cat_features, 
                  eval_set=(X_test, y_test), verbose=False)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'fold': fold, 'roc_auc': roc_auc, 'precision': precision,
            'recall': recall, 'f1': f1, 'f2': f2, 'threshold': optimal_threshold,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
        
        print(f"  fold {fold}: auc={roc_auc:.4f} rec={recall:.4f} prec={precision:.4f} f1={f1:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("results")
    print("=" * 60)
    
    mean_auc = results_df['roc_auc'].mean()
    std_auc = results_df['roc_auc'].std()
    mean_prec = results_df['precision'].mean()
    std_prec = results_df['precision'].std()
    mean_rec = results_df['recall'].mean()
    std_rec = results_df['recall'].std()
    mean_f1 = results_df['f1'].mean()
    std_f1 = results_df['f1'].std()
    mean_f2 = results_df['f2'].mean()
    std_f2 = results_df['f2'].std()
    
    print(f"\nroc-auc:   {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"precision: {mean_prec:.4f} +/- {std_prec:.4f}")
    print(f"recall:    {mean_rec:.4f} +/- {std_rec:.4f}")
    print(f"f1:        {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"f2:        {mean_f2:.4f} +/- {std_f2:.4f}")
    
    total_tp = results_df['tp'].sum()
    total_fp = results_df['fp'].sum()
    total_fn = results_df['fn'].sum()
    total_tn = results_df['tn'].sum()
    
    print(f"\naggregated confusion matrix:")
    print(f"  tn: {total_tn}  fp: {total_fp}")
    print(f"  fn: {total_fn}  tp: {total_tp}")
    
    results_df.to_csv('models/cv_results.csv', index=False)
    
    mean_threshold = results_df['threshold'].mean()
    report = f"""cross-validation report
=======================
folds: {N_FOLDS}
threshold: optimal per fold (avg: {mean_threshold:.3f})

roc-auc:   {mean_auc:.4f} +/- {std_auc:.4f}
precision: {mean_prec:.4f} +/- {std_prec:.4f}
recall:    {mean_rec:.4f} +/- {std_rec:.4f}
f1:        {mean_f1:.4f} +/- {std_f1:.4f}
f2:        {mean_f2:.4f} +/- {std_f2:.4f}

confusion matrix (all folds):
tp: {total_tp}, fp: {total_fp}, fn: {total_fn}, tn: {total_tn}
"""
    
    with open('models/cv_stability_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nsaved: models/cv_results.csv, models/cv_stability_report.txt")
    
    return results_df


if __name__ == "__main__":
    run_cross_validation()
