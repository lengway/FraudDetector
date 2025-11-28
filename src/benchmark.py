import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
import warnings

from train_catboost import load_and_clean_data, engineer_features, add_composite_features

warnings.filterwarnings('ignore')

MODEL_PATH = "models/catboost_fraud_model.cbm"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
N_ITERATIONS = 10
BATCH_SIZES = [1, 10, 100, 500, 1000]


def scorecard_score(row):
    """быстрый скоринг по правилам"""
    score = 0
    if row.get('monthly_os_changes', 0) >= 3:
        score += 2
    if row.get('monthly_phone_model_changes', 0) >= 2:
        score += 2
    if row.get('logins_last_7_days', 0) >= 10:
        score += 1
    if row.get('login_frequency_7d', 0) >= 2:
        score += 1
    if row.get('logins_7d_over_30d_ratio', 0) >= 0.7:
        score += 2
    if row.get('direction_out', 0) == 1:
        score += 1
    if row.get('amount_vs_avg', 1) >= 3:
        score += 2
    if row.get('amount_vs_avg', 1) >= 5:
        score += 1
    if row.get('user_hist_fraud', 0) >= 1:
        score += 3
    return score


def benchmark_feature_engineering(df, n_iterations=N_ITERATIONS):
    """замер времени feature engineering"""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = add_composite_features(df)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark_scorecard(df, n_iterations=N_ITERATIONS):
    """замер времени scorecard"""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = df.apply(scorecard_score, axis=1)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark_ml_inference(model, X, n_iterations=N_ITERATIONS):
    """замер времени ml inference"""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict_proba(X)[:, 1]
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark_full_pipeline(model, df, feature_names, n_iterations=N_ITERATIONS):
    """замер полного пайплайна"""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        df_feat = add_composite_features(df)
        _ = df_feat.apply(scorecard_score, axis=1)
        available_features = [f for f in feature_names if f in df_feat.columns]
        X = df_feat[available_features].fillna(0)
        _ = model.predict_proba(X)[:, 1]
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000


def run_benchmark():
    print("=" * 60)
    print("benchmark")
    print("=" * 60)
    
    print("\nloading data...")
    df = load_and_clean_data()
    df_full = engineer_features(df)
    print(f"total records: {len(df_full)}")
    
    print("loading model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"model features: {len(feature_names)}")
    
    print("\n" + "=" * 60)
    print("results")
    print("=" * 60)
    
    results = []
    
    for batch_size in BATCH_SIZES:
        if batch_size <= len(df_full):
            df_sample = df_full.sample(n=batch_size, random_state=42)
        else:
            df_sample = df_full.copy()
            batch_size = len(df_full)
        
        df_sample = add_composite_features(df_sample)
        available_features = [f for f in feature_names if f in df_sample.columns]
        X = df_sample[available_features].fillna(0)
        
        feat_mean, _ = benchmark_feature_engineering(df_sample)
        score_mean, _ = benchmark_scorecard(df_sample)
        ml_mean, _ = benchmark_ml_inference(model, X)
        full_mean, _ = benchmark_full_pipeline(model, df_sample, feature_names)
        
        per_tx_full = full_mean / batch_size
        
        results.append({
            'batch_size': batch_size,
            'feature_eng_ms': feat_mean,
            'scorecard_ms': score_mean,
            'ml_inference_ms': ml_mean,
            'full_pipeline_ms': full_mean,
            'per_tx_ms': per_tx_full
        })
        
        print(f"\nbatch {batch_size}:")
        print(f"  feature eng: {feat_mean:.3f} ms ({feat_mean/batch_size:.4f} ms/tx)")
        print(f"  scorecard:   {score_mean:.3f} ms ({score_mean/batch_size:.4f} ms/tx)")
        print(f"  ml:          {ml_mean:.3f} ms ({ml_mean/batch_size:.4f} ms/tx)")
        print(f"  total:       {full_mean:.3f} ms ({per_tx_full:.4f} ms/tx)")
    
    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    
    r1000 = [r for r in results if r['batch_size'] == 1000][0]
    tps = 1000 / (r1000['full_pipeline_ms'] / 1000)
    
    print(f"\nthroughput (batch=1000): {tps:,.0f} tx/sec")
    print(f"latency: {r1000['per_tx_ms']:.4f} ms/tx")
    print(f"real-time: {'yes' if r1000['per_tx_ms'] < 100 else 'no'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/benchmark_results.csv', index=False)
    
    report = f"""benchmark report
================
throughput: {tps:,.0f} tx/sec
latency: {r1000['per_tx_ms']:.4f} ms/tx
real-time: {'yes' if r1000['per_tx_ms'] < 100 else 'no'}

breakdown (batch=1000):
- feature eng: {r1000['feature_eng_ms']:.2f} ms
- scorecard: {r1000['scorecard_ms']:.2f} ms  
- ml: {r1000['ml_inference_ms']:.2f} ms
- total: {r1000['full_pipeline_ms']:.2f} ms
"""
    
    with open('models/benchmark_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nsaved: models/benchmark_results.csv, models/benchmark_report.txt")


if __name__ == "__main__":
    run_benchmark()
