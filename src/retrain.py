import os
import sys
import shutil
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, fbeta_score
)

# add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, clean_and_merge, add_derived_features
from src.config import USE_GPU, GPU_DEVICE_ID


# model parameters
CATBOOST_PARAMS = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'task_type': 'GPU' if USE_GPU else 'CPU',
    'devices': str(GPU_DEVICE_ID) if USE_GPU else None,
    'verbose': False,
}

CATEGORICAL_FEATURES = ['direction', 'last_phone_model_categorical', 'last_os_categorical']
TARGET_COLUMN = 'target'


class ModelRetrainer:
    """handles model retraining with comparison and rollback capability."""
    
    def __init__(self, models_dir='models', backup_dir='models/backup'):
        self.models_dir = models_dir
        self.backup_dir = backup_dir
        self.model_path = os.path.join(models_dir, 'catboost_fraud_model.cbm')
        self.features_path = os.path.join(models_dir, 'feature_names.pkl')
        self.metrics_path = os.path.join(models_dir, 'model_metrics.txt')
        
        os.makedirs(backup_dir, exist_ok=True)
    
    def load_current_metrics(self):
        """load metrics from current model."""
        metrics = {}
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.strip().split(':', 1)
                        try:
                            metrics[key.strip().lower()] = float(val.strip().split()[0])
                        except:
                            pass
        return metrics
    
    def backup_current_model(self):
        """backup current model before retraining."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = os.path.join(self.backup_dir, timestamp)
        os.makedirs(backup_subdir, exist_ok=True)
        
        files_to_backup = [
            self.model_path,
            self.features_path,
            self.metrics_path
        ]
        
        for src in files_to_backup:
            if os.path.exists(src):
                dst = os.path.join(backup_subdir, os.path.basename(src))
                shutil.copy2(src, dst)
        
        print(f"backup saved to {backup_subdir}")
        return backup_subdir
    
    def prepare_data(self, data_path='docs'):
        """load and prepare training data."""
        print("loading data...")
        df_trans, df_behavior = load_data(data_path)
        df = clean_and_merge(df_trans, df_behavior)
        df = add_derived_features(df)
        
        # use all numeric + categorical features (exclude target and IDs)
        exclude_cols = [TARGET_COLUMN, 'cst_dim_id', 'transdate', 'trans_id']
        available_features = [c for c in df.columns if c not in exclude_cols]
        
        X = df[available_features].copy()
        y = df[TARGET_COLUMN].copy()
        
        # handle categoricals
        cat_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]
        for col in cat_features:
            X[col] = X[col].astype(str).fillna('unknown')
        
        # fill numeric
        for col in X.columns:
            if col not in cat_features:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, y, available_features, cat_features
    
    def train_new_model(self, X, y, feature_names, cat_features, test_size=0.2):
        """train new model and evaluate."""
        print(f"training on {len(X)} samples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        cat_indices = [feature_names.index(f) for f in cat_features if f in feature_names]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        test_pool = Pool(X_test, y_test, cat_features=cat_indices)
        
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=test_pool, verbose=100)
        
        # evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'f2': fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            'samples': len(X),
            'fraud_rate': y.mean()
        }
        
        print(f"new model: roc_auc={metrics['roc_auc']:.4f}, recall={metrics['recall']:.4f}")
        
        return model, metrics, feature_names
    
    def compare_models(self, old_metrics, new_metrics):
        """compare old and new model metrics."""
        print("\nmodel comparison:")
        print("-" * 50)
        
        # key metrics to compare (higher is better)
        key_metrics = ['roc_auc', 'recall', 'f1', 'precision']
        
        improvements = {}
        for metric in key_metrics:
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            diff = new_val - old_val
            pct = (diff / old_val * 100) if old_val > 0 else 0
            improvements[metric] = {'old': old_val, 'new': new_val, 'diff': diff, 'pct': pct}
            
            status = '+' if diff >= 0 else ''
            print(f"  {metric}: {old_val:.4f} -> {new_val:.4f} ({status}{pct:.1f}%)")
        
        print("-" * 50)
        
        # decision: new model is better if recall improved or stayed same AND roc_auc improved
        recall_ok = improvements['recall']['diff'] >= -0.02  # allow 2% drop
        auc_ok = improvements['roc_auc']['diff'] >= -0.01  # allow 1% drop
        any_improvement = any(improvements[m]['diff'] > 0.01 for m in key_metrics)
        
        is_better = recall_ok and auc_ok and any_improvement
        
        return is_better, improvements
    
    def save_new_model(self, model, metrics, feature_names):
        """save new model and metrics."""
        model.save_model(self.model_path)
        
        with open(self.features_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"retrained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"samples: {metrics['samples']}\n")
            f.write(f"fraud_rate: {metrics['fraud_rate']:.4f}\n")
            f.write(f"roc_auc: {metrics['roc_auc']:.4f}\n")
            f.write(f"precision: {metrics['precision']:.4f}\n")
            f.write(f"recall: {metrics['recall']:.4f}\n")
            f.write(f"f1: {metrics['f1']:.4f}\n")
            f.write(f"f2: {metrics['f2']:.4f}\n")
        
        print(f"model saved to {self.model_path}")
    
    def rollback(self, backup_path):
        """restore model from backup."""
        for filename in os.listdir(backup_path):
            src = os.path.join(backup_path, filename)
            dst = os.path.join(self.models_dir, filename)
            shutil.copy2(src, dst)
        print(f"rolled back to {backup_path}")
    
    def retrain(self, data_path='docs', force=False, min_improvement=0.0):
        """
        full retraining pipeline.
        
        args:
            data_path: path to data directory
            force: if True, save new model even if worse
            min_improvement: minimum improvement threshold
        
        returns:
            dict with results
        """
        print("=" * 60)
        print("MODEL RETRAINING")
        print("=" * 60)
        
        # 1. backup current model
        backup_path = self.backup_current_model()
        old_metrics = self.load_current_metrics()
        
        # 2. prepare data
        X, y, features, cat_features = self.prepare_data(data_path)
        
        # 3. train new model
        model, new_metrics, feature_names = self.train_new_model(
            X, y, features, cat_features
        )
        
        # 4. compare
        is_better, improvements = self.compare_models(old_metrics, new_metrics)
        
        # 5. decide
        if is_better or force:
            self.save_new_model(model, new_metrics, feature_names)
            decision = "new model saved"
        else:
            decision = "kept old model (new model not better)"
        
        print(f"\ndecision: {decision}")
        print("=" * 60)
        
        return {
            'backup_path': backup_path,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'is_better': is_better,
            'improvements': improvements,
            'decision': decision
        }


def main():
    """run retraining."""
    import argparse
    
    parser = argparse.ArgumentParser(description='retrain fraud detection model')
    parser.add_argument('--data', default='docs', help='path to data directory')
    parser.add_argument('--force', action='store_true', help='save model even if worse')
    args = parser.parse_args()
    
    retrainer = ModelRetrainer()
    result = retrainer.retrain(data_path=args.data, force=args.force)
    
    # summary
    print("\nsummary:")
    for metric, vals in result['improvements'].items():
        print(f"  {metric}: {vals['old']:.4f} -> {vals['new']:.4f}")


if __name__ == '__main__':
    main()
