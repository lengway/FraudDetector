"""
Two-Stage Fraud Detection Pipeline
===================================

Stage 1: Scorecard (Rule-based fast filter)
Stage 2: ML Model (Deep analysis for suspicious cases)

Архитектура:
    Transaction → Scorecard → Low risk? → APPROVE
                           → High risk? → ML Model → FRAUD/NOT_FRAUD
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import pickle
from catboost import CatBoostClassifier
import config


class ScorecardFilter:
    """Stage 1: Fast rule-based filter using scorecard logic."""
    
    def __init__(self, threshold_low: int = 3, threshold_high: int = 5):
        """
        Args:
            threshold_low: Скор <= этого значения → AUTO APPROVE
            threshold_high: Скор >= этого значения → SEND TO ML MODEL
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
    def calculate_scorecard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление scorecard баллов для каждой транзакции.
        
        Правила (из анализа total_fraud.ipynb):
        DEVICE/OS RULES:
        - rare_os_flag = 1            → +2 балла
        - rare_device_flag = 1        → +2 балла
        - suspicious_device_combo = 1 → +2 балла
        - high_device_volatility = 1  → +1 балл
        - high_login_volatility = 1   → +1 балл
        
        LOGIN FREQUENCY RULES (NEW - based on data analysis):
        - freq_change_suspicious = 1  → +2 балла (резкий рост частоты логинов)
        - large_login_interval = 1    → +2 балла (большой интервал между логинами)
        - low_login_activity = 1      → +1 балл  (мало логинов за 7 дней)
        - high_login_ratio = 1        → +1 балл  (высокая доля 7d/30d)
        """
        df = df.copy()
        
        # =====================================================================
        # DEVICE/OS RULES (existing)
        # =====================================================================
        
        # 1. Rare OS (< 1% транзакций)
        if 'last_os_ver' in df.columns:
            os_counts = df['last_os_ver'].value_counts(normalize=True)
            df['rare_os_flag'] = df['last_os_ver'].map(
                lambda x: 1 if os_counts.get(x, 0) < 0.01 else 0
            )
        else:
            df['rare_os_flag'] = 0
        
        # 2. Rare Device (< 1% транзакций)
        if 'last_phone_model' in df.columns:
            device_counts = df['last_phone_model'].value_counts(normalize=True)
            df['rare_device_flag'] = df['last_phone_model'].map(
                lambda x: 1 if device_counts.get(x, 0) < 0.01 else 0
            )
        else:
            df['rare_device_flag'] = 0
        
        # 3. High Device Volatility (частая смена device/OS)
        volatility_features = ['os_count_30d', 'device_count_30d']
        if all(f in df.columns for f in volatility_features):
            volatility_threshold = df[volatility_features].mean(axis=1).quantile(0.75)
            df['high_device_volatility'] = (
                df[volatility_features].mean(axis=1) > volatility_threshold
            ).astype(int)
        else:
            df['high_device_volatility'] = 0
        
        # 4. Suspicious Device Combo
        df['suspicious_device_combo'] = df['rare_device_flag'] * df['high_device_volatility']
        
        # 5. High Login Volatility (burstiness)
        if 'login_volatility_factor' in df.columns:
            login_vol_threshold = df['login_volatility_factor'].quantile(0.80)
            df['high_login_volatility'] = (
                df['login_volatility_factor'] > login_vol_threshold
            ).astype(int)
        else:
            df['high_login_volatility'] = 0
        
        # =====================================================================
        # LOGIN FREQUENCY RULES (NEW - based on data analysis)
        # =====================================================================
        
        # 6. Резкий рост частоты логинов (fraud mean +98% vs non-fraud)
        # rel_freq_change_7_30d > 1.0 ловит 23.6% fraud при 14% FP
        if 'rel_freq_change_7_30d' in df.columns:
            df['rel_freq_change_7_30d'] = pd.to_numeric(df['rel_freq_change_7_30d'], errors='coerce').fillna(0)
            df['freq_change_suspicious'] = (df['rel_freq_change_7_30d'] > 1.0).astype(int)
        else:
            df['freq_change_suspicious'] = 0
        
        # 7. Большой интервал между логинами (fraud mean +57% vs non-fraud)
        # avg_login_interval > 200000 ловит 13.9% fraud при 8.7% FP
        if 'avg_login_interval' in df.columns:
            df['avg_login_interval'] = pd.to_numeric(df['avg_login_interval'], errors='coerce').fillna(0)
            df['large_login_interval'] = (df['avg_login_interval'] > 200000).astype(int)
        else:
            df['large_login_interval'] = 0
        
        # 8. Мало логинов за 7 дней (fraud mean -13% vs non-fraud)
        # logins_7d < 3 ловит 20.6% fraud при 15.8% FP
        if 'logins_7d' in df.columns:
            df['logins_7d'] = pd.to_numeric(df['logins_7d'], errors='coerce').fillna(0)
            df['low_login_activity'] = (df['logins_7d'] < 3).astype(int)
        else:
            df['low_login_activity'] = 0
        
        # 9. Высокая доля логинов 7d/30d (fraud mean +19% vs non-fraud)
        # login_share_7_30d > 0.5 — активность сконцентрирована недавно
        if 'login_share_7_30d' in df.columns:
            df['login_share_7_30d'] = pd.to_numeric(df['login_share_7_30d'], errors='coerce').fillna(0)
            df['high_login_ratio'] = (df['login_share_7_30d'] > 0.5).astype(int)
        else:
            df['high_login_ratio'] = 0
        
        # =====================================================================
        # AMOUNT & TRANSACTION RULES (NEW - catches remaining 46 fraudsters)
        # =====================================================================
        
        # 10. Большая сумма транзакции (снизили порог: 75k ловит 28% при 15% FP)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df['high_amount_flag'] = (df['amount'] > 75000).astype(int)
        else:
            df['high_amount_flag'] = 0
        
        # 11. Fast bot (ловит 28.3% пропущенных при всего 3.1% FP!)
        if 'is_fast_bot' in df.columns:
            df['is_fast_bot'] = pd.to_numeric(df['is_fast_bot'], errors='coerce').fillna(0)
            df['fast_bot_flag'] = (df['is_fast_bot'] == 1).astype(int)
        else:
            df['fast_bot_flag'] = 0
        
        # 12. Сумма выше средней по клиенту (снизили порог: 1.8x ловит 19% при 14% FP)
        if 'amount_to_avg_ratio' in df.columns:
            df['amount_to_avg_ratio'] = pd.to_numeric(df['amount_to_avg_ratio'], errors='coerce').fillna(0)
            df['unusual_amount_flag'] = (df['amount_to_avg_ratio'] > 1.8).astype(int)
        else:
            df['unusual_amount_flag'] = 0
        
        # 13. Очень низкая активность недавно (ловит 58.7% пропущенных)
        if 'login_share_7_30d' in df.columns:
            df['very_low_recent_activity'] = (df['login_share_7_30d'] < 0.15).astype(int)
        else:
            df['very_low_recent_activity'] = 0
        
        # =====================================================================
        # NEW RULES FOR BETTER RECALL (catches scorecard-missed fraud)
        # =====================================================================
        
        # 14. Ночная транзакция (0-6 часов) - повышенный риск
        if 'is_night_transaction' in df.columns:
            df['night_tx_flag'] = pd.to_numeric(df['is_night_transaction'], errors='coerce').fillna(0).astype(int)
        elif 'hour' in df.columns:
            hour = pd.to_numeric(df['hour'], errors='coerce').fillna(12)
            df['night_tx_flag'] = ((hour >= 0) & (hour <= 6)).astype(int)
        else:
            df['night_tx_flag'] = 0
        
        # 15. Device hopper - частая смена устройств
        if 'is_device_hopper' in df.columns:
            df['device_hopper_flag'] = pd.to_numeric(df['is_device_hopper'], errors='coerce').fillna(0).astype(int)
        elif 'device_count_30d' in df.columns:
            device_count = pd.to_numeric(df['device_count_30d'], errors='coerce').fillna(1)
            df['device_hopper_flag'] = (device_count > 1).astype(int)
        else:
            df['device_hopper_flag'] = 0
        
        # 16. Login burst - резкий всплеск активности
        if 'is_login_burst' in df.columns:
            df['login_burst_flag'] = pd.to_numeric(df['is_login_burst'], errors='coerce').fillna(0).astype(int)
        else:
            df['login_burst_flag'] = 0
        
        # 17. Высокая сумма (90-й перцентиль)
        if 'is_high_amount' in df.columns:
            df['high_amount_p90_flag'] = pd.to_numeric(df['is_high_amount'], errors='coerce').fillna(0).astype(int)
        else:
            df['high_amount_p90_flag'] = 0
        
        # 18. Комбо: ночь + высокая сумма (очень подозрительно!)
        df['night_high_amount_combo'] = (df['night_tx_flag'] * df.get('high_amount_flag', 0)).astype(int)
        
        # =====================================================================
        # TOTAL SCORECARD SCORE (UPDATED)
        # =====================================================================
        df['scorecard_total'] = (
            # Device/OS rules
            df['rare_os_flag'] * 2 +
            df['rare_device_flag'] * 2 +
            df['suspicious_device_combo'] * 2 +
            df['high_device_volatility'] * 1 +
            df['high_login_volatility'] * 1 +
            # Login frequency rules
            df['freq_change_suspicious'] * 2 +
            df['large_login_interval'] * 2 +
            df['low_login_activity'] * 1 +
            df['high_login_ratio'] * 1 +
            # Amount & transaction rules
            df['high_amount_flag'] * 2 +
            df['fast_bot_flag'] * 3 +  # Высокий вес - низкий FP!
            df['unusual_amount_flag'] * 2 +
            df['very_low_recent_activity'] * 1 +
            # NEW rules for better recall
            df['night_tx_flag'] * 1 +
            df['device_hopper_flag'] * 1 +
            df['login_burst_flag'] * 2 +
            df['high_amount_p90_flag'] * 1 +
            df['night_high_amount_combo'] * 2  # Комбо-правило
        )
        
        return df
    
    def filter_transactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Разделяет транзакции на 3 категории.
        
        Returns:
            (auto_approve, needs_ml_check, scorecard_results)
        """
        df_scored = self.calculate_scorecard(df)
        
        # Категоризация
        auto_approve = df_scored[df_scored['scorecard_total'] <= self.threshold_low].copy()
        needs_ml_check = df_scored[df_scored['scorecard_total'] > self.threshold_low].copy()
        
        # Статистика
        stats = {
            'total': len(df_scored),
            'auto_approve': len(auto_approve),
            'needs_ml_check': len(needs_ml_check),
            'approve_rate': len(auto_approve) / len(df_scored) * 100,
            'ml_check_rate': len(needs_ml_check) / len(df_scored) * 100
        }
        
        print(f"\n📊 SCORECARD FILTER RESULTS:")
        print(f"   Total transactions: {stats['total']}")
        print(f"   ✅ Auto-approved: {stats['auto_approve']} ({stats['approve_rate']:.1f}%)")
        print(f"   🔍 Needs ML check: {stats['needs_ml_check']} ({stats['ml_check_rate']:.1f}%)")
        
        return auto_approve, needs_ml_check, df_scored


class MLModelDetector:
    """Stage 2: Deep ML-based fraud detection for suspicious cases."""
    
    def __init__(self, model_path: str = 'models/catboost_fraud_model.cbm',
                 feature_names_path: str = 'models/feature_names.pkl'):
        """Загрузка обученной CatBoost модели."""
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✅ ML Model loaded: {model_path}")
        print(f"   Features: {len(self.feature_names)}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предсказание фрода для подозрительных транзакций.
        
        Returns:
            DataFrame с колонками: fraud_probability, fraud_prediction, risk_level
        """
        df = df.copy()
        
        # Проверка наличия всех нужных фичей
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            print(f"⚠️ Warning: Missing features: {missing_features}")
            for feat in missing_features:
                df[feat] = 0
        
        # Предсказание
        X = df[self.feature_names]
        df['fraud_probability'] = self.model.predict_proba(X)[:, 1]
        
        # Используем пороги из config
        df['fraud_prediction'] = (df['fraud_probability'] > 0.5).astype(int)
        
        # Risk levels
        df['risk_level'] = pd.cut(
            df['fraud_probability'],
            bins=[0, config.THRESHOLDS['low'], config.THRESHOLDS['medium'], 
                  config.THRESHOLDS['high'], 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        return df


class TwoStageDetector:
    """Основной класс двухэтапной системы детекции."""
    
    def __init__(self, scorecard_threshold_low: int = 2,
                 model_path: str = 'models/catboost_fraud_model.cbm'):
        """
        Args:
            scorecard_threshold_low: Скор <= этого → авто-одобрение
            model_path: Путь к обученной ML модели
        """
        self.scorecard = ScorecardFilter(threshold_low=scorecard_threshold_low)
        self.ml_model = MLModelDetector(model_path=model_path)
        
    def detect_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Полный пайплайн детекции мошенничества.
        
        Returns:
            DataFrame со всеми результатами и финальным решением
        """
        print("\n" + "="*60)
        print("🚀 TWO-STAGE FRAUD DETECTION PIPELINE")
        print("="*60)
        
        # STAGE 1: Scorecard фильтр
        print("\n📋 STAGE 1: Scorecard Filter...")
        auto_approve, needs_ml, df_scored = self.scorecard.filter_transactions(df)
        
        # Для авто-одобренных: fraud_probability = 0
        auto_approve['fraud_probability'] = 0.0
        auto_approve['fraud_prediction'] = 0
        auto_approve['risk_level'] = 'LOW'
        auto_approve['detection_stage'] = 'scorecard'
        
        if len(needs_ml) == 0:
            print("\n✅ All transactions auto-approved by scorecard!")
            return auto_approve
        
        # STAGE 2: ML Model для подозрительных
        print(f"\n🤖 STAGE 2: ML Model Analysis ({len(needs_ml)} transactions)...")
        needs_ml_analyzed = self.ml_model.predict(needs_ml)
        needs_ml_analyzed['detection_stage'] = 'ml_model'
        
        # Детальная статистика по ML-проверенным
        print(f"\n🔍 ML CHECK DETAILS:")
        print(f"   Analyzed: {len(needs_ml_analyzed)} suspicious transactions")
        if 'target' in needs_ml_analyzed.columns:
            # Если есть истинные метки (для анализа)
            actual_fraud = needs_ml_analyzed['target'].sum()
            detected_fraud = needs_ml_analyzed['fraud_prediction'].sum()
            print(f"   Actual fraud (target=1): {actual_fraud}")
            print(f"   Predicted fraud: {detected_fraud}")
            
            # Confusion matrix для ML-проверенных
            true_positives = ((needs_ml_analyzed['target'] == 1) & (needs_ml_analyzed['fraud_prediction'] == 1)).sum()
            false_positives = ((needs_ml_analyzed['target'] == 0) & (needs_ml_analyzed['fraud_prediction'] == 1)).sum()
            false_negatives = ((needs_ml_analyzed['target'] == 1) & (needs_ml_analyzed['fraud_prediction'] == 0)).sum()
            true_negatives = ((needs_ml_analyzed['target'] == 0) & (needs_ml_analyzed['fraud_prediction'] == 0)).sum()
            
            print(f"\n   Confusion Matrix (ML-checked only):")
            print(f"   TP: {true_positives} | FP: {false_positives}")
            print(f"   FN: {false_negatives} | TN: {true_negatives}")
        else:
            detected_fraud = needs_ml_analyzed['fraud_prediction'].sum()
            print(f"   Predicted fraud: {detected_fraud}")
        
        print(f"\n   Scorecard scores distribution (ML-checked):")
        print(needs_ml_analyzed['scorecard_total'].value_counts().sort_index().to_string())
        
        # Объединение результатов
        final_results = pd.concat([auto_approve, needs_ml_analyzed], ignore_index=True)
        
        # Финальная статистика
        fraud_count = final_results['fraud_prediction'].sum()
        fraud_rate = fraud_count / len(final_results) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Total transactions: {len(final_results)}")
        print(f"   Fraud detected: {fraud_count} ({fraud_rate:.2f}%)")
        print(f"   Risk breakdown:")
        print(final_results['risk_level'].value_counts().to_string())
        print("="*60)
        
        # Сохраняем отдельно ML-проверенные транзакции для анализа
        self.ml_checked_transactions = needs_ml_analyzed
        
        return final_results


if __name__ == '__main__':
    # Пример использования
    from preprocessing import load_data, clean_and_merge, preprocess
    from train_catboost import engineer_features
    
    print("Loading data...")
    df_trans, df_behavior = load_data()
    df = clean_and_merge(df_trans, df_behavior)
    
    print("Preprocessing (adding derived features)...")
    df = preprocess(df)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Двухэтапная детекция с более строгим порогом для recall
    # Скор ≤1 = auto-approve (только самые безопасные)
    detector = TwoStageDetector(scorecard_threshold_low=1)
    results = detector.detect_fraud(df)
    
    # Сохранение результатов
    results.to_csv('docs/two_stage_detection_results.csv', index=False)
    print("\n✅ All results saved to 'docs/two_stage_detection_results.csv'")
    
    # Сохранение ML-проверенных транзакций отдельно для анализа
    if hasattr(detector, 'ml_checked_transactions'):
        ml_checked = detector.ml_checked_transactions
        ml_checked.to_csv('docs/ml_checked_transactions.csv', index=False)
        print(f"✅ ML-checked transactions saved to 'docs/ml_checked_transactions.csv' ({len(ml_checked)} rows)")
