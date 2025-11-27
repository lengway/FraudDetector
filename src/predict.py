"""
🔮 CatBoost Fraud Detection - Inference Module
ForteBank Hackathon

This module provides real-time fraud prediction using trained CatBoost model
"""

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from typing import Dict, List, Tuple
import warnings
import config  # Import centralized config
warnings.filterwarnings('ignore')


class FraudDetector:
    """Real-time fraud detection using CatBoost"""
    
    def __init__(self, model_path: str = 'models/catboost_fraud_model.cbm',
                 features_path: str = 'models/feature_names.pkl'):
        """
        Initialize fraud detector
        
        Args:
            model_path: Path to saved CatBoost model
            features_path: Path to saved feature names
        """
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and feature names"""
        try:
            # Load CatBoost model
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load feature names
            with open(self.features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Features loaded: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single transaction
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Ensure all features present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Predict
        fraud_proba = self.model.predict_proba(X)[0, 1]
        fraud_pred = self.model.predict(X)[0]
        
        result = {
            'is_fraud': bool(fraud_pred),
            'fraud_probability': float(fraud_proba),
            'fraud_score': float(fraud_proba * 100),
            'risk_level': self._get_risk_level(fraud_proba),
            'recommendation': self._get_recommendation(fraud_proba)
        }
        
        return result
    
    def predict_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions: DataFrame with transaction features
            
        Returns:
            DataFrame with predictions
        """
        # Ensure all features present
        for feat in self.feature_names:
            if feat not in transactions.columns:
                transactions[feat] = 0
        
        # Select features
        X = transactions[self.feature_names]
        
        # Predict
        fraud_proba = self.model.predict_proba(X)[:, 1]
        fraud_pred = self.model.predict(X)
        
        # Add results
        results = transactions.copy()
        results['is_fraud'] = fraud_pred
        results['fraud_probability'] = fraud_proba
        results['fraud_score'] = fraud_proba * 100
        results['risk_level'] = results['fraud_probability'].apply(self._get_risk_level)
        results['recommendation'] = results['fraud_probability'].apply(self._get_recommendation)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from model"""
        importance = self.model.get_feature_importance()
        
        fi_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return fi_df
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Determine risk level based on fraud probability (using config thresholds)"""
        if probability < config.THRESHOLDS['low']:
            return "LOW"
        elif probability < config.THRESHOLDS['medium']:
            return "MEDIUM"
        elif probability < config.THRESHOLDS['high']:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @staticmethod
    def _get_recommendation(probability: float) -> str:
        """Get action recommendation based on fraud probability (using config thresholds)"""
        if probability < config.THRESHOLDS['low']:
            return "APPROVE"
        elif probability < config.THRESHOLDS['medium']:
            return "REVIEW"
        elif probability < config.THRESHOLDS['high']:
            return "ADDITIONAL_VERIFICATION"
        else:
            return "BLOCK"


def example_usage():
    """Example of how to use FraudDetector"""
    
    # Initialize detector
    detector = FraudDetector()
    
    # Example transaction
    transaction = {
        'amount': 15000,
        'amount_log': np.log1p(15000),
        'amount_vs_user_avg': 2.5,
        'amount_percentile': 0.9,
        'hour': 23,
        'day_of_week': 5,
        'day_of_month': 15,
        'month': 11,
        'is_weekend': 1,
        'is_night': 1,
        'is_business_hours': 0,
        'user_avg_amount': 5000,
        'user_std_amount': 2000,
        'user_tx_count': 50,
        'user_fraud_count': 0,
        'monthly_os_changes': 3,
        'monthly_phone_model_changes': 2,
        'logins_last_7_days': 15,
        'logins_last_30_days': 60,
        'login_frequency_7d': 2.14,
        'login_frequency_30d': 2.0,
        'freq_change_7d_vs_mean': 0.07,
        'logins_7d_over_30d_ratio': 0.25,
        'avg_login_interval_30d': 43200,
        'std_login_interval_30d': 10800,
        'burstiness_login_interval': 0.5,
        'fano_factor_login_interval': 0.25,
        'zscore_avg_login_interval_7d': 1.5
    }
    
    # Predict
    result = detector.predict_single(transaction)
    
    print("\n" + "=" * 60)
    print("🔮 FRAUD DETECTION RESULT")
    print("=" * 60)
    print(f"Is Fraud:           {result['is_fraud']}")
    print(f"Fraud Probability:  {result['fraud_probability']:.4f}")
    print(f"Fraud Score:        {result['fraud_score']:.2f}%")
    print(f"Risk Level:         {result['risk_level']}")
    print(f"Recommendation:     {result['recommendation']}")
    print("=" * 60)
    
    # Feature importance
    print("\n📈 Top 10 Important Features:")
    fi = detector.get_feature_importance()
    print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    example_usage()
