import os
import sqlite3
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, clean_and_merge


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.db')


def engineer_features(df):
    """Add all features required by the model."""
    print("engineering features...")
    
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

    # Composite high-risk flag
    if 'amount' in df.columns and 'std_login_interval_30d' in df.columns:
        df['is_high_risk_combo'] = ((df['amount'] > 10000.0) & (df['std_login_interval_30d'] > 100000.0)).astype(int)

    # Behavioral flags
    if 'monthly_phone_model_changes' in df.columns:
        df['is_device_hopper'] = (df['monthly_phone_model_changes'] > 1).astype(int)
    if 'avg_login_interval_30d' in df.columns:
        df['is_fast_bot'] = (df['avg_login_interval_30d'] < 10).astype(int)

    # Login velocity
    if 'logins_last_7_days' in df.columns and 'logins_last_30_days' in df.columns:
        logins_7d = pd.to_numeric(df['logins_last_7_days'], errors='coerce').fillna(0)
        logins_30d = pd.to_numeric(df['logins_last_30_days'], errors='coerce').fillna(0)
        df['login_velocity'] = logins_7d / (logins_30d + 1e-6)
    
    # Device change rate
    if 'monthly_phone_model_changes' in df.columns and 'logins_last_30_days' in df.columns:
        device_count = pd.to_numeric(df['monthly_phone_model_changes'], errors='coerce').fillna(0)
        logins_30d = pd.to_numeric(df['logins_last_30_days'], errors='coerce').fillna(0)
        df['device_change_rate'] = device_count / (logins_30d + 1)
    
    # Time since last login
    if 'hour' in df.columns:
        df['time_since_last_login'] = (24 - df['hour']).clip(lower=0)

    # User-level aggregates
    if 'cst_dim_id' in df.columns and 'amount' in df.columns:
        user_amt_agg = df.groupby('cst_dim_id').agg({
            'amount': ['mean', 'std', 'count'],
        }).reset_index()
        user_amt_agg.columns = ['cst_dim_id', 'user_avg_amt', 'user_std_amt', 'user_tx_count']
        df = df.merge(user_amt_agg, on='cst_dim_id', how='left')
        df.fillna(0, inplace=True)
        df['amount_to_avg_ratio'] = df['amount'] / df['user_avg_amt'].replace(0, 1e-6)
        df['amount_to_avg_ratio'].replace([np.inf, -np.inf], 99999.0, inplace=True)
        
        # user_hist_fraud: CUMULATIVE fraud count UP TO current transaction (no leakage!)
        if 'transdate' in df.columns:
            df = df.sort_values(['cst_dim_id', 'transdate'])
            df['user_hist_fraud'] = df.groupby('cst_dim_id')['target'].cumsum().shift(1).fillna(0).astype(int)
        else:
            df['user_hist_fraud'] = 0

    print(f"features engineered: {df.shape}")
    return df


def init_database():
    """Create and populate SQLite database with transaction data."""
    
    print("initializing database...")
    
    # ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # load and merge data
    df_trans, df_behavior = load_data('docs')
    df = clean_and_merge(df_trans, df_behavior)
    
    # add all model features
    df = engineer_features(df)
    
    # add columns for fraud detection status
    df['checked'] = 0  # 0 = not checked, 1 = checked
    df['is_fraud_detected'] = None  # ML prediction result
    df['fraud_probability'] = None
    df['checked_at'] = None
    
    # create unique trans_id if not exists
    if 'trans_id' not in df.columns:
        df['trans_id'] = range(1, len(df) + 1)
    
    # connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # save to database
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # create index on checked column for fast queries
    conn.execute('CREATE INDEX IF NOT EXISTS idx_checked ON transactions(checked)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fraud ON transactions(is_fraud_detected)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_trans_id ON transactions(trans_id)')
    
    conn.commit()
    
    # verify
    cursor = conn.execute('SELECT COUNT(*) FROM transactions')
    total = cursor.fetchone()[0]
    
    cursor = conn.execute('SELECT COUNT(*) FROM transactions WHERE target = 1')
    fraud_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"database created: {DB_PATH}")
    print(f"total transactions: {total}")
    print(f"fraud transactions (target=1): {fraud_count}")
    print(f"columns: {len(df.columns)}")
    
    return DB_PATH


def reset_checked_status():
    """Reset all transactions to unchecked state (for demo)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE transactions SET checked = 0, is_fraud_detected = NULL, fraud_probability = NULL, checked_at = NULL')
    conn.commit()
    conn.close()
    print("all transactions reset to unchecked")


if __name__ == "__main__":
    init_database()
