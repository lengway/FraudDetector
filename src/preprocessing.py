import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def clean_columns(df):
    """Remove spaces and BOM markers from column names."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('\ufeff', '')
        .str.replace('"', '')
    )
    return df


def load_data(base_path='docs'):
    """
    Load transaction and behavioral data from CSV files.
    
    Args:
        base_path: Directory containing CSV files (default: 'docs')
    
    Returns:
        tuple: (df_transactions, df_behavioral)
    
    Raises:
        FileNotFoundError: If CSV files are not found
    """
    print("loading data...")
    
    # Try different paths
    trans_paths = [
        os.path.join(base_path, 'транзакции в Мобильном интернет Банкинге.csv'),
        'транзакции в Мобильном интернет Банкинге.csv'
    ]
    
    behav_paths = [
        os.path.join(base_path, 'поведенческие паттерны клиентов.csv'),
        'поведенческие паттерны клиентов.csv'
    ]
    
    # Load transactions
    df_trans = None
    for path in trans_paths:
        if os.path.exists(path):
            try:
                df_trans = pd.read_csv(
                    path,
                    sep=';',
                    encoding='cp1251',
                    header=1,
                    engine='python',
                )
                df_trans = clean_columns(df_trans)
                print(f"loaded transactions from {path} (shape: {df_trans.shape})")
                break
            except Exception as e:
                print(f"failed to load {path}: {e}")
                continue
    
    if df_trans is None:
        raise FileNotFoundError(f"Transaction CSV not found in paths: {trans_paths}")
    
    # Load behavioral patterns
    df_behavior = None
    for path in behav_paths:
        if os.path.exists(path):
            try:
                df_behavior = pd.read_csv(
                    path,
                    sep=';',
                    encoding='cp1251',
                    header=1,  # header=1 gives English column names directly
                    engine='python',
                    on_bad_lines='skip',
                )
                df_behavior = clean_columns(df_behavior)
                print(f"loaded behavioral data from {path} (shape: {df_behavior.shape})")
                break
            except Exception as e:
                print(f"failed to load {path}: {e}")
                continue
    
    if df_behavior is None:
        raise FileNotFoundError(f"Behavioral CSV not found in paths: {behav_paths}")
    
    return df_trans, df_behavior


def clean_and_merge(df_trans, df_behavior):
    """
    Clean column names, normalize data types, and merge datasets.
    
    Args:
        df_trans: Transaction DataFrame
        df_behavior: Behavioral patterns DataFrame
    
    Returns:
        pd.DataFrame: Merged and cleaned dataset
    """
    print("cleaning and merging data...")
    
    # header=1 gives English column names directly - no rename needed!
    
    # Define expected numeric columns (using header=1 English names)
    numeric_cols = [
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
    
    # Force numeric types on known columns
    for df_temp in [df_trans, df_behavior]:
        for col in numeric_cols:
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
    print("merging datasets...")
    df = df_trans.merge(df_behavior, on=['cst_dim_id', 'transdate'], how='left')
    
    # Fill categorical NaNs (using header=1 column names)
    for c in ['last_phone_model_categorical', 'last_os_categorical', 'direction']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')
    
    # Final numeric cleaning
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                .fillna(0)
                .astype(float)
            )
    
    df.fillna(0, inplace=True)
    print(f"dataset ready: {df.shape}, fraud rate: {df['target'].mean()*100:.2f}%")
    
    return df


def add_derived_features(df):
    """
    Add derived features that improve model performance.
    These features were identified as missing in the ML pipeline.
    
    Args:
        df: DataFrame with merged transaction and behavioral data
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("adding derived features...")
    df = df.copy()
    
    # =========================================================================
    # 1. monthly_phone_model_changes - already from behavioral, just verify
    # =========================================================================
    if 'monthly_phone_model_changes' not in df.columns:
        # If missing - compute as count of unique devices
        if 'last_phone_model_categorical' in df.columns and 'cst_dim_id' in df.columns:
            device_counts = df.groupby('cst_dim_id')['last_phone_model_categorical'].transform('nunique')
            df['monthly_phone_model_changes'] = device_counts
        else:
            df['monthly_phone_model_changes'] = 1
    
    # =========================================================================
    # 2. login_volatility_factor - (std - mean) / (std + mean) for intervals
    # =========================================================================
    if 'burstiness_login_interval' not in df.columns:
        if 'std_login_interval_30d' in df.columns and 'avg_login_interval_30d' in df.columns:
            std = pd.to_numeric(df['std_login_interval_30d'], errors='coerce').fillna(0)
            mean = pd.to_numeric(df['avg_login_interval_30d'], errors='coerce').fillna(0)
            # Avoid division by zero
            denominator = std + mean
            df['burstiness_login_interval'] = np.where(
                denominator > 0,
                (std - mean) / denominator,
                0
            )
        else:
            df['burstiness_login_interval'] = 0
    
    # =========================================================================
    # 3. is_device_hopper - frequent device changes (>1 in 30 days)
    # =========================================================================
    if 'is_device_hopper' not in df.columns:
        device_count = pd.to_numeric(df.get('monthly_phone_model_changes', 1), errors='coerce').fillna(1)
        df['is_device_hopper'] = (device_count > 1).astype(int)
    
    # =========================================================================
    # 4. BONUS: Additional features for better recall
    # =========================================================================
    
    # 4a. is_new_device - transaction from rare device
    if 'last_phone_model_categorical' in df.columns:
        device_freq = df['last_phone_model_categorical'].value_counts(normalize=True)
        df['is_rare_device'] = df['last_phone_model_categorical'].map(
            lambda x: 1 if device_freq.get(x, 0) < 0.01 else 0
        )
    
    # 4b. login_burst - sudden activity spike (7d >> 30d average)
    if 'logins_last_7_days' in df.columns and 'logins_last_30_days' in df.columns:
        logins_7d = pd.to_numeric(df['logins_last_7_days'], errors='coerce').fillna(0)
        logins_30d = pd.to_numeric(df['logins_last_30_days'], errors='coerce').fillna(0)
        avg_7d_expected = logins_30d / 4.28  # 30/7 = 4.28
        df['login_burst'] = np.where(
            avg_7d_expected > 0,
            logins_7d / avg_7d_expected,
            0
        )
        df['is_login_burst'] = (df['login_burst'] > 2.0).astype(int)
    
    # 4c. is_high_amount - сумма выше 90-го перцентиля
    if 'amount' in df.columns:
        amount = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        p90 = amount.quantile(0.90)
        df['is_high_amount'] = (amount > p90).astype(int)
    
    # 4d. suspicious_time - транзакция в подозрительное время (ночь 0-6)
    if 'transdatetime' in df.columns:
        df['transdatetime'] = pd.to_datetime(df['transdatetime'].astype(str).str.strip("'"), errors='coerce')
        df['hour'] = df['transdatetime'].dt.hour.fillna(12).astype(int)
        df['is_night_transaction'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
    
    # 4e. device_os_mismatch - несовпадение OS и device (подозрительно)
    if 'last_os_ver' in df.columns and 'last_phone_model' in df.columns:
        # iOS на Android устройстве или наоборот
        is_ios = df['last_os_ver'].astype(str).str.lower().str.contains('ios|iphone', na=False)
        is_android_device = df['last_phone_model'].astype(str).str.lower().str.contains('samsung|xiaomi|huawei|oppo|vivo|realme|poco', na=False)
        df['device_os_mismatch'] = (is_ios & is_android_device).astype(int)
    
    print(f"derived features added. shape: {df.shape}")
    return df


def preprocess(df):
    """
    Apply full preprocessing pipeline including derived features.
    
    Args:
        df: DataFrame to preprocess
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with all features
    """
    # Add derived features
    df = add_derived_features(df)
    
    return df