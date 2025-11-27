"""
Data Loading and Preprocessing Module
Centralized data loading to avoid code duplication
"""

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
    print("📊 Loading data...")
    
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
                print(f"✓ Loaded transactions from {path} (shape: {df_trans.shape})")
                break
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
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
                    header=0,
                    engine='python',
                    on_bad_lines='skip',
                )
                df_behavior = clean_columns(df_behavior)
                print(f"✓ Loaded behavioral data from {path} (shape: {df_behavior.shape})")
                break
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
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
    print("🔧 Cleaning and merging data...")
    
    # Rename behavioral columns for consistency
    behav_map = {
        'Уникальный идентификатор клиента': 'cst_dim_id',
        'Дата совершенной транзакции': 'transdate',
        'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент': 'os_count_30d',
        'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент "менял устройство" по логам': 'device_count_30d',
        'Модель телефона из самой последней сессии (по времени) перед transdate': 'last_phone_model',
        'Версия ОС из самой последней сессии перед transdate': 'last_os_ver',
        'Количество уникальных логин-сессий (минутных тайм-слотов) за последние 7 дней до transdate': 'logins_7d',
        'Количество уникальных логин-сессий за последние 30 дней до транзакции': 'logins_30d',
        'Среднее число логинов в день за последние 7 дней: logins_last_7_days / 7': 'avg_logins_7d',
        'Среднее число логинов в день за последние 30 дней: logins_last_30_days / 30': 'avg_logins_30d',
        'Относительное изменение частоты логинов за 7 дней к средней частоте за 30 дней:\n(freq7d?freq30d)/freq30d(freq_{7d} - freq_{30d}) / freq_{30d}(freq7d?freq30d)/freq30d — показывает, стал клиент заходить чаще или реже недавно': 'rel_freq_change_7_30d',
        'Доля логинов за 7 дней от логинов за 30 дней': 'login_share_7_30d',
        'Средний интервал (в секундах) между соседними сессиями за последние 30 дней': 'avg_login_interval',
        'Стандартное отклонение интервалов между логинами за 30 дней (в секундах), измеряет разброс интервалов': 'std_login_interval',
        'Показатель "взрывности" логинов: (std?mean)/(std+mean)(std - mean)/(std + mean)(std?mean)/(std+mean) для интервалов': 'login_volatility_factor',
        'Fano-factor интервалов: variance / mean': 'fano_factor_interval',
        'Z-скор среднего интервала за последние 7 дней относительно среднего за 30 дней: насколько сильно недавние интервалы отличаются от типичных, в единицах стандартного отклонения': 'z_score_avg_interval_7d_vs_30d',
        'Экспоненциально взвешенное среднее интервалов между логинами за 7 дней, где более свежие сессии имеют больший вес (коэффициент затухания 0.3)': 'weighted_avg_interval_7d',
        'Дисперсия интервалов между логинами за 30 дней (в секундах?), ещё одна мера разброса': 'interval_variance_30d',
    }
    df_behavior.rename(columns=behav_map, inplace=True)
    
    # Define expected numeric columns
    numeric_cols = [
        'amount',
        'os_count_30d', 'device_count_30d',
        'logins_7d', 'logins_30d',
        'avg_logins_7d', 'avg_logins_30d',
        'avg_login_interval', 'std_login_interval',
        'rel_freq_change_7_30d',
        'login_share_7_30d',
        'weighted_avg_interval_7d',
        'login_volatility_factor',
        'fano_factor_interval',
        'z_score_avg_interval_7d_vs_30d',
        'interval_variance_30d',
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
    print("🔗 Merging datasets...")
    df = df_trans.merge(df_behavior, on=['cst_dim_id', 'transdate'], how='left')
    
    # Fill categorical NaNs
    for c in ['last_phone_model', 'last_os_ver', 'direction']:
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
    print(f"✓ Dataset ready: {df.shape}, Fraud rate: {df['target'].mean()*100:.2f}%")
    
    return df


def preprocess(df):
    """
    Apply additional preprocessing steps (placeholder for future enhancements).
    
    Args:
        df: DataFrame to preprocess
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Future: scaling, encoding, additional cleaning
    return df