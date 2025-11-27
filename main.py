# run_login_analysis.py
# Запуск: python run_login_analysis.py
import pandas as pd
import numpy as np
import os, json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nbformat as nbf

# --- Настройки путей (изменяй при необходимости) ---
BEH_PATH = "docs/поведенческие паттерны клиентов.csv"
TX_PATH = "docs/транзакции в Мобильном интернет Банкинге.csv"
OUT_MERGED = "docs/login_features_merged.csv"
OUT_CLUSTER = "docs/cluster_summary_k4.csv"
OUT_NOTEBOOK = "notebooks/data_analysis2.ipynb"

# --- Список ожидаемых логин-фичей (как ты прислал) ---
login_features = [
    'logins_last_7_days', 'logins_last_30_days', 'login_frequency_7d', 'login_frequency_30d',
    'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio', 'avg_login_interval_30d',
    'std_login_interval_30d', 'var_login_interval_30d', 'ewm_login_interval_7d',
    'burstiness_login_interval', 'fano_factor_login_interval', 'zscore_avg_login_interval_7d'
]

# --- Вспомогательные функции ---
def try_load(path):
    for enc in ("cp1251","utf-8","latin1","cp866"):
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python', header = 2)
                if df.shape[1] >= 2:
                    print(f"Loaded {path} enc={enc} sep='{sep}' shape={df.shape}")
                    return df
            except Exception:
                pass
    # fallback: raw lines -> one-column DF
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    return pd.DataFrame({0: raw.splitlines()})

def smart_split_if_single(df):
    if df.shape[1] == 1:
        s = df.iloc[:,0].astype(str)
        if s.str.contains(";").sum() > 0:
            split = s.str.split(";", expand=True)
            split.columns = [f"c{i}" for i in range(split.shape[1])]
            return split
    return df

def find_col(cols, candidates):
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    return None

def clean_series(s):
    return s.astype(str).str.strip().str.replace("'", "").str.replace('"','').str.replace("\\","")

# --- Загрузка ---
tx = try_load(TX_PATH)
beh = try_load(BEH_PATH)
tx = smart_split_if_single(tx)
beh = smart_split_if_single(beh)

tx.columns = [str(c).strip() for c in tx.columns]
beh.columns = [str(c).strip() for c in beh.columns]

# --- Определяем id-колонки и label ---
tx_cust = find_col(tx.columns, ["cst_dim_id","cust","client","customer","cst_id"])
beh_cust = find_col(beh.columns, ["cst_dim_id","cust","client","customer","cst_id"])
beh_label = find_col(beh.columns, ["target","is_fraud","label","fraud"])

# если не нашли — берем первый подходящий numeric-ish столбец
if tx_cust is None:
    for c in tx.columns[:6]:
        if tx[c].astype(str).str.contains(r"\d{6,}").any():
            tx_cust = c; break
if beh_cust is None:
    for c in beh.columns[:6]:
        if beh[c].astype(str).str.contains(r"\d{6,}").any():
            beh_cust = c; break

# создаём standardized cust_id
if tx_cust:
    tx['cust_id'] = clean_series(tx[tx_cust])
else:
    tx['cust_id'] = tx.index.astype(str)
if beh_cust:
    beh['cust_id'] = clean_series(beh[beh_cust])
else:
    beh['cust_id'] = beh.index.astype(str)

# парсим label
if beh_label:
    # извлекаем цифру (0/1) если есть
    beh['target'] = beh[beh_label].astype(str).str.extract(r'(\d)')[0].fillna("0").astype(int)
else:
    beh['target'] = 0
    print("Warning: label column not found in behaviour data; 'target' set to 0 for all rows.")

# --- Выбираем присутствующие login-фичи в транзакциях ---
present_login_features = [f for f in login_features if f in tx.columns]

# простая fuzzy попытка сопоставить колонки, если имена немного отличаются
missing = [f for f in login_features if f not in present_login_features]
for f in missing[:]:
    for c in tx.columns:
        lowc = c.lower()
        if all(part in lowc for part in f.split('_') if len(part)>3) or f.split('_')[0] in lowc:
            present_login_features.append(c)
            missing.remove(f)
            break

print("Present login features:", present_login_features)
print("Missing (not found):", missing)

# Берём подтаблицу и приводим к numeric
tx_sub = tx[['cust_id'] + present_login_features].copy()
for c in present_login_features:
    tx_sub[c] = pd.to_numeric(tx_sub[c].astype(str).str.replace(",", "."), errors='coerce')

# Merge по cust_id (left join на транзакции)
merged = tx_sub.merge(beh[['cust_id','target']], on='cust_id', how='left')
merged['target'] = merged['target'].fillna(0).astype(int)

# --- EDA: основные статистики и корреляции ---
summary = {
    'rows': len(merged),
    'unique_customers': merged['cust_id'].nunique(),
    'fraud_count': int(merged['target'].sum()),
    'nonfraud_count': int((merged['target']==0).sum())
}
print("Basic summary:", summary)

# Per-feature group stats
feat_stats = merged.groupby('target')[present_login_features].agg(['count','mean','std','min','25%','50%','75%','max']).T
feat_stats.columns = ['_'.join(map(str,c)) for c in feat_stats.columns]

# Correlation (Spearman)
corr = merged[present_login_features].corr(method='spearman')

# --- Clustering pipeline ---
# Impute + scale
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(merged[present_login_features])
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA for 2D projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xs)

# KMeans k=4 by default
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = km.fit_predict(Xs)
merged['cluster_k4'] = clusters
merged['pca1'] = X_pca[:,0]; merged['pca2'] = X_pca[:,1]

# Cluster profiles
cluster_profiles = merged.groupby('cluster_k4')[present_login_features].mean().T
cluster_counts = merged.groupby('cluster_k4')['cust_id'].nunique()
cluster_target = merged.groupby('cluster_k4')['target'].agg(['sum','count'])
cluster_summary = pd.concat([cluster_profiles, cluster_counts.rename('unique_customers'), cluster_target], axis=1)

# Save merged file and cluster summary
merged.to_csv(OUT_MERGED, index=False)
cluster_summary.to_csv(OUT_CLUSTER)

# --- Create a short notebook data_analysis2.ipynb with a summary cell ---
import os
os.makedirs(os.path.dirname(OUT_NOTEBOOK), exist_ok=True)
os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell("# Data Analysis 2 — Login-feature clustering (auto-generated)\nThis notebook contains merged features and cluster summary."))
cells.append(nbf.v4.new_markdown_cell("## Basic summary\n" + json.dumps(summary, indent=2)))
cells.append(nbf.v4.new_markdown_cell("## Present login features:\n" + json.dumps(present_login_features, indent=2)))
cells.append(nbf.v4.new_code_cell(f"import pandas as pd\npd.read_csv('{OUT_MERGED}').head()"))
cells.append(nbf.v4.new_code_cell(f"pd.read_csv('{OUT_CLUSTER}').head()"))
nb['cells'] = cells
with open(OUT_NOTEBOOK, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Saved merged features to", OUT_MERGED)
print("Saved notebook to", OUT_NOTEBOOK)
print("Cluster summary saved to", OUT_CLUSTER)
