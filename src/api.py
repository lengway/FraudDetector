import os
import sys
import sqlite3
import pickle
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostClassifier

# add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ML_PREDICTION_THRESHOLD


# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_fraud_model.cbm')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')


# Response Models
class FraudTransaction(BaseModel):
    trans_id: int
    cst_dim_id: str
    amount: float
    transdate: str
    direction: str
    probability: float
    risk_level: str
    checked_at: str


class CheckResult(BaseModel):
    checked_count: int
    fraud_count: int
    frauds: List[FraudTransaction]


# Global state
model = None
feature_names = None
last_check_time = None
total_checked = 0
total_fraud_found = 0


def get_db():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def load_model():
    """Load CatBoost model and feature names."""
    global model, feature_names
    
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"model loaded: {len(feature_names)} features")


def predict_fraud(df: pd.DataFrame) -> tuple:
    """Predict fraud probabilities for transactions."""
    
    # ensure all features exist
    X = df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    
    X = X[feature_names].copy()
    
    # handle categoricals
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).fillna('unknown')
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # predict
    probas = model.predict_proba(X)[:, 1]
    is_fraud = probas >= ML_PREDICTION_THRESHOLD
    
    return probas, is_fraud


def get_risk_level(prob: float) -> str:
    """Get risk level from probability."""
    if prob >= 0.9:
        return "CRITICAL"
    elif prob >= 0.7:
        return "HIGH"
    elif prob >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def check_new_transactions(batch_size: int = 100) -> CheckResult:
    """Check unchecked transactions from database."""
    global total_checked, total_fraud_found
    
    conn = get_db()
    
    # get unchecked transactions
    query = f"SELECT * FROM transactions WHERE checked = 0 LIMIT {batch_size}"
    df = pd.read_sql_query(query, conn)
    
    if len(df) == 0:
        conn.close()
        return CheckResult(checked_count=0, fraud_count=0, frauds=[])
    
    # predict
    probas, is_fraud = predict_fraud(df)
    
    # update database
    now = datetime.now().isoformat()
    frauds = []
    
    for idx, row in df.iterrows():
        trans_id = int(row['trans_id'])
        prob = float(probas[idx])
        fraud = bool(is_fraud[idx])
        
        conn.execute(
            """UPDATE transactions 
               SET checked = 1, is_fraud_detected = ?, fraud_probability = ?, checked_at = ?
               WHERE trans_id = ?""",
            (1 if fraud else 0, prob, now, trans_id)
        )
        
        if fraud:
            frauds.append(FraudTransaction(
                trans_id=trans_id,
                cst_dim_id=str(row.get('cst_dim_id', '')),
                amount=float(row.get('amount', 0)),
                transdate=str(row.get('transdate', '')),
                direction=str(row.get('direction', 'unknown')),
                probability=prob,
                risk_level=get_risk_level(prob),
                checked_at=now
            ))
    
    conn.commit()
    conn.close()
    
    total_checked += len(df)
    total_fraud_found += len(frauds)
    
    return CheckResult(
        checked_count=len(df),
        fraud_count=len(frauds),
        frauds=frauds
    )


def check_all_unchecked():
    """Check ALL unchecked transactions at once."""
    global last_check_time, total_checked, total_fraud_found
    
    conn = get_db()
    cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE checked = 0")
    unchecked_count = cursor.fetchone()[0]
    conn.close()
    
    if unchecked_count == 0:
        return
    
    print(f"checking all {unchecked_count} unchecked transactions...")
    
    total_frauds = 0
    batch_size = 1000
    
    while True:
        result = check_new_transactions(batch_size=batch_size)
        if result.checked_count == 0:
            break
        total_frauds += result.fraud_count
    
    last_check_time = datetime.now().isoformat()
    print(f"[{last_check_time}] initial check complete: {total_checked} checked, {total_fraud_found} frauds found")


async def periodic_fraud_check():
    """Background task: check new transactions every 60 seconds."""
    global last_check_time
    
    # initial check - process all existing transactions
    check_all_unchecked()
    
    while True:
        # wait 60 seconds first
        await asyncio.sleep(60)
        
        try:
            last_check_time = datetime.now().isoformat()
            result = check_new_transactions(batch_size=1000)
            
            if result.fraud_count > 0:
                print(f"[{last_check_time}] checked {result.checked_count}, found {result.fraud_count} new frauds")
            elif result.checked_count > 0:
                print(f"[{last_check_time}] checked {result.checked_count}, no new fraud")
            else:
                print(f"[{last_check_time}] no new transactions")
                
        except Exception as e:
            print(f"error in fraud check: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    load_model()
    
    # check if database exists
    if not os.path.exists(DB_PATH):
        print(f"WARNING: database not found at {DB_PATH}")
        print("run 'python src/init_db.py' to create it")
    else:
        conn = get_db()
        cursor = conn.execute("SELECT COUNT(*) FROM transactions")
        total = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE checked = 0")
        unchecked = cursor.fetchone()[0]
        conn.close()
        print(f"database loaded: {total} transactions, {unchecked} unchecked")
    
    # start background task
    asyncio.create_task(periodic_fraud_check())
    print("fraud detection service started (checking every 60s)")
    yield
    print("fraud detection service stopped")


# FastAPI app
app = FastAPI(
    title="FraudDetector API",
    description="Real-time fraud detection for mobile banking transactions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Service status."""
    return {
        "service": "FraudDetector",
        "status": "running",
        "last_check": last_check_time,
        "total_checked": total_checked,
        "total_fraud_found": total_fraud_found,
        "threshold": ML_PREDICTION_THRESHOLD
    }


@app.get("/frauds", response_model=List[FraudTransaction])
async def get_frauds(limit: int = Query(100, description="Max results")):
    """Get all detected fraud transactions from database."""
    conn = get_db()
    query = f"""
        SELECT trans_id, cst_dim_id, amount, transdate, direction, 
               fraud_probability, checked_at
        FROM transactions 
        WHERE is_fraud_detected = 1 
        ORDER BY fraud_probability DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    results = []
    for _, row in df.iterrows():
        results.append(FraudTransaction(
            trans_id=int(row['trans_id']),
            cst_dim_id=str(row['cst_dim_id']),
            amount=float(row['amount']),
            transdate=str(row['transdate']),
            direction=str(row['direction']),
            probability=float(row['fraud_probability']),
            risk_level=get_risk_level(float(row['fraud_probability'])),
            checked_at=str(row['checked_at'])
        ))
    
    return results


@app.post("/check_now", response_model=CheckResult)
async def check_now(batch_size: int = Query(500, description="Batch size")):
    """Force immediate check of unchecked transactions."""
    global last_check_time
    last_check_time = datetime.now().isoformat()
    result = check_new_transactions(batch_size=batch_size)
    return result


@app.post("/reset")
async def reset_database():
    """Reset all transactions to unchecked state (for demo)."""
    global total_checked, total_fraud_found
    
    conn = get_db()
    conn.execute("""
        UPDATE transactions 
        SET checked = 0, is_fraud_detected = NULL, 
            fraud_probability = NULL, checked_at = NULL
    """)
    conn.commit()
    conn.close()
    
    total_checked = 0
    total_fraud_found = 0
    
    return {"status": "reset", "message": "all transactions reset to unchecked"}


@app.get("/stats")
async def get_stats():
    """Get database and detection statistics."""
    conn = get_db()
    
    cursor = conn.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE checked = 1")
    checked = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud_detected = 1")
    fraud_detected = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE target = 1")
    actual_fraud = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_transactions": total,
        "checked": checked,
        "unchecked": total - checked,
        "fraud_detected": fraud_detected,
        "actual_fraud_in_data": actual_fraud,
        "last_check": last_check_time,
        "threshold": ML_PREDICTION_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
