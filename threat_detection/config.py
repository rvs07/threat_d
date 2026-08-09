# config.py

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR       = os.path.join(BASE_DIR, "models")

RAW_LOG_FILE    = os.path.join(DATA_RAW_DIR,  "system_logs.csv")
CLEAN_LOG_FILE  = os.path.join(DATA_PROC_DIR, "cleaned_logs.csv")
FEATURE_FILE    = os.path.join(DATA_PROC_DIR, "features.csv")
MODEL_FILE      = os.path.join(MODEL_DIR,     "isolation_forest.pkl")

# ── Model Hyperparameters ──────────────────────────────────────────────────
CONTAMINATION   = 0.05      # Expected fraction of anomalies (5 %)
RANDOM_STATE    = 42
N_ESTIMATORS    = 100

# ── Threat-Detection Rules ─────────────────────────────────────────────────
MAX_FAILED_LOGINS     = 5   # Brute-force threshold
MAX_REQUESTS_PER_MIN  = 100 # DDoS threshold
SUSPICIOUS_PORTS      = [22, 23, 3389, 4444, 8080]

# ── Alert Settings ─────────────────────────────────────────────────────────
ALERT_LOG_FILE  = os.path.join(BASE_DIR, "alerts.log")
HIGH_RISK_SCORE = 0.8       # Normalised anomaly score → HIGH alert