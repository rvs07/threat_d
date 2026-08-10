# src/preprocess.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import RAW_LOG_FILE, CLEAN_LOG_FILE, DATA_PROC_DIR


def _load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and parse timestamps."""
    print("  [+] Loading raw log file …")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


def _report(stage: str, df: pd.DataFrame):
    """Print a short status line after each cleaning step."""
    print(f"      [{stage}] shape={df.shape}  "
          f"nulls={df.isnull().sum().sum()}")


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"  [+] Duplicate rows removed : {dropped:,}")
    return df


def _fix_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs with sensible defaults per column."""
    fills = {
        "user"        : "unknown",
        "action"      : "UNKNOWN",
        "protocol"    : "UNKNOWN",
        "status_code" : 0,
        "bytes_sent"  : 0,
        "port"        : 0,
        "duration_ms" : 0,
    }
    before = df.isnull().sum().sum()
    df.fillna(fills, inplace=True)
    after  = df.isnull().sum().sum()
    print(f"  [+] Missing values fixed   : {before:,} → {after:,}")
    return df


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip physically impossible values.
    Negative durations or bytes make no sense.
    """
    df["duration_ms"] = df["duration_ms"].clip(lower=0, upper=60_000)
    df["bytes_sent"]  = df["bytes_sent"].clip(lower=0, upper=10_000_000)
    df["port"]        = df["port"].clip(lower=0, upper=65_535)
    df["status_code"] = df["status_code"].clip(lower=0, upper=599)
    print("  [+] Outlier clipping applied  (duration, bytes, port, status)")
    return df


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features from the timestamp column."""
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"]= df["timestamp"].dt.dayofweek   # 0=Mon … 6=Sun
    df["is_night"]   = df["hour"].apply(
        lambda h: 1 if (h >= 22 or h <= 5) else 0    # 10 PM – 5 AM
    )
    print("  [+] Time features added       : hour, day_of_week, is_night")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode low-cardinality categorical columns.
    Encoders are fitted on the full dataset (no train/test split yet).
    """
    cat_cols = ["action", "protocol"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        classes = list(le.classes_)
        print(f"  [+] Encoded '{col}' → '{col}_enc'  classes={classes}")

    # Store mapping as readable comment in returned df attribute
    df.attrs["label_encoders"] = encoders
    return df


def _flag_sensitive_ports(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the port is commonly targeted by attackers."""
    SENSITIVE = {21, 22, 23, 25, 53, 135, 139, 445,
                 1433, 1521, 3306, 3389, 4444, 8080, 9999}
    df["sensitive_port"] = df["port"].apply(
        lambda p: 1 if p in SENSITIVE else 0
    )
    print(f"  [+] sensitive_port flag added  "
          f"({df['sensitive_port'].sum():,} flagged rows)")
    return df


def _normalize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MinMax scaling to skewed numeric columns.
    bytes_sent is first log-transformed to reduce right skew.
    """
    # Log-transform bytes_sent (add 1 to avoid log(0))
    df["bytes_log"] = np.log1p(df["bytes_sent"])

    scale_cols = ["bytes_log", "duration_ms", "port", "status_code"]
    scaler = MinMaxScaler()
    df[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(
        df[scale_cols]
    )

    scaled_names = [f"{c}_scaled" for c in scale_cols]
    print(f"  [+] Normalized columns         : {scaled_names}")
    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text label → integer for model use.
    normal=0, brute_force=1, ddos=2, port_scan=3, data_exfil=4
    """
    label_map = {
        "normal"      : 0,
        "brute_force" : 1,
        "ddos"        : 2,
        "port_scan"   : 3,
        "data_exfil"  : 4,
    }
    df["label_enc"] = df["label"].map(label_map).fillna(0).astype(int)
    print(f"  [+] Label encoded              : {label_map}")
    return df


# ── Summary printer ────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame):
    print("\n  📊 Cleaned Dataset Summary:")
    print(f"     Shape          : {df.shape}")
    print(f"     Memory usage   : "
          f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\n  🔢 Numeric Column Stats:")
    num_cols = ["bytes_sent", "duration_ms", "port",
                "status_code", "hour", "bytes_log"]
    print(df[num_cols].describe().round(2).to_string())

    print("\n  🏷️  Label Distribution (cleaned):")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"     {label:<15} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n  📋 Final Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"     {i:>2}. {col}")


# ── Public runner ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    print("\n[Phase 2] Data Preprocessing")
    print("-" * 40)

    # Pipeline steps
    df = _load_raw(RAW_LOG_FILE)
    _report("raw", df)

    df = _drop_duplicates(df)
    df = _fix_missing_values(df)
    df = _clip_outliers(df)
    df = _extract_time_features(df)
    df = _encode_categoricals(df)
    df = _flag_sensitive_ports(df)
    df = _normalize_numerics(df)
    df = _encode_label(df)
    _report("final", df)

    # Save
    df.to_csv(CLEAN_LOG_FILE, index=False)
    print(f"\n  [✔] Cleaned data saved → {CLEAN_LOG_FILE}")

    _print_summary(df)
    return df


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()