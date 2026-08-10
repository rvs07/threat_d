# src/features.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import CLEAN_LOG_FILE, FEATURE_FILE, DATA_PROC_DIR


# ── Row-level features ─────────────────────────────────────────────────────

def _add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple per-row binary / derived flags."""

    # Failed login flag
    df["is_failed_login"] = (df["action"] == "LOGIN_FAILED").astype(int)

    # Large data transfer flag  (> 50 KB)
    df["is_large_transfer"] = (df["bytes_sent"] > 50_000).astype(int)

    # Odd-hour activity  (11 PM – 5 AM)
    df["is_odd_hour"] = df["hour"].apply(
        lambda h: 1 if (h >= 23 or h <= 5) else 0
    )

    # Connection attempt flag
    df["is_conn_attempt"] = (df["action"] == "CONNECTION_ATTEMPT").astype(int)

    # Anonymous / unknown user flag
    df["is_anon_user"] = df["user"].apply(
        lambda u: 1 if u in {"anonymous", "unknown", "guest"} else 0
    )

    # Status-code category
    df["is_client_error"]  = ((df["status_code"] >= 400) &
                               (df["status_code"] < 500)).astype(int)
    df["is_server_error"]  = ((df["status_code"] >= 500) &
                               (df["status_code"] < 600)).astype(int)
    df["is_success"]       = ((df["status_code"] >= 200) &
                               (df["status_code"] < 300)).astype(int)

    added = [
        "is_failed_login", "is_large_transfer", "is_odd_hour",
        "is_conn_attempt", "is_anon_user",
        "is_client_error", "is_server_error", "is_success",
    ]
    print(f"  [+] Row-level features added   : {added}")
    return df


# ── IP-level aggregate features ────────────────────────────────────────────

def _add_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by ip_address and compute behavioural aggregates.
    These are then merged back onto every row for that IP.
    """

    grp = df.groupby("ip_address")

    agg = pd.DataFrame({
        # Volume signals
        "ip_request_count"  : grp["action"].transform("count"),
        "ip_failed_logins"  : grp["is_failed_login"].transform("sum"),
        "ip_large_transfers": grp["is_large_transfer"].transform("sum"),
        "ip_conn_attempts"  : grp["is_conn_attempt"].transform("sum"),

        # Diversity signals  (more unique = more suspicious)
        "ip_unique_ports"   : grp["port"].transform("nunique"),
        "ip_unique_actions" : grp["action"].transform("nunique"),

        # Size / speed signals
        "ip_avg_bytes"      : grp["bytes_sent"].transform("mean"),
        "ip_max_bytes"      : grp["bytes_sent"].transform("max"),
        "ip_avg_duration"   : grp["duration_ms"].transform("mean"),
        "ip_min_duration"   : grp["duration_ms"].transform("min"),

        # Error-rate signal
        "ip_error_rate"     : grp["is_client_error"].transform("mean"),
    })

    df = pd.concat([df, agg], axis=1)

    added = list(agg.columns)
    print(f"  [+] IP-aggregate features added: {added}")
    return df


# ── Time-window features ───────────────────────────────────────────────────

def _add_time_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count events per IP per HOUR window.
    High count in a short window → DDoS / brute-force signal.
    """
    # Create hour-bucket key
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    window_grp = df.groupby(["ip_address", "hour_bucket"])

    df["ip_requests_per_hour"] = window_grp["action"].transform("count")
    df["ip_failures_per_hour"] = window_grp["is_failed_login"].transform("sum")
    df["ip_ports_per_hour"]    = window_grp["port"].transform("nunique")

    # Drop helper column
    df.drop(columns=["hour_bucket"], inplace=True)

    added = ["ip_requests_per_hour", "ip_failures_per_hour",
             "ip_ports_per_hour"]
    print(f"  [+] Time-window features added : {added}")
    return df


# ── Composite risk score ───────────────────────────────────────────────────

def _add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight heuristic risk score (0–1) combining multiple signals.
    This is NOT the ML model — it's a human-interpretable pre-score
    that also becomes a feature for the Isolation Forest.
    """

    # Normalise each component to 0-1
    def _norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    score = (
        0.25 * _norm(df["ip_failed_logins"])
      + 0.20 * _norm(df["ip_requests_per_hour"])
      + 0.20 * _norm(df["ip_unique_ports"])
      + 0.15 * _norm(df["ip_avg_bytes"])
      + 0.10 * df["is_odd_hour"]
      + 0.05 * df["sensitive_port"]
      + 0.05 * df["is_anon_user"]
    )

    df["heuristic_risk_score"] = score.clip(0, 1).round(4)
    print("  [+] Heuristic risk score added : heuristic_risk_score  (0–1)")
    return df


# ── Feature selection ──────────────────────────────────────────────────────

# These are the columns the Isolation Forest will train on.
MODEL_FEATURES = [
    # Row-level
    "is_failed_login",
    "is_large_transfer",
    "is_odd_hour",
    "is_conn_attempt",
    "is_anon_user",
    "is_client_error",
    "sensitive_port",

    # IP-aggregate
    "ip_request_count",
    "ip_failed_logins",
    "ip_unique_ports",
    "ip_unique_actions",
    "ip_avg_bytes",
    "ip_max_bytes",
    "ip_avg_duration",
    "ip_error_rate",

    # Time-window
    "ip_requests_per_hour",
    "ip_failures_per_hour",
    "ip_ports_per_hour",

    # Composite
    "heuristic_risk_score",

    # Scaled numerics from Phase 2
    "bytes_log_scaled",
    "duration_ms_scaled",
    "status_code_scaled",
    "hour",
    "day_of_week",
    "is_night",
]


def _scale_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final MinMax pass on aggregate features so all MODEL_FEATURES
    are on a comparable 0-1 scale for Isolation Forest.
    """
    cols_to_scale = [
        "ip_request_count", "ip_failed_logins", "ip_unique_ports",
        "ip_unique_actions", "ip_avg_bytes", "ip_max_bytes",
        "ip_avg_duration", "ip_error_rate",
        "ip_requests_per_hour", "ip_failures_per_hour", "ip_ports_per_hour",
        "ip_large_transfers", "ip_conn_attempts", "ip_min_duration",
        "ip_max_bytes",
    ]
    # Only scale columns that exist and aren't already 0-1
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]

    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"  [+] Final scaling applied on   : {len(cols_to_scale)} columns")
    return df


# ── Summary ────────────────────────────────────────────────────────────────

def _print_feature_summary(df: pd.DataFrame):
    print("\n  📊 Feature Engineering Summary:")
    print(f"     Total columns now    : {len(df.columns)}")
    print(f"     Model feature count  : {len(MODEL_FEATURES)}")

    print("\n  🔍 Top 10 Riskiest IPs (by heuristic_risk_score):")
    top_ips = (
        df.groupby("ip_address")["heuristic_risk_score"]
          .max()
          .sort_values(ascending=False)
          .head(10)
    )
    for ip, score in top_ips.items():
        bar = "█" * int(score * 20)
        print(f"     {ip:<18}  score={score:.4f}  {bar}")

    print("\n  📋 Model Features Preview (first 3 rows):")
    preview_cols = [
        "ip_address", "label",
        "ip_failed_logins", "ip_unique_ports",
        "ip_requests_per_hour", "heuristic_risk_score"
    ]
    print(df[preview_cols].head(3).to_string(index=False))

    print("\n  🏷️  Correlation with label_enc (top features):")
    corr = (
        df[MODEL_FEATURES + ["label_enc"]]
          .corr()["label_enc"]
          .drop("label_enc")
          .abs()
          .sort_values(ascending=False)
          .head(8)
    )
    for feat, val in corr.items():
        bar = "█" * int(val * 30)
        print(f"     {feat:<30}  r={val:.3f}  {bar}")


# ── Public runner ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    print("\n[Phase 3] Feature Engineering")
    print("-" * 40)

    df = pd.read_csv(CLEAN_LOG_FILE, parse_dates=["timestamp"])
    print(f"  [+] Loaded cleaned data        : {df.shape}")

    df = _add_row_features(df)
    df = _add_ip_features(df)
    df = _add_time_window_features(df)
    df = _add_risk_score(df)
    df = _scale_model_features(df)

    # Save full feature set
    df.to_csv(FEATURE_FILE, index=False)
    print(f"\n  [✔] Feature dataset saved → {FEATURE_FILE}")

    _print_feature_summary(df)
    return df


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()