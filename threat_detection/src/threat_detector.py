# src/threat_detector.py

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROC_DIR, ALERT_LOG_FILE,
                    MAX_FAILED_LOGINS, MAX_REQUESTS_PER_MIN,
                    SUSPICIOUS_PORTS, HIGH_RISK_SCORE)


# ── Severity levels ────────────────────────────────────────────────────────

SEVERITY_RANK = {
    "CRITICAL" : 4,
    "HIGH"     : 3,
    "MEDIUM"   : 2,
    "LOW"      : 1,
    "NORMAL"   : 0,
}

SEVERITY_COLOR = {
    "CRITICAL" : "🔴",
    "HIGH"     : "🟠",
    "MEDIUM"   : "🟡",
    "LOW"      : "🔵",
    "NORMAL"   : "🟢",
}


# ── Individual rule detectors ──────────────────────────────────────────────

def _detect_brute_force(row: pd.Series) -> dict | None:
    """
    Rule: Same IP has ≥ MAX_FAILED_LOGINS failed logins per hour
          AND this specific row is a failed login attempt.
    """
    # ip_failures_per_hour is already normalised 0-1 in Phase 3
    # We reverse-engineer the threshold:
    # raw failures ≥ MAX_FAILED_LOGINS  →  after MinMax this ≈ > 0.05
    # Use is_failed_login + ip_failures_per_hour combined
    if (row.get("is_failed_login", 0) == 1 and
            row.get("ip_failures_per_hour", 0) > 0.04):
        return {
            "threat_type" : "BRUTE_FORCE",
            "severity"    : "HIGH",
            "confidence"  : min(0.99,
                                0.6 + row.get("ip_failures_per_hour", 0)),
            "description" : (
                f"Repeated login failures detected from "
                f"{row.get('ip_address','?')}. "
                f"Possible brute-force / credential stuffing attack."
            ),
            "rule"        : "ip_failures_per_hour > threshold "
                            "AND is_failed_login=1",
        }
    return None


def _detect_ddos(row: pd.Series) -> dict | None:
    """
    Rule: Very high request volume per hour from an IP
          AND very short average duration (flood, not real traffic).
    """
    high_volume   = row.get("ip_requests_per_hour", 0) > 0.5
    fast_requests = row.get("duration_ms_scaled", 1) < 0.05

    if high_volume and fast_requests:
        return {
            "threat_type" : "DDOS",
            "severity"    : "CRITICAL",
            "confidence"  : min(0.99,
                                0.7 + row.get("ip_requests_per_hour", 0) * 0.3),
            "description" : (
                f"Abnormally high request rate from "
                f"{row.get('ip_address','?')}. "
                f"Possible Distributed Denial-of-Service (DDoS) attack."
            ),
            "rule"        : "ip_requests_per_hour > 0.5 "
                            "AND duration_ms_scaled < 0.05",
        }
    return None


def _detect_port_scan(row: pd.Series) -> dict | None:
    """
    Rule: IP hits many unique ports per hour
          AND this row is a connection attempt.
    """
    many_ports   = row.get("ip_ports_per_hour", 0) > 0.1
    conn_attempt = row.get("is_conn_attempt", 0) == 1

    if many_ports and conn_attempt:
        return {
            "threat_type" : "PORT_SCAN",
            "severity"    : "HIGH",
            "confidence"  : min(0.99,
                                0.55 + row.get("ip_ports_per_hour", 0)),
            "description" : (
                f"Multiple port connection attempts from "
                f"{row.get('ip_address','?')}. "
                f"Possible network reconnaissance / port scanning."
            ),
            "rule"        : "ip_ports_per_hour > 0.1 "
                            "AND is_conn_attempt=1",
        }
    return None


def _detect_data_exfil(row: pd.Series) -> dict | None:
    """
    Rule: Very large bytes sent by this row
          AND IP has multiple large transfers overall.
    """
    large_row      = row.get("is_large_transfer", 0) == 1
    many_transfers = row.get("ip_large_transfers", 0) > 0.01

    if large_row and many_transfers:
        return {
            "threat_type" : "DATA_EXFILTRATION",
            "severity"    : "CRITICAL",
            "confidence"  : min(0.99,
                                0.65 + row.get("ip_large_transfers", 0) * 0.5),
            "description" : (
                f"Unusually large data transfers detected from "
                f"{row.get('ip_address','?')}. "
                f"Possible data exfiltration in progress."
            ),
            "rule"        : "is_large_transfer=1 "
                            "AND ip_large_transfers > threshold",
        }
    return None


def _detect_suspicious_time(row: pd.Series) -> dict | None:
    """
    Rule: Activity during odd hours on sensitive ports.
    """
    if (row.get("is_odd_hour", 0) == 1 and
            row.get("sensitive_port", 0) == 1 and
            row.get("anomaly_score", 0) > 0.5):
        return {
            "threat_type" : "SUSPICIOUS_ACTIVITY",
            "severity"    : "MEDIUM",
            "confidence"  : 0.5 + row.get("anomaly_score", 0) * 0.3,
            "description" : (
                f"Sensitive port activity during off-hours from "
                f"{row.get('ip_address','?')}. "
                f"Warrants investigation."
            ),
            "rule"        : "is_odd_hour=1 AND sensitive_port=1 "
                            "AND anomaly_score > 0.5",
        }
    return None


def _detect_ml_anomaly(row: pd.Series) -> dict | None:
    """
    Fallback: ML flagged it but no specific rule matched.
    """
    score = row.get("anomaly_score", 0)
    pred  = row.get("prediction", 0)

    if pred == 1 and score >= HIGH_RISK_SCORE:
        return {
            "threat_type" : "ML_ANOMALY",
            "severity"    : "HIGH",
            "confidence"  : float(score),
            "description" : (
                f"Isolation Forest flagged anomalous behaviour from "
                f"{row.get('ip_address','?')} "
                f"(score={score:.3f}). No matching rule pattern."
            ),
            "rule"        : f"ML anomaly_score >= {HIGH_RISK_SCORE}",
        }
    elif pred == 1 and score >= 0.5:
        return {
            "threat_type" : "ML_ANOMALY",
            "severity"    : "MEDIUM",
            "confidence"  : float(score),
            "description" : (
                f"Mildly anomalous behaviour from "
                f"{row.get('ip_address','?')} "
                f"(score={score:.3f})."
            ),
            "rule"        : "ML anomaly_score >= 0.5",
        }
    return None


# ── Rule engine ────────────────────────────────────────────────────────────

RULE_PIPELINE = [
    _detect_brute_force,
    _detect_ddos,
    _detect_port_scan,
    _detect_data_exfil,
    _detect_suspicious_time,
    _detect_ml_anomaly,          # ML fallback — always last
]


def _classify_row(row: pd.Series) -> dict:
    """
    Run every rule on a single row.
    Return the highest-severity match.
    If no rule fires → NORMAL.
    """
    hits = []
    for rule_fn in RULE_PIPELINE:
        result = rule_fn(row)
        if result:
            hits.append(result)

    if not hits:
        return {
            "threat_type" : "NORMAL",
            "severity"    : "NORMAL",
            "confidence"  : 1 - row.get("anomaly_score", 0),
            "description" : "No threat detected.",
            "rule"        : "none",
        }

    # Return highest severity hit
    hits.sort(key=lambda h: SEVERITY_RANK[h["severity"]], reverse=True)
    return hits[0]


# ── Batch classification ───────────────────────────────────────────────────

def classify_threats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule engine to every row.
    Adds columns: threat_type, severity, confidence, description, rule_matched
    """
    print("  [+] Running rule-based threat classification …")

    results = df.apply(_classify_row, axis=1)
    result_df = pd.DataFrame(list(results))

    df = df.copy()
    df["threat_type"]  = result_df["threat_type"].values
    df["severity"]     = result_df["severity"].values
    df["confidence"]   = result_df["confidence"].round(4).values
    df["description"]  = result_df["description"].values
    df["rule_matched"] = result_df["rule"].values

    # Final combined risk = weighted average of ML score + rule confidence
    df["final_risk"] = (
        0.6 * df["anomaly_score"] +
        0.4 * df["confidence"]
    ).round(4)

    total_threats = (df["severity"] != "NORMAL").sum()
    print(f"  [✔] Classification complete — "
          f"{total_threats:,} threat rows identified.")
    return df


# ── Threat summary ─────────────────────────────────────────────────────────

def build_threat_summary(df: pd.DataFrame) -> dict:
    """Build a structured summary dict for the dashboard."""
    threats = df[df["severity"] != "NORMAL"]

    summary = {
        "generated_at"      : datetime.now().isoformat(),
        "total_logs"        : int(len(df)),
        "total_threats"     : int(len(threats)),
        "threat_rate_pct"   : round(len(threats) / len(df) * 100, 2),
        "by_severity"       : threats["severity"].value_counts().to_dict(),
        "by_threat_type"    : threats["threat_type"].value_counts().to_dict(),
        "top_attacker_ips"  : (
            threats.groupby("ip_address")["final_risk"]
                   .mean()
                   .sort_values(ascending=False)
                   .head(10)
                   .round(4)
                   .to_dict()
        ),
        "critical_events"   : int(
            (df["severity"] == "CRITICAL").sum()
        ),
        "high_events"       : int(
            (df["severity"] == "HIGH").sum()
        ),
    }
    return summary


# ── Alert logger ───────────────────────────────────────────────────────────

def log_alerts(df: pd.DataFrame, path: str = ALERT_LOG_FILE):
    """Write HIGH / CRITICAL threats to alerts.log as JSON lines."""
    alerts = df[df["severity"].isin(["HIGH", "CRITICAL"])].copy()
    alerts = alerts.sort_values("final_risk", ascending=False)

    with open(path, "w") as f:
        for _, row in alerts.iterrows():
            entry = {
                "timestamp"   : str(row.get("timestamp", "")),
                "ip_address"  : str(row.get("ip_address", "")),
                "threat_type" : str(row.get("threat_type", "")),
                "severity"    : str(row.get("severity", "")),
                "confidence"  : float(row.get("confidence", 0)),
                "final_risk"  : float(row.get("final_risk", 0)),
                "description" : str(row.get("description", "")),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  [✔] {len(alerts):,} alert(s) written → {path}")


# ── Summary printer ────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, summary: dict):
    print("\n  " + "="*52)
    print("   🛡️  THREAT DETECTION REPORT")
    print("  " + "="*52)
    print(f"   Generated   : {summary['generated_at']}")
    print(f"   Total Logs  : {summary['total_logs']:,}")
    print(f"   Threats     : {summary['total_threats']:,}  "
          f"({summary['threat_rate_pct']}%)")
    print(f"   Critical    : {summary['critical_events']:,}")
    print(f"   High        : {summary['high_events']:,}")

    print("\n  📊 By Severity:")
    for sev, count in sorted(
            summary["by_severity"].items(),
            key=lambda x: SEVERITY_RANK.get(x[0], 0),
            reverse=True):
        icon = SEVERITY_COLOR.get(sev, "⚪")
        bar  = "█" * int(count / summary["total_threats"] * 30)
        print(f"     {icon} {sev:<10} {count:>5}  {bar}")

    print("\n  🔎 By Threat Type:")
    for ttype, count in sorted(
            summary["by_threat_type"].items(),
            key=lambda x: x[1], reverse=True):
        print(f"     {ttype:<22} {count:>5}")

    print("\n  🚨 Top 5 Attacker IPs:")
    for ip, risk in list(summary["top_attacker_ips"].items())[:5]:
        bar = "█" * int(risk * 20)
        print(f"     {ip:<20}  risk={risk:.4f}  {bar}")

    print("\n  📋 Sample CRITICAL Threats:")
    sample = df[df["severity"] == "CRITICAL"][
        ["timestamp", "ip_address", "threat_type",
         "confidence", "final_risk", "description"]
    ].head(3)
    if not sample.empty:
        for _, r in sample.iterrows():
            print(f"\n     🔴 {r['threat_type']}")
            print(f"        IP        : {r['ip_address']}")
            print(f"        Time      : {r['timestamp']}")
            print(f"        Confidence: {r['confidence']:.3f}")
            print(f"        Risk      : {r['final_risk']:.3f}")
            print(f"        Detail    : {r['description'][:80]}…")


# ── Public runner ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    scored_path = os.path.join(DATA_PROC_DIR, "scored_logs.csv")
    output_path = os.path.join(DATA_PROC_DIR, "threat_logs.csv")

    print("\n[Phase 5] Threat Detection Logic")
    print("-" * 40)

    df = pd.read_csv(scored_path, parse_dates=["timestamp"])
    print(f"  [+] Loaded scored logs : {df.shape}")

    df      = classify_threats(df)
    summary = build_threat_summary(df)
    log_alerts(df)

    df.to_csv(output_path, index=False)
    print(f"  [✔] Threat-labelled data saved → {output_path}")

    _print_summary(df, summary)
    return df


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()