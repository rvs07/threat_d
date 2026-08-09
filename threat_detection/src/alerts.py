# src/alerts.py

import os
import sys
import json
import hashlib
import textwrap
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from config import DATA_PROC_DIR, ALERT_LOG_FILE, HIGH_RISK_SCORE


# ── Constants ──────────────────────────────────────────────────────────────

ALERT_THRESHOLD_SEVERITY = {"HIGH", "CRITICAL"}
COOLDOWN_MINUTES         = 60    # suppress duplicate alerts within this window

ESCALATION_MAP = {
    "CRITICAL" : "IMMEDIATE",
    "HIGH"     : "INVESTIGATE",
    "MEDIUM"   : "MONITOR",
    "LOW"      : "LOG_ONLY",
    "NORMAL"   : "NONE",
}

RECOMMENDED_ACTIONS = {
    "BRUTE_FORCE"       : (
        "Block source IP immediately. "
        "Enable account lockout policy. "
        "Review authentication logs for compromised accounts. "
        "Consider enabling MFA on targeted services."
    ),
    "DDOS"              : (
        "Activate rate limiting on affected endpoints. "
        "Block IP range at firewall level. "
        "Enable DDoS mitigation service. "
        "Alert network operations team."
    ),
    "PORT_SCAN"         : (
        "Block scanning IP at perimeter firewall. "
        "Review exposed services on detected ports. "
        "Check for follow-up exploitation attempts. "
        "Enable port-scan detection IDS rules."
    ),
    "DATA_EXFILTRATION" : (
        "URGENT: Isolate affected host immediately. "
        "Block outbound traffic from source IP. "
        "Identify and secure the data pathway used. "
        "Initiate incident response procedure."
    ),
    "SUSPICIOUS_ACTIVITY": (
        "Increase monitoring on this IP. "
        "Correlate with other log sources. "
        "Review user activity if internal IP."
    ),
    "ML_ANOMALY"        : (
        "Investigate manually — ML flagged unusual pattern. "
        "Correlate with network flow logs. "
        "Escalate if pattern persists."
    ),
}

SEVERITY_ICONS = {
    "CRITICAL" : "🔴",
    "HIGH"     : "🟠",
    "MEDIUM"   : "🟡",
    "LOW"      : "🔵",
    "NORMAL"   : "🟢",
}


# ── Alert ID generator ─────────────────────────────────────────────────────

def _make_alert_id(ip: str, threat_type: str, hour_bucket: str) -> str:
    """
    Deterministic ID based on IP + threat_type + hour.
    Same IP doing same attack in the same hour → same ID → deduplicated.
    """
    raw = f"{ip}|{threat_type}|{hour_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:12].upper()


# ── Alert builder ──────────────────────────────────────────────────────────

def _build_alerts(df: pd.DataFrame) -> list[dict]:
    """
    Aggregate threat rows into deduplicated alert objects.
    One alert per (IP, threat_type, hour_bucket).
    """
    # Filter to actionable threats only
    threat_df = df[
        df["severity"].isin(ALERT_THRESHOLD_SEVERITY)
    ].copy()

    if threat_df.empty:
        print("  [!] No HIGH/CRITICAL threats found — no alerts generated.")
        return []

    # Create hour bucket for deduplication window
    threat_df["hour_bucket"] = (
        threat_df["timestamp"].dt.floor("h").astype(str)
    )

    alerts = []
    groups = threat_df.groupby(["ip_address", "threat_type", "hour_bucket"])

    for (ip, threat_type, hour_bucket), grp in groups:
        severity   = grp["severity"].mode()[0]
        alert_id   = _make_alert_id(ip, threat_type, hour_bucket)
        avg_risk   = round(float(grp["final_risk"].mean()),  4)
        max_risk   = round(float(grp["final_risk"].max()),   4)
        event_count= int(len(grp))
        first_seen = str(grp["timestamp"].min())[:19]
        last_seen  = str(grp["timestamp"].max())[:19]

        # Escalation level
        escalation = ESCALATION_MAP.get(severity, "MONITOR")

        # Recommended action
        action = RECOMMENDED_ACTIONS.get(
            threat_type,
            "Review logs and apply appropriate security controls."
        )

        # Auto-description
        description = (
            f"{event_count} {threat_type.replace('_',' ').title()} "
            f"event(s) detected from {ip} between "
            f"{first_seen} and {last_seen}. "
            f"Average risk score: {avg_risk:.4f}."
        )

        alerts.append({
            "alert_id"           : alert_id,
            "generated_at"       : datetime.now().isoformat(),
            "first_seen"         : first_seen,
            "last_seen"          : last_seen,
            "ip_address"         : ip,
            "threat_type"        : threat_type,
            "severity"           : severity,
            "escalation"         : escalation,
            "event_count"        : event_count,
            "avg_risk"           : avg_risk,
            "max_risk"           : max_risk,
            "recommended_action" : action,
            "description"        : description,
            "ports_targeted"     : sorted(
                grp["port"].dropna().astype(int).unique().tolist()
            )[:10],
            "actions_seen"       : grp["action"].unique().tolist(),
        })

    # Sort: CRITICAL first, then by max_risk descending
    severity_order = {"CRITICAL":0, "HIGH":1, "MEDIUM":2, "LOW":3}
    alerts.sort(key=lambda a: (
        severity_order.get(a["severity"], 9),
        -a["max_risk"]
    ))

    return alerts


# ── Cooldown / deduplication ───────────────────────────────────────────────

def _apply_cooldown(alerts: list[dict],
                    existing_path: str) -> list[dict]:
    """
    Load previously written alerts and suppress any alert whose
    alert_id was already fired within COOLDOWN_MINUTES.
    Returns only new/non-suppressed alerts.
    """
    seen_ids: set[str] = set()

    if os.path.exists(existing_path):
        with open(existing_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)
                    # Check if within cooldown window
                    prev_time = datetime.fromisoformat(
                        prev.get("generated_at", "2000-01-01")
                    )
                    age_mins = (
                        datetime.now() - prev_time
                    ).total_seconds() / 60

                    if age_mins <= COOLDOWN_MINUTES:
                        seen_ids.add(prev.get("alert_id", ""))
                except Exception:
                    pass

    new_alerts = [a for a in alerts
                  if a["alert_id"] not in seen_ids]

    suppressed = len(alerts) - len(new_alerts)
    if suppressed:
        print(f"  [!] {suppressed} alert(s) suppressed "
              f"(cooldown {COOLDOWN_MINUTES} min).")
    return new_alerts


# ── Writers ────────────────────────────────────────────────────────────────

def _write_jsonl(alerts: list[dict], path: str):
    """Append alerts to JSONL file (one JSON object per line)."""
    with open(path, "w") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")
    print(f"  [✔] JSONL alerts saved  → {path}  "
          f"({len(alerts)} alerts)")


def _write_text_report(alerts: list[dict], path: str):
    """
    Write a human-readable alert report — suitable for
    email body or printed incident report.
    """
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    border = "=" * 62

    lines = [
        border,
        "  🛡️  AI THREAT DETECTION — ALERT REPORT",
        f"  Generated : {now}",
        f"  Alerts    : {len(alerts)}  "
        f"(CRITICAL: "
        f"{sum(1 for a in alerts if a['severity']=='CRITICAL')}  |  "
        f"HIGH: "
        f"{sum(1 for a in alerts if a['severity']=='HIGH')})",
        border,
        "",
    ]

    for i, alert in enumerate(alerts, 1):
        icon = SEVERITY_ICONS.get(alert["severity"], "⚪")
        lines += [
            f"  ALERT #{i:03d}  [{alert['alert_id']}]",
            f"  {icon} Severity   : {alert['severity']} "
            f"— {alert['escalation']}",
            f"  🎯 Threat     : {alert['threat_type']}",
            f"  🌐 Source IP  : {alert['ip_address']}",
            f"  📅 First Seen : {alert['first_seen']}",
            f"  📅 Last Seen  : {alert['last_seen']}",
            f"  📊 Events     : {alert['event_count']}  "
            f"(Avg Risk: {alert['avg_risk']:.4f}  |  "
            f"Max Risk: {alert['max_risk']:.4f})",
            f"  🔌 Ports      : {alert['ports_targeted'][:5]}",
            "",
            f"  📝 Description:",
        ]
        # Word-wrap description
        for chunk in textwrap.wrap(alert["description"], width=56):
            lines.append(f"     {chunk}")

        lines += [
            "",
            f"  ✅ Recommended Action:",
        ]
        for chunk in textwrap.wrap(
                alert["recommended_action"], width=56):
            lines.append(f"     {chunk}")

        lines += ["", "-" * 62, ""]

    lines += [
        border,
        "  END OF REPORT",
        border,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [✔] Text report saved   → {path}")


def _write_json_report(alerts: list[dict], path: str):
    """Write full structured JSON report."""
    report = {
        "report_meta" : {
            "generated_at"    : datetime.now().isoformat(),
            "total_alerts"    : len(alerts),
            "critical_count"  : sum(
                1 for a in alerts if a["severity"] == "CRITICAL"),
            "high_count"      : sum(
                1 for a in alerts if a["severity"] == "HIGH"),
            "unique_ips"      : len(
                set(a["ip_address"] for a in alerts)),
        },
        "alerts" : alerts,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [✔] JSON report saved   → {path}")


# ── Console summary ────────────────────────────────────────────────────────

def _print_console_summary(alerts: list[dict]):
    print("\n  " + "=" * 55)
    print("   🚨 ALERT SYSTEM SUMMARY")
    print("  " + "=" * 55)

    total    = len(alerts)
    critical = [a for a in alerts if a["severity"] == "CRITICAL"]
    high     = [a for a in alerts if a["severity"] == "HIGH"]

    print(f"   Total Alerts  : {total}")
    print(f"   🔴 Critical   : {len(critical)}")
    print(f"   🟠 High       : {len(high)}")
    print(f"   Unique IPs    : "
          f"{len(set(a['ip_address'] for a in alerts))}")

    if critical:
        print("\n  🔴 CRITICAL ALERTS — IMMEDIATE ACTION REQUIRED:")
        for a in critical[:5]:
            print(f"\n     [{a['alert_id']}] {a['threat_type']}")
            print(f"     IP        : {a['ip_address']}")
            print(f"     Events    : {a['event_count']}  |  "
                  f"Max Risk: {a['max_risk']:.4f}")
            print(f"     Escalation: {a['escalation']}")
            print(f"     Action    : "
                  + textwrap.shorten(
                      a['recommended_action'], width=55))

    if high:
        print("\n  🟠 HIGH ALERTS — INVESTIGATE:")
        for a in high[:5]:
            print(f"     [{a['alert_id']}] {a['threat_type']} "
                  f"— {a['ip_address']}  "
                  f"(risk={a['max_risk']:.4f})")

    print("\n  " + "=" * 55)


# ── Statistics ─────────────────────────────────────────────────────────────

def get_alert_statistics(alerts: list[dict]) -> dict:
    """Return summary statistics dict — used by dashboard."""
    if not alerts:
        return {}

    by_severity    = defaultdict(int)
    by_threat_type = defaultdict(int)
    by_escalation  = defaultdict(int)
    ip_risk        = defaultdict(float)

    for a in alerts:
        by_severity[a["severity"]]       += 1
        by_threat_type[a["threat_type"]] += 1
        by_escalation[a["escalation"]]   += 1
        ip_risk[a["ip_address"]] = max(
            ip_risk[a["ip_address"]], a["max_risk"]
        )

    return {
        "total"          : len(alerts),
        "by_severity"    : dict(by_severity),
        "by_threat_type" : dict(by_threat_type),
        "by_escalation"  : dict(by_escalation),
        "top_ips"        : dict(
            sorted(ip_risk.items(),
                   key=lambda x: x[1], reverse=True)[:10]
        ),
        "avg_risk"       : round(
            sum(a["avg_risk"] for a in alerts) / len(alerts), 4
        ),
    }


def run() -> list[dict]:
    threat_path  = os.path.join(DATA_PROC_DIR, "threat_logs.csv")
    jsonl_path   = ALERT_LOG_FILE
    report_txt   = os.path.join(DATA_PROC_DIR, "alert_report.txt")
    report_json  = os.path.join(DATA_PROC_DIR, "alert_report.json")

    print("\n[Phase 7] Alert System")
    print("-" * 40)

    # Load threat-labelled data
    df = pd.read_csv(threat_path, parse_dates=["timestamp"])
    print(f"  [+] Loaded threat logs : {df.shape}")

    # Build deduplicated alerts
    alerts = _build_alerts(df)
    print(f"  [+] Raw alerts built   : {len(alerts)}")

    # Apply cooldown deduplication
    alerts = _apply_cooldown(alerts, jsonl_path)
    print(f"  [+] After cooldown     : {len(alerts)} alerts")

    if not alerts:
        print("  [i] No new alerts to dispatch.")
        return []

    # Write all outputs
    _write_jsonl(alerts, jsonl_path)
    _write_text_report(alerts, report_txt)
    _write_json_report(alerts, report_json)

    # Console summary
    _print_console_summary(alerts)

    stats = get_alert_statistics(alerts)
    print(f"\n  📊 Alert Statistics:")
    print(f"     Avg Risk Score : {stats['avg_risk']:.4f}")
    print(f"     By Escalation  : {stats['by_escalation']}")
    print(f"     By Threat Type : {stats['by_threat_type']}")

    return alerts


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()