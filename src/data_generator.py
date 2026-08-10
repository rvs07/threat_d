import os
import sys
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

from config import RAW_LOG_FILE, DATA_RAW_DIR

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────

NORMAL_ACTIONS = [
    "LOGIN_SUCCESS", "LOGOUT", "FILE_ACCESS",
    "FILE_DOWNLOAD", "API_CALL", "PAGE_VIEW", "DB_QUERY"
]

ATTACK_ACTIONS = {
    "brute_force"  : "LOGIN_FAILED",
    "ddos"         : "API_CALL",
    "port_scan"    : "CONNECTION_ATTEMPT",
    "data_exfil"   : "FILE_DOWNLOAD",
}

PROTOCOLS   = ["TCP", "UDP", "HTTP", "HTTPS", "SSH", "FTP"]
NORMAL_PORTS= [80, 443, 8080, 3306, 5432, 27017]
ATTACK_PORTS= [22, 23, 3389, 4444, 1337, 9999]

USERS = (
    [fake.user_name() for _ in range(30)]   # legitimate users
    + ["admin", "root", "administrator", "guest", "test"]
)


# ── Helper generators ──────────────────────────────────────────────────────

def _random_ip(private: bool = True) -> str:
    if private:
        return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    return fake.ipv4_public()


def _make_normal_entry(ts: datetime) -> dict:
    return {
        "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
        "ip_address"  : _random_ip(private=True),
        "user"        : random.choice(USERS[:30]),   # only legit users
        "action"      : random.choice(NORMAL_ACTIONS),
        "status_code" : random.choices([200, 201, 204, 301, 304],
                                        weights=[60,10,5,15,10])[0],
        "bytes_sent"  : random.randint(100, 5_000),
        "port"        : random.choice(NORMAL_PORTS),
        "protocol"    : random.choice(["HTTP", "HTTPS", "TCP"]),
        "duration_ms" : random.randint(10, 500),
        "label"       : "normal",
    }


def _make_brute_force_block(start_ts: datetime, n: int = 20) -> list:
    """Single IP hammering login endpoint repeatedly."""
    ip = _random_ip(private=False)          # attacker from outside
    entries = []
    for i in range(n):
        ts = start_ts + timedelta(seconds=i * random.randint(1, 4))
        entries.append({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address"  : ip,
            "user"        : random.choice(["admin", "root", "administrator"]),
            "action"      : "LOGIN_FAILED",
            "status_code" : 401,
            "bytes_sent"  : random.randint(50, 200),
            "port"        : 22,
            "protocol"    : "SSH",
            "duration_ms" : random.randint(200, 800),
            "label"       : "brute_force",
        })
    return entries


def _make_ddos_block(start_ts: datetime, n: int = 50) -> list:
    """Flood of requests from multiple IPs in a short window."""
    entries = []
    for i in range(n):
        ts = start_ts + timedelta(seconds=random.uniform(0, 30))
        entries.append({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address"  : _random_ip(private=False),
            "user"        : "anonymous",
            "action"      : "API_CALL",
            "status_code" : random.choice([200, 429, 503]),
            "bytes_sent"  : random.randint(50, 300),
            "port"        : 80,
            "protocol"    : "HTTP",
            "duration_ms" : random.randint(1, 50),   # very fast
            "label"       : "ddos",
        })
    return entries


def _make_port_scan_block(start_ts: datetime, n: int = 30) -> list:
    """One IP probing many ports quickly."""
    ip = _random_ip(private=False)
    entries = []
    ports = random.sample(range(1, 10_000), n)
    for i, port in enumerate(ports):
        ts = start_ts + timedelta(seconds=i * 0.5)
        entries.append({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address"  : ip,
            "user"        : "unknown",
            "action"      : "CONNECTION_ATTEMPT",
            "status_code" : random.choice([0, 111, 403]),
            "bytes_sent"  : random.randint(0, 60),
            "port"        : port,
            "protocol"    : "TCP",
            "duration_ms" : random.randint(1, 30),
            "label"       : "port_scan",
        })
    return entries


def _make_data_exfil_block(start_ts: datetime, n: int = 10) -> list:
    """Unusually large file downloads from a single IP."""
    ip = _random_ip(private=False)
    entries = []
    for i in range(n):
        ts = start_ts + timedelta(minutes=i * 2)
        entries.append({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address"  : ip,
            "user"        : random.choice(USERS[:30]),
            "action"      : "FILE_DOWNLOAD",
            "status_code" : 200,
            "bytes_sent"  : random.randint(50_000, 500_000),  # huge
            "port"        : random.choice([21, 443]),
            "protocol"    : random.choice(["FTP", "HTTPS"]),
            "duration_ms" : random.randint(2_000, 10_000),
            "label"       : "data_exfil",
        })
    return entries


# ── Main generator ─────────────────────────────────────────────────────────

def generate_logs(
    n_normal : int = 4_000,
    seed     : int = 42
) -> pd.DataFrame:
    """
    Generate a mixed dataset of normal + attack log entries.

    Parameters
    ----------
    n_normal : int   – number of normal baseline entries
    seed     : int   – reproducibility seed

    Returns
    -------
    pd.DataFrame sorted by timestamp
    """
    random.seed(seed)
    np.random.seed(seed)

    start_time = datetime(2024, 1, 1, 0, 0, 0)
    all_entries: list[dict] = []

    # ── Normal traffic ────────────────────────────────────────────────
    print(f"  [+] Generating {n_normal:,} normal log entries …")
    for i in range(n_normal):
        ts = start_time + timedelta(seconds=i * random.randint(1, 60))
        all_entries.append(_make_normal_entry(ts))

    # ── Attack injections ─────────────────────────────────────────────
    attack_schedule = [
        # (offset_hours, attack_type,  kwargs)
        (  6,  "brute_force",  {"n": 25}),
        ( 12,  "ddos",         {"n": 60}),
        ( 18,  "port_scan",    {"n": 35}),
        ( 30,  "brute_force",  {"n": 30}),
        ( 42,  "data_exfil",   {"n": 12}),
        ( 55,  "ddos",         {"n": 80}),
        ( 68,  "port_scan",    {"n": 40}),
        ( 75,  "data_exfil",   {"n": 15}),
        ( 88,  "brute_force",  {"n": 20}),
        (100,  "ddos",         {"n": 50}),
    ]

    for offset_h, attack_type, kwargs in attack_schedule:
        attack_start = start_time + timedelta(hours=offset_h)
        if attack_type == "brute_force":
            all_entries.extend(_make_brute_force_block(attack_start, **kwargs))
        elif attack_type == "ddos":
            all_entries.extend(_make_ddos_block(attack_start, **kwargs))
        elif attack_type == "port_scan":
            all_entries.extend(_make_port_scan_block(attack_start, **kwargs))
        elif attack_type == "data_exfil":
            all_entries.extend(_make_data_exfil_block(attack_start, **kwargs))

    # ── Assemble DataFrame ────────────────────────────────────────────
    df = pd.DataFrame(all_entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── Runner ─────────────────────────────────────────────────────────────────

def run():

    df = generate_logs(n_normal=4_000)

    # Save
    df.to_csv(RAW_LOG_FILE, index=False)
    print(f"\n  [✔] Saved {len(df):,} log entries → {RAW_LOG_FILE}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n  📊 Dataset Summary:")
    print(f"     Total records  : {len(df):,}")
    print(f"     Date range     : {df['timestamp'].min()} "
          f"→ {df['timestamp'].max()}")
    print(f"     Columns        : {list(df.columns)}")

    print("\n  🏷️  Label Distribution:")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"     {label:<15} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n  🔍 Sample Rows (first 5):")
    print(df.head().to_string(index=False))

    return df


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()