import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from config import DATA_PROC_DIR, MODEL_DIR, ALERT_LOG_FILE

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Threat Detection Dashboard",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color : #1e2130;
        border           : 1px solid #2e3250;
        border-radius    : 10px;
        padding          : 15px;
    }

    /* Severity badge colours */
    .badge-critical { background:#ff4b4b; color:white;
                      padding:3px 10px; border-radius:12px;
                      font-weight:bold; font-size:12px; }
    .badge-high     { background:#ff8c00; color:white;
                      padding:3px 10px; border-radius:12px;
                      font-weight:bold; font-size:12px; }
    .badge-medium   { background:#ffd700; color:black;
                      padding:3px 10px; border-radius:12px;
                      font-weight:bold; font-size:12px; }
    .badge-low      { background:#1e90ff; color:white;
                      padding:3px 10px; border-radius:12px;
                      font-weight:bold; font-size:12px; }
    .badge-normal   { background:#2e8b57; color:white;
                      padding:3px 10px; border-radius:12px;
                      font-weight:bold; font-size:12px; }

    /* Section headers */
    .section-header {
        font-size: 18px; font-weight: bold;
        color: #00d4ff; margin: 10px 0;
        border-bottom: 1px solid #2e3250;
        padding-bottom: 5px;
    }

    /* Alert box */
    .alert-box {
        background-color : #2d1515;
        border-left      : 4px solid #ff4b4b;
        padding          : 12px;
        border-radius    : 5px;
        margin           : 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour maps ────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "CRITICAL" : "#ff4b4b",
    "HIGH"     : "#ff8c00",
    "MEDIUM"   : "#ffd700",
    "LOW"      : "#1e90ff",
    "NORMAL"   : "#2e8b57",
}

THREAT_COLORS = {
    "DDOS"             : "#ff4b4b",
    "BRUTE_FORCE"      : "#ff8c00",
    "PORT_SCAN"        : "#ffd700",
    "DATA_EXFILTRATION": "#9b59b6",
    "SUSPICIOUS_ACTIVITY":"#1e90ff",
    "ML_ANOMALY"       : "#e67e22",
    "NORMAL"           : "#2e8b57",
}

# ── Data loader ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_PROC_DIR, "threat_logs.csv")
    if not os.path.exists(path):
        st.error(f"❌ Data file not found: {path}\n\n"
                 "Please run `python main.py` first.")
        st.stop()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"]     = df["timestamp"].dt.floor("h")
    df["date"]     = df["timestamp"].dt.date
    return df


@st.cache_data(ttl=60)
def load_alerts() -> list:
    if not os.path.exists(ALERT_LOG_FILE):
        return []
    alerts = []
    with open(ALERT_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
    return alerts

# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image(
        "https://img.icons8.com/color/96/shield.png",
        width=60
    )
    st.sidebar.title("🛡️ Dashboard Controls")
    st.sidebar.markdown("---")

    # Date range filter
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    st.sidebar.markdown("**📅 Date Range**")
    date_from = st.sidebar.date_input("From", min_date,
                                       min_value=min_date,
                                       max_value=max_date)
    date_to   = st.sidebar.date_input("To",   max_date,
                                       min_value=min_date,
                                       max_value=max_date)

    # Severity filter
    st.sidebar.markdown("**🚨 Severity**")
    severities = ["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"]
    sel_sev = st.sidebar.multiselect(
        "Select severities", severities,
        default=["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"]
    )

    # Threat type filter
    st.sidebar.markdown("**🎯 Threat Type**")
    threat_types = sorted(df["threat_type"].unique().tolist())
    sel_threats  = st.sidebar.multiselect(
        "Select threat types", threat_types,
        default=threat_types
    )

    # Risk score slider
    st.sidebar.markdown("**⚡ Min Risk Score**")
    min_risk = st.sidebar.slider("", 0.0, 1.0, 0.0, 0.01)

    # Apply filters
    mask = (
        (df["date"] >= date_from)
      & (df["date"] <= date_to)
      & (df["severity"].isin(sel_sev))
      & (df["threat_type"].isin(sel_threats))
      & (df["final_risk"] >= min_risk)
    )
    filtered = df[mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Records", f"{len(filtered):,}")
    st.sidebar.metric("Threats in View",
                      f"{(filtered['severity'] != 'NORMAL').sum():,}")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return filtered

# ── KPI cards ──────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame):
    total       = len(df)
    threats     = (df["severity"] != "NORMAL").sum()
    critical    = (df["severity"] == "CRITICAL").sum()
    high        = (df["severity"] == "HIGH").sum()
    avg_risk    = df["final_risk"].mean()
    threat_rate = threats / total * 100 if total else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📋 Total Logs",     f"{total:,}")
    c2.metric("⚠️ Threats",        f"{threats:,}",
              delta=f"{threat_rate:.1f}%",
              delta_color="inverse")
    c3.metric("🔴 Critical",       f"{critical:,}",
              delta_color="inverse")
    c4.metric("🟠 High",           f"{high:,}",
              delta_color="inverse")
    c5.metric("📈 Avg Risk",       f"{avg_risk:.3f}")
    c6.metric("🌐 Unique IPs",
              f"{df['ip_address'].nunique():,}")

# ── Tab 1 — Overview ───────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame):
    st.markdown('<p class="section-header">📊 Attack Timeline</p>',
                unsafe_allow_html=True)

    # Timeline: threats per hour
    timeline = (
        df[df["severity"] != "NORMAL"]
          .groupby(["hour", "severity"])
          .size()
          .reset_index(name="count")
    )
    if not timeline.empty:
        fig = px.bar(
            timeline, x="hour", y="count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            title="Threats Per Hour — by Severity",
            labels={"hour":"Time","count":"Events"},
            height=350,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No threat events in selected range.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">🍩 Severity Distribution</p>',
                    unsafe_allow_html=True)
        sev_counts = df[df["severity"] != "NORMAL"]["severity"].value_counts()
        fig = px.pie(
            values=sev_counts.values,
            names=sev_counts.index,
            color=sev_counts.index,
            color_discrete_map=SEVERITY_COLORS,
            hole=0.45,
            height=320,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">🔎 Threat Type Breakdown</p>',
                    unsafe_allow_html=True)
        type_counts = (
            df[df["threat_type"] != "NORMAL"]["threat_type"]
              .value_counts()
              .reset_index()
        )
        type_counts.columns = ["threat_type", "count"]
        fig = px.bar(
            type_counts,
            x="count", y="threat_type",
            orientation="h",
            color="threat_type",
            color_discrete_map=THREAT_COLORS,
            height=320,
            labels={"count":"Events","threat_type":""},
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 — Threat Log ─────────────────────────────────────────────────────

def render_threat_log(df: pd.DataFrame):
    st.markdown('<p class="section-header">🚨 Live Threat Log</p>',
                unsafe_allow_html=True)

    threats = df[df["severity"] != "NORMAL"].sort_values(
        "final_risk", ascending=False
    )

    # Colour-code severity column
    def _badge(sev: str) -> str:
        cls = f"badge-{sev.lower()}"
        return f'<span class="{cls}">{sev}</span>'

    display_cols = [
        "timestamp","ip_address","threat_type",
        "severity","confidence","final_risk","description"
    ]
    display = threats[display_cols].copy()
    display["timestamp"] = display["timestamp"].astype(str).str[:19]
    display["confidence"] = display["confidence"].round(3)
    display["final_risk"] = display["final_risk"].round(3)

    # Search box
    search = st.text_input("🔍 Search IP / Threat Type",
                           placeholder="e.g.  192.168  or  DDOS")
    if search:
        mask = (
            display["ip_address"].str.contains(search, case=False, na=False)
          | display["threat_type"].str.contains(search, case=False, na=False)
          | display["description"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    st.caption(f"Showing {len(display):,} threat entries")

    # Risk score heatmap styling
    st.dataframe(
        display.reset_index(drop=True),
        use_container_width=True,
        height=420,
        column_config={
            "final_risk"  : st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1,
                format="%.3f",
            ),
            "confidence"  : st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1,
                format="%.3f",
            ),
            "severity"    : st.column_config.TextColumn("Severity"),
            "description" : st.column_config.TextColumn(
                "Description", width="large"
            ),
        }
    )

    # Download button
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Threats CSV",
        csv, "threats.csv", "text/csv"
    )

# ── Tab 3 — IP Analysis ────────────────────────────────────────────────────

def render_ip_analysis(df: pd.DataFrame):
    st.markdown('<p class="section-header">🌐 Top Attacker IPs</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        ip_stats = (
            df[df["severity"] != "NORMAL"]
              .groupby("ip_address")
              .agg(
                  events      = ("ip_address","count"),
                  avg_risk    = ("final_risk","mean"),
                  max_risk    = ("final_risk","max"),
                  threat_types= ("threat_type",
                                 lambda x: ", ".join(x.unique())),
              )
              .sort_values("max_risk", ascending=False)
              .head(20)
              .reset_index()
        )
        ip_stats["avg_risk"] = ip_stats["avg_risk"].round(4)
        ip_stats["max_risk"] = ip_stats["max_risk"].round(4)

        fig = px.scatter(
            ip_stats,
            x="events", y="max_risk",
            size="events",
            color="max_risk",
            color_continuous_scale="Reds",
            hover_data=["ip_address","threat_types"],
            labels={"events":"Total Events",
                    "max_risk":"Max Risk Score"},
            title="IP Risk Bubble Chart — Size = Event Count",
            height=380,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 15 Risky IPs**")
        st.dataframe(
            ip_stats[["ip_address","events",
                       "max_risk","threat_types"]].head(15),
            use_container_width=True,
            height=380,
        )

    # IP drilldown
    st.markdown('<p class="section-header">🔍 IP Drilldown</p>',
                unsafe_allow_html=True)
    all_ips = sorted(df["ip_address"].unique().tolist())
    sel_ip  = st.selectbox("Select an IP address to investigate", all_ips)

    if sel_ip:
        ip_df = df[df["ip_address"] == sel_ip].sort_values("timestamp")
        st.caption(f"**{len(ip_df):,} events** from `{sel_ip}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",
                  ip_df["ip_address"].count())
        m2.metric("Max Risk",
                  f"{ip_df['final_risk'].max():.4f}")
        m3.metric("Threat Types",
                  ip_df[ip_df['threat_type']!='NORMAL']
                  ['threat_type'].nunique())
        m4.metric("Unique Ports",
                  int(ip_df['port'].nunique()))

        # Timeline for this IP
        ip_time = (
            ip_df.groupby(["hour","severity"])
                 .size()
                 .reset_index(name="count")
        )
        fig = px.bar(
            ip_time, x="hour", y="count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            title=f"Event Timeline — {sel_ip}",
            height=280,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4 — Model Insights ─────────────────────────────────────────────────

def render_model_insights(df: pd.DataFrame):
    st.markdown('<p class="section-header">🤖 Model Insights</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Anomaly score distribution
        fig = px.histogram(
            df, x="anomaly_score",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            nbins=60,
            title="Anomaly Score Distribution",
            labels={"anomaly_score":"Anomaly Score",
                    "count":"Frequency"},
            height=320,
            barmode="overlay",
            opacity=0.7,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk score vs anomaly score
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x="anomaly_score", y="final_risk",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            title="Anomaly Score vs Final Risk",
            opacity=0.6,
            height=320,
        )
        fig.update_layout(
            plot_bgcolor="#1e2130",
            paper_bgcolor="#1e2130",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Saved plots from Phase 4
    st.markdown('<p class="section-header">📈 Diagnostic Plots</p>',
                unsafe_allow_html=True)

    plots_dir = os.path.join(MODEL_DIR, "plots")
    plot_files = {
        "Score Distribution"  : "score_distribution.png",
        "Confusion Matrix"    : "confusion_matrix.png",
        "PCA Scatter"         : "pca_scatter.png",
        "Feature Importance"  : "feature_importance.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(plot_files.items()):
        path = os.path.join(plots_dir, fname)
        if os.path.exists(path):
            cols[i % 2].image(path, caption=title,
                              use_container_width=True)

# # ── Tab 5 — Live Alerts ────────────────────────────────────────────────────

# def render_alerts(df: pd.DataFrame):
#     st.markdown('<p class="section-header">🚨 Live Alert Feed</p>',
#                 unsafe_allow_html=True)

#     alerts = load_alerts()
#     if not alerts:
#         st.info("No alerts found. Run the pipeline first.")
#         return

#     # Summary banner
#     critical_alerts = [a for a in alerts if a.get("severity") == "CRITICAL"]
#     high_alerts     = [a for a in alerts if a.get("severity") == "HIGH"]

#     b1, b2, b3 = st.columns(3)
#     b1.metric("🔴 Critical Alerts", len(critical_alerts))
#     b2.metric("🟠 High Alerts",     len(high_alerts))
#     b3.metric("📋 Total Alerts",    len(alerts))

#     st.markdown("---")

#     # Render each alert as a styled card
#     st.markdown("**Most Recent Alerts (Top 20)**")
#     for alert in alerts[:20]:
#         sev   = alert.get("severity","")
#         color = SEVERITY_COLORS.get(sev, "#888")
#         icon  = {"CRITICAL":"🔴","HIGH":"🟠",
#                  "MEDIUM":"🟡","LOW":"🔵"}.get(sev, "⚪")

#         st.markdown(f"""
#         <div style="
#             border-left: 4px solid {color};
#             background : #1e2130;
#             padding    : 12px 16px;
#             border-radius: 5px;
#             margin-bottom: 8px;
#         ">
#             <b>{icon} [{sev}] {alert.get('threat_type','')}</b>
#             &nbsp;&nbsp;
#             <code>{alert.get('ip_address','')}</code>
#             &nbsp;&nbsp;
#             <span style="color:#888;font-size:12px">
#                 {str(alert.get('timestamp',''))[:19]}
#             </span>
#             <br>
#             <span style="font-size:13px;color:#ccc">
#                 {alert.get('description','')}
#             </span>
#             <br>
#             <small style="color:#888">
#                 Risk: <b style="color:{color}">
#                     {alert.get('final_risk',0):.4f}
#                 </b>
#                 &nbsp;|&nbsp;
#                 Confidence: {alert.get('confidence',0):.3f}
#             </small>
#         </div>
#         """, unsafe_allow_html=True)
def render_alerts(df: pd.DataFrame):
    st.markdown('<p class="section-header">🚨 Live Alert Feed</p>',
                unsafe_allow_html=True)

    alerts = load_alerts()
    if not alerts:
        st.info("No alerts found. Run the pipeline first.")
        return

    critical_alerts = [a for a in alerts if a.get("severity") == "CRITICAL"]
    high_alerts     = [a for a in alerts if a.get("severity") == "HIGH"]

    b1, b2, b3 = st.columns(3)
    b1.metric("🔴 Critical Alerts", len(critical_alerts))
    b2.metric("🟠 High Alerts",     len(high_alerts))
    b3.metric("📋 Total Alerts",    len(alerts))

    st.markdown("---")
    st.markdown("**Most Recent Alerts (Top 20)**")

    for alert in alerts[:20]:
        sev   = alert.get("severity", "")
        color = SEVERITY_COLORS.get(sev, "#888")
        icon  = {"CRITICAL": "🔴", "HIGH": "🟠",
                 "MEDIUM": "🟡", "LOW": "🔵"}.get(sev, "⚪")

        # ✅ Fixed: use correct field names from the JSON
        timestamp   = str(alert.get("first_seen", alert.get("generated_at", "")))[:19]
        avg_risk    = alert.get("avg_risk", 0)
        max_risk    = alert.get("max_risk", 0)
        ports       = ", ".join(str(p) for p in alert.get("ports_targeted", []))
        actions     = ", ".join(alert.get("actions_seen", []))
        escalation  = alert.get("escalation", "")

        st.markdown(f"""
        <div style="
            border-left  : 4px solid {color};
            background   : #1e2130;
            padding      : 12px 16px;
            border-radius: 5px;
            margin-bottom: 8px;
        ">
            <b>{icon} [{sev}] {alert.get('threat_type', '')}</b>
            &nbsp;&nbsp;
            <code>{alert.get('ip_address', '')}</code>
            &nbsp;&nbsp;
            <span style="color:#888;font-size:12px">{timestamp}</span>
            <br>
            <span style="font-size:13px;color:#ccc">
                {alert.get('description', '')}
            </span>
            <br>
            <small style="color:#888">
                Avg Risk: <b style="color:{color}">{avg_risk:.4f}</b>
                &nbsp;|&nbsp;
                Max Risk: <b style="color:{color}">{max_risk:.4f}</b>
                &nbsp;|&nbsp;
                Escalation: <b>{escalation}</b>
            </small>
            <br>
            <small style="color:#666">
                Ports: {ports} &nbsp;|&nbsp; Actions: {actions}
            </small>
        </div>
        """, unsafe_allow_html=True)
# ── Main app ───────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <h1 style='text-align:center; color:#00d4ff;'>
        🛡️ AI-Based Log Analysis &amp; Threat Detection
    </h1>
    <p style='text-align:center; color:#888; margin-top:-10px;'>
        Powered by Isolation Forest + Rule-Based Engine
    </p>
    <hr style='border-color:#2e3250;'>
    """, unsafe_allow_html=True)

    # Load & filter
    df = load_data()
    filtered = render_sidebar(df)

    # KPI cards
    render_kpis(filtered)
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🚨 Threat Log",
        "🌐 IP Analysis",
        "🤖 Model Insights",
        "🔔 Alerts",
    ])

    with tab1:
        render_overview(filtered)
    with tab2:
        render_threat_log(filtered)
    with tab3:
        render_ip_analysis(filtered)
    with tab4:
        render_model_insights(filtered)
    with tab5:
        render_alerts(filtered)

    # Footer
    st.markdown("""
    <hr style='border-color:#2e3250; margin-top:40px;'>
    <p style='text-align:center; color:#555; font-size:12px;'>
        AI Threat Detection Dashboard &nbsp;|&nbsp;
        Built with Streamlit + Isolation Forest &nbsp;|&nbsp;
        Refresh every 60s
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
