from __future__ import annotations

import json
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.alerts import send_alert_email
from app.auth import login_form
from app.config import settings
from app.database import FraudTransaction, SessionLocal, init_db
from app.etl import load_data, transform_preprocessor
from app.models import load_artifacts, risk_score


def hex_to_rgba(hex_color: str, alpha: float = 0.14) -> str:
    hex_clean = hex_color.lstrip("#")
    if len(hex_clean) == 8:
        hex_clean = hex_clean[:6]
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon=":rotating_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #050b18; color: #e2e8f0; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    .ffd-hero {
      border-radius: 20px; padding: 22px 28px; margin-bottom: 18px;
      background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #071220 100%);
      border: 1px solid rgba(59,130,246,0.25); position: relative; overflow: hidden;
    }
    .ffd-hero::before {
      content:''; position:absolute; top:-60px; right:-60px;
      width:280px; height:280px;
      background:radial-gradient(circle,rgba(59,130,246,0.18) 0%,transparent 70%);
      border-radius:50%;
    }
    .ffd-title { font-size:1.55rem; font-weight:800; letter-spacing:-0.02em; margin:0; color:#f8fafc; }
    .ffd-subtitle { margin:10px 0 0 0; opacity:0.78; font-size:1rem; color:#cbd5e1; line-height:1.6; max-width:900px; }
    .ffd-title { font-size:2.15rem; }
    .ffd-badges { margin-top:12px; display:flex; gap:8px; flex-wrap:wrap; }
    .ffd-badge {
      font-size:0.75rem; font-family:'Space Mono',monospace;
      padding:5px 12px; border-radius:999px;
      border:1px solid rgba(59,130,246,0.35); background:rgba(59,130,246,0.10); color:#93c5fd;
    }

    .metric-card {
      background:linear-gradient(135deg,#0d1f3c 0%,#0a1628 100%);
      border:1px solid rgba(59,130,246,0.20); border-radius:16px; padding:18px 20px; margin-bottom:4px;
    }
    .metric-card .label { font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-family:'Space Mono',monospace; }
    .metric-card .value { font-size:2rem; font-weight:800; color:#f8fafc; margin:4px 0; }
    .metric-card .delta { font-size:0.82rem; }
    .metric-card .delta.up { color:#10b981; }
    .metric-card .delta.down { color:#ef4444; }

    .risk-badge {
      display:inline-flex; align-items:center; gap:8px;
      padding:10px 20px; border-radius:12px; font-weight:700;
      font-size:1.05rem; font-family:'Space Mono',monospace;
    }
    .risk-critical { background:rgba(239,68,68,0.15); border:2px solid #ef4444; color:#fca5a5; }
    .risk-high     { background:rgba(249,115,22,0.15); border:2px solid #f97316; color:#fdba74; }
    .risk-medium   { background:rgba(245,158,11,0.15); border:2px solid #f59e0b; color:#fcd34d; }
    .risk-low      { background:rgba(16,185,129,0.15); border:2px solid #10b981; color:#6ee7b7; }

    .explain-card {
      background:#0d1f3c; border-radius:14px;
      padding:16px 20px; margin:8px 0; border-left:4px solid;
    }
    .explain-card.danger  { border-left-color:#ef4444; }
    .explain-card.warning { border-left-color:#f59e0b; }
    .explain-card.success { border-left-color:#10b981; }
    .explain-card h4 { margin:0 0 6px 0; font-size:0.92rem; font-weight:700; color:#f8fafc; }
    .explain-card p  { margin:0; font-size:0.85rem; color:#94a3b8; line-height:1.5; }

    .live-ticker {
      background:#0a1628; border:1px solid rgba(59,130,246,0.20); border-radius:12px;
      padding:10px 16px; font-family:'Space Mono',monospace; font-size:0.78rem;
      display:flex; align-items:center; gap:12px; margin-bottom:10px; color:#94a3b8;
    }
    .live-dot {
      width:8px; height:8px; border-radius:50%; background:#10b981;
      display:inline-block; animation:pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
      0%,100%{opacity:1;transform:scale(1);}
      50%{opacity:0.4;transform:scale(0.7);}
    }

    .fraud-record-card {
      background:linear-gradient(135deg,#0d1f3c 0%,#0a1628 100%);
      border:1px solid rgba(59,130,246,0.18); border-radius:14px;
      padding:14px 18px; margin-bottom:8px;
      display:flex; justify-content:space-between; align-items:center;
    }
    .fraud-record-card.high-risk { border-left:4px solid #ef4444; }
    .fraud-record-card.med-risk  { border-left:4px solid #f59e0b; }
    .fraud-record-card.low-risk  { border-left:4px solid #10b981; }

    .section-title {
      font-size:1.05rem; font-weight:700; color:#f8fafc;
      margin:20px 0 12px 0; display:flex; align-items:center; gap:8px;
    }
    .section-title::after {
      content:''; flex:1; height:1px;
      background:linear-gradient(90deg,rgba(59,130,246,0.30),transparent);
    }

    .ffd-auth-wrap { max-width:960px; margin:0 auto; }
    .ffd-auth-card {
      border-radius:20px; padding:20px 24px;
      border:1px solid rgba(59,130,246,0.22); background:#0a1628;
    }

    div.stButton > button {
      border-radius:10px; padding:0.65rem 0.9rem; font-weight:700;
      font-family:'Syne',sans-serif;
      background:linear-gradient(135deg,#1d4ed8,#1e40af);
      border:none; color:#fff; transition:all 0.2s;
    }
    div.stButton > button:hover { opacity:0.88; }
    .stTabs [data-baseweb="tab-list"] { gap:12px; background:transparent; }
    .stTabs [data-baseweb="tab"] {
      border-radius:10px; padding:10px 18px; font-weight:600;
      background:rgba(59,130,246,0.08); color:#94a3b8;
      border:1px solid rgba(59,130,246,0.18);
    }
    .stTabs [aria-selected="true"] {
      background:rgba(59,130,246,0.22)!important;
      color:#93c5fd!important; border-color:rgba(59,130,246,0.45)!important;
    }
    .stSidebar { background:#070f1e!important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Layout tokens ────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,31,60,0.5)",
    font=dict(color="#94a3b8", family="Syne"),
    xaxis=dict(gridcolor="rgba(59,130,246,0.12)", zerolinecolor="rgba(59,130,246,0.15)"),
    yaxis=dict(gridcolor="rgba(59,130,246,0.12)", zerolinecolor="rgba(59,130,246,0.15)"),
)
PLOTLY_LIGHT = dict(
    paper_bgcolor="rgba(255,255,255,0.97)",
    plot_bgcolor="rgba(255,255,255,0.95)",
    font=dict(color="#0f172a", family="Syne"),
    xaxis=dict(gridcolor="rgba(226,232,240,0.8)", zerolinecolor="rgba(226,232,240,0.9)"),
    yaxis=dict(gridcolor="rgba(226,232,240,0.8)", zerolinecolor="rgba(226,232,240,0.9)"),
)


def render_theme_css(theme: str) -> None:
    if theme == "Light":
        st.markdown(
            """
            <style>
            body, .stApp { background: #f8fafc !important; color: #0f172a !important; }
            .block-container { background: #f8fafc !important; color: #0f172a !important; }
            .stSidebar, .css-1d391kg { background: #e2e8f0 !important; }
            .ffd-hero { background: linear-gradient(135deg, #e2e7f2 0%, #dbeafe 50%, #eef2ff 100%) !important; border:1px solid rgba(37,99,235,0.25) !important; }
            .ffd-title { color: #0f172a !important; }
            .ffd-subtitle { color: #475569 !important; }
            .ffd-badge { background: rgba(59,130,246,0.12) !important; color: #1d4ed8 !important; border-color: rgba(59,130,246,0.2) !important; }
            .metric-card, .explain-card, .fraud-record-card { background: #ffffff !important; border-color: rgba(148,163,184,0.3) !important; color:#0f172a !important; }
            .live-ticker { background: #ffffff !important; color:#0f172a !important; border-color: rgba(37,99,235,0.18) !important; }
            .risk-badge { background: rgba(224,242,254,0.55) !important; color:#0f172a !important; }
            .risk-critical { background: rgba(254,226,226,0.65) !important; color:#b91c1c !important; border-color:#f87171 !important; }
            .risk-high { background: rgba(255,247,237,0.65) !important; color:#c2410c !important; border-color:#fb923c !important; }
            .risk-medium { background: rgba(254,240,138,0.5) !important; color:#b45309 !important; border-color:#f59e0b !important; }
            .risk-low { background: rgba(220,252,231,0.75) !important; color:#047857 !important; border-color:#10b981 !important; }
            .section-title { color:#0f172a !important; }
            .delta.up { color:#16a34a !important; }
            .delta.down { color:#dc2626 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body, .stApp { background: #050b18 !important; color: #e2e8f0 !important; }
            .block-container { background: #050b18 !important; color: #e2e8f0 !important; }
            .ffd-hero { background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #071220 100%) !important; }
            .ffd-title { color:#f8fafc !important; }
            .ffd-subtitle { color:#94a3b8 !important; }
            .ffd-badge { background: rgba(59,130,246,0.10) !important; color:#93c5fd !important; }
            .metric-card, .explain-card, .fraud-record-card { background: linear-gradient(135deg,#0d1f3c 0%,#0a1628 100%) !important; border-color: rgba(59,130,246,0.20) !important; color:#e2e8f0 !important; }
            .live-ticker { background:#0a1628 !important; color:#94a3b8 !important; }
            .risk-badge { color:#f8fafc !important; }
            .section-title { color:#f8fafc !important; }
            .delta.up { color:#10b981 !important; }
            .delta.down { color:#ef4444 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.session_state["app_theme"] = theme


def get_plotly_style():
    return PLOTLY_DARK if st.session_state.get("app_theme", "Dark") == "Dark" else PLOTLY_LIGHT

# ─── Market signal constants ──────────────────────────────────────────────────
FRAUD_CATEGORIES = [
    "Card-Not-Present", "Account Takeover", "Synthetic Identity",
    "Phishing", "Money Mule", "Merchant Fraud", "Wire Fraud",
    "Cheque Fraud", "ATM Skimming", "Insider Threat",
]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"]
SIMULATED_TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
SIMULATED_REASONS = [
    "Unusual merchant pattern", "High-value transfer", "New beneficiary", "Multiple quick withdrawals",
    "Rapid balance drain", "Remote login behavior", "Suspicious destination account",
]
SIMULATED_SOURCE_NAMES = [
    "eCommerce", "Retail POS", "Online Transfer", "ATM Withdrawal", "Mobile Payment",
]


def get_risk_level_label(score: float) -> str:
    if score >= 85:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def generate_spike_transaction() -> dict:
    tx = generate_simulated_transaction()
    tx["amount"] = max(tx["amount"], float(np.random.uniform(120_000, 240_000)))
    tx["fraud_probability"] = float(min(1.0, tx["fraud_probability"] + 0.18))
    tx["anomaly_score"] = float(min(1.0, tx["anomaly_score"] + 0.22))
    tx["risk_score"] = float(min(100.0, tx["fraud_probability"] * 80 + np.tanh(tx["anomaly_score"]) * 20 + 18))
    tx["risk_level"] = get_risk_level_label(tx["risk_score"])
    tx["source"] = "Threat Intelligence"
    tx["reason"] = "Simulated high-risk alert"
    return tx


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def hero_header():
    ts = datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
    st.markdown(
        f"""
        <div class="ffd-hero">
                    <h1 class="ffd-title">⚡ Financial Fraud Detection &amp; Real-Time Monitoring</h1>
                    <p class="ffd-subtitle">Interactive fraud intelligence dashboard with live simulation, risk scoring, explainable predictions, and operational reporting.</p>
          <div class="ffd-badges">
            <span class="ffd-badge">ML Scoring</span>
            <span class="ffd-badge">Anomaly Detection</span>
            <span class="ffd-badge">Live Market</span>
            <span class="ffd-badge">Alerts &amp; Reports</span>
            <span class="ffd-badge">{ts}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_sample_data():
    return load_data(str(settings.data_file), nrows=50_000)


@st.cache_resource(show_spinner=True)
def get_artifacts():
    return load_artifacts(settings.artifacts_dir)


def check_artifacts_available() -> bool:
    required = [
        "classifier.joblib",
        "feature_columns.joblib",
        "imputer.joblib",
        "scaler.joblib",
    ]
    missing = [name for name in required if not (settings.artifacts_dir / name).exists()]
    if missing:
        st.error(
            "Missing ML artifacts for prediction."
            f" The files {', '.join(missing)} were not found in `{settings.artifacts_dir}`."
        )
        st.info(
            "Generate the trained model artifacts by running `python -m app.train_models` "
            "from the project root, then refresh this page."
        )
        return False
    return True


def run_single_prediction(payload: dict):
    if not check_artifacts_available():
        raise FileNotFoundError("Model artifacts are required for prediction.")
    model, feature_cols, imputer, scaler = get_artifacts()
    source = pd.DataFrame([payload])
    source["type"] = source["type"].fillna("PAYMENT")
    processed = transform_preprocessor(source, imputer=imputer, scaler=scaler, feature_columns=feature_cols)
    prob = float(model.predict_proba(processed)[0, 1])
    anomaly = 0.0
    risk = float(risk_score(np.array([prob]))[0])
    return prob, anomaly, risk


def run_batch_predictions(batch_df: pd.DataFrame) -> pd.DataFrame:
    if not check_artifacts_available():
        raise FileNotFoundError("Model artifacts are required for batch prediction.")
    model, feature_cols, imputer, scaler = get_artifacts()
    source = batch_df.copy()
    source["type"] = source["type"].fillna("PAYMENT")
    processed = transform_preprocessor(source, imputer=imputer, scaler=scaler, feature_columns=feature_cols)
    probs = model.predict_proba(processed)[:, 1]
    anomalies = np.zeros_like(probs, dtype=float)
    risks = risk_score(probs)
    output = batch_df.copy()
    output["fraud_probability"] = probs
    output["anomaly_score"] = anomalies
    output["risk_score"] = risks
    output["risk_level"] = output["risk_score"].apply(get_risk_level_label)
    return output


def save_prediction(payload, fraud_probability, anomaly_score, risk, alert_sent):
    session = SessionLocal()
    try:
        rec = FraudTransaction(
            step=int(payload["step"]),
            tx_type=str(payload["type"]),
            amount=float(payload["amount"]),
            oldbalance_org=float(payload["oldbalanceOrg"]),
            newbalance_orig=float(payload["newbalanceOrig"]),
            oldbalance_dest=float(payload["oldbalanceDest"]),
            newbalance_dest=float(payload["newbalanceDest"]),
            is_fraud=fraud_probability >= 0.5,
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
            risk_score=risk,
            alert_sent=alert_sent,
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()


def fetch_live_records(limit: int = 500, window_minutes: int = 30) -> pd.DataFrame:
    session = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        records = (
            session.query(FraudTransaction)
            .filter(FraudTransaction.created_at >= cutoff)
            .order_by(FraudTransaction.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "id": r.id,
            "created_at": r.created_at,
            "tx_type": r.tx_type,
            "amount": float(r.amount),
            "oldbalance_org": float(r.oldbalance_org),
            "newbalance_orig": float(r.newbalance_orig),
            "oldbalance_dest": float(r.oldbalance_dest),
            "newbalance_dest": float(r.newbalance_dest),
            "is_fraud": bool(r.is_fraud),
            "fraud_probability": float(r.fraud_probability),
            "anomaly_score": float(r.anomaly_score),
            "risk_score": float(r.risk_score),
            "alert_sent": bool(r.alert_sent),
            "step": r.step,
        }
        for r in records
    ]
    return pd.DataFrame(rows).sort_values("created_at", ascending=True).reset_index(drop=True)


def fetch_all_records(limit: int = 2000) -> pd.DataFrame:
    session = SessionLocal()
    try:
        records = (
            session.query(FraudTransaction)
            .order_by(FraudTransaction.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "id": r.id, "created_at": r.created_at, "tx_type": r.tx_type,
            "amount": r.amount, "is_fraud": bool(r.is_fraud),
            "fraud_probability": float(r.fraud_probability),
            "anomaly_score": float(r.anomaly_score),
            "risk_score": float(r.risk_score),
            "alert_sent": bool(r.alert_sent), "step": r.step,
            "oldbalance_org": float(r.oldbalance_org),
            "newbalance_orig": float(r.newbalance_orig),
        }
        for r in records
    ]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Real-time market signal generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_live_market_signals() -> dict:
    """Generates deterministic-but-shifting signals based on current 30-s bucket."""
    seed = int(time.time() // 30)
    rng = np.random.default_rng(seed)

    fraud_index = int(42 + rng.integers(-8, 9))
    prev_index  = int(42 + rng.integers(-8, 9))

    category_vols = {c: int(rng.integers(120, 980)) for c in FRAUD_CATEGORIES}
    region_risk   = {r: int(rng.integers(20, 91))   for r in REGIONS}

    anomaly_bps = round(float(rng.uniform(3.5, 18.5)), 2)
    prev_bps    = round(float(rng.uniform(3.5, 18.5)), 2)
    alerts_30m  = int(rng.integers(0, 43))
    latency_ms  = int(rng.integers(14, 121))

    top_threat = FRAUD_CATEGORIES[int(rng.integers(0, len(FRAUD_CATEGORIES)))]

    trend_vals  = list(np.cumsum(rng.normal(0, 1.2, 60)).clip(-15, 25) + 38)
    trend_times = [
        (datetime.utcnow() - timedelta(minutes=59 - i)).strftime("%H:%M")
        for i in range(60)
    ]

    cp_pct = float(rng.uniform(28, 60))
    vectors = {
        "Automated Bots": int(rng.integers(20, 46)),
        "Human Operators": int(rng.integers(15, 36)),
        "Compromised Devices": int(rng.integers(10, 31)),
        "Insider Threat": int(rng.integers(2, 16)),
        "Third-Party Breach": int(rng.integers(5, 21)),
    }

    return dict(
        fraud_index=fraud_index, prev_fraud_index=prev_index,
        category_vols=category_vols, region_risk=region_risk,
        anomaly_bps=anomaly_bps, prev_bps=prev_bps,
        alerts_30m=alerts_30m, latency_ms=latency_ms,
        top_threat=top_threat, trend_vals=trend_vals, trend_times=trend_times,
        cp_pct=cp_pct, vectors=vectors,
    )


def generate_simulated_transaction() -> dict:
    seed = int(time.time() // 10)
    rng = np.random.default_rng(seed + int(time.time()) % 10)
    tx_type = rng.choice(SIMULATED_TX_TYPES, p=[0.35, 0.2, 0.2, 0.15, 0.1])
    amount = float(rng.normal(35_000, 45_000))
    amount = max(120.0, min(amount, 250_000.0))
    if tx_type in ("TRANSFER", "CASH_OUT"):
        amount = max(amount, float(rng.uniform(18_000, 220_000)))
    oldbalance_org = float(max(0.0, amount * rng.uniform(1.0, 3.5)))
    newbalance_org = float(max(0.0, oldbalance_org - amount * rng.uniform(0.75, 1.0)))
    oldbalance_dest = float(rng.uniform(0.0, 12_000.0))
    newbalance_dest = float(oldbalance_dest + amount * rng.uniform(0.7, 1.0))
    fraud_probability = float(min(1.0, rng.beta(2.5, 3.8) + (0.12 if tx_type in ("TRANSFER", "CASH_OUT") else 0.0) + (0.08 if amount > 80_000 else 0.0)))
    anomaly_score = float(min(1.0, rng.normal(0.22, 0.18) + 0.05 * (fraud_probability > 0.45)))
    risk_score_val = float(min(100.0, fraud_probability * 70 + anomaly_score * 30 + (10 if amount > 120_000 else 0)))
    return {
        "timestamp": datetime.utcnow() - timedelta(seconds=int(rng.integers(0, 180))),
        "source": rng.choice(SIMULATED_SOURCE_NAMES),
        "type": tx_type,
        "reason": rng.choice(SIMULATED_REASONS),
        "amount": amount,
        "fraud_probability": fraud_probability,
        "anomaly_score": anomaly_score,
        "risk_score": risk_score_val,
        "risk_level": ("Critical" if risk_score_val >= 85 else "High" if risk_score_val >= 65 else "Medium" if risk_score_val >= 40 else "Low"),
    }


def get_simulated_feed(max_records: int = 8) -> pd.DataFrame:
    if "simulator_history" not in st.session_state:
        st.session_state.simulator_history = []
    intensity = st.session_state.get("simulator_intensity", 3)
    inject_spike = st.session_state.get("simulator_inject_spike", False)
    new_records = [generate_simulated_transaction() for _ in range(max(1, intensity))]
    if inject_spike:
        new_records.extend([generate_spike_transaction() for _ in range(2)])
        st.session_state.simulator_inject_spike = False
    st.session_state.simulator_history = (st.session_state.simulator_history + new_records)[-max_records:]
    df = pd.DataFrame(st.session_state.simulator_history)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MARKET ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def live_analytics_section(df_db: pd.DataFrame, refresh_sec: int):
    st_autorefresh(interval=refresh_sec * 1000, key="live_analytics_refresh")
    sig = generate_live_market_signals()

    theme = st.session_state.get("app_theme", "Dark")
    ticker_bg = "#ffffff" if theme == "Light" else "#0a1628"
    ticker_text = "#0f172a" if theme == "Light" else "#f8fafc"
    detail_text = "#1e293b" if theme == "Light" else "#94a3b8"

    delta_idx = sig["fraud_index"] - sig["prev_fraud_index"]
    arrow = "▲" if delta_idx >= 0 else "▼"
    idx_color = "#ef4444" if delta_idx >= 0 else "#10b981"

    st.markdown(
        f"""
        <div class="live-ticker" style="background:{ticker_bg}; color:{detail_text}; border-color: rgba(37,99,235,0.18);">
          <span class="live-dot"></span>
          <strong style="color:{ticker_text}">LIVE MARKET SIGNALS</strong>
          &nbsp;|&nbsp; Fraud Index: <span style="color:{idx_color}">{sig['fraud_index']} {arrow}{abs(delta_idx)}</span>
          &nbsp;|&nbsp; Anomaly Rate: <span style="color:#fcd34d">{sig['anomaly_bps']} bps</span>
          &nbsp;|&nbsp; Alerts (30m): <span style="color:#f97316">{sig['alerts_30m']}</span>
          &nbsp;|&nbsp; Top Threat: <span style="color:#c084fc">{sig['top_threat']}</span>
          &nbsp;|&nbsp; Latency: {sig['latency_ms']}ms
          &nbsp;|&nbsp; {datetime.utcnow().strftime('%H:%M:%S')} UTC
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:12px 14px; border-radius:14px; background:rgba(59,130,246,0.08); '
        'border:1px solid rgba(59,130,246,0.18); margin-bottom:20px;'>
        '<strong>Simulator mode:</strong> This dashboard uses a local simulated fraud feed for live observation without external Kafka or streaming services.</div>',
        unsafe_allow_html=True,
    )

    control_col, spike_col = st.columns([4, 1])
    with control_col:
        intensity = st.slider("Simulation intensity", 1, 8, 3,
                              help="Higher intensity generates more simulated events in the live feed.")
        st.session_state["simulator_intensity"] = intensity
    with spike_col:
        if st.button("Inject High-Risk Spike", key="inject_spike"):
            st.session_state["simulator_inject_spike"] = True

    sim_df = get_simulated_feed(max_records=12)

    st.markdown(
        '<div style="padding:12px 14px; border-radius:14px; background:rgba(16,185,129,0.08); '
        'border:1px solid rgba(16,185,129,0.18); margin-bottom:18px;'>
        '<strong>Live Simulator:</strong> This tab generates a simulated real-time fraud feed for understanding market activity and risk signals.</div>',
        unsafe_allow_html=True,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Global Fraud Index", f"{sig['fraud_index']}", "/100",
         delta_idx, True),
        (c2, "Anomaly Rate", f"{sig['anomaly_bps']}", " bps",
         sig["anomaly_bps"] - sig["prev_bps"], True),
        (c3, "Alerts Fired (30m)", f"{sig['alerts_30m']}", "", None, None),
        (c4, "CNP Transaction %", f"{sig['cp_pct']:.1f}", "%", None, None),
        (c5, "Avg API Latency", f"{sig['latency_ms']}", " ms", None, None),
    ]
    for col, label, val, suffix, delta, higher_is_bad in kpis:
        if delta is not None:
            d_sign = "+" if delta >= 0 else ""
            d_cls = ("down" if (higher_is_bad and delta > 0) else
                     "up"   if (higher_is_bad and delta <= 0) else "up")
            delta_html = f'<div class="delta {d_cls}">{d_sign}{delta:.2f} vs prev</div>'
        else:
            delta_html = ""
        col.markdown(
            f"""<div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val}{suffix}</div>
              {delta_html}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── 60-min rolling trend ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Real-Time Fraud Trend (60-min rolling)</div>',
                unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=sig["trend_times"], y=sig["trend_vals"],
        mode="lines", line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
    ))
    fig_trend.add_hline(y=50, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                        annotation_text="High Risk Threshold",
                        annotation_font_color="#fca5a5")
    fig_trend.update_layout(
        **get_plotly_style(), height=240, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, xaxis_title="Time (UTC)", yaxis_title="Fraud Index",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(
        '<div style="padding:12px 14px; border-radius:14px; background:rgba(16,185,129,0.08); '
        'border:1px solid rgba(16,185,129,0.18); margin-bottom:18px;'>
        '<strong>Live Simulator:</strong> This tab generates a simulated real-time fraud feed for understanding market activity and risk signals.</div>',
        unsafe_allow_html=True,
    )

    # ── Category + Region ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">🎯 Fraud by Category (Live)</div>', unsafe_allow_html=True)
        cat_df = pd.DataFrame({
            "Category": list(sig["category_vols"].keys()),
            "Volume": list(sig["category_vols"].values()),
        }).sort_values("Volume", ascending=True)
        fig_cat = go.Figure(go.Bar(
            x=cat_df["Volume"], y=cat_df["Category"], orientation="h",
            marker=dict(color=cat_df["Volume"],
                        colorscale=[[0, "#1d4ed8"], [0.5, "#7c3aed"], [1, "#ef4444"]]),
        ))
        fig_cat.update_layout(**get_plotly_style(), height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">🌍 Regional Risk Scores</div>', unsafe_allow_html=True)
        reg_df = pd.DataFrame({
            "Region": list(sig["region_risk"].keys()),
            "Risk": list(sig["region_risk"].values()),
        }).sort_values("Risk", ascending=False)
        fig_reg = go.Figure(go.Bar(
            x=reg_df["Region"], y=reg_df["Risk"],
            marker=dict(color=reg_df["Risk"],
                        colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]]),
        ))
        fig_reg.update_layout(**get_plotly_style(), height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_reg, use_container_width=True)

    # ── Attack vectors + CNP donut ────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">⚔️ Attack Vector Breakdown</div>', unsafe_allow_html=True)
        vec_df = pd.DataFrame({
            "Vector": list(sig["vectors"].keys()),
            "Share": list(sig["vectors"].values()),
        })
        fig_vec = go.Figure(go.Pie(
            labels=vec_df["Vector"], values=vec_df["Share"], hole=0.55,
            marker=dict(colors=["#3b82f6", "#7c3aed", "#ef4444", "#f59e0b", "#10b981"]),
        ))
        fig_vec.update_layout(**get_plotly_style(), height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_vec, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">💳 Card-Present vs Card-Not-Present</div>',
                    unsafe_allow_html=True)
        cnp, cp = sig["cp_pct"], 100 - sig["cp_pct"]
        fig_cp = go.Figure(go.Pie(
            labels=["Card-Not-Present", "Card-Present"], values=[cnp, cp], hole=0.6,
            marker=dict(colors=["#ef4444", "#10b981"]),
        ))
        fig_cp.update_layout(**get_plotly_style(), height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cp, use_container_width=True)

    # ── DB overlay ────────────────────────────────────────────────────────────
    if not df_db.empty:
        st.markdown('<div class="section-title">🗄️ Your Predictions (Live Window)</div>',
                    unsafe_allow_html=True)
        df_db["created_at"] = pd.to_datetime(df_db["created_at"])
        total_db = len(df_db)
        fraud_db = int(df_db["is_fraud"].sum())
        avg_rs   = df_db["risk_score"].mean()

        dc1, dc2, dc3 = st.columns(3)
        for col, lbl, val in [
            (dc1, "Live Predictions", f"{total_db}"),
            (dc2, "Fraud Flagged",    f"{fraud_db}"),
            (dc3, "Avg Risk Score",   f"{avg_rs:.1f}/100"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="label">{lbl}</div>'
                f'<div class="value">{val}</div></div>',
                unsafe_allow_html=True,
            )

        col_e, col_f = st.columns(2)
        with col_e:
            fig_rs = go.Figure(go.Histogram(
                x=df_db["risk_score"], nbinsx=20,
                marker=dict(color="#3b82f6", line=dict(color="#1d4ed8", width=1)),
            ))
            fig_rs.update_layout(**get_plotly_style(), title="Risk Score Distribution (DB)",
                                 height=260, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_rs, use_container_width=True)

        with col_f:
            if len(df_db) > 1:
                freq = (df_db.set_index("created_at")["id"]
                        .resample("1min").count()
                        .reset_index(name="count"))
                fig_freq = go.Figure(go.Scatter(
                    x=freq["created_at"], y=freq["count"],
                    mode="lines+markers", line=dict(color="#10b981", width=2),
                ))
                fig_freq.update_layout(**get_plotly_style(), title="Prediction Frequency / min",
                                       height=260, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Submit predictions via 'Risk Prediction' to overlay your data here.")

    top_alerts = sim_df[sim_df["risk_score"] >= 65].sort_values("risk_score", ascending=False).head(4)
    st.markdown('<div class="section-title">🚨 Simulated Live Transaction Feed</div>', unsafe_allow_html=True)
    alert_count = int((sim_df["risk_score"] >= 65).sum())
    st.markdown(
        f'<div class="metric-card"><div class="label">Simulated Alerts</div>'
        f'<div class="value">{alert_count}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card"><div class="label">Latest Simulator Value</div>'
        f'<div class="value">{sim_df["risk_score"].mean():.1f}/100</div></div>',
        unsafe_allow_html=True,
    )
    if not top_alerts.empty:
        st.markdown('<div style="margin-top:10px;margin-bottom:10px"><strong>Top simulated high-risk events</strong></div>', unsafe_allow_html=True)
        for _, row in top_alerts.iterrows():
            st.markdown(
                f'<div class="fraud-record-card high-risk">'
                f'<div><strong>{row["type"]}</strong> · {row["source"]}</div>'
                f'<div>${row["amount"]:,.0f}</div>'
                f'<div><span style="color:#fca5a5">{row["risk_level"]}</span> · {row["fraud_probability"]:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="section-title">🚨 Real-Time Fraud Simulator</div>', unsafe_allow_html=True)
    if sim_df.empty:
        st.write("No live simulator records available yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="label">Simulated Events</div>'
        f'<div class="value">{len(sim_df)}</div></div>', unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-card"><div class="label">High-Risk Alerts</div>'
        f'<div class="value">{int((sim_df["risk_score"] >= 65).sum())}</div></div>', unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-card"><div class="label">Avg Risk Score</div>'
        f'<div class="value">{sim_df["risk_score"].mean():.1f}/100</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Simulator Trend & Live Feed</div>', unsafe_allow_html=True)
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(
        x=sim_df["timestamp"].dt.strftime("%H:%M:%S"),
        y=sim_df["risk_score"],
        marker_color=["#ef4444" if s >= 85 else "#f97316" if s >= 65 else "#f59e0b" if s >= 40 else "#10b981" for s in sim_df["risk_score"]],
    ))
    fig_sim.update_layout(**get_plotly_style(), height=260, margin=dict(l=0, r=0, t=30, b=0),
                           title="Simulated Risk Score Flow", xaxis_title="Time", yaxis_title="Risk Score")
    st.plotly_chart(fig_sim, use_container_width=True)

    st.dataframe(
        sim_df.rename(columns={
            "timestamp": "Time",
            "source": "Source",
            "type": "Type",
            "reason": "Trigger",
            "amount": "Amount",
            "fraud_probability": "Fraud Probability",
            "anomaly_score": "Anomaly Score",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
        })[
            ["Time", "Source", "Type", "Trigger", "Amount", "Fraud Probability", "Anomaly Score", "Risk Score", "Risk Level"]
        ].assign(**{
            "Amount": lambda df: df["Amount"].map(lambda v: f"${v:,.0f}"),
            "Fraud Probability": lambda df: df["Fraud Probability"].map(lambda v: f"{v:.1%}"),
            "Anomaly Score": lambda df: df["Anomaly Score"].map(lambda v: f"{v:.3f}"),
        }),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTION (detailed explanations)
# ═══════════════════════════════════════════════════════════════════════════════

def _risk_level(score: float) -> tuple[str, str, str]:
    if score >= 85:
        return "CRITICAL RISK", "risk-critical", "🔴"
    if score >= 65:
        return "HIGH RISK",     "risk-high",     "🟠"
    if score >= 40:
        return "MEDIUM RISK",   "risk-medium",   "🟡"
    return     "LOW RISK",      "risk-low",      "🟢"


def _explain_prediction(payload: dict, fraud_prob: float, risk: float) -> list[dict]:
    factors: list[dict] = []
    amount   = payload["amount"]
    old_org  = payload["oldbalanceOrg"]
    new_org  = payload["newbalanceOrig"]
    old_dest = payload["oldbalanceDest"]
    new_dest = payload["newbalanceDest"]
    tx_type  = payload["type"]

    # Balance drain
    if old_org > 0 and new_org == 0:
        factors.append(dict(severity="danger",
            title="🔴 Complete Balance Drain",
            text=(f"The origin account balance dropped from {old_org:,.2f} to 0. "
                  "Complete balance draining is a hallmark of account-takeover fraud — "
                  "attackers empty accounts in a single operation to maximise theft before detection.")))
    elif old_org > 0 and new_org < old_org * 0.05:
        factors.append(dict(severity="danger",
            title="🟠 Near-Complete Balance Drain",
            text=(f"Only {new_org / old_org * 100:.1f}% of the origin balance remains post-transaction. "
                  "Near-total depletion is strongly correlated with account-takeover scenarios.")))

    # Destination anomaly
    if old_dest == 0 and new_dest >= amount * 0.95:
        factors.append(dict(severity="danger",
            title="🔴 Funds Land in Zero-Balance Account",
            text=("The destination account held zero funds before receiving this transfer. "
                  "Mule accounts used for money laundering are often freshly opened and maintain "
                  "near-zero balances between transactions.")))

    # Transaction type risk
    if tx_type in ("TRANSFER", "CASH_OUT"):
        factors.append(dict(severity="warning",
            title="🟡 High-Risk Transaction Type",
            text=(f"'{tx_type}' transactions account for the vast majority of confirmed fraud cases. "
                  "These operations move funds out of the network and are the primary vectors for "
                  "money-laundering and account-takeover attacks.")))
    else:
        factors.append(dict(severity="success",
            title="🟢 Lower-Risk Transaction Type",
            text=(f"'{tx_type}' transactions carry a lower inherent fraud rate compared to "
                  "TRANSFER and CASH_OUT operations.")))

    # Large amount
    if amount > 200_000:
        factors.append(dict(severity="warning",
            title="🟡 Large Transaction Amount",
            text=(f"Transaction value of {amount:,.2f} is significantly above median legitimate amounts. "
                  "Large single-step transfers are disproportionately represented in fraud cases, "
                  "especially when combined with balance-drain patterns.")))

    # Risk factors are based on classifier output and transaction features.

    # Classifier confidence
    if fraud_prob >= 0.80:
        factors.append(dict(severity="danger",
            title="🔴 Classifier: Very High Fraud Confidence",
            text=(f"The ML classifier assigns {fraud_prob:.1%} fraud probability — very high confidence "
                  "driven by the combination of transaction type, amount, and balance patterns. "
                  "Immediate manual review or automated block is strongly recommended.")))
    elif fraud_prob >= 0.50:
        factors.append(dict(severity="warning",
            title="🟡 Classifier: Fraud Probable",
            text=(f"The classifier assigns {fraud_prob:.1%} — more characteristics in common with "
                  "fraudulent cases than legitimate ones. Enhanced monitoring or step-up authentication recommended.")))
    elif fraud_prob >= 0.25:
        factors.append(dict(severity="warning",
            title="🟡 Classifier: Borderline Suspicion",
            text=(f"Fraud probability {fraud_prob:.1%} — below the 50% decision boundary but elevated. "
                  "Some features overlap with known fraud patterns. Low-friction step-up auth may be appropriate.")))
    else:
        factors.append(dict(severity="success",
            title="🟢 Classifier: Likely Legitimate",
            text=(f"Fraud probability is only {fraud_prob:.1%}. The transaction shares most "
                  "characteristics with normal, legitimate activity. No immediate action required.")))

    return factors


def prediction_form():
    st.markdown('<div class="section-title">🔮 Transaction Risk Prediction</div>', unsafe_allow_html=True)

    if not check_artifacts_available():
        return

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    if st.button("🔄 Reset Prediction"):
        st.session_state["last_prediction"] = None
        st.rerun()

    with st.expander("📥 Upload CSV for Batch Prediction", expanded=False):
        st.markdown(
            "Upload a CSV file with columns: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`."
        )
        batch_file = st.file_uploader("Transaction CSV", type=["csv"], key="batch_upload")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
            except Exception as exc:
                st.error(f"Unable to read CSV: {exc}")
                batch_df = pd.DataFrame()
            required_cols = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
            if not batch_df.empty:
                if not required_cols.issubset(batch_df.columns):
                    missing = required_cols - set(batch_df.columns)
                    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                else:
                    with st.spinner("Scoring uploaded transactions…"):
                        results_df = run_batch_predictions(batch_df)
                    st.success("Batch prediction complete.")
                    st.dataframe(results_df.head(10), use_container_width=True)
                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download scored batch", data=csv_data,
                                       file_name="batch_fraud_predictions.csv", mime="text/csv")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        step    = col1.number_input("Step (hour)", min_value=1, value=1,
                                    help="Simulation hour — higher steps may correlate with off-hours fraud spikes.")
        tx_type = col2.selectbox("Transaction Type",
                                 ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                                 help="TRANSFER and CASH_OUT carry the highest inherent risk.")
        amount  = col3.number_input("Amount", min_value=1.0, value=10_000.0,
                                    help="Transaction value in currency units.")
        col4, col5 = st.columns(2)
        old_org = col4.number_input("Old Balance — Origin",      min_value=0.0, value=50_000.0)
        new_org = col5.number_input("New Balance — Origin",      min_value=0.0, value=30_000.0)
        col6, col7 = st.columns(2)
        old_dest = col6.number_input("Old Balance — Destination", min_value=0.0, value=0.0)
        new_dest = col7.number_input("New Balance — Destination", min_value=0.0, value=20_000.0)
        submitted = st.form_submit_button("⚡ Predict Fraud Risk", use_container_width=True)

    if submitted:
        payload = {
            "step": step, "type": tx_type, "amount": amount,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }
        with st.spinner("Running ML scoring…"):
            fraud_prob, anomaly_score, risk = run_single_prediction(payload)
        st.session_state["last_prediction"] = {
            "payload": payload,
            "fraud_prob": fraud_prob,
            "anomaly_score": anomaly_score,
            "risk": risk,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        alert_sent = False
        threshold = st.session_state.get("risk_alert_threshold", 80)
        if risk >= threshold:
            alert_sent = send_alert_email(
                "High Risk Fraud Alert",
                f"High-risk transaction at {datetime.utcnow().isoformat()} UTC\n"
                f"Risk Score: {risk:.2f}\nFraud Probability: {fraud_prob:.2%}\n"
                f"Payload: {json.dumps(payload)}",
            )
        save_prediction(payload, fraud_prob, anomaly_score, risk, alert_sent)

    last = st.session_state.get("last_prediction")
    if not last:
        st.info("Fill in the transaction details above and click **Predict Fraud Risk** to see the full analysis.")
        return

    risk_val   = last["risk"]
    fraud_prob = last["fraud_prob"]
    label, css_cls, emoji = _risk_level(risk_val)

    # ── Verdict banner ────────────────────────────────────────────────────────
    st.markdown(
        f'<br><span class="risk-badge {css_cls}">'
        f'{emoji} {label} &nbsp;|&nbsp; Score: {risk_val:.1f} / 100'
        f'</span>',
        unsafe_allow_html=True,
    )

    # ── Gauges ────────────────────────────────────────────────────────────────
    def make_gauge(val, title, max_val, suffix):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"size": 26, "color": "#f8fafc", "family": "Space Mono"}},
            title={"text": title, "font": {"size": 13, "color": "#94a3b8", "family": "Syne"}},
            gauge={
                "axis": {"range": [0, max_val], "tickfont": {"color": "#64748b"}},
                "bar": {"color": "#3b82f6", "thickness": 0.28},
                "bgcolor": "rgba(13,31,60,0)",
                "bordercolor": "rgba(59,130,246,0.25)",
                "steps": [
                    {"range": [0, max_val * 0.35], "color": "rgba(16,185,129,0.18)"},
                    {"range": [max_val * 0.35, max_val * 0.70], "color": "rgba(245,158,11,0.18)"},
                    {"range": [max_val * 0.70, max_val], "color": "rgba(239,68,68,0.22)"},
                ],
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=240, margin=dict(l=20, r=20, t=40, b=0))
        return fig

    g1, g2 = st.columns(2)
    g1.plotly_chart(make_gauge(fraud_prob * 100, "Fraud Probability", 100, "%"), use_container_width=True)
    g2.plotly_chart(make_gauge(risk_val, "Overall Risk Score", 100, "/100"), use_container_width=True)

    # ── Numeric summary cards ─────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    for col, lbl, val in [
        (m1, "Fraud Probability", f"{fraud_prob:.2%}"),
        (m2, "Risk Score",        f"{risk_val:.1f}/100"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Factor explanations ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Why This Risk Score?</div>', unsafe_allow_html=True)
    for f in _explain_prediction(last["payload"], fraud_prob, risk_val):
        st.markdown(
            f'<div class="explain-card {f["severity"]}">'
            f'<h4>{f["title"]}</h4><p>{f["text"]}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Recommended actions ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🛡️ Recommended Actions</div>', unsafe_allow_html=True)
    if risk_val >= 85:
        actions = [
            ("danger",  "🚫 Block Transaction",    "Immediately suspend this transaction pending manual review."),
            ("danger",  "📞 Contact Account Holder","Verify intent via an out-of-band channel (phone/OTP)."),
            ("danger",  "🗂️ File SAR",             "If fraud confirmed, file a Suspicious Activity Report with compliance."),
            ("warning", "🔒 Freeze Origin Account", "Temporarily freeze the origin account to prevent further unauthorized transfers."),
        ]
    elif risk_val >= 65:
        actions = [
            ("warning", "⚠️ Flag for Review",        "Route transaction to the fraud operations queue for analyst review."),
            ("warning", "🔐 Require Step-Up Auth",    "Prompt the customer with biometric or OTP challenge before proceeding."),
            ("warning", "👁️ Enhanced Monitoring",    "Flag account for heightened surveillance over the next 72 hours."),
        ]
    elif risk_val >= 40:
        actions = [
            ("success", "👀 Passive Monitoring",  "No immediate block — apply passive monitoring and log for review."),
            ("success", "📋 Audit Logging",        "Retain transaction record with elevated flags for periodic audit cycles."),
        ]
    else:
        actions = [
            ("success", "✅ Approve Transaction", "Low risk — transaction may proceed normally."),
            ("success", "📝 Routine Logging",     "Log as standard activity with no additional flags."),
        ]

    for sev, title, desc in actions:
        st.markdown(
            f'<div class="explain-card {sev}"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("📄 Raw Prediction Payload", expanded=False):
        st.json(last["payload"])
        st.caption(f"Scored at {last['created_at']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FRAUD DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def fraud_dashboard(df: pd.DataFrame):
    st.markdown('<div class="section-title">🏛️ Fraud Operations Dashboard</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No records yet — submit predictions via 'Risk Prediction' to populate the dashboard.")
        return

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["risk_level"] = df["risk_score"].apply(
        lambda s: "Critical" if s >= 85 else ("High" if s >= 65 else ("Medium" if s >= 40 else "Low"))
    )
    df["date"] = df["created_at"].dt.date

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total      = len(df)
    fraud_cnt  = int(df["is_fraud"].sum())
    high_risk  = int((df["risk_score"] >= 65).sum())
    alerts_cnt = int(df["alert_sent"].sum())
    avg_risk   = df["risk_score"].mean()
    total_amt  = df["amount"].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_data = [
        (k1, "Total Records",     f"{total:,}",            None),
        (k2, "Fraud Flagged",     f"{fraud_cnt:,}",        "down" if fraud_cnt else None),
        (k3, "High/Critical",     f"{high_risk:,}",        "down" if high_risk else None),
        (k4, "Alerts Sent",       f"{alerts_cnt:,}",       None),
        (k5, "Avg Risk Score",    f"{avg_risk:.1f}/100",   None),
        (k6, "Total Amount",      f"${total_amt:,.0f}",    None),
    ]
    for col, lbl, val, d_cls in kpi_data:
        d_html = f'<div class="delta {d_cls}">⚠️ Elevated</div>' if d_cls else ""
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div>{d_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🚨 Top Fraud Alerts</div>', unsafe_allow_html=True)
    top_alerts = df.sort_values("risk_score", ascending=False).head(4)
    alert_cols = st.columns(4)
    for col, (_, row) in zip(alert_cols, top_alerts.iterrows()):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{row["tx_type"]} · {row["risk_level"]}</div>'
            f'<div class="value">{row["risk_score"]:.0f}/100</div>'
            f'<div class="delta down">${row["amount"]:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Risk Distribution &amp; Transaction Mix</div>',
                unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    cmap = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}

    with col_a:
        lvc = df["risk_level"].value_counts().reset_index()
        lvc.columns = ["Level", "Count"]
        fig = go.Figure(go.Pie(
            labels=lvc["Level"], values=lvc["Count"], hole=0.6,
            marker=dict(colors=[cmap.get(l, "#94a3b8") for l in lvc["Level"]]),
        ))
        fig.update_layout(**get_plotly_style(), title="Risk Level Distribution",
                          height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        tc = df["tx_type"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]
        fig = go.Figure(go.Bar(
            x=tc["Type"], y=tc["Count"],
            marker=dict(color=tc["Count"],
                        colorscale=[[0, "#1d4ed8"], [1, "#7c3aed"]]),
        ))
        fig.update_layout(**get_plotly_style(), title="Transactions by Type",
                          height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        fr = df.groupby("tx_type")["is_fraud"].mean().reset_index()
        fr.columns = ["Type", "FraudRate"]
        fig = go.Figure(go.Bar(
            x=fr["Type"], y=fr["FraudRate"] * 100,
            marker=dict(color=fr["FraudRate"],
                        colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]]),
        ))
        fig.update_layout(**get_plotly_style(), title="Fraud Rate by Type (%)",
                          height=300, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Risk Score &amp; Amount Analysis</div>',
                unsafe_allow_html=True)
    col_d, col_e = st.columns(2)

    with col_d:
        fig = go.Figure(go.Histogram(
            x=df["risk_score"], nbinsx=25,
            marker=dict(color="#3b82f6", line=dict(color="#1d4ed8", width=0.8)),
        ))
        fig.add_vline(x=65, line_dash="dash", line_color="#f97316",
                      annotation_text="High Risk", annotation_font_color="#fdba74")
        fig.add_vline(x=85, line_dash="dash", line_color="#ef4444",
                      annotation_text="Critical",  annotation_font_color="#fca5a5")
        fig.update_layout(**get_plotly_style(), title="Risk Score Distribution",
                          height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_e:
        fig = go.Figure(go.Scatter(
            x=df["fraud_probability"] * 100, y=df["risk_score"],
            mode="markers",
            marker=dict(
                color=df["risk_score"],
                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                size=7, opacity=0.75,
            ),
            text=df["tx_type"],
        ))
        fig.add_hline(y=65, line_dash="dot", line_color="rgba(249,115,22,0.5)")
        fig.add_hline(y=85, line_dash="dot", line_color="rgba(239,68,68,0.5)")
        fig.update_layout(**get_plotly_style(), title="Fraud Probability vs Risk Score",
                          xaxis_title="Fraud Probability (%)", yaxis_title="Risk Score",
                          height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Box plot by risk level ────────────────────────────────────────────────
    st.markdown('<div class="section-title">💰 Transaction Amount by Risk Level</div>',
                unsafe_allow_html=True)
    fig_box = go.Figure()
    for lvl, color in [("Low","#10b981"),("Medium","#f59e0b"),("High","#f97316"),("Critical","#ef4444")]:
        subset = df[df["risk_level"] == lvl]["amount"]
        if not subset.empty:
            fig_box.add_trace(go.Box(
                y=subset,
                name=lvl,
                marker_color=color,
                line_color=color,
                fillcolor=hex_to_rgba(color, 0.18),
            ))
    fig_box.update_layout(**get_plotly_style(), title="Amount Distribution per Risk Band",
                          height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Daily trend (if multi-day) ────────────────────────────────────────────
    if df["date"].nunique() > 1:
        st.markdown('<div class="section-title">📅 Daily Fraud Activity</div>', unsafe_allow_html=True)
        daily = df.groupby("date").agg(
            total=("id","count"), fraud=("is_fraud","sum"), avg_risk=("risk_score","mean")
        ).reset_index()
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=daily["date"], y=daily["total"], name="Total", marker_color="#3b82f6"))
        fig_d.add_trace(go.Bar(x=daily["date"], y=daily["fraud"], name="Fraud", marker_color="#ef4444"))
        fig_d.update_layout(**get_plotly_style(), title="Daily Transaction Volume",
                            barmode="overlay", height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Filterable record cards ───────────────────────────────────────────────
    st.markdown('<div class="section-title">🗂️ Transaction Records</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    search_text = fc1.text_input("Search by type or ID", "",
                                help="Filter the transaction feed by type or transaction id.")
    date_range = fc2.date_input("Date range",
                               [df["date"].min(), df["date"].max()])
    flt_min = fc3.slider("Minimum Risk Score", 0, 100, 0)
    flt_type  = st.selectbox("Filter by Type", ["All"] + sorted(df["tx_type"].unique().tolist()))
    flt_level = st.selectbox("Filter by Risk Level", ["All","Critical","High","Medium","Low"])

    view = df.copy()
    if search_text:
        view = view[view["tx_type"].str.contains(search_text, case=False, na=False) |
                    view["id"].astype(str).str.contains(search_text, na=False)]
    if len(date_range) == 2:
        start_date, end_date = date_range
        view = view[(view["date"] >= start_date) & (view["date"] <= end_date)]
    if flt_type  != "All": view = view[view["tx_type"]   == flt_type]
    if flt_level != "All": view = view[view["risk_level"] == flt_level]
    view = view[view["risk_score"] >= flt_min].head(50)

    for _, row in view.iterrows():
        rc = "high-risk" if row["risk_score"] >= 65 else ("med-risk" if row["risk_score"] >= 40 else "low-risk")
        fb = "🔴 FRAUD" if row["is_fraud"] else "🟢 LEGIT"
        ab = "🔔" if row["alert_sent"] else ""
        st.markdown(
            f"""<div class="fraud-record-card {rc}">
              <div>
                <strong style="color:#f8fafc">#{row['id']} &nbsp; {row['tx_type']}</strong>
                &nbsp;<span style="color:#64748b;font-size:0.8rem;font-family:'Space Mono',monospace">
                  {row['created_at'].strftime('%d %b %H:%M')}
                </span>
              </div>
              <div><span style="font-family:'Space Mono',monospace;color:#94a3b8">${row['amount']:,.0f}</span></div>
              <div style="text-align:right">
                {fb} &nbsp;
                <span style="font-family:'Space Mono',monospace;color:#fcd34d">Risk {row['risk_score']:.0f}/100</span>
                &nbsp;{ab}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Filtered Records (CSV)", data=csv,
                       file_name="fraud_dashboard_export.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def model_metrics_section():
    st.markdown('<div class="section-title">📐 Model Performance Metrics</div>', unsafe_allow_html=True)
    metrics_path = settings.artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        st.info("Train models first (`python -m app.train_models`) to generate metrics.")
        return
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_payload = json.load(f)
    best_model = metrics_payload.get("best_model")
    metrics = metrics_payload.get("metrics", metrics_payload)
    if best_model:
        st.write(f"**Best model:** {best_model.replace('_', ' ').title()}")
    for model_name, m in metrics.items():
        st.subheader(model_name.replace("_", " ").title())
        cols = st.columns(5)
        for col, (key, lbl) in zip(cols, [
            ("accuracy","Accuracy"),("roc_auc","ROC AUC"),
            ("precision","Precision"),("recall","Recall"),("f1","F1 Score"),
        ]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{lbl}</div>'
                f'<div class="value">{m.get(key,0):.3f}</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_db()
    hero_header()
    if not login_form():
        return

    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "Dark"
    theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0 if st.session_state["app_theme"] == "Dark" else 1)
    st.session_state["app_theme"] = theme
    render_theme_css(theme)
    px.defaults.template = "plotly_dark" if theme == "Dark" else "plotly_white"

    st.sidebar.markdown(
        f"""<div style="padding:14px;background:{'#0a1628' if theme == 'Dark' else '#f1f5f9'};border-radius:12px;
          border:1px solid rgba(59,130,246,0.2);margin-bottom:12px;">
          <div style="font-size:0.75rem;color:{'#64748b' if theme == 'Dark' else '#475569'};font-family:'Space Mono',monospace;
            text-transform:uppercase">Logged in as</div>
          <div style="font-size:1rem;font-weight:700;color:{'#93c5fd' if theme == 'Dark' else '#1d4ed8'}">
            {st.session_state.get('username','unknown')}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### ⚙️ Controls")
    threshold    = st.sidebar.slider("High-Risk Alert Threshold", 60, 99, 80)
    refresh_sec  = st.sidebar.slider("Live Refresh (seconds)",    5,  60, 15)
    window_min   = st.sidebar.slider("Live DB Window (minutes)",  1, 120, 30)
    st.session_state["risk_alert_threshold"] = threshold

    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

    tabs = st.tabs(["🌐 Live Analytics", "🔮 Risk Prediction", "🏛️ Fraud Dashboard", "📐 Model Metrics"])

    with tabs[0]:
        df_live = fetch_live_records(limit=1200, window_minutes=window_min)
        live_analytics_section(df_live, refresh_sec)

    with tabs[1]:
        prediction_form()

    with tabs[2]:
        df_all = fetch_all_records(limit=2000)
        fraud_dashboard(df_all)

    with tabs[3]:
        model_metrics_section()


if __name__ == "__main__":
    main()
