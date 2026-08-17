


"""
REVORA
Machine-learning powered automotive horsepower predictor.

Built around an existing, already-trained scikit-learn pipeline
(ExtraTreesRegressor wrapped in preprocessing) produced by the
source notebook. This app does NOT retrain or alter that pipeline -
it only loads it and feeds it correctly-shaped input.

Run with:
    streamlit run app.py
"""

import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================

APP_TITLE = "REVORA"
APP_TAGLINE = "Automotive Performance Intelligence"

try:
    _APP_DIR = Path(__file__).parent
except NameError:
    _APP_DIR = Path.cwd()

CANDIDATE_MODEL_DIRS = [
    _APP_DIR / "models",
    Path("/content/models"),
    Path("models"),
]

TARGET_COL_DEFAULT = "Power (hp)"

# ---- Exact model input contract. DO NOT rename / remove / reorder meaning. ----
HIGH_CARDINALITY_COLS = ["Brand_Manufacturer"]
LOW_CARDINALITY_COLS = ["Origin Country", "Body Type", "Additional Type", "gear_type"]
NUMERIC_COLS = [
    "Approx Cost",
    "Model Year",
    "Weight",
    "Fuel Econ (L/100km)",
    "Fuel Econ (km/L)",
    "Performance 0-100 kph (sec)",
    "Top speed (kph)",
    "gear_count",
]
REQUIRED_MODEL_COLUMNS = HIGH_CARDINALITY_COLS + LOW_CARDINALITY_COLS + NUMERIC_COLS

PERFORMANCE_TIERS = [
    (0, "City Cruiser", "\U0001F697"),
    (120, "Daily Driver", "\U0001F6E3\uFE0F"),
    (200, "Sporty", "\U0001F525"),
    (300, "Performance", "\U0001F3CE\uFE0F"),
    (450, "Supercar Territory", "\U0001F680"),
    (600, "Hypercar Beast", "\U0001F451"),
]

GEAR_TYPE_LABELS = {
    "A": "Automatic",
    "M": "Manual",
    "AM": "Automated Manual",
    "AT": "Automatic (AT)",
    "CVT": "CVT",
}

BODY_TYPE_ICONS = {
    "SUV": "\U0001F699",
    "Sedan": "\U0001F697",
    "Coupe": "\U0001F3CE\uFE0F",
    "Hatchback": "\U0001F697",
    "Convertible": "\U0001F3CE\uFE0F",
    "Pickup": "\U0001F6FB",
    "Van": "\U0001F690",
    "Wagon": "\U0001F699",
}

# Metrics used across the app for scoring / comparison. Direction = "higher" or "lower".
COMPARISON_METRICS = [
    ("Power (hp)", "higher", "HP"),
    ("Weight", "lower_neutral", "kg"),
    ("Top speed (kph)", "higher", "km/h"),
    ("Performance 0-100 kph (sec)", "lower", "s"),
    ("Fuel Econ (km/L)", "higher", "km/L"),
]

# ============================================================================
# DESIGN TOKENS
# ============================================================================

COLORS = {
    "bg": "#0A0C0F",
    "bg_alt": "#111418",
    "surface": "rgba(255,255,255,0.045)",
    "surface_border": "rgba(255,255,255,0.09)",
    "text": "#F2F4F6",
    "text_dim": "#8B939C",
    "accent": "#FF5A1F",
    "accent_soft": "rgba(255,90,31,0.15)",
    "telemetry": "#2FD4C0",
    "telemetry_soft": "rgba(47,212,192,0.15)",
    "redline": "#FF2D55",
    "divider": "rgba(255,255,255,0.08)",
}

CHART_SEQUENCE = [COLORS["accent"], COLORS["telemetry"], "#8C7BFF", "#FFC24B", "#4B9FFF", "#FF7BAC"]


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(ellipse 1200px 600px at 20% -10%, rgba(255,90,31,0.08), transparent 60%),
                radial-gradient(ellipse 900px 500px at 90% 0%, rgba(47,212,192,0.06), transparent 60%),
                {COLORS['bg']};
            color: {COLORS['text']};
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.4rem;
            max-width: 1240px;
        }}

        .ap-display {{
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.04em;
            line-height: 1;
        }}
        .ap-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .ap-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {COLORS['telemetry']};
        }}

        .ap-hero {{
            padding: 2.2rem 2.2rem 1.8rem 2.2rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
            border: 1px solid {COLORS['surface_border']};
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }}
        .ap-hero::after {{
            content: "";
            position: absolute;
            right: -60px; top: -60px;
            width: 260px; height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, {COLORS['accent_soft']}, transparent 70%);
        }}
        .ap-hero h1 {{ font-size: 3.1rem; margin: 0; color: {COLORS['text']}; }}
        .ap-hero h1 span {{ color: {COLORS['accent']}; }}
        .ap-hero p.tagline {{ font-size: 1.0rem; color: {COLORS['text_dim']}; margin: 0.3rem 0 0.7rem 0; }}
        .ap-hero p.sub {{ max-width: 680px; color: {COLORS['text_dim']}; font-size: 0.9rem; line-height: 1.5; }}

        .ap-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['surface_border']};
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
        }}
        .ap-card-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: {COLORS['telemetry']};
            margin-bottom: 0.75rem;
            display: flex; align-items: center; gap: 0.4rem;
        }}

        .ap-pill-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.4rem; }}
        .ap-pill {{
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            border: 1px solid {COLORS['surface_border']};
            background: rgba(255,255,255,0.03);
            font-size: 0.82rem;
            color: {COLORS['text_dim']};
        }}
        .ap-pill.active {{
            background: {COLORS['accent_soft']};
            border-color: rgba(255,90,31,0.5);
            color: {COLORS['text']};
        }}

        .ap-result {{ text-align: center; padding: 1.1rem 0 0.5rem 0; }}
        .ap-result .label {{
            font-family: 'JetBrains Mono', monospace; letter-spacing: 0.2em; font-size: 0.75rem;
            color: {COLORS['text_dim']}; text-transform: uppercase;
        }}
        .ap-result .hp {{
            font-family: 'Bebas Neue', sans-serif; font-size: 5.2rem; color: {COLORS['accent']};
            line-height: 1; margin: 0.1rem 0; text-shadow: 0 0 40px {COLORS['accent_soft']};
        }}
        .ap-result .kw {{ font-family: 'JetBrains Mono', monospace; color: {COLORS['telemetry']}; font-size: 0.92rem; }}
        .ap-tier {{
            display: inline-block; margin-top: 0.6rem; padding: 0.35rem 1rem; border-radius: 999px;
            background: {COLORS['accent_soft']}; border: 1px solid rgba(255,90,31,0.35); font-size: 0.88rem;
        }}

        .ap-metric {{ text-align: center; padding: 0.7rem 0.4rem; }}
        .ap-metric .v {{ font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: {COLORS['text']}; }}
        .ap-metric .l {{ font-size: 0.7rem; color: {COLORS['text_dim']}; text-transform: uppercase; letter-spacing: 0.07em; }}

        .ap-spec-row {{
            display: flex; justify-content: space-between; padding: 0.42rem 0;
            border-bottom: 1px solid {COLORS['divider']}; font-size: 0.86rem;
        }}
        .ap-spec-row:last-child {{ border-bottom: none; }}
        .ap-spec-row .k {{ color: {COLORS['text_dim']}; }}
        .ap-spec-row .v {{ font-family: 'JetBrains Mono', monospace; color: {COLORS['text']}; }}

        hr.ap-div {{ border: none; border-top: 1px solid {COLORS['divider']}; margin: 0.9rem 0; }}

        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['accent']}, #E8460F);
            color: white; font-weight: 700; font-size: 1.0rem; letter-spacing: 0.03em;
            border: none; border-radius: 14px; padding: 0.8rem 1.1rem; width: 100%;
            box-shadow: 0 10px 30px -8px rgba(255,90,31,0.55); transition: transform 0.15s ease;
        }}
        .stButton > button:hover {{ transform: translateY(-1px); }}

        section[data-testid="stSidebar"] {{ background: {COLORS['bg_alt']}; border-right: 1px solid {COLORS['divider']}; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['surface']}; border-radius: 10px 10px 0 0; border: 1px solid {COLORS['surface_border']};
        }}
        .ap-footer {{ text-align: center; color: {COLORS['text_dim']}; font-size: 0.78rem; padding: 1.6rem 0 1rem 0; }}
        .ap-disclaimer {{
            font-size: 0.78rem; color: {COLORS['text_dim']}; border-left: 3px solid {COLORS['accent']};
            padding: 0.5rem 0.9rem; background: rgba(255,90,31,0.06); border-radius: 6px;
        }}
        .ap-winner-banner {{
            text-align: center; padding: 1.2rem; border-radius: 16px;
            background: linear-gradient(135deg, rgba(255,90,31,0.16), rgba(47,212,192,0.10));
            border: 1px solid rgba(255,90,31,0.35); margin: 0.6rem 0 1rem 0;
        }}
        .ap-winner-banner .t {{ font-family: 'Bebas Neue', sans-serif; font-size: 2.1rem; color: {COLORS['accent']}; }}
        .ap-empty-state {{
            text-align: center; color: {COLORS['text_dim']}; padding: 2.4rem 1rem;
            border: 1px dashed {COLORS['surface_border']}; border-radius: 16px;
        }}

        .revora-status {{
            display:flex; align-items:center; gap:.55rem; margin-top:1rem;
            font-family:'JetBrains Mono',monospace; font-size:.68rem;
            letter-spacing:.12em; color:#7f8a93;
        }}
        .status-dot {{
            width:7px; height:7px; border-radius:50%; background:#2FD4C0;
            box-shadow:0 0 12px rgba(47,212,192,.9);
            animation:statusPulse 1.6s infinite;
        }}
        .status-sep {{ opacity:.35; }}
        .revora-cluster {{
            position:relative; overflow:hidden; border-radius:24px; padding:1.25rem;
            background:radial-gradient(circle at 50% 105%, rgba(255,90,31,.15), transparent 45%),linear-gradient(145deg,#11161b,#080a0d);
            border:1px solid rgba(255,255,255,.11);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 22px 55px rgba(0,0,0,.35);
            margin:.7rem 0 1rem;
        }}
        .revora-cluster:before {{
            content:""; position:absolute; inset:0; pointer-events:none;
            background:repeating-linear-gradient(0deg,transparent 0,transparent 3px,rgba(255,255,255,.012) 4px);
        }}
        .cluster-top {{ display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1; }}
        .cluster-label {{ font:700 .66rem 'JetBrains Mono',monospace; letter-spacing:.18em; color:#69747d; }}
        .cluster-mode {{ font:700 .64rem 'JetBrains Mono',monospace; color:#2FD4C0; letter-spacing:.12em; }}
        .cluster-speed {{
            position:relative; z-index:1; text-align:center; margin:.5rem 0 .15rem;
            font:400 5.8rem/1 'Bebas Neue',sans-serif; color:#f2f4f6;
            text-shadow:0 0 35px rgba(255,90,31,.18);
            animation:speedIn .8s cubic-bezier(.2,.8,.2,1) both;
        }}
        .cluster-speed span {{ color:#FF5A1F; font-size:1rem; font-family:'JetBrains Mono',monospace; letter-spacing:.15em; margin-left:.3rem; }}
        .cluster-sub {{ position:relative; z-index:1; text-align:center; font:600 .66rem 'JetBrains Mono',monospace; color:#68727b; letter-spacing:.16em; }}
        .cluster-grid {{
            position:relative; z-index:1; display:grid; grid-template-columns:repeat(3,1fr);
            gap:.55rem; margin-top:1rem;
        }}
        .cluster-cell {{
            border:1px solid rgba(255,255,255,.07); background:rgba(255,255,255,.025);
            border-radius:12px; padding:.65rem .45rem; text-align:center;
        }}
        .cluster-cell .v {{ font:700 1rem 'JetBrains Mono',monospace; color:#eef1f3; }}
        .cluster-cell .k {{ margin-top:.18rem; font:500 .56rem 'JetBrains Mono',monospace; color:#68727b; letter-spacing:.08em; }}
        .rpm-wrap {{ position:relative; z-index:1; margin-top:1rem; }}
        .rpm-track {{ height:8px; border-radius:99px; background:#1a2025; overflow:hidden; box-shadow:inset 0 1px 3px #000; }}
        .rpm-fill {{ height:100%; border-radius:99px; background:linear-gradient(90deg,#2FD4C0 0 55%,#FF5A1F 78%,#FF2D55); transform-origin:left; animation:rpmFill 1.1s ease-out both; }}
        .rpm-scale {{ display:flex; justify-content:space-between; margin-top:.25rem; color:#56616a; font:500 .52rem 'JetBrains Mono',monospace; }}
        .gauge-wrap {{ position:relative; z-index:1; display:flex; justify-content:center; margin:.9rem 0 .3rem; }}
        .gauge-dial {{
            position:relative; width:180px; height:180px; border-radius:50%;
            background: conic-gradient(from 220deg, #2FD4C0 0deg, #FF5A1F var(--gauge-fill,120deg), rgba(255,255,255,.06) var(--gauge-fill,120deg) 300deg, transparent 300deg 360deg);
            display:flex; align-items:center; justify-content:center;
            box-shadow: inset 0 0 25px rgba(0,0,0,.55);
            animation: dialSweep 1.1s cubic-bezier(.2,.8,.2,1) both;
        }}
        .gauge-dial:before {{
            content:""; position:absolute; width:148px; height:148px; border-radius:50%;
            background:#0b0e11; box-shadow:inset 0 0 15px rgba(0,0,0,.6);
        }}
        .gauge-needle {{
            position:absolute; width:3px; height:66px; background:linear-gradient(#fff,#FF5A1F);
            top:24px; left:50%; transform-origin:bottom center;
            transform:translateX(-50%) rotate(var(--needle-angle,-130deg));
            border-radius:4px; animation:needleSweep 1.3s cubic-bezier(.2,.8,.2,1) both;
            box-shadow:0 0 10px rgba(255,90,31,.7);
        }}
        .gauge-center {{ position:absolute; z-index:2; text-align:center; }}
        .gauge-center .val {{ font:400 2.05rem/1 'Bebas Neue',sans-serif; color:#f2f4f6; }}
        .gauge-center .lbl {{ margin-top:.2rem; font:600 .56rem 'JetBrains Mono',monospace; letter-spacing:.14em; color:#68727b; }}
        @keyframes dialSweep {{ from {{ opacity:0; transform:scale(.85); }} to {{ opacity:1; transform:scale(1); }} }}
        @keyframes needleSweep {{ from {{ transform:translateX(-50%) rotate(-130deg); }} to {{ transform:translateX(-50%) rotate(var(--needle-angle,-130deg)); }} }}
        .predict-reveal {{ animation:panelReveal .75s ease-out both; }}
        .battle-stage {{
            position:relative; overflow:hidden; border-radius:22px; min-height:230px; padding:1rem;
            background:linear-gradient(180deg,#0d1115,#080a0d);
            border:1px solid rgba(255,255,255,.1);
        }}
        .battle-lane {{ position:absolute; left:4%; right:4%; height:1px; background:rgba(255,255,255,.08); }}
        .battle-lane.one {{ top:38%; }} .battle-lane.two {{ top:68%; }}
        .speed-streak {{
            position:absolute; height:2px; width:60px; left:0;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent);
            animation:streakMove 1.1s linear infinite;
        }}
        .speed-streak.one {{ top:34%; animation-delay:0s; }}
        .speed-streak.two {{ top:36%; animation-delay:.35s; }}
        .speed-streak.three {{ top:64%; animation-delay:.15s; }}
        .speed-streak.four {{ top:66%; animation-delay:.5s; }}
        @keyframes streakMove {{ 0% {{ transform:translateX(-10%); opacity:0; }} 20% {{ opacity:1; }} 100% {{ transform:translateX(560px); opacity:0; }} }}
        .battle-car {{
            position:absolute; width:170px; height:42px; border-radius:18px 28px 10px 10px;
            background:linear-gradient(180deg,#343c44,#151a1f); border:1px solid rgba(255,255,255,.14);
            box-shadow:0 12px 22px rgba(0,0,0,.45);
        }}
        .battle-car:before {{
            content:""; position:absolute; left:32px; top:-13px; width:70px; height:22px;
            border-radius:18px 18px 4px 4px; background:#20272d; border:1px solid rgba(255,255,255,.1);
        }}
        .battle-car:after {{
            content:""; position:absolute; left:16px; right:-80px; top:15px; height:3px;
            background:linear-gradient(90deg,transparent,rgba(255,90,31,.5),transparent); filter:blur(1px);
        }}
        .battle-car.a {{ top:26%; left:3%; border-color:rgba(255,90,31,.55); animation:raceA 2.7s cubic-bezier(.18,.8,.2,1) forwards; }}
        .battle-car.b {{ top:56%; left:3%; border-color:rgba(47,212,192,.55); animation:raceB 2.7s cubic-bezier(.18,.8,.2,1) forwards; }}
        .battle-wheel {{ position:absolute; bottom:-7px; width:17px; height:17px; border-radius:50%; background:#050607; border:3px solid #252c32; }}
        .battle-wheel.w1 {{ left:25px; }} .battle-wheel.w2 {{ right:25px; }}
        .battle-name {{ position:absolute; top:8px; font:700 .62rem 'JetBrains Mono',monospace; letter-spacing:.12em; }}
        .battle-name.a {{ left:1rem; color:#FF5A1F; }} .battle-name.b {{ left:1rem; top:50%; color:#2FD4C0; }}
        .winner-pop {{ animation:winnerPop .75s cubic-bezier(.2,.9,.2,1) .15s both; text-align:center; padding:1rem; }}
        .winner-pop .cup {{ font-size:2rem; filter:drop-shadow(0 0 14px rgba(255,194,75,.5)); }}
        .winner-pop .winner-text {{ font:400 2.5rem/1 'Bebas Neue',sans-serif; color:#FF5A1F; letter-spacing:.05em; }}
        .winner-pop .winner-meta {{ color:#7f8a93; font:600 .7rem 'JetBrains Mono',monospace; letter-spacing:.08em; }}
        @keyframes statusPulse {{ 50% {{ opacity:.35; transform:scale(.7); }} }}
        @keyframes speedIn {{ from {{ opacity:0; transform:translateY(15px) scale(.92); }} to {{ opacity:1; transform:none; }} }}
        @keyframes rpmFill {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}
        @keyframes panelReveal {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:none; }} }}
        @keyframes raceA {{ from {{ transform:translateX(0); }} to {{ transform:translateX(540px); }} }}
        @keyframes raceB {{ from {{ transform:translateX(0); }} to {{ transform:translateX(430px); }} }}
        @keyframes winnerPop {{ from {{ opacity:0; transform:scale(.75); }} to {{ opacity:1; transform:scale(1); }} }}
        @media (max-width: 760px) {{
            .cluster-speed {{ font-size:4.3rem; }}
            .battle-car {{ width:125px; }}
            @keyframes raceA {{ to {{ transform:translateX(240px); }} }}
            @keyframes raceB {{ to {{ transform:translateX(190px); }} }}
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA / MODEL LOADING
# ============================================================================

def _find_model_dir():
    for d in CANDIDATE_MODEL_DIRS:
        if d.exists() and (d / "best_model.joblib").exists():
            return d
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_dir_str: str):
    model_dir = Path(model_dir_str)
    return joblib.load(model_dir / "best_model.joblib")


@st.cache_data(show_spinner=False)
def load_artifacts(model_dir_str: str):
    model_dir = Path(model_dir_str)

    dataset = joblib.load(model_dir / "dataset.joblib")
    dataset = normalize_dataset_columns(dataset)

    results_df = None
    if (model_dir / "results_df.joblib").exists():
        results_df = joblib.load(model_dir / "results_df.joblib")

    target_col = TARGET_COL_DEFAULT
    if (model_dir / "target_col.joblib").exists():
        target_col = joblib.load(model_dir / "target_col.joblib")

    feature_importance = None
    if (model_dir / "feature_importance.joblib").exists():
        feature_importance = joblib.load(model_dir / "feature_importance.joblib")

    dataset = clean_dataset(dataset)

    return dataset, results_df, target_col, feature_importance

def normalize_dataset_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the dataset contains the columns required by the GUI.

    The ML model uses Brand_Manufacturer, while the GUI expects
    separate Manufacturer and Brand columns.
    """
    df = dataset.copy()

    # Clean column names first
    df.columns = [str(c).strip() for c in df.columns]

    # If Manufacturer and Brand already exist, keep them.
    if "Manufacturer" in df.columns and "Brand" in df.columns:
        return df

    # Try to reconstruct them from Brand_Manufacturer.
    if "Brand_Manufacturer" in df.columns:
        parts = (
            df["Brand_Manufacturer"]
            .astype(str)
            .str.strip()
            .str.split(n=1, expand=True)
        )

        if "Manufacturer" not in df.columns:
            df["Manufacturer"] = parts[0]

        if "Brand" not in df.columns:
            df["Brand"] = (
                parts[1]
                if parts.shape[1] > 1
                else df["Brand_Manufacturer"].astype(str)
            )

        return df

    raise ValueError(
        "Dataset must contain either "
        "'Manufacturer' + 'Brand' or 'Brand_Manufacturer'. "
        f"Found columns: {list(df.columns)}"
    )

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Defensive cleanup: dedupe, coerce numerics, drop rows missing the target,
    replace inf, add a stable unique row id used across the app (Power Battle etc.)."""
    df = dataset.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    for col in NUMERIC_COLS + ["Power (hp)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    if "Power (hp)" in df.columns:
        df = df.dropna(subset=["Power (hp)"]).reset_index(drop=True)

    for col in HIGH_CARDINALITY_COLS + LOW_CARDINALITY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Stable unique identifier, independent of Manufacturer+Brand+Year duplicates.
    df["_car_id"] = [f"car_{i}" for i in range(len(df))]
    return df


def best_model_name(results_df):
    if results_df is not None and len(results_df) > 0 and "Test Accuracy" in results_df.columns:
        try:
            return str(results_df.sort_values("Test Accuracy", ascending=False).iloc[0]["Model"])
        except Exception:
            pass
    return "Extra Trees"


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def brand_color(name: str):
    h = int(hashlib.md5(str(name).encode()).hexdigest(), 16)
    hue1 = h % 360
    hue2 = (hue1 + 46) % 360
    c1 = f"hsl({hue1}, 72%, 58%)"
    c2 = f"hsl({hue2}, 72%, 42%)"
    return c1, c2


def initials(name: str):
    parts = [p for p in str(name).replace("-", " ").split(" ") if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def classify_performance(hp: float):
    tier = PERFORMANCE_TIERS[0]
    for min_hp, label, emoji in PERFORMANCE_TIERS:
        if hp >= min_hp:
            tier = (min_hp, label, emoji)
    return tier[1], tier[2]


def safe_numeric_range(series: pd.Series, default=(0.0, 100.0)):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return default
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def safe_unique(df: pd.DataFrame, col: str, fallback=None):
    if col not in df.columns:
        return fallback or []
    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
    return vals if vals else (fallback or [])


def build_input_dataframe(values: dict) -> pd.DataFrame:
    """Build input using exactly the feature names used during training."""

    # Safety check: recreate Brand_Manufacturer if it is missing.
    if not values.get("Brand_Manufacturer"):
        manufacturer = str(values.get("Manufacturer", "")).strip()
        brand = str(values.get("Brand", "")).strip()
        values["Brand_Manufacturer"] = f"{manufacturer} {brand}".strip()

    row = {
        col: values.get(col)
        for col in REQUIRED_MODEL_COLUMNS
    }

    return pd.DataFrame([row])


def predict_power(model, input_df: pd.DataFrame) -> float:
    pred = model.predict(input_df)
    return float(np.asarray(pred).ravel()[0])


def pearson_label(r: float) -> str:
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "Not available"
    a = abs(r)
    strength = "Very strong" if a >= 0.9 else "Strong" if a >= 0.7 else "Moderate" if a >= 0.5 else \
        "Weak" if a >= 0.3 else "Very weak / negligible"
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    return f"{strength} {direction} correlation"


def normalize_series(values, higher_is_better=True):
    """0-100 min-max normalization for scoring."""
    values = np.array(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi == lo:
        return np.full_like(values, 50.0)
    norm = (values - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        norm = 100.0 - norm
    return norm


def segmented_choice(label, options, default=None, key=None, help_=None):
    """Use st.segmented_control if available (Streamlit >= 1.36), else fall back
    to a horizontal radio so the UI stays button/pill-like regardless of version."""
    idx_default = options.index(default) if (default in options) else 0
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options, default=options[idx_default], key=key, help=help_)
        return val if val is not None else options[idx_default]
    return st.radio(label, options, index=idx_default, key=key, horizontal=True, help=help_)


# ============================================================================
# RENDER: HERO / SIDEBAR / FOOTER
# ============================================================================

def render_hero():
    st.markdown(
        f"""
        <div class="ap-hero">
            <div class="ap-eyebrow">REVORA &middot; Automotive Performance Intelligence</div>
            <h1 class="ap-display"><span>REVORA</span></h1>
            <p class="tagline">{APP_TAGLINE}</p>
            <p class="sub">Build a vehicle configuration, predict its engine output, then read the result like a real performance instrument cluster.</p>
            <div class="revora-status">
                <span class="status-dot"></span> SYSTEM ONLINE
                <span class="status-sep">|</span> MODEL YEAR RANGE: <b>2026</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(dataset: pd.DataFrame, best_name: str):
    with st.sidebar:
        st.markdown(
            "<div class='ap-eyebrow'>REVORA</div>"
            "<div class='ap-display' style='font-size:1.5rem;margin-bottom:0.5rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )
        st.markdown("Predictor &middot; Power Battle &middot; Explore Garage &middot; Model Insights")
        st.markdown("<hr class='ap-div'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Fun Statistics</div>", unsafe_allow_html=True)

        n_cars = len(dataset)
        n_makers = dataset["Manufacturer"].nunique() if "Manufacturer" in dataset.columns else 0
        if "Model Year" in dataset.columns and dataset["Model Year"].notna().any():
            yr_min, yr_max = int(dataset["Model Year"].min()), int(dataset["Model Year"].max())
        else:
            yr_min, yr_max = 0, 0

        st.markdown(
            f"<div class='ap-metric' style='text-align:left;'><div class='v'>{n_cars:,}</div>"
            f"<div class='l'>Cars in Garage</div></div>"
            f"<div class='ap-metric' style='text-align:left;'><div class='v'>{n_makers}</div>"
            f"<div class='l'>Manufacturers</div></div>"
            f"<div class='ap-metric' style='text-align:left;'><div class='v'>{yr_min}&ndash;{yr_max}</div>"
            f"<div class='l'>Model Years</div></div>",
            unsafe_allow_html=True,
        )

        if "Power (hp)" in dataset.columns and dataset["Power (hp)"].notna().any():
            max_hp_row = dataset.loc[dataset["Power (hp)"].idxmax()]
            st.markdown(
                f"<div class='ap-metric' style='text-align:left;'>"
                f"<div class='v'>{int(max_hp_row['Power (hp)'])} HP</div>"
                f"<div class='l'>Highest recorded ({max_hp_row.get('Manufacturer','?')} {max_hp_row.get('Brand','?')})</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='ap-div'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='ap-card-title'>Active Model</div>"
            f"<div style='font-size:0.88rem;color:{COLORS['text_dim']}'>{best_name}</div>",
            unsafe_allow_html=True,
        )


def render_footer():
    st.markdown("<hr class='ap-div'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ap-footer">
            REVORA &middot; Machine Learning Automotive Performance Predictor<br>
            Built with Python &middot; Scikit-learn &middot; Streamlit<br>
            &copy; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STYLIZED 3D-LOOKING VEHICLE VISUALIZATION (procedural, Plotly WebGL)
# ============================================================================
# NOTE: This is a stylized, procedurally generated "AI vehicle visualization",
# not a manufacturer CAD model and not a copyrighted logo/design. Shape reacts
# to body type / weight / performance so every configuration looks distinct.

def _box(x0, x1, y0, y1, z0, z1):
    """Return vertices/faces (i,j,k) for a rectangular box, for Mesh3d."""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return xs, ys, zs, i, j, k


def render_vehicle_visual(manufacturer: str, brand: str, body_type: str, weight: float,
                           top_speed: float, accel: float, gear_type: str, height_px: int = 320):
    """Procedural stylized 3D car built from primitive boxes; proportions vary by body type."""
    c1, _ = brand_color(f"{manufacturer}{brand}")
    body_type_key = (body_type or "").strip().lower()

    # Base proportions
    length, width, cabin_h, ground_clear = 4.2, 1.8, 0.55, 0.16

    if "suv" in body_type_key or "pickup" in body_type_key or "van" in body_type_key:
        cabin_h, ground_clear, width = 0.85, 0.30, 1.95
    elif "coupe" in body_type_key or "convertible" in body_type_key:
        cabin_h, ground_clear = 0.42, 0.11
    elif "hatchback" in body_type_key:
        length, cabin_h = 3.7, 0.62
    elif "sedan" in body_type_key:
        length, cabin_h = 4.5, 0.5

    # Weight nudges width/length slightly so heavier cars look bulkier.
    weight = weight or 1500.0
    bulk = np.clip((weight - 1200) / 1800, -0.15, 0.35)
    width += bulk * 0.25
    length += bulk * 0.4

    wheel_r = 0.32 + (0.05 if "suv" in body_type_key or "pickup" in body_type_key else 0.0)

    fig = go.Figure()

    # Ground shadow
    theta = np.linspace(0, 2 * np.pi, 40)
    shadow_x = (length / 2 + 0.2) * np.cos(theta)
    shadow_y = (width / 2 + 0.15) * np.sin(theta)
    fig.add_trace(go.Scatter3d(
        x=shadow_x, y=shadow_y, z=np.zeros_like(shadow_x),
        mode="lines", surfaceaxis=2, surfacecolor="rgba(0,0,0,0.45)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    # Lower body
    xs, ys, zs, i, j, k = _box(-length / 2, length / 2, -width / 2, width / 2,
                                ground_clear, ground_clear + cabin_h * 0.55)
    fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                             color=c1, opacity=0.96, flatshading=True,
                             lighting=dict(ambient=0.55, diffuse=0.75, specular=0.9, roughness=0.25),
                             lightposition=dict(x=100, y=200, z=300),
                             name="Body", showlegend=False, hoverinfo="skip"))

    # Cabin / roof (skipped visually lowered for convertible)
    if "convertible" not in body_type_key:
        cx0, cx1 = -length / 4.6, length / 3.2
        cz0 = ground_clear + cabin_h * 0.55
        cz1 = cz0 + cabin_h * 0.62
        xs, ys, zs, i, j, k = _box(cx0, cx1, -width / 2 + 0.12, width / 2 - 0.12, cz0, cz1)
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                                 color="#1b1f24", opacity=0.9, flatshading=True,
                                 lighting=dict(ambient=0.5, diffuse=0.6, specular=0.6),
                                 name="Cabin", showlegend=False, hoverinfo="skip"))

    # Pickup cargo bed cut-out illusion: small raised rails at the rear
    if "pickup" in body_type_key:
        bx0, bx1 = -length / 2 + 0.1, -length / 8
        bz0 = ground_clear + cabin_h * 0.55
        bz1 = bz0 + cabin_h * 0.35
        xs, ys, zs, i, j, k = _box(bx0, bx1, -width / 2 + 0.05, width / 2 - 0.05, bz0, bz1)
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                                 color="#2a2f36", opacity=0.85, flatshading=True,
                                 name="Bed", showlegend=False, hoverinfo="skip"))

    # Wheels (4 cylinders approximated with disks via scatter3d markers for simplicity/perf)
    wheel_positions = [
        (length / 2 - wheel_r - 0.25, width / 2 - 0.05),
        (length / 2 - wheel_r - 0.25, -width / 2 + 0.05),
        (-length / 2 + wheel_r + 0.25, width / 2 - 0.05),
        (-length / 2 + wheel_r + 0.25, -width / 2 + 0.05),
    ]
    for wx, wy in wheel_positions:
        theta = np.linspace(0, 2 * np.pi, 16)
        ring_y = wy + 0.01 * np.sign(wy)
        fig.add_trace(go.Scatter3d(
            x=wx + wheel_r * np.cos(theta), y=np.full(16, ring_y),
            z=wheel_r + wheel_r * np.sin(theta),
            mode="lines", line=dict(color="#111418", width=10),
            surfaceaxis=1, surfacecolor="#111418",
            showlegend=False, hoverinfo="skip",
        ))

    # Headlights / taillights as bright markers
    fig.add_trace(go.Scatter3d(
        x=[length / 2, length / 2, -length / 2, -length / 2],
        y=[width / 2 - 0.18, -width / 2 + 0.18, width / 2 - 0.18, -width / 2 + 0.18],
        z=[ground_clear + cabin_h * 0.35] * 4,
        mode="markers",
        marker=dict(size=5, color=[COLORS["telemetry"], COLORS["telemetry"], COLORS["redline"], COLORS["redline"]]),
        showlegend=False, hoverinfo="skip",
    ))

    max_dim = max(length, width) * 0.95
    fig.update_layout(
        height=height_px,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False, range=[-max_dim, max_dim]),
            yaxis=dict(visible=False, range=[-max_dim, max_dim]),
            zaxis=dict(visible=False, range=[0, max_dim]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    return fig


# ============================================================================
# VEHICLE SELECTOR (predictor tab) — typed / segmented inputs, minimal sliders
# ============================================================================

def render_vehicle_selector(dataset: pd.DataFrame, key_prefix="pred") -> dict:
    values = {}

    # ---------------- Vehicle identity ----------------
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>🚗 Vehicle Identity</div>", unsafe_allow_html=True)

    year_series = pd.to_numeric(dataset["Model Year"], errors="coerce").dropna()
    if year_series.empty:
        min_model_year, max_model_year = 2020, 2026
    else:
        min_model_year = int(year_series.min())
        max_model_year = 2026
        if min_model_year > max_model_year:
            min_model_year = max_model_year

    manufacturers = safe_unique(dataset, "Manufacturer", ["Unknown"])
    default_idx = manufacturers.index("Mercedes-Benz") if "Mercedes-Benz" in manufacturers else 0

    c1, c2 = st.columns(2)
    with c1:
        manufacturer = st.selectbox("Manufacturer", manufacturers, index=default_idx, key=f"{key_prefix}_mfr")

    manufacturer_df = dataset[dataset["Manufacturer"] == manufacturer]
    brands = safe_unique(manufacturer_df, "Brand", ["Unknown"])
    brands = sorted(dict.fromkeys(str(x).strip() for x in brands if str(x).strip()), key=str.lower) or ["Unknown"]
    with c2:
        brand = st.selectbox("Model", brands, key=f"{key_prefix}_brand")

    pool = dataset[(dataset["Manufacturer"] == manufacturer) & (dataset["Brand"] == brand)].copy()
    pool["_year_numeric"] = pd.to_numeric(pool["Model Year"], errors="coerce")
    available_years = sorted(pool["_year_numeric"].dropna().astype(int).loc[lambda x: x <= 2026].unique().tolist())
    default_year = int(np.clip(2026, min_model_year, max_model_year))

    model_year = st.number_input(
        "Model Year",
        min_value=min_model_year,
        max_value=max_model_year,
        value=default_year,
        step=1,
        format="%d",
        key=f"{key_prefix}_year",
        help=f"Type a year or use +/-. Dataset starts at {min_model_year}; REVORA supports through 2026.",
    )
    model_year = int(model_year)

    if available_years and model_year not in available_years:
        st.warning(f"{brand} has no exact {model_year} record in the dataset. Available years: {', '.join(map(str, available_years))}")

    st.markdown(
        f"<div style='color:{COLORS['text_dim']};font-size:.78rem;margin-top:-8px;margin-bottom:8px;'>"
        f"Selected: <b>{manufacturer} {brand}</b> · {model_year}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Drivetrain & body ----------------
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>⚙️ Drivetrain &amp; Body</div>", unsafe_allow_html=True)

    body_types = safe_unique(dataset, "Body Type", ["Sedan"])
    add_types = safe_unique(dataset, "Additional Type", ["Standard"])
    origins = safe_unique(dataset, "Origin Country", ["Unknown"])
    gear_types = safe_unique(dataset, "gear_type", ["A"])
    gear_counts = sorted(pd.to_numeric(dataset.get("gear_count", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()) or [6]

    c1, c2 = st.columns(2)
    with c1:
        body_type = st.selectbox("Body Type", body_types, key=f"{key_prefix}_body")
    with c2:
        gear_type = st.selectbox(
            "Gearbox Type", gear_types, key=f"{key_prefix}_gear",
            help="AT=Automatic, M=Manual, AM=Automated Manual, CVT=Continuously Variable",
        )
    st.caption(f"Selected gearbox: **{GEAR_TYPE_LABELS.get(gear_type, gear_type)}**")

    c1, c2 = st.columns(2)
    with c1:
        origin_country = st.selectbox("Origin Country", origins, key=f"{key_prefix}_origin")
    with c2:
        additional_type = st.selectbox("Additional Type", add_types, key=f"{key_prefix}_addtype")
    gear_count = st.selectbox("Number of Gears", gear_counts, index=len(gear_counts)//2, key=f"{key_prefix}_gearcount")
    st.markdown("</div>", unsafe_allow_html=True)

    values.update({
        "Manufacturer": manufacturer,
        "Brand": brand,
        "Brand_Manufacturer": f"{manufacturer} {brand}".strip(),
        "Model Year": model_year,
        "Body Type": body_type,
        "Additional Type": additional_type,
        "Origin Country": origin_country,
        "gear_type": gear_type,
        "gear_count": float(gear_count),
    })

    # ---------------- Performance ----------------
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>🏎️ Performance &amp; Efficiency</div>", unsafe_allow_html=True)

    weight_lo, weight_hi = safe_numeric_range(dataset["Weight"], (900.0, 3200.0))
    perf_lo, perf_hi = safe_numeric_range(dataset["Performance 0-100 kph (sec)"], (2.0, 20.0))
    top_lo, top_hi = safe_numeric_range(dataset["Top speed (kph)"], (120.0, 350.0))
    fuel_lo, fuel_hi = safe_numeric_range(dataset["Fuel Econ (L/100km)"], (3.0, 20.0))
    cost_lo, cost_hi = safe_numeric_range(dataset["Approx Cost"], (50000.0, 2000000.0))

    c1, c2 = st.columns(2)
    with c1:
        weight = st.number_input("Weight (kg)", min_value=float(round(weight_lo)), max_value=float(round(weight_hi)), value=float(round(dataset["Weight"].median())), step=10.0, key=f"{key_prefix}_weight")
        perf_0_100 = st.number_input("0–100 km/h (sec)", min_value=round(perf_lo,1), max_value=round(perf_hi,1), value=round(float(dataset["Performance 0-100 kph (sec)"].median()),1), step=0.1, key=f"{key_prefix}_perf")
        top_speed = st.number_input("Top Speed (km/h)", min_value=float(round(top_lo)), max_value=float(round(top_hi)), value=float(round(dataset["Top speed (kph)"].median())), step=1.0, key=f"{key_prefix}_topspeed")
    with c2:
        fuel_l100 = st.number_input("Fuel Economy (L/100km)", min_value=round(fuel_lo,1), max_value=round(fuel_hi,1), value=round(float(dataset["Fuel Econ (L/100km)"].median()),1), step=0.1, key=f"{key_prefix}_fuel")
        fuel_kml = round(100.0 / fuel_l100, 1) if fuel_l100 > 0 else 0.0
        st.caption(f"≈ {fuel_kml} km/L")
        approx_cost = st.number_input("Approx Cost (AED)", min_value=float(round(cost_lo,-3)), max_value=float(round(cost_hi,-3)), value=float(round(dataset["Approx Cost"].median(),-3)), step=1000.0, key=f"{key_prefix}_cost")
    st.markdown("</div>", unsafe_allow_html=True)

    values.update({
        "Weight": weight,
        "Performance 0-100 kph (sec)": perf_0_100,
        "Top speed (kph)": top_speed,
        "Fuel Econ (L/100km)": fuel_l100,
        "Fuel Econ (km/L)": fuel_kml,
        "Approx Cost": approx_cost,
    })
    return values


def render_vehicle_preview(values: dict, dataset: pd.DataFrame):
    """Right-hand panel shown BEFORE prediction: no empty space, full spec + 3D preview."""
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>\U0001F441\uFE0F Vehicle Preview</div>", unsafe_allow_html=True)

    fig = render_vehicle_visual(
        values["Manufacturer"], values["Brand"], values["Body Type"], values["Weight"],
        values["Top speed (kph)"], values["Performance 0-100 kph (sec)"], values["gear_type"],
        height_px=280,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div style='text-align:center;'><div class='ap-display' style='font-size:1.4rem;'>"
        f"{values['Manufacturer']}</div><div style='color:{COLORS['text_dim']};'>{values['Brand']} "
        f"&middot; {values['Model Year']}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='ap-div'>", unsafe_allow_html=True)

    ptw = values["Weight"] and (dataset["Power (hp)"].median() / values["Weight"] * 1000)
    spec_rows = [
        ("Body Type", values["Body Type"]),
        ("Gearbox", f"{values['gear_type']} \u2014 {GEAR_TYPE_LABELS.get(values['gear_type'], values['gear_type'])}"),
        ("Weight", f"{values['Weight']:,.0f} kg"),
        ("0\u2013100 km/h", f"{values['Performance 0-100 kph (sec)']:.1f} s"),
        ("Top Speed", f"{values['Top speed (kph)']:,.0f} km/h"),
        ("Fuel Economy", f"{values['Fuel Econ (L/100km)']:.1f} L/100km ({values['Fuel Econ (km/L)']:.1f} km/L)"),
        ("Approx Cost", f"AED {values['Approx Cost']:,.0f}"),
    ]
    rows_html = "".join(
        f"<div class='ap-spec-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>"
        for k, v in spec_rows
    )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='ap-empty-state'>\u26A1 Ready to predict.<br>"
        "Click <b>PREDICT ENGINE POWER</b> to estimate this configuration's horsepower.</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# RENDER: PREDICTION RESULT + GAUGE
# ============================================================================

def render_power_gauge(hp: float, max_scale: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=hp,
            number={"suffix": " HP", "font": {"size": 1, "color": "rgba(0,0,0,0)"}},
            gauge={
                "axis": {"range": [0, max_scale], "tickcolor": COLORS["text_dim"], "tickfont": {"color": COLORS["text_dim"]}},
                "bar": {"color": COLORS["accent"]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_scale * 0.5], "color": "rgba(47,212,192,0.18)"},
                    {"range": [max_scale * 0.5, max_scale * 0.8], "color": "rgba(255,90,31,0.20)"},
                    {"range": [max_scale * 0.8, max_scale], "color": "rgba(255,45,85,0.28)"},
                ],
                "threshold": {"line": {"color": COLORS["redline"], "width": 3}, "thickness": 0.85, "value": hp},
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _safe_norm(value, series, higher=True):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty or pd.isna(value):
        return 50.0
    lo, hi = float(series.min()), float(series.max())
    if hi == lo:
        return 50.0
    x = (float(value) - lo) / (hi - lo) * 100.0
    return float(np.clip(x if higher else 100.0 - x, 0.0, 100.0))


def render_performance_dna(hp: float, values: dict, dataset: pd.DataFrame):
    """Transparent Performance DNA: normalized against the real dataset."""
    ptw = hp / values["Weight"] * 1000 if values.get("Weight") else np.nan
    labels = ["Power", "Acceleration", "Top Speed", "Efficiency", "Power/Weight"]
    scores = [
        _safe_norm(hp, dataset["Power (hp)"], True),
        _safe_norm(values["Performance 0-100 kph (sec)"], dataset["Performance 0-100 kph (sec)"], False),
        _safe_norm(values["Top speed (kph)"], dataset["Top speed (kph)"], True),
        _safe_norm(values["Fuel Econ (km/L)"], dataset["Fuel Econ (km/L)"], True),
        _safe_norm(ptw, (dataset["Power (hp)"] / dataset["Weight"].replace(0, np.nan) * 1000), True),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]], theta=labels + [labels[0]], fill="toself",
        line_color=COLORS["accent"], fillcolor="rgba(255,90,31,.18)",
        name="Vehicle",
    ))
    fig.update_layout(
        title="Performance DNA",
        polar=dict(radialaxis=dict(visible=True, range=[0,100], color=COLORS["text_dim"]), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", font={"color":COLORS["text"]}, height=400,
        margin=dict(l=20,r=20,t=55,b=20), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.caption("Scores are normalized against the current dataset. They are relative performance indicators, not probabilities.")



def render_instrument_cluster(hp: float, values: dict):
    top_speed = float(values.get("Top speed (kph)", 0))
    accel = float(values.get("Performance 0-100 kph (sec)", 0))
    rpm_pct = int(np.clip((hp / 900.0) * 100, 8, 100))
    cluster_speed = int(np.clip(top_speed, 0, 420))
    gauge_fill_deg = 300 * rpm_pct / 100
    needle_angle = -130 + (260 * rpm_pct / 100)
    st.markdown(
        f"""
        <div class="revora-cluster predict-reveal">
            <div class="cluster-top">
                <div class="cluster-label">REVORA / PERFORMANCE CLUSTER</div>
                <div class="cluster-mode">● DRIVE READY</div>
            </div>
            <div class="gauge-wrap">
                <div class="gauge-dial" style="--gauge-fill:{gauge_fill_deg:.0f}deg;">
                    <div class="gauge-needle" style="--needle-angle:{needle_angle:.0f}deg;"></div>
                    <div class="gauge-center">
                        <div class="val">{hp:,.0f}</div>
                        <div class="lbl">HORSEPOWER</div>
                    </div>
                </div>
            </div>
            <div class="cluster-speed">{cluster_speed}<span>KM/H</span></div>
            <div class="cluster-sub">ESTIMATED PERFORMANCE OUTPUT</div>
            <div class="cluster-grid">
                <div class="cluster-cell"><div class="v">{hp:,.0f}</div><div class="k">HORSEPOWER</div></div>
                <div class="cluster-cell"><div class="v">{hp*0.7457:,.0f}</div><div class="k">KW</div></div>
                <div class="cluster-cell"><div class="v">{accel:.1f}s</div><div class="k">0–100 KM/H</div></div>
            </div>
            <div class="rpm-wrap">
                <div class="cluster-label">POWER LOAD / RPM ZONE</div>
                <div class="rpm-track"><div class="rpm-fill" style="width:{rpm_pct}%"></div></div>
                <div class="rpm-scale"><span>0</span><span>2K</span><span>4K</span><span>6K</span><span>8K+</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_prediction_result(hp: float, results_df, dataset: pd.DataFrame, values: dict, feature_importance=None):
    kw = hp * 0.7457
    tier_label, tier_emoji = classify_performance(hp)
    render_instrument_cluster(hp, values)
    ptw = hp / values["Weight"] * 1000 if values.get("Weight") else float("nan")

    fig = render_vehicle_visual(
        values["Manufacturer"], values["Brand"], values["Body Type"], values["Weight"],
        values["Top speed (kph)"], values["Performance 0-100 kph (sec)"], values["gear_type"], height_px=240,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown(
        f"""<div class="ap-result"><div class="label">Predicted Power</div>
        <div class="hp ap-mono predict-reveal">{hp:,.0f}</div>
        <div class="kw">Horsepower · ≈ {kw:,.0f} kW</div>
        <div class="ap-tier">{tier_emoji}&nbsp; {tier_label}</div></div>""",
        unsafe_allow_html=True,
    )

    max_scale = max(800.0, float(dataset["Power (hp)"].max()) * 1.05)
    render_power_gauge(hp, max_scale)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"<div class='ap-metric'><div class='v'>{ptw:.1f}</div><div class='l'>HP / Tonne</div></div>", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"<div class='ap-metric'><div class='v'>{values['Performance 0-100 kph (sec)']:.1f}s</div><div class='l'>0–100 km/h</div></div>", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"<div class='ap-metric'><div class='v'>{values['Top speed (kph)']:.0f}</div><div class='l'>Top Speed km/h</div></div>", unsafe_allow_html=True)

    if results_df is not None and len(results_df) > 0 and {"Test Accuracy", "MAE", "RMSE"}.issubset(results_df.columns):
        try:
            best_row = results_df.sort_values("Test Accuracy", ascending=False).iloc[0]
            a,b,c = st.columns(3)
            with a: st.markdown(f"<div class='ap-metric'><div class='v'>{best_row['Test Accuracy']:.1f}%</div><div class='l'>Model R²</div></div>", unsafe_allow_html=True)
            with b: st.markdown(f"<div class='ap-metric'><div class='v'>±{best_row['MAE']:.0f} HP</div><div class='l'>Typical Error</div></div>", unsafe_allow_html=True)
            with c: st.markdown(f"<div class='ap-metric'><div class='v'>{best_row['RMSE']:.0f} HP</div><div class='l'>RMSE</div></div>", unsafe_allow_html=True)
        except Exception:
            pass

    # Save to session garage / favorites
    if "favorite_cars" not in st.session_state:
        st.session_state.favorite_cars = []
    favorite = {
        "Manufacturer": values["Manufacturer"], "Brand": values["Brand"], "Model Year": values["Model Year"],
        "Body Type": values["Body Type"], "Predicted HP": float(hp), "Power/Tonne": float(ptw),
    }
    fav_key = f"{favorite['Manufacturer']}|{favorite['Brand']}|{favorite['Model Year']}"
    existing_keys = {f"{x['Manufacturer']}|{x['Brand']}|{x['Model Year']}" for x in st.session_state.favorite_cars}
    if st.button("☆ Save to REVORA Garage", key="save_prediction_car", use_container_width=True):
        if fav_key not in existing_keys:
            st.session_state.favorite_cars.append(favorite)
            st.success("Vehicle saved to your REVORA Garage.")
        else:
            st.info("This vehicle is already saved.")

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Performance DNA</div>", unsafe_allow_html=True)
    render_performance_dna(hp, values, dataset)
    st.markdown("</div>", unsafe_allow_html=True)

    # Transparent model-level explanation
    if feature_importance is not None:
        try:
            top_features = feature_importance.head(5)
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown("<div class='ap-card-title'>What the Model Learned</div>", unsafe_allow_html=True)
            st.caption("These are the model's strongest global feature signals; they are not a claim about the exact causal reason for this individual prediction.")
            st.dataframe(top_features, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

    render_similar_cars(dataset, values, hp)

    # Find something faster
    candidates = dataset.copy()
    candidates["_ptw"] = candidates["Power (hp)"] / candidates["Weight"].replace(0, np.nan) * 1000
    candidates = candidates[
        (candidates["Power (hp)"] > hp) &
        (candidates["Performance 0-100 kph (sec)"] < values["Performance 0-100 kph (sec)"])
    ].copy()
    if not candidates.empty:
        candidates["_performance_gap"] = (candidates["Power (hp)"]-hp) + (values["Performance 0-100 kph (sec)"]-candidates["Performance 0-100 kph (sec)"])*20
        faster = candidates.sort_values("_performance_gap", ascending=False).head(5)
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>⚡ Find Something Faster</div>", unsafe_allow_html=True)
        st.caption("Vehicles in the dataset with both higher recorded horsepower and quicker 0–100 km/h than the selected configuration.")
        show = faster[["Manufacturer","Brand","Model Year","Power (hp)","Performance 0-100 kph (sec)","Top speed (kph)"]].copy()
        show.columns = ["Manufacturer","Model","Year","HP","0–100 (s)","Top Speed (km/h)"]
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-disclaimer'>This is an ML-based point estimate, not a calibrated confidence interval, and may differ from manufacturer specifications or real-world measurements.</div>", unsafe_allow_html=True)


def render_similar_cars(dataset: pd.DataFrame, values: dict, predicted_hp: float):
    same_maker = dataset[dataset["Manufacturer"] == values["Manufacturer"]].copy()
    pool = same_maker if len(same_maker) >= 3 else dataset
    pool = pool.copy()
    pool["_diff"] = (pool["Power (hp)"] - predicted_hp).abs()
    similar = pool.sort_values("_diff").head(5)
    if similar.empty:
        return

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Similar Cars in the Dataset</div>", unsafe_allow_html=True)
    show = similar[["Manufacturer", "Brand", "Model Year", "Power (hp)"]].rename(
        columns={"Power (hp)": "Recorded Power (hp)"}
    )
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Actual recorded values from the dataset \u2014 not model predictions for these cars.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: MODEL INSIGHTS TAB
# ============================================================================

def render_model_insights(results_df, feature_importance, dataset: pd.DataFrame, best_name: str):
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>\U0001F9E0 About the AI Model</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        - **Model used:** {best_name}
        - **Training dataset size:** {len(dataset):,} vehicles
        - **Target variable:** Power (hp)
        - **Feature preprocessing:** target encoding (Brand_Manufacturer), one-hot encoding
          (Origin Country, Body Type, Additional Type, Gearbox Type),
          median imputation + standard scaling (numeric specs)
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if results_df is not None and {"Model", "Train Accuracy", "Test Accuracy"}.issubset(results_df.columns):
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Train vs Test R&sup2; by Model</div>", unsafe_allow_html=True)
        try:
            melt = results_df.melt(id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy"],
                                    var_name="Dataset", value_name="R2")
            melt["Dataset"] = melt["Dataset"].replace({"Train Accuracy": "Train R\u00b2", "Test Accuracy": "Test R\u00b2"})
            fig = px.bar(melt, x="Model", y="R2", color="Dataset", barmode="group",
                         color_discrete_map={"Train R\u00b2": COLORS["telemetry"], "Test R\u00b2": COLORS["accent"]})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": COLORS["text"]}, height=360, yaxis_range=[0, 100],
                               yaxis_title="R\u00b2 (%)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            fmt = {"Train Accuracy": "{:.2f}%", "Test Accuracy": "{:.2f}%"}
            if "MAE" in results_df.columns:
                fmt["MAE"] = "{:.2f}"
            if "RMSE" in results_df.columns:
                fmt["RMSE"] = "{:.2f}"
            st.dataframe(results_df.rename(columns={"Train Accuracy": "Train R\u00b2 (%)",
                                                      "Test Accuracy": "Test R\u00b2 (%)"}).style.format(
                {"Train R\u00b2 (%)": "{:.2f}", "Test R\u00b2 (%)": "{:.2f}",
                 **({"MAE": "{:.2f}"} if "MAE" in results_df.columns else {}),
                 **({"RMSE": "{:.2f}"} if "RMSE" in results_df.columns else {})}),
                hide_index=True, use_container_width=True)

            best_row = results_df.sort_values("Test Accuracy", ascending=False).iloc[0]
            st.markdown(
                f"<div class='ap-disclaimer'>"
                f"<b>R&sup2;</b> describes how much variance in horsepower the model explains "
                f"(here, about {best_row['Test Accuracy']:.1f}% on held-out test data). "
                + (f"<b>MAE</b> means predictions are typically off by roughly &plusmn;{best_row['MAE']:.0f} HP."
                   if "MAE" in results_df.columns else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning("Could not render the model comparison chart.")
            with st.expander("Technical details"):
                st.code(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    if feature_importance is not None:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>What Influences Horsepower?</div>", unsafe_allow_html=True)
        try:
            top = feature_importance.head(10).sort_values("Importance")
            fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[COLORS["accent"]])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": COLORS["text"]}, height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning("Could not render feature importance.")
            with st.expander("Technical details"):
                st.code(str(e))
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# POWER BATTLE
# ============================================================================

def _car_label(row) -> str:
    return f"{row['Manufacturer']} {row['Brand']} \u2014 {int(row['Model Year'])} [{row['_car_id']}]"


def render_car_card(row: pd.Series, title: str):
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='ap-card-title'>{title}</div>", unsafe_allow_html=True)
    fig = render_vehicle_visual(row["Manufacturer"], row["Brand"], row.get("Body Type", "Sedan"),
                                 row.get("Weight", 1500), row.get("Top speed (kph)", 200),
                                 row.get("Performance 0-100 kph (sec)", 8), row.get("gear_type", "A"),
                                 height_px=220)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div style='text-align:center;'><div class='ap-display' style='font-size:1.3rem;'>"
        f"{row['Manufacturer']}</div><div style='color:{COLORS['text_dim']};'>{row['Brand']} "
        f"&middot; {int(row['Model Year'])}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='ap-div'>", unsafe_allow_html=True)
    spec_rows = [
        ("Body Type", row.get("Body Type", "\u2014")),
        ("Power", f"{row.get('Power (hp)', float('nan')):,.0f} HP"),
        ("Weight", f"{row.get('Weight', float('nan')):,.0f} kg"),
        ("Top Speed", f"{row.get('Top speed (kph)', float('nan')):,.0f} km/h"),
        ("0\u2013100 km/h", f"{row.get('Performance 0-100 kph (sec)', float('nan')):.1f} s"),
        ("Fuel Economy", f"{row.get('Fuel Econ (km/L)', float('nan')):.1f} km/L"),
    ]
    rows_html = "".join(
        f"<div class='ap-spec-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>"
        for k, v in spec_rows
    )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_power_battle(dataset: pd.DataFrame):
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>⚔️ Choose Your Fighters</div>", unsafe_allow_html=True)
    manufacturers = safe_unique(dataset, "Manufacturer", ["Unknown"])
    ca, cb = st.columns(2)

    def choose_battle_car(side, default_mfr_index=0):
        with (ca if side == "A" else cb):
            st.markdown(f"**Car {side}**")
            mfr_key = f"battle_mfr_{side.lower()}"
            brand_key = f"battle_brand_{side.lower()}"
            year_key = f"battle_year_{side.lower()}"
            mfr = st.selectbox(f"Manufacturer ({side})", manufacturers, index=min(default_mfr_index, len(manufacturers)-1), key=mfr_key)
            brands = sorted(dict.fromkeys(str(x).strip() for x in safe_unique(dataset[dataset["Manufacturer"] == mfr], "Brand", ["Unknown"]) if str(x).strip()), key=str.lower)
            brands = brands or ["Unknown"]
            brand = st.selectbox(f"Model ({side})", brands, key=brand_key)
            pool = dataset[(dataset["Manufacturer"] == mfr) & (dataset["Brand"] == brand)].copy()
            pool["_year_numeric"] = pd.to_numeric(pool["Model Year"], errors="coerce")
            years = sorted(pool["_year_numeric"].dropna().astype(int).loc[lambda x: x <= 2026].unique().tolist())
            global_years = pd.to_numeric(dataset["Model Year"], errors="coerce").dropna().astype(int)
            min_year = int(global_years.min()) if not global_years.empty else 2020
            default_year = int(np.clip(2026, min_year, 2026))
            year = st.number_input(f"Model Year ({side})", min_value=min_year, max_value=2026, value=int(default_year), step=1, format="%d", key=year_key)
            year = int(year)
            rows = pool[pool["_year_numeric"] == year]
            if rows.empty:
                st.warning(f"No exact {year} record for {mfr} {brand}. Available years: {', '.join(map(str, years)) if years else 'none'}")
                rows = pool.sort_values("_year_numeric").tail(1)
            if len(rows) > 1:
                ids = rows["_car_id"].tolist()
                cid = st.selectbox(f"Configuration ({side})", ids, format_func=lambda x: f"{x} — {rows.loc[rows['_car_id']==x,'Power (hp)'].iloc[0]:.0f} HP", key=f"battle_dup_{side.lower()}")
                row = rows[rows["_car_id"] == cid].iloc[0]
            else:
                row = rows.iloc[0]
            return row

    row_a = choose_battle_car("A", 0)
    row_b = choose_battle_car("B", 1 if len(manufacturers) > 1 else 0)
    st.markdown("</div>", unsafe_allow_html=True)

    if row_a["_car_id"] == row_b["_car_id"]:
        st.warning("Car A and Car B are the same vehicle record — pick a different Car B for a real battle.")

    c1, c2 = st.columns(2)
    with c1: render_car_card(row_a, "🛡️ Car A")
    with c2: render_car_card(row_b, "🛡️ Car B")

    # Head-to-head metrics
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>📊 Head-to-Head Metrics</div>", unsafe_allow_html=True)
    metrics_data = []
    for col, direction, unit in COMPARISON_METRICS:
        va, vb = row_a.get(col, np.nan), row_b.get(col, np.nan)
        if direction == "higher": better = "A" if va > vb else ("B" if vb > va else "Tie")
        elif direction == "lower": better = "A" if va < vb else ("B" if vb < va else "Tie")
        else: better = "—"
        metrics_data.append({"Metric": col, "Car A": va, "Car B": vb, "Unit": unit, "Better": better if direction != "lower_neutral" else "Context"})
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Normalized radar
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Performance DNA — Head to Head</div>", unsafe_allow_html=True)
    radar_labels, radar_a, radar_b = [], [], []
    for col, direction, unit in COMPARISON_METRICS:
        series = pd.to_numeric(dataset[col], errors="coerce").dropna()
        if series.empty or series.max() == series.min(): continue
        higher = direction != "lower"
        radar_labels.append(col)
        radar_a.append(_safe_norm(row_a.get(col, np.nan), series, higher))
        radar_b.append(_safe_norm(row_b.get(col, np.nan), series, higher))
    if radar_labels:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar_a+[radar_a[0]], theta=radar_labels+[radar_labels[0]], fill="toself", name="Car A", line_color=COLORS["accent"], fillcolor="rgba(255,90,31,.15)"))
        fig.add_trace(go.Scatterpolar(r=radar_b+[radar_b[0]], theta=radar_labels+[radar_labels[0]], fill="toself", name="Car B", line_color=COLORS["telemetry"], fillcolor="rgba(47,212,192,.12)"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100]),bgcolor="rgba(0,0,0,0)"),paper_bgcolor="rgba(0,0,0,0)",font={"color":COLORS["text"]},height=400)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Power-to-weight
    ptw_a = row_a["Power (hp)"] / row_a["Weight"] * 1000 if row_a.get("Weight") else np.nan
    ptw_b = row_b["Power (hp)"] / row_b["Weight"] * 1000 if row_b.get("Weight") else np.nan
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Power-to-Weight</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=["Car A","Car B"],y=[ptw_a,ptw_b],marker_color=[COLORS["accent"],COLORS["telemetry"]],text=[f"{ptw_a:.1f}",f"{ptw_b:.1f}"],textposition="outside"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":COLORS["text"]},height=300,showlegend=False)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)


    # REVORA race result: the faster car wins.
    speed_a = float(row_a.get("Top speed (kph)", 0) or 0)
    speed_b = float(row_b.get("Top speed (kph)", 0) or 0)
    accel_a = float(row_a.get("Performance 0-100 kph (sec)", 99) or 99)
    accel_b = float(row_b.get("Performance 0-100 kph (sec)", 99) or 99)

    if speed_a > speed_b:
        winner = "A"
    elif speed_b > speed_a:
        winner = "B"
    elif accel_a < accel_b:
        winner = "A"
    elif accel_b < accel_a:
        winner = "B"
    else:
        winner = "Tie"

    winner_name = (
        f"{row_a['Manufacturer']} {row_a['Brand']}" if winner == "A"
        else f"{row_b['Manufacturer']} {row_b['Brand']}" if winner == "B"
        else "Perfect Tie"
    )

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>🏁 REVORA SPEED BATTLE</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="battle-stage">
            <div class="battle-lane one"></div>
            <div class="battle-lane two"></div>
            <div class="speed-streak one"></div>
            <div class="speed-streak two"></div>
            <div class="speed-streak three"></div>
            <div class="speed-streak four"></div>
            <div class="battle-name a">CAR A · {speed_a:.0f} KM/H</div>
            <div class="battle-name b">CAR B · {speed_b:.0f} KM/H</div>
            <div class="battle-car a">
                <div class="battle-wheel w1"></div><div class="battle-wheel w2"></div>
            </div>
            <div class="battle-car b">
                <div class="battle-wheel w1"></div><div class="battle-wheel w2"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if winner == "Tie":
        winner_html = """
        <div class="winner-pop">
            <div class="cup">⚖️</div>
            <div class="winner-text">PERFECT TIE</div>
            <div class="winner-meta">SAME TOP SPEED · SAME ACCELERATION</div>
        </div>
        """
    else:
        winning_speed = speed_a if winner == "A" else speed_b
        losing_speed = speed_b if winner == "A" else speed_a
        winner_html = f"""
        <div class="winner-pop">
            <div class="cup">🏆</div>
            <div class="winner-text">{winner_name}</div>
            <div class="winner-meta">CAR {winner} WINS · {winning_speed:.0f} KM/H vs {losing_speed:.0f} KM/H</div>
        </div>
        """

    st.markdown(winner_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Live Battle Telemetry</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"<div class='ap-metric'><div class='v'>{speed_a:.0f} km/h</div><div class='l'>Car A Top Speed</div></div>", unsafe_allow_html=True)
    with b:
        st.markdown(f"<div class='ap-metric'><div class='v'>{speed_b:.0f} km/h</div><div class='l'>Car B Top Speed</div></div>", unsafe_allow_html=True)
    with c:
        gap = abs(speed_a - speed_b)
        st.markdown(f"<div class='ap-metric'><div class='v'>{gap:.0f} km/h</div><div class='l'>Speed Gap</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "battle_history" not in st.session_state:
        st.session_state.battle_history = []
    signature = f"{row_a['_car_id']}|{row_b['_car_id']}"
    if not any(x["signature"] == signature for x in st.session_state.battle_history):
        st.session_state.battle_history.insert(
            0,
            {
                "signature": signature,
                "A": f"{row_a['Manufacturer']} {row_a['Brand']} ({int(row_a['Model Year'])})",
                "B": f"{row_b['Manufacturer']} {row_b['Brand']} ({int(row_b['Model Year'])})",
                "Winner": winner_name,
                "Result": f"{speed_a:.0f}–{speed_b:.0f} km/h",
            },
        )
        st.session_state.battle_history = st.session_state.battle_history[:10]

    if st.session_state.battle_history:
        with st.expander("⚔️ Recent Battle History"):
            st.dataframe(
                pd.DataFrame(st.session_state.battle_history).drop(columns=["signature"]),
                hide_index=True,
                use_container_width=True,
            )



def apply_filters(dataset: pd.DataFrame) -> pd.DataFrame:
    with st.expander("\U0001F50D Advanced Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            makers = ["All"] + safe_unique(dataset, "Manufacturer")
            pick_maker = st.multiselect("Manufacturer", makers[1:], default=[], key="flt_mfr")
            body_types = safe_unique(dataset, "Body Type")
            pick_body = st.multiselect("Body Type", body_types, default=[], key="flt_body")
        with c2:
            origins = safe_unique(dataset, "Origin Country")
            pick_origin = st.multiselect("Origin Country", origins, default=[], key="flt_origin")
            gear_types = safe_unique(dataset, "gear_type")
            pick_gear = st.multiselect("Gear Type", gear_types, default=[], key="flt_gear")
        with c3:
            year_values = pd.to_numeric(dataset["Model Year"], errors="coerce").dropna()
            if not year_values.empty:
                yr_lo, yr_hi = int(year_values.min()), min(int(year_values.max()), 2026)
            else:
                yr_lo, yr_hi = 2020, 2026
            start_year = st.number_input("From Year", min_value=yr_lo, max_value=yr_hi, value=yr_lo, key="flt_yr_start")
            end_year = st.number_input("To Year", min_value=yr_lo, max_value=yr_hi, value=yr_hi, key="flt_yr_end")

        if st.button("Clear Filters", key="flt_clear"):
            for k in ["flt_mfr", "flt_body", "flt_origin", "flt_gear"]:
                st.session_state[k] = []
            st.session_state["flt_yr_start"] = yr_lo
            st.session_state["flt_yr_end"] = yr_hi
            st.rerun()

    filtered = dataset.copy()
    if pick_maker:
        filtered = filtered[filtered["Manufacturer"].isin(pick_maker)]
    if pick_body:
        filtered = filtered[filtered["Body Type"].isin(pick_body)]
    if pick_origin:
        filtered = filtered[filtered["Origin Country"].isin(pick_origin)]
    if pick_gear:
        filtered = filtered[filtered["gear_type"].isin(pick_gear)]
    lo_y, hi_y = min(start_year, end_year), max(start_year, end_year)
    filtered = filtered[(filtered["Model Year"] >= lo_y) & (filtered["Model Year"] <= hi_y)]

    st.caption(f"**{len(filtered):,}** of {len(dataset):,} vehicles match your filters.")
    return filtered


def numeric_columns(dataset: pd.DataFrame):
    cols = []
    for c in NUMERIC_COLS + ["Power (hp)"]:
        if c in dataset.columns and pd.to_numeric(dataset[c], errors="coerce").notna().any():
            cols.append(c)
    return cols


def categorical_columns(dataset: pd.DataFrame):
    return [c for c in (HIGH_CARDINALITY_COLS + LOW_CARDINALITY_COLS) if c in dataset.columns]


def viz_dataset_overview(df: pd.DataFrame):
    if df.empty:
        st.info("No vehicles match the current filters.")
        return
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Overview</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    stats = [
        ("Total Vehicles", f"{len(df):,}"),
        ("Manufacturers", f"{df['Manufacturer'].nunique():,}"),
        ("Brands", f"{df['Brand'].nunique():,}"),
        ("Avg HP", f"{df['Power (hp)'].mean():,.0f}"),
        ("Median HP", f"{df['Power (hp)'].median():,.0f}"),
    ]
    for c, (label, val) in zip(cols, stats):
        with c:
            st.markdown(f"<div class='ap-metric'><div class='v'>{val}</div><div class='l'>{label}</div></div>",
                        unsafe_allow_html=True)
    cols2 = st.columns(5)
    stats2 = [
        ("Max HP", f"{df['Power (hp)'].max():,.0f}"),
        ("Min HP", f"{df['Power (hp)'].min():,.0f}"),
        ("Avg Weight", f"{df['Weight'].mean():,.0f} kg"),
        ("Avg Top Speed", f"{df['Top speed (kph)'].mean():,.0f} km/h"),
        ("Year Range", f"{int(df['Model Year'].min())}\u2013{int(df['Model Year'].max())}"),
    ]
    for c, (label, val) in zip(cols2, stats2):
        with c:
            st.markdown(f"<div class='ap-metric'><div class='v'>{val}</div><div class='l'>{label}</div></div>",
                        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Power Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x="Power (hp)", nbins=40, color_discrete_sequence=[COLORS["accent"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Cars by Manufacturer (Top 10)</div>", unsafe_allow_html=True)
        top_makers = df["Manufacturer"].value_counts().head(10).sort_values()
        fig = px.bar(x=top_makers.values, y=top_makers.index, orientation="h",
                     color_discrete_sequence=[COLORS["telemetry"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=320, xaxis_title="Cars", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Cars by Body Type</div>", unsafe_allow_html=True)
        counts = df["Body Type"].value_counts().sort_values()
        fig = px.bar(x=counts.values, y=counts.index, orientation="h", color_discrete_sequence=[COLORS["accent"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=320, xaxis_title="Cars", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Cars by Year</div>", unsafe_allow_html=True)
        by_year = df.groupby(df["Model Year"].astype(int)).size().reset_index(name="Count")
        fig = px.bar(by_year, x="Model Year", y="Count", color_discrete_sequence=[COLORS["accent"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


def viz_histogram(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    if not num_cols:
        st.info("No numeric columns available.")
        return
    col = st.selectbox("Column", num_cols, index=num_cols.index("Power (hp)") if "Power (hp)" in num_cols else 0)
    bins = st.slider("Bins", 10, 100, 40)
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        st.info("No data for this column with current filters.")
        return
    fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=[COLORS["accent"]])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    cols = st.columns(5)
    stats = [("Mean", series.mean()), ("Median", series.median()), ("Std Dev", series.std()),
             ("Min", series.min()), ("Max", series.max())]
    for c, (label, val) in zip(cols, stats):
        with c:
            st.markdown(f"<div class='ap-metric'><div class='v'>{val:,.1f}</div><div class='l'>{label}</div></div>",
                        unsafe_allow_html=True)


def viz_scatter(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for a scatter plot.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_col = st.selectbox("X Axis", num_cols, index=num_cols.index("Weight") if "Weight" in num_cols else 0)
    with c2:
        y_default = num_cols.index("Power (hp)") if "Power (hp)" in num_cols else min(1, len(num_cols) - 1)
        y_col = st.selectbox("Y Axis", num_cols, index=y_default)
    with c3:
        color_col = st.selectbox("Color By", ["None"] + cat_cols)
    with c4:
        size_options = ["None"] + [c for c in ["Power (hp)", "Weight", "Approx Cost"] if c in num_cols]
        size_col = st.selectbox("Size By", size_options)

    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        st.info("No data available for the selected columns.")
        return

    hover_cols = [c for c in ["Manufacturer", "Brand", "Model Year", "Power (hp)", "Weight", "Body Type"] if c in plot_df.columns]
    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        color=None if color_col == "None" else color_col,
        size=None if size_col == "None" else plot_df[size_col].clip(lower=0),
        hover_data=hover_cols, color_discrete_sequence=CHART_SEQUENCE,
    )

    try:
        xv = plot_df[x_col].astype(float).values
        yv = plot_df[y_col].astype(float).values
        if len(xv) >= 2 and np.std(xv) > 0:
            slope, intercept = np.polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                      line=dict(color=COLORS["redline"], dash="dash"), name="Trend"))
    except Exception:
        pass

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=460)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    try:
        r = plot_df[x_col].astype(float).corr(plot_df[y_col].astype(float))
        st.markdown(f"<div class='ap-metric'><div class='v'>{r:.2f}</div><div class='l'>Pearson Correlation</div></div>",
                    unsafe_allow_html=True)
        st.caption(pearson_label(r))
    except Exception:
        st.caption("Correlation could not be computed for this column pair.")


def viz_correlation_pair(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        st.info("Not enough numeric columns.")
        return
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X Column", num_cols, index=num_cols.index("Weight") if "Weight" in num_cols else 0, key="corr_pair_x")
    with c2:
        y_default = num_cols.index("Power (hp)") if "Power (hp)" in num_cols else min(1, len(num_cols) - 1)
        y_col = st.selectbox("Y Column", num_cols, index=y_default, key="corr_pair_y")

    sub = df[[x_col, y_col, "Manufacturer", "Brand", "Model Year", "Body Type"]].dropna(subset=[x_col, y_col])
    if sub.empty or x_col == y_col:
        st.info("Select two different numeric columns with available data.")
        return

    r = sub[x_col].astype(float).corr(sub[y_col].astype(float))
    fig = px.scatter(sub, x=x_col, y=y_col, color_discrete_sequence=[COLORS["accent"]],
                      hover_data=["Manufacturer", "Brand", "Model Year", "Body Type"])
    try:
        xv, yv = sub[x_col].astype(float).values, sub[y_col].astype(float).values
        if np.std(xv) > 0:
            slope, intercept = np.polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                      line=dict(color=COLORS["telemetry"], dash="dash"), name="Trendline"))
    except Exception:
        pass
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=420)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"<div class='ap-result'><div class='label'>Correlation</div>"
        f"<div class='hp ap-mono' style='font-size:3.2rem;'>{r:.2f}</div>"
        f"<div class='kw'>{pearson_label(r)}</div></div>",
        unsafe_allow_html=True,
    )


def viz_correlation_heatmap(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for a correlation heatmap.")
        return
    scope = st.radio("Feature scope", ["All numeric features", "Selected features"], horizontal=True, key="corr_hm_scope")
    if scope == "Selected features":
        chosen = st.multiselect("Select columns", num_cols, default=num_cols[:min(6, len(num_cols))])
    else:
        chosen = num_cols
    if len(chosen) < 2:
        st.info("Select at least two columns.")
        return

    threshold = st.select_slider("Show only strong correlations \u2265", options=[0.0, 0.50, 0.70, 0.80, 0.90, 0.95], value=0.0)

    corr = df[chosen].apply(pd.to_numeric, errors="coerce").corr()
    display_corr = corr.copy()
    if threshold > 0:
        display_corr = display_corr.where(display_corr.abs() >= threshold)

    fig = px.imshow(display_corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=460)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    pairs = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({"Feature A": chosen[i], "Feature B": chosen[j], "Correlation": val})
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False) if pairs else pd.DataFrame(
        columns=["Feature A", "Feature B", "Correlation"])

    st.markdown("<div class='ap-card-title' style='margin-top:0.6rem;'>Highly Correlated Pairs</div>", unsafe_allow_html=True)
    if pairs_df.empty:
        st.caption("No feature pairs meet the selected threshold.")
    else:
        st.dataframe(pairs_df.style.format({"Correlation": "{:.2f}"}), hide_index=True, use_container_width=True)


def viz_box_plot(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    if not num_cols or not cat_cols:
        st.info("Not enough columns for a box plot.")
        return
    c1, c2 = st.columns(2)
    with c1:
        num_col = st.selectbox("Numeric Column", num_cols, index=num_cols.index("Power (hp)") if "Power (hp)" in num_cols else 0)
    with c2:
        group_col = st.selectbox("Group By", cat_cols, index=cat_cols.index("Manufacturer") if "Manufacturer" in cat_cols else 0)

    sub = df.dropna(subset=[num_col, group_col])
    if sub.empty:
        st.info("No data available for this combination.")
        return
    top_groups = sub[group_col].value_counts().head(12).index.tolist()
    sub = sub[sub[group_col].isin(top_groups)]
    fig = px.box(sub, x=group_col, y=num_col, points="outliers", color=group_col, color_discrete_sequence=CHART_SEQUENCE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=460, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Showing up to the 12 most common groups; points beyond the whiskers are outliers.")


def viz_violin_plot(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    if not num_cols or not cat_cols:
        st.info("Not enough columns for a violin plot.")
        return
    c1, c2 = st.columns(2)
    with c1:
        num_col = st.selectbox("Numeric Column", num_cols, index=num_cols.index("Power (hp)") if "Power (hp)" in num_cols else 0, key="violin_num")
    with c2:
        group_col = st.selectbox("Group By", cat_cols, index=cat_cols.index("Body Type") if "Body Type" in cat_cols else 0, key="violin_grp")

    sub = df.dropna(subset=[num_col, group_col])
    if sub.empty:
        st.info("No data available for this combination.")
        return
    top_groups = sub[group_col].value_counts().head(10).index.tolist()
    sub = sub[sub[group_col].isin(top_groups)]
    fig = px.violin(sub, x=group_col, y=num_col, box=True, points=False, color=group_col,
                     color_discrete_sequence=CHART_SEQUENCE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=460, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def viz_bar_chart(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    if not num_cols or not cat_cols:
        st.info("Not enough columns for a bar chart.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox("Metric", num_cols, index=num_cols.index("Power (hp)") if "Power (hp)" in num_cols else 0, key="bar_metric")
    with c2:
        group_col = st.selectbox("Grouping Column", cat_cols, index=cat_cols.index("Manufacturer") if "Manufacturer" in cat_cols else 0, key="bar_group")
    with c3:
        agg = st.selectbox("Aggregation", ["Mean", "Median", "Min", "Max", "Count"], key="bar_agg")

    sub = df.dropna(subset=[metric, group_col])
    if sub.empty:
        st.info("No data available.")
        return

    if agg == "Count":
        grouped = sub.groupby(group_col).size().reset_index(name=metric)
    else:
        func = {"Mean": "mean", "Median": "median", "Min": "min", "Max": "max"}[agg]
        grouped = sub.groupby(group_col)[metric].agg(func).reset_index()

    ascending = metric == "Performance 0-100 kph (sec)"
    grouped = grouped.sort_values(metric, ascending=ascending).head(15)

    fig = px.bar(grouped, x=group_col, y=metric, color_discrete_sequence=[COLORS["accent"]])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=440)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if metric == "Performance 0-100 kph (sec)":
        st.caption("Sorted ascending \u2014 lower 0\u2013100 time is better.")


def viz_year_analysis(df: pd.DataFrame):
    if df["Model Year"].dropna().empty:
        st.info("No Model Year data available.")
        return
    yr_lo, yr_hi = int(df["Model Year"].min()), int(df["Model Year"].max())
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start Year", min_value=yr_lo, max_value=yr_hi, value=yr_lo, key="ya_start")
    with c2:
        end_year = st.number_input("End Year", min_value=yr_lo, max_value=yr_hi, value=yr_hi, key="ya_end")

    lo_y, hi_y = min(start_year, end_year), max(start_year, end_year)
    sub = df[(df["Model Year"] >= lo_y) & (df["Model Year"] <= hi_y)]
    if sub.empty:
        st.info("No vehicles in this year range.")
        return

    by_year = sub.groupby(sub["Model Year"].astype(int))
    summary = by_year.agg(
        avg_hp=("Power (hp)", "mean"),
        median_hp=("Power (hp)", "median"),
        count=("Power (hp)", "size"),
        avg_top_speed=("Top speed (kph)", "mean"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Average &amp; Median Horsepower by Year</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary["Model Year"], y=summary["avg_hp"], mode="lines+markers",
                                  name="Average HP", line=dict(color=COLORS["accent"])))
        fig.add_trace(go.Scatter(x=summary["Model Year"], y=summary["median_hp"], mode="lines+markers",
                                  name="Median HP", line=dict(color=COLORS["telemetry"])))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=360, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Number of Cars by Year</div>", unsafe_allow_html=True)
        fig = px.bar(summary, x="Model Year", y="count", color_discrete_sequence=[COLORS["accent"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Average Top Speed by Year</div>", unsafe_allow_html=True)
    fig = px.line(summary, x="Model Year", y="avg_top_speed", markers=True, color_discrete_sequence=[COLORS["telemetry"]])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=340)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Top Manufacturer by Year</div>", unsafe_allow_html=True)
    top_by_year = sub.groupby([sub["Model Year"].astype(int), "Manufacturer"]).size().reset_index(name="count")
    top_by_year = top_by_year.sort_values(["Model Year", "count"], ascending=[True, False]).drop_duplicates("Model Year")
    st.dataframe(top_by_year.rename(columns={"count": "Cars"}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def viz_manufacturer_comparison(df: pd.DataFrame):
    if df.empty:
        st.info("No data available.")
        return
    makers = safe_unique(df, "Manufacturer")
    chosen = st.multiselect("Compare Manufacturers", makers, default=makers[:min(5, len(makers))])
    sub = df[df["Manufacturer"].isin(chosen)] if chosen else df
    if sub.empty:
        st.info("No data for the selected manufacturers.")
        return
    summary = sub.groupby("Manufacturer").agg(
        avg_hp=("Power (hp)", "mean"), avg_top_speed=("Top speed (kph)", "mean"),
        avg_0_100=("Performance 0-100 kph (sec)", "mean"), count=("Power (hp)", "size"),
    ).reset_index().sort_values("avg_hp", ascending=False)
    fig = px.bar(summary, x="Manufacturer", y="avg_hp", color="Manufacturer",
                 color_discrete_sequence=CHART_SEQUENCE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(summary.rename(columns={"avg_hp": "Avg HP", "avg_top_speed": "Avg Top Speed",
                                          "avg_0_100": "Avg 0-100 (s)", "count": "Cars"}),
                hide_index=True, use_container_width=True)


def viz_body_type_comparison(df: pd.DataFrame):
    if df.empty:
        st.info("No data available.")
        return
    summary = df.groupby("Body Type").agg(
        avg_hp=("Power (hp)", "mean"), avg_weight=("Weight", "mean"),
        avg_top_speed=("Top speed (kph)", "mean"), count=("Power (hp)", "size"),
    ).reset_index().sort_values("avg_hp", ascending=False)
    fig = px.bar(summary, x="Body Type", y="avg_hp", color="Body Type", color_discrete_sequence=CHART_SEQUENCE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(summary.rename(columns={"avg_hp": "Avg HP", "avg_weight": "Avg Weight",
                                          "avg_top_speed": "Avg Top Speed", "count": "Cars"}),
                hide_index=True, use_container_width=True)


def viz_power_analysis(df: pd.DataFrame):
    if df.empty or df["Power (hp)"].dropna().empty:
        st.info("No power data available.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Power vs Weight</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="Weight", y="Power (hp)", color="Body Type",
                          hover_data=["Manufacturer", "Brand"], color_discrete_sequence=CHART_SEQUENCE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Power vs 0\u2013100 km/h</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="Performance 0-100 kph (sec)", y="Power (hp)", color="Body Type",
                          hover_data=["Manufacturer", "Brand"], color_discrete_sequence=CHART_SEQUENCE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": COLORS["text"]}, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Power-to-Weight Leaders</div>", unsafe_allow_html=True)
    ptw_df = df.copy()
    ptw_df["Power-to-Weight (HP/t)"] = ptw_df["Power (hp)"] / ptw_df["Weight"].replace(0, np.nan) * 1000
    top = ptw_df.dropna(subset=["Power-to-Weight (HP/t)"]).sort_values("Power-to-Weight (HP/t)", ascending=False).head(10)
    st.dataframe(top[["Manufacturer", "Brand", "Model Year", "Power (hp)", "Weight", "Power-to-Weight (HP/t)"]],
                hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def viz_top_cars(df: pd.DataFrame):
    if df.empty:
        st.info("No data available.")
        return
    c1, c2 = st.columns(2)
    with c1:
        top_n = st.selectbox("Top N", [5, 10, 20, 50], index=1)
    with c2:
        metric = st.selectbox("Rank By", ["Power (hp)", "Top speed (kph)", "Performance 0-100 kph (sec)",
                                           "Power-to-Weight", "Fuel Econ (km/L)"])

    ranking_df = df.copy()
    if metric == "Power-to-Weight":
        ranking_df["Power-to-Weight"] = ranking_df["Power (hp)"] / ranking_df["Weight"].replace(0, np.nan) * 1000
        ranking_df = ranking_df.dropna(subset=["Power-to-Weight"])
        top = ranking_df.sort_values("Power-to-Weight", ascending=False).head(top_n)
    elif metric == "Performance 0-100 kph (sec)":
        top = ranking_df.dropna(subset=[metric]).sort_values(metric, ascending=True).head(top_n)
    else:
        top = ranking_df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)

    if top.empty:
        st.info("No vehicles to rank with the current filters.")
        return

    for _, row in top.iterrows():
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = render_vehicle_visual(row["Manufacturer"], row["Brand"], row.get("Body Type", "Sedan"),
                                         row.get("Weight", 1500), row.get("Top speed (kph)", 200),
                                         row.get("Performance 0-100 kph (sec)", 8), row.get("gear_type", "A"),
                                         height_px=160)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.markdown(
                f"<div class='ap-display' style='font-size:1.4rem;'>{row['Manufacturer']} {row['Brand']}</div>"
                f"<div style='color:{COLORS['text_dim']};margin-bottom:0.4rem;'>{int(row['Model Year'])} &middot; {row.get('Body Type','')}</div>"
                f"<div class='ap-spec-row'><span class='k'>Power</span><span class='v'>{row['Power (hp)']:.0f} HP</span></div>"
                f"<div class='ap-spec-row'><span class='k'>Top Speed</span><span class='v'>{row.get('Top speed (kph)', float('nan')):.0f} km/h</span></div>"
                f"<div class='ap-spec-row'><span class='k'>0-100</span><span class='v'>{row.get('Performance 0-100 kph (sec)', float('nan')):.1f} s</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def viz_custom_comparison(df: pd.DataFrame):
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    if len(num_cols) < 1 or not cat_cols:
        st.info("Not enough columns for a custom comparison.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        group_col = st.selectbox("Group By", cat_cols, key="custom_group")
    with c2:
        metric_a = st.selectbox("Metric A", num_cols, key="custom_metric_a")
    with c3:
        metric_b_options = ["None"] + [c for c in num_cols if c != metric_a]
        metric_b = st.selectbox("Metric B (optional)", metric_b_options, key="custom_metric_b")

    sub = df.dropna(subset=[group_col, metric_a])
    if sub.empty:
        st.info("No data for this combination.")
        return
    top_groups = sub[group_col].value_counts().head(12).index.tolist()
    sub = sub[sub[group_col].isin(top_groups)]

    agg_dict = {metric_a: "mean"}
    if metric_b != "None":
        agg_dict[metric_b] = "mean"
    grouped = sub.groupby(group_col).agg(agg_dict).reset_index().sort_values(metric_a, ascending=False)

    fig = go.Figure()
    fig.add_bar(name=metric_a, x=grouped[group_col], y=grouped[metric_a], marker_color=COLORS["accent"])
    if metric_b != "None":
        fig.add_trace(go.Scatter(x=grouped[group_col], y=grouped[metric_b], name=metric_b,
                                  mode="lines+markers", yaxis="y2", line=dict(color=COLORS["telemetry"])))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                       tickfont={"color": COLORS["telemetry"]}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": COLORS["text"]}, height=440, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


VIZ_OPTIONS = {
    "Dataset Overview": viz_dataset_overview,
    "Histogram": viz_histogram,
    "Scatter Plot": viz_scatter,
    "Box Plot": viz_box_plot,
    "Violin Plot": viz_violin_plot,
    "Correlation Heatmap": viz_correlation_heatmap,
    "Correlation Between Two Columns": viz_correlation_pair,
    "Bar Chart": viz_bar_chart,
    "Year Analysis": viz_year_analysis,
    "Manufacturer Comparison": viz_manufacturer_comparison,
    "Body Type Comparison": viz_body_type_comparison,
    "Power Analysis": viz_power_analysis,
    "Top Cars": viz_top_cars,
    "Custom Comparison": viz_custom_comparison,
}


def _render_favorite_garage():
    if not st.session_state.get("favorite_cars"):
        return
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>⭐ My REVORA Garage</div>", unsafe_allow_html=True)
    fav=pd.DataFrame(st.session_state.favorite_cars)
    st.dataframe(fav, hide_index=True, use_container_width=True)
    if st.button("Clear Saved Cars", key="clear_favorites"):
        st.session_state.favorite_cars=[]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_dataset_explorer(dataset: pd.DataFrame):
    _render_favorite_garage()
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>\U0001F697 Explore the Garage</div>", unsafe_allow_html=True)
    filtered = apply_filters(dataset)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    viz_type = st.selectbox("Visualization Type", list(VIZ_OPTIONS.keys()), key="viz_type")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        VIZ_OPTIONS[viz_type](filtered)
    except Exception as e:
        st.error("\u26A0\uFE0F This visualization couldn't be rendered with the current data/filters.")
        with st.expander("Technical details"):
            st.code(str(e))


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="REVORA", page_icon="\U0001F3CE\uFE0F", layout="wide",
                        initial_sidebar_state="expanded")
    inject_css()

    model_dir = _find_model_dir()
    if model_dir is None:
        render_hero()
        st.error(
            "\u26A0\uFE0F Model files not found. Expected a `models/` folder containing "
            "`best_model.joblib`, `dataset.joblib`, `results_df.joblib`, `target_col.joblib` "
            "and `feature_importance.joblib` next to `app.py` (or at `/content/models`). "
            "Run the notebook's export cell first, then place that folder alongside this app."
        )
        st.stop()

    try:
        model = load_model(str(model_dir))
        dataset, results_df, target_col, feature_importance = load_artifacts(str(model_dir))
    except Exception as e:
        render_hero()
        st.error(
            "\u26A0\uFE0F We couldn't load the trained model or its artifacts. "
            "The files may be corrupted or built with an incompatible library version "
            "(this pipeline requires `category_encoders` for the Brand target-encoding step)."
        )
        with st.expander("Technical details"):
            st.code(str(e))
        st.stop()

    if dataset.empty:
        render_hero()
        st.error("\u26A0\uFE0F The saved dataset is empty after cleaning (no rows with a valid Power (hp) value).")
        st.stop()

    missing_cols = [c for c in REQUIRED_MODEL_COLUMNS if c not in dataset.columns]
    if missing_cols:
        render_hero()
        st.error(f"\u26A0\uFE0F The saved dataset is missing expected columns: {missing_cols}")
        st.stop()

    name = best_model_name(results_df)
    render_sidebar(dataset, name)
    render_hero()

    tab_predict, tab_battle, tab_explore, tab_insights = st.tabs(
        ["\U0001F3CE\uFE0F Predictor", "\u2694\uFE0F Power Battle", "\U0001F697 Explore Garage", "\U0001F9E0 Model Insights"]
    )

    with tab_predict:
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### Configure Your Machine")
            values = render_vehicle_selector(dataset)
            predict_clicked = st.button("\u26A1 PREDICT ENGINE POWER", use_container_width=True)

        with right:
            st.markdown("### REVORA Instrument Cluster")
            if predict_clicked:
                try:
                    input_df = build_input_dataframe(values)
                    hp = predict_power(model, input_df)
                    hp = max(hp, 0.0)
                    render_prediction_result(hp, results_df, dataset, values, feature_importance)
                except Exception as e:
                    st.error(
                        "\u26A0\uFE0F We couldn't generate the prediction. "
                        "Please check the vehicle configuration and try again."
                    )
                    with st.expander("Technical details"):
                        st.code(str(e))
            else:
                render_vehicle_preview(values, dataset)

    with tab_battle:
        try:
            render_power_battle(dataset)
        except Exception as e:
            st.error("\u26A0\uFE0F Power Battle couldn't be rendered with the current dataset.")
            with st.expander("Technical details"):
                st.code(str(e))

    with tab_explore:
        render_dataset_explorer(dataset)

    with tab_insights:
        render_model_insights(results_df, feature_importance, dataset, name)

    render_footer()


if __name__ == "__main__":
    main()


