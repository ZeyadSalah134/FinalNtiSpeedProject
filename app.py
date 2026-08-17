"""
REVORA
Machine-learning powered automotive horsepower predictor.
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
# CONFIG & PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="REVORA | Automotive Performance Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    (0, "City Cruiser", "🚗"),
    (120, "Daily Driver", "🛣️"),
    (200, "Sporty", "🔥"),
    (300, "Performance", "🏎️"),
    (450, "Supercar Territory", "🚀"),
    (600, "Hypercar Beast", "👑"),
]

GEAR_TYPE_LABELS = {
    "A": "Automatic",
    "M": "Manual",
    "AM": "Automated Manual",
    "AT": "Automatic (AT)",
    "CVT": "CVT",
}

COLORS = {
    "bg": "#0B0E14",
    "bg_alt": "#121721",
    "surface": "rgba(255, 255, 255, 0.035)",
    "surface_border": "rgba(255, 255, 255, 0.08)",
    "text": "#F3F5F7",
    "text_dim": "#94A0B0",
    "accent": "#FF5A1F",
    "accent_soft": "rgba(255, 90, 31, 0.15)",
    "telemetry": "#2FD4C0",
    "telemetry_soft": "rgba(47, 212, 192, 0.15)",
    "divider": "rgba(255, 255, 255, 0.06)",
}

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

        #MainMenu, footer, header {{ visibility: hidden; display: none !important; }}

        /* Zero out excess space at the top */
        .block-container {{
            padding-top: 0.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }}

        .ap-display {{
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.04em;
            line-height: 1;
        }}
        .ap-mono {{ font-family: 'JetBrains Mono', monospace; }}

        .ap-hero {{
            padding: 1.5rem 2rem 1.2rem 2rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border: 1px solid {COLORS['surface_border']};
            margin-bottom: 1.2rem;
            margin-top: 0rem;
        }}
        .ap-hero h1 {{ font-size: 2.8rem; margin: 0; color: {COLORS['text']}; }}
        .ap-hero h1 span {{ color: {COLORS['accent']}; }}
        .ap-hero p.tagline {{ font-size: 0.95rem; color: {COLORS['text_dim']}; margin: 0.2rem 0 0.5rem 0; }}

        .ap-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['surface_border']};
            border-radius: 16px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }}
        .ap-card-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: {COLORS['telemetry']};
            margin-bottom: 0.8rem;
        }}

        .car-photo-card {{
            width: 100%;
            height: 280px;
            border-radius: 14px;
            object-fit: cover;
            border: 1px solid {COLORS['surface_border']};
        }}

        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['accent']}, #E8460F);
            color: white; font-weight: 700; font-size: 1.0rem;
            border: none; border-radius: 12px; padding: 0.75rem 1rem; width: 100%;
            box-shadow: 0 8px 24px -6px rgba(255,90,31,0.5);
        }}

        section[data-testid="stSidebar"] {{
            background: {COLORS['bg_alt']};
            border-right: 1px solid {COLORS['surface_border']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ============================================================================
# DATA & MODEL LOADERS
# ============================================================================

def _find_model_dir():
    for d in CANDIDATE_MODEL_DIRS:
        if d.exists() and (d / "best_model.joblib").exists():
            return d
    return None

@st.cache_resource(show_spinner=False)
def load_model(model_dir_str: str):
    return joblib.load(Path(model_dir_str) / "best_model.joblib")

@st.cache_data(show_spinner=False)
def load_artifacts(model_dir_str: str):
    model_dir = Path(model_dir_str)
    dataset = joblib.load(model_dir / "dataset.joblib")
    
    # Normalize columns
    df = dataset.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Manufacturer" not in df.columns and "Brand_Manufacturer" in df.columns:
        parts = df["Brand_Manufacturer"].astype(str).str.split(n=1, expand=True)
        df["Manufacturer"] = parts[0]
        df["Brand"] = parts[1] if parts.shape[1] > 1 else df["Brand_Manufacturer"]

    df = df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC_COLS + ["Power (hp)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Power (hp)" in df.columns:
        df = df.dropna(subset=["Power (hp)"]).reset_index(drop=True)

    return df

model_dir = _find_model_dir()
if model_dir is None:
    st.error("No trained model directory found. Please check your path configuration.")
    st.stop()

dataset = load_artifacts(str(model_dir))
model = load_model(str(model_dir))

# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    f"""
    <div class="ap-hero">
        <h1 class="ap-display"><span>REVORA</span></h1>
        <p class="tagline">{APP_TAGLINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

col_input, col_preview = st.columns([1.2, 1], gap="medium")

with col_input:
    st.markdown("<div class='ap-card-title'>Vehicle Configuration</div>", unsafe_allow_html=True)
    
    # Structured drop-downs replacing free/spread controls
    manufacturers = sorted(dataset["Manufacturer"].dropna().unique().tolist())
    selected_mfg = st.selectbox("Manufacturer", options=manufacturers, index=0)
    
    filtered_brands = sorted(dataset[dataset["Manufacturer"] == selected_mfg]["Brand"].dropna().unique().tolist())
    selected_brand = st.selectbox("Brand / Model", options=filtered_brands if filtered_brands else ["Base"])
    
    body_types = sorted(dataset["Body Type"].dropna().unique().tolist())
    selected_body = st.selectbox("Body Type", options=body_types, index=0)
    
    origins = sorted(dataset["Origin Country"].dropna().unique().tolist())
    selected_origin = st.selectbox("Origin Country", options=origins, index=0)
    
    c1, c2 = st.columns(2)
    with c1:
        gear_type = st.radio("Gear Type", options=list(GEAR_TYPE_LABELS.keys()), format_func=lambda x: GEAR_TYPE_LABELS[x], horizontal=True)
        gear_count = st.slider("Gear Count", min_value=1, max_value=10, value=6)
        model_year = st.slider("Model Year", min_value=1990, max_value=2026, value=2024)
    with c2:
        weight = st.number_input("Weight (kg)", min_value=500, max_value=4000, value=1500, step=50)
        top_speed = st.number_input("Top Speed (km/h)", min_value=80, max_value=500, value=220, step=5)
        accel = st.number_input("0-100 km/h (sec)", min_value=1.5, max_value=25.0, value=6.5, step=0.1)

    fuel_econ_l = st.slider("Fuel Economy (L/100km)", min_value=2.0, max_value=30.0, value=8.5, step=0.5)
    fuel_econ_km = 100.0 / fuel_econ_l if fuel_econ_l > 0 else 10.0
    approx_cost = st.number_input("Approx Cost ($)", min_value=5000, max_value=2000000, value=35000, step=2500)

with col_preview:
    st.markdown("<div class='ap-card-title'>Vehicle Visual & Prediction</div>", unsafe_allow_html=True)
    
    # Check if a photo column exists in the CSV data
    photo_col = next((c for c in dataset.columns if c.lower() in ["image", "photo", "image_url", "car_photo"]), None)
    
    matched_row = dataset[(dataset["Manufacturer"] == selected_mfg) & (dataset["Brand"] == selected_brand)]
    
    if photo_col and not matched_row.empty and pd.notna(matched_row.iloc[0][photo_col]):
        img_url = matched_row.iloc[0][photo_col]
        st.image(img_url, use_column_width=True, caption=f"{selected_mfg} {selected_brand}")
    else:
        # High-style fallback photo card block replacing the 3D canvas
        st.markdown(
            f"""
            <div class="ap-card" style="text-align:center; padding: 2rem 1rem;">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🚗</div>
                <div style="font-family:'Bebas Neue'; font-size:1.8rem; color:{COLORS['text']};">
                    {selected_mfg} {selected_brand}
                </div>
                <div style="color:{COLORS['telemetry']}; font-family:'JetBrains Mono'; font-size:0.85rem;">
                    {selected_body} &middot; {selected_origin}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Prediction Pipeline
    brand_mfg_str = f"{selected_mfg} {selected_brand}".strip()
    input_data = pd.DataFrame([{
        "Brand_Manufacturer": brand_mfg_str,
        "Origin Country": selected_origin,
        "Body Type": selected_body,
        "Additional Type": "Standard",
        "gear_type": gear_type,
        "Approx Cost": approx_cost,
        "Model Year": model_year,
        "Weight": weight,
        "Fuel Econ (L/100km)": fuel_econ_l,
        "Fuel Econ (km/L)": fuel_econ_km,
        "Performance 0-100 kph (sec)": accel,
        "Top speed (kph)": top_speed,
        "gear_count": gear_count,
    }])

    if st.button("RUN PERFORMANCE PREDICTION"):
        pred_hp = model.predict(input_data)[0]
        
        tier_label, tier_emoji = "Sporty", "🔥"
        for min_hp, label, emoji in PERFORMANCE_TIERS:
            if pred_hp >= min_hp:
                tier_label, tier_emoji = label, emoji

        st.markdown(
            f"""
            <div class="ap-card" style="text-align:center; margin-top:1rem;">
                <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:{COLORS['text_dim']};">PREDICTED OUTPUT</div>
                <div style="font-family:'Bebas Neue'; font-size:4.5rem; color:{COLORS['accent']}; line-height:1;">
                    {int(pred_hp)} <span style="font-size:2rem;">HP</span>
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:0.9rem; color:{COLORS['telemetry']};">
                    ~ {int(pred_hp * 0.7457)} kW
                </div>
                <div style="margin-top:0.8rem; display:inline-block; padding:0.3rem 1rem; border-radius:99px; background:{COLORS['accent_soft']}; border:1px solid rgba(255,90,31,0.3); font-size:0.85rem;">
                    {tier_emoji} {tier_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
