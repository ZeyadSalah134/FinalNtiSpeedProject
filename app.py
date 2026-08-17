

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Revora",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "best_model.joblib"
DATA_FILE = MODEL_DIR / "dataset.joblib"
RESULTS_FILE = MODEL_DIR / "results_df.joblib"
IMPORTANCE_FILE = MODEL_DIR / "feature_importance.joblib"
TARGET_FILE = MODEL_DIR / "target_col.joblib"

REQUIRED_COLUMNS = [
    "Brand_Manufacturer",
    "Origin Country",
    "Body Type",
    "Additional Type",
    "gear_type",
    "Approx Cost",
    "Model Year",
    "Weight",
    "Fuel Econ (L/100km)",
    "Fuel Econ (km/L)",
    "Performance 0-100 kph (sec)",
    "Top speed (kph)",
    "gear_count",
]

NUMERIC_COLUMNS = [
    "Approx Cost",
    "Model Year",
    "Weight",
    "Fuel Econ (L/100km)",
    "Fuel Econ (km/L)",
    "Performance 0-100 kph (sec)",
    "Top speed (kph)",
    "gear_count",
]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101827, #172554);
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}
.hero {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(37,99,235,.35), rgba(124,58,237,.30));
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 3rem;
    margin: 0;
}
.hero p {
    color: #cbd5e1;
    font-size: 1.05rem;
}
.result-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    text-align: center;
    margin-top: 20px;
}
.result-label {
    color: #dbeafe;
    font-size: 15px;
    font-weight: 700;
}
.result-value {
    color: white;
    font-size: 60px;
    font-weight: 900;
}
.card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(15,23,42,.70);
    border: 1px solid rgba(148,163,184,.15);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return joblib.load(DATA_FILE)


@st.cache_data
def load_results():
    if RESULTS_FILE.exists():
        return joblib.load(RESULTS_FILE)
    return None


@st.cache_data
def load_importance():
    if IMPORTANCE_FILE.exists():
        return joblib.load(IMPORTANCE_FILE)
    return None


@st.cache_data
def load_target():
    if TARGET_FILE.exists():
        return joblib.load(TARGET_FILE)
    return "Power (hp)"


def clean_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Brand_Manufacturer" in df.columns:
        parts = df["Brand_Manufacturer"].astype(str).str.strip().str.split(n=1, expand=True)

        if "Manufacturer" not in df.columns:
            df["Manufacturer"] = parts[0]

        if "Brand" not in df.columns:
            if parts.shape[1] > 1:
                df["Brand"] = parts[1]
            else:
                df["Brand"] = df["Brand_Manufacturer"]

    for col in NUMERIC_COLUMNS + ["Power (hp)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    if "Power (hp)" in df.columns:
        df = df.dropna(subset=["Power (hp)"])

    return df.reset_index(drop=True)


def get_options(df, column, fallback="Unknown"):
    if column not in df.columns:
        return [fallback]

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    values = sorted([x for x in values if x])
    return values if values else [fallback]


def make_input(
    manufacturer,
    brand,
    origin,
    body,
    additional,
    gear_type,
    cost,
    year,
    weight,
    fuel_l,
    fuel_km,
    zero100,
    top_speed,
    gear_count,
):
    return pd.DataFrame([{
        "Brand_Manufacturer": f"{manufacturer} {brand}".strip(),
        "Origin Country": origin,
        "Body Type": body,
        "Additional Type": additional,
        "gear_type": gear_type,
        "Approx Cost": cost,
        "Model Year": year,
        "Weight": weight,
        "Fuel Econ (L/100km)": fuel_l,
        "Fuel Econ (km/L)": fuel_km,
        "Performance 0-100 kph (sec)": zero100,
        "Top speed (kph)": top_speed,
        "gear_count": gear_count,
    }], columns=REQUIRED_COLUMNS)


def normalize_results(results):
    if results is None:
        return None

    results = pd.DataFrame(results).copy()

    # Support the notebook's R2 column names.
    if "Train R2" in results.columns:
        results["Train Accuracy (%)"] = results["Train R2"] * 100

    if "Validation R2" in results.columns:
        results["Validation Accuracy (%)"] = results["Validation R2"] * 100

    if "Test R2" in results.columns:
        results["Test Accuracy (%)"] = results["Test R2"] * 100

    # Support older artifact naming if present.
    if "Train Accuracy" in results.columns and "Train Accuracy (%)" not in results.columns:
        results["Train Accuracy (%)"] = results["Train Accuracy"]

    if "Test Accuracy" in results.columns and "Test Accuracy (%)" not in results.columns:
        results["Test Accuracy (%)"] = results["Test Accuracy"]

    if "Train R2" in results.columns and "Test R2" in results.columns:
        results["Train-Test Gap (%)"] = (
            results["Train R2"] - results["Test R2"]
        ) * 100

    if "Train Accuracy (%)" in results.columns and "Test Accuracy (%)" in results.columns:
        results["Train-Test Gap (%)"] = (
            results["Train Accuracy (%)"] - results["Test Accuracy (%)"]
        )

    return results


try:
    model = load_model()
    df = clean_dataset(load_dataset())
    results = normalize_results(load_results())
    importance = load_importance()
    target_col = load_target()
except Exception as e:
    st.error("The model files could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure the GitHub repository contains the models folder with "
        "best_model.joblib and dataset.joblib."
    )
    st.stop()


# ---------------- Header ----------------

st.markdown("""
<div class="hero">
    <h1>🚗 Car Power AI</h1>
    <p>
        Predict vehicle horsepower from specifications and evaluate the
        machine-learning models using train, validation and test performance.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- Best model ----------------

best_row = None
best_name = "Best Model"

if results is not None and "Test R2" in results.columns:
    best_row = results.sort_values("Test R2", ascending=False).iloc[0]
    best_name = str(best_row["Model"])
elif results is not None and "Test Accuracy (%)" in results.columns:
    best_row = results.sort_values("Test Accuracy (%)", ascending=False).iloc[0]
    best_name = str(best_row["Model"])


# ---------------- Sidebar ----------------

with st.sidebar:
    st.title("🚗 Car Power AI")

    page = st.radio(
        "Navigation",
        [
            "🔮 Prediction",
            "📊 Data Analytics",
            "🏆 Model Check",
            "🌲 Feature Importance",
        ],
    )

    st.markdown("---")
    st.write(f"**Target:** {target_col}")
    st.write(f"**Cars:** {len(df):,}")
    st.write(f"**Best model:** {best_name}")


# ============================================================
# Prediction
# ============================================================

if page == "🔮 Prediction":

    st.subheader("🔮 Predict Horsepower")
    st.caption("Enter the vehicle specifications below.")

    left, right = st.columns(2)

    with left:
        manufacturers = get_options(df, "Manufacturer")
        manufacturer = st.selectbox("🏭 Manufacturer", manufacturers)

        brand_df = df[df["Manufacturer"].astype(str) == str(manufacturer)]
        brands = get_options(brand_df, "Brand", "Unknown")
        brand = st.selectbox("🏷️ Brand", brands)

        origin = st.selectbox(
            "🌍 Origin Country",
            get_options(df, "Origin Country")
        )

        body = st.selectbox(
            "🚘 Body Type",
            get_options(df, "Body Type")
        )

        additional = st.selectbox(
            "🔧 Additional Type",
            get_options(df, "Additional Type")
        )

        gear_type = st.selectbox(
            "⚙️ Gear Type",
            get_options(df, "gear_type", "A")
        )

    with right:

        def numeric_range(column, default_min, default_max):
            s = pd.to_numeric(df[column], errors="coerce").dropna()

            if s.empty:
                return default_min, default_max

            return float(s.min()), float(s.max())

        year_min, year_max = numeric_range("Model Year", 1950, 2030)
        cost_min, cost_max = numeric_range("Approx Cost", 0, 5000000)
        weight_min, weight_max = numeric_range("Weight", 500, 5000)
        fuel_l_min, fuel_l_max = numeric_range("Fuel Econ (L/100km)", 1, 40)
        fuel_km_min, fuel_km_max = numeric_range("Fuel Econ (km/L)", 1, 100)
        zero_min, zero_max = numeric_range(
            "Performance 0-100 kph (sec)", 1, 30
        )
        speed_min, speed_max = numeric_range("Top speed (kph)", 50, 500)

        year_default = int(df["Model Year"].median())
        cost_default = float(df["Approx Cost"].median())
        weight_default = float(df["Weight"].median())
        fuel_l_default = float(df["Fuel Econ (L/100km)"].median())
        fuel_km_default = float(df["Fuel Econ (km/L)"].median())
        zero_default = float(df["Performance 0-100 kph (sec)"].median())
        speed_default = float(df["Top speed (kph)"].median())

        gear_counts = (
            pd.to_numeric(df["gear_count"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        gear_counts = sorted(gear_counts) or [6]
        gear_default = int(np.median(gear_counts))

        year = st.number_input(
            "📅 Model Year",
            min_value=int(year_min),
            max_value=int(year_max),
            value=year_default,
            step=1,
        )

        cost = st.number_input(
            "💰 Approx Cost",
            min_value=float(cost_min),
            max_value=float(cost_max),
            value=cost_default,
        )

        weight = st.number_input(
            "⚖️ Weight",
            min_value=float(weight_min),
            max_value=float(weight_max),
            value=weight_default,
        )

        fuel_l = st.number_input(
            "⛽ Fuel Economy (L/100km)",
            min_value=float(fuel_l_min),
            max_value=float(fuel_l_max),
            value=fuel_l_default,
        )

        fuel_km = st.number_input(
            "⛽ Fuel Economy (km/L)",
            min_value=float(fuel_km_min),
            max_value=float(fuel_km_max),
            value=fuel_km_default,
        )

        zero100 = st.number_input(
            "🏁 0-100 km/h (sec)",
            min_value=float(zero_min),
            max_value=float(zero_max),
            value=zero_default,
        )

        top_speed = st.number_input(
            "💨 Top Speed (km/h)",
            min_value=float(speed_min),
            max_value=float(speed_max),
            value=speed_default,
        )

        gear_count = st.selectbox(
            "⚙️ Number of Gears",
            gear_counts,
            index=gear_counts.index(gear_default)
            if gear_default in gear_counts
            else len(gear_counts) // 2,
        )

    if st.button("🚀 Predict Horsepower", use_container_width=True):

        input_df = make_input(
            manufacturer,
            brand,
            origin,
            body,
            additional,
            gear_type,
            cost,
            year,
            weight,
            fuel_l,
            fuel_km,
            zero100,
            top_speed,
            gear_count,
        )

        try:
            prediction = float(np.asarray(model.predict(input_df)).ravel()[0])

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED ENGINE POWER</div>
                    <div class="result-value">{prediction:.1f} HP</div>
                    <div style="color:#ede9fe">
                        Powered by {best_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Horsepower", f"{prediction:.1f} HP")

            with col2:
                st.metric("Kilowatts", f"{prediction * 0.7457:.1f} kW")

            with col3:
                if prediction < 120:
                    tier = "City Cruiser"
                elif prediction < 200:
                    tier = "Daily Driver"
                elif prediction < 300:
                    tier = "Sporty"
                elif prediction < 450:
                    tier = "Performance"
                elif prediction < 600:
                    tier = "Supercar Territory"
                else:
                    tier = "Hypercar Beast"

                st.metric("Performance Tier", tier)

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


# ============================================================
# Data Analytics
# ============================================================

elif page == "📊 Data Analytics":

    st.subheader("📊 Data Analytics")

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="🔥 Correlation Heatmap",
        )

        fig_corr.update_layout(height=700)

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

        if "Power (hp)" in corr.columns:

            power_corr = (
                corr["Power (hp)"]
                .drop("Power (hp)")
                .sort_values(key=lambda x: x.abs(), ascending=False)
                .reset_index()
            )

            power_corr.columns = ["Feature", "Correlation"]

            fig_power = px.bar(
                power_corr,
                x="Correlation",
                y="Feature",
                orientation="h",
                title="🎯 Correlation with Horsepower",
            )

            st.plotly_chart(
                fig_power,
                use_container_width=True
            )

    st.subheader("Dataset Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Model Check
# ============================================================

elif page == "🏆 Model Check":

    st.subheader("🏆 Train / Validation / Test Model Check")

    if results is None:
        st.warning("results_df.joblib was not found.")
        st.stop()

    display = results.copy()

    wanted = [
        "Model",
        "Train Accuracy (%)",
        "Validation Accuracy (%)",
        "Test Accuracy (%)",
        "Train-Test Gap (%)",
    ]

    available = [c for c in wanted if c in display.columns]

    if available:
        st.dataframe(
            display[available].style.format({
                "Train Accuracy (%)": "{:.2f}%",
                "Validation Accuracy (%)": "{:.2f}%",
                "Test Accuracy (%)": "{:.2f}%",
                "Train-Test Gap (%)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    if {
        "Train Accuracy (%)",
        "Validation Accuracy (%)",
        "Test Accuracy (%)",
    }.issubset(display.columns):

        chart_df = display[
            [
                "Model",
                "Train Accuracy (%)",
                "Validation Accuracy (%)",
                "Test Accuracy (%)",
            ]
        ].melt(
            id_vars="Model",
            var_name="Dataset",
            value_name="Accuracy",
        )

        fig = px.bar(
            chart_df,
            x="Model",
            y="Accuracy",
            color="Dataset",
            barmode="group",
            text_auto=".2f",
            title="📈 Train vs Validation vs Test Accuracy",
        )

        fig.update_yaxes(range=[0, 100])

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if "Train-Test Gap (%)" in display.columns:

        gap_df = display[
            ["Model", "Train-Test Gap (%)"]
        ].sort_values("Train-Test Gap (%)")

        fig_gap = px.bar(
            gap_df,
            x="Model",
            y="Train-Test Gap (%)",
            text_auto=".2f",
            title="⚠️ Train-Test Gap / Overfitting Check",
        )

        st.plotly_chart(
            fig_gap,
            use_container_width=True,
        )

        st.markdown("### Generalization Status")

        for _, row in display.iterrows():

            gap = float(row["Train-Test Gap (%)"])

            if gap <= 5:
                status = "🟢 GOOD GENERALIZATION"
            elif gap <= 10:
                status = "🟡 WATCH — SOME GAP"
            else:
                status = "🔴 OVERFITTING RISK"

            st.write(
                f"**{row['Model']}** — {status} "
                f"| Train: {row['Train Accuracy (%)']:.2f}% "
                f"| Test: {row['Test Accuracy (%)']:.2f}% "
                f"| Gap: {gap:.2f} points"
            )

    st.info(
        "R² is converted to a percentage only for easier display. "
        "For regression, this is not classification accuracy."
    )


# ============================================================
# Feature Importance
# ============================================================

elif page == "🌲 Feature Importance":

    st.subheader("🌲 Feature Importance")

    if importance is None:
        st.warning("feature_importance.joblib was not found.")
        st.stop()

    importance_df = pd.DataFrame(importance).copy()

    if {"Feature", "Importance"}.issubset(importance_df.columns):

        top = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(20)
            .sort_values("Importance")
        )

        fig = px.bar(
            top,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"🌲 Top 20 Features — {best_name}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            importance_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption(
    "🚗 Car Power AI • XGBoost-first regression • "
    "70 / 15 / 15 Train / Validation / Test split"
)

