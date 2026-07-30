"""
app.py – Streamlit web application for Li-ion Battery Remaining Useful Life (RUL) Prediction.
Supports both single-cycle manual prediction and full .mat file upload for analysis.
"""

import os
import io
import numpy as np
import pandas as pd
import scipy.io as sio
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery RUL Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "cycle_num", "mean_voltage", "min_voltage",
    "std_voltage", "max_temp", "mean_temp", "discharge_duration",
]
EOL_CAPACITY = 1.4
BATTERY_IDS  = ["B0005", "B0006", "B0007", "B0018"]


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model_path  = os.path.join(base, "..", "models", "rf_rul.pkl")
    scaler_path = os.path.join(base, "..", "models", "scaler.pkl")
    if not os.path.exists(model_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_discharge_features(mat_bytes: bytes, batt_id: str) -> pd.DataFrame:
    """Parse a .mat file (as bytes) and return per-cycle feature DataFrame."""
    mat = sio.loadmat(io.BytesIO(mat_bytes), struct_as_record=False, squeeze_me=True)
    cycles = mat[batt_id].cycle
    discharge_cycles = [c for c in cycles if c.type == "discharge"]
    total = len(discharge_cycles)
    capacities = [float(dc.data.Capacity) for dc in discharge_cycles]

    records = []
    for i, dc in enumerate(discharge_cycles):
        d   = dc.data
        cap = capacities[i]
        rul = sum(1 for c in capacities[i:] if c >= EOL_CAPACITY)
        v   = np.array(d.Voltage_measured, dtype=float)
        t   = np.array(d.Temperature_measured, dtype=float)
        tv  = np.array(d.Time, dtype=float)

        records.append({
            "cycle_num"         : i + 1,
            "mean_voltage"      : np.mean(v),
            "min_voltage"       : np.min(v),
            "std_voltage"       : np.std(v),
            "max_temp"          : np.max(t),
            "mean_temp"         : np.mean(t),
            "discharge_duration": tv[-1] - tv[0],
            "capacity"          : cap,
            "rul_true"          : rul,
        })
    return pd.DataFrame(records)


def rul_color(rul: float) -> str:
    if rul > 60:  return "🟢"
    if rul > 20:  return "🟡"
    return "🔴"


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔋 Li-ion Battery RUL Predictor")
st.markdown("*Random Forest model trained on NASA Battery Prognostics dataset (B0005–B0018)*")
st.divider()

model, scaler = load_model()
if model is None:
    st.warning(
        "⚠️ Trained model not found. "
        "Run `python src/train.py --data_dir data --models_dir models` first, "
        "then restart the app."
    )

tab1, tab2, tab3 = st.tabs(["📁 Upload .mat File", "🎛️ Manual Input", "📊 Dataset Overview"])


# ── Tab 1: Upload .mat file ──────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a NASA Battery .mat File")
    col_sel, col_up = st.columns([1, 2])

    with col_sel:
        batt_id = st.selectbox("Battery ID in file", BATTERY_IDS)

    with col_up:
        uploaded = st.file_uploader("Choose .mat file", type=["mat"])

    if uploaded and model is not None:
        with st.spinner("Extracting features …"):
            df = extract_discharge_features(uploaded.read(), batt_id)
            X  = scaler.transform(df[FEATURES].values)
            df["rul_pred"] = model.predict(X)

        st.success(f"✅ Processed {len(df)} discharge cycles for **{batt_id}**")

        # KPI row
        latest = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cycles",          f"{len(df)}")
        c2.metric("Current Capacity",      f"{latest['capacity']:.3f} Ahr",
                  delta=f"{latest['capacity'] - df.iloc[0]['capacity']:.3f} Ahr")
        c3.metric("Predicted RUL (latest)", f"{max(0, latest['rul_pred']):.0f} cycles",
                  delta=f"{rul_color(latest['rul_pred'])} {'Good' if latest['rul_pred']>60 else 'Caution' if latest['rul_pred']>20 else 'Near EOL'}")
        c4.metric("True RUL (latest)",      f"{latest['rul_true']:.0f} cycles")

        # Capacity fade chart
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(x=df["cycle_num"], y=df["capacity"],
                                     mode="lines", name="Capacity (Ahr)",
                                     line=dict(color="#2563eb", width=2)))
        fig_cap.add_hline(y=EOL_CAPACITY, line_dash="dash", line_color="red",
                          annotation_text="EOL Threshold (1.4 Ahr)")
        fig_cap.update_layout(title=f"{batt_id} — Capacity Fade",
                               xaxis_title="Cycle Number", yaxis_title="Capacity (Ahr)",
                               template="plotly_white", height=320)
        st.plotly_chart(fig_cap, use_container_width=True)

        # True vs Predicted RUL
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=df["cycle_num"], y=df["rul_true"],
                                     mode="lines", name="True RUL",
                                     line=dict(color="#64748b", width=2)))
        fig_rul.add_trace(go.Scatter(x=df["cycle_num"], y=df["rul_pred"].clip(0),
                                     mode="lines", name="Predicted RUL",
                                     line=dict(color="#2563eb", width=2, dash="dash")))
        fig_rul.update_layout(title=f"{batt_id} — Predicted vs True RUL",
                               xaxis_title="Cycle Number", yaxis_title="RUL (cycles)",
                               template="plotly_white", height=320)
        st.plotly_chart(fig_rul, use_container_width=True)

        # Data table
        with st.expander("📋 View cycle-level data"):
            show_df = df[["cycle_num", "capacity", "mean_voltage", "discharge_duration",
                          "max_temp", "rul_true", "rul_pred"]].copy()
            show_df["rul_pred"] = show_df["rul_pred"].clip(0).round(1)
            st.dataframe(show_df.style.format(precision=3), use_container_width=True)


# ── Tab 2: Manual Input ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Predict RUL from Manual Cycle Measurements")
    st.info("Enter values from a single discharge cycle to get an instant RUL estimate.")

    col1, col2 = st.columns(2)
    with col1:
        cycle_num    = st.number_input("Cycle Number",          min_value=1,    max_value=500, value=50)
        mean_voltage = st.number_input("Mean Voltage (V)",      min_value=2.5,  max_value=4.5, value=3.53, step=0.01)
        min_voltage  = st.number_input("Min Voltage (V)",       min_value=2.0,  max_value=4.5, value=2.65, step=0.01)
        std_voltage  = st.number_input("Voltage Std Dev (V)",   min_value=0.0,  max_value=1.0, value=0.23, step=0.01)
    with col2:
        max_temp  = st.number_input("Max Temperature (°C)",     min_value=20.0, max_value=60.0, value=39.0, step=0.5)
        mean_temp = st.number_input("Mean Temperature (°C)",    min_value=15.0, max_value=55.0, value=32.5, step=0.5)
        duration  = st.number_input("Discharge Duration (sec)", min_value=500,  max_value=8000, value=3600, step=10)

    if st.button("🔮 Predict RUL", type="primary") and model is not None:
        X_input = np.array([[cycle_num, mean_voltage, min_voltage, std_voltage,
                             max_temp, mean_temp, duration]])
        X_scaled = scaler.transform(X_input)
        rul_pred = max(0, model.predict(X_scaled)[0])

        st.divider()
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Predicted RUL", f"{rul_pred:.0f} cycles",
                      delta=f"{rul_color(rul_pred)} {'Healthy' if rul_pred > 60 else 'Caution' if rul_pred > 20 else 'Near EOL'}")
        with col_r2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rul_pred,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Remaining Useful Life (cycles)"},
                gauge={
                    "axis": {"range": [0, 170]},
                    "bar" : {"color": "#2563eb"},
                    "steps": [
                        {"range": [0,   20], "color": "#fee2e2"},
                        {"range": [20,  60], "color": "#fef9c3"},
                        {"range": [60, 170], "color": "#dcfce7"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": 20},
                },
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)


# ── Tab 3: Dataset Overview ──────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance — Leave-One-Battery-Out Cross-Validation")
    st.markdown(
        "The model was evaluated using a **Leave-One-Battery-Out (LOBO)** protocol "
        "— the strictest possible cross-validation for multi-asset datasets."
    )

    perf_data = {
        "Battery": ["B0005", "B0006", "B0007", "B0018", "**Overall**"],
        "MAE (cycles)": [19.05, 7.89, 40.29, 13.74, 20.61],
        "RMSE (cycles)": [22.15, 11.38, 42.28, 16.32, 26.29],
        "R²": [0.711, 0.902, 0.240, 0.762, 0.660],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("""
**Feature Importances (descending):**
| Feature | Importance |
|---|---|
| Mean Voltage | 73.3% |
| Min Voltage | 10.6% |
| Cycle Number | 6.5% |
| Discharge Duration | 6.4% |
| Voltage Std Dev | 1.5% |
| Mean Temperature | 1.2% |
| Max Temperature | 0.6% |
""")

    st.info(
        "B0007 showed the weakest LOBO performance (R²=0.24) due to its significantly "
        "different degradation pattern compared to the other three batteries, "
        "making it harder to predict from out-of-distribution training data."
    )
