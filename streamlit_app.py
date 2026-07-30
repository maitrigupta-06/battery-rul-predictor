"""
streamlit_app.py – Li-ion Battery RUL Predictor
Deploy to Streamlit Cloud: place this file + requirements.txt at repo root.
Upload B0005–B0018 .mat files via the sidebar to run predictions.
"""

import io
import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery RUL Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
EOL_CAPACITY = 1.4
BATTERY_IDS  = ["B0005", "B0006", "B0007", "B0018"]
FEATURES     = [
    "cycle_num", "mean_voltage", "min_voltage",
    "std_voltage", "max_temp", "mean_temp", "discharge_duration",
]
COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_features(mat_bytes: bytes, batt_id: str) -> pd.DataFrame:
    mat    = sio.loadmat(io.BytesIO(mat_bytes), struct_as_record=False, squeeze_me=True)
    cycles = mat[batt_id].cycle
    dc     = [c for c in cycles if c.type == "discharge"]
    caps   = [float(c.data.Capacity) for c in dc]
    rows   = []
    for i, c in enumerate(dc):
        d    = c.data
        v    = np.array(d.Voltage_measured, dtype=float)
        t    = np.array(d.Temperature_measured, dtype=float)
        time = np.array(d.Time, dtype=float)
        rul  = sum(1 for cap in caps[i:] if cap >= EOL_CAPACITY)
        rows.append({
            "cycle_num"         : i + 1,
            "mean_voltage"      : np.mean(v),
            "min_voltage"       : np.min(v),
            "std_voltage"       : np.std(v),
            "max_temp"          : np.max(t),
            "mean_temp"         : np.mean(t),
            "discharge_duration": time[-1] - time[0],
            "capacity"          : caps[i],
            "rul_true"          : rul,
        })
    return pd.DataFrame(rows)


def fit_model(df: pd.DataFrame):
    scaler = StandardScaler()
    X      = scaler.fit_transform(df[FEATURES].values)
    rf     = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, df["rul_true"].values)
    return rf, scaler


def lobo_metrics(df: pd.DataFrame):
    results = []
    for test_b in df["battery"].unique():
        tr  = df[df["battery"] != test_b]
        te  = df[df["battery"] == test_b]
        sc  = StandardScaler()
        Xtr = sc.fit_transform(tr[FEATURES].values)
        Xte = sc.transform(te[FEATURES].values)
        rf  = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(Xtr, tr["rul_true"].values)
        pred = rf.predict(Xte)
        true = te["rul_true"].values
        results.append({
            "Battery": test_b,
            "MAE"    : round(mean_absolute_error(true, pred), 2),
            "RMSE"   : round(mean_squared_error(true, pred) ** 0.5, 2),
            "R²"     : round(r2_score(true, pred), 3),
        })
    return results


def health_badge(rul):
    if rul > 60:  return "🟢 Healthy"
    if rul > 20:  return "🟡 Caution"
    return "🔴 Near EOL"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔋 Battery RUL Predictor")
    st.caption("GAIL Battery Prognostics Dataset")
    st.divider()
    st.subheader("Upload .mat Files")
    uploaded_files = st.file_uploader(
        "B0005 / B0006 / B0007 / B0018",
        type=["mat"], accept_multiple_files=True,
    )
    st.divider()
    st.markdown("""
**Model:** Random Forest (200 trees)  
**Validation:** Leave-One-Battery-Out  
**EOL threshold:** 1.4 Ahr (30% fade)
""")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔋 Li-ion Battery Remaining Useful Life Predictor")
st.markdown("*Random Forest model trained on GAIL Battery Prognostics dataset*")
st.divider()

if not uploaded_files:
    st.info("👈 Upload one or more `.mat` files from the sidebar to begin.")
    st.stop()

# ── Parse uploaded files ──────────────────────────────────────────────────────
dfs_list, parsed_ids = [], []
for uf in uploaded_files:
    batt_id = uf.name.replace(".mat", "").upper()
    if batt_id not in BATTERY_IDS:
        st.sidebar.warning(f"Unrecognised file: {uf.name}")
        continue
    with st.spinner(f"Parsing {batt_id} …"):
        df_b = extract_features(uf.read(), batt_id)
        df_b["battery"] = batt_id
        dfs_list.append(df_b)
        parsed_ids.append(batt_id)

if not dfs_list:
    st.error("No valid battery files found.")
    st.stop()

# ── Train ─────────────────────────────────────────────────────────────────────
df_all = pd.concat(dfs_list, ignore_index=True)

with st.spinner("Training Random Forest …"):
    rf, scaler = fit_model(df_all)
    df_all["rul_pred"] = rf.predict(scaler.transform(df_all[FEATURES].values)).clip(0)

st.success(f"✅ Trained on {len(df_all)} discharge cycles from: {', '.join(parsed_ids)}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Capacity Fade", "📊 RUL Predictions", "🏅 Model Performance", "🎛️ Manual Prediction"
])

# ── Tab 1: Capacity fade ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Capacity Fade Over Discharge Cycles")
    fig = go.Figure()
    for batt, col in zip(parsed_ids, COLORS):
        bdf = df_all[df_all["battery"] == batt]
        fig.add_trace(go.Scatter(
            x=bdf["cycle_num"], y=bdf["capacity"],
            mode="lines", name=batt, line=dict(color=col, width=2),
        ))
    fig.add_hline(y=EOL_CAPACITY, line_dash="dash", line_color="red",
                  annotation_text="EOL Threshold (1.4 Ahr)", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Cycle Number", yaxis_title="Capacity (Ahr)",
                      template="plotly_white", height=400, legend_title="Battery")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(parsed_ids))
    for col_ui, batt in zip(cols, parsed_ids):
        bdf  = df_all[df_all["battery"] == batt]
        fade = ((bdf["capacity"].iloc[0] - bdf["capacity"].iloc[-1]) / bdf["capacity"].iloc[0]) * 100
        col_ui.metric(batt, f"{bdf['capacity'].iloc[-1]:.3f} Ahr", delta=f"-{fade:.1f}% fade")

# ── Tab 2: RUL predictions ────────────────────────────────────────────────────
with tab2:
    st.subheader("Predicted vs True RUL Over Cycles")
    selected = st.selectbox("Select battery", parsed_ids)
    bdf = df_all[df_all["battery"] == selected]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bdf["cycle_num"], y=bdf["rul_true"],
                              mode="lines", name="True RUL",
                              line=dict(color="#64748b", width=2)))
    fig2.add_trace(go.Scatter(x=bdf["cycle_num"], y=bdf["rul_pred"],
                              mode="lines", name="Predicted RUL",
                              line=dict(color="#2563eb", width=2, dash="dash")))
    fig2.update_layout(xaxis_title="Cycle Number", yaxis_title="RUL (cycles)",
                       template="plotly_white", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  f"{mean_absolute_error(bdf['rul_true'], bdf['rul_pred']):.1f} cycles")
    c2.metric("RMSE", f"{mean_squared_error(bdf['rul_true'], bdf['rul_pred'])**0.5:.1f} cycles")
    c3.metric("R²",   f"{r2_score(bdf['rul_true'], bdf['rul_pred']):.3f}")

# ── Tab 3: Model performance ──────────────────────────────────────────────────
with tab3:
    st.subheader("Leave-One-Battery-Out Cross-Validation")
    st.caption("Train on N-1 batteries → test on the held-out one. Strictest cross-battery evaluation.")

    if len(parsed_ids) >= 2:
        with st.spinner("Running LOBO …"):
            lobo_res = lobo_metrics(df_all)
        st.dataframe(pd.DataFrame(lobo_res), use_container_width=True, hide_index=True)

        st.subheader("Feature Importances")
        fi    = dict(zip(FEATURES, rf.feature_importances_))
        labels = {
            "mean_voltage": "Mean Voltage", "min_voltage": "Min Voltage",
            "cycle_num": "Cycle No.", "discharge_duration": "Discharge Duration",
            "std_voltage": "Voltage StDev", "mean_temp": "Mean Temperature",
            "max_temp": "Max Temperature",
        }
        pairs = sorted([(labels[k], v) for k, v in fi.items()], key=lambda x: x[1])
        fig_fi = go.Figure(go.Bar(
            x=[v for _, v in pairs], y=[n for n, _ in pairs],
            orientation="h", marker_color="#2563eb",
            text=[f"{v:.1%}" for _, v in pairs], textposition="outside",
        ))
        fig_fi.update_layout(xaxis_title="Importance", template="plotly_white",
                             height=320, margin=dict(l=150))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Upload at least 2 battery files to run LOBO cross-validation.")

# ── Tab 4: Manual prediction ──────────────────────────────────────────────────
with tab4:
    st.subheader("Single-Cycle Manual Prediction")
    st.caption("Uses the model trained on your uploaded files.")

    col1, col2 = st.columns(2)
    with col1:
        mn_cycle = st.number_input("Cycle Number",           min_value=1,    max_value=500,  value=80)
        mn_mv    = st.number_input("Mean Voltage (V)",       min_value=2.5,  max_value=4.5,  value=3.52, step=0.01)
        mn_minv  = st.number_input("Min Voltage (V)",        min_value=2.0,  max_value=4.5,  value=2.68, step=0.01)
        mn_stdv  = st.number_input("Voltage Std Dev (V)",    min_value=0.0,  max_value=1.0,  value=0.23, step=0.01)
    with col2:
        mn_mxt   = st.number_input("Max Temperature (°C)",   min_value=20.0, max_value=60.0, value=38.5, step=0.5)
        mn_mnt   = st.number_input("Mean Temperature (°C)",  min_value=15.0, max_value=55.0, value=32.0, step=0.5)
        mn_dur   = st.number_input("Discharge Duration (s)", min_value=500,  max_value=8000, value=3580, step=10)

    if st.button("🔮 Predict RUL", type="primary"):
        X_in  = np.array([[mn_cycle, mn_mv, mn_minv, mn_stdv, mn_mxt, mn_mnt, mn_dur]])
        rul_p = max(0, rf.predict(scaler.transform(X_in))[0])

        col_g, col_m = st.columns([1, 2])
        with col_g:
            st.metric("Predicted RUL", f"{rul_p:.0f} cycles", delta=health_badge(rul_p))
        with col_m:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=rul_p,
                title={"text": "Remaining Useful Life (cycles)"},
                gauge={
                    "axis": {"range": [0, 170]},
                    "bar" : {"color": "#2563eb"},
                    "steps": [{"range": [0, 20],  "color": "#fee2e2"},
                               {"range": [20, 60], "color": "#fef9c3"},
                               {"range": [60, 170],"color": "#dcfce7"}],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": 20},
                },
            ))
            fig_g.update_layout(height=240, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_g, use_container_width=True)
