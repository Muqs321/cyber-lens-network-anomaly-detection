# app/streamlit_app.py
# Cyber Lens — Streamlit dashboard for network traffic anomaly detection
# Run: streamlit run app/streamlit_app.py

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ------------------------- Page setup -------------------------
st.set_page_config(page_title="Cyber Lens — Network Traffic Anomaly Detection",
                   page_icon="🌐", layout="wide")

st.title("🌐 Cyber Lens — Network Traffic Anomaly Detection")
st.caption("When packets go rogue, we catch them.")

# ------------------------- Sidebar: data -------------------------
st.sidebar.header("Data")
DEFAULT_PATH = "data/ee7e1853-d39e-48dc-a7c4-1eefedb2b1ee.csv"
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def load_csv(path_or_file):
    df = pd.read_csv(path_or_file)
    return df

if uploaded is not None:
    df = load_csv(uploaded)
elif os.path.exists(DEFAULT_PATH):
    df = load_csv(DEFAULT_PATH)
    st.sidebar.success(f"Loaded default dataset: {DEFAULT_PATH}")
else:
    st.info("👋 Upload a CSV to start (columns like packet_size, inter_arrival_time, src_port, dst_port, spectral_entropy, frequency_band_energy, tcp_flags_SYN, ...)")
    st.stop()

# ------------------------- Preprocess -------------------------
df_encoded = df.copy()
for c in df_encoded.select_dtypes('bool'):
    df_encoded[c] = df_encoded[c].astype(int)

feature_cols = [c for c in df_encoded.columns if c != "label"]
X = df_encoded[feature_cols].copy()

# ------------------------- Sidebar: models -------------------------
st.sidebar.header("Models")
contam = st.sidebar.slider("Expected anomaly fraction (contamination)", 0.01, 0.30, 0.10, 0.01)
use_lof = st.sidebar.checkbox("Add LOF validation", value=True)
neighbors = st.sidebar.slider("LOF neighbors", 5, 50, 20, 1)

# ------------------------- Isolation Forest -------------------------
@st.cache_resource(show_spinner=False)
def fit_iforest(X, contamination):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X)
    return model

iso = fit_iforest(X, contam)
iso_pred = iso.predict(X)  # 1 normal, -1 anomaly
df["iso_anomaly"] = pd.Series(iso_pred, index=X.index).replace({1: 0, -1: 1}).astype(int)
df["iso_score"] = -iso.decision_function(X)  # higher = more anomalous

# ------------------------- LOF (optional) -------------------------
if use_lof:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=neighbors, contamination=contam)
    lof_pred = lof.fit_predict(Xs)  # 1 normal, -1 anomaly
    df["lof_anomaly"] = (lof_pred == -1).astype(int)
    df["agree"] = (df["lof_anomaly"] == df["iso_anomaly"]).astype(int)
else:
    df["lof_anomaly"] = np.nan
    df["agree"] = np.nan

# ------------------------- KPIs -------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("IF anomalies", int(df["iso_anomaly"].sum()))
k2.metric("LOF anomalies", int(df["lof_anomaly"].sum()) if use_lof else 0)
k3.metric("Agreement %", f"{df['agree'].mean()*100:.2f}%" if use_lof else "—")
k4.metric("Total", f"{len(df):,}")

with st.expander("🔎 Data Preview", expanded=False):
    st.dataframe(df.head(20))
    meta1, meta2, meta3 = st.columns(3)
    meta1.write(f"**Rows:** {df.shape[0]:,}")
    meta2.write(f"**Columns:** {df.shape[1]:,}")
    meta3.write(f"**Missing values:** {int(df.isnull().sum().sum())}")

st.markdown("---")

# ------------------------- Plots -------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

def pick(pref, fallback_idx=0):
    return pref if pref in num_cols else (num_cols[fallback_idx] if len(num_cols) > fallback_idx else None)

left, right = st.columns(2)

with left:
    x_opt = st.selectbox("X-axis", options=num_cols, index=num_cols.index(pick("packet_size", 0)) if pick("packet_size", 0) in num_cols else 0)
    y_opt = st.selectbox("Y-axis", options=num_cols, index=num_cols.index(pick("inter_arrival_time", 1)) if pick("inter_arrival_time", 1) in num_cols else min(1, len(num_cols)-1))
    fig1 = px.scatter(
        df, x=x_opt, y=y_opt, color="iso_anomaly", opacity=0.75,
        title="IF — Anomaly Scatter",
        color_continuous_scale=["#16a34a", "#ef4444"] if False else None,
        color_discrete_map={0: "#16a34a", 1: "#ef4444"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with right:
    if {"spectral_entropy", "frequency_band_energy"}.issubset(df.columns):
        if use_lof:
            fig2 = px.scatter(
                df, x="spectral_entropy", y="frequency_band_energy",
                color="agree", opacity=0.75, title="Agreement Map (Blue = both models agree)",
                color_discrete_map={1: "#2563eb", 0: "#fb923c"}
            )
        else:
            fig2 = px.scatter(
                df, x="spectral_entropy", y="frequency_band_energy",
                color="iso_anomaly", opacity=0.75, title="Entropy vs Energy (IF)",
                color_discrete_map={0: "#16a34a", 1: "#ef4444"}
            )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Add columns 'spectral_entropy' and 'frequency_band_energy' to see the agreement map.")

st.markdown("---")

# SYN distribution plot
if "tcp_flags_SYN" in df.columns:
    syn = df.groupby(["tcp_flags_SYN", "iso_anomaly"]).size().reset_index(name="count")
    syn["SYN"] = syn["tcp_flags_SYN"].map({0: "No SYN", 1: "SYN=1"})
    syn["Type"] = syn["iso_anomaly"].map({0: "Normal", 1: "Anomaly"})
    fig3 = px.bar(syn, x="SYN", y="count", color="Type", barmode="group", title="SYN Flag Distribution by Type")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------- Download anomalies -------------------------
anom_cols = ["iso_score", "iso_anomaly", "lof_anomaly", "agree"] + [
    c for c in ["packet_size", "inter_arrival_time", "src_port", "dst_port",
                "spectral_entropy", "frequency_band_energy", "tcp_flags_SYN"]
    if c in df.columns
]
anom_df = df.sort_values("iso_score", ascending=False)[anom_cols]
buf = io.StringIO()
anom_df.to_csv(buf, index=False)
st.download_button("⬇️ Download anomalies CSV", buf.getvalue(),
                   file_name="cyber_lens_anomalies.csv", mime="text/csv")

st.caption("© 2025 Muqadas Aijaz — Cyber Lens")
