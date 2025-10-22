
import os, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

st.set_page_config(page_title="Cyber Lens Dashboard", layout="wide")
st.title("🌐 Cyber Lens — Network Traffic Anomaly Detection")
st.caption("When packets go rogue, we catch them.")

# Data
st.sidebar.header("Data")
default_path = "data/ee7e1853-d39e-48dc-a7c4-1eefedb2b1ee.csv"
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.sidebar.info(f"Loaded default dataset: {default_path}")
else:
    st.warning("Upload a CSV to get started."); st.stop()

# Settings
st.sidebar.header("Model Settings")
contam = st.sidebar.slider("Contamination", 0.01, 0.30, 0.10, 0.01)
use_lof = st.sidebar.checkbox("Add LOF validation", value=True)
neighbors = st.sidebar.slider("LOF neighbors", 5, 50, 20, 1)

# Encode
df_encoded = df.copy()
for c in df_encoded.select_dtypes('bool'):
    df_encoded[c] = df_encoded[c].astype(int)

X_cols = [c for c in df_encoded.columns if c != "label"]
X = df_encoded[X_cols].copy()

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=contam, random_state=42)
iso.fit(X)
df["iso_anomaly"] = pd.Series(iso.predict(X)).replace({1:0, -1:1}).values
df["iso_score"] = -iso.decision_function(X)

# LOF
if use_lof:
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=neighbors, contamination=contam)
    df["lof_anomaly"] = (lof.fit_predict(Xs) == -1).astype(int)
    df["agree"] = (df["lof_anomaly"] == df["iso_anomaly"]).astype(int)
else:
    df["lof_anomaly"] = np.nan; df["agree"] = np.nan

# KPIs
k1,k2,k3,k4 = st.columns(4)
k1.metric("IF anomalies", int(df["iso_anomaly"].sum()))
k2.metric("LOF anomalies", 0 if not use_lof else int(df["lof_anomaly"].sum()))
k3.metric("Agreement %", "—" if not use_lof else f"{df['agree'].mean()*100:.2f}%")
k4.metric("Total", f"{len(df):,}")

# Plots
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
c1,c2 = st.columns(2)
with c1:
    x = "packet_size" if "packet_size" in num_cols else num_cols[0]
    y = "inter_arrival_time" if "inter_arrival_time" in num_cols else (num_cols[1] if len(num_cols)>1 else num_cols[0])
    fig = px.scatter(df, x=x, y=y, color="iso_anomaly", title="IF — Anomaly Scatter", opacity=0.7, color_discrete_map={0:"#16a34a",1:"#ef4444"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    if "spectral_entropy" in df.columns and "frequency_band_energy" in df.columns:
        if use_lof:
            fig2 = px.scatter(df, x="spectral_entropy", y="frequency_band_energy", color="agree", title="Agreement Map", opacity=0.7, color_discrete_map={1:"#2563eb",0:"#fb923c"})
        else:
            fig2 = px.scatter(df, x="spectral_entropy", y="frequency_band_energy", color="iso_anomaly", title="Entropy vs Energy (IF)", opacity=0.7, color_discrete_map={0:"#16a34a",1:"#ef4444"})
        st.plotly_chart(fig2, use_container_width=True)

# SYN distribution
if "tcp_flags_SYN" in df.columns:
    cc = df.groupby(["tcp_flags_SYN","iso_anomaly"]).size().reset_index(name="count")
    cc["SYN"] = cc["tcp_flags_SYN"].map({0:"No SYN",1:"SYN=1"})
    cc["Type"] = cc["iso_anomaly"].map({0:"Normal",1:"Anomaly"})
    st.plotly_chart(px.bar(cc, x="SYN", y="count", color="Type", barmode="group", title="SYN by Type"), use_container_width=True)

# Download
anom_cols = ["iso_score","iso_anomaly","lof_anomaly","agree"] + [c for c in ["packet_size","inter_arrival_time","src_port","dst_port","spectral_entropy","frequency_band_energy"] if c in df.columns]
csv = df.sort_values("iso_score", ascending=False)[anom_cols].to_csv(index=False)
st.download_button("⬇️ Download anomalies CSV", csv, file_name="cyber_lens_anomalies.csv", mime="text/csv")
