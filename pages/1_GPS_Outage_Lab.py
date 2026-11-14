import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (C_SERIES, available_sensor_sets, cached_experiment, current_key, data_selector, error_figure,
                    fmt, map_figure, page_setup, state_timeline)

page_setup("GPS Outage Lab")
streams = data_selector()
mode, seq = current_key(streams)
t0, t1 = streams.time_span()
dur = t1 - t0
sets = available_sensor_sets(streams)
gps_sets = {k: v for k, v in sets.items() if "gps" in v}

st.markdown("Withhold GPS for a chosen window and watch how each method keeps estimating. GPS rows inside the "
            "window are removed **before** they reach the estimator. Every number here is computed live.")

with st.form("outage"):
    c = st.columns(4)
    method = c[0].selectbox("Localization method", list(gps_sets), index=len(gps_sets) - 1)
    d = c[1].selectbox("Outage duration (s)", [10, 30, 60, 120], index=2)
    start = c[2].slider("Outage start (s)", 10, int(max(20, dur - d - 10)), int(min(120, dur / 3)), 5)
    n_out = c[3].number_input("Number of outages", 1, 4, 1, help="Repeated at equal spacing after the first.")
    c2 = st.columns(4)
    jump = c2[0].checkbox("Add multipath jump (tree canopy)", help="A sudden GPS offset for 5 s, after the outage(s).")
    jump_m = c2[1].slider("Jump size (m)", 0.5, 5.0, 2.0, 0.5)
    sigma = c2[2].slider("GPS proxy noise σ (m)", 0.01, 0.20, 0.03, 0.01,
                         disabled=streams.available().get("gps", False),
                         help="Only used when live GPS is the simulated proxy (raw fixes not yet extracted).")
    log_y = c2[3].checkbox("Log-scale error axis")
    st.form_submit_button("Run experiment", type="primary")

spacing = (dur - start) / n_out
outages = tuple((float(start + k * spacing), float(d)) for k in range(int(n_out)) if start + k * spacing + d < dur)
jumps = ((float(outages[-1][0] + outages[-1][1] + 20), 5.0, float(jump_m)),) if jump else ()
res = cached_experiment(mode, seq, gps_sets[method], outages, jumps, sigma)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.plotly_chart(map_figure(streams, res, height=520), use_container_width=True)
with c2:
    st.plotly_chart(error_figure(res, t0, height=360, log_y=log_y), use_container_width=True)
    st.plotly_chart(state_timeline(res, t0), use_container_width=True)

st.subheader("Outage metrics")
ot = res.outage_table.copy()
if len(ot):
    ot["start_s"] = ot.start_s - t0
    st.dataframe(ot.rename(columns={
        "start_s": "start (s)", "duration_s": "duration (s)", "max_err_m": "peak error (m)",
        "end_err_m": "error at GPS return (m)", "mean_err_m": "mean error (m)",
        "time_to_exceed_s": "time to exceed 0.3 m (s)", "pct_within_target": "% of outage within 0.3 m",
        "growth_rate_m_per_s": "error growth (m/s)", "recovery_s": "recovery after GPS return (s)"}).round(3),
        hide_index=True, use_container_width=True)
m = res.metrics
k = st.columns(5)
k[0].metric("Mean error (full run)", fmt(m["mean_m"], 3, " m"))
k[1].metric("RMSE", fmt(m["rmse_m"], 3, " m"))
k[2].metric("Max error", fmt(m["max_m"], 2, " m"))
k[3].metric("Time within 0.3 m", fmt(m["pct_time_within_target"], 1, " %"))
k[4].metric("GPS fixes rejected (NIS gate)", m["gps_rejected"])

st.subheader("Compare methods under the identical outage")
if st.button("Run all available methods"):
    rows, fig = [], go.Figure()
    for i, (name, sens) in enumerate(gps_sets.items()):
        r = cached_experiment(mode, seq, sens, outages, jumps, sigma)
        o = r.outage_table.iloc[0] if len(r.outage_table) else {}
        rows.append({"method": name, "mean error (m)": r.metrics["mean_m"], "max error (m)": r.metrics["max_m"],
                     "peak in 1st outage (m)": o.get("max_err_m", np.nan),
                     "time to exceed 0.3 m (s)": o.get("time_to_exceed_s", np.nan),
                     "% time within 0.3 m": r.metrics["pct_time_within_target"],
                     "truth inside 95% radius (%)": r.metrics["coverage_r95_pct"]})
        fig.add_trace(go.Scatter(x=r.error.t - t0, y=r.error.err, mode="lines", name=name,
                                 line=dict(color=C_SERIES[i], width=2)))
    fig.add_hline(y=0.3, line=dict(color="#52514e", dash="dash", width=1), annotation_text="0.3 m target")
    for a, b in outages:
        fig.add_vrect(x0=a, x1=a + b, fillcolor="rgba(82,81,78,0.12)", line_width=0)
    fig.update_layout(height=340, hovermode="x unified", xaxis_title="time since start (s)", yaxis_title="error (m)",
                      yaxis_type="log" if log_y else "linear", legend=dict(orientation="h", y=1.15),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
