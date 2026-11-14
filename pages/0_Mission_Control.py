"""Mission Control — live view of one run (entry point: app.py)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.reliability_monitor import STATE_LABEL, transitions
from src.ui import (SENSOR_NAMES, STREAM_OF, available_sensor_sets, cached_experiment, current_key, data_selector,
                    error_figure, fmt, map_figure, page_setup, state_card, state_timeline)

page_setup("Mission Control")
streams = data_selector()
mode, seq = current_key(streams)
t0, t1 = streams.time_span()
dur = t1 - t0

RECOMMEND = {
    "TRUSTED": "Continue autonomous operation. Position estimate meets the configured integrity limits.",
    "DEGRADED": "Increase operator attention. Reduce speed and avoid entering occupied rows (simulated recommendation).",
    "STOP": "Stop recommended (simulated). Wait for GPS re-acquisition or take manual control before continuing.",
}

with st.sidebar:
    st.header("Scenario")
    sets = available_sensor_sets(streams)
    names = [k for k in sets if "gps" in sets[k]]
    choice = st.selectbox("Localization method (EKF sensors)", names, index=len(names) - 1)
    out_dur = st.select_slider("GPS outage duration (s)", [0, 10, 30, 60, 120], value=60)
    out_start = st.slider("Outage start (s after start)", 10, int(max(20, dur - 130)), int(min(150, dur / 3)), 5)

outages = ((float(out_start), float(out_dur)),) if out_dur > 0 else ()
res = cached_experiment(mode, seq, sets[choice], outages)
est = res.estimate

t_rel = st.slider("Playback — time since start (s)", 0.0, float(est.t.iloc[-1] - t0),
                  float(min(out_start + out_dur * 0.9 if out_dur else dur / 2, est.t.iloc[-1] - t0)), 0.5,
                  help="Scrub through the run. Everything on this page reflects the estimator at this moment.")
t_now = t0 + t_rel
i = int(np.searchsorted(est.t.to_numpy(), t_now, side="right") - 1)
row, srow = est.iloc[max(i, 0)], res.states.iloc[max(i, 0)]
e_now = res.error[res.error.t <= t_now]
err_now = float(e_now.err.iloc[-1]) if len(e_now) else float("nan")
gps_ok = row.time_since_gps <= 2.0

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(map_figure(streams, res, t_now=t_now, height=600), use_container_width=True)
with right:
    st.subheader("Navigation state")
    state_card(srow.state, srow.reason)
    st.markdown(f"**Operator recommendation:** {RECOMMEND[srow.state]}")
    c1, c2 = st.columns(2)
    c1.metric("95% position uncertainty", fmt(row.r95, 2, " m"))
    c2.metric("GPS", "Available" if gps_ok else "Unavailable",
              delta=None if gps_ok else f"{row.time_since_gps:.0f} s since last fix", delta_color="inverse")
    c1.metric("Estimated position (x, y)", f"{row.x_ant:.1f}, {row.y_ant:.1f} m")
    c2.metric("Actual error (offline check)", fmt(err_now, 2, " m"),
              help="Only computable offline because the RTK reference is available. A deployed robot would "
                   "only see the uncertainty estimate.")
    c1.metric("Heading", f"{np.degrees(row.theta):.0f}°")
    c2.metric("Speed (est.)", fmt(row.v, 2, " m/s"))
    st.markdown("**Sensor availability**")
    av = streams.available()
    chips = []
    for k in ["odom", "imu", "lidar", "gps"]:
        used = k in res.sensors
        have = av.get(STREAM_OF[k]) or (k == "gps")
        if k == "gps" and used:
            lbl = "🟢 live" if gps_ok else "⚪ withheld"
            src = " (simulated proxy)" if res.gps_source == "proxy_from_reference" else ""
            chips.append(f"<span class='chip'>{lbl} {SENSOR_NAMES[k]}{src}</span>")
        elif used:
            chips.append(f"<span class='chip'>🟢 {SENSOR_NAMES[k]}</span>")
        else:
            chips.append(f"<span class='chip'>{'⚪ not used' if have else '⛔ not extracted'} — {SENSOR_NAMES[k]}</span>")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

st.subheader("Localization integrity over time")
st.plotly_chart(state_timeline(res, t0), use_container_width=True)
st.plotly_chart(error_figure(res, t0, t_now=t_now), use_container_width=True)

c1, c2 = st.columns([2, 3])
with c1:
    st.markdown("**Alert log** (state changes)")
    log = transitions(res.states)
    log = log[log.t <= t_now].copy()
    log["time (s)"] = (log.t - t0).round(1)
    log["state"] = log.state.map(STATE_LABEL)
    st.dataframe(log[["time (s)", "state", "reason", "r95"]].rename(columns={"r95": "r95 (m)"}).round(2),
                 hide_index=True, use_container_width=True, height=240)
with c2:
    st.markdown("**Run summary** (full run, computed against the reference)")
    m = res.metrics
    k = st.columns(4)
    k[0].metric("Mean error", fmt(m["mean_m"], 3, " m"))
    k[1].metric("Max error", fmt(m["max_m"], 2, " m"))
    k[2].metric("Time within 0.3 m", fmt(m["pct_time_within_target"], 1, " %"))
    k[3].metric("Truth inside 95% radius", fmt(m["coverage_r95_pct"], 1, " %"),
                help="Consistency of the uncertainty estimate: ~95% expected; lower = overconfident.")
    if len(res.outage_table):
        o = res.outage_table.iloc[0]
        w = res.warnings["per_outage"][0] if res.warnings["per_outage"] else {}
        st.markdown(
            f"During the **{o.duration_s:.0f} s outage**: peak error **{o.max_err_m:.2f} m**, error exceeded 0.3 m after "
            f"**{fmt(o.time_to_exceed_s, 1, ' s')}**; monitor warned after **{fmt(w.get('detect_latency_s'), 1, ' s')}** "
            f"and recommended stop after **{fmt(w.get('stop_latency_s'), 1, ' s')}**; recovery after GPS return "
            f"**{fmt(o.recovery_s, 1, ' s')}**.")
