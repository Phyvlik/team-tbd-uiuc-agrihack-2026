import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import SensorStreams
from src.reliability_monitor import STATE_LABEL, transitions
from src.sensor_comparison import CONFIG_LABEL, COST_TIERS, compare_configs, recommend, summarize_outages
from src.sensor_fusion import CHI2_2DOF
from src.ui import (C_REF, C_SERIES, SENSOR_NAMES, STREAM_OF, _load, cached_experiment, current_key, data_selector,
                    page_setup)

page_setup("Sensor Intelligence")
streams = data_selector()
mode, seq = current_key(streams)
t0, _ = streams.time_span()

# ---- modalities -------------------------------------------------------------
st.subheader("Sensing modalities")
av = streams.available()
CF_SOURCE = {"odom": "odom_*.bag · /jackal_velocity_controller/odom", "imu": "base_*.bag · /microstrain/imu/data",
             "lidar": "base_*.bag · /velodyne_points → scan matching", "gps": "base_*.bag · /piksi/navsatfix_best_fix"}
rows = []
for k in ["odom", "imu", "lidar", "gps"]:
    s = getattr(streams, STREAM_OF[k])
    have = av.get(STREAM_OF[k])
    rate = f"{1 / np.median(np.diff(s.t)):.0f} Hz" if have else "—"
    status = "✅ loaded" if have else ("🟠 simulated proxy (reference + noise)" if k == "gps" else "⛔ not yet extracted")
    rows.append({"sensor": SENSOR_NAMES[k], "status": status, "samples": len(s) if have else 0, "rate": rate,
                 "CitrusFarm source": CF_SOURCE[k]})
rows.append({"sensor": "GPS receiver quality", "status": "✅ loaded" if av.get("gps_quality") else "⛔ not yet extracted",
             "samples": len(streams.gps_quality) if av.get("gps_quality") else 0, "rate": "—",
             "CitrusFarm source": "base_*.bag · /piksi/debug/receiver_state"})
rows.append({"sensor": "R-G-NIR canopy index", "status": "✅ loaded" if av.get("canopy") else "⛔ not yet extracted",
             "samples": len(streams.canopy) if av.get("canopy") else 0, "rate": "—",
             "CitrusFarm source": "mapir_*.bag · /mapir_cam/image_raw"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---- sensor agreement ---------------------------------------------------------
st.subheader("Sensor agreement — GPS vs. dead-reckoning prediction")
st.markdown("Each GPS fix is checked against the filter's prediction with the Normalised Innovation Squared (NIS) "
            f"test. Fixes above the χ² gate ({CHI2_2DOF[0.999]:.1f}, 99.9 %, 2 dof) are rejected as inconsistent.")
c = st.columns(3)
jump = c[0].checkbox("Inject a 2 m multipath jump at 60 s for 5 s", value=True)
sens = ("odom", "imu", "gps") if av.get("imu") else ("odom", "gps")
res = cached_experiment(mode, seq, sens, (), ((60.0, 5.0, 2.0),) if jump else ())
ev = res.gps_events
fig = go.Figure()
if ev is not None and len(ev):
    fig.add_trace(go.Scattergl(x=ev.t - t0, y=ev.nis, mode="markers", name="NIS per GPS fix",
                               marker=dict(size=5, color=np.where(ev.accepted, C_SERIES[0], C_SERIES[7]))))
fig.add_hline(y=CHI2_2DOF[0.999], line=dict(color=C_REF, dash="dash"), annotation_text="rejection gate")
fig.update_layout(height=300, yaxis_type="log", xaxis_title="time since start (s)", yaxis_title="NIS (log)",
                  margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)
n_rej = int((~ev.accepted).sum()) if ev is not None else 0
st.caption(f"{len(ev) if ev is not None else 0} GPS fixes checked, {n_rej} rejected (red). "
           f"Sensors: {'+'.join(sens)}. GPS source: {res.gps_source}.")

st.markdown("**Degradation events detected by the integrity monitor**")
tr = transitions(res.states)
tr = tr[tr.state != "TRUSTED"].copy()
tr["time (s)"] = (tr.t - t0).round(1)
tr["state"] = tr.state.map(STATE_LABEL)
st.dataframe(tr[["time (s)", "state", "reason", "r95"]].round(3), hide_index=True, use_container_width=True, height=200)

# ---- configuration comparison -----------------------------------------------------
st.subheader("Sensor configurations: accuracy vs. cost (SP5)")
st.markdown("Costs are **qualitative tiers** from the team plan, not prices. Outage robustness = each "
            "configuration's dead-reckoning sensors + GPS under identical outage windows (2 start times × 10/30/60/120 s).")


@st.cache_data(show_spinner="Running all configurations under identical outages (≈40 EKF runs)…")
def _compare(mode, seq):
    return compare_configs(_load(mode, seq), n_starts=2)


if st.button("Run configuration comparison", type="primary") or st.session_state.get(f"cmp_{mode}_{seq}"):
    st.session_state[f"cmp_{mode}_{seq}"] = True
    overall, otab = _compare(mode, seq)
    summ = summarize_outages(otab)
    st.dataframe(overall.round(3), hide_index=True, use_container_width=True)
    if not summ.empty:
        fig = go.Figure()
        for i, cid in enumerate(summ.config.unique()):
            s = summ[summ.config == cid]
            fig.add_trace(go.Scatter(x=s.duration_s, y=s.max_err_m, mode="lines+markers",
                                     name=f"{cid}: {CONFIG_LABEL[cid]} + GPS ({COST_TIERS[cid]['tier']} cost)",
                                     line=dict(color=C_SERIES[i], width=2), marker=dict(size=9)))
        fig.add_hline(y=0.3, line=dict(color=C_REF, dash="dash"), annotation_text="0.3 m target")
        fig.update_layout(height=380, xaxis_title="GPS outage duration (s)", yaxis_title="mean peak error (m, log)",
                          yaxis_type="log", hovermode="x unified", legend=dict(orientation="h", y=1.18),
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.info(recommend(overall, summ, outage_s=30.0))
        st.info(recommend(overall, summ, outage_s=10.0))
        st.dataframe(summ.round(3), hide_index=True, use_container_width=True)
