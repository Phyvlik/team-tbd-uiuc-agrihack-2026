import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.coordinate_utils import interp_series
from src.worker_safety import (DECISION_LEVEL, LIGHT, MESSAGES, SafetyConfig, evaluate_safety,
                               place_workers_along_route, score_safety)
from src.ui import (C_SERIES, available_sensor_sets, cached_experiment, current_key, data_selector, fmt, map_figure,
                    page_setup)

page_setup("Worker Safety")
streams = data_selector()
mode, seq = current_key(streams)
t0, t1 = streams.time_span()
dur = t1 - t0

st.markdown("<div class='banner banner-syn'>👷 <b>Worker positions are SIMULATED.</b> CitrusFarm has no annotated "
            "workers and this system does not detect people. Decisions shown are simulated recommendations from "
            "explicit rules — not a certified safety controller and not commands sent to a robot.</div>",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Safety rules")
    cfg = SafetyConfig(
        min_buffer_m=st.slider("Minimum buffer (m)", 0.5, 5.0, 2.0, 0.5),
        k_uncertainty=st.slider("Uncertainty multiplier × r95", 0.0, 3.0, 1.0, 0.25),
        slow_margin_m=st.slider("Slow-down margin (m)", 0.5, 10.0, 3.0, 0.5),
        risk_margin_m=st.slider("Extra margin in high-canopy zones (m)", 0.0, 3.0, 1.0, 0.5),
    )
    proactive = st.checkbox("Proactive mode (widen buffer under dense canopy)", value=streams.available().get("canopy", False),
                            disabled=not streams.available().get("canopy", False),
                            help="Needs the canopy index stream (R-G-NIR camera).")
    st.header("Scenario")
    f1 = st.slider("Crew 1 position (fraction of route)", 0.05, 0.95, 0.40, 0.05)
    f2 = st.slider("Crew 2 position (fraction of route)", 0.05, 0.95, 0.75, 0.05)
    lat = st.slider("Crew distance from robot path (m)", 0.0, 6.0, 2.5, 0.5)
    out_dur = st.select_slider("GPS outage (s)", [0, 10, 30, 60, 120], value=60)

zones = place_workers_along_route(streams.reference, (f1, f2), lat)
# place the outage so it ends as the robot reaches crew 1 (scenario design uses the reference route)
ref = streams.reference
d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ref.x), np.diff(ref.y)))])
t_crew1 = float(ref.t.iloc[int(np.searchsorted(d, f1 * d[-1]))]) - t0
start = max(10.0, t_crew1 - 0.8 * out_dur)
sets = available_sensor_sets(streams)
sens = sets.get("Wheel odometry + IMU + GPS", sets["Wheel odometry + GPS"])
res = cached_experiment(mode, seq, sens, ((start, float(out_dur)),) if out_dur else ())
est = res.estimate
risk = None
if proactive:
    nd = interp_series(est.t.to_numpy(), streams.canopy.t.to_numpy(), streams.canopy.ndvi.to_numpy())
    risk = (np.nan_to_num(nd) > np.nanpercentile(streams.canopy.ndvi, 75)).astype(float)
dec = evaluate_safety(est, res.states, zones, cfg, risk)

t_rel = st.slider("Playback — time since start (s)", 0.0, float(est.t.iloc[-1] - t0),
                  float(min(t_crew1, est.t.iloc[-1] - t0)), 0.5)
i = max(int(np.searchsorted(est.t.to_numpy(), t0 + t_rel, side="right") - 1), 0)
row = dec.iloc[i]
decision = row.decision

# map with workers and buffer
ang = np.linspace(0, 2 * np.pi, 64)
traces = []
for k, z in enumerate(zones):
    traces.append(go.Scatter(x=z.x + z.radius * np.cos(ang), y=z.y + z.radius * np.sin(ang), fill="toself",
                             mode="lines", line=dict(color=C_SERIES[6], width=1), fillcolor="rgba(98,80,214,0.25)",
                             name="Worker zone (SIMULATED)", showlegend=k == 0,
                             hovertemplate=f"{z.zone_id} — simulated crew<extra></extra>"))
    traces.append(go.Scatter(x=[z.x], y=[z.y], mode="text", text=[f"👷 {z.zone_id}"], showlegend=False,
                             hoverinfo="skip"))
col = LIGHT[decision][1]
traces.append(go.Scatter(x=row.x + row.buffer_m * np.cos(ang), y=row.y + row.buffer_m * np.sin(ang), mode="lines",
                         line=dict(color=col, width=3), name=f"Dynamic safety buffer ({row.buffer_m:.2f} m)",
                         hoverinfo="skip"))
fig = map_figure(streams, res, t_now=t0 + t_rel, extra_traces=traces, height=560, show_gps=False)
zx = [z.x for z in zones] + [row.x]
zy = [z.y for z in zones] + [row.y]
span = max(25.0, row.buffer_m * 3)
if st.toggle("Zoom to robot", value=True):
    fig.update_xaxes(range=[row.x - span, row.x + span])
    fig.update_yaxes(range=[row.y - span, row.y + span])

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Worker status light")
    lights = "".join(
        f"<div style='width:62px;height:62px;border-radius:50%;margin:6px auto;background:"
        f"{LIGHT[n][1] if n == decision else 'rgba(82,81,78,0.15)'};border:2px solid rgba(82,81,78,0.4)'></div>"
        for n in ["STOP", "SLOW", "CONTINUE"])
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div style='background:#2b2b2a;border-radius:14px;padding:8px;width:84px'>{lights}</div>",
                unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {MESSAGES[decision]['icon']} {decision}")
        st.caption(f"Reason: {row.reason}")
        st.metric("Buffer radius", f"{row.buffer_m:.2f} m",
                  help="min buffer + multiplier × 95% uncertainty (+ canopy margin when proactive)")
        st.metric("Clearance to nearest crew", fmt(row.clearance_m, 1, " m"))
    st.markdown("**Crew-lead alert**")
    st.markdown(f"🇺🇸 **EN:** {MESSAGES[decision]['en']}")
    st.markdown(f"🇲🇽 **ES:** {MESSAGES[decision]['es']}")
    st.caption("Spanish wording is a draft — to be reviewed by a native speaker / domain lead.")

# decision timeline
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dec.t - t0, y=dec.buffer_m, name="Buffer radius (m)", line=dict(color=C_SERIES[0], width=2)))
fig2.add_trace(go.Scatter(x=dec.t - t0, y=dec.dist_to_worker_m.clip(upper=30), name="Distance to nearest crew (m, capped 30)",
                          line=dict(color=C_SERIES[1], width=2)))
for n, lv in DECISION_LEVEL.items():
    if lv == 0:
        continue
    m = dec.level == lv
    fig2.add_trace(go.Scatter(x=(dec.t - t0)[m], y=np.zeros(m.sum()), mode="markers", name=f"{MESSAGES[n]['icon']} {n}",
                              marker=dict(color=LIGHT[n][1], size=8, symbol="square")))
for a, b in res.outages_abs:
    fig2.add_vrect(x0=a - t0, x1=b - t0, fillcolor="rgba(82,81,78,0.12)", line_width=0, annotation_text="GPS withheld")
fig2.add_vline(x=t_rel, line=dict(color="#52514e", width=1))
fig2.update_layout(height=320, hovermode="x unified", xaxis_title="time since start (s)", yaxis_title="metres",
                   legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig2, use_container_width=True)

sc = score_safety(dec, streams.reference, zones, cfg)
k = st.columns(4)
k[0].metric("True position inside uncertainty margin", f"{sc['true_inside_buffer_pct']:.1f} %",
            help="Share of time the true (RTK) position was within k × r95 of the estimate.")
k[1].metric("Truly unsafe moments (true dist < min buffer)", sc["unsafe_true_samples"])
k[2].metric("…of which decision was CONTINUE (missed)", sc["missed_unsafe"])
k[3].metric("Time with STOP / SLOW recommended", f"{sc['stop_pct']:.0f} % / {sc['slow_pct']:.0f} %")
