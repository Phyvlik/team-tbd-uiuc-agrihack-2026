"""Worker Safety — turn localization uncertainty into a dynamic safety buffer,
a worker status light and bilingual crew alerts (SIMULATED crews).

Rules come from the backend's worker_safety module:
buffer = min_buffer + k · r95 (+ margin in high-canopy zones);
overlap → STOP, near → SLOW, monitor STOP/DEGRADED escalates the decision.
"""
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import backend as be
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import state as S
from dashboard import team_ui
from dashboard.theme import CRIT, DECISION_STATE, GOOD, INK, STATE, WARN
from src.coordinate_utils import interp_series

ss = st.session_state
if team_ui.active():          # Team Slip-Learning Results: teammate summary tables
    team_ui.safety_page()
    st.stop()
key = ss.source
streams = be.load_streams(key)
info = be.source_info(key)
t0, t1 = streams.time_span()
eng = S.is_engineer()
trusted, stop = S.thresholds()
methods = be.methods_for(streams)
if ss.method not in methods:
    ss.method = methods[-1]
canopy_ok = streams.available().get("canopy", False)

ui.page_header(
    "Crew & farmworker view", "Worker Safety",
    "The robot keeps a safety buffer that <b>grows with its position uncertainty</b>. It slows when the buffer "
    "nears a crew and stops when it overlaps — and tells the crew why, in English and Spanish.",
    [ui.provenance_badge(info.provenance), ui.provenance_badge("sim_workers")])
ui.notice("👷 <b>Crew positions are simulated.</b> CitrusFarm has no annotated workers and this system does not detect "
          "people. Decisions are recommendations from explicit, configurable rules — not a certified safety "
          "controller. The buffer grows with the estimator's 95 % uncertainty, which is <b>not yet calibrated</b> "
          "(docs/UNCERTAINTY_CALIBRATION.md).", "syn")

with st.container(border=True):
    c = st.columns([2, 2, 2, 2.2, 1.4])
    c[0].slider("Crew 1 along route", 0.05, 0.95, step=0.05, format="%.2f", **S.bind("ws_crew1", "ws"),
                help="Position as a fraction of the route length.")
    c[1].slider("Crew 2 along route", 0.05, 0.95, step=0.05, format="%.2f", **S.bind("ws_crew2", "ws"))
    c[2].slider("Crew distance from path", 0.0, 6.0, step=0.5, format="%.1f m", **S.bind("ws_offset", "ws"))
    c[3].segmented_control("GPS outage before crew 1", sorted({0, 10, 30, 60, 120, int(ss.outage_dur)}),
                           format_func=lambda v: "Off" if v == 0 else f"{v}s", required=True,
                           **S.bind("outage_dur", "ws"))
    with c[4]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        demo = st.button("▶ Crew approach", type="primary", width="stretch",
                         help="Replays the robot approaching crew 1 while GPS is out.")
    if eng:
        e = st.columns([2, 2, 2, 2.2, 1.4])
        e[0].slider("Minimum buffer", 0.5, 5.0, step=0.5, format="%.1f m", **S.bind("ws_min_buffer", "ws"))
        e[1].slider("Uncertainty multiplier (× r95)", 0.0, 3.0, step=0.25, **S.bind("ws_k", "ws"))
        e[2].slider("Slow-down margin", 0.5, 10.0, step=0.5, format="%.1f m", **S.bind("ws_slow", "ws"))
        with e[3]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            st.toggle("Proactive mode (canopy)", disabled=not canopy_ok, **S.bind("ws_proactive", "ws"),
                      help="Widen the buffer under dense canopy. Needs the canopy index stream.")
        e[4].selectbox("Localization", methods, format_func=lambda m: be.METHOD_LABEL[m].replace("Wheel ", ""),
                       **S.bind("method", "ws"))

cfg = be.SafetyConfig(min_buffer_m=float(ss.ws_min_buffer), k_uncertainty=float(ss.ws_k),
                      slow_margin_m=float(ss.ws_slow), risk_margin_m=1.0)
ref = streams.reference
zones = be.place_workers_along_route(ref, (float(ss.ws_crew1), float(ss.ws_crew2)), float(ss.ws_offset))
d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ref.x), np.diff(ref.y)))])
t_crew1 = float(ref.t.iloc[min(int(np.searchsorted(d, ss.ws_crew1 * d[-1])), len(ref) - 1)]) - t0
out_d = float(ss.outage_dur)
start = max(10.0, t_crew1 - 0.8 * out_d)
res = be.experiment(key, be.METHODS[ss.method], ((start, out_d),) if out_d else (), (), trusted, stop)
est = res.estimate
proactive = bool(ss.ws_proactive and canopy_ok)
risk = None
if proactive:
    nd = interp_series(est.t.to_numpy(), streams.canopy.t.to_numpy(), streams.canopy.ndvi.to_numpy())
    risk = (np.nan_to_num(nd) > np.nanpercentile(streams.canopy.ndvi, 75)).astype(float)
dec = be.evaluate_safety(est, res.states, zones, cfg, risk)
t_max = float(est.t.iloc[-1] - t0)
if ss.ws_t is None:
    ss.ws_t = max(0.0, t_crew1 - max(out_d, 20.0) - 5.0)
if demo:
    S.set_values(ws_t=max(0.0, start - 5.0), playing="ws")
    st.rerun()

dec_states = SimpleNamespace(
    states=dec.assign(state=dec.decision.map(DECISION_STATE)), outages_abs=res.outages_abs)
DEC_LABEL = {"TRUSTED": "Continue", "DEGRADED": "Slow down", "STOP": "Stop"}


def light_html(decision: str) -> str:
    lamps = [("STOP", CRIT, "STOP"), ("SLOW", WARN, "SLOW"), ("CONTINUE", GOOD, "GO")]
    return "<div class='light'>" + "".join(
        f"<div class='lamp{' on' if decision == k else ''}' style='background:{c};"
        f"{'box-shadow:0 0 28px ' + c + ';' if decision == k else ''}'>{lab}</div>" for k, c, lab in lamps) + "</div>"


def live():
    ui.advance("ws", "ws_t", t_max, 0.5 * ss.speed)
    t_now = t0 + float(ss.ws_t)
    i = be.at_time(dec, t_now)
    row = dec.iloc[i]
    decision = row.decision
    stt = DECISION_STATE[decision]
    color = STATE[stt]["color"]

    bar = ui.playback_controls("ws", "ws_t", t_max, jump_to=[("Crew 1", t_crew1 - 8.0)], label="Playback time")
    bar.segmented_control("Speed", [1, 2, 4, 10], format_func=lambda s: f"{s}×", required=True,
                          label_visibility="collapsed", **S.bind("speed", "ws"))
    left, mid, right = st.columns([2.1, 0.55, 1.25], gap="medium")
    with left:
        extra = []
        for k_, z in enumerate(zones):
            zx, zy = ch.circle(z.x, z.y, z.radius)
            extra.append(go.Scatter(x=zx, y=zy, mode="lines", fill="toself", fillcolor="rgba(238,242,239,0.10)",
                                    line=dict(color=INK, width=1.5, dash="dot"), showlegend=k_ == 0,
                                    name="Crew work zone (simulated)", hoverinfo="skip"))
            extra.append(go.Scatter(x=[z.x], y=[z.y], mode="markers+text", text=[f"👷 Crew {k_ + 1}"],
                                    textposition="bottom center", textfont=dict(color=INK, size=13),
                                    marker=dict(size=10, color=INK, symbol="circle"), showlegend=False,
                                    hovertemplate=f"Crew {k_ + 1} (simulated)<extra></extra>"))
        bx, by = ch.circle(row.x, row.y, row.buffer_m)
        extra.append(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                                name=f"Safety buffer {row.buffer_m:.1f} m ({DEC_LABEL[stt]})", hoverinfo="skip"))
        fig = ch.trajectory_map(streams, res, t_now, height=540, follow=14.0, extra=extra, show_gps=False)
        st.plotly_chart(fig, key="ws_map", theme=None, config={"displaylogo": False, "scrollZoom": True})
    with mid:
        st.markdown("<div class='card-label' style='text-align:center'>Worker light</div>" + light_html(decision),
                    unsafe_allow_html=True)
    with right:
        msg = be.MESSAGES[decision]
        ui.status_hero(stt, row.reason, label="Robot decision (simulated)", title=DEC_LABEL[stt],
                       meaning={"TRUSTED": "Robot continues", "DEGRADED": "Robot slows down",
                                "STOP": "Robot stops"}[stt])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.segmented_control("Alert language", ["English", "Español", "English + Español"], required=True,
                             label_visibility="collapsed", **S.bind("lang", "ws"))
        langs = {"English": ["en"], "Español": ["es"]}.get(ss.lang, ["en", "es"])
        for lg in langs:
            st.markdown(f"<div class='msg' style='border-left:4px solid {color}'>"
                        f"<div class='m-lang'>{'Crew alert · English' if lg == 'en' else 'Alerta al equipo · Español'}"
                        f"</div><div class='m-text'>{msg['icon']} {ui.esc(msg[lg])}</div></div>",
                        unsafe_allow_html=True)
        clearance = row.clearance_m
        ui.kpi_row([
            ui.kpi_html("Distance to crew", ui.fmt(row.dist_to_worker_m, 1, " m", "—"),
                        f"crew {int(row.nearest_zone) + 1}" if row.nearest_zone >= 0 else ""),
            ui.kpi_html("Safety buffer", f"{row.buffer_m:.1f} m",
                        f"{cfg.min_buffer_m:.1f} m + {cfg.k_uncertainty:g} × {row.r95:.2f} m uncertainty"
                        + (" + canopy margin" if proactive and risk is not None and risk[i] else "")),
            ui.kpi_html("Clearance", ui.fmt(clearance, 1, " m", "—"),
                        "buffer overlaps crew" if clearance <= 0 else "free space beyond buffer", accent=color),
        ])
    st.plotly_chart(ch.state_ribbon(dec_states, t0, t_now, height=86, labels=DEC_LABEL), key="ws_ribbon",
                    theme=None, config={"displayModeBar": False})


st.fragment(run_every=0.5 if ss.playing == "ws" else None)(live)()
ui.status_legend([("Safety buffer (dashed, colour = decision)", "#eef2ef", "dash"),
                  ("Crew work zone (simulated, dotted circle)", "#eef2ef", "dot")], map_items=True)

if not eng:
    with st.expander("What do the colours mean for the crew?"):
        st.markdown("- ● **Green / GO** — robot working normally; its position is reliable.\n"
                    "- ▲ **Yellow / SLOW** — robot is near your work area or unsure of its position. Keep clear.\n"
                    "- ■ **Red / STOP** — robot is stopping; do not approach until it is green again.\n\n"
                    "The buffer grows when GPS is lost, so the robot stops *earlier* when it is less sure where it is.")
    st.stop()

# ----------------------------------------------------------- engineer view --
st.divider()
ui.section("Safety evaluation against the RTK truth (offline)", "backend worker_safety.score_safety + plan SP7 metric")
sc = be.score_safety(dec, ref, zones, cfg)
err = res.error
margin_dyn = interp_series(err.t.to_numpy(), est.t.to_numpy(), cfg.k_uncertainty * est.r95.to_numpy())
fixed = float(np.nanmean(margin_dyn))
inside_dyn = 100 * float(np.mean(err.err.to_numpy() <= margin_dyn))
inside_fix = 100 * float(np.mean(err.err.to_numpy() <= fixed))
ui.kpi_row([
    ui.kpi_html("Truth inside dynamic buffer margin", f"{inside_dyn:.1f} %", f"k·r95, k = {cfg.k_uncertainty:g}"),
    ui.kpi_html("…inside a fixed margin of equal mean width", f"{inside_fix:.1f} %", f"{fixed:.2f} m everywhere"),
    ui.kpi_html("Truly unsafe samples", f"{sc['unsafe_true_samples']}", f"true robot < {cfg.min_buffer_m:.1f} m from a crew"),
    ui.kpi_html("…of which decision was CONTINUE", f"{sc['missed_unsafe']}", "should be 0",
                accent=GOOD if sc["missed_unsafe"] == 0 else CRIT),
    ui.kpi_html("Time slowed / stopped", f"{sc['slow_pct']:.1f} % / {sc['stop_pct']:.1f} %", "whole run"),
])
st.plotly_chart(ch.safety_timeline(dec, t0, t0 + float(ss.ws_t), height=280), theme=None, key="ws_tl")
log = dec[dec.level.diff().fillna(1) != 0][["t", "decision", "reason", "buffer_m", "dist_to_worker_m"]].copy()
log["t"] = (log.t - t0).round(1)
ui.df_table(log.rename(columns={"t": "time (s)", "buffer_m": "buffer (m)", "dist_to_worker_m": "distance (m)"})
            .round(2), height=240)
if not canopy_ok:
    st.caption("Proactive (canopy) mode: **Not evaluated** on this source — no canopy index stream.")
