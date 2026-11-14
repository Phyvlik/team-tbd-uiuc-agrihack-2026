"""Streamlit views for the "Team Slip-Learning Results" source.

The teammate's CSVs (dashboard/team_slip.py) drive the replay directly: robot pose
(x_est, y_est, yaw_est), navigation state, GPS outage flag, model uncertainty
(2*sigma), safety bubble (bubble_m), simulated workers and simulated spray outcomes.
Nothing here re-estimates anything. One replay clock (session key ``team_t``,
seconds since the first gt.csv sample) drives the 3D view, metrics, checklist,
alerts and events.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import charts as ch
from . import components as ui
from . import opui, ops
from . import orchard as orc
from . import scene3d
from . import state as S
from . import team_slip as ts
from .theme import CRIT, GOOD, INK, MUTED, REF, SERIES, STATE, WARN

PAGE, TKEY = "team", "team_t"
EKF_LABEL = "Backend EKF (live)"
SOURCES = (EKF_LABEL, ts.SOURCE_LABEL)
DEMO = {"sequence": "02_13B_Jackal", "mode": ts.SAFE, "start": 50.0, "end": 100.0, "speed": 4}
ACTION_THEME = {ts.GO: "TRUSTED", ts.SLOW: "DEGRADED", ts.STOP: "STOP"}
ACTION_MEANING = {ts.GO: "Continue normal operation", ts.SLOW: "Reduce speed and stay alert",
                  ts.STOP: "Stop and wait for the situation to clear"}
SPRAY_CODE = {"unreached": orc.NOT_REACHED, "correct": orc.TREATED, "missed": orc.MISSED, "double": orc.DOUBLE,
              "wrong_tree": orc.WRONG_TREE}
SPRAY_TEXT = {"correct": "Correct dose", "wrong_tree": "Wrong tree", "missed": "Missed", "double": "Double dose",
              "unreached": "Not reached yet"}
# sidebar key for the teammate's states (same colours, teammate's names and 2*sigma limits)
STATE_KEY = {
    "TRUSTED": {**STATE["TRUSTED"], "label": "Trusted", "meaning": "model 2σ ≤ 0.3 m"},
    "DEGRADED": {**STATE["DEGRADED"], "label": "Degraded", "meaning": "0.3 m < model 2σ ≤ 1.5 m"},
    "STOP": {**STATE["STOP"], "label": "Lost", "meaning": "model 2σ > 1.5 m"},
}


def active() -> bool:
    return st.session_state.get("results_src") == ts.SOURCE_LABEL


# ------------------------------------------------------------------ cached data --

@st.cache_data(show_spinner=False)
def inventory() -> ts.Inventory:
    return ts.discover()


@st.cache_data(show_spinner="Loading teammate results…")
def load(seq: str) -> ts.TeamRun:
    return ts.load_run(seq)


@st.cache_data(show_spinner=False)
def decisions(seq: str, mode: str) -> pd.DataFrame:
    return ts.decision_series(load(seq), mode)


@st.cache_data(show_spinner=False)
def summaries() -> dict:
    return ts.load_summaries()


@st.cache_data(show_spinner=False)
def scene_frames(seq: str):
    """Teammate columns renamed to what scene3d draws (no values changed)."""
    d = load(seq).dash
    est = pd.DataFrame({"t": d.t, "x_ant": d.x_est, "y_ant": d.y_est, "theta": d.yaw_est, "r95": 2 * d.sigma})
    ref = pd.DataFrame({"t": d.t, "x": d.x_true, "y": d.y_true})
    return est, ref, pd.DataFrame({"level": d.state})


def sequences() -> list[str]:
    return inventory().sequences


def seq_label(seq: str) -> str:
    return f"{seq.replace('_Jackal', '')} · teammate results"


# --------------------------------------------------------------------- helpers --

def _status_codes(run: ts.TeamRun, t: float, show_final: bool) -> tuple[np.ndarray, np.ndarray]:
    labels = ts.tree_status_at(run, t, show_final)
    return labels, np.array([SPRAY_CODE[s] for s in labels], int)


def _visible_workers(run: ts.TeamRun, t: float) -> list:
    if not len(run.crew):
        return []
    vis = ts.worker_visible(run.crew, [t])[:, 0]
    return [SimpleNamespace(x=float(w.x), y=float(w.y), radius=ts.WORKER_RADIUS_M, k=k)
            for k, w in enumerate(run.crew.itertuples()) if vis[k]]


def scene_figure(seq: str, t: float, mode: str, *, show_final: bool, show_reference: bool, overview: bool,
                 height: int = 470) -> go.Figure:
    run, dec = load(seq), decisions(seq, mode)
    est, ref, states = scene_frames(seq)
    _, codes = _status_codes(run, t, show_final)
    res = SimpleNamespace(estimate=est, states=states, outages_abs=run.outages)
    dec_ns = pd.DataFrame({"t": dec.t, "buffer_m": dec.bubble_m, "decision": dec.action})
    i = run.index_at(t)
    return scene3d.orchard_scene(SimpleNamespace(reference=ref), res, t, SimpleNamespace(trees=run.trees), codes,
                                 dec=dec_ns, zones=_visible_workers(run, t), show_reference=show_reference,
                                 height=height, overview=overview,
                                 uncertainty_radius=float(2 * run.dash.sigma.iloc[i]))


def map_figure(seq: str, t: float, mode: str, *, show_final: bool, show_reference: bool, follow: float | None,
               height: int = 470) -> go.Figure:
    """2D top view, equal axes, metres in the gt.csv frame."""
    run, dec = load(seq), decisions(seq, mode)
    d = run.dash
    i = run.index_at(t)
    fig = go.Figure()
    labels, _ = _status_codes(run, t, show_final)
    for lab in ("unreached", *ts.SPRAY_STATUSES):
        m = labels == lab
        if m.any():
            fig.add_trace(go.Scatter(x=run.trees.x[m], y=run.trees.y[m], mode="markers", name=SPRAY_TEXT[lab],
                                     marker=dict(size=8, color=scene3d.SPRAY_COLOR[SPRAY_CODE[lab]], opacity=0.85),
                                     hovertemplate=f"{SPRAY_TEXT[lab]} (simulated)<extra></extra>"))
    if show_reference:
        fig.add_trace(go.Scatter(x=d.x_true, y=d.y_true, mode="lines", name="RTK reference (gt.csv)",
                                 line=dict(color=REF, width=1.5, dash="dot"), hoverinfo="skip"))
    past = d.iloc[: i + 1]
    for lvl in (0, 1, 2):
        y = np.where(past.state == lvl, past.y_est, np.nan)
        fig.add_trace(go.Scatter(x=past.x_est, y=y, mode="lines", name=f"Estimate · {ts.STATE_NAMES[lvl]}",
                                 line=dict(color=STATE[ts.STATE_THEME[lvl]]["color"], width=3), hoverinfo="skip",
                                 connectgaps=False))
    r = d.iloc[i]
    for rad, col, dash, name in ((2 * r.sigma, STATE[ts.STATE_THEME[int(r.state)]]["color"], "solid",
                                  "Model uncertainty (2σ)"),
                                 (r.bubble_m, scene3d.DECISION_COLOR[dec.action.iloc[i]], "dash", "Safety bubble")):
        cx, cy = ch.circle(r.x_est, r.y_est, rad)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=name, line=dict(color=col, dash=dash, width=2),
                                 hoverinfo="skip"))
    for w in _visible_workers(run, t):
        cx, cy = ch.circle(w.x, w.y, w.radius)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", fill="toself", fillcolor="rgba(238,242,239,0.12)",
                                 line=dict(color=INK, dash="dot"), name="Simulated worker (2 m zone)",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r.x_est], y=[r.y_est], mode="markers", name="Robot (estimate)",
                             marker=dict(size=13, color=STATE[ts.STATE_THEME[int(r.state)]]["color"],
                                         symbol="triangle-up", angle=90 - np.degrees(r.yaw_est),
                                         line=dict(color="#0b0f14", width=2))))
    ch._base_layout(fig, height, legend_bottom=True, uirevision=None if follow else f"team-map-{seq}")
    ch._equal_axes(fig)
    fig.update_xaxes(title_text="x (m, gt.csv frame)", showgrid=True)
    fig.update_yaxes(title_text="y (m)", showgrid=True)
    if follow:
        fig.update_xaxes(range=[r.x_est - follow, r.x_est + follow])
        fig.update_yaxes(range=[r.y_est - follow, r.y_est + follow])
    return fig


def _fmt_len(v: float) -> str:
    return f"{100 * v:.0f} cm" if v < 1 else f"{v:.2f} m"


def tiles_html(run: ts.TeamRun, s: ts.Situation, show_final: bool) -> str:
    stt = STATE[ts.STATE_THEME[s.state]]
    conf = opui.tile("Navigation confidence", f"<span style='color:{stt['color']}'>{stt['icon']}</span> {s.state_name}",
                     f"teammate state {s.state} · from model 2σ", stt["color"])
    if s.gps_out:
        gps = opui.tile("GPS (simulated outage)", f"<span style='color:{CRIT}'>■</span> Lost",
                        f"outage running for {s.outage_elapsed:.0f} s", CRIT)
    else:
        gps = opui.tile("GPS", f"<span style='color:{GOOD}'>●</span> Available",
                        f"back for {s.since_restored:.0f} s" if s.since_restored is not None else "no outage yet", GOOD)
    unc = opui.tile("Position uncertainty (model 2σ)", f"± {_fmt_len(s.two_sigma)}",
                    "Model-estimated, not measured accuracy · target 0.3 m"
                    + ui.uncertainty_gauge(s.two_sigma, ts.TRUSTED_2SIGMA_M, ts.LOST_2SIGMA_M, "lost"),
                    stt["color"], wide=True, tag=opui.kind("computed", "Model estimate"))
    if s.dist is None:
        wk = opui.tile("Nearest worker (simulated)", "None active", f"no simulated worker now · bubble {s.bubble:.1f} m",
                       None)
    else:
        col = CRIT if s.dist < s.bubble else WARN if s.dist < s.bubble + ts.SLOW_MARGIN_M else GOOD
        wk = opui.tile("Nearest worker (simulated)", f"{s.dist:.1f} m",
                       f"bubble {s.bubble:.1f} m · slow within {s.bubble + ts.SLOW_MARGIN_M:.1f} m", col)
    labels = ts.tree_status_at(run, s.t, show_final)
    total = len(labels)
    if run.tree_reveal == "progressive" or show_final:
        seen = labels != "unreached"
        ok = int((labels == "correct").sum())
        spray = opui.tile("Spray outcome (simulated)", f"{ok} / {int(seen.sum())} correct",
                          f"trees passed so far, of {total}" if not show_final else f"final result, {total} trees",
                          None)
    else:
        spray = opui.tile("Spray outcome (simulated)", "Final only",
                          "spray timing cannot be reconstructed for this run", None)
    return "<div class='mgrid'>" + conf + gps + unc + wk + spray + "</div>"


def checklist_items(s: ts.Situation) -> list:
    items = [("Overall recommendation", {ts.GO: ops.YES, ts.SLOW: ops.ADVISED, ts.STOP: ops.NO}[s.action],
              f"{s.action}: {s.reason}")]
    for name, ok, lim in ts.capabilities(s.two_sigma):
        items.append((name, ops.YES if ok else ops.NO, f"model 2σ {_fmt_len(s.two_sigma)}, limit {lim:g} m"))
    return items


CHECK_NOTE = ("Prototype decision rules from the teammate's 2σ limits, not validated safety guarantees. "
              "Localization capability and the GO/SLOW/STOP recommendation are separate: a task can be possible "
              "while the robot should still stop for a worker. The dashboard does not control a robot.")


# ------------------------------------------------------------------- toolbar --

def start_demo():
    S.set_values(team_seq=DEMO["sequence"], team_mode=DEMO["mode"], speed=DEMO["speed"], playing=PAGE,
                 demo_end=DEMO["end"], viz="3D orchard", view3d="Follow robot")
    st.session_state[TKEY] = DEMO["start"]


def reset():
    S.set_values(playing=None, demo_end=None)
    st.session_state[TKEY] = DEMO["start"] if st.session_state.team_seq == DEMO["sequence"] else 0.0


def toolbar(run: ts.TeamRun):
    ss = st.session_state
    with st.container(border=True):
        c = st.columns([1.6, 1.4, 1.2, 1.55, 0.9], vertical_alignment="bottom", gap="small")
        c[0].segmented_control("Safety mode", list(ts.MODES), required=True, **S.bind("team_mode", "team"),
                               help="Safe: STOP when a worker is inside the bubble or confidence is Lost. "
                                    "Productive: STOP only for a worker inside the bubble; Lost means SLOW.")
        c[1].segmented_control("Speed", [1, 2, 4, 10], format_func=lambda v: f"{v}×", required=True,
                               **S.bind("speed", "team"))
        if run.tree_reveal == "final_only":
            c[2].toggle("Show final spray results", **S.bind("team_show_final", "team"),
                        help="Spray timing cannot be reconstructed for this sequence (rows are driven more than "
                             "once), so outcomes are hidden during the replay unless you show the final result.")
        else:
            c[2].markdown("<div class='small'>Spray outcomes appear as the robot passes each tree "
                          "(approximate timing).</div>", unsafe_allow_html=True)
        demo = c[3].button("▶ Run GPS Failure Demo", type="primary", width="stretch", key="team_demo",
                           help="02_13B · Safe mode · plays from 10 s before the first simulated GPS outage "
                                "through the first simulated worker encounter.")
        rst = c[4].button("⟲ Reset", width="stretch", key="team_reset")
    if demo:
        start_demo()
        st.rerun()
    if rst:
        reset()
        st.rerun()


# ------------------------------------------------------------- mission control --

def header(seq: str, eng: bool):
    kinds = (opui.kind("recorded", "RTK reference (gt.csv)") + opui.kind("computed", "Teammate estimates")
             + opui.kind("simulated", "Simulated outages · workers · trees · spray"))
    ui.page_header(f"{'Engineer' if eng else 'Operator'} view · {ts.SOURCE_LABEL} · {seq}", "Mission Control",
                   "Where is the robot, can we trust that position, and what should happen next?",
                   [ui.badge("TEAMMATE RESULTS", "real"), kinds])


def not_available() -> bool:
    inv = inventory()
    if inv.sequences:
        return False
    ui.not_evaluated(ts.SOURCE_LABEL, f"no complete teammate run found in {ts.TEAM_DIR}.",
                     "<SEQ>_dashboard.csv, <SEQ>_trees.csv, <SEQ>_crew.csv (see README)")
    return True


def live_panel():
    ss = st.session_state
    seq, mode = ss.team_seq, ss.team_mode
    run, dec = load(seq), decisions(seq, mode)
    ui.advance(PAGE, TKEY, run.t_max, 0.5 * ss.speed)
    S.clamp(TKEY, 0.0, run.t_max)
    t = float(ss[TKEY])
    s = ts.situation(run, dec, t, mode)
    ev = ts.events(run, dec)
    eng = S.is_engineer()
    show_final = bool(ss.team_show_final) and run.tree_reveal == "final_only"

    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        bar = st.container(horizontal=True, gap="small", vertical_alignment="center")
        view = ss.viz if ss.viz in ("3D orchard", "2D map") else "3D orchard"
        ss.viz = view
        bar.segmented_control("View", ["3D orchard", "2D map"], required=True, label_visibility="collapsed",
                              **S.bind("viz", "team"))
        bar.segmented_control("Camera", ["Follow robot", "Whole orchard"], required=True,
                              label_visibility="collapsed", **S.bind("view3d", "team"))
        if ss.viz == "3D orchard":
            fig = scene_figure(seq, t, mode, show_final=show_final, show_reference=eng,
                               overview=ss.view3d == "Whole orchard")
            st.plotly_chart(fig, key="team_3d", theme=None, config={"displaylogo": False, "displayModeBar": False})
        else:
            fig = map_figure(seq, t, mode, show_final=show_final, show_reference=eng,
                             follow=18.0 if ss.view3d == "Follow robot" else None)
            st.plotly_chart(fig, key="team_map", theme=None, config={"displaylogo": False, "scrollZoom": True})
        labels = ts.tree_status_at(run, t, show_final)
        st.markdown(
            "<div class='legend'>"
            + "".join(f"<span class='l'><span class='sw' style='background:{STATE[ts.STATE_THEME[k]]['color']}'>"
                      f"{STATE[ts.STATE_THEME[k]]['icon']}</span>Trail · {ts.STATE_NAMES[k]}</span>" for k in (0, 1, 2))
            + "<span class='l'><span class='sw' style='background:rgba(238,242,239,.25)'></span>Bubble = model 2σ"
              "</span><span class='l'>Dashed ring = safety bubble (bubble_m), coloured GO / SLOW / STOP</span>"
              "<span class='l'>◇ Simulated worker · 2 m zone</span>"
            + "".join(f"<span class='l'><span class='sw' style='background:{scene3d.SPRAY_COLOR[SPRAY_CODE[k]]}'>"
                      f"</span>{SPRAY_TEXT[k]} {int((labels == k).sum())}</span>"
                      for k in ("correct", "wrong_tree", "missed", "double", "unreached"))
            + f"</div><div class='small'>{opui.kind('simulated', 'Simulated spray')} Tree positions and spray "
              f"outcomes come from the teammate's simulation, not a field survey or real spraying. "
              + ("Each outcome appears when the RTK path passes closest to the tree; this approximate timing "
                 f"matches the teammate's own during-outage flags for {100 * run.tree_timing_agreement:.0f} % of "
                 "trees." if run.tree_reveal == "progressive" else
                 f"Spray timing cannot be reconstructed for this run (closest-approach timing matches only "
                 f"{100 * run.tree_timing_agreement:.0f} % of the teammate's flags), so outcomes stay hidden "
                 "unless 'Show final spray results' is on.")
            + " Robot drawn 2× size.</div>", unsafe_allow_html=True)
        jumps = []
        nxt_gps = ts.next_time([a for a, _ in run.outages], t, lead=3.0)
        if nxt_gps is not None:
            jumps.append(("⏭ Next GPS outage", nxt_gps))
        nxt_w = ts.next_time(run.crew.t_outage_start, t)
        if nxt_w is not None:
            jumps.append(("⏭ Next worker", nxt_w))
        ui.playback_controls(PAGE, TKEY, run.t_max, step=0.1, label="Replay time", jump_to=jumps)
        opui.event_chips(ev[(ev.t_rel >= t - 30) & (ev.t_rel <= t + 150)], PAGE, TKEY, run.t_max, limit=8)

    with right:
        ui.status_hero(ACTION_THEME[s.action], s.reason, label=f"Recommended action · {mode} mode · t = {s.t:.0f} s",
                       title=s.action, meaning=ACTION_MEANING[s.action])
        st.markdown(tiles_html(run, s, show_final), unsafe_allow_html=True)
        if eng:
            st.caption(f"Actual error vs RTK now (replay only): **{s.error:.2f} m** · learned slip scale "
                       f"**{s.slip_scale:.3f}**")
        st.caption("Simulated decision support from the teammate's rules; no command is sent to a robot.")

    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1:
        ui.section("What the robot can do now", "Teammate's 2σ limits · updates with the replay")
        st.markdown(opui.checklist_html(checklist_items(s), note=CHECK_NOTE), unsafe_allow_html=True)
    with c2:
        h = st.container(horizontal=True, vertical_alignment="center")
        h.markdown("<div class='sec'>Alerts · Alertas</div>", unsafe_allow_html=True)
        h.segmented_control("Language", ["English", "Español", "English + Español"], required=True,
                            label_visibility="collapsed", **S.bind("lang", "team"))
        st.markdown(opui.alerts_html(ts.alert_keys(s), ss.lang, table=ts.ALERTS), unsafe_allow_html=True)
        st.caption("Spanish drafted for the demo, pending native-speaker review.")


# ------------------------------------------------------------------- engineer --

UNAVAILABLE = ("IMU gyro bias", "GPS NIS / consistency-gate statistics", "GPS rejection history",
               "Raw GPS fixes used by the teammate's filter", "LiDAR measurements or point clouds",
               "Full position covariance (only a scalar sigma is exported)", "Robot speed")


def _time_axes(fig, run, height=280, title="metres"):
    ch.add_outage_bands(fig, run.outages, 0.0)
    ch._base_layout(fig, height, legend_bottom=False, hovermode="x unified")
    fig.update_xaxes(title_text="time since start (s)")
    fig.update_yaxes(title_text=title)
    return fig


def error_uncertainty_figure(run: ts.TeamRun, t_now: float | None = None) -> go.Figure:
    d = run.dash
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.t, y=d.error_m, mode="lines", name="Actual error vs RTK (error_m)",
                             line=dict(color=SERIES[0], width=1.8)))
    fig.add_trace(go.Scatter(x=d.t, y=2 * d.sigma, mode="lines", name="Model uncertainty 2σ",
                             line=dict(color=SERIES[1], width=1.8, dash="dash")))
    fig.add_hline(y=ts.TRUSTED_2SIGMA_M, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text="0.3 m target", annotation_font=dict(color=MUTED, size=11))
    if t_now is not None:
        fig.add_vline(x=t_now, line=dict(color=INK, width=1.2))
    return _time_axes(fig, run, 300)


def slip_figure(run: ts.TeamRun, t_now: float | None = None) -> go.Figure:
    d = run.dash
    fig = go.Figure(go.Scatter(x=d.t, y=d.slip_scale, mode="lines", name="Learned slip scale",
                               line=dict(color=SERIES[2], width=2)))
    fig.add_hline(y=1.0, line=dict(color=MUTED, dash="dot", width=1), annotation_text="1.0 = no correction",
                  annotation_font=dict(color=MUTED, size=11))
    if t_now is not None:
        fig.add_vline(x=t_now, line=dict(color=INK, width=1.2))
    return _time_axes(fig, run, 260, "wheel-speed scale")


def decision_figure(run: ts.TeamRun) -> go.Figure:
    fig = go.Figure()
    for mode, col in ((ts.SAFE, SERIES[0]), (ts.PRODUCTIVE, SERIES[1])):
        dec = decisions(run.sequence, mode)
        fig.add_trace(go.Scatter(x=dec.t, y=dec.level + (0.04 if mode == ts.SAFE else -0.04), mode="lines",
                                 name=f"{mode} mode", line=dict(color=col, width=2, shape="hv")))
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["GO", "SLOW", "STOP"], range=[-0.3, 2.3])
    return _time_axes(fig, run, 220, "action")


def worker_figure(run: ts.TeamRun, mode: str) -> go.Figure:
    dec = decisions(run.sequence, mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec.t, y=dec.dist_m, mode="lines", name="Distance to visible worker (estimate)",
                             line=dict(color=SERIES[0], width=2), connectgaps=False))
    fig.add_trace(go.Scatter(x=dec.t, y=dec.bubble_m, mode="lines", name="Safety bubble (STOP inside)",
                             line=dict(color=CRIT, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=dec.t, y=dec.bubble_m + ts.SLOW_MARGIN_M, mode="lines", name="Bubble + 5 m (SLOW)",
                             line=dict(color=WARN, width=1.5, dash="dot")))
    return _time_axes(fig, run, 260)


@st.cache_data(show_spinner="Running the backend EKF with the teammate's outage windows…")
def backend_comparison(seq: str):
    """Backend EKF on the same sequence with the SAME outage windows, if the CitrusFarm data is local."""
    from . import backend as be
    if seq not in be.source_keys():
        return None
    streams = be.load_streams(seq)
    methods = be.methods_for(streams)
    method = "odom+imu+gps" if "odom+imu+gps" in methods else methods[0]
    run = load(seq)
    outages = tuple((round(a, 3), round(b - a, 3)) for a, b in run.outages)
    res = be.experiment(seq, be.METHODS[method], outages)
    t0 = streams.time_span()[0]
    est = res.estimate
    return {"method": be.METHOD_LABEL[method], "gps_source": res.gps_source,
            "t": est.t.to_numpy() - t0, "x": est.x_ant.to_numpy(), "y": est.y_ant.to_numpy(),
            "err_t": res.error.t.to_numpy() - t0, "err": res.error.err.to_numpy()}


def trajectory_figure(run: ts.TeamRun, bk: dict | None) -> go.Figure:
    d = run.dash
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.x_true, y=d.y_true, mode="lines", name="RTK reference (gt.csv)",
                             line=dict(color=REF, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=d.x_est, y=d.y_est, mode="lines", name="Teammate estimate (slip learning)",
                             line=dict(color=SERIES[2], width=2)))
    if bk is not None:
        fig.add_trace(go.Scatter(x=bk["x"], y=bk["y"], mode="lines", name=f"Backend EKF ({bk['method']})",
                                 line=dict(color=SERIES[0], width=1.6)))
    for k, (a, b) in enumerate(run.outages):
        m = (d.t >= a) & (d.t <= b)
        fig.add_trace(go.Scatter(x=d.x_true[m], y=d.y_true[m], mode="lines", line=dict(color=WARN, width=7),
                                 opacity=0.35, name="GPS outage (simulated)", showlegend=k == 0, hoverinfo="skip"))
    ch_idx = np.flatnonzero(np.diff(d.state.to_numpy()) != 0) + 1
    fig.add_trace(go.Scatter(x=d.x_est[ch_idx], y=d.y_est[ch_idx], mode="markers", name="Confidence state change",
                             marker=dict(size=7, color=[STATE[ts.STATE_THEME[int(v)]]["color"] for v in d.state[ch_idx]],
                                         line=dict(color="#0b0f14", width=1))))
    ch._base_layout(fig, 520, legend_bottom=True, uirevision=f"team-traj-{run.sequence}")
    ch._equal_axes(fig)
    fig.update_xaxes(title_text="x (m, gt.csv frame)")
    fig.update_yaxes(title_text="y (m)")
    return fig


@st.cache_data(show_spinner=False)
def slip_table() -> pd.DataFrame:
    rows = []
    for seq in sequences():
        r = load(seq)
        s = r.dash.slip_scale
        rows.append({"sequence": seq, "final slip scale": round(float(s.iloc[-1]), 4),
                     "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                     "reduction vs uncorrected (final)": f"{100 * (1 - s.iloc[-1]):.1f} %"})
    return pd.DataFrame(rows)


def _metrics(err: np.ndarray, t: np.ndarray, outages) -> dict:
    m = np.zeros(len(t), bool)
    for a, b in outages:
        m |= (t >= a) & (t < b)
    return {"mean error (m)": round(float(np.mean(err)), 3), "max error (m)": round(float(np.max(err)), 2),
            "time within 0.3 m (%)": round(float(100 * np.mean(err <= 0.3)), 1),
            "peak error in outages (m)": round(float(np.max(err[m])), 2) if m.any() else None}


def engineer(seq: str):
    ss = st.session_state
    run = load(seq)
    t_now = float(ss[TKEY])
    st.divider()
    tabs = st.tabs(["Trajectory comparison", "Error vs uncertainty", "Wheel-slip learning", "Decisions",
                    "Event log", "Validation & provenance"])
    with tabs[0]:
        bk = backend_comparison(seq)
        st.plotly_chart(trajectory_figure(run, bk), key="team_traj", theme=None, config={"displaylogo": False})
        rows = [{"run": "Teammate: odometry + IMU + GPS + slip learning", **_metrics(run.dash.error_m.to_numpy(),
                                                                                       run.dash.t.to_numpy(),
                                                                                       run.outages)}]
        if bk is not None:
            rows.append({"run": f"Backend EKF: {bk['method']}", **_metrics(bk["err"], bk["err_t"], run.outages)})
        ui.df_table(pd.DataFrame(rows))
        if bk is None:
            ui.not_evaluated("Backend EKF comparison", "the CitrusFarm data for this sequence is not available "
                             "locally.", "scripts/download_citrusfarm_subset.py and scripts/validate_baseline.py")
        else:
            gps = ("a SIMULATED GPS proxy (reference + 3 cm noise)" if bk["gps_source"] == "proxy_from_reference"
                   else "recorded GPS fixes")
            st.caption(f"Two different implementations on the same RTK reference and the same simulated outage "
                       f"windows. The backend EKF here uses {bk['method'].lower()} with {gps}; the teammate's filter "
                       f"adds a learned wheel-slip correction and its GPS input is not described in the export. "
                       f"Sensor sets and GPS inputs differ, so this is not a controlled comparison.")
    with tabs[1]:
        st.plotly_chart(error_uncertainty_figure(run, t_now), key="team_err", theme=None,
                        config={"displaylogo": False})
        d = run.dash
        inside = d.error_m <= 2 * d.sigma
        m = d.gps_out.to_numpy()
        ui.kpi_row([
            ui.kpi_html("Error ≤ 2σ (all)", f"{100 * inside.mean():.1f} %", "share of samples"),
            ui.kpi_html("Error ≤ 2σ (outages)", f"{100 * inside[m].mean():.1f} %" if m.any() else "n/a",
                        "during simulated outages"),
            ui.kpi_html("Max error", f"{d.error_m.max():.2f} m", "full run"),
            ui.kpi_html("Time within 0.3 m", f"{100 * (d.error_m <= 0.3).mean():.1f} %", "full run"),
        ])
        st.caption("**Actual error** (error_m) is the distance to the RTK reference and is only known afterwards. "
                   "**Model uncertainty** (2σ) is what the teammate's filter believes in real time. If sigma is a "
                   "per-axis standard deviation of an ideal 2D Gaussian estimate, a 2σ circle would contain the truth "
                   "about 86 % of the time (the export does not define sigma further). The shares above are an "
                   "empirical check, not a calibration proof.")
    with tabs[2]:
        st.plotly_chart(slip_figure(run, t_now), key="team_slip", theme=None, config={"displaylogo": False})
        s = run.dash.slip_scale
        st.markdown(f"Learned wheel-speed scale on **{seq}**: starts at {s.iloc[0]:.3f}, ends at **{s.iloc[-1]:.3f}** "
                    f"(about {100 * (1 - s.iloc[-1]):.1f} % less than the uncorrected wheel speed). A value near "
                    f"0.885 means the filter scales the wheel-odometry speed down by about 11.5 % to account for "
                    f"wheel slip. The correction is learned per run; values differ between runs:")
        ui.df_table(slip_table())
        st.caption("Computed by the teammate's filter; not independently validated here.")
    with tabs[3]:
        st.plotly_chart(decision_figure(run), key="team_dec", theme=None, config={"displaylogo": False})
        st.plotly_chart(worker_figure(run, ss.team_mode), key="team_wd", theme=None, config={"displaylogo": False})
        rows = []
        for mode in ts.MODES:
            dec = decisions(seq, mode)
            dt = np.diff(np.r_[dec.t.to_numpy(), dec.t.iloc[-1]])
            rows.append({"mode": mode, **{f"time {a} (s)": round(float(dt[dec.action == a].sum()), 1)
                                          for a in (ts.GO, ts.SLOW, ts.STOP)},
                         "STOP episodes": int(((dec.level == 2) & (dec.level.shift(fill_value=0) != 2)).sum())})
        ui.df_table(pd.DataFrame(rows))
        st.caption("Decisions follow the teammate's documented rules on the supplied columns (dashboard/team_slip.py). "
                   "Workers and decisions are simulated; no command is sent to a robot.")
    with tabs[4]:
        ev = ts.events(run, decisions(seq, ss.team_mode)).rename(columns={"t_rel": "time (s)"})
        ui.df_table(ev.assign(**{"time (s)": ev["time (s)"].round(1)}), height=360)
    with tabs[5]:
        st.markdown("**Validation of the supplied files** (values are shown as supplied; nothing is corrected)")
        st.markdown("\n".join(f"- {c}" for c in run.checks))
        st.markdown("- `x_true, y_true` equal the CitrusFarm `gt.csv` reference on the same clock "
                    "(checked on 02_13B and 06_14B: median difference below 1 mm, no rotation, no time offset)")
        st.markdown("**Not available in this teammate export**")
        ui.df_table(pd.DataFrame({"field": UNAVAILABLE, "status": "Not available in this teammate export"}))
        ui.df_table(pd.DataFrame([
            ("RTK reference trajectory", "Recorded (CitrusFarm gt.csv, post-processed RTK)"),
            ("Estimated pose, sigma, state, slip scale, error", "Computed by the teammate's localization analysis"),
            ("GPS outages", "Simulated (30 s every 90 s from 60 s)"),
            ("Trees and spray outcomes", "Simulated by the teammate, not surveyed or measured"),
            ("Workers", "Simulated positions, not detected people"),
            ("GO / SLOW / STOP", "Teammate's rules applied in this dashboard (simulated decision support)"),
            ("Files", ", ".join(Path(p).name for p in run.files.values())),
        ], columns=["item", "origin"]))


# ---------------------------------------------------------- results pages --

SETUP_COLOR = {ts.PRIMARY_SETUPS[0]: SERIES[0], ts.PRIMARY_SETUPS[1]: SERIES[1], ts.PRIMARY_SETUPS[2]: SERIES[2],
               "GPS always on (reference)": REF}
SETUP_SENSORS = [  # qualitative only: no prices
    (ts.PRIMARY_SETUPS[0], "wheel odometry, GPS", "Lowest", "no extra sensor"),
    (ts.PRIMARY_SETUPS[1], "wheel odometry, IMU, GPS", "Low", "adds a low-cost IMU"),
    (ts.PRIMARY_SETUPS[2], "wheel odometry, IMU, GPS", "Low (same hardware as row 2)",
     "software only: learned wheel-slip correction"),
]
TEST_METRICS = {"worst": ("Worst error during outage (m)", "lower is better"),
                "err_end": ("Error at outage end (m)", "lower is better"),
                "time_within": ("Time within 0.3 m target (s)", "higher is better")}


def _results_header(title: str, eyebrow: str, sub: str):
    ui.page_header(eyebrow, title, sub, [ui.badge("TEAMMATE RESULTS", "real"),
                                          opui.kind("computed", "Teammate batch results"),
                                          opui.kind("simulated", "Simulated outages")])


def _filters(df: pd.DataFrame, scope: str, groups: bool = False) -> pd.DataFrame:
    c = st.columns(2 if groups else 1)
    if groups:
        g = c[1].multiselect("Group", sorted(df.group.unique()), default=sorted(df.group.unique()), key=f"{scope}_grp")
        df = df[df.group.isin(g)]
    seqs = sorted(df.seq.unique())
    pick = c[0].multiselect("Sequences", seqs, default=seqs, key=f"{scope}_seq",
                            format_func=lambda s: s.replace("_Jackal", ""))
    return df[df.seq.isin(pick)]


def _matched(df: pd.DataFrame, setups) -> pd.DataFrame:
    """Keep only (sequence, outage) cells that have every selected setup, so means compare like with like."""
    df = df[df.setup.isin(setups)]
    full = df.groupby(["seq", "outage_s"]).setup.nunique() == len(setups)
    keep = set(full[full].index)
    return df[[(s, o) in keep for s, o in zip(df.seq, df.outage_s)]]


def gps_page():
    """GPS Failure Simulator, team mode: test_results.csv (held-out sequences, 10 to 120 s outages)."""
    _results_header("GPS Failure Simulator", "Teammate results · localization under simulated outages",
                    "How far does each sensor setup drift when GPS drops out for 10, 30, 60 or 120 s?")
    t = summaries()["test_results"]
    if t is None:
        ui.not_evaluated("test_results.csv", "file not found in team_runs/slip_learning.")
        return
    t = ts.primary(t)
    t = _filters(t, "tgps", groups=True)
    metric = st.segmented_control("Metric", list(TEST_METRICS), format_func=lambda m: TEST_METRICS[m][0],
                                  required=True, default="worst", key="tgps_metric")
    setups = st.multiselect("Setups", list(ts.PRIMARY_SETUPS), default=list(ts.PRIMARY_SETUPS), key="tgps_setups")
    m = _matched(t, setups)
    if m.empty:
        st.info("No matching rows for this selection.")
        return
    agg = m.groupby(["setup", "outage_s"]).agg(value=(metric, "mean"), seqs=("seq", "nunique"),
                                               windows=("n", "sum")).reset_index()
    fig = go.Figure()
    for sname in setups:
        a = agg[agg.setup == sname]
        fig.add_trace(go.Bar(x=a.outage_s.map(lambda v: f"{v:g} s"), y=a.value, name=sname,
                             marker_color=SETUP_COLOR.get(sname, MUTED),
                             customdata=np.c_[a.seqs, a.windows],
                             hovertemplate="%{y:.2f}<br>%{customdata[0]} sequences · %{customdata[1]} outage windows"
                                           "<extra>" + sname + "</extra>"))
    if metric != "time_within":
        fig.add_hline(y=0.3, line=dict(color=MUTED, dash="dot"), annotation_text="0.3 m target",
                      annotation_font=dict(color=MUTED, size=11))
    ch._base_layout(fig, 360, legend_bottom=True, barmode="group")
    fig.update_xaxes(title_text="simulated GPS outage length")
    fig.update_yaxes(title_text=TEST_METRICS[metric][0])
    st.plotly_chart(fig, key="tgps_bar", theme=None, config={"displaylogo": False})
    st.caption(f"Bars: mean over the selected sequences of each sequence's value ({TEST_METRICS[metric][1]}). "
               "Each sequence value is the teammate's average over its n outage windows. Only sequence and outage "
               "combinations that have every selected setup are included, so setups are compared on the same "
               "experiments. 02_13B is not in this table (the teammate used it for tuning). Simulated outages on "
               "recorded CitrusFarm runs; computed by the teammate.")
    with st.expander("Rows used"):
        ui.df_table(m.sort_values(["seq", "outage_s", "setup"]).round(3))


def cost_page():
    """Sensor Cost Comparison, team mode: sensors per setup (qualitative) + simulated spray outcomes."""
    _results_header("Sensor Cost Comparison", "Teammate results · sensors vs outcome",
                    "Does learned wheel-slip correction buy accuracy without new hardware?")
    ui.df_table(pd.DataFrame(SETUP_SENSORS, columns=["setup", "sensors", "relative hardware cost (qualitative)",
                                                     "difference"]))
    st.caption("Qualitative tiers only; no prices are claimed.")
    sp = summaries()["spray_results"]
    if sp is None:
        ui.not_evaluated("spray_results.csv", "file not found in team_runs/slip_learning.")
        return
    ui.section("Simulated spraying outcome by setup", "trees passed during the simulated outages")
    sp = _filters(sp, "tcost")
    outage = st.segmented_control("Outage length", sorted(sp[sp.outage_s > 0].outage_s.unique()),
                                  format_func=lambda v: f"{v:g} s", required=True, default=30, key="tcost_out")
    sub = _matched(sp[sp.outage_s == outage], ts.PRIMARY_SETUPS)
    if sub.empty:
        st.info("No matching rows for this selection.")
        return
    cols = {"correct_pct": ("Correct", SERIES[0]), "wrong_tree_pct": ("Wrong tree", SERIES[1]),
            "missed_pct": ("Missed", "#e8e2d0"), "double_pct": ("Double", "#d55181")}
    agg = sub.groupby("setup").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.trees) for c in
                                                         list(cols) + ["wrong_row_pct"]} | {"trees": g.trees.sum()}),
                                     include_groups=False).reindex(ts.PRIMARY_SETUPS)
    fig = go.Figure()
    for c, (lab, colr) in cols.items():
        fig.add_trace(go.Bar(y=agg.index, x=agg[c], name=lab, orientation="h", marker_color=colr,
                             hovertemplate="%{x:.1f} %<extra>" + lab + "</extra>"))
    ch._base_layout(fig, 260, legend_bottom=True, barmode="stack")
    fig.update_xaxes(title_text="% of trees passed during outages (tree-weighted over selected sequences)",
                     range=[0, 100])
    st.plotly_chart(fig, key="tcost_bar", theme=None, config={"displaylogo": False})
    ui.df_table(agg.reset_index().round(1).rename(columns={"wrong_row_pct": "wrong row %", "trees": "trees"}))
    st.caption(f"Simulated spraying, not real agricultural performance. Percentages are weighted by the number of "
               f"trees each sequence passed during {outage} s outages; only sequences with all three setups are "
               f"included. 'Wrong row' is reported separately by the teammate. 'Fixed slip from 02 (no learning)' is "
               f"excluded from this presentation comparison at the teammate's request.")


def safety_page():
    """Worker Safety, team mode: safety_results.csv + Safe vs Productive on the selected sequence."""
    _results_header("Worker Safety", "Teammate results · simulated worker encounters",
                    "Does an uncertainty-dependent safety bubble stop the robot in time?")
    sf = summaries()["safety_results"]
    if sf is None:
        ui.not_evaluated("safety_results.csv", "file not found in team_runs/slip_learning.")
    else:
        sf = _filters(sf, "tsafe")
        outage = st.segmented_control("Outage length", sorted(sf.outage_s.unique()), format_func=lambda v: f"{v:g} s",
                                      required=True, default=30, key="tsafe_out")
        sub = sf[sf.outage_s == outage]
        agg = sub.groupby("setup").agg(
            encounters=("encounters", "sum"), unsafe=("unsafe", "sum"),
            median_stop_dist_m=("median_stop_dist_m", "median"),
            closest_unsafe_m=("closest_unsafe_m", "min"),
            bubble_contains_truth_pct=("bubble_contains_truth_outage_pct", "mean"),
            green_pct=("time_green_pct", "mean"), yellow_pct=("time_yellow_pct", "mean"),
            red_pct=("time_red_pct", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg.setup, y=agg.encounters - agg.unsafe, name="Safe encounters", marker_color=GOOD))
        fig.add_trace(go.Bar(x=agg.setup, y=agg.unsafe, name="Unsafe encounters", marker_color=CRIT))
        ch._base_layout(fig, 300, legend_bottom=True, barmode="stack")
        fig.update_yaxes(title_text="simulated encounters (sum over selected sequences)")
        st.plotly_chart(fig, key="tsafe_bar", theme=None, config={"displaylogo": False})
        ui.df_table(agg.round(1))
        st.caption("Simulated workers and encounters, not detected people. Encounters and unsafe counts are summed "
                   "over the selected sequences; the median stopping distance is the median of per-sequence medians; "
                   "closest unsafe distance is the minimum (empty when no encounter was unsafe); percentages are "
                   "unweighted means. Green, yellow and red are the teammate's Trusted, Degraded and Lost shares.")
    ui.section("Safe vs Productive mode on one sequence", "rules applied in this dashboard to the teammate's replay")
    seq = st.session_state.team_seq
    rows = []
    for mode in ts.MODES:
        dec = decisions(seq, mode)
        dt = np.diff(np.r_[dec.t.to_numpy(), dec.t.iloc[-1]])
        stops = int(((dec.level == 2) & (dec.level.shift(fill_value=0) != 2)).sum())
        rows.append({"mode": mode, "time GO (s)": round(float(dt[dec.action == ts.GO].sum()), 1),
                     "time SLOW (s)": round(float(dt[dec.action == ts.SLOW].sum()), 1),
                     "time STOP (s)": round(float(dt[dec.action == ts.STOP].sum()), 1), "STOP episodes": stops,
                     "closest estimated worker distance (m)": round(float(np.nanmin(dec.dist_m)), 2)
                     if np.isfinite(dec.dist_m).any() else None})
    ui.df_table(pd.DataFrame(rows))
    st.caption(f"{seq}: Safe mode stops when a simulated worker is inside the safety bubble or confidence is Lost; "
               "Productive mode stops only for a worker inside the bubble. Open Mission Control to replay it.")


def risk_map_notice():
    ui.notice("<b>Team Slip-Learning Results</b> do not include receiver-quality or canopy data, so this page "
              "keeps using the backend data source selected under Backend EKF. Switch Localization results back to "
              "Backend EKF to change it.", "")


def mission_control():
    ss = st.session_state
    eng = S.is_engineer()
    if not_available():
        return
    if ss.team_seq not in sequences():
        ss.team_seq = DEMO["sequence"] if DEMO["sequence"] in sequences() else sequences()[0]
    seq = ss.team_seq
    header(seq, eng)
    run = load(seq)
    toolbar(run)
    st.fragment(run_every=0.5 if ss.playing == PAGE else None)(live_panel)()
    if eng:
        engineer(seq)
    else:
        st.markdown("<div class='small' style='margin-top:6px'>Switch to <b>Engineer</b> mode (sidebar) for the "
                    "reference trajectory, error vs uncertainty, wheel-slip learning and the Rerun replay.</div>",
                    unsafe_allow_html=True)
        st.button("Inspect this moment in Engineer mode →", on_click=lambda: ss.update(view_mode="Engineer"),
                  key="team_to_eng")
