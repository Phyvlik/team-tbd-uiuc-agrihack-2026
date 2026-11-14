"""Streamlit views for the "Team Slip-Learning Results" source.

The teammate's CSVs (dashboard/team_slip.py) drive the replay directly: robot pose
(x_est, y_est, yaw_est), navigation state, GPS outage flag, model uncertainty
(2*sigma), safety bubble (bubble_m), simulated workers and simulated spray outcomes.
Nothing here re-estimates anything. One replay clock (session key ``team_t``,
seconds since the first gt.csv sample) drives the 3D view, metrics, checklist,
alerts and events.
"""
from __future__ import annotations

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
    if not eng:
        st.markdown("<div class='small' style='margin-top:6px'>Switch to <b>Engineer</b> mode (sidebar) for the "
                    "reference trajectory, error vs uncertainty, wheel-slip learning and the Rerun replay.</div>",
                    unsafe_allow_html=True)
        st.button("Inspect this moment in Engineer mode →", on_click=lambda: ss.update(view_mode="Engineer"),
                  key="team_to_eng")
