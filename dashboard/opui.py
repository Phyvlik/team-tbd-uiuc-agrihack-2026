"""Operator-dashboard components: data-kind tags, experiment toolbar, metric tiles,
"What the robot can do now" checklist, bilingual alerts, timestamp-linked events,
and the error-vs-uncertainty explainer."""
from __future__ import annotations

import numpy as np
import streamlit as st

from . import backend as be
from . import ops
from . import state as S
from .components import esc
from .scenario import DEMO
from .theme import CRIT, GOOD, MUTED, NEUTRAL, STATE, WARN

KIND_TEXT = {
    "recorded": ("Recorded", "Measured by the robot's sensors (CitrusFarm logs)."),
    "computed": ("Computed", "Our estimator's output from recorded data (team EKF + integrity monitor)."),
    "predicted": ("Predicted", "Model output about places/times not directly measured (e.g. canopy → GPS risk)."),
    "simulated": ("Simulated", "Made up on purpose to test behaviour: GPS outages, crews, spray, synthetic orchard."),
    "illustrative": ("Illustrative", "Drawn for context only, not measured (e.g. tree positions inferred from the path)."),
    "uncalibrated": ("Uncalibrated", "The estimator's own 95 % uncertainty has not been validated against ground truth "
                                     "(see docs/UNCERTAINTY_CALIBRATION.md)."),
}


def kind(k: str, text: str | None = None) -> str:
    lab, tip = KIND_TEXT[k]
    return f"<span class='kind {k}' title='{esc(tip)}'>{esc(text or lab)}</span>"


def kind_legend_html() -> str:
    return "".join(f"<div style='margin:3px 0'>{kind(k)} <span class='small'>{esc(t)}</span></div>"
                   for k, (_, t) in KIND_TEXT.items())


# ------------------------------------------------------------- toolbar ----

def experiment_toolbar(scope: str, streams, *, demo_page: str | None = None, show_method: bool = True,
                       label_outage: str = "GPS outage") -> dict:
    """Compact one-row experiment toolbar bound to the shared scenario state."""
    ss = st.session_state
    t0, t1 = streams.time_span()
    dur = t1 - t0
    methods = be.methods_for(streams)
    if ss.method not in methods:
        ss.method = methods[-1]
    S.clamp("outage_start", 10.0, max(10.0, dur - 40.0))
    eng = S.is_engineer()
    widths = ([1.45] if show_method else []) + [2.25, 1.6] + ([0.9] if eng else []) + ([1.55, 0.9] if demo_page else [])
    with st.container(border=True):
        c = st.columns(widths, vertical_alignment="bottom", gap="small")
        i = 0
        if show_method:
            c[i].selectbox("Localization method", methods, format_func=lambda m: be.METHOD_LABEL[m],
                           help="Sensors fused by the team's EKF. GPS is used whenever it is available.",
                           **S.bind("method", scope))
            i += 1
        c[i].segmented_control(f"{label_outage} (simulated)", sorted({0, 10, 30, 60, 120, int(ss.outage_dur)}),
                               format_func=lambda v: "Off" if v == 0 else f"{v}s", required=True,
                               help="GPS fixes are withheld from the estimator for this long, as under dense canopy.",
                               **S.bind("outage_dur", scope))
        c[i + 1].slider("Outage starts at", 10.0, float(max(11.0, dur - 40.0)), step=5.0, format="%.0f s",
                        disabled=ss.outage_dur == 0, **S.bind("outage_start", scope))
        i += 2
        if eng:
            c[i].toggle("Multipath", help="Inject a 2.5 m GPS jump for 4 s before the outage (tests the "
                                          "consistency gate).", **S.bind("multipath", scope))
            i += 1
        out = {"demo": False, "reset": False}
        if demo_page:
            out["demo"] = c[i].button("▶ Run GPS Failure Demo", type="primary", width="stretch", key=f"demo_{scope}",
                                      help="Odometry + IMU + GPS · 60 s simulated outage at 100 s · plays from "
                                           "15 s before GPS drops.")
            out["reset"] = c[i + 1].button("⟲ Reset", width="stretch", key=f"reset_{scope}",
                                           help="Stop playback and return to the moment before the outage.")
    return out


def start_demo(page: str, tkey: str, streams):
    ss = st.session_state
    methods = be.methods_for(streams)
    a = DEMO["outage_start"]
    S.set_values(method=DEMO["method"] if DEMO["method"] in methods else methods[-1], outage_dur=DEMO["outage_dur"],
                 outage_start=a, multipath=False, speed=4, playing=page,
                 demo_end=a + DEMO["outage_dur"] + DEMO["tail_s"], viz="3D orchard")
    ss[tkey] = a - DEMO["lead_s"]


def reset_demo(tkey: str):
    ss = st.session_state
    ss.playing = None
    ss.demo_end = None
    ss[tkey] = max(0.0, float(ss.outage_start) - DEMO["lead_s"]) if ss.outage_dur else 0.0


# -------------------------------------------------------------- tiles -----

def tile(label: str, value_html: str, sub: str = "", accent: str | None = None, wide: bool = False,
         tag: str = "") -> str:
    top = f"border-top:3px solid {accent};" if accent else ""
    # the tag sits next to the value (the label line truncates with an ellipsis, which could hide it)
    tag_html = f" <span style='font-size:.8rem;vertical-align:middle'>{tag}</span>" if tag else ""
    return (f"<div class='mtile{' wide' if wide else ''}' style='{top}'><div class='m-l'>{esc(label)}</div>"
            f"<div class='m-v'>{value_html}{tag_html}</div><div class='m-s'>{sub}</div></div>")


GPS_STYLE = {ops.GPS_AVAILABLE: (GOOD, "●"), ops.GPS_DEGRADED: (WARN, "▲"), ops.GPS_LOST: (CRIT, "■")}


def operator_tiles(sit: ops.Situation, trusted: float, stop: float, gauge_html: str, speed_label: str,
                   gps_source: str | None = None) -> str:
    gc, gi = GPS_STYLE[sit.gps]
    gps_sub = ("fix received %.1f s ago" % sit.gps_age if sit.gps == ops.GPS_AVAILABLE else
               "a reading was rejected (disagreed with motion sensors)" if sit.gps == ops.GPS_DEGRADED else
               f"no fix for {sit.gps_age:.0f} s")
    gps_label = "GPS status"
    if gps_source == "proxy_from_reference":
        gps_label, gps_sub = "GPS status (simulated proxy)", gps_sub + " · proxy GPS, not recorded fixes"
    s = STATE[sit.nav_state]
    return ("<div class='mgrid'>"
            + tile(gps_label, f"<span style='color:{gc}'>{gi}</span> {sit.gps}", gps_sub, gc)
            + tile("Robot speed", f"{sit.speed:.2f} m/s", speed_label, None)
            + tile("Position uncertainty (model 95 %)", f"± {sit.r95:.2f} m",
                   f"The estimator's 95 % region — not yet validated against ground truth{gauge_html}",
                   s["color"], wide=True, tag=kind("uncalibrated"))
            + "</div>")


# ----------------------------------------------------------- checklist ----

CHK_STYLE = {ops.YES: ("✓", GOOD, "#0b0b0b", "Allowed"), ops.ADVISED: ("!", WARN, "#0b0b0b", "Recommended now"),
             ops.NO: ("✕", CRIT, "#ffffff", "Not advised"), ops.NA: ("–", "#2a3542", MUTED, "Not needed")}


CHECKLIST_NOTE = ("Advisory only — derived from the team's integrity monitor and the simulated-crew safety rules. "
                  "The dashboard does not control the robot.")


def checklist_html(items, note: str = CHECKLIST_NOTE) -> str:
    rows = []
    for name, status, why in items:
        ic, bg, fg, lab = CHK_STYLE[status]
        if name == "Stop recommended" and status == ops.ADVISED:
            bg, fg = CRIT, "#ffffff"
        dim = " dim" if status == ops.NA else ""
        rows.append(f"<div class='c{dim}'><div class='ic' style='background:{bg};color:{fg}'>{ic}</div>"
                    f"<div><div class='t'>{esc(name)}</div><div class='w'>{esc(lab)} · {esc(why)}</div></div></div>")
    return ("<div class='chk'>" + "".join(rows) + f"</div><div class='small' style='margin-top:6px'>{esc(note)}</div>")


def alerts_html(keys: list[str], lang: str, table: dict | None = None) -> str:
    """``table``: alert texts keyed like ops.ALERTS (default). Teammate replay passes its own."""
    out = []
    for k in keys[:3]:
        icon, state, en, en_do, es, es_do = (table or ops.ALERTS)[k]
        col = STATE[state]["color"]
        en_b = (f"<div><span class='lang'>EN</span><span class='a-en'>{esc(en)}</span>"
                f"<div class='a-do'>→ {esc(en_do)}</div></div>")
        es_b = (f"<div class='a-es'><span class='lang'>ES</span>{esc(es)}<div class='a-do'>→ {esc(es_do)}</div></div>")
        body = (en_b if lang != "Español" else "") + (es_b if lang != "English" else "")
        out.append(f"<div class='alrt' style='border-left:4px solid {col}'><div class='a-h'>"
                   f"<div class='a-i' style='background:{col}'>{icon}</div><div style='min-width:0'>{body}</div>"
                   f"</div></div>")
    return "".join(out)


# ------------------------------------------------------------- events -----

def event_chips(evs, page: str, tkey: str, t_max: float, limit: int = 8):
    """Timestamp-linked events: each chip jumps the shared replay time."""
    if evs is None or not len(evs):
        return
    bar = st.container(horizontal=True, gap="small", vertical_alignment="center")
    bar.markdown("<span class='tb-label'>Jump to</span>", unsafe_allow_html=True)
    for k, r in evs.head(limit).iterrows():
        icon = STATE[r.state]["icon"]
        if bar.button(f"{icon} {r.label} · {r.t_rel:.0f}s", key=f"ev_{page}_{k}",
                      help=f"Jump to {r.t_rel:.1f} s ({r.kind} event)"):
            st.session_state[tkey] = float(min(max(r.t_rel + 0.3, 0.0), t_max))
            st.session_state.playing = None
            st.rerun()


def explainer():
    with st.popover("ⓘ Uncertainty vs error"):
        st.markdown(
            "**Position uncertainty** (shown to operators) is what the robot *computes itself, in real time*, "
            "from its sensors: a 95 % radius — \"I'm probably within this distance of where I think I am\". "
            "It grows while GPS is missing.\n\n"
            "**Actual error** is the distance between the estimate and the RTK ground truth. It can only be "
            "computed **afterwards** (in replay), because a working robot doesn't know the truth. It's shown "
            "in Engineer mode.\n\n"
            "A trustworthy system keeps the uncertainty **at or above** the actual error about 95 % of the time. "
            "**This has not been validated yet**: on the real 06_14B run the region contained the RTK reference only "
            "~53 % of the time. The audit traced most of that to a 0.19 % scale mismatch between the live GPS "
            "conversion and the reference file (not yet resolved), so treat the value as **uncalibrated** "
            "(docs/UNCERTAINTY_CALIBRATION.md).")
