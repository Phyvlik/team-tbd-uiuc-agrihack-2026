"""Reusable UI components (HTML rendered through st.markdown).

Every status indicator carries an icon + text label next to its colour, so
state is never conveyed by colour alone.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from . import backend as be
from .theme import CRIT, GOOD, INK, MUTED, NEUTRAL, STATE, SURFACE_2, WARN


def _md(s: str):
    st.markdown(s, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def fmt(v, nd: int = 2, unit: str = "", ne: str = "Not evaluated") -> str:
    if not be.finite(v):
        return ne
    return f"{float(v):.{nd}f}{unit}"


# ------------------------------------------------------------------ badges --

def badge(text: str, kind: str = "") -> str:
    return f"<span class='badge {kind}'>{esc(text)}</span>"


def provenance_badge(kind: str) -> str:
    return {
        "real": badge("● REAL CITRUSFARM DATA", "real"),
        "synthetic": badge("▲ SYNTHETIC DEMO", "synthetic"),
        "batch": badge("■ TEAM BATCH RESULTS", "batch"),
        "sim_gps": badge("SIMULATED OUTAGE", "synthetic"),
        "sim_workers": badge("SIMULATED WORKERS", "synthetic"),
        "ne": badge("NOT EVALUATED", "ne"),
    }[kind]


def state_pill(state: str) -> str:
    s = STATE[state]
    return (f"<span class='state-pill' style='background:{s['color']}26;border:1px solid {s['color']}'>"
            f"<span style='color:{s['color']}'>{s['icon']}</span>{esc(s['label'].upper())}</span>")


# ------------------------------------------------------------------ layout --

def page_header(eyebrow: str, title: str, subtitle: str, badges: list[str]):
    _md(f"<div class='pg-head'><div><div class='pg-eyebrow'>{esc(eyebrow)}</div>"
        f"<div class='pg-title'>{esc(title)}</div><div class='pg-sub'>{subtitle}</div></div>"
        f"<div class='badges'>{''.join(badges)}</div></div>")


def section(title: str, sub: str | None = None):
    _md(f"<div class='sec'>{esc(title)}</div>" + (f"<div class='sec-sub'>{sub}</div>" if sub else ""))


def notice(text: str, kind: str = ""):
    _md(f"<div class='notice {kind}'>{text}</div>")


def source_notice(info: be.SourceInfo, gps_source: str | None = None):
    if info.provenance == "synthetic":
        notice(f"<b>{esc(be.SYNTHETIC_LABEL)}.</b> Error models are invented to exercise the software; "
               f"numbers here say nothing about real-world accuracy.", "syn")
    else:
        gps = ""
        if gps_source == "proxy_from_reference":
            gps = (" · Live GPS is a <b>simulated proxy</b> (reference + 3 cm noise) because raw fixes "
                   "were not extracted for this sequence.")
        elif gps_source:
            gps = f" · Live GPS: <b>recorded receiver fixes</b> (<code>{esc(gps_source)}</code>)."
        notice(f"<b>Real CitrusFarm data — {esc(info.key)}.</b> Sensors: {esc(info.streams_text)}{gps} "
               f"Ground truth (post-processed RTK) is used only for scoring, never by the estimator.", "real")


def not_evaluated(what: str, why: str, needs: str | None = None):
    n = f"<br><span class='small'>Needed: {needs}</span>" if needs else ""
    _md(f"<div class='ne-box'><b>Not evaluated — {esc(what)}.</b> {why}{n}</div>")


# ------------------------------------------------------------------- cards --

def kpi_html(label: str, value: str, sub: str = "", accent: str | None = None, icon: str = "") -> str:
    bar = f"border-top:3px solid {accent};" if accent else ""
    ic = f"<span style='color:{accent};margin-right:6px'>{icon}</span>" if icon else ""
    return (f"<div class='card' style='{bar}'><div class='card-label'>{esc(label)}</div>"
            f"<div class='card-value'>{ic}{value}</div><div class='card-sub'>{sub}</div></div>")


def kpi_row(items: list[str]):
    _md(f"<div class='kpi-row'>{''.join(items)}</div>")


def status_hero(state: str, reason: str, label: str = "Navigation state", extra: str = "",
                title: str | None = None, meaning: str | None = None):
    s = STATE[state]
    pulse = " pulse" if state == "STOP" else ""
    _md(f"<div class='hero' style='background:linear-gradient(135deg,{s['color']}2e,{s['color']}0a);"
        f"border-color:{s['color']}88'><div class='h-top'>"
        f"<div class='h-icon{pulse}' style='background:{s['color']}'>{s['icon']}</div>"
        f"<div><div class='h-label'>{esc(label)}</div><div class='h-state'>{esc(title or s['label'])}</div></div></div>"
        f"<div class='h-reason'>{esc(meaning or s['meaning'])} · <b>{esc(reason)}</b>{extra}</div></div>")


def uncertainty_gauge(r95: float, trusted: float, stop: float, stop_label: str = "stop") -> str:
    """Bullet gauge: r95 against the Trusted / Stop limits (scale 0 → 2·stop)."""
    top = 2.0 * stop
    pos = lambda v: 100.0 * min(max(v, 0.0), top) / top  # noqa: E731
    a, b = pos(trusted), pos(stop)
    over = "" if r95 <= top else " +"
    return (f"<div class='gauge'><div class='g-track'>"
            f"<div class='g-seg' style='width:{a}%;background:{GOOD}'></div>"
            f"<div class='g-seg' style='width:{b - a}%;background:{WARN}'></div>"
            f"<div class='g-seg' style='width:{100 - b}%;background:{CRIT}'></div>"
            f"<div class='g-marker' style='left:calc({pos(r95)}% - 1.5px)'></div></div>"
            f"<div class='g-ticks'><span style='left:{a}%'>{trusted:.1f} target</span>"
            f"<span style='left:{b}%'>{stop:.1f} {esc(stop_label)}</span>"
            f"<span style='right:0;left:auto;transform:none'>{top:.0f} m{over}</span></div></div>")


SENSOR_STATUS = {
    "active": ("✓", GOOD, "Active"),
    "lost": ("!", WARN, "Signal lost"),
    "standby": ("○", NEUTRAL, "Available · not used"),
    "missing": ("✕", "#3a4540", "Not extracted"),
}


def sensor_panel(rows: list[tuple[str, str, str]]):
    """rows: (sensor name, status key, detail)."""
    cells = []
    for name, key, detail in rows:
        icon, color, label = SENSOR_STATUS[key]
        fg = "#0b0b0b" if key in ("active", "lost") else INK
        cells.append(f"<div class='sensor'><div class='s-dot' style='background:{color};color:{fg}'>{icon}</div>"
                     f"<div><div class='s-name'>{esc(name)}</div><div class='s-state'>{esc(label)}"
                     f"{(' · ' + esc(detail)) if detail else ''}</div></div></div>")
    _md(f"<div class='sensors'>{''.join(cells)}</div>")


def sensor_rows(streams, sensors: tuple, gps_ok: bool, gps_age: float, gps_source: str) -> list:
    av = streams.available()
    rows = []
    for k in ("odom", "imu", "lidar", "gps"):
        name = be.SENSOR_NAMES[k]
        have = av.get(be.STREAM_OF[k]) or k == "gps"
        if k == "gps":
            src = "simulated proxy" if gps_source == "proxy_from_reference" else (
                "synthetic" if gps_source == "synthetic" else "recorded fixes")
            if not gps_ok:
                rows.append((name, "lost", f"{gps_age:.0f} s since last fix"))
            else:
                rows.append((name, "active", src))
        elif k in sensors:
            rows.append((name, "active", ""))
        else:
            rows.append((name, "standby" if have else "missing", ""))
    return rows


RECOMMEND = {
    "TRUSTED": ("Continue autonomous operation", [
        "Position estimate is within the 0.3 m target.",
        "No action needed — keep monitoring the alert log.",
    ]),
    "DEGRADED": ("Slow down and watch the robot", [
        "Reduce speed; avoid entering rows where people are working.",
        "Expect GPS to return soon under thinner canopy.",
        "Be ready to pause if the state turns red.",
    ]),
    "STOP": ("Stop the robot and send a person", [
        "Position can no longer be trusted (uncertainty above the stop limit).",
        "Pause the robot, or take manual control.",
        "Resume only after GPS returns and the state turns green.",
    ]),
}


def recommendation(state: str, context: str = ""):
    title, bullets = RECOMMEND[state]
    s = STATE[state]
    lis = "".join(f"<li>{esc(b)}</li>" for b in bullets)
    ctx = f"<div class='small' style='margin-top:8px'>{context}</div>" if context else ""
    _md(f"<div class='rec' style='border-left:4px solid {s['color']}'>"
        f"<div class='card-label'>Recommended action</div>"
        f"<div class='r-title'><span style='color:{s['color']}'>{s['icon']}</span> {esc(title)}</div>"
        f"<ul>{lis}</ul>{ctx}</div>")


def alert_log(log: pd.DataFrame, t0: float, empty_text: str = "No alerts yet."):
    """log: rows with t, state, reason, r95 (from reliability_monitor.transitions)."""
    if log is None or not len(log):
        _md(f"<div class='small'>{esc(empty_text)}</div>")
        return
    items = []
    for _, r in log.iloc[::-1].iterrows():
        s = STATE[r.state]
        items.append(f"<div class='a' style='border-left:3px solid {s['color']}'>"
                     f"<div class='a-t'>{r.t - t0:7.1f} s</div>"
                     f"<div><div class='a-m'><span style='color:{s['color']}'>{s['icon']}</span> {esc(s['label'])}</div>"
                     f"<div class='a-r'>{esc(r.reason)} · r95 {r.r95:.2f} m</div></div></div>")
    _md(f"<div class='alog'>{''.join(items)}</div>")


def status_legend(extra: list[tuple[str, str, str]] | None = None, map_items: bool = False):
    """Always-visible key for the status colours (+ optional line items: label, colour, 'line'|'dash'|'dot')."""
    parts = [f"<span class='l'><span class='sw' style='background:{v['color']}'>{v['icon']}</span>"
             f"<b style='color:{INK}'>{esc(v['label'])}</b>&nbsp;{esc(v['meaning'].lower())}</span>"
             for v in STATE.values()]
    if map_items:
        parts.append("<span class='l'><span style='color:#eef2ef'>➤</span>Robot (estimated, heading)</span>")
        parts.append("<span class='l'><span class='sw' style='background:rgba(238,242,239,.18);border:1px solid #c3c9c4'>"
                     "</span>95 % uncertainty ellipse</span>")
        parts.append(f"<span class='l'><span style='color:{WARN};font-weight:800'>✕</span>GPS lost · "
                     f"<span style='color:{WARN};font-weight:800'>○</span>GPS restored</span>")
    for label, color, style in extra or []:
        bs = {"line": "solid", "dash": "dashed", "dot": "dotted"}[style]
        parts.append(f"<span class='l'><span class='ln' style='border-top-color:{color};border-top-style:{bs}'>"
                     f"</span>{esc(label)}</span>")
    _md(f"<div class='legend'>{''.join(parts)}</div>")


def cost_dots(rank: int, n: int = 4) -> str:
    return "".join("$" if i < rank else "<span class='off'>$</span>" for i in range(n))


def playback_controls(page: str, key: str, t_max: float, step: float = 0.5,
                      jump_to: list[tuple[str, float]] | None = None, label: str = "Playback"):
    """Play/pause + scrubber bound to session key ``key`` (seconds since start)."""
    from . import state as S  # local import to avoid cycle
    S.clamp(key, 0.0, float(t_max))
    playing = st.session_state.playing == page
    st.slider(label, 0.0, float(t_max), step=step, format="%.1f s", label_visibility="collapsed",
              **S.bind(key, page))
    bar = st.container(horizontal=True, gap="small", vertical_alignment="center")
    c = [bar] * (2 + len(jump_to or []))
    if c[0].button("⏸ Pause" if playing else "▶ Play", key=f"pp_{page}",
                   type="primary" if not playing else "secondary"):
        st.session_state.playing = None if playing else page
        if not playing and st.session_state[key] >= t_max - 1:
            st.session_state[key] = 0.0
        st.rerun()
    if c[1].button("⟲ Reset", key=f"rs_{page}"):
        st.session_state[key] = 0.0
        st.session_state.playing = None
        st.rerun()
    for i, (lab, tv) in enumerate(jump_to or []):
        if c[2 + i].button(lab, key=f"jt_{page}_{i}"):
            st.session_state[key] = float(min(max(tv, 0.0), t_max))
            st.session_state.playing = None
            st.rerun()
    return bar


def advance(page: str, key: str, t_max: float, dt: float):
    if st.session_state.playing == page:
        end = st.session_state.get("demo_end")
        limit = min(t_max, end) if end is not None else t_max
        nv = float(st.session_state[key]) + dt
        if nv >= limit:
            nv = limit
            st.session_state.playing = None
            st.session_state.demo_end = None
        st.session_state[key] = nv


def df_table(df: pd.DataFrame, height: int | None = None):
    st.dataframe(df, hide_index=True, width="stretch", height=height or "auto")


def ne_value(v, nd=2, unit="") -> str:
    return fmt(v, nd, unit, ne="Not evaluated")


def first_finite(*vals):
    for v in vals:
        if be.finite(v):
            return float(v)
    return np.nan
