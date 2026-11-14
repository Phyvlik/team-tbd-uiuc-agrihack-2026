"""Shared session state.

One canonical value per setting lives in ``st.session_state`` under a plain
key (e.g. ``"outage_dur"``). Widgets on different pages bind to that key via
:func:`bind`, so a change on the GPS simulator is what Mission Control shows
next, and values survive page switches (Streamlit otherwise drops the state of
widgets that are not rendered on the current page).
"""
from __future__ import annotations

import streamlit as st

from . import backend as be

DEFAULTS = {
    "view_mode": "Operator",
    "source": None,                 # filled with backend.default_source()
    "method": "odom+imu+gps",
    "outage_dur": 60,
    "outage_start": 100.0,
    "multipath": False,
    "trusted_r95": 0.30,
    "stop_r95": 1.00,
    "mc_t": 104.0,                  # Mission Control playback (s since start)
    "sim_t": None,                  # GPS simulator replay (s since start)
    "sim_fallbacks": ["C1", "C2"],
    "ws_t": None,                   # Worker safety playback
    "ws_crew1": 0.40,
    "ws_crew2": 0.75,
    "ws_offset": 2.5,
    "ws_min_buffer": 2.0,
    "ws_k": 1.0,
    "ws_slow": 3.0,
    "ws_proactive": True,
    "lang": "English + Español",
    "cmp_metric": "peak",
    "cmp_duration": 30,
    "cmp_basis": "auto",
    "playing": None,                # which page is auto-playing, if any
    "speed": 4,                     # playback speed multiplier
    "map_view": "Follow robot",
    "viz": "3D orchard",            # Mission Control visual: 3D orchard | 2D map | Rerun 3D
    "view3d": "Follow robot",       # 3D camera: Follow robot | Whole orchard
    "demo_end": None,               # playback auto-stops here during the one-click demo
    "show_crew": True,
    "rrd_eng": None,
    "rrd_op": None,
}


def init():
    ss = st.session_state
    for k, v in DEFAULTS.items():
        if k not in ss:
            ss[k] = be.default_source() if k == "source" else v
    if ss.source not in be.source_keys():
        ss.source = be.default_source()


def _copy(wkey: str, key: str):
    st.session_state[key] = st.session_state[wkey]


def bind(key: str, scope: str) -> dict:
    """Widget kwargs that read/write the canonical ``key``. Pass as ``**bind(...)``;
    never also pass ``value=``/``index=``."""
    wkey = f"_w_{scope}_{key}"
    st.session_state[wkey] = st.session_state[key]
    return {"key": wkey, "on_change": _copy, "args": (wkey, key)}


def clamp(key: str, lo: float, hi: float):
    v = st.session_state.get(key)
    if v is None or v < lo or v > hi:
        st.session_state[key] = min(max(v if v is not None else lo, lo), hi)


def set_values(**kw):
    for k, v in kw.items():
        st.session_state[k] = v


def is_engineer() -> bool:
    return st.session_state.get("view_mode") == "Engineer"


def thresholds() -> tuple[float, float]:
    return float(st.session_state.trusted_r95), float(st.session_state.stop_r95)


def scenario_outages() -> tuple:
    d = float(st.session_state.outage_dur)
    return ((float(st.session_state.outage_start), d),) if d > 0 else ()


def scenario_jumps() -> tuple:
    """Optional multipath jump: 2.5 m offset for 4 s just before the outage (tree multipath)."""
    if not st.session_state.multipath:
        return ()
    return ((max(5.0, float(st.session_state.outage_start) - 25.0), 4.0, 2.5),)
