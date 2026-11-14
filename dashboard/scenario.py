"""Cached scenario pieces shared by the operator view, Worker Safety and the Rerun export."""
from __future__ import annotations

import numpy as np
import streamlit as st

from . import backend as be
from . import orchard as orc

DEMO = dict(method="odom+imu+gps", outage_start=100.0, outage_dur=60, lead_s=15.0, tail_s=15.0)


@st.cache_resource(show_spinner=False)
def orchard(key: str) -> orc.Orchard:
    geo = be.APP_DIR / "team_runs" / f"{key}__orchard_geometry.csv"
    return orc.build_orchard(be.load_streams(key), geo)


@st.cache_data(show_spinner=False, max_entries=48)
def spray(key: str, sensors: tuple, outages: tuple, jumps: tuple, trusted: float, stop: float):
    res = be.experiment(key, sensors, outages, jumps, trusted, stop)
    o = orchard(key)
    return orc.simulate_spray(o.trees, res.estimate, be.load_streams(key).reference)


def demo_crew(key: str, outages: tuple, offset: float = 2.5):
    """One SIMULATED crew placed beside the route where the robot will be ~25 s into the outage
    (or 40 % along the route if there is no outage)."""
    s = be.load_streams(key)
    ref = s.reference
    t0 = s.time_span()[0]
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ref.x), np.diff(ref.y)))])
    if outages:
        a, dur = outages[0]
        i = int(np.searchsorted(ref.t.to_numpy(), t0 + a + min(25.0, 0.6 * dur)))
        frac = d[min(i, len(d) - 1)] / d[-1]
    else:
        frac = 0.4
    return be.place_workers_along_route(ref, (float(np.clip(frac, 0.05, 0.95)),), offset)


@st.cache_data(show_spinner=False, max_entries=48)
def safety(key: str, sensors: tuple, outages: tuple, jumps: tuple, trusted: float, stop: float,
           min_buffer: float, k: float, slow: float, offset: float):
    res = be.experiment(key, sensors, outages, jumps, trusted, stop)
    zones = demo_crew(key, outages, offset)
    cfg = be.SafetyConfig(min_buffer_m=min_buffer, k_uncertainty=k, slow_margin_m=slow, risk_margin_m=1.0)
    return be.evaluate_safety(res.estimate, res.states, zones, cfg), zones
