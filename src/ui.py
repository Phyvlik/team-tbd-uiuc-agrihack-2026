"""Shared Streamlit helpers: data-mode selection, caching, banners, plots.

Kept separate from the engine modules so the engine stays UI-free and
testable. Real-data mode never falls back to synthetic data: if loading
fails the page shows the error and stops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data_loader import (CITRUSFARM_SEQUENCES, DataUnavailableError, SensorStreams, list_processed_sequences,
                          load_processed)
from .outage_simulator import OutageScenario
from .pipeline import ExperimentResult, run_experiment
from .reliability_monitor import STATE_COLOR, STATE_ICON, STATE_LABEL
from .sensor_fusion import ellipse_points
from .synthetic_data import SYNTHETIC_LABEL, make_synthetic_sequence

APP_TITLE = "Agricultural Robot Localization Dashboard"

# Reference palette (dataviz skill): fixed categorical order, reserved status colours.
C_REF = "#52514e"      # reference trajectory: neutral ink, not a series hue
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_OUTAGE = "rgba(82,81,78,0.12)"
C_BAND = "rgba(42,120,214,0.15)"

MODE_SYN, MODE_REAL = "Synthetic Demo Mode", "Real CitrusFarm Mode"
SENSOR_NAMES = {"odom": "Wheel odometry", "imu": "IMU gyro", "lidar": "LiDAR odometry", "gps": "GPS"}
STREAM_OF = {"odom": "odom", "imu": "imu", "lidar": "lidar_odom", "gps": "gps"}


def page_setup(subtitle: str):
    st.set_page_config(page_title=f"{subtitle} · {APP_TITLE}", page_icon="🛰️", layout="wide")
    st.markdown(
        """<style>
        .block-container {padding-top: 2.2rem; max-width: 1400px;}
        div[data-testid="stMetricValue"] {font-size: 1.55rem;}
        .chip {display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.85rem;margin:2px 4px 2px 0;
               border:1px solid rgba(82,81,78,0.35);}
        .banner {padding:8px 14px;border-radius:6px;margin-bottom:10px;font-size:0.92rem;
                 border-left:5px solid #52514e;background:rgba(82,81,78,0.08);}
        .banner-syn {border-left-color:#eda100;background:rgba(237,161,0,0.10);}
        .state {padding:14px 18px;border-radius:8px;color:#0b0b0b;font-weight:600;font-size:1.15rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.caption(APP_TITLE)
    st.title(subtitle)


@st.cache_resource(show_spinner="Loading data…")
def _load(mode: str, seq: str | None) -> SensorStreams:
    if mode == MODE_SYN:
        return make_synthetic_sequence()
    return load_processed(seq)


def data_selector() -> SensorStreams:
    """Sidebar: data mode + sequence. Stops the page on real-data failure."""
    with st.sidebar:
        st.header("Data source")
        mode = st.radio("Mode", [MODE_SYN, MODE_REAL], key="mode",
                        help="Synthetic mode tests the workflow. Real mode uses extracted CitrusFarm recordings.")
        seq = None
        if mode == MODE_REAL:
            avail = list_processed_sequences()
            if not avail:
                st.error("No processed CitrusFarm sequences found in data/processed/. "
                         "Run `python scripts/validate_baseline.py` first.")
                st.stop()
            default = avail.index("06_14B_Jackal") if "06_14B_Jackal" in avail else 0
            seq = st.selectbox("Sequence", avail, index=default, key="seq")
            not_extracted = [s for s in CITRUSFARM_SEQUENCES if s not in avail]
            if not_extracted:
                st.caption(f"Not extracted yet: {', '.join(not_extracted)}")
    try:
        streams = _load(mode, seq)
    except DataUnavailableError as e:
        st.error(f"Real data could not be loaded — nothing will be substituted.\n\n{e}")
        st.stop()
    banner(streams)
    return streams


def banner(streams: SensorStreams, gps_source: str | None = None):
    if streams.mode == "synthetic":
        st.markdown(f"<div class='banner banner-syn'>⚠️ <b>{SYNTHETIC_LABEL}</b>. Error models are invented to "
                    f"exercise the software; numbers below say nothing about real-world accuracy.</div>",
                    unsafe_allow_html=True)
    else:
        av = streams.available()
        have = ", ".join(SENSOR_NAMES[k] for k, s in STREAM_OF.items() if av.get(s)) or "none"
        gps_note = ""
        if not av.get("gps"):
            gps_note = (" · Live GPS: <b>SIMULATED proxy</b> (reference + 3 cm noise, causal) until raw "
                        "<code>/piksi/navsatfix_best_fix</code> is extracted from base_*.bag")
        st.markdown(f"<div class='banner'>📡 <b>Real CitrusFarm data — {streams.name}</b> · "
                    f"Recorded sensor streams: {have}{gps_note}. Reference = post-processed RTK "
                    f"(<code>gt.csv</code>), used only for scoring.</div>", unsafe_allow_html=True)


def available_sensor_sets(streams: SensorStreams) -> dict[str, tuple]:
    av = streams.available()
    sets = {
        "Wheel odometry + GPS": ("odom", "gps"),
        "Wheel odometry + IMU + GPS": ("odom", "imu", "gps"),
        "Wheel odometry + IMU + LiDAR + GPS": ("odom", "imu", "lidar", "gps"),
        "Wheel odometry only (no GPS)": ("odom",),
        "Wheel odometry + IMU (no GPS)": ("odom", "imu"),
    }
    return {k: v for k, v in sets.items() if all(s == "gps" or av.get(STREAM_OF[s]) for s in v)}


@st.cache_data(show_spinner="Running estimator…", max_entries=64)
def cached_experiment(mode: str, seq: str | None, sensors: tuple, outages: tuple, jumps: tuple = (),
                      proxy_sigma: float = 0.03, trusted_r95: float = 0.3, stop_r95: float = 1.0) -> ExperimentResult:
    from .reliability_monitor import MonitorConfig
    streams = _load(mode, seq)
    sc = OutageScenario(outages=list(outages), multipath_jumps=list(jumps))
    return run_experiment(streams, sensors, sc, monitor=MonitorConfig(trusted_r95=trusted_r95, stop_r95=stop_r95),
                          proxy_sigma=proxy_sigma)


def current_key(streams: SensorStreams) -> tuple[str, str | None]:
    return (MODE_SYN, None) if streams.mode == "synthetic" else (MODE_REAL, streams.name)


# ---------------------------------------------------------------- plotting ---

def _decimate(df: pd.DataFrame, n: int = 4000) -> pd.DataFrame:
    step = max(1, len(df) // n)
    return df.iloc[::step]


def map_figure(streams: SensorStreams, res: ExperimentResult | None = None, t_now: float | None = None,
               extra_traces: list | None = None, height: int = 560, show_gps: bool = True) -> go.Figure:
    fig = go.Figure()
    ref = _decimate(streams.reference)
    fig.add_trace(go.Scattergl(x=ref.x, y=ref.y, mode="lines", name="Reference (RTK, scoring only)",
                               line=dict(color=C_REF, width=2, dash="dot"),
                               hovertemplate="reference<br>x %{x:.2f} m<br>y %{y:.2f} m<extra></extra>"))
    if res is not None:
        est = res.estimate if t_now is None else res.estimate[res.estimate.t <= t_now]
        e = _decimate(est)
        fig.add_trace(go.Scattergl(x=e.x_ant, y=e.y_ant, mode="lines", name=f"Estimate ({'+'.join(res.sensors)})",
                                   line=dict(color=C_SERIES[0], width=2),
                                   hovertemplate="estimate<br>x %{x:.2f} m<br>y %{y:.2f} m<extra></extra>"))
        for a, b in res.outages_abs:
            seg = est[(est.t >= a) & (est.t <= b)]
            if len(seg):
                fig.add_trace(go.Scattergl(x=seg.x_ant, y=seg.y_ant, mode="lines", name="Estimate during GPS outage",
                                           line=dict(color=C_SERIES[1], width=3), showlegend=bool(a == res.outages_abs[0][0]),
                                           hovertemplate="GPS withheld<br>x %{x:.2f}<br>y %{y:.2f}<extra></extra>"))
        if show_gps and res.gps_log is not None and len(res.gps_log):
            g = res.gps_log if t_now is None else res.gps_log[res.gps_log.t <= t_now]
            g = _decimate(g[g.available], 1500)
            fig.add_trace(go.Scattergl(x=g.x, y=g.y, mode="markers", name=f"GPS fixes used ({res.gps_source})",
                                       marker=dict(color=C_SERIES[2], size=4, opacity=0.5),
                                       hovertemplate="GPS fix<br>x %{x:.2f}<br>y %{y:.2f}<extra></extra>"))
        if t_now is not None and len(est):
            row = est.iloc[-1]
            ex, ey = ellipse_points(row, 0.95)
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself", fillcolor=C_BAND,
                                     line=dict(color=C_SERIES[0], width=1), name="95% position uncertainty",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[row.x_ant], y=[row.y_ant], mode="markers", name="Robot (estimated)",
                                     marker=dict(color=C_SERIES[0], size=13, symbol="circle",
                                                 line=dict(color="white", width=2)),
                                     hovertemplate="robot estimate<br>x %{x:.2f}<br>y %{y:.2f}<extra></extra>"))
    for tr in extra_traces or []:
        fig.add_trace(tr)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, font=dict(size=11)),
                      xaxis_title="x (m)", yaxis_title="y (m)", hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, gridcolor="rgba(82,81,78,0.12)")
    fig.update_xaxes(gridcolor="rgba(82,81,78,0.12)")
    return fig


def add_outage_bands(fig: go.Figure, outages_abs, t0: float, label: bool = True):
    for i, (a, b) in enumerate(outages_abs):
        fig.add_vrect(x0=a - t0, x1=b - t0, fillcolor=C_OUTAGE, line_width=0,
                      annotation_text="GPS withheld" if (label and i == 0) else None, annotation_position="top left")


def error_figure(res: ExperimentResult, t0: float, height: int = 330, target: float = 0.3,
                 log_y: bool = False, t_now: float | None = None) -> go.Figure:
    e, est = res.error, res.estimate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est.t - t0, y=est.r95, mode="lines", name="Estimator 95% radius (own estimate)",
                             line=dict(color=C_SERIES[0], width=0), fill="tozeroy", fillcolor=C_BAND,
                             hovertemplate="t %{x:.1f} s<br>r95 %{y:.2f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=e.t - t0, y=e.err, mode="lines", name="Actual error vs reference",
                             line=dict(color=C_SERIES[1], width=2),
                             hovertemplate="t %{x:.1f} s<br>error %{y:.3f} m<extra></extra>"))
    fig.add_hline(y=target, line=dict(color=C_REF, dash="dash", width=1),
                  annotation_text=f"{target} m target", annotation_position="bottom right")
    add_outage_bands(fig, res.outages_abs, t0)
    if t_now is not None:
        fig.add_vline(x=t_now - t0, line=dict(color=C_REF, width=1))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12), xaxis_title="time since start (s)",
                      yaxis_title="metres", yaxis_type="log" if log_y else "linear")
    return fig


def state_timeline(res: ExperimentResult, t0: float, height: int = 120) -> go.Figure:
    s = res.states
    ch = np.flatnonzero(np.r_[True, s.level.to_numpy()[1:] != s.level.to_numpy()[:-1]])
    fig = go.Figure()
    seen = set()
    for i, j in zip(ch, list(ch[1:]) + [len(s) - 1]):
        name = s.state.iloc[i]
        fig.add_trace(go.Bar(x=[s.t.iloc[j] - s.t.iloc[i] + 0.1], base=[s.t.iloc[i] - t0], y=["state"],
                             orientation="h", marker=dict(color=STATE_COLOR[name], line=dict(width=0)),
                             name=f"{STATE_ICON[name]} {STATE_LABEL[name]}", showlegend=name not in seen,
                             hovertemplate=f"{STATE_LABEL[name]}<br>%{{base:.1f}}–%{{x:.1f}} s<br>"
                                           f"{s.reason.iloc[i]}<extra></extra>"))
        seen.add(name)
    add_outage_bands(fig, res.outages_abs, t0, label=False)
    fig.update_layout(height=height, barmode="overlay", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.5), yaxis=dict(showticklabels=False),
                      xaxis_title="time since start (s)")
    return fig


def state_card(state: str, reason: str):
    st.markdown(f"<div class='state' style='background:{STATE_COLOR[state]}33;border-left:8px solid "
                f"{STATE_COLOR[state]}'>{STATE_ICON[state]} {STATE_LABEL[state]}<br>"
                f"<span style='font-weight:400;font-size:0.9rem'>{reason}</span></div>", unsafe_allow_html=True)


def fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}{unit}"
