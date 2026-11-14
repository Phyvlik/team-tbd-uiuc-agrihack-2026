"""Plotly figure builders. All take backend outputs and return go.Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import backend as be
from .theme import CONFIG_COLOR, CRIT, GOOD, GPS_FIX, INK, INK_2, MUTED, REF, SERIES, STATE, SURFACE, WARN

OUTAGE_FILL = "rgba(250,178,25,0.07)"
LEVEL_STATE = {0: "TRUSTED", 1: "DEGRADED", 2: "STOP"}
CONFIG_SYMBOL = {"C1": "circle", "C2": "diamond", "C3": "square", "C4": "triangle-up"}


def _decimate(df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    step = max(1, len(df) // n)
    return df.iloc[::step] if step > 1 else df


def _equal_axes(fig: go.Figure):
    fig.update_yaxes(scaleanchor="x", scaleratio=1)


def _base_layout(fig: go.Figure, height: int, legend_bottom: bool = True, **kw):
    kw.setdefault("hovermode", "closest")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", y=-0.13 if legend_bottom else 1.02,
                                  yanchor="top" if legend_bottom else "bottom", x=0, font=dict(size=11)),
                      uirevision=kw.pop("uirevision", None), **kw)
    fig.update_xaxes(automargin=True, title_standoff=6)
    fig.update_yaxes(automargin=True, title_standoff=6)


LOG_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def _log_y(fig: go.Figure, title: str, data_min: float | None = None, data_max: float | None = None):
    """Log y-axis with clean ticks. Also moves add_hline annotations into log10 units
    (Plotly places them in raw units on log axes, which misplaces the labels)."""
    kw = dict(title_text=title, type="log", tickvals=LOG_TICKS, ticktext=[f"{v:g}" for v in LOG_TICKS])
    if data_min is not None and data_max is not None and np.isfinite(data_min) and np.isfinite(data_max):
        lo, hi = max(min(data_min, 0.3) / 1.6, 1e-3), max(data_max, 0.3) * 1.8
        kw["range"] = [np.log10(lo), np.log10(hi)]
    fig.update_yaxes(**kw)
    for a in fig.layout.annotations or []:
        if a.yref == "y" and a.y is not None and a.y > 0 and not getattr(a, "_logfixed", False):
            a.y = float(np.log10(a.y))


def _segments_by_state(est: pd.DataFrame, levels: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """x/y arrays per state level, with NaN breaks; each run includes the next point so the path is continuous."""
    x, y = est["x_ant"].to_numpy(), est["y_ant"].to_numpy()
    out = {0: ([], []), 1: ([], []), 2: ([], [])}
    if len(x) == 0:
        return {k: (np.array([]), np.array([])) for k in out}
    ch = np.flatnonzero(np.r_[True, levels[1:] != levels[:-1]])
    ends = list(ch[1:]) + [len(x)]
    for a, b in zip(ch, ends):
        lv = int(levels[a])
        j = min(b + 1, len(x))
        out[lv][0].extend(list(x[a:j]) + [np.nan])
        out[lv][1].extend(list(y[a:j]) + [np.nan])
    return {k: (np.array(v[0], float), np.array(v[1], float)) for k, v in out.items()}


def orchard_rows_traces(streams) -> list[go.BaseTraceType]:
    """Tree lines for the synthetic orchard (layout known by construction). Real data has no row map."""
    rows = streams.meta.get("rows") if streams.mode == "synthetic" else None
    if not rows:
        return []
    sp = float(streams.meta.get("row_spacing", 6.5))
    xs, ys = [], []
    for i, (x0, y0, x1, y1) in enumerate(rows):
        for dx in (-sp / 2, sp / 2):
            xs += [x0 + dx, x1 + dx, None]
            ys += [y0, y1, None]
    return [go.Scatter(x=xs, y=ys, mode="lines", name="Tree rows (synthetic layout)",
                       line=dict(color="rgba(25,158,112,0.35)", width=7), hoverinfo="skip")]


def trajectory_map(streams, res, t_now: float | None, *, height: int = 620, follow: float | None = None,
                   show_reference: bool = True, show_gps: bool = True, extra: list | None = None,
                   window: tuple | None = None, uirevision: str | None = None,
                   ref_label: str = "True path (RTK reference · scoring only)", legend: bool = False) -> go.Figure:
    """Main map: estimate coloured by integrity state up to ``t_now``, reference, GPS fixes,
    95 % uncertainty ellipse and a heading arrow for the robot."""
    fig = go.Figure()
    for tr in orchard_rows_traces(streams):
        fig.add_trace(tr)
    if show_reference:
        ref = _decimate(streams.reference, 2500)
        fig.add_trace(go.Scatter(x=ref.x, y=ref.y, mode="lines", name=ref_label,
                                   line=dict(color=REF, width=1.5, dash="dot"), opacity=0.75,
                                   hovertemplate="true path<br>x %{x:.1f} m · y %{y:.1f} m<extra></extra>"))
    if show_gps and res.gps_log is not None and len(res.gps_log):
        g = res.gps_log if t_now is None else res.gps_log[res.gps_log.t <= t_now]
        g = _decimate(g[g.available], 1200)
        src = {"proxy_from_reference": "simulated proxy", "synthetic": "synthetic"}.get(res.gps_source, "recorded")
        fig.add_trace(go.Scatter(x=g.x, y=g.y, mode="markers", name=f"GPS fixes received ({src})",
                                   marker=dict(color=GPS_FIX, size=3, opacity=0.35),
                                   hovertemplate="GPS fix<br>x %{x:.2f} · y %{y:.2f}<extra></extra>"))
    est_all, states = res.estimate, res.states
    m = np.ones(len(est_all), bool) if t_now is None else (est_all.t.to_numpy() <= t_now)
    est, lv = est_all[m], states.level.to_numpy()[m]
    step = max(1, len(est) // 3000)
    segs = _segments_by_state(est.iloc[::step], lv[::step])
    for level in (0, 1, 2):
        s = STATE[LEVEL_STATE[level]]
        xs, ys = segs[level]
        fig.add_trace(go.Scatter(x=xs if len(xs) else [None], y=ys if len(ys) else [None], mode="lines",
                                   name=f"{s['icon']} Estimate · {s['label']}",
                                   line=dict(color=s["color"], width=3.2),
                                   hovertemplate=f"estimate · {s['label']}<br>x %{{x:.2f}} m · y %{{y:.2f}} m<extra></extra>"))
    # outage start / end markers
    for k, (a, b) in enumerate(res.outages_abs):
        for tt, lab, sym in ((a, "GPS lost", "x-thin"), (b, "GPS restored", "circle-open")):
            if t_now is not None and tt > t_now:
                continue
            i = be.at_time(est_all, tt)
            r = est_all.iloc[i]
            fig.add_trace(go.Scatter(x=[r.x_ant], y=[r.y_ant], mode="markers+text", text=[lab],
                                     textposition="top center", textfont=dict(color=INK, size=11),
                                     marker=dict(symbol=sym, size=14, color=WARN, line=dict(width=3, color=WARN)),
                                     name=lab, showlegend=False,
                                     hovertemplate=f"{lab} at %{{x:.1f}}, %{{y:.1f}}<extra></extra>"))
    for tr in extra or []:
        fig.add_trace(tr)
    if t_now is not None and len(est):
        row = est.iloc[-1]
        stt = STATE[LEVEL_STATE[int(lv[-1])]]
        ex, ey = be.ellipse_points(row, 0.95)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself", fillcolor=_rgba(stt["color"], 0.18),
                                 line=dict(color=stt["color"], width=1.5), name="95 % position uncertainty",
                                 hovertemplate=f"95 % uncertainty radius {row.r95:.2f} m<extra></extra>"))
        ang = 90.0 - np.degrees(row.theta)
        fig.add_trace(go.Scatter(x=[row.x_ant], y=[row.y_ant], mode="markers", name="Robot (estimated)",
                                 marker=dict(symbol="arrow", size=22, angle=ang, color=INK,
                                             line=dict(color=stt["color"], width=2.5)),
                                 hovertemplate=(f"robot estimate<br>x %{{x:.2f}} · y %{{y:.2f}}<br>"
                                                f"heading {np.degrees(row.theta) % 360:.0f}°<br>"
                                                f"state {stt['label']}<extra></extra>")))
        if follow:
            r = max(follow, 2.5 * float(row.r95))
            fig.update_xaxes(range=[row.x_ant - r, row.x_ant + r])
            fig.update_yaxes(range=[row.y_ant - r * 0.62, row.y_ant + r * 0.62])
    if window is not None:
        (x0, x1), (y0, y1) = window
        fig.update_xaxes(range=[x0, x1])
        fig.update_yaxes(range=[y0, y1])
    _base_layout(fig, height, uirevision=uirevision, showlegend=legend)
    fig.update_xaxes(title_text="East–west (m, local frame)", showgrid=True)
    fig.update_yaxes(title_text="North–south (m)", showgrid=True)
    _equal_axes(fig)
    return fig


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def outage_window(res, streams, pad_s: float = 15.0, margin: float = 6.0):
    """Axis window covering the reference + estimate around the first outage."""
    if not res.outages_abs:
        return None
    a, b = res.outages_abs[0]
    ref, est = streams.reference, res.estimate
    r = ref[(ref.t >= a - pad_s) & (ref.t <= b + pad_s)]
    e = est[(est.t >= a - pad_s) & (est.t <= b + pad_s)]
    xs = np.r_[r.x.to_numpy(), e.x_ant.to_numpy()]
    ys = np.r_[r.y.to_numpy(), e.y_ant.to_numpy()]
    if not len(xs):
        return None
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), (ys.max() - ys.min()) * 1.6, 12.0) / 2 + margin
    return (cx - half, cx + half), (cy - half / 1.6, cy + half / 1.6)


def add_outage_bands(fig: go.Figure, outages_abs, t0: float, label: bool = True, row=None, col=None):
    for i, (a, b) in enumerate(outages_abs):
        ann = (dict(annotation_text="GPS outage", annotation_position="top left",
                    annotation_font=dict(color=WARN, size=11)) if (label and i == 0) else {})
        fig.add_vrect(x0=a - t0, x1=b - t0, fillcolor=OUTAGE_FILL, line=dict(color=WARN, width=1, dash="dot"),
                      layer="below", **ann)


def error_timeline(res, t0: float, t_now: float | None = None, target: float = 0.3, stop: float = 1.0,
                   height: int = 300, log_y: bool = True, xrange: tuple | None = None) -> go.Figure:
    est, err = res.estimate, res.error
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est.t - t0, y=est.r95.clip(lower=1e-3), mode="lines",
                             name="Estimator's own 95 % uncertainty (what the robot knows)",
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate="t %{x:.1f} s<br>uncertainty %{y:.2f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=err.t - t0, y=err.err.clip(lower=1e-3), mode="lines",
                             name="Actual error vs RTK truth (offline check)",
                             line=dict(color=SERIES[1], width=2),
                             hovertemplate="t %{x:.1f} s<br>actual error %{y:.3f} m<extra></extra>"))
    fig.add_hline(y=target, line=dict(color=GOOD, dash="dash", width=1),
                  annotation_text=f"{target:.1f} m target", annotation_position="bottom left",
                  annotation_font=dict(color=INK_2, size=11))
    fig.add_hline(y=stop, line=dict(color=CRIT, dash="dash", width=1),
                  annotation_text=f"{stop:.1f} m stop limit", annotation_position="top left",
                  annotation_font=dict(color=INK_2, size=11))
    add_outage_bands(fig, res.outages_abs, t0)
    if t_now is not None:
        fig.add_vline(x=t_now - t0, line=dict(color=INK, width=1.2))
    _base_layout(fig, height, legend_bottom=False, hovermode="x unified")
    fig.update_xaxes(title_text="time since start (s)", range=list(xrange) if xrange else None)
    if log_y:
        _log_y(fig, "metres (log scale)")
    else:
        fig.update_yaxes(title_text="metres")
    return fig


def state_ribbon(res, t0: float, t_now: float | None = None, height: int = 96,
                 xrange: tuple | None = None, labels: dict | None = None) -> go.Figure:
    s = res.states
    lv = s.level.to_numpy()
    ch = np.flatnonzero(np.r_[True, lv[1:] != lv[:-1]])
    fig = go.Figure()
    seen = set()
    for i, j in zip(ch, list(ch[1:]) + [len(s) - 1]):
        name = s.state.iloc[i]
        st_ = STATE[name]
        lab = (labels or {}).get(name, st_["label"])
        fig.add_trace(go.Bar(x=[max(s.t.iloc[j] - s.t.iloc[i], 0.1)], base=[s.t.iloc[i] - t0], y=[""],
                             orientation="h", marker=dict(color=st_["color"], line=dict(width=0)),
                             name=f"{st_['icon']} {lab}", showlegend=name not in seen,
                             legendgroup=name,
                             hovertemplate=(f"{lab}<br>from %{{base:.1f}} s for %{{x:.1f}} s<br>"
                                            f"{s.reason.iloc[i]}<extra></extra>")))
        seen.add(name)
    add_outage_bands(fig, res.outages_abs, t0, label=False)
    if t_now is not None:
        fig.add_vline(x=t_now - t0, line=dict(color=INK, width=1.5))
    _base_layout(fig, height, legend_bottom=False, barmode="overlay", bargap=0.15)
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8), legend=dict(y=1.0, yanchor="bottom", x=0))
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(title_text=None, range=list(xrange) if xrange else None)
    return fig


def gps_consistency(res, t0: float, gate: float = 13.816, height: int = 240) -> go.Figure:
    """Engineer: GPS innovation consistency (NIS) — accepted vs rejected fixes."""
    ev = res.gps_events
    fig = go.Figure()
    if ev is None or not len(ev):
        return fig
    for acc, name, color, sym in ((True, "Accepted fix", SERIES[2], "circle"),
                                  (False, "Rejected by consistency gate", SERIES[1], "x")):
        e = _decimate(ev[ev.accepted == acc], 1500)
        if len(e):
            fig.add_trace(go.Scatter(x=e.t - t0, y=e.nis.clip(lower=1e-3), mode="markers", name=name,
                                       marker=dict(color=color, size=5 if acc else 9, symbol=sym, opacity=0.7),
                                       hovertemplate="t %{x:.1f} s<br>NIS %{y:.2f}<extra></extra>"))
    fig.add_hline(y=gate, line=dict(color=CRIT, dash="dash", width=1), annotation_text="rejection gate (χ², 99.9 %)",
                  annotation_position="top left", annotation_font=dict(color=INK_2, size=11))
    add_outage_bands(fig, res.outages_abs, t0)
    _base_layout(fig, height, legend_bottom=False)
    fig.update_xaxes(title_text="time since start (s)")
    _log_y(fig, "NIS (log)")
    return fig


def compare_trajectories(streams, runs: dict, t_now: float | None = None, height: int = 460,
                         window: tuple | None = None) -> go.Figure:
    """Engineer: several estimates on one map, categorical colours in fixed order (by config id)."""
    fig = go.Figure()
    ref = _decimate(streams.reference, 2500)
    fig.add_trace(go.Scatter(x=ref.x, y=ref.y, mode="lines", name="True path (RTK)",
                               line=dict(color=REF, width=1.5, dash="dot")))
    for i, (label, (cid, res)) in enumerate(runs.items()):
        e = res.estimate if t_now is None else res.estimate[res.estimate.t <= t_now]
        e = _decimate(e, 2500)
        fig.add_trace(go.Scatter(x=e.x_ant, y=e.y_ant, mode="lines", name=label,
                                   line=dict(color=CONFIG_COLOR.get(cid, SERIES[i % 4]), width=2.2),
                                   hovertemplate=f"{label}<br>x %{{x:.2f}} · y %{{y:.2f}}<extra></extra>"))
    if window is not None:
        fig.update_xaxes(range=list(window[0]))
        fig.update_yaxes(range=list(window[1]))
    _base_layout(fig, height)
    fig.update_xaxes(title_text="East–west (m)")
    fig.update_yaxes(title_text="North–south (m)")
    _equal_axes(fig)
    return fig


def compare_errors(runs: dict, t0: float, outages_abs, target: float = 0.3, height: int = 300,
                   xrange: tuple | None = None, t_now: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (label, (cid, res)) in enumerate(runs.items()):
        e = res.error
        fig.add_trace(go.Scatter(x=e.t - t0, y=e.err.clip(lower=1e-3), mode="lines", name=label,
                                 line=dict(color=CONFIG_COLOR.get(cid, SERIES[i % 4]), width=2),
                                 hovertemplate=f"{label}<br>t %{{x:.1f}} s<br>error %{{y:.2f}} m<extra></extra>"))
    fig.add_hline(y=target, line=dict(color=GOOD, dash="dash", width=1), annotation_text=f"{target} m target",
                  annotation_position="bottom left", annotation_font=dict(color=INK_2, size=11))
    add_outage_bands(fig, outages_abs, t0)
    if t_now is not None:
        fig.add_vline(x=t_now - t0, line=dict(color=INK, width=1.2))
    _base_layout(fig, height, legend_bottom=False, hovermode="x unified")
    fig.update_xaxes(title_text="time since start (s)", range=list(xrange) if xrange else None)
    _log_y(fig, "actual error (m, log)")
    return fig


def config_bars(df: pd.DataFrame, value: str, label: str, target: float | None = 0.3, height: int = 300,
                log_y: bool = True) -> go.Figure:
    """One bar per config C1–C4; Not-evaluated configs get a dashed placeholder with a text label."""
    fig = go.Figure()
    ymax = np.nanmax(df[value].to_numpy(float)) if df[value].notna().any() else 1.0
    for _, r in df.iterrows():
        cid = r.config
        ok = be.finite(r[value])
        if ok:
            fig.add_trace(go.Bar(x=[f"{cid}"], y=[r[value]], name=f"{cid} · {be.CONFIG_LABEL[cid]}",
                                 marker=dict(color=CONFIG_COLOR[cid], cornerradius=4),
                                 text=[f"{r[value]:.2f}"], textposition="outside",
                                 textfont=dict(color=INK, size=12),
                                 hovertemplate=f"{cid} · {be.CONFIG_LABEL[cid]}<br>{label}: %{{y:.3f}}<extra></extra>"))
        else:
            fig.add_trace(go.Bar(x=[f"{cid}"], y=[ymax if not log_y else ymax], name=f"{cid} · not evaluated",
                                 marker=dict(color="rgba(0,0,0,0)", line=dict(color=MUTED, width=1.2),
                                             pattern=dict(shape="/", fgcolor="#3a4540", size=8)),
                                 text=["Not evaluated"], textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color=MUTED, size=11), showlegend=False,
                                 hovertemplate=f"{cid}: not evaluated<extra></extra>"))
    if target is not None:
        fig.add_hline(y=target, line=dict(color=GOOD, dash="dash", width=1), annotation_text=f"{target} m target",
                      annotation_position="top right", annotation_font=dict(color=INK_2, size=11))
    _base_layout(fig, height, legend_bottom=True, bargap=0.35)
    vals = df[value].to_numpy(float)
    if log_y:
        _log_y(fig, label, np.nanmin(vals) if np.isfinite(vals).any() else 0.3,
               np.nanmax(vals) if np.isfinite(vals).any() else 1.0)
    else:
        fig.update_yaxes(title_text=label)
    fig.update_layout(showlegend=False)
    return fig


def robustness_chart(summary: pd.DataFrame, target: float = 0.3, height: int = 330) -> go.Figure:
    """Peak error vs outage duration per config: median line + min–max whiskers (one axis)."""
    fig = go.Figure()
    for cid in sorted(summary.config.unique()):
        d = summary[summary.config == cid].sort_values("duration_s")
        fig.add_trace(go.Scatter(
            x=d.duration_s, y=d["median"], mode="lines+markers+text",
            name=f"{cid} · {be.CONFIG_LABEL[cid]} + GPS",
            line=dict(color=CONFIG_COLOR[cid], width=2),
            marker=dict(size=9, symbol=CONFIG_SYMBOL[cid], color=CONFIG_COLOR[cid], line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=d["max"] - d["median"],
                         arrayminus=d["median"] - d["min"], color=CONFIG_COLOR[cid], thickness=1, width=4),
            text=[cid if i == len(d) - 1 else "" for i in range(len(d))], textposition="middle right",
            textfont=dict(color=INK_2, size=11),
            customdata=np.c_[d["n"], d["min"], d["max"]],
            hovertemplate=(f"{cid}<br>%{{x:.0f}} s outage<br>median peak error %{{y:.2f}} m<br>"
                           f"range %{{customdata[1]:.2f}}–%{{customdata[2]:.2f}} m<br>n = %{{customdata[0]}} runs<extra></extra>")))
    fig.add_hline(y=target, line=dict(color=GOOD, dash="dash", width=1), annotation_text=f"{target} m target",
                  annotation_position="bottom right", annotation_font=dict(color=INK_2, size=11))
    _base_layout(fig, height, legend_bottom=False, hovermode="closest")
    fig.update_xaxes(title_text="GPS outage length (s)", type="log", tickvals=[10, 30, 60, 120])
    _log_y(fig, "peak position error during outage (m, log)", float(summary["min"].min()), float(summary["max"].max()))
    return fig


def quality_map(df: pd.DataFrame, value: str, label: str, levels: np.ndarray, height: int = 520,
                ref: pd.DataFrame | None = None, extra: list | None = None, square: bool = False) -> go.Figure:
    """Risk map: points (or grid cells) coloured by risk level 0/1/2 with icon labels in the legend."""
    fig = go.Figure()
    if ref is not None:
        r = _decimate(ref, 2500)
        fig.add_trace(go.Scatter(x=r.x, y=r.y, mode="lines", name="Route", line=dict(color=REF, width=1, dash="dot"),
                                   hoverinfo="skip"))
    names = {0: ("Low risk", GOOD, "●"), 1: ("Elevated risk", WARN, "▲"), 2: ("High risk", CRIT, "■")}
    sym = {0: "circle", 1: "triangle-up", 2: "square"} if not square else {0: "square", 1: "square", 2: "square"}
    xcol, ycol = ("cx", "cy") if square else ("x", "y")
    for lv, (nm, col, ic) in names.items():
        d = df[levels == lv]
        if not len(d):
            continue
        fig.add_trace(go.Scatter(
            x=d[xcol], y=d[ycol], mode="markers", name=f"{ic} {nm}",
            marker=dict(color=col, size=13 if square else 6, symbol=sym[lv], opacity=0.85 if square else 0.8,
                        line=dict(width=0)),
            customdata=d[[value]].to_numpy(),
            hovertemplate=f"{nm}<br>{label}: %{{customdata[0]:.3f}}<br>x %{{x:.0f}} · y %{{y:.0f}}<extra></extra>"))
    for tr in extra or []:
        fig.add_trace(tr)
    _base_layout(fig, height, showlegend=False)
    fig.update_xaxes(title_text="East–west (m)")
    fig.update_yaxes(title_text="North–south (m)")
    _equal_axes(fig)
    return fig


def series_over_time(t, y, label: str, height: int = 220, color: str = SERIES[0], hline=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=1.6), name=label,
                                 hovertemplate=f"t %{{x:.1f}} s<br>{label} %{{y:.2f}}<extra></extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=MUTED, dash="dash", width=1))
    _base_layout(fig, height, legend_bottom=False, showlegend=False)
    fig.update_xaxes(title_text="time since start (s)")
    fig.update_yaxes(title_text=label)
    return fig


def safety_timeline(dec: pd.DataFrame, t0: float, t_now: float | None, height: int = 260) -> go.Figure:
    """Clearance to nearest crew vs time, with buffer, and decision ribbon colour."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec.t - t0, y=dec.dist_to_worker_m, mode="lines", name="Distance to nearest crew zone",
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate="t %{x:.1f} s<br>distance %{y:.1f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=dec.t - t0, y=dec.buffer_m, mode="lines", name="Safety buffer (grows with uncertainty)",
                             line=dict(color=SERIES[1], width=2, dash="dash"),
                             hovertemplate="t %{x:.1f} s<br>buffer %{y:.2f} m<extra></extra>"))
    if t_now is not None:
        fig.add_vline(x=t_now - t0, line=dict(color=INK, width=1.2))
    _base_layout(fig, height, legend_bottom=False, hovermode="x unified")
    fig.update_xaxes(title_text="time since start (s)")
    fig.update_yaxes(title_text="metres", rangemode="tozero")
    return fig


def circle(x: float, y: float, r: float, n: int = 60):
    a = np.linspace(0, 2 * np.pi, n)
    return x + r * np.cos(a), y + r * np.sin(a)

