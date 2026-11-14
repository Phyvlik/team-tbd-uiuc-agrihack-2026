"""Simplified 3D orchard view (Plotly) for the Farmer / Operator dashboard.

Rendered inside Streamlit so it stays frame-synchronised with the time slider,
metrics, checklist and alerts. The tree geometry is illustrative (see
dashboard/orchard.py). Robot size is exaggerated 2× for visibility.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import orchard as orc
from .theme import CRIT, GOOD, INK, REF, SERIES, STATE, WARN

LEVEL_STATE = {0: "TRUSTED", 1: "DEGRADED", 2: "STOP"}
SPRAY_COLOR = {orc.NOT_REACHED: "#2f6b3f", orc.TREATED: SERIES[0], orc.MISSED: "#e8e2d0", orc.DOUBLE: "#d55181",
               orc.WRONG_TREE: SERIES[1]}
DECISION_COLOR = {"CONTINUE": GOOD, "GO": GOOD, "SLOW": WARN, "STOP": CRIT}

# unit low-poly sphere (2 poles + 3 rings × 7) for tree canopies
_RINGS, _SEG = 3, 7


def _unit_sphere():
    verts = [(0, 0, 1)]
    for r in range(1, _RINGS + 1):
        phi = np.pi * r / (_RINGS + 1)
        for s in range(_SEG):
            th = 2 * np.pi * s / _SEG + (0.4 if r % 2 else 0)
            verts.append((np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)))
    verts.append((0, 0, -1))
    v = np.array(verts)
    faces = []
    for s in range(_SEG):
        faces.append((0, 1 + s, 1 + (s + 1) % _SEG))
    for r in range(_RINGS - 1):
        a, b = 1 + r * _SEG, 1 + (r + 1) * _SEG
        for s in range(_SEG):
            s2 = (s + 1) % _SEG
            faces += [(a + s, b + s, b + s2), (a + s, b + s2, a + s2)]
    last, bottom = 1 + (_RINGS - 1) * _SEG, len(v) - 1
    for s in range(_SEG):
        faces.append((bottom, last + (s + 1) % _SEG, last + s))
    return v, np.array(faces)


SPHERE_V, SPHERE_F = _unit_sphere()


def _canopies(trees, status, r=1.5, h=2.3):
    n = len(trees)
    if n == 0:
        return None
    nv = len(SPHERE_V)
    scale = np.array([r, r, r * 0.85])
    V = (SPHERE_V[None] * scale + np.c_[trees.x, trees.y, np.full(n, h)][:, None]).reshape(-1, 3)
    F = (SPHERE_F[None] + (np.arange(n) * nv)[:, None, None]).reshape(-1, 3)
    fc = np.repeat([SPRAY_COLOR[int(s)] for s in status], len(SPHERE_F))
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], facecolor=fc,
                     flatshading=True, lighting=dict(ambient=0.55, diffuse=0.7, specular=0.05, roughness=0.9),
                     lightposition=dict(x=100, y=200, z=400), name="Trees (illustrative)", hoverinfo="skip",
                     showlegend=False)


def _trunks(trees):
    xs, ys, zs = [], [], []
    for x, y in zip(trees.x, trees.y):
        xs += [x, x, None]; ys += [y, y, None]; zs += [0, 1.1, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#5b4636", width=5), hoverinfo="skip",
                        showlegend=False)


def _box(cx, cy, th, L=1.5, W=1.25, H=0.8, z0=0.12):
    c, s = np.cos(th), np.sin(th)
    pts = []
    for dz in (0, H):
        for dx, dy in ((L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2)):
            pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, z0 + dz))
    P = np.array(pts)
    F = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                  (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    return P, F


def _ellipsoid(est_row, height, radius: float | None = None):
    """Covariance 95 % ellipse, or a circle of ``radius`` when the source gives a scalar uncertainty."""
    if radius is not None:
        ax, vecs = np.array([radius, radius]), np.eye(2)
    else:
        cov = np.array([[est_row.var_x, est_row.cov_xy], [est_row.cov_xy, est_row.var_y]])
        vals, vecs = np.linalg.eigh(cov)
        ax = np.sqrt(5.991 * np.maximum(vals, 1e-6))          # 95 % in 2D
    V = SPHERE_V.copy()
    xy = (V[:, :2] * ax) @ vecs.T
    return np.c_[xy[:, 0] + est_row.x_ant, xy[:, 1] + est_row.y_ant, V[:, 2] * height + 0.2], SPHERE_F, ax


def _empty_line():
    return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", hoverinfo="skip", showlegend=False)


def _empty_mesh():
    return go.Mesh3d(x=[0, 0, 0], y=[0, 0, 0], z=[-1, -1, -1], i=[0], j=[1], k=[2], opacity=0, hoverinfo="skip",
                     showlegend=False)


def _ring(x, y, r, z=0.05, n=64):
    a = np.linspace(0, 2 * np.pi, n)
    return x + r * np.cos(a), y + r * np.sin(a), np.full(n, z)


def orchard_scene(streams, res, t_now: float, orchard: orc.Orchard, spray: np.ndarray | None,
                  *, dec=None, zones=None, half: float = 17.0, show_reference: bool = False,
                  height: int = 470, overview: bool = False, uncertainty_radius: float | None = None) -> go.Figure:
    """``uncertainty_radius``: draw the bubble as a circle of this radius (m) instead of the 95 %
    covariance ellipse, for sources that supply a scalar uncertainty (teammate 2*sigma)."""
    est, states = res.estimate, res.states
    m = est.t.to_numpy() <= t_now
    e, lv = est[m], states.level.to_numpy()[m]
    row = e.iloc[-1] if len(e) else est.iloc[0]
    level = int(lv[-1]) if len(lv) else 0
    stt = STATE[LEVEL_STATE[level]]
    cx, cy = float(row.x_ant), float(row.y_ant)
    if overview:
        ref = streams.reference
        x0, x1, y0, y1 = ref.x.min() - 8, ref.x.max() + 8, ref.y.min() - 8, ref.y.max() + 8
    else:
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    inwin = lambda X, Y: (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)  # noqa: E731

    fig = go.Figure()
    # ground + light grid (10 m)
    fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#1b2420", opacity=1.0, hoverinfo="skip", showlegend=False))
    gx, gy, gz = [], [], []
    for v in np.arange(np.ceil(x0 / 10) * 10, x1, 10):
        gx += [v, v, None]; gy += [y0, y1, None]; gz += [0.01, 0.01, None]
    for v in np.arange(np.ceil(y0 / 10) * 10, y1, 10):
        gx += [x0, x1, None]; gy += [v, v, None]; gz += [0.01, 0.01, None]
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="#26332d", width=1),
                               hoverinfo="skip", showlegend=False))
    # trees (spray status colours)
    tr = orchard.trees
    # NOTE: the number and order of traces is kept CONSTANT between frames so the
    # browser updates the WebGL scene in place (a changing trace list forces a full
    # re-plot, which flickers during playback). Empty slots use invisible placeholders.
    sub = tr.iloc[:0] if not len(tr) else tr
    st_sub = np.zeros(0, int)
    if len(tr):
        sel = inwin(tr.x.to_numpy(), tr.y.to_numpy())
        if not overview:
            sel &= np.hypot(tr.x - cx, tr.y - cy).to_numpy() <= half * 1.25
        sub = tr[sel]
        st_sub = (spray[sel] if spray is not None else np.zeros(len(sub), int))
    fig.add_trace(_trunks(sub) if len(sub) else _empty_line())
    fig.add_trace(_canopies(sub, st_sub, r=1.4 if not overview else 1.6) if len(sub) else _empty_mesh())
    # reference path (Engineer only)
    if show_reference:
        r = streams.reference
        rr = r[inwin(r.x.to_numpy(), r.y.to_numpy())].iloc[::3]
        fig.add_trace(go.Scatter3d(x=rr.x, y=rr.y, z=np.full(len(rr), 0.08), mode="lines",
                                   line=dict(color=REF, width=3, dash="dot"), name="True path (RTK, scoring only)",
                                   hoverinfo="skip"))
    else:
        fig.add_trace(_empty_line())
    # trail coloured by navigation state
    step = max(1, len(e) // 2500)
    ee, ll = e.iloc[::step], lv[::step]
    xs, ys = ee.x_ant.to_numpy(), ee.y_ant.to_numpy()
    for lvl in (0, 1, 2):
        s = STATE[LEVEL_STATE[lvl]]
        X, Y = [], []
        ch = np.flatnonzero(np.r_[True, ll[1:] != ll[:-1]])
        for a, b in zip(ch, list(ch[1:]) + [len(ll)]):
            if len(ll) and ll[a] == lvl:
                j = min(b + 1, len(xs))
                X += list(xs[a:j]) + [None]; Y += list(ys[a:j]) + [None]
        fig.add_trace(go.Scatter3d(x=X or [None], y=Y or [None], z=[0.15 if v is not None else None for v in X] or [None],
                                   mode="lines", line=dict(color=s["color"], width=7),
                                   name=f"{s['icon']} Trail · {s['label']}", hoverinfo="skip"))
    # GPS lost / restored markers (one trace: poles + labels)
    mx, my, mz, mt = [], [], [], []
    for a, b in res.outages_abs:
        for tt, lab in ((a, "GPS lost"), (b, "GPS restored")):
            if tt <= t_now:
                k = min(int(np.searchsorted(est.t.to_numpy(), tt)), len(est) - 1)
                px, py = est.x_ant.iloc[k], est.y_ant.iloc[k]
                if overview or inwin(np.array([px]), np.array([py]))[0]:
                    mx += [px, px, None]
                    my += [py, py, None]
                    mz += [0, 3.2, None]
                    mt += ["", lab, ""]
    fig.add_trace(go.Scatter3d(x=mx or [None], y=my or [None], z=mz or [None], mode="lines+text", text=mt or [""],
                               textposition="top center", textfont=dict(color=WARN, size=12),
                               line=dict(color=WARN, width=4), hoverinfo="skip", showlegend=False))
    # crews (rings + poles, labels: fixed 2 traces) + safety buffer (fixed 1 trace)
    zx, zy, zz, px_, py_, pz_, lab = [], [], [], [], [], [], []
    for k, z in enumerate(zones or []):
        if not overview and not inwin(np.array([z.x]), np.array([z.y]))[0]:
            continue
        rx, ry, rz = _ring(z.x, z.y, z.radius, 0.06)
        zx += list(rx) + [None, z.x, z.x, None]
        zy += list(ry) + [None, z.y, z.y, None]
        zz += list(rz) + [None, 0, 1.6, None]
        px_.append(z.x)
        py_.append(z.y)
        pz_.append(1.75)
        lab.append(f"Crew {k + 1} (simulated)")
    fig.add_trace(go.Scatter3d(x=zx or [None], y=zy or [None], z=zz or [None], mode="lines",
                               line=dict(color=INK, width=4), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=px_ or [None], y=py_ or [None], z=pz_ or [None], mode="markers+text",
                               text=lab or [""], textposition="top center", textfont=dict(color=INK, size=12),
                               marker=dict(symbol="diamond", size=7, color=INK), hoverinfo="skip", showlegend=False))
    if dec is not None:
        i = min(int(np.searchsorted(dec.t.to_numpy(), t_now, side="right")) - 1, len(dec) - 1)
        d = dec.iloc[max(i, 0)]
        bx, by, bz = _ring(cx, cy, float(d.buffer_m), 0.07)
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                                   line=dict(color=DECISION_COLOR[d.decision], width=5, dash="dash"),
                                   name=f"Safety buffer {d.buffer_m:.1f} m", hoverinfo="skip"))
    else:
        fig.add_trace(_empty_line())
    # uncertainty bubble
    V, F, ax = _ellipsoid(row, height=min(1.2 + 0.4 * float(row.r95), 3.0), radius=uncertainty_radius)
    fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color=stt["color"],
                            opacity=0.22, flatshading=False, hoverinfo="skip", name="Position uncertainty",
                            showlegend=False))
    ex_, ey_, ez_ = _ring(0, 0, 1)
    vals = np.c_[ex_, ey_]
    if uncertainty_radius is not None:
        ring = vals * uncertainty_radius
    else:
        cov = np.array([[row.var_x, row.cov_xy], [row.cov_xy, row.var_y]])
        w, vec = np.linalg.eigh(cov)
        ring = (vals * np.sqrt(5.991 * np.maximum(w, 1e-6))) @ vec.T
    fig.add_trace(go.Scatter3d(x=ring[:, 0] + cx, y=ring[:, 1] + cy, z=np.full(len(ring), 0.1), mode="lines",
                               line=dict(color=stt["color"], width=4), hoverinfo="skip", showlegend=False))
    # robot body (2× scale) + status beacon + heading arrow
    P, Fb = _box(cx, cy, float(row.theta))
    fig.add_trace(go.Mesh3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], i=Fb[:, 0], j=Fb[:, 1], k=Fb[:, 2], color="#d9dee3",
                            flatshading=True, hoverinfo="skip", showlegend=False))
    hx, hy = np.cos(row.theta), np.sin(row.theta)
    fig.add_trace(go.Scatter3d(x=[cx, cx + 1.8 * hx], y=[cy, cy + 1.8 * hy], z=[0.9, 0.9], mode="lines",
                               line=dict(color=INK, width=6), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[1.25], mode="markers",
                               marker=dict(size=11, color=stt["color"], line=dict(color="#0b0f14", width=2)),
                               name="Robot status light", hovertemplate=f"Robot · {stt['label']}<extra></extra>"))

    span_x, span_y = x1 - x0, y1 - y0
    zmax = 8.0
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        uirevision="orchard3d-overview" if overview else "orchard3d",
        scene=dict(
            xaxis=dict(range=[x0, x1], visible=False), yaxis=dict(range=[y0, y1], visible=False),
            zaxis=dict(range=[0, zmax], visible=False), bgcolor="#0e141b",
            aspectmode="manual",
            aspectratio=dict(x=1, y=span_y / span_x, z=zmax / span_x * (2.2 if not overview else 4.0)),
            camera=dict(eye=dict(x=-0.42, y=-0.82, z=0.58) if not overview else dict(x=-0.4, y=-0.9, z=1.1),
                        center=dict(x=0, y=0, z=-0.12)),
            dragmode="orbit"),
    )
    return fig
