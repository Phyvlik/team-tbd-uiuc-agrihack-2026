"""Smooth Mission Control replay: a client-side player (dashboard/replay_player.js).

Root cause of the old jerky playback: the replay advanced by a FIXED step on every
Streamlit fragment rerun (``0.5 s * speed`` every 0.5 s, i.e. 2 s jumps at 4x, ~2 frames
per second, longer jumps whenever a rerun ran late) and each rerun rebuilt the whole
Plotly figure server-side. In the 3D orchard the tree mesh was re-selected from a
moving window every frame, so the WebGL scene changed shape and redrew from scratch.

Now the browser owns the clock while playing (replay time = anchor + speed x elapsed
wall time) and draws every display frame on a canvas (3D orchard or 2D map, ~1 ms per
frame), interpolating the robot between the 10 Hz samples. Plotly was measured at 28 to
50 ms per animated frame in 3D and ~34 ms in 2D, too slow for smooth motion. The player
reports the time back ~4 times a second, so the Streamlit side panel stays on the same
clock. Nothing here changes any estimate: the payload holds the same arrays the old
figures drew, sent once per scenario.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import orchard as orc
from . import scene3d
from .theme import CRIT, GOOD, GPS_FIX, INK, REF, STATE, WARN

APP_DIR = Path(__file__).resolve().parents[1]
JS = (Path(__file__).with_name("replay_player.js")).read_text(encoding="utf-8")
CSS = ".orchard-replay { width: 100%; }"
LEVEL_COLORS = [STATE["TRUSTED"]["color"], STATE["DEGRADED"]["color"], STATE["STOP"]["color"]]
DECISION_LEVEL = {"CONTINUE": 0, "GO": 0, "SLOW": 1, "STOP": 2}
SPEEDS = (1, 2, 4, 10)


_component = None


def _player():
    global _component
    if _component is None:
        _component = st.components.v2.component("orchard_replay", js=JS, css=CSS, isolate_styles=False)
    return _component


def _r(a, nd=3) -> list:
    return np.round(np.asarray(a, float), nd).tolist()


def _colors() -> dict:
    return {"state": LEVEL_COLORS, "ref": REF, "warn": WARN, "ink": INK, "gps": GPS_FIX, "good": GOOD, "crit": CRIT}


def _scene_id(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _bounds(x, y, pad=8.0) -> list:
    return [float(np.min(x) - pad), float(np.max(x) + pad), float(np.min(y) - pad), float(np.max(y) + pad)]


def _tree_colors() -> dict:
    return {str(k): v for k, v in scene3d.SPRAY_COLOR.items()}


def spray_changes(n_trees: int, events: pd.DataFrame, t_pass: np.ndarray, t_offset: float = 0.0) -> list:
    """Per-tree status changes [(tree, t, code)] that reproduce orchard.spray_status at every time:
    MISSED once the true path passed with no hit yet, TREATED after one hit, DOUBLE after two."""
    per: dict[int, list] = {}
    for j in range(n_trees):
        if np.isfinite(t_pass[j]):
            per.setdefault(j, []).append((float(t_pass[j]), "pass"))
    ev = events[events.hit >= 0] if len(events) else events
    for t, h in zip(ev.t if len(ev) else [], ev.hit if len(ev) else []):
        per.setdefault(int(h), []).append((float(t), "hit"))
    out = []
    for j, evs in per.items():
        hits, passed, cur = 0, False, orc.NOT_REACHED
        for t, kind in sorted(evs, key=lambda e: (e[0], e[1] != "pass")):
            hits += kind == "hit"
            passed |= kind == "pass"
            new = orc.DOUBLE if hits >= 2 else orc.TREATED if hits == 1 else orc.MISSED if passed else orc.NOT_REACHED
            if new != cur:
                out.append((j, float(t - t_offset), int(new)))
                cur = new
    return sorted(out, key=lambda c: c[1])


# ------------------------------------------------------------------ payloads --

@st.cache_data(show_spinner=False, max_entries=24)
def team_scene(seq: str, mode: str, eng: bool, show_final: bool) -> dict:
    from . import team_slip as ts
    from . import team_ui as tu
    run, dec = tu.load(seq), tu.decisions(seq, mode)
    d = run.dash
    code = {"correct": orc.TREATED, "missed": orc.MISSED, "double": orc.DOUBLE, "wrong_tree": orc.WRONG_TREE}
    if show_final:
        changes = [(k, 0.0, code[s]) for k, s in enumerate(run.trees.status)]
    elif run.tree_reveal == "progressive":
        changes = sorted(((k, round(float(tp), 3), code[s]) for k, (tp, s) in
                          enumerate(zip(run.tree_t_pass, run.trees.status))), key=lambda c: c[1])
    else:
        changes = []
    markers = []
    for a, b in run.outages:
        for tm, lab in ((a, "GPS lost"), (b, "GPS restored")):
            if tm <= run.t_max:
                k = run.index_at(tm)
                markers.append({"t": round(tm, 3), "x": float(d.x_est.iloc[k]), "y": float(d.y_est.iloc[k]), "label": lab})
    two_s = 2 * d.sigma
    return {
        "id": _scene_id("team", seq, mode, eng, show_final),
        "t": _r(d.t), "x": _r(d.x_est), "y": _r(d.y_est), "th": _r(d.yaw_est, 4), "lvl": d.state.astype(int).tolist(),
        "ua": _r(two_s), "ub": _r(two_s), "ang": [0.0] * len(d),
        "ring": {"r": _r(dec.bubble_m), "lvl": dec.level.astype(int).tolist()},
        "ref": {"x": _r(d.x_true[::2]), "y": _r(d.y_true[::2])} if eng else None,
        "trees": {"x": _r(run.trees.x), "y": _r(run.trees.y), "changes": changes, "colors": _tree_colors()},
        "workers": [{"x": float(w.x), "y": float(w.y), "r": ts.WORKER_RADIUS_M, "t0": float(w.t_outage_start),
                     "t1": float(w.t + ts.WORKER_VISIBLE_AFTER_S), "label": f"Worker {k + 1} (simulated)"}
                    for k, w in enumerate(run.crew.itertuples())],
        "markers": markers, "fixes": None,
        "bounds": _bounds(np.r_[d.x_true, d.x_est], np.r_[d.y_true, d.y_est]),
        "t_max": float(run.t_max), "colors": _colors(),
    }


@st.cache_data(show_spinner=False, max_entries=24)
def ekf_scene(key: str, method: str, outages: tuple, jumps: tuple, trusted: float, stop: float,
              crew: tuple | None, eng: bool) -> dict:
    """crew = (min_buffer, k, slow, offset) when the simulated crew is shown, else None."""
    from . import backend as be
    from . import scenario as scn
    sens = be.METHODS[method]
    streams = be.load_streams(key)
    t0 = streams.time_span()[0]
    res = be.experiment(key, sens, outages, jumps, trusted, stop)
    est, lv = res.estimate, res.states.level.to_numpy().astype(int)
    t = est.t.to_numpy() - t0
    a_ = est.var_x.to_numpy(); b_ = est.var_y.to_numpy(); c_ = est.cov_xy.to_numpy()
    tr = 0.5 * (a_ + b_); dd = np.sqrt(0.25 * (a_ - b_) ** 2 + c_ ** 2)
    l1, l2 = np.maximum(tr + dd, 1e-12), np.maximum(tr - dd, 1e-12)
    ang = 0.5 * np.arctan2(2 * c_, a_ - b_)                      # major-axis angle
    orchard = scn.orchard(key)
    spray_ev, t_pass = scn.spray(key, sens, outages, jumps, trusted, stop)
    ring, workers = None, []
    if crew is not None:
        dec, zones = scn.safety(key, sens, outages, jumps, trusted, stop, *crew)
        ring = {"r": _r(dec.buffer_m), "lvl": [DECISION_LEVEL[x] for x in dec.decision]}
        workers = [{"x": float(z.x), "y": float(z.y), "r": float(z.radius), "t0": None, "t1": None,
                    "label": f"Crew {k + 1} (simulated)"} for k, z in enumerate(zones)]
    markers = []
    for a, b in res.outages_abs:
        for tm, lab in ((a, "GPS lost"), (b, "GPS restored")):
            k = min(int(np.searchsorted(est.t.to_numpy(), tm)), len(est) - 1)
            markers.append({"t": round(tm - t0, 3), "x": float(est.x_ant.iloc[k]), "y": float(est.y_ant.iloc[k]),
                            "label": lab})
    fixes = None
    if res.gps_log is not None and len(res.gps_log):
        g = res.gps_log[res.gps_log.available].iloc[::2]
        fixes = {"t": _r(g.t - t0), "x": _r(g.x), "y": _r(g.y)}
    ref = streams.reference
    return {
        "id": _scene_id("ekf", key, method, outages, jumps, trusted, stop, crew, eng),
        "t": _r(t), "x": _r(est.x_ant), "y": _r(est.y_ant), "th": _r(est.theta, 4), "lvl": lv.tolist(),
        "ua": _r(np.sqrt(5.991 * l1)), "ub": _r(np.sqrt(5.991 * l2)), "ang": _r(ang, 4),
        "ring": ring,
        "ref": {"x": _r(ref.x[::3]), "y": _r(ref.y[::3])} if eng else None,
        "trees": {"x": _r(orchard.trees.x), "y": _r(orchard.trees.y),
                  "changes": spray_changes(len(orchard.trees), spray_ev, t_pass, t0), "colors": _tree_colors()},
        "workers": workers, "markers": markers, "fixes": fixes,
        "bounds": _bounds(ref.x, ref.y),
        "t_max": float(t[-1]), "colors": _colors(),
    }


# ------------------------------------------------------------- clock sync --

def _keys(page: str) -> tuple[str, str, str, str]:
    return f"_rp_js_t_{page}", f"_rp_rev_{page}", f"_rp_seek_{page}", f"_rp_seekid_{page}"


def detect_seek(page: str, tkey: str) -> bool:
    """A time set by the server (slider, jump, event chip, reset, demo) becomes a seek for the player."""
    ss = st.session_state
    k_js, _, k_seek, k_id = _keys(page)
    cur = float(ss[tkey])
    if ss.get(k_js) is None or abs(cur - ss[k_js]) > 1e-6:
        ss[k_seek], ss[k_id], ss[k_js] = cur, ss.get(k_id, 0) + 1, cur
        return True
    return False


def apply_clock(page: str, tkey: str, clock, seeked: bool = False) -> bool:
    """Take the time the player reported (once per report). Returns True if it was applied.
    In a run where the server seeked, a report that was in flight is consumed but not applied."""
    ss = st.session_state
    k_js, k_rev, _, _ = _keys(page)
    rev = clock.get("rev") if isinstance(clock, dict) else None
    if rev is None or (ss.get(k_rev) is not None and rev <= ss[k_rev]):
        return False      # nothing new, or an older report whose rerun ran late: never step backwards
    ss[k_rev] = rev
    if seeked:
        return False
    ss[tkey] = ss[k_js] = float(clock["t"])
    if not clock.get("playing") and ss.get("playing") == page:
        ss.playing = None
        ss.demo_end = None
    return True


def player(page: str, tkey: str, scene: dict, *, view: str, follow: bool, height: int = 470) -> None:
    """Mount (or update) the player. The clock it reports is applied to ``tkey`` before returning,
    so everything rendered after this call uses the player's time."""
    ss = st.session_state
    seeked = detect_seek(page, tkey)
    _, _, k_seek, k_id = _keys(page)
    end = scene["t_max"] if ss.get("demo_end") is None else min(float(ss.demo_end), scene["t_max"])
    ctl = {"playing": ss.get("playing") == page, "speed": float(ss.get("speed", 1)), "seek_t": float(ss[k_seek]),
           "seek_id": int(ss[k_id]), "end": float(end), "view": "3d" if view == "3d" else "2d", "follow": bool(follow),
           "height": int(height)}
    res = _player()(key=f"replay_{page}", data={"scene": scene, "ctl": ctl}, height=int(height),
                    on_clock_change=lambda: None)
    apply_clock(page, tkey, res.get("clock") if hasattr(res, "get") else getattr(res, "clock", None), seeked)
