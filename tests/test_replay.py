"""Playback regression tests: client-side replay player (dashboard/replay.py, replay_player.js)."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dashboard import orchard as orc
from dashboard import replay
from dashboard import team_slip as ts

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "dashboard" / "replay_player.js"


class _State(dict):
    """Minimal stand-in for st.session_state (attribute and item access)."""
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture
def ss(monkeypatch):
    state = _State(playing="team", demo_end=None, team_t=50.0)
    monkeypatch.setattr(replay, "st", SimpleNamespace(session_state=state))
    return state


# ------------------------------------------------------------- JS helpers --

def _node(expr: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    code = f"import(process.argv[1]).then(m => {{ console.log(JSON.stringify({expr})); }})"
    out = subprocess.run([node, "-e", code, JS.as_uri()], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_js_clock_is_time_based():
    r = _node("[m.simTime(50, 1000, 3000, 4, 1e9), m.simTime(50, 1000, 3000, 1, 1e9), "
              "m.simTime(50, 1000, 3000, 10, 60), m.simTime(50, 1000, 1000, 10, 60)]")
    assert r == [58, 52, 60, 50]          # 2 s of wall time at 4x = 8 s; clamped at the end


def test_js_interpolation_and_lookup():
    r = _node("[m.lastIndex([0,0.1,0.2,0.3], 0.25), m.lastIndex([0,0.1,0.2], -1), m.lastIndex([0,0.1,0.2], 9),"
              " m.interp([0,1,2,3],[0,0.1,0.2,0.3],0.25), m.interpAngle([3.1,-3.1],[0,1],0.5)]")
    assert r[:3] == [2, 0, 2] and abs(r[3] - 2.5) < 1e-9
    assert abs(abs(r[4]) - np.pi) < 0.01   # wraps through +-pi, not through 0


def test_js_tree_status_and_workers():
    r = _node("[m.treeStatus([[1,5,1],[0,6,2],[1,9,3]],3,7), "
              "m.visibleWorkers([{t0:60,t1:99,label:'a'},{t0:null,t1:null,label:'b'}], 50).map(w=>w.label)]")
    assert r == [[2, 1, 0], ["b"]]


# ---------------------------------------------------------- Python payloads --

def test_spray_changes_reproduce_spray_status():
    """The per-tree change list sent to the player gives the same status as orchard.spray_status."""
    rng = np.random.default_rng(1)
    n = 40
    t_pass = np.where(rng.random(n) < 0.8, rng.uniform(0, 100, n), np.nan)
    ev = pd.DataFrame({"t": rng.uniform(0, 100, 60), "intended": rng.integers(0, n, 60),
                       "hit": np.where(rng.random(60) < 0.85, rng.integers(0, n, 60), -1)})
    changes = replay.spray_changes(n, ev, t_pass)
    for t in np.linspace(-1, 101, 120):
        status = np.zeros(n, int)
        for k, tc, code in changes:
            if tc <= t:
                status[k] = code
        assert (status == orc.spray_status(n, ev, t_pass, t)).all(), t


def test_team_scene_carries_csv_values_unchanged():
    run = ts.load_run("02_13B_Jackal")
    s = replay.team_scene("02_13B_Jackal", ts.SAFE, False, False)
    d = run.dash
    assert len(s["t"]) == len(d) and s["t_max"] == pytest.approx(run.t_max)
    assert np.allclose(s["x"], d.x_est, atol=1e-3) and np.allclose(s["y"], d.y_est, atol=1e-3)
    assert np.allclose(s["ua"], 2 * d.sigma, atol=1e-3) and s["lvl"] == d.state.tolist()
    assert np.allclose(s["ring"]["r"], d.bubble_m, atol=1e-3)
    assert s["ref"] is None and len(s["workers"]) == len(run.crew)
    # progressive spray reveal = closest-approach time of the reference path
    assert len(s["trees"]["changes"]) == len(run.trees)
    assert replay.team_scene("07_14B_Jackal", ts.SAFE, False, False)["trees"]["changes"] == []


# --------------------------------------------------------------- clock sync --

def test_server_time_change_becomes_a_seek(ss):
    assert replay.detect_seek("team", "team_t") is True          # first render
    assert replay.detect_seek("team", "team_t") is False
    ss["team_t"] = 84.0                                           # slider, jump, chip, reset
    assert replay.detect_seek("team", "team_t") is True and ss["_rp_seek_team"] == 84.0


def test_player_reports_are_applied_in_order_only(ss):
    replay.detect_seek("team", "team_t")
    assert replay.apply_clock("team", "team_t", {"t": 55.0, "rev": 100, "playing": True})
    assert ss["team_t"] == 55.0 and replay.detect_seek("team", "team_t") is False
    assert not replay.apply_clock("team", "team_t", {"t": 52.0, "rev": 90, "playing": True})   # late, older
    assert ss["team_t"] == 55.0
    assert not replay.apply_clock("team", "team_t", {"t": 56.0, "rev": 100, "playing": True})  # repeated
    assert replay.apply_clock("team", "team_t", {"t": 100.0, "rev": 120, "playing": False})    # ended
    assert ss["team_t"] == 100.0 and ss.playing is None


def test_report_in_flight_during_a_seek_is_ignored(ss):
    replay.detect_seek("team", "team_t")
    ss["team_t"] = 20.0
    seeked = replay.detect_seek("team", "team_t")
    assert not replay.apply_clock("team", "team_t", {"t": 61.0, "rev": 200, "playing": True}, seeked)
    assert ss["team_t"] == 20.0


def test_server_clock_uses_elapsed_wall_time(monkeypatch):
    from dashboard import components as ui
    state = _State(playing="ws", demo_end=None, ws_t=10.0)
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state=state))
    clock = iter([100.0, 100.5, 101.0, 109.0])
    monkeypatch.setattr(ui.time, "monotonic", lambda: next(clock))
    ui.advance("ws", "ws_t", 300.0, 4)            # first tick: no elapsed time yet
    assert state["ws_t"] == 10.0
    ui.advance("ws", "ws_t", 300.0, 4)            # 0.5 s at 4x
    assert state["ws_t"] == pytest.approx(12.0)
    ui.advance("ws", "ws_t", 300.0, 4)
    assert state["ws_t"] == pytest.approx(14.0)
    ui.advance("ws", "ws_t", 300.0, 4)            # an 8 s stall is capped, never a leap
    assert state["ws_t"] == pytest.approx(14.0 + 4 * ui.MAX_STEP_S)


# --------------------------------------------------------------------- app --

def test_progress_bar_replaces_slider_while_playing():
    from streamlit.testing.v1 import AppTest
    app = str(ROOT / "app.py")
    for src in ("Backend EKF (live)", ts.SOURCE_LABEL):
        at = AppTest.from_file(app, default_timeout=300)
        at.session_state["results_src"] = src
        at.run()
        assert not at.exception and len(at.slider) >= 1
        at.session_state["playing"] = "team" if src == ts.SOURCE_LABEL else "mc"
        at.run()
        assert not at.exception
        assert not any("Replay time" in (s.label or "") for s in at.slider)
        assert any("pause to scrub" in str(p.proto.text) for p in at.get("progress"))
