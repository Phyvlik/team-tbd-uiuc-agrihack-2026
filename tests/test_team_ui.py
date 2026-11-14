"""Team Slip-Learning Results in the Streamlit app (headless AppTest)."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import team_slip as ts

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["views/mission_control.py", "views/gps_simulator.py", "views/sensor_comparison.py", "views/risk_map.py",
         "views/worker_safety.py"]


def _open(page="views/mission_control.py", mode="Operator", **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["results_src"] = ts.SOURCE_LABEL
    at.session_state["view_mode"] = mode
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page != "views/mission_control.py":
        at.switch_page(page).run()
    return at


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.fixture(scope="module")
def run02():
    return ts.load_run("02_13B_Jackal")


@pytest.mark.parametrize("mode", ["Operator", "Engineer"])
@pytest.mark.parametrize("page", PAGES)
def test_team_mode_pages_load(page, mode):
    at = _open(page, mode)
    assert not at.exception, [e.value for e in at.exception]
    assert "eammate" not in _md(at)               # user-facing text names the method, not a person


def test_ekf_remains_default():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert at.session_state["results_src"] == "Backend EKF (live)" and not at.exception
    assert "Team Slip-Learning" not in at.markdown[0].value


def test_demo_button_starts_02_safe(run02):
    at = _open(team_seq="05_13D_Jackal", team_mode=ts.PRODUCTIVE, team_t=500.0)
    at.button(key="team_demo").click().run()
    ss = at.session_state
    assert (ss["team_seq"], ss["team_mode"], ss["playing"]) == ("02_13B_Jackal", ts.SAFE, "team")
    assert run02.outages[0][0] - 10.5 < ss["team_t"] < run02.outages[0][0] + 5   # starts before the first outage


def test_jump_to_next_outage_and_worker(run02):
    at = _open(team_t=0.0)
    at.button(key="jt_team_0").click().run()             # Next GPS outage (3 s lead)
    assert at.session_state["team_t"] == pytest.approx(run02.outages[0][0] - 3.0)
    at.button(key="jt_team_1").click().run()             # Next worker
    assert at.session_state["team_t"] == pytest.approx(run02.crew.t_outage_start.iloc[0])


def test_replay_is_synchronised_at_known_times(run02):
    w = run02.crew.iloc[0]
    at = _open(team_t=float(w.t))                          # worker standing on the path
    md = _md(at)
    assert "h-state'>STOP" in md and "Nearest worker" in md and "Simulated worker inside the robot" in md
    assert "Lost" in md and "outage running for" in md
    at = _open(team_t=20.0)                                # before any outage
    md = _md(at)
    assert "h-state'>GO" in md and "None active" in md and "Available" in md and "All systems normal" in md


def test_safe_and_productive_modes_differ(run02):
    safe, prod = ts.decision_series(run02, ts.SAFE), ts.decision_series(run02, ts.PRODUCTIVE)
    i = int(np.flatnonzero((safe.action == "STOP").to_numpy() & (prod.action == "SLOW").to_numpy())[0])
    t = float(run02.dash.t.iloc[i])
    assert "h-state'>STOP" in _md(_open(team_t=t, team_mode=ts.SAFE))
    assert "h-state'>SLOW" in _md(_open(team_t=t, team_mode=ts.PRODUCTIVE))


def test_final_only_sequence_hides_spray_until_requested():
    at = _open(team_seq="07_14B_Jackal", team_t=500.0)
    assert "Final only" in _md(at)
    at = _open(team_seq="07_14B_Jackal", team_t=500.0, team_show_final=True)
    assert "final result" in _md(at)


def test_engineer_reports_unavailable_fields():
    at = _open(mode="Engineer")
    tables = [df.value for df in at.dataframe]
    assert any("Not available in this export" in set(t.get("status", [])) for t in tables)
