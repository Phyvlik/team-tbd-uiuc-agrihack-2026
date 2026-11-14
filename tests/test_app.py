"""Headless UI tests (Streamlit AppTest). Run from this folder:

    ..\\UIUC-AgriRobotics-2026\\.venv\\Scripts\\python -m pytest -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = {
    "views/mission_control.py": "Mission Control",
    "views/gps_simulator.py": "GPS Failure Simulator",
    "views/sensor_comparison.py": "Sensor Cost Comparison",
    "views/risk_map.py": "Orchard Risk Map",
    "views/worker_safety.py": "Worker Safety",
}


def _open(page: str, source: str = "06_14B_Jackal", mode: str = "Operator", **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["source"] = source
    at.session_state["view_mode"] = mode
    at.session_state["cmp_basis"] = "Team batch"  # avoid the (button-gated) live evaluation in tests
    for k, v in state.items():
        at.session_state[k] = v
    at.run()                      # first run registers st.navigation pages
    at.switch_page(page).run()
    return at


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("mode", ["Operator", "Engineer"])
@pytest.mark.parametrize("source", ["06_14B_Jackal", "SYNTHETIC", "01_13B_Jackal"])
@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_renders(page, source, mode):
    at = _open(page, source, mode)
    assert not at.exception, [e.value for e in at.exception]
    assert PAGES[page] in _text(at)


def test_synthetic_is_always_labelled():
    for page in PAGES:
        at = _open(page, "SYNTHETIC")
        assert "SYNTHETIC" in _text(at).upper(), page


def test_missing_streams_show_not_evaluated():
    # 01_13B has wheel odometry only: risk map and C2/C3 must say Not evaluated, never fake values
    at = _open("views/risk_map.py", "01_13B_Jackal")
    assert "Not evaluated" in _text(at)
    at = _open("views/sensor_comparison.py", "01_13B_Jackal")
    assert "Not evaluated" in _text(at)


def test_demo_button_sets_shared_scenario():
    at = _open("views/mission_control.py", outage_dur=0)
    demo = [b for b in at.button if b.label == "▶ Run GPS Failure Demo"]
    assert demo, "Demo button missing"
    demo[0].click().run()
    assert not at.exception
    assert at.session_state["outage_dur"] == 60
    assert at.session_state["outage_start"] == 100.0
    assert at.session_state["method"] == "odom+imu+gps"
    assert 85.0 <= at.session_state["mc_t"] < 100.0 and at.session_state["playing"] == "mc"
    reset = [b for b in at.button if b.label == "⟲ Reset"]
    reset[0].click().run()
    assert at.session_state["playing"] is None and at.session_state["mc_t"] == 85.0


def test_operator_panel_has_metrics_checklist_and_bilingual_alerts():
    at = _open("views/mission_control.py", mc_t=112.0, outage_dur=60, outage_start=100.0, method="odom+imu+gps")
    txt = _text(at)
    for needle in ("What the robot can do now", "GPS status", "Position uncertainty", "Robot speed",
                   "Stop recommended", "GPS no disponible"):
        assert needle in txt, needle
    assert len(at.get("plotly_chart")) >= 1, "3D orchard figure missing"


def test_all_three_visualisations_render():
    for viz in ("3D orchard", "2D map", "Rerun 3D"):
        at = _open("views/mission_control.py", viz=viz)
        assert not at.exception, (viz, [e.value for e in at.exception])


def test_outage_setting_is_shared_between_pages():
    at = _open("views/gps_simulator.py", outage_dur=45)
    assert not at.exception
    at.switch_page("views/mission_control.py").run()   # custom duration must be accepted there too
    assert not at.exception
    assert at.session_state["outage_dur"] == 45


def test_engineer_thresholds_change_monitor():
    at = _open("views/mission_control.py", mode="Engineer", trusted_r95=0.5, stop_r95=2.0)
    assert not at.exception
    assert "trusted ≤ 0.5 m · stop > 2.0 m" in " ".join(str(d.value) for d in at.dataframe)


def test_ui_only_writes_to_its_own_cache():
    """The UI imports the team backend read-only. The only file writes are the live
    C1–C4 tables (.cache/) and Rerun recordings (recordings/), both in this app folder."""
    from dashboard import backend as be
    app_dir = Path(APP).parent
    assert be.BACKEND_DIR != app_dir and be.CACHE_DIR.parent == app_dir
    writers = ("to_csv(", "write_text(", "write_bytes(", ".mkdir(")
    for f in list((app_dir / "dashboard").glob("*.py")) + list((app_dir / "views").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        hits = [w for w in writers if w in src]
        if f.name == "backend.py":
            assert src.count("to_csv(") == 2 and "CACHE_DIR.mkdir(" in src
        elif f.name == "rerun_export.py":          # .rrd recordings go to this app's recordings/ folder
            from dashboard import rerun_export as rx
            assert rx.RECORDINGS_DIR.parent == app_dir and hits == [".mkdir("]
        else:
            assert not hits, (f.name, hits)


def test_adapter_roundtrip(tmp_path):
    """A teammate CSV (epoch ns timestamps, covariance) is normalised and validated against the sequence."""
    import numpy as np
    import pandas as pd
    from dashboard import adapter as ad
    from dashboard import backend as be
    res = be.experiment("06_14B_Jackal", ("odom", "gps"), ((100.0, 30.0),))
    e = res.estimate.iloc[::5]
    df = pd.DataFrame({"t": (e.t * 1e9).astype("int64"), "est_x": e.x_ant, "est_y": e.y_ant,
                       "var_x": e.var_x, "var_y": e.var_y, "cov_xy": e.cov_xy})
    f = tmp_path / "06_14B_Jackal__teammate-odom-gps.csv"
    df.to_csv(f, index=False)
    b = ad.from_csv(f)
    assert b.sequence == "06_14B_Jackal" and b.method == "teammate-odom-gps"
    assert abs(b.t[0]) < 1.0                      # converted from epoch ns to seconds since start
    assert b.r95 is not None and b.gyro_bias is None and b.speed is None   # nothing inferred
    assert np.nanmax(b.err) < 5 and any("time unit: ns" in c for c in b.checks)


def test_rerun_export_builds(tmp_path):
    from dashboard import rerun_export as rx
    ok, _ = rx.available()
    if not ok:
        pytest.skip("rerun-sdk not installed in this interpreter")
    p = rx.build_recording("SYNTHETIC", "odom+imu+gps", ((100.0, 30.0),), (), 0.3, 1.0, kind="engineer",
                           out=tmp_path / "x.rrd")
    assert p.exists() and p.stat().st_size > 100_000


def test_uncertainty_is_flagged_uncalibrated_everywhere_it_is_shown():
    at = _open("views/mission_control.py", mc_t=112.0)
    assert "Uncalibrated" in _text(at)
    at = _open("views/mission_control.py", mode="Engineer")
    txt = _text(at)
    assert "Uncertainty status: Uncalibrated" in txt and "nothing is rescaled" in txt


def test_proxy_gps_is_labelled_simulated():
    at = _open("views/mission_control.py", source="01_13B_Jackal")
    assert "simulated proxy" in _text(at)
    at = _open("views/sensor_comparison.py", source="01_13B_Jackal")
    assert "GPS PROXY (SIMULATED)" in _text(at)


def test_estimator_covariance_is_not_modified():
    """The UI must display the EKF's r95 exactly as computed (no silent inflation)."""
    import numpy as np
    from dashboard import backend as be
    from src.outage_simulator import OutageScenario
    from src.pipeline import run_experiment
    s = be.load_streams("06_14B_Jackal")
    direct = run_experiment(s, ("odom", "imu", "gps"), OutageScenario([(100.0, 60.0)]))
    via_ui = be.experiment("06_14B_Jackal", ("odom", "imu", "gps"), ((100.0, 60.0),))
    assert np.array_equal(direct.estimate.r95.to_numpy(), via_ui.estimate.r95.to_numpy())
