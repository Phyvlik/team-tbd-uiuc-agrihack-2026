import inspect

import numpy as np
import pandas as pd
import pytest

from src.coordinate_utils import (apply_se2, body_to_map, fit_se2, heading_from_track, interp_series, latlon_to_enu,
                                  quat_to_yaw, wrap_angle)
from src.data_loader import RAW_DIR, SensorStreams, load_gt_csv
from src.evaluation import error_series, outage_metrics, summary_metrics
from src.localization import InitialPose, dead_reckon, initial_pose_from_track
from src.outage_simulator import OutageScenario, apply_scenario, gps_proxy_from_reference, standard_outage_suite
from src.pipeline import run_experiment
from src.reliability_monitor import MonitorConfig, classify
from src.sensor_comparison import compare_configs
from src.sensor_fusion import EKFConfig, run_ekf
from src.synthetic_data import make_synthetic_sequence
from src.teammate_import import normalize, verify
from src.worker_safety import STOP, SafetyConfig, WorkerZone, evaluate_safety


@pytest.fixture(scope="module")
def syn():
    return make_synthetic_sequence(seed=1, n_rows=3, row_length=40.0)


# --------------------------------------------------------------- coordinates ---
def test_wrap_angle():
    assert np.allclose(wrap_angle([2 * np.pi + 0.5, -2 * np.pi - 0.5, 7 * np.pi / 2, 0.1]),
                       [0.5, -0.5, -np.pi / 2, 0.1])
    assert np.all(np.abs(wrap_angle(np.linspace(-20, 20, 401))) <= np.pi)


def test_quat_to_yaw():
    yaw = 0.7
    assert np.isclose(quat_to_yaw(0, 0, np.sin(yaw / 2), np.cos(yaw / 2)), yaw)


def test_latlon_to_enu_small_offsets():
    lat0, lon0 = 33.97, -117.33  # Riverside, CA
    e, n, u = latlon_to_enu(lat0 + 1e-4, lon0 + 1e-4, 0.0, lat0, lon0, 0.0)
    assert np.isclose(n, 11.09, atol=0.05)                     # ~111 km/deg latitude
    assert np.isclose(e, 11.09 * np.cos(np.radians(lat0)), atol=0.05)
    assert abs(u) < 0.01


def test_fit_se2_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(-10, 10, (50, 2))
    dst = apply_se2(src, 0.6, [3.0, -2.0])
    th, t = fit_se2(src, dst)
    assert np.isclose(th, 0.6) and np.allclose(t, [3.0, -2.0])


def test_lever_arm_rotates_with_heading():
    x, y = body_to_map(0.0, 0.0, np.pi / 2, (-0.18, 0.0))
    assert np.isclose(x, 0.0, atol=1e-12) and np.isclose(y, -0.18)


def test_heading_from_track():
    th, i = heading_from_track([0, 1, 2, 3], [0, 1, 2, 3], 2.0)
    assert np.isclose(th, np.pi / 4) and i == 2


# --------------------------------------------------------------- timestamps ---
def test_interp_rejects_unordered_and_no_extrapolation():
    with pytest.raises(ValueError):
        interp_series([0.5], [0, 2, 1], [0, 1, 2])
    out = interp_series([-1, 0.5, 5], [0, 1], [0, 10])
    assert np.isnan(out[0]) and np.isclose(out[1], 5) and np.isnan(out[2])


def test_streams_validate_detects_bad_time():
    s = SensorStreams("x", "synthetic", odom=pd.DataFrame({"t": [0, 2, 1], "v": [1, 1, 1], "w": [0, 0, 0]}))
    assert any("strictly increasing" in p for p in s.validate())


def test_synthetic_streams_valid(syn):
    assert syn.validate() == []


@pytest.mark.skipif(not (RAW_DIR / "06_14B_Jackal" / "gt.csv").exists(), reason="CitrusFarm gt.csv not downloaded")
def test_real_gt_csv_format():
    gt = load_gt_csv(RAW_DIR / "06_14B_Jackal" / "gt.csv")
    assert list(gt.columns) == ["t", "x", "y", "z"]
    assert np.all(np.diff(gt.t) > 0)
    assert gt.x.iloc[0] == 0.0 and gt.y.iloc[0] == 0.0
    assert 9.0 < 1 / np.median(np.diff(gt.t)) < 11.0


# --------------------------------------------------------- motion integration ---
def test_dead_reckon_straight_and_circle():
    t = np.arange(0, 10.001, 0.01)
    tr = dead_reckon(t, np.ones_like(t), np.zeros_like(t), theta0=np.pi / 2)
    assert np.isclose(tr.x.iloc[-1], 0, atol=1e-9) and np.isclose(tr.y.iloc[-1], 10, atol=1e-6)
    w = 2 * np.pi / 10  # one full circle in 10 s
    tr = dead_reckon(t, np.ones_like(t), np.full_like(t, w))
    assert np.hypot(tr.x.iloc[-1], tr.y.iloc[-1]) < 1e-3  # back to start


def test_initial_pose_accounts_for_lever_arm():
    track = pd.DataFrame({"t": np.arange(5.0), "x": np.arange(5.0), "y": np.zeros(5)})
    p = initial_pose_from_track(track, lever_arm=(-0.18, 0.0), align_distance=2.0)
    assert np.isclose(p.theta, 0) and np.isclose(p.x, 0.18)


# ------------------------------------------------------------------ outages ---
def test_outage_removes_exactly_the_window():
    ref = pd.DataFrame({"t": np.arange(0, 100, 0.1), "x": np.zeros(1000), "y": np.zeros(1000)})
    gps = gps_proxy_from_reference(ref, sigma=0.0)
    used, log = apply_scenario(gps, OutageScenario(outages=[(20.0, 30.0)]), t0=0.0)
    assert not ((used.t >= 20.0) & (used.t < 50.0)).any()
    assert (~log.available).sum() == 300
    assert len(used) == 700


def test_standard_suite_same_windows():
    a, b = standard_outage_suite(600), standard_outage_suite(600)
    assert [s.outages for s in a] == [s.outages for s in b]
    assert {s.outages[0][1] for s in a} == {10, 30, 60, 120}


def test_ekf_never_sees_reference():
    params = inspect.signature(run_ekf).parameters
    assert "reference" not in params and "ref" not in params


def test_ekf_ignores_withheld_gps(syn):
    r_full = run_experiment(syn, ("odom", "gps"), OutageScenario())
    r_out = run_experiment(syn, ("odom", "gps"), OutageScenario(outages=[(40.0, 30.0)]))
    t0 = syn.time_span()[0]
    m = (r_out.estimate.t > t0 + 41) & (r_out.estimate.t < t0 + 69)
    assert not r_out.estimate.gps_used[m].any()
    assert r_out.estimate.time_since_gps[m].max() > 25


# ------------------------------------------------------------ error metrics ---
def test_error_series_and_summary():
    ref = pd.DataFrame({"t": np.arange(10.0), "x": np.arange(10.0), "y": np.zeros(10)})
    est = pd.DataFrame({"t": np.arange(10.0), "x": np.arange(10.0), "y": np.full(10, 0.2)})
    e = error_series(est, ref)
    m = summary_metrics(e, 0.3)
    assert np.allclose(e.err, 0.2) and np.isclose(m["rmse_m"], 0.2) and m["pct_time_within_target"] == 100


def test_outage_metrics_time_to_exceed():
    t = np.arange(0, 60, 0.1)
    err = pd.DataFrame({"t": t, "err": np.where((t >= 10) & (t < 40), 0.05 * (t - 10), 0.01)})
    om = outage_metrics(err, [(10.0, 40.0)])
    assert np.isclose(om.time_to_exceed_s[0], 6.1, atol=0.11)
    assert om.recovery_s[0] < 0.2


# ----------------------------------------------------- uncertainty behaviour ---
def test_uncertainty_grows_during_outage_and_shrinks_after(syn):
    r = run_experiment(syn, ("odom", "imu", "gps"), OutageScenario(outages=[(40.0, 30.0)]))
    t0 = syn.time_span()[0]
    e = r.estimate
    at = lambda s: float(e.r95[np.searchsorted(e.t, t0 + s)])
    assert at(39) < at(55) < at(69)
    assert at(75) < 0.5 * at(69)


def test_monitor_states_escalate_in_outage(syn):
    r = run_experiment(syn, ("odom", "gps"), OutageScenario(outages=[(40.0, 60.0)]))
    t0 = syn.time_span()[0]
    st = r.states
    assert st.level[(st.t > t0 + 20) & (st.t < t0 + 38)].max() <= 1
    assert st.level[(st.t > t0 + 95) & (st.t < t0 + 100)].max() == 2


def test_monitor_trusted_with_small_uncertainty():
    est = pd.DataFrame({"t": np.arange(0, 10, 0.1), "r95": 0.05, "time_since_gps": 0.1, "gps_rejected": False})
    assert (classify(est, MonitorConfig()).state == "TRUSTED").all()


# ------------------------------------------------------- sensor comparisons ---
def test_compare_configs_ranks_fusion_first(syn):
    overall, otab = compare_configs(syn, outage_durations=(30,), n_starts=1)
    o = overall.set_index("config")
    assert o.loc["C4", "mean_m"] < o.loc["C2", "mean_m"] < o.loc["C1", "mean_m"]
    assert set(otab.config) == {"C1", "C2", "C3"}


def test_missing_stream_is_not_run():
    s = make_synthetic_sequence(seed=2, n_rows=2, row_length=30.0)
    s.lidar_odom = None
    overall, _ = compare_configs(s, outage_durations=(10,), n_starts=1)
    o = overall.set_index("config")
    assert o.loc["C3", "status"].startswith("NOT RUN")
    assert "lidar" not in o.loc["C4", "sensors_used"]


# ----------------------------------------------------------- worker safety ---
def _est(r95):
    return pd.DataFrame({"t": [0.0], "x_ant": [0.0], "y_ant": [0.0], "r95": [r95]})


def test_buffer_grows_with_uncertainty_and_triggers_stop():
    zones = [WorkerZone("W1", 4.0, 0.0, radius=1.0)]   # zone edge 3 m away
    cfg = SafetyConfig(min_buffer_m=2.0, k_uncertainty=1.0, slow_margin_m=3.0, use_monitor_state=False)
    calm = evaluate_safety(_est(0.1), None, zones, cfg).iloc[0]
    tense = evaluate_safety(_est(1.5), None, zones, cfg).iloc[0]
    assert np.isclose(calm.buffer_m, 2.1) and np.isclose(tense.buffer_m, 3.5)
    assert calm.decision == "SLOW" and tense.decision == STOP


def test_monitor_stop_forces_stop_far_from_workers():
    zones = [WorkerZone("W1", 100.0, 0.0)]
    states = pd.DataFrame({"level": [2]})
    assert evaluate_safety(_est(0.1), states, zones, SafetyConfig()).decision.iloc[0] == STOP


# ---------------------------------------------------------- teammate import ---
def test_import_aliases_units_and_frame_diagnosis():
    t = np.arange(0, 100.0, 0.1)
    rx, ry = t, np.zeros_like(t)
    th = np.radians(30)  # estimate rotated 30 deg: a frame error, not drift
    df = pd.DataFrame({"time_ns": (1.6e9 + t) * 1e9, "gt_x": rx * 1000, "gt_y": ry * 1000,
                       "odom_x": (np.cos(th) * rx) * 1000, "odom_y": (np.sin(th) * rx) * 1000})
    norm, notes = normalize(df, length_unit="mm")
    assert any("ns" in n for n in notes)
    assert np.isclose(norm.reference_x.iloc[-1], 99.9)
    rep = verify(norm)
    assert np.isclose(abs(rep["rigid_fit_rotation_deg"]), 30, atol=0.5)
    assert any("rigid rotation" in f for f in rep["findings"])
