"""Rerun engineer replay vs Streamlit operator view: same coordinates, units and clock.

Both are built from the same backend run. These tests read the .rrd back and compare it to
the exact values the operator 3D scene draws, so a frame/unit/timestamp drift between the
two viewers would fail here.
"""
from pathlib import Path

import numpy as np
import pytest

from dashboard import backend as be
from dashboard import scene3d
from dashboard import scenario as scn

KEY, METHOD, OUT = "06_14B_Jackal", "odom+imu+gps", ((100.0, 60.0),)
SENS = be.METHODS[METHOD]


def _chunks(path: Path, entity: str):
    from rerun.chunk import RrdReader
    rows = []
    for c in RrdReader(str(path)).stream():
        if c.entity_path == entity and not c.is_static:
            rows.append(c.to_record_batch().to_pydict())
    assert rows, f"{entity} not found in {path.name}"
    out = {}
    for r in rows:
        for k, v in r.items():
            out.setdefault(k, []).extend(v)
    t = np.array([x.total_seconds() for x in out["time"]])
    order = np.argsort(t)
    return t[order], {k: [v[i] for i in order] for k, v in out.items() if k != "time"}


@pytest.fixture(scope="module")
def rrd(tmp_path_factory):
    pytest.importorskip("rerun")
    from dashboard import rerun_export as rx
    zones = scn.demo_crew(KEY, OUT)
    return rx.build_recording(KEY, METHOD, OUT, (), 0.3, 1.0, kind="engineer", zones=zones,
                              out=tmp_path_factory.mktemp("rrd") / "eng.rrd")


@pytest.fixture(scope="module")
def run():
    s = be.load_streams(KEY)
    return s, s.time_span()[0], be.experiment(KEY, SENS, OUT, (), 0.3, 1.0)


def test_robot_pose_same_frame_units_and_clock(rrd, run):
    s, t0, res = run
    t, cols = _chunks(rrd, "/world/robot")
    xyz = np.array([v[0] for v in cols["Transform3D:translation"]])
    est = res.estimate
    ex = np.interp(t, est.t - t0, est.x_ant)
    ey = np.interp(t, est.t - t0, est.y_ant)
    assert np.max(np.hypot(xyz[:, 0] - ex, xyz[:, 1] - ey)) < 2e-3        # metres (float32 storage)
    assert abs(t[0] - (est.t.iloc[0] - t0)) < 1e-6                        # same seconds-since-start clock
    # the operator 3D scene draws the robot status light at the same point at the same replay time
    for tt in (50.0, 120.0, 200.0):
        fig = scene3d.orchard_scene(s, res, t0 + tt, scn.orchard(KEY), None)
        light = [tr for tr in fig.data if tr.name == "Robot status light"][0]
        k = int(np.searchsorted(t, tt, side="right") - 1)                  # rrd is decimated to ~5 Hz
        i = be.at_time(est, t0 + tt)
        assert abs(light.x[0] - est.x_ant.iloc[i]) < 1e-9 and abs(light.y[0] - est.y_ant.iloc[i]) < 1e-9
        assert np.hypot(light.x[0] - xyz[k, 0], light.y[0] - xyz[k, 1]) < 0.3   # ≤ one 0.2 s step at ~1.2 m/s


def test_error_and_uncertainty_series_match(rrd, run):
    s, t0, res = run
    t, cols = _chunks(rrd, "/plots/error_vs_rtk/C2_odom_imu_gps")
    v = np.array([x[0] for x in cols["Scalars:scalars"]])
    assert np.allclose(v, np.interp(t, res.error.t - t0, res.error.err), atol=1e-5)   # ns time quantisation
    t, cols = _chunks(rrd, "/plots/uncertainty_r95/C2_odom_imu_gps")
    v = np.array([x[0] for x in cols["Scalars:scalars"]])
    assert np.allclose(v, np.interp(t, res.estimate.t - t0, res.estimate.r95), rtol=1e-6, atol=1e-9)


def test_outage_events_on_same_clock(rrd, run):
    _, t0, res = run
    t, cols = _chunks(rrd, "/events")
    texts = [str(x[0]) if isinstance(x, list) else str(x) for x in cols["TextLog:text"]]
    lost = [tt for tt, x in zip(t, texts) if "GPS lost" in x]
    restored = [tt for tt, x in zip(t, texts) if "GPS restored" in x]
    assert lost and abs(lost[0] - 100.0) < 1e-6 and restored and abs(restored[0] - 160.0) < 1e-6


def test_reference_is_scoring_only_and_same_frame(rrd, run):
    s, t0, res = run
    from rerun.chunk import RrdReader
    pts = None
    for c in RrdReader(str(rrd)).stream():
        if c.entity_path == "/world/reference_rtk":
            pts = np.array(c.to_record_batch().to_pydict()["LineStrips3D:strips"][0][0])
    assert pts is not None
    assert np.allclose(pts[:, 0], s.reference.x.to_numpy()[::2], atol=1e-3)
    assert np.allclose(pts[:, 1], s.reference.y.to_numpy()[::2], atol=1e-3)
