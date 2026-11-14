"""Rerun (.rrd) recordings of a run — Engineer replay and a simplified operator scene.

Everything logged comes from the backend pipeline (run_experiment) on the
selected data source, plus the illustrative orchard / simulated spray and crews
from this app. Timeline: ``time`` = seconds since the sequence start, the same
clock as the Streamlit slider.

Rerun is an independent viewer: the .rrd is a snapshot of one scenario. The
Streamlit slider does NOT drive the Rerun cursor (see README: "Rerun").

Needs ``rerun-sdk`` (installed in this app's .venv). Import is lazy so the rest
of the dashboard works without it.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend as be
from . import ops
from . import orchard as orc

RECORDINGS_DIR = be.APP_DIR / "recordings"
STATE_RGB = {0: (12, 163, 12), 1: (250, 178, 25), 2: (208, 59, 59)}
METHOD_RGB = {"C1": (57, 135, 229), "C2": (217, 89, 38), "C3": (25, 158, 112), "imported": (201, 133, 0),
              "B1": (138, 148, 142)}
SPRAY_RGB = {orc.NOT_REACHED: (47, 107, 63), orc.TREATED: (57, 135, 229), orc.MISSED: (232, 226, 208),
             orc.DOUBLE: (213, 81, 129), orc.WRONG_TREE: (217, 89, 38)}


def available() -> tuple[bool, str]:
    try:
        import rerun as rr  # noqa: F401
        return True, rr.__version__
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _quat_z(theta):
    return np.c_[np.zeros_like(theta), np.zeros_like(theta), np.sin(theta / 2), np.cos(theta / 2)]


def build_recording(key: str, method: str, outages: tuple, jumps: tuple, trusted: float, stop: float,
                    *, kind: str = "engineer", zones=None, safety_cfg=None, imported: list | None = None,
                    out: Path | None = None) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    streams = be.load_streams(key)
    t0, _ = streams.time_span()
    sens = be.METHODS[method]
    res = be.experiment(key, sens, outages, jumps, trusted, stop)
    est, states = res.estimate, res.states
    trel = est.t.to_numpy() - t0
    lv = states.level.to_numpy()
    orchard = orc.build_orchard(streams)
    spray_ev, t_pass = orc.simulate_spray(orchard.trees, est, streams.reference)
    dec = None
    if zones:
        dec = be.evaluate_safety(est, states, zones, safety_cfg or be.SafetyConfig())

    RECORDINGS_DIR.mkdir(exist_ok=True)
    tag = f"{key}_{method.replace('+', '-')}_{'_'.join(f'{a:.0f}s-{d:.0f}s' for a, d in outages) or 'no-outage'}"
    out = out or RECORDINGS_DIR / f"{kind}_{tag}.rrd"
    rec = rr.RecordingStream(f"orchard_robot_{kind}", recording_id=f"{kind}-{tag}")
    TL = "time"

    def scal(path, tt, vals):
        vals = np.asarray(vals, float)
        m = np.isfinite(vals)
        if m.any():
            rec.send_columns(path, indexes=[rr.TimeColumn(TL, duration=np.asarray(tt)[m])],
                             columns=rr.Scalars.columns(scalars=vals[m]))

    # ---------------- static scene ----------------
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    ref = streams.reference
    rec.log("world/info", rr.TextDocument(
        f"Source: {be.source_label(key)}\nMethod: {be.METHOD_LABEL[method]}\nGPS input: {res.gps_source}\n"
        f"Scenario: {', '.join(f'{d:.0f} s outage at {a:.0f} s' for a, d in outages) or 'no outage'} (SIMULATED)\n"
        f"Trees: {orchard.note}\nSpray: SIMULATED application scenario (rules in dashboard/orchard.py)\n"
        f"Crews: {'SIMULATED' if zones else 'none'}\nReference = post-processed RTK, scoring only.\n"
        f"Uncertainty (r95, bubble): {be.CALIBRATION_STATUS.upper()} — see {be.CALIBRATION_DOC}.\n"
        f"{be.reference_caveat(key, res.gps_source)}"), static=True)
    if len(orchard.trees):
        T = orchard.trees
        rec.log("world/orchard/trunks", rr.LineStrips3D([[(x, y, 0), (x, y, 1.1)] for x, y in zip(T.x, T.y)],
                                                         colors=[(91, 70, 54)], radii=0.08), static=True)
        centers = np.c_[T.x, T.y, np.full(len(T), 2.3)]
        rec.set_time(TL, duration=0.0)
        rec.log("world/orchard/canopy", rr.Ellipsoids3D(centers=centers, half_sizes=[(1.4, 1.4, 1.2)] * len(T),
                                                        colors=[SPRAY_RGB[0]] * len(T), fill_mode="solid"))
        if len(spray_ev):
            for tt in sorted(set(spray_ev.t.round(1))):
                st = orc.spray_status(len(T), spray_ev, t_pass, tt + 1e-6)
                rec.set_time(TL, duration=float(tt - t0))
                rec.log("world/orchard/canopy", rr.Ellipsoids3D.from_fields(colors=[SPRAY_RGB[int(s)] for s in st]))
    if kind == "engineer":
        rec.log("world/reference_rtk", rr.LineStrips3D([np.c_[ref.x, ref.y, np.full(len(ref), 0.05)][::2]],
                                                         colors=[(195, 194, 183)], radii=0.04), static=True)
    if zones:
        for k, z in enumerate(zones):
            a = np.linspace(0, 2 * np.pi, 48)
            rec.log(f"world/crews/crew_{k + 1}", rr.Points3D([(z.x, z.y, 1.6)], radii=0.25, colors=[(238, 242, 239)],
                                                              labels=[f"Crew {k + 1} (simulated)"], show_labels=True),
                    rr.LineStrips3D([np.c_[z.x + z.radius * np.cos(a), z.y + z.radius * np.sin(a), np.full(48, 0.05)]],
                                    colors=[(238, 242, 239)], radii=0.03), static=True)

    # ---------------- robot over time ----------------
    step = max(1, int(round(0.2 / np.median(np.diff(est.t)))))  # ~5 Hz for the 3D scene
    idx = np.arange(0, len(est), step)
    E = est.iloc[idx]
    tt = trel[idx]
    th = E.theta.to_numpy()
    cols = [STATE_RGB[int(v)] for v in lv[idx]]
    rec.log("world/robot/body", rr.Boxes3D(half_sizes=[(0.5, 0.43, 0.28)], centers=[(0, 0, 0.4)],
                                           colors=[(217, 222, 227)], fill_mode="solid"), static=True)
    rec.log("world/robot/heading", rr.Arrows3D(vectors=[(1.8, 0, 0)], origins=[(0, 0, 0.9)], colors=[(238, 242, 239)],
                                               radii=0.06), static=True)
    rec.send_columns("world/robot", indexes=[rr.TimeColumn(TL, duration=tt)],
                     columns=rr.Transform3D.columns(translation=np.c_[E.x_ant, E.y_ant, np.zeros(len(E))],
                                                    quaternion=_quat_z(th)))
    rec.send_columns("world/robot/status_light", indexes=[rr.TimeColumn(TL, duration=tt)],
                     columns=rr.Points3D.columns(positions=np.tile([0, 0, 1.1], (len(E), 1)), colors=cols,
                                                 radii=np.full(len(E), 0.22)))
    # trail = path so far, split into runs coloured by navigation state (re-logged at ~1 Hz)
    XY = np.c_[E.x_ant, E.y_ant, np.full(len(E), 0.15)]
    L = lv[idx]
    for k in range(1, len(E), 5):
        strips, scol = [], []
        ch = np.flatnonzero(np.r_[True, L[1:k + 1] != L[:k]])
        for a_, b_ in zip(ch, list(ch[1:]) + [k + 1]):
            strips.append(XY[a_:min(b_ + 1, k + 1)])
            scol.append(STATE_RGB[int(L[a_])])
        rec.set_time(TL, duration=float(tt[k]))
        rec.log("world/trail", rr.LineStrips3D(strips, colors=scol, radii=0.14))
    # 95 % uncertainty bubble (axes from covariance eigen-decomposition)
    hs, q = [], []
    for _, r in E.iterrows():
        w, v = np.linalg.eigh(np.array([[r.var_x, r.cov_xy], [r.cov_xy, r.var_y]]))
        a = np.sqrt(5.991 * np.maximum(w, 1e-6))
        ang = np.arctan2(v[1, 1], v[0, 1])
        hs.append((a[1], a[0], min(1.2 + 0.4 * r.r95, 3.0)))
        q.append(ang)
    rgba = [(*c, 70) for c in cols]
    rec.send_columns("world/uncertainty_95", indexes=[rr.TimeColumn(TL, duration=tt)],
                     columns=rr.Ellipsoids3D.columns(centers=np.c_[E.x_ant, E.y_ant, np.full(len(E), 0.2)],
                                                     half_sizes=np.array(hs), quaternions=_quat_z(np.array(q)),
                                                     colors=rgba))
    if dec is not None:
        D = dec.iloc[idx]
        a = np.linspace(0, 2 * np.pi, 40)
        dc = {"CONTINUE": STATE_RGB[0], "SLOW": STATE_RGB[1], "STOP": STATE_RGB[2]}
        for (x, y, b, d), ti in zip(D[["x", "y", "buffer_m", "decision"]].to_numpy(), tt):
            rec.set_time(TL, duration=float(ti))
            rec.log("world/safety_buffer", rr.LineStrips3D([np.c_[x + b * np.cos(a), y + b * np.sin(a),
                                                                  np.full(40, 0.06)]], colors=[dc[d]], radii=0.05))
    # GPS fixes received / rejected
    if res.gps_log is not None and len(res.gps_log):
        g = res.gps_log[res.gps_log.available].iloc[::2]
        rec.send_columns("world/gps/fixes_received", indexes=[rr.TimeColumn(TL, duration=g.t.to_numpy() - t0)],
                         columns=rr.Points3D.columns(positions=np.c_[g.x, g.y, np.full(len(g), 0.3)],
                                                     colors=[(57, 135, 229)] * len(g), radii=np.full(len(g), 0.06)))
    # outage markers
    for a_, b_ in res.outages_abs:
        for tm, lab in ((a_, "GPS lost"), (b_, "GPS restored")):
            k = min(int(np.searchsorted(est.t.to_numpy(), tm)), len(est) - 1)
            rec.set_time(TL, duration=float(tm - t0))
            rec.log(f"world/events/{lab.replace(' ', '_').lower()}",
                    rr.Points3D([(est.x_ant.iloc[k], est.y_ant.iloc[k], 3.0)], colors=[(250, 178, 25)], radii=0.3,
                                labels=[f"{lab} (simulated)"], show_labels=True))

    # ---------------- engineer time series ----------------
    if kind == "engineer":
        runs = {}
        cfg_of = {"odom+gps": "C1", "odom+imu+gps": "C2", "odom+imu+lidar+gps": "C3"}
        for m in be.methods_for(streams):
            runs[f"{cfg_of[m]}_{m.replace('+', '_')}"] = (cfg_of[m], be.experiment(key, be.METHODS[m], outages, jumps,
                                                                                    trusted, stop))
        runs["B1_wheel_odometry_only_no_gps"] = ("B1", be.experiment(key, ("odom",), (), (), trusted, stop))
        for name, (cid, r) in runs.items():
            e = r.estimate.iloc[::step]
            if cid != "B1":   # B1 drifts tens of metres; kept in the plots (hidden by default), not the 3D scene
                rec.log(f"world/methods/{name}", rr.LineStrips3D([np.c_[e.x_ant, e.y_ant, np.full(len(e), 0.1)]],
                                                                 colors=[METHOD_RGB[cid]], radii=0.05), static=True)
            rec.log(f"plots/error_vs_rtk/{name}", rr.SeriesLines(colors=[METHOD_RGB[cid]], names=[name],
                                                                 visible_series=[cid != "B1"]), static=True)
            scal(f"plots/error_vs_rtk/{name}", r.error.t - t0, r.error.err)
            if cid != "B1":
                rec.log(f"plots/uncertainty_r95/{name}", rr.SeriesLines(colors=[METHOD_RGB[cid]], names=[name]),
                        static=True)
                scal(f"plots/uncertainty_r95/{name}", r.estimate.t - t0, r.estimate.r95)
        for b in imported or []:
            nm = f"imported_{b.method}"
            rec.log(f"world/methods/{nm}", rr.LineStrips3D([np.c_[b.est_x, b.est_y, np.full(len(b.t), 0.1)]],
                                                           colors=[METHOD_RGB['imported']], radii=0.05), static=True)
            if b.err is not None:
                scal(f"plots/error_vs_rtk/{nm}", b.t, b.err)
        for path, val in (("plots/thresholds/target_0.3m", trusted), ("plots/thresholds/stop_limit", stop)):
            scal(path, [trel[0], trel[-1]], [val, val])
        scal("plots/integrity/state_level_0_trusted_1_degraded_2_stop", trel, lv)
        scal("plots/gps/time_since_fix_s", trel, est.time_since_gps)
        gs = ops.gps_status_series(est)
        scal("plots/gps/available_1_degraded_0.5_lost_0", trel,
             np.select([gs == ops.GPS_AVAILABLE, gs == ops.GPS_DEGRADED], [1.0, 0.5], 0.0))
        if streams.gps_quality is not None:
            gq = streams.gps_quality
            for c in ("num_sat", "num_sats", "h_sigma_m", "cn0_mean"):
                if c in gq:
                    scal(f"plots/gps/quality_{c}", gq.t - t0, gq[c])
        ev = res.gps_events
        if ev is not None and len(ev):
            scal("plots/gps/nis_accepted", ev.t[ev.accepted] - t0, ev.nis[ev.accepted])
            rej = ev[~ev.accepted]
            if len(rej):
                rec.log("plots/gps/nis_rejected", rr.SeriesPoints(colors=[(208, 59, 59)], markers=["cross"],
                                                                  marker_sizes=[6]), static=True)
                scal("plots/gps/nis_rejected", rej.t - t0, rej.nis)
                for _, r in rej.iterrows():
                    rec.set_time(TL, duration=float(r.t - t0))
                    rec.log("events", rr.TextLog(f"GPS fix rejected by consistency gate (NIS {r.nis:.1f}, "
                                                 f"innovation {r.innovation_m:.2f} m)", level=rr.TextLogLevel.WARN))
        if "imu" in sens:
            scal("plots/fusion/gyro_bias_rad_s", trel, est.b_g)
        else:
            rec.log("plots/fusion/gyro_bias_rad_s", rr.TextDocument("Not evaluated: IMU not fused in this method."),
                    static=True)
        scal("plots/fusion/odom_speed_scale_k", trel, est.k_odom)
        if streams.imu is not None and len(streams.imu):
            o = streams.odom
            ti = o.t.to_numpy()
            wz = np.interp(ti, streams.imu.t, streams.imu.wz)
            dis = pd.Series(o.w.to_numpy() - wz).rolling(50, center=True, min_periods=5).mean().to_numpy()
            scal("plots/fusion/sensor_disagreement_yaw_rate_odom_minus_imu", (ti - t0)[::10], dis[::10])
        # event log
        for _, r in ops.events(est, states, gs, dec, res.outages_abs, t0).iterrows():
            rec.set_time(TL, duration=float(r.t_rel))
            lvl = {"TRUSTED": rr.TextLogLevel.INFO, "DEGRADED": rr.TextLogLevel.WARN,
                   "STOP": rr.TextLogLevel.ERROR}[r.state]
            rec.log("events", rr.TextLog(f"{r.label}", level=lvl))
        rec.log("plots/lidar", rr.TextDocument("Data unavailable: no LiDAR point clouds or lidar_odom.csv extracted."),
                static=True)

    # ---------------- blueprint ----------------
    eye = rrb.EyeControls3D(tracking_entity="world/robot")
    cxr, cyr = float(ref.x.mean()), float(ref.y.mean())
    span = float(max(ref.x.max() - ref.x.min(), ref.y.max() - ref.y.min(), 30))
    eng_eye = rrb.EyeControls3D(position=(cxr - 0.5 * span, cyr - 0.9 * span, 0.7 * span), look_target=(cxr, cyr, 0),
                                eye_up=(0, 0, 1))
    if kind == "operator":
        bp = rrb.Blueprint(
            rrb.Spatial3DView(origin="world", name="Orchard (illustrative trees · simulated outage, spray, crew)",
                              contents=["+ world/**", "- world/methods/**", "- world/reference_rtk",
                                        "- world/gps/**", "- world/info"],
                              background=(14, 20, 27), line_grid=rrb.LineGrid3D(visible=False),
                              eye_controls=eye),
            rrb.TimePanel(state="collapsed", timeline=TL), collapse_panels=True)
    else:
        bp = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(origin="world", name="Orchard 3D · trajectories (trees illustrative, spray simulated)",
                                      background=(14, 20, 27),
                                      contents=["+ world/**", "- world/info"],
                                      eye_controls=eng_eye),
                    rrb.TextLogView(origin="events", name="Event log"), row_shares=[3, 1]),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="plots/error_vs_rtk", name="Position error vs RTK (m)"),
                    rrb.TimeSeriesView(origin="plots/uncertainty_r95", name="Estimated 95 % uncertainty (m) · UNCALIBRATED"),
                    rrb.TimeSeriesView(origin="plots/gps", name="GPS status · quality · NIS"),
                    rrb.TimeSeriesView(origin="plots/fusion", name="Fusion: gyro bias · odom scale · disagreement"),
                    rrb.TimeSeriesView(origin="plots/integrity", name="Integrity state (0 trusted · 1 degraded · 2 stop)"),
                ),
                column_shares=[3, 2]),
            rrb.TimePanel(timeline=TL, state="expanded"), rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"))
    rec.save(str(out), default_blueprint=bp)
    rec.flush()
    return out


# ------------------------------------------------------------ web viewer ---

def _free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def rerun_exe() -> str | None:
    exe = Path(sys.executable).with_name("rerun.exe" if sys.platform == "win32" else "rerun")
    return str(exe) if exe.exists() else shutil.which("rerun")


_PROCS: dict[str, subprocess.Popen] = {}


def serve_web(rrd: Path, web_port: int = 9090, grpc_port: int = 9877, slot: str = "eng") -> str:
    """Host Rerun's own web viewer for ``rrd`` (separate local process). Returns the viewer URL."""
    exe = rerun_exe()
    if exe is None:
        raise RuntimeError("rerun executable not found in this environment")
    old = _PROCS.get(slot)
    if old is not None and old.poll() is None:
        old.terminate()
        try:
            old.wait(5)
        except Exception:
            old.kill()
    for _ in range(30):
        if _free(web_port) and _free(grpc_port):
            break
        time.sleep(0.2)
    _PROCS[slot] = subprocess.Popen([exe, "--serve-web", "--web-viewer-port", str(web_port), "--port", str(grpc_port),
                                     "--hide-welcome-screen", str(rrd)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(50):
        if not _free(web_port):
            break
        time.sleep(0.2)
    return f"http://localhost:{web_port}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{grpc_port}%2Fproxy"


def open_native(rrd: Path) -> None:
    exe = rerun_exe()
    if exe is None:
        raise RuntimeError("rerun executable not found")
    subprocess.Popen([exe, str(rrd)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
