"""SP2 — multimodal sensor fusion with an Extended Kalman Filter.

State (base_link in the map frame)::

    s = [x, y, theta, v, w, b_g, k]
        x, y   position (m)            theta  heading (rad)
        v      forward speed (m/s)     w      yaw rate (rad/s)
        b_g    IMU gyro z bias (rad/s) k      wheel-odometry speed scale
                                              (v_wheel = k * v_true)

Process model (constant velocity / turn rate, random-walk accelerations)::

    x' = x + v cos(theta) dt      v'  = v   + noise
    y' = y + v sin(theta) dt      w'  = w   + noise
    theta' = theta + w dt         b_g' = b_g + noise,  k' = k + noise

Measurement models — each sensor is a genuine EKF update with its own noise:

    wheel odometry  z = [k v, w]                    (non-linear in k, v)
    IMU gyro        z = w + b_g                     (linear)
    LiDAR odometry  z = [v, w]                      (linear)
    GPS position    z = p + R(theta) @ lever_arm    (non-linear, antenna point)

GPS updates are gated with the Normalised Innovation Squared (NIS) test; a
rejected fix is logged as a sensor-disagreement event (used by SP4).

The filter receives only sensor streams. The evaluation reference is not an
argument of :func:`run_ekf`, so ground truth cannot leak into the estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .localization import InitialPose

CHI2_2DOF = {0.95: 5.991, 0.99: 9.210, 0.999: 13.816}

ALL_SENSORS = ("odom", "imu", "lidar", "gps")


@dataclass
class EKFConfig:
    sensors: tuple = ("odom", "imu", "gps")
    # process noise spectral densities
    q_xy: float = 0.02 ** 2      # m^2/s   (unmodelled lateral slip etc.)
    q_theta: float = 0.02 ** 2   # rad^2/s  (tuned on 02_13B, checked on held-out 06_14B)
    q_v: float = 0.5 ** 2        # (m/s^2)^2 * s
    q_w: float = 1.0 ** 2        # (rad/s^2)^2 * s
    q_bg: float = 1e-5 ** 2      # gyro bias random walk
    q_k: float = 1e-5 ** 2       # odom scale random walk
    odom_scale_init: float = 1.0   # prior on k (e.g. calibrated on a training sequence)
    odom_scale_sigma: float = 0.15  # prior 1-sigma on k
    estimate_odom_scale: bool = True
    # measurement noise (1-sigma)
    odom_sigma_v: float = 0.05
    odom_sigma_w: float = 0.20   # skid-steer wheel yaw rate is poor (tuned on 02_13B)
    imu_sigma_w: float = 0.01
    lidar_sigma_v: float = 0.05
    lidar_sigma_w: float = 0.02
    gps_sigma_floor: float = 0.02  # min 1-sigma used for GPS (m)
    gps_sigma_scale: float = 1.0   # inflate reported GPS sigma
    gps_gate: float = CHI2_2DOF[0.999]  # NIS rejection threshold
    gps_max_consecutive_reject: int = 10  # then accept (avoid permanent lock-out)
    lever_arm: tuple = (0.0, 0.0)  # GPS antenna in body frame
    output_dt: float = 0.1
    init_sigma_xy: float = 0.05
    init_sigma_theta: float = math.radians(2.0)

    def label(self) -> str:
        return "+".join(s for s in ALL_SENSORS if s in self.sensors)


@dataclass
class EKFResult:
    estimate: pd.DataFrame
    events: pd.DataFrame
    config: EKFConfig
    counts: dict = field(default_factory=dict)


def _merge_events(streams: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, kind, row_index) sorted by time. kind: 0 odom,1 imu,2 lidar,3 gps."""
    ts, kinds, idxs = [], [], []
    for k, name in enumerate(["odom", "imu", "lidar", "gps"]):
        df = streams.get(name)
        if df is None or len(df) == 0:
            continue
        t = df["t"].to_numpy(float)
        ts.append(t); kinds.append(np.full(len(t), k)); idxs.append(np.arange(len(t)))
    t = np.concatenate(ts); kind = np.concatenate(kinds); idx = np.concatenate(idxs)
    order = np.lexsort((kind, t))
    return t[order], kind[order], idx[order]


def run_ekf(init: InitialPose, odom: pd.DataFrame | None = None, imu: pd.DataFrame | None = None,
            lidar_odom: pd.DataFrame | None = None, gps: pd.DataFrame | None = None,
            config: EKFConfig | None = None, t_end: float | None = None) -> EKFResult:
    """Run the EKF over all available measurements from ``init.t0`` to ``t_end``.

    ``gps`` must already have outages removed (see ``outage_simulator``) — the
    filter uses every GPS row it is given.
    """
    cfg = config or EKFConfig()
    use = set(cfg.sensors)
    streams = {
        "odom": odom if "odom" in use else None,
        "imu": imu if "imu" in use else None,
        "lidar": lidar_odom if "lidar" in use else None,
        "gps": gps if "gps" in use else None,
    }
    for k, df in list(streams.items()):
        if df is not None:
            df = df[df["t"] >= init.t0]
            if t_end is not None:
                df = df[df["t"] <= t_end]
            streams[k] = df.reset_index(drop=True)
    if all(df is None or len(df) == 0 for df in streams.values()):
        raise ValueError("EKF has no measurements for the selected sensors")

    arr = {k: (df.to_dict("list") if df is not None else None) for k, df in streams.items()}
    times, kinds, rows = _merge_events(streams)

    N = 7
    s = np.array([init.x, init.y, init.theta, 0.0, 0.0, 0.0, cfg.odom_scale_init])
    k_sig = cfg.odom_scale_sigma if cfg.estimate_odom_scale else 0.0
    P = np.diag([cfg.init_sigma_xy ** 2, cfg.init_sigma_xy ** 2, cfg.init_sigma_theta ** 2,
                 0.5 ** 2, 0.5 ** 2, 0.005 ** 2, k_sig ** 2])
    q_k = cfg.q_k if cfg.estimate_odom_scale else 0.0
    lx, ly = cfg.lever_arm
    I_N = np.eye(N)

    H_vw = np.zeros((2, N)); H_vw[0, 3] = 1; H_vw[1, 4] = 1
    H_imu = np.zeros((1, N)); H_imu[0, 4] = 1; H_imu[0, 5] = 1
    R_odom = np.diag([cfg.odom_sigma_v ** 2, cfg.odom_sigma_w ** 2])
    R_lidar = np.diag([cfg.lidar_sigma_v ** 2, cfg.lidar_sigma_w ** 2])
    R_imu = np.array([[cfg.imu_sigma_w ** 2]])

    t_cur = init.t0
    last_gps_t = init.t0 if "gps" in use else -np.inf
    out, ev = [], []
    next_out = init.t0
    n_gps_used = n_gps_rej = consecutive_rej = 0
    last_nis = np.nan
    gps_used_since_out = gps_rej_since_out = False

    def kf_update(z, h, H, R, freeze_k=True):
        # The odometry scale k is only *estimated* from GPS / LiDAR updates
        # (the only sensors that observe distance independently of the wheels).
        # In every other update it is a Schmidt "consider" state: its gain row
        # is zeroed, so its uncertainty still inflates P but k is not changed.
        # The Joseph-form covariance update stays valid for this sub-optimal gain.
        nonlocal s, P
        y = z - h
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        if freeze_k:
            K[6, :] = 0.0
        s = s + K @ y
        IKH = I_N - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form
        return y, S

    for t, kind, r in zip(times, kinds, rows):
        dt = t - t_cur
        if dt > 0:
            # ---- predict ----
            c, sn = math.cos(s[2]), math.sin(s[2])
            v = s[3]
            s[0] += v * c * dt
            s[1] += v * sn * dt
            s[2] += s[4] * dt
            F = I_N.copy()
            F[0, 2] = -v * sn * dt; F[0, 3] = c * dt
            F[1, 2] = v * c * dt;   F[1, 3] = sn * dt
            F[2, 4] = dt
            Q = np.diag([cfg.q_xy * dt, cfg.q_xy * dt, cfg.q_theta * dt, cfg.q_v * dt, cfg.q_w * dt,
                         cfg.q_bg * dt, q_k * dt])
            P = F @ P @ F.T + Q
            t_cur = t
            # ---- record output on a fixed grid ----
            while next_out <= t_cur:
                out.append((next_out, s[0], s[1], s[2], s[3], s[4], s[5], s[6], P[0, 0], P[1, 1], P[0, 1], P[2, 2],
                            gps_used_since_out, gps_rej_since_out, last_nis, next_out - last_gps_t))
                gps_used_since_out = gps_rej_since_out = False
                next_out += cfg.output_dt

        if kind == 0:      # wheel odometry
            d = arr["odom"]
            H = np.zeros((2, N)); H[0, 3] = s[6]; H[0, 6] = s[3]; H[1, 4] = 1
            kf_update(np.array([d["v"][r], d["w"][r]]), np.array([s[6] * s[3], s[4]]), H, R_odom)
        elif kind == 1:    # IMU gyro
            kf_update(np.array([arr["imu"]["wz"][r]]), np.array([s[4] + s[5]]), H_imu, R_imu)
        elif kind == 2:    # LiDAR odometry
            d = arr["lidar"]
            kf_update(np.array([d["v"][r], d["w"][r]]), s[[3, 4]], H_vw, R_lidar, freeze_k=not cfg.estimate_odom_scale)
        else:              # GPS
            d = arr["gps"]
            c, sn = math.cos(s[2]), math.sin(s[2])
            h = np.array([s[0] + c * lx - sn * ly, s[1] + sn * lx + c * ly])
            H = np.zeros((2, N)); H[0, 0] = 1; H[1, 1] = 1
            H[0, 2] = -sn * lx - c * ly; H[1, 2] = c * lx - sn * ly
            sig = max(cfg.gps_sigma_floor, d["sigma"][r] * cfg.gps_sigma_scale)
            R = np.eye(2) * sig ** 2
            z = np.array([d["x"][r], d["y"][r]])
            innov = z - h
            S = H @ P @ H.T + R
            nis = float(innov @ np.linalg.solve(S, innov))
            last_nis = nis
            accept = nis <= cfg.gps_gate or consecutive_rej >= cfg.gps_max_consecutive_reject
            if accept:
                kf_update(z, h, H, R, freeze_k=not cfg.estimate_odom_scale)
                n_gps_used += 1; consecutive_rej = 0; last_gps_t = t
                gps_used_since_out = True
            else:
                n_gps_rej += 1; consecutive_rej += 1
                gps_rej_since_out = True
            ev.append((t, nis, accept, float(np.hypot(*innov))))

    cols = ["t", "x", "y", "theta", "v", "w", "b_g", "k_odom", "var_x", "var_y", "cov_xy", "var_theta",
            "gps_used", "gps_rejected", "nis", "time_since_gps"]
    est = pd.DataFrame(out, columns=cols)
    est["theta"] = np.arctan2(np.sin(est["theta"]), np.cos(est["theta"]))
    # antenna-point estimate (what the reference measures)
    ct, stt = np.cos(est["theta"]), np.sin(est["theta"])
    est["x_ant"] = est["x"] + ct * lx - stt * ly
    est["y_ant"] = est["y"] + stt * lx + ct * ly
    est["r95"] = position_radius(est, 0.95)
    events = pd.DataFrame(ev, columns=["t", "nis", "accepted", "innovation_m"])
    counts = {"gps_used": n_gps_used, "gps_rejected": n_gps_rej,
              **{k: (0 if v is None else len(v)) for k, v in streams.items()}}
    return EKFResult(est, events, cfg, counts)


def position_radius(est: pd.DataFrame, p: float = 0.95) -> np.ndarray:
    """Radius of the circle enclosing the p-probability position ellipse
    (conservative: uses the major axis). Units: metres."""
    a, b, c = est["var_x"].to_numpy(), est["var_y"].to_numpy(), est["cov_xy"].to_numpy()
    lam_max = 0.5 * (a + b) + np.sqrt(0.25 * (a - b) ** 2 + c ** 2)
    return np.sqrt(CHI2_2DOF[p] * np.maximum(lam_max, 0.0))


def ellipse_points(est_row, p: float = 0.95, n: int = 48):
    """Boundary of the p-probability position ellipse around (x_ant, y_ant)."""
    cov = np.array([[est_row["var_x"], est_row["cov_xy"]], [est_row["cov_xy"], est_row["var_y"]]])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    ang = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(ang), np.sin(ang)])
    pts = vecs @ (np.sqrt(CHI2_2DOF[p] * vals)[:, None] * circle)
    return est_row["x_ant"] + pts[0], est_row["y_ant"] + pts[1]
