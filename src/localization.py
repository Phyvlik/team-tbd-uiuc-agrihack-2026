"""SP1 — trajectory estimation baselines (dead reckoning) and initial alignment.

B1: wheel odometry only (v and w from the wheel encoders).
B2: wheel odometry speed + IMU gyro yaw rate (heading from the IMU).

Initial alignment policy (applies identically to every method so comparisons
are fair, and uses **only data from the first few seconds**):

* initial position = first reference/GPS sample (converted from the antenna to
  base_link using the known lever arm and the initial heading);
* initial heading  = direction of the reference/GPS track over the first
  ``align_distance`` metres of travel.

A full-trajectory SE(2) fit (``coordinate_utils.fit_se2``) is available only
as a *diagnostic* for frame problems — it uses future data and is never used
to report headline errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coordinate_utils import body_to_map, heading_from_track, interp_series, wrap_angle


@dataclass
class InitialPose:
    t0: float
    x: float
    y: float
    theta: float
    method: str


def initial_pose_from_track(track: pd.DataFrame, lever_arm=(0.0, 0.0), align_distance: float = 2.0,
                            t_start: float | None = None, odom: pd.DataFrame | None = None) -> InitialPose:
    """Initial base_link pose from a position track (reference or GPS).

    Only uses samples from ``t_start`` until the robot has moved
    ``align_distance`` metres. The track direction describes the heading in
    the *middle* of that window; if ``odom`` is given, the yaw the robot
    turned between ``t_start`` and that midpoint (from its own wheel yaw rate)
    is subtracted so the heading refers to ``t_start``."""
    tr = track if t_start is None else track[track["t"] >= t_start]
    x, y = tr["x"].to_numpy(), tr["y"].to_numpy()
    theta, idx = heading_from_track(x, y, align_distance)
    if idx < 0:
        raise ValueError(f"track never moves {align_distance} m; cannot initialise heading")
    if odom is not None:
        t0 = tr["t"].iloc[0]
        t_mid = 0.5 * (t0 + tr["t"].iloc[idx])
        o = odom[(odom["t"] >= t0) & (odom["t"] <= t_mid)]
        if len(o) > 1:
            theta = float(wrap_angle(theta - np.sum(o["w"].to_numpy()[:-1] * np.diff(o["t"].to_numpy()))))
    # antenna -> base_link: p_base = p_ant - R(theta) @ lever
    bx, by = body_to_map(x[0], y[0], theta, (-lever_arm[0], -lever_arm[1]))
    return InitialPose(float(tr["t"].iloc[0]), float(bx), float(by), theta,
                       f"track heading over first {align_distance:.1f} m "
                       f"({tr['t'].iloc[idx] - tr['t'].iloc[0]:.1f} s)")


def dead_reckon(t, v, w, x0=0.0, y0=0.0, theta0=0.0) -> pd.DataFrame:
    """Integrate a unicycle model with midpoint heading.

    ``v[k], w[k]`` are the body twist measured at ``t[k]`` and are held over
    [t[k], t[k+1]). Returns t, x, y, theta."""
    t, v, w = (np.asarray(a, float) for a in (t, v, w))
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    dt = np.diff(t)
    dtheta = w[:-1] * dt
    theta = theta0 + np.concatenate([[0.0], np.cumsum(dtheta)])
    th_mid = theta[:-1] + 0.5 * dtheta
    x = x0 + np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(th_mid) * dt)])
    y = y0 + np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(th_mid) * dt)])
    return pd.DataFrame({"t": t, "x": x, "y": y, "theta": wrap_angle(theta)})


def baseline_b1(odom: pd.DataFrame, init: InitialPose) -> pd.DataFrame:
    """B1: wheel odometry only."""
    o = odom[odom["t"] >= init.t0]
    return dead_reckon(o["t"], o["v"], o["w"], init.x, init.y, init.theta)


def baseline_b2(odom: pd.DataFrame, imu: pd.DataFrame, init: InitialPose) -> pd.DataFrame:
    """B2: wheel odometry speed + IMU gyro yaw rate (interpolated to odom times)."""
    o = odom[(odom["t"] >= init.t0) & (odom["t"] <= imu["t"].iloc[-1]) & (odom["t"] >= imu["t"].iloc[0])]
    wz = interp_series(o["t"].to_numpy(), imu["t"].to_numpy(), imu["wz"].to_numpy())
    return dead_reckon(o["t"], o["v"], wz, init.x, init.y, init.theta)
