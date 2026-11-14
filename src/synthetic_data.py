"""SYNTHETIC demo data — for testing the interface and workflow only.

Generates a serpentine orchard route (rows + headland turns) and noisy sensor
streams with known error models. Nothing produced here is CitrusFarm data and
no metric computed on it says anything about real-world accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import SensorStreams

SYNTHETIC_LABEL = "SYNTHETIC DEMO DATA — not CitrusFarm, not a real-world accuracy result"


def orchard_layout(n_rows: int = 6, row_length: float = 80.0, row_spacing: float = 6.5):
    """Row centre lines (x0, y0, x1, y1). Rows run along +y; tree lines between them."""
    return [(i * row_spacing, 0.0, i * row_spacing, row_length) for i in range(n_rows)]


def _serpentine_path(n_rows, row_length, row_spacing, speed, dt):
    """Ground-truth (t, x, y, theta, v, w) for a lane-by-lane serpentine drive."""
    pts = []
    r = row_spacing / 2.0
    x, y, th = 0.0, 0.0, np.pi / 2
    t = 0.0

    def straight(L):
        nonlocal x, y, t
        n = int(L / (speed * dt))
        for _ in range(n):
            pts.append((t, x, y, th, speed, 0.0))
            x += speed * np.cos(th) * dt; y += speed * np.sin(th) * dt; t += dt

    def turn(direction):
        nonlocal x, y, th, t
        w = direction * speed / r
        n = int(np.pi / (abs(w) * dt))
        for _ in range(n):
            pts.append((t, x, y, th, speed, w))
            x += speed * np.cos(th + 0.5 * w * dt) * dt
            y += speed * np.sin(th + 0.5 * w * dt) * dt
            th += w * dt; t += dt
        th = np.round(th / (np.pi / 2)) * (np.pi / 2)

    for i in range(n_rows):
        straight(row_length)
        if i < n_rows - 1:
            turn(-1 if i % 2 == 0 else +1)
    a = np.array(pts)
    return pd.DataFrame({"t": a[:, 0], "x": a[:, 1], "y": a[:, 2], "theta": a[:, 3], "v": a[:, 4], "w": a[:, 5]})


def canopy_density(x, y, n_rows, row_length, row_spacing, seed=3):
    """Synthetic canopy index in [0, 1]: dense patches along some rows."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform([0, 10], [(n_rows - 1) * row_spacing, row_length - 10], size=(4, 2))
    d = np.zeros_like(np.asarray(x, float))
    for cx, cy in centers:
        d += np.exp(-(((x - cx) / 6.0) ** 2 + ((y - cy) / 12.0) ** 2))
    return np.clip(0.25 + 0.7 * d, 0, 1)


def make_synthetic_sequence(seed: int = 7, n_rows: int = 6, row_length: float = 80.0,
                            row_spacing: float = 6.5, speed: float = 1.2) -> SensorStreams:
    rng = np.random.default_rng(seed)
    truth = _serpentine_path(n_rows, row_length, row_spacing, speed, dt=0.02)  # 50 Hz
    n = len(truth)

    # wheel odometry: scale error + skid on turns + noise, 50 Hz
    odom = pd.DataFrame({
        "t": truth.t,
        "v": truth.v * 1.06 + rng.normal(0, 0.02, n),
        "w": truth.w * 0.93 + 0.004 + rng.normal(0, 0.03, n),
    })
    # IMU gyro: small bias + noise, 50 Hz
    t_imu = np.arange(truth.t.iloc[0], truth.t.iloc[-1], 0.02)  # 50 Hz
    w_imu = np.interp(t_imu, truth.t, truth.w)
    imu = pd.DataFrame({"t": t_imu, "wz": w_imu + 0.002 + rng.normal(0, 0.005, len(t_imu))})
    # LiDAR scan-matching odometry, 10 Hz
    li = truth.iloc[::5].reset_index(drop=True)
    lidar = pd.DataFrame({"t": li.t, "v": li.v * 1.01 + rng.normal(0, 0.03, len(li)),
                          "w": li.w + rng.normal(0, 0.01, len(li))})

    ref = truth[["t", "x", "y"]].iloc[::5].reset_index(drop=True)  # 10 Hz reference
    # GPS 10 Hz: quality depends on canopy (synthetic relationship, by construction)
    g = truth.iloc[::5].reset_index(drop=True)
    can = canopy_density(g.x.to_numpy(), g.y.to_numpy(), n_rows, row_length, row_spacing)
    sigma = 0.02 + 0.25 * np.clip(can - 0.5, 0, None)
    nsat = np.round(22 - 12 * can + rng.normal(0, 1, len(g))).clip(4, 30)
    gps = pd.DataFrame({"t": g.t, "x": g.x + rng.normal(0, 1, len(g)) * sigma,
                        "y": g.y + rng.normal(0, 1, len(g)) * sigma, "sigma": sigma,
                        "source": "synthetic"})
    gps_quality = pd.DataFrame({"t": g.t, "x": g.x, "y": g.y, "num_sats": nsat, "h_sigma_m": sigma})
    canopy = pd.DataFrame({"t": g.t, "x": g.x, "y": g.y, "ndvi": 0.2 + 0.6 * can + rng.normal(0, 0.03, len(g))})

    return SensorStreams(
        name="SYNTHETIC_orchard_demo", mode="synthetic", reference=ref, odom=odom, imu=imu,
        gps=gps, lidar_odom=lidar, gps_quality=gps_quality, canopy=canopy,
        reference_lever_arm=(0.0, 0.0),
        meta={"label": SYNTHETIC_LABEL, "seed": seed, "n_rows": n_rows, "row_length": row_length,
              "row_spacing": row_spacing, "rows": orchard_layout(n_rows, row_length, row_spacing),
              "error_model": "odom v x1.06, w x0.93 +0.004 rad/s; gyro bias 0.002 rad/s; "
                             "GPS sigma grows with synthetic canopy density"},
    )
