"""Coordinate-frame utilities.

Conventions used throughout the project
---------------------------------------
* Positions are 2-D, metres, in a local planar frame ("map" frame).
  For CitrusFarm the map frame is the frame of ``gt.csv`` (local metres,
  origin = first ground-truth sample). For raw NavSatFix data we convert
  WGS84 lat/lon to a local East-North-Up tangent plane.
* Heading ``theta`` is radians, counter-clockwise from the map +x axis,
  wrapped to (-pi, pi].
* Robot body frame (ROS REP-103): x forward, y left, z up.
"""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    return np.arctan2(np.sin(a), np.cos(a))


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def quat_to_yaw(qx, qy, qz, qw):
    """Yaw (rotation about z) from a quaternion, vectorised."""
    qx, qy, qz, qw = (np.asarray(v, dtype=float) for v in (qx, qy, qz, qw))
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """WGS84 geodetic -> local ENU (metres) about (lat0, lon0, alt0).

    Exact ECEF-based conversion (not a flat-earth approximation)."""
    def to_ecef(la, lo, h):
        la, lo = np.radians(la), np.radians(lo)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
        x = (n + h) * np.cos(la) * np.cos(lo)
        y = (n + h) * np.cos(la) * np.sin(lo)
        z = (n * (1 - WGS84_E2) + h) * np.sin(la)
        return np.stack([x, y, z], axis=-1)

    lat, lon, alt = (np.asarray(v, dtype=float) for v in (lat, lon, alt))
    d = to_ecef(lat, lon, alt) - to_ecef(lat0, lon0, alt0)
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    r = np.array([
        [-np.sin(lo0), np.cos(lo0), 0.0],
        [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)],
        [np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)],
    ])
    enu = d @ r.T
    return enu[..., 0], enu[..., 1], enu[..., 2]


def body_to_map(x, y, theta, offset_xy):
    """Position of a body-fixed point (e.g. GPS antenna lever arm) in the map frame.

    ``offset_xy`` is (dx, dy) of the point in the robot body frame."""
    dx, dy = offset_xy
    x, y, theta = (np.asarray(v, dtype=float) for v in (x, y, theta))
    return x + np.cos(theta) * dx - np.sin(theta) * dy, y + np.sin(theta) * dx + np.cos(theta) * dy


def fit_se2(src: np.ndarray, dst: np.ndarray):
    """Least-squares rigid 2-D transform (R, t) minimising ||R @ src + t - dst||.

    Umeyama/Kabsch without scale. ``src``/``dst`` are (N, 2) arrays of
    corresponding points. Returns (theta, t[2])."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 2:
        raise ValueError("need >= 2 correspondences to fit SE(2)")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    h = (src - mu_s).T @ (dst - mu_d)
    theta = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    t = mu_d - rot2(theta) @ mu_s
    return float(theta), t


def apply_se2(xy: np.ndarray, theta: float, t) -> np.ndarray:
    return np.asarray(xy, float) @ rot2(theta).T + np.asarray(t, float)


def heading_from_track(x, y, min_distance: float = 2.0):
    """Initial heading from the first ``min_distance`` metres of a position track.

    Used to initialise heading from GPS/reference at start-up (the way a real
    robot would after driving a couple of metres with GPS). Returns
    (theta, index_used) or (nan, -1) if the track never moves that far."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.hypot(x - x[0], y - y[0])
    idx = np.flatnonzero(d >= min_distance)
    if idx.size == 0:
        return float("nan"), -1
    i = int(idx[0])
    return float(np.arctan2(y[i] - y[0], x[i] - x[0])), i


def interp_series(t_query, t, values):
    """Linear interpolation of (N,) or (N, k) values onto ``t_query``.

    Query times outside [t[0], t[-1]] return NaN (no extrapolation)."""
    t_query, t = np.asarray(t_query, float), np.asarray(t, float)
    values = np.asarray(values, float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    out_of_range = (t_query < t[0]) | (t_query > t[-1])
    if values.ndim == 1:
        out = np.interp(t_query, t, values)
        out[out_of_range] = np.nan
        return out
    out = np.column_stack([np.interp(t_query, t, values[:, k]) for k in range(values.shape[1])])
    out[out_of_range] = np.nan
    return out


def interp_angle(t_query, t, theta):
    """Interpolate angles via unit vectors (handles wrap-around)."""
    c = interp_series(t_query, t, np.cos(theta))
    s = interp_series(t_query, t, np.sin(theta))
    return np.arctan2(s, c)


def path_length(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))
