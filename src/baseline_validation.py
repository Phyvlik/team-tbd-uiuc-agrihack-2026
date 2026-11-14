"""Baseline verification — is a large wheel-odometry error real drift or a
processing artefact?

For one trajectory (reference + odometry) we compute several variants side by
side so a frame/alignment bug can be told apart from genuine drift:

V0  naive_ctrl_pose      controller's integrated pose used as-is vs reference
                         (no re-zeroing, no rotation) — a common pitfall
V1  ctrl_pose_rezeroed   controller pose shifted to start at the reference
                         origin but NOT rotated into the reference frame
V2  b1_aligned           our B1 integration of (v, w), initial pose from the
                         first 2 m of reference, GPS lever arm applied  <- headline B1
V3  b1_scale_cal         V2 with wheel speed scale calibrated on OTHER
                         sequences (leave-one-out, no leakage)
V4  b1_scale_yaw_cal     V3 plus wheel yaw-rate gain calibrated on OTHER
                         sequences (leave-one-out)
D   b1_oracle_se2        V2 re-aligned with a full-trajectory SE(2) fit.
                         DIAGNOSTIC ONLY (uses future data): if D is small
                         but V2 is large, the problem is the initial heading.

Also reported: wheel-distance / reference-distance ratio (speed scale) and
yaw-rate gain of wheel odometry vs reference-derived turn rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coordinate_utils import apply_se2, body_to_map, fit_se2, interp_series, path_length, rot2
from .evaluation import error_series, summary_metrics
from .localization import baseline_b1, initial_pose_from_track


def _with_antenna(traj: pd.DataFrame, lever) -> pd.DataFrame:
    traj = traj.copy()
    traj["x_ant"], traj["y_ant"] = body_to_map(traj["x"], traj["y"], traj["theta"], lever)
    return traj


def speed_scale_ratio(ref: pd.DataFrame, odom: pd.DataFrame) -> float:
    """Wheel-integrated distance / reference path length over the common span."""
    t0, t1 = max(ref.t.iloc[0], odom.t.iloc[0]), min(ref.t.iloc[-1], odom.t.iloc[-1])
    o = odom[(odom.t >= t0) & (odom.t <= t1)]
    r = ref[(ref.t >= t0) & (ref.t <= t1)]
    d_odom = float(np.sum(np.abs(o.v.to_numpy()[:-1]) * np.diff(o.t.to_numpy())))
    return d_odom / path_length(r.x, r.y)


def yaw_rate_gain(ref: pd.DataFrame, odom: pd.DataFrame, min_speed: float = 0.5, smooth_s: float = 1.0) -> float:
    """Slope of wheel yaw rate vs reference track turn rate (1.0 = perfect).

    Reference turn rate comes from the direction of travel of the reference
    track (it has no orientation), smoothed over ``smooth_s``."""
    dt = float(np.median(np.diff(ref.t)))
    k = max(1, int(round(smooth_s / dt)))
    x = ref.x.rolling(k, center=True, min_periods=1).mean().to_numpy()
    y = ref.y.rolling(k, center=True, min_periods=1).mean().to_numpy()
    t = ref.t.to_numpy()
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    spd = np.hypot(vx, vy)
    hd = np.unwrap(np.arctan2(vy, vx))
    w_ref = np.gradient(hd, t)
    w_odom = interp_series(t, odom.t.to_numpy(), odom.w.to_numpy())
    m = (spd > min_speed) & np.isfinite(w_odom) & (np.abs(w_ref) < 2.0)
    if m.sum() < 50:
        return float("nan")
    return float(np.sum(w_odom[m] * w_ref[m]) / np.sum(w_ref[m] ** 2))


def validate_sequence(ref: pd.DataFrame, odom: pd.DataFrame, lever=(0.0, 0.0),
                      scale_from_other_sequences: float | None = None,
                      yaw_gain_from_other_sequences: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (per-variant metrics table, trajectories dict)."""
    init = initial_pose_from_track(ref, lever_arm=lever, align_distance=2.0, odom=odom)
    trajs, rows = {}, []

    # V0 naive controller pose
    if {"x_ctrl", "y_ctrl"} <= set(odom.columns):
        v0 = pd.DataFrame({"t": odom.t, "x": odom.x_ctrl, "y": odom.y_ctrl})
        trajs["V0_naive_ctrl_pose"] = v0
        # V1 re-zeroed (translation only)
        v1 = v0.copy()
        v1["x"] -= v0.x.iloc[0] - ref.x.iloc[0]
        v1["y"] -= v0.y.iloc[0] - ref.y.iloc[0]
        trajs["V1_ctrl_pose_rezeroed_no_rotation"] = v1

    # V2 B1 aligned
    b1 = _with_antenna(baseline_b1(odom, init), lever)
    trajs["V2_B1_aligned"] = b1

    # V3 B1 with leave-one-out speed-scale calibration
    if scale_from_other_sequences and np.isfinite(scale_from_other_sequences):
        o = odom.copy()
        o["v"] = o["v"] / scale_from_other_sequences
        trajs["V3_B1_scale_calibrated_LOO"] = _with_antenna(baseline_b1(o, init), lever)
        if yaw_gain_from_other_sequences and np.isfinite(yaw_gain_from_other_sequences):
            o["w"] = o["w"] / yaw_gain_from_other_sequences
            trajs["V4_B1_scale_and_yaw_calibrated_LOO"] = _with_antenna(baseline_b1(o, init), lever)

    # D oracle SE(2) — diagnostic only
    err_b1 = error_series(b1, ref)
    exy = err_b1[["est_x", "est_y"]].to_numpy()
    th, tt = fit_se2(exy, err_b1[["ref_x", "ref_y"]].to_numpy())
    d = b1.copy()
    d[["x_ant", "y_ant"]] = apply_se2(b1[["x_ant", "y_ant"]].to_numpy(), th, tt)
    trajs["D_B1_oracle_SE2_diagnostic"] = d

    for name, tr in trajs.items():
        m = summary_metrics(error_series(tr, ref))
        rows.append({"variant": name, **m})
    table = pd.DataFrame(rows)
    table.attrs["oracle_rotation_deg"] = float(np.degrees(th))
    table.attrs["init_method"] = init.method
    return table, trajs


def diagnose(table: pd.DataFrame, scale_ratio: float, yaw_gain: float) -> list[str]:
    """Human-readable findings derived from the numbers (no guesses beyond them)."""
    get = lambda v: table.loc[table.variant == v].iloc[0] if (table.variant == v).any() else None
    notes = []
    v0, v2, d = get("V0_naive_ctrl_pose"), get("V2_B1_aligned"), get("D_B1_oracle_SE2_diagnostic")
    if v0 is not None and v2 is not None and v0["max_m"] > 2 * v2["max_m"]:
        notes.append(f"Using the controller pose without frame alignment inflates max error "
                     f"from {v2['max_m']:.1f} m to {v0['max_m']:.1f} m -> frame/alignment artefact.")
    if abs(scale_ratio - 1) > 0.03:
        notes.append(f"Wheel distance is {100 * (scale_ratio - 1):+.1f}% vs RTK path -> systematic speed-scale "
                     f"error (real wheel-odometry drift source; calibratable).")
    if np.isfinite(yaw_gain) and abs(yaw_gain - 1) > 0.05:
        notes.append(f"Wheel yaw-rate gain {yaw_gain:.2f} (1.00 ideal) -> skid-steer turn-rate error "
                     f"(real drift source; IMU gyro should fix).")
    if v2 is not None and d is not None and d["max_m"] < 0.5 * v2["max_m"]:
        notes.append(f"A rigid re-alignment cuts max error {v2['max_m']:.1f} -> {d['max_m']:.1f} m: part of the "
                     f"error is a heading offset (initial heading or early yaw error).")
    return notes
