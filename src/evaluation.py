"""Evaluation metrics (shared by SP1–SP7).

All errors are horizontal Euclidean distances (m) between the estimated
reference point (GPS-antenna point for CitrusFarm) and the reference
trajectory, evaluated **at reference timestamps** inside the estimate's time
span. Nothing here feeds back into an estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coordinate_utils import interp_series, path_length

TARGET_M = 0.3  # project target from the team plan — a goal, not an assumed result


def error_series(est: pd.DataFrame, ref: pd.DataFrame, x_col: str = "x_ant", y_col: str = "y_ant") -> pd.DataFrame:
    """Interpolate the estimate onto reference times and compute position error."""
    if x_col not in est.columns:
        x_col, y_col = "x", "y"
    t = ref["t"].to_numpy()
    mask = (t >= est["t"].iloc[0]) & (t <= est["t"].iloc[-1])
    t = t[mask]
    exy = interp_series(t, est["t"].to_numpy(), est[[x_col, y_col]].to_numpy())
    rx, ry = ref["x"].to_numpy()[mask], ref["y"].to_numpy()[mask]
    out = pd.DataFrame({"t": t, "ref_x": rx, "ref_y": ry, "est_x": exy[:, 0], "est_y": exy[:, 1]})
    out["err"] = np.hypot(out["est_x"] - rx, out["est_y"] - ry)
    out["dist"] = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rx), np.diff(ry)))])
    return out


def summary_metrics(err: pd.DataFrame, threshold: float = TARGET_M) -> dict:
    e = err["err"].to_numpy()
    if len(e) == 0:
        return {}
    dist = float(err["dist"].iloc[-1])
    dur = float(err["t"].iloc[-1] - err["t"].iloc[0])
    return {
        "n": int(len(e)),
        "duration_s": dur,
        "distance_m": dist,
        "mean_m": float(np.mean(e)),
        "median_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "p95_m": float(np.percentile(e, 95)),
        "max_m": float(np.max(e)),
        "final_m": float(e[-1]),
        "final_drift_pct_of_distance": float(100 * e[-1] / dist) if dist > 0 else np.nan,
        "pct_time_within_target": float(100 * np.mean(e <= threshold)),
    }


def outage_metrics(err: pd.DataFrame, outages: list[tuple[float, float]], threshold: float = TARGET_M,
                   recovery_window: float = 60.0) -> pd.DataFrame:
    """Per-outage robustness metrics.

    ``outages`` = list of (t_start, t_end) absolute times.

    * max_err_m / end_err_m: error peak and error when GPS returns
    * time_to_exceed_s: time from outage start until error first > threshold
      (NaN = never exceeded during the outage)
    * pct_within_target: share of outage time with error <= threshold
    * growth_rate_m_per_s: least-squares slope of error during the outage
    * recovery_s: time after GPS returns until error <= threshold again
      (0 if it never left; NaN if not recovered within ``recovery_window``)
    """
    rows = []
    t, e = err["t"].to_numpy(), err["err"].to_numpy()
    for (a, b) in outages:
        m = (t >= a) & (t <= b)
        if m.sum() < 2:
            continue
        te, ee = t[m], e[m]
        exceed = np.flatnonzero(ee > threshold)
        after = (t > b) & (t <= b + recovery_window)
        if ee[-1] <= threshold:
            rec = 0.0
        else:
            ok = np.flatnonzero(e[after] <= threshold)
            rec = float(t[after][ok[0]] - b) if ok.size else np.nan
        rows.append({
            "start_s": a, "duration_s": b - a,
            "max_err_m": float(ee.max()), "end_err_m": float(ee[-1]),
            "mean_err_m": float(ee.mean()),
            "time_to_exceed_s": float(te[exceed[0]] - a) if exceed.size else np.nan,
            "pct_within_target": float(100 * np.mean(ee <= threshold)),
            "growth_rate_m_per_s": float(np.polyfit(te - a, ee, 1)[0]),
            "recovery_s": rec,
        })
    return pd.DataFrame(rows)


def coverage(err: pd.DataFrame, est: pd.DataFrame, radius_col: str = "r95") -> float:
    """Fraction of reference samples whose true error lies inside the
    estimator's own uncertainty radius (consistency check; ~95% expected for r95)."""
    r = interp_series(err["t"].to_numpy(), est["t"].to_numpy(), est[radius_col].to_numpy())
    return float(np.mean(err["err"].to_numpy() <= r))


def trajectory_stats(df: pd.DataFrame) -> dict:
    return {"path_length_m": path_length(df["x"], df["y"]),
            "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0])}
