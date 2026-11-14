"""SP4 — localization integrity monitor and fallback states.

Inputs are the estimator's *own* outputs only (no ground truth):

* ``r95``            95% position-uncertainty radius from the EKF covariance
* ``time_since_gps`` seconds since the last accepted GPS fix
* ``gps_rejected``   GPS fix failed the NIS consistency gate (sensors disagree)

States
------
TRUSTED   r95 <= trusted_r95 and GPS recent and no recent disagreement
DEGRADED  r95 <= stop_r95, but GPS stale / recent disagreement / r95 > trusted_r95
STOP      r95 > stop_r95, or sustained disagreement  ("Fallback / Stop recommended")

Thresholds are configurable. Default trusted_r95 = 0.30 m (the team's target)
and stop_r95 = 1.0 m (≈ half the free space between a 0.6 m-wide robot and a
tree line in a ~6.5 m row; an assumption to be confirmed by the domain lead).

:func:`evaluate_warnings` scores states against the *actual* error, which is
only possible offline with a reference trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coordinate_utils import interp_series

TRUSTED, DEGRADED, STOP = "TRUSTED", "DEGRADED", "STOP"
STATE_LEVEL = {TRUSTED: 0, DEGRADED: 1, STOP: 2}
STATE_COLOR = {TRUSTED: "#0ca30c", DEGRADED: "#fab219", STOP: "#d03b3b"}
STATE_ICON = {TRUSTED: "🟢", DEGRADED: "🟡", STOP: "🔴"}
STATE_LABEL = {TRUSTED: "Trusted", DEGRADED: "Degraded", STOP: "Fallback / Stop recommended"}


@dataclass
class MonitorConfig:
    trusted_r95: float = 0.30
    stop_r95: float = 1.00
    gps_stale_s: float = 2.0
    disagreement_window_s: float = 3.0
    stop_after_disagreement_s: float = 10.0
    hysteresis_s: float = 1.0  # minimum time before de-escalating


def classify(est: pd.DataFrame, cfg: MonitorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MonitorConfig()
    t = est["t"].to_numpy()
    r95 = est["r95"].to_numpy()
    stale = est["time_since_gps"].to_numpy() > cfg.gps_stale_s
    rej = est["gps_rejected"].to_numpy().astype(bool)

    # time since last rejection, and duration of continuous disagreement
    last_rej, rej_start = -np.inf, None
    recent_rej = np.zeros(len(t), bool)
    sustained = np.zeros(len(t), bool)
    for i in range(len(t)):
        if rej[i]:
            last_rej = t[i]
            rej_start = t[i] if rej_start is None else rej_start
        elif rej_start is not None and t[i] - last_rej > cfg.disagreement_window_s:
            rej_start = None
        recent_rej[i] = t[i] - last_rej <= cfg.disagreement_window_s
        sustained[i] = rej_start is not None and t[i] - rej_start >= cfg.stop_after_disagreement_s

    raw = np.where((r95 > cfg.stop_r95) | sustained, 2,
                   np.where((r95 > cfg.trusted_r95) | stale | recent_rej, 1, 0))
    # hysteresis: escalate immediately, de-escalate only after hysteresis_s at lower level
    level = np.empty_like(raw)
    cur, since = raw[0], t[0]
    for i in range(len(t)):
        if raw[i] > cur:
            cur, since = raw[i], t[i]
        elif raw[i] < cur:
            if t[i] - since >= cfg.hysteresis_s:
                cur, since = raw[i], t[i]
        else:
            since = t[i]
        level[i] = cur
    names = np.array([TRUSTED, DEGRADED, STOP])
    reason = np.where(sustained, "sustained GPS/odometry disagreement",
             np.where(r95 > cfg.stop_r95, "uncertainty above stop limit",
             np.where(recent_rej, "GPS fix rejected by consistency check",
             np.where(stale, "GPS unavailable",
             np.where(r95 > cfg.trusted_r95, "uncertainty above target", "nominal")))))
    out = est[["t", "r95", "time_since_gps", "gps_rejected"]].copy()
    out["level"] = level
    out["state"] = names[level]
    out["reason"] = reason
    return out


def evaluate_warnings(states: pd.DataFrame, err: pd.DataFrame, error_threshold: float = 0.30,
                      outages_abs: list | None = None) -> dict:
    """Do warnings coincide with real errors?

    Ground-truth "bad" = actual error > error_threshold. Reports, per reference
    sample: confusion of (state != TRUSTED) vs bad, missed-hazard rate
    (bad while TRUSTED), false-alarm rate (not bad while warning), and per
    outage the detection latency (state escalation after the outage starts)
    and warning lead time (escalation before the error crosses the threshold).
    """
    lv = interp_series(err["t"].to_numpy(), states["t"].to_numpy(), states["level"].to_numpy())
    lv = np.round(np.nan_to_num(lv, nan=0)).astype(int)
    bad = err["err"].to_numpy() > error_threshold
    warn = lv >= 1
    tp, fp = int(np.sum(warn & bad)), int(np.sum(warn & ~bad))
    fn, tn = int(np.sum(~warn & bad)), int(np.sum(~warn & ~bad))
    res = {
        "samples": len(bad), "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
        "recall_warn_when_bad": tp / (tp + fn) if tp + fn else np.nan,
        "precision_bad_when_warn": tp / (tp + fp) if tp + fp else np.nan,
        "missed_hazard_rate": fn / (tp + fn) if tp + fn else np.nan,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        "pct_time_trusted": 100 * float(np.mean(lv == 0)),
        "pct_time_degraded": 100 * float(np.mean(lv == 1)),
        "pct_time_stop": 100 * float(np.mean(lv == 2)),
    }
    lat = []
    if outages_abs:
        st_t, st_l = states["t"].to_numpy(), states["level"].to_numpy()
        et, ee = err["t"].to_numpy(), err["err"].to_numpy()
        for a, b in outages_abs:
            m = (st_t >= a) & (st_t <= b)
            esc = st_t[m][st_l[m] >= 1]
            stop = st_t[m][st_l[m] >= 2]
            em = (et >= a) & (et <= b) & (ee > error_threshold)
            cross = et[em][0] if em.any() else np.nan
            lat.append({"outage_start": a, "detect_latency_s": (esc[0] - a) if esc.size else np.nan,
                        "stop_latency_s": (stop[0] - a) if stop.size else np.nan,
                        "error_crosses_target_s": cross - a if np.isfinite(cross) else np.nan,
                        "lead_time_s": (cross - esc[0]) if (esc.size and np.isfinite(cross)) else np.nan})
    res["per_outage"] = lat
    return res


def transitions(states: pd.DataFrame) -> pd.DataFrame:
    """Alert log: one row per state change."""
    ch = states["level"].diff().fillna(1) != 0
    return states.loc[ch, ["t", "state", "reason", "r95"]].reset_index(drop=True)
