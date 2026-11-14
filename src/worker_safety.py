"""SP7 — worker-aware safety decision support (SIMULATED workers).

CitrusFarm contains no annotated worker positions, so worker zones here are
**synthetic** and always labelled as such. This module does not detect people
and is not a certified safety controller; it shows how localization
uncertainty could be turned into explicit, configurable recommendations.

Rule set (all parameters in :class:`SafetyConfig`)::

    buffer = min_buffer_m + k_uncertainty * r95 + risk_margin_m * in_high_risk_zone
    clearance = distance(estimated robot position, worker zone) - buffer

    clearance <= 0                -> STOP      (buffer overlaps a worker zone)
    clearance <= slow_margin_m    -> SLOW      (buffer approaching a worker zone)
    monitor state STOP            -> STOP      (position not trustworthy)
    monitor state DEGRADED        -> at least SLOW
    otherwise                     -> CONTINUE
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .coordinate_utils import interp_series

CONTINUE, SLOW, STOP = "CONTINUE", "SLOW", "STOP"
DECISION_LEVEL = {CONTINUE: 0, SLOW: 1, STOP: 2}
LIGHT = {CONTINUE: ("green", "#0ca30c"), SLOW: ("yellow", "#fab219"), STOP: ("red", "#d03b3b")}

# Spanish wording drafted for the demo; to be reviewed by a native speaker / domain lead.
MESSAGES = {
    CONTINUE: {
        "en": "Robot operating normally. Position is reliable.",
        "es": "El robot opera normalmente. La posición es confiable.",
        "icon": "🟢",
    },
    SLOW: {
        "en": "Caution: robot near the work area or its position is uncertain. Slow-down recommended (simulated).",
        "es": "Precaución: el robot está cerca del área de trabajo o su posición es incierta. Se recomienda reducir la velocidad (simulado).",
        "icon": "🟡",
    },
    STOP: {
        "en": "STOP: robot position uncertain near the crew. Stop recommended. Stay clear of the robot (simulated).",
        "es": "ALTO: posición del robot incierta cerca del equipo. Se recomienda detener el robot. Manténgase alejado (simulado).",
        "icon": "🔴",
    },
}


@dataclass
class WorkerZone:
    """Synthetic worker zone: a circle (crew member / small group)."""
    zone_id: str
    x: float
    y: float
    radius: float = 1.0
    label: str = "SIMULATED worker"


@dataclass
class SafetyConfig:
    min_buffer_m: float = 2.0
    k_uncertainty: float = 1.0
    slow_margin_m: float = 3.0
    risk_margin_m: float = 1.0
    use_monitor_state: bool = True


def place_workers_along_route(ref: pd.DataFrame, fractions=(0.35, 0.7), lateral_offset: float = 2.5,
                              radius: float = 1.0) -> list[WorkerZone]:
    """SIMULATED worker zones beside the reference route at given fractions of
    the distance travelled (reference used only to place the scenario)."""
    x, y = ref["x"].to_numpy(), ref["y"].to_numpy()
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    zones = []
    for k, f in enumerate(fractions):
        i = int(np.searchsorted(d, f * d[-1]))
        i = min(max(i, 1), len(x) - 2)
        hx, hy = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
        n = np.hypot(hx, hy) or 1.0
        zones.append(WorkerZone(f"W{k + 1}", float(x[i] - hy / n * lateral_offset),
                                float(y[i] + hx / n * lateral_offset), radius))
    return zones


def _zone_distance(px, py, zones: list[WorkerZone]):
    """Distance from points to the nearest zone edge, and nearest zone index."""
    if not zones:
        return np.full(len(px), np.inf), np.full(len(px), -1)
    d = np.stack([np.hypot(px - z.x, py - z.y) - z.radius for z in zones])
    return d.min(0), d.argmin(0)


def evaluate_safety(est: pd.DataFrame, states: pd.DataFrame | None, zones: list[WorkerZone],
                    cfg: SafetyConfig | None = None, risk_flag: np.ndarray | None = None) -> pd.DataFrame:
    """Per-timestep decision from estimator outputs only (no ground truth)."""
    cfg = cfg or SafetyConfig()
    px, py = est["x_ant"].to_numpy(), est["y_ant"].to_numpy()
    r95 = est["r95"].to_numpy()
    risk = np.zeros(len(est)) if risk_flag is None else np.asarray(risk_flag, float)
    buffer = cfg.min_buffer_m + cfg.k_uncertainty * r95 + cfg.risk_margin_m * risk
    dist, nearest = _zone_distance(px, py, zones)
    clearance = dist - buffer
    level = np.where(clearance <= 0, 2, np.where(clearance <= cfg.slow_margin_m, 1, 0))
    reason = np.where(clearance <= 0, "safety buffer overlaps worker zone",
                      np.where(clearance <= cfg.slow_margin_m, "safety buffer approaching worker zone", "clear"))
    if cfg.use_monitor_state and states is not None:
        ml = states["level"].to_numpy()
        reason = np.where((ml == 2) & (level < 2), "localization not trustworthy (monitor STOP)",
                          np.where((ml == 1) & (level < 1), "localization degraded (monitor)", reason))
        level = np.maximum(level, ml)
    names = np.array([CONTINUE, SLOW, STOP])
    return pd.DataFrame({"t": est["t"], "x": px, "y": py, "r95": r95, "buffer_m": buffer,
                         "dist_to_worker_m": dist, "nearest_zone": nearest, "clearance_m": clearance,
                         "level": level, "decision": names[level], "reason": reason})


def score_safety(dec: pd.DataFrame, ref: pd.DataFrame, zones: list[WorkerZone],
                 cfg: SafetyConfig | None = None) -> dict:
    """Offline check against the reference trajectory.

    * true_inside_buffer_pct: % of time the true position lies within the
      uncertainty part of the buffer (estimate ± k*r95) — the plan's
      "how often the true position fell outside the buffer" metric.
    * unsafe_true_samples: true robot closer than min_buffer to a worker zone.
    * missed_unsafe: of those, how many had decision CONTINUE (should be 0).
    """
    cfg = cfg or SafetyConfig()
    t = ref["t"].to_numpy()
    m = (t >= dec["t"].iloc[0]) & (t <= dec["t"].iloc[-1])
    t, rx, ry = t[m], ref["x"].to_numpy()[m], ref["y"].to_numpy()[m]
    ex = interp_series(t, dec["t"].to_numpy(), dec["x"].to_numpy())
    ey = interp_series(t, dec["t"].to_numpy(), dec["y"].to_numpy())
    margin = interp_series(t, dec["t"].to_numpy(), cfg.k_uncertainty * dec["r95"].to_numpy())
    lvl = np.round(interp_series(t, dec["t"].to_numpy(), dec["level"].to_numpy())).astype(int)
    err = np.hypot(ex - rx, ey - ry)
    true_dist, _ = _zone_distance(rx, ry, zones)
    unsafe = true_dist < cfg.min_buffer_m
    return {
        "true_inside_buffer_pct": 100 * float(np.mean(err <= margin)),
        "unsafe_true_samples": int(unsafe.sum()),
        "missed_unsafe": int(np.sum(unsafe & (lvl == 0))),
        "stop_pct": 100 * float(np.mean(lvl == 2)),
        "slow_pct": 100 * float(np.mean(lvl == 1)),
    }
