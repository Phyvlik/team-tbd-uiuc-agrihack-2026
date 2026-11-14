"""Operator-facing derivations from backend outputs (no new estimation).

* GPS status (Available / Degraded / Lost) — from the estimator's own inputs:
  time since the last accepted fix and recent consistency-gate rejections.
* "What the robot can do now" — explicit advisory rules on top of the backend
  integrity monitor (reliability_monitor) and worker-safety decision
  (worker_safety). Advisory only: the dashboard does not control a robot.
* Bilingual alerts (EN/ES) — what happened + what to do.
* Timestamp-linked events — GPS lost/restored, state changes, crew warnings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GPS_STALE_S = 2.0              # same rule as MonitorConfig.gps_stale_s
REJECT_WINDOW_S = 3.0          # same rule as MonitorConfig.disagreement_window_s
ATTENTION_AFTER_S = 5.0        # degraded / GPS lost this long → ask for operator attention
VERIFY_AFTER_STOP_S = 20.0     # after a STOP, ask for manual verification for this long

GPS_AVAILABLE, GPS_DEGRADED, GPS_LOST = "Available", "Degraded", "Lost"


def gps_status_series(est: pd.DataFrame) -> np.ndarray:
    t = est.t.to_numpy()
    lost = est.time_since_gps.to_numpy() > GPS_STALE_S
    rej = est.gps_rejected.to_numpy().astype(bool)
    last = -np.inf
    recent = np.zeros(len(t), bool)
    for i in range(len(t)):
        if rej[i]:
            last = t[i]
        recent[i] = t[i] - last <= REJECT_WINDOW_S
    return np.where(lost, GPS_LOST, np.where(recent, GPS_DEGRADED, GPS_AVAILABLE))


def time_in_state(levels: np.ndarray, t: np.ndarray, i: int, min_level: int) -> float:
    """Seconds the series has continuously been at >= min_level up to index i."""
    if levels[i] < min_level:
        return 0.0
    j = i
    while j > 0 and levels[j - 1] >= min_level and (min_level > 0 or levels[j - 1] == levels[i]):
        j -= 1
    return float(t[i] - t[j])


@dataclass
class Situation:
    t_rel: float
    nav_state: str           # TRUSTED | DEGRADED | STOP
    nav_reason: str
    gps: str                 # Available | Degraded | Lost
    gps_age: float
    r95: float
    speed: float
    safety: str | None       # CONTINUE | SLOW | STOP (simulated crews) or None
    safety_reason: str
    dist_crew: float
    buffer_m: float
    degraded_for: float
    since_stop: float        # seconds since the last STOP state ended (inf if none)
    just_restored: bool


def situation(est, states, gps_series, dec, i: int, t0: float, outages_abs) -> Situation:
    t = est.t.to_numpy()
    lv = states.level.to_numpy()
    row, srow = est.iloc[i], states.iloc[i]
    stop_idx = np.flatnonzero(lv[: i + 1] == 2)
    since_stop = float(t[i] - t[stop_idx[-1]]) if len(stop_idx) and lv[i] < 2 else (0.0 if lv[i] == 2 else np.inf)
    restored = any(b <= t[i] < b + 10.0 for _, b in outages_abs)
    d = dec.iloc[i] if dec is not None else None
    return Situation(
        t_rel=float(t[i] - t0), nav_state=srow.state, nav_reason=srow.reason, gps=str(gps_series[i]),
        gps_age=float(row.time_since_gps), r95=float(row.r95), speed=float(abs(row.v)),
        safety=None if d is None else d.decision, safety_reason="" if d is None else d.reason,
        dist_crew=np.inf if d is None else float(d.dist_to_worker_m),
        buffer_m=np.nan if d is None else float(d.buffer_m),
        degraded_for=time_in_state(lv, t, i, 1), since_stop=since_stop, just_restored=restored)


# ------------------------------------------------------------- checklist ---
YES, NO, ADVISED, NA = "yes", "no", "advised", "na"


def capabilities(s: Situation) -> list[tuple[str, str, str]]:
    """(item, status, why). Status: yes = allowed, advised = recommended now, no = not advised, na = not needed."""
    stop = s.nav_state == "STOP" or s.safety == "STOP"
    caution = s.nav_state == "DEGRADED" or s.safety == "SLOW"
    items = []
    items.append(("Continue autonomous operation",
                  YES if not stop and not caution else NO,
                  "position trusted and no crew conflict" if not stop and not caution else
                  ("stop recommended" if stop else "position less certain or crew nearby")))
    items.append(("Continue at reduced speed",
                  NO if stop else (ADVISED if caution else NA),
                  "not advised while stop is recommended" if stop else
                  (s.nav_reason if s.nav_state == "DEGRADED" else s.safety_reason) if caution else "not needed"))
    attention = stop or s.degraded_for >= ATTENTION_AFTER_S or (s.gps == "Lost" and s.gps_age >= ATTENTION_AFTER_S)
    items.append(("Request operator attention", ADVISED if attention else NA,
                  (f"GPS lost for {s.gps_age:.0f} s" if s.gps == "Lost" else
                   f"degraded for {s.degraded_for:.0f} s" if s.degraded_for else "stop recommended")
                  if attention else "not needed"))
    items.append(("Stop recommended", ADVISED if stop else NA,
                  (s.nav_reason if s.nav_state == "STOP" else s.safety_reason) if stop else "not needed"))
    verify = s.nav_state == "STOP" or s.since_stop <= VERIFY_AFTER_STOP_S
    items.append(("Manual position check before resuming", ADVISED if verify else NA,
                  "a stop was recommended recently — confirm the robot is in the right row" if verify else "not needed"))
    return items


# ---------------------------------------------------------------- alerts ---
# Spanish drafted for the demo; needs native-speaker review (see docs).
ALERTS = {
    "stop": ("■", "STOP", "Stop recommended. The robot cannot confirm its position.",
             "Send a person or take manual control.",
             "Se recomienda detener el robot. No puede confirmar su posición.",
             "Envíe a una persona o tome el control manual."),
    "crew_stop": ("■", "STOP", "Robot's safety zone overlaps a work crew (simulated).",
                  "Stop recommended. Crew: stay clear of the robot.",
                  "La zona de seguridad del robot alcanza a un equipo de trabajo (simulado).",
                  "Se recomienda detener el robot. Equipo: manténgase alejado."),
    "gps_lost": ("▲", "DEGRADED", "GPS unavailable. Position estimate becoming less reliable.",
                 "The robot is navigating with wheel and motion sensors. Reduce speed.",
                 "GPS no disponible. La estimación de posición es menos confiable.",
                 "El robot navega con sensores de ruedas y de movimiento. Reduzca la velocidad."),
    "gps_rejected": ("▲", "DEGRADED", "GPS reading rejected — it disagreed with the motion sensors.",
                     "Likely signal reflection under the canopy. Watch the status light.",
                     "Lectura de GPS rechazada: no coincide con los sensores de movimiento.",
                     "Probable reflejo de la señal bajo el follaje. Observe la luz de estado."),
    "uncertain": ("▲", "DEGRADED", "Position uncertainty is above the 0.3 m target.",
                  "Slow down and avoid rows where people are working.",
                  "La incertidumbre de posición supera el objetivo de 0.3 m.",
                  "Reduzca la velocidad y evite las hileras con personas trabajando."),
    "crew_near": ("▲", "DEGRADED", "Robot approaching a work crew (simulated).",
                  "Robot slows down. Crew: keep clear of the robot.",
                  "El robot se acerca a un equipo de trabajo (simulado).",
                  "El robot reduce la velocidad. Equipo: manténgase alejado."),
    "restored": ("●", "TRUSTED", "GPS restored. Position confidence is recovering.",
                 "Confirm the robot is in the correct row before resuming full speed.",
                 "GPS restablecido. La confianza en la posición se está recuperando.",
                 "Confirme que el robot está en la hilera correcta antes de volver a la velocidad normal."),
    "ok": ("●", "TRUSTED", "All systems normal. Position is reliable.", "No action needed.",
           "Todo normal. La posición es confiable.", "No se requiere ninguna acción."),
}


def active_alerts(s: Situation) -> list[str]:
    keys = []
    if s.nav_state == "STOP":
        keys.append("stop")
    if s.safety == "STOP" and "monitor" not in s.safety_reason:
        keys.append("crew_stop")
    if s.gps == "Lost":
        keys.append("gps_lost")
    elif s.gps == "Degraded":
        keys.append("gps_rejected")
    if s.nav_state == "DEGRADED" and s.r95 > 0.3 and s.gps != "Lost":
        keys.append("uncertain")
    if s.safety == "SLOW" and "worker" in s.safety_reason:
        keys.append("crew_near")
    if s.just_restored and s.gps == "Available":
        keys.append("restored")
    return keys or ["ok"]


# ---------------------------------------------------------------- events ---

def events(est, states, gps_series, dec, outages_abs, t0: float) -> pd.DataFrame:
    """Timestamp-linked events for the replay: (t_rel, kind, label, state)."""
    rows = []
    for a, b in outages_abs:
        rows.append((a - t0, "gps", "GPS lost", "DEGRADED"))
        rows.append((b - t0, "gps", "GPS restored", "TRUSTED"))
    lv = states.level.to_numpy()
    t = states.t.to_numpy()
    names = {0: ("Trusted again", "TRUSTED"), 1: ("Degraded", "DEGRADED"), 2: ("Stop recommended", "STOP")}
    for i in np.flatnonzero(np.diff(lv) != 0) + 1:
        lab, stt = names[int(lv[i])]
        rows.append((t[i] - t0, "state", lab, stt))
    g = np.asarray(gps_series)
    for i in np.flatnonzero((g[1:] == GPS_DEGRADED) & (g[:-1] != GPS_DEGRADED)) + 1:
        rows.append((est.t.iloc[i] - t0, "gps", "GPS rejected", "DEGRADED"))
    if dec is not None:
        dl = dec.level.to_numpy()
        reason = dec.reason.to_numpy()
        for i in np.flatnonzero(np.diff(dl) > 0) + 1:
            if "worker" in reason[i]:
                rows.append((dec.t.iloc[i] - t0, "crew", "Crew warning" if dl[i] == 1 else "Crew stop",
                             "DEGRADED" if dl[i] == 1 else "STOP"))
    ev = pd.DataFrame(rows, columns=["t_rel", "kind", "label", "state"]).sort_values("t_rel")
    # drop rapid duplicates (hysteresis flicker) for readability
    ev = ev[~((ev.label == ev.label.shift()) & (ev.t_rel - ev.t_rel.shift() < 3))]
    return ev.reset_index(drop=True)
