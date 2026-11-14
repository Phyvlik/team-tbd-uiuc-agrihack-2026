"""SP3 — controlled GPS degradation scenarios.

Two kinds of GPS input exist and are always labelled:

* ``raw``   — the receiver's own live fixes (CitrusFarm ``/piksi/navsatfix_best_fix``
              in ``base_*.bag``), converted to the map frame.
* ``proxy`` — a **simulated** live GPS made from the reference trajectory plus
              white noise, used only until raw fixes are extracted. Each proxy
              fix at time t uses the reference at time t only (causal). Because
              CitrusFarm's RTK ran in fixed mode the whole time, a proxy with
              σ≈2–3 cm is a fair stand-in, but results using it must say so.

Outages remove GPS rows *before* they reach the estimator, so withheld
readings cannot be used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OUTAGE_DURATIONS_S = (10, 30, 60, 120)


@dataclass
class OutageScenario:
    """Outages are given relative to the sequence start time (seconds)."""
    outages: list = field(default_factory=list)  # list of (start_rel_s, duration_s)
    multipath_jumps: list = field(default_factory=list)  # (start_rel_s, duration_s, offset_m)
    extra_noise_sigma: float = 0.0
    seed: int = 0

    def absolute(self, t0: float) -> list[tuple[float, float]]:
        return [(t0 + a, t0 + a + d) for a, d in self.outages]

    def describe(self) -> str:
        parts = [f"{d:.0f} s outage @ {a:.0f} s" for a, d in self.outages]
        parts += [f"{o:.1f} m multipath jump @ {a:.0f} s for {d:.0f} s" for a, d, o in self.multipath_jumps]
        return "; ".join(parts) or "no injected degradation"


def gps_proxy_from_reference(ref: pd.DataFrame, rate_hz: float = 10.0, sigma: float = 0.03,
                             seed: int = 0) -> pd.DataFrame:
    """SIMULATED GPS: reference positions sampled at ``rate_hz`` + N(0, sigma²) noise."""
    rng = np.random.default_rng(seed)
    t = ref["t"].to_numpy()
    step = max(1, int(round((1.0 / rate_hz) / np.median(np.diff(t)))))
    sel = ref.iloc[::step]
    n = len(sel)
    return pd.DataFrame({
        "t": sel["t"].to_numpy(),
        "x": sel["x"].to_numpy() + rng.normal(0, sigma, n),
        "y": sel["y"].to_numpy() + rng.normal(0, sigma, n),
        "sigma": np.full(n, sigma),
        "source": "proxy_from_reference",
    })


def apply_scenario(gps: pd.DataFrame, scenario: OutageScenario, t0: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gps_for_estimator, gps_log).

    ``gps_for_estimator`` contains only the fixes the robot would receive.
    ``gps_log`` keeps every fix with an ``available`` flag and a ``degraded``
    flag (for plotting / SP4 evaluation)."""
    g = gps.copy()
    rng = np.random.default_rng(scenario.seed + 1)
    t_rel = g["t"].to_numpy() - t0
    available = np.ones(len(g), bool)
    for a, d in scenario.outages:
        available &= ~((t_rel >= a) & (t_rel < a + d))
    degraded = np.zeros(len(g), bool)
    for a, d, off in scenario.multipath_jumps:
        m = (t_rel >= a) & (t_rel < a + d)
        ang = rng.uniform(0, 2 * np.pi)
        g.loc[m, "x"] += off * np.cos(ang)
        g.loc[m, "y"] += off * np.sin(ang)
        degraded |= m
    if scenario.extra_noise_sigma > 0:
        g["x"] += rng.normal(0, scenario.extra_noise_sigma, len(g))
        g["y"] += rng.normal(0, scenario.extra_noise_sigma, len(g))
        # receiver does not know about the injected noise: sigma unchanged
    g["available"] = available
    g["degraded_injected"] = degraded
    return g[available].drop(columns=["available", "degraded_injected"]).reset_index(drop=True), g


def outage_mask(t, outages_abs) -> np.ndarray:
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    for a, b in outages_abs:
        m |= (t >= a) & (t < b)
    return m


def standard_outage_suite(duration_total: float, durations=OUTAGE_DURATIONS_S, warmup: float = 30.0,
                          n_starts: int = 3) -> list[OutageScenario]:
    """Same outage windows for every method: for each duration, ``n_starts``
    evenly spaced start times after a GPS warm-up period."""
    out = []
    for d in durations:
        latest = duration_total - d - 30.0  # leave 30 s to observe recovery
        if latest <= warmup:
            continue
        for a in np.linspace(warmup, latest, n_starts):
            out.append(OutageScenario(outages=[(float(a), float(d))]))
    return out
