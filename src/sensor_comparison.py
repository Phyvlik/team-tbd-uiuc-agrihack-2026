"""SP5 — reduced-sensor configurations and cost/accuracy trade-off.

Costs are QUALITATIVE TIERS from the team plan, not prices. No dollar figures
or savings are computed here; add externally verified prices to
``COST_TIERS[...]["price_note"]`` only with a source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import SensorStreams
from .outage_simulator import OutageScenario, standard_outage_suite
from .pipeline import missing_streams, run_experiment

CONFIGS = {
    "C1": ("odom",),
    "C2": ("odom", "imu"),
    "C3": ("odom", "imu", "lidar"),
    "C4": ("odom", "imu", "lidar", "gps"),
}
CONFIG_LABEL = {
    "C1": "Wheel odometry only",
    "C2": "Wheel odometry + IMU",
    "C3": "Wheel odometry + IMU + LiDAR",
    "C4": "Full fusion with GPS-RTK",
}
COST_TIERS = {
    "C1": {"tier": "Lowest", "rank": 1, "complexity": "1 sensor, built into drive base", "price_note": "qualitative tier"},
    "C2": {"tier": "Low", "rank": 2, "complexity": "2 sensors, IMU time-sync", "price_note": "qualitative tier"},
    "C3": {"tier": "Medium", "rank": 3, "complexity": "3 sensors, LiDAR extrinsics + scan matching compute", "price_note": "qualitative tier"},
    "C4": {"tier": "Highest", "rank": 4, "complexity": "4 sensors, RTK base station / corrections link", "price_note": "qualitative tier"},
}


def effective_sensors(streams: SensorStreams, cfg_id: str) -> tuple[tuple, list]:
    """Sensors actually usable for a config; C4 drops LiDAR if not extracted
    (and says so) instead of pretending it was used."""
    sens = CONFIGS[cfg_id]
    miss = missing_streams(streams, sens)
    if cfg_id == "C4" and miss:
        sens = tuple(s for s in sens if s not in miss)
    return sens, miss


def compare_configs(streams: SensorStreams, outage_durations=(10, 30, 60, 120), n_starts: int = 3,
                    ekf=None, proxy_sigma: float = 0.03) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (overall table, outage table).

    * Overall: each config on the full sequence (C1–C3 never see GPS).
    * Outage robustness: GPS + each dead-reckoning config, under identical
      outage windows (same start times and durations for every config).
    """
    rows, orows = [], []
    t_span = streams.time_span()
    suite = standard_outage_suite(t_span[1] - t_span[0], outage_durations, n_starts=n_starts)
    for cid in CONFIGS:
        sens, miss = effective_sensors(streams, cid)
        base = {"config": cid, "label": CONFIG_LABEL[cid], "sensors_used": "+".join(sens),
                "cost_tier": COST_TIERS[cid]["tier"], "complexity": COST_TIERS[cid]["complexity"]}
        if miss and cid != "C4":
            rows.append({**base, "status": f"NOT RUN — missing stream(s): {', '.join(miss)}"})
            continue
        status = "ok" if not miss else f"ran without {', '.join(miss)} (not available)"
        r = run_experiment(streams, sens, OutageScenario(), ekf=ekf, proxy_sigma=proxy_sigma)
        rows.append({**base, "status": status, **{k: r.metrics[k] for k in
                     ("mean_m", "rmse_m", "max_m", "final_m", "pct_time_within_target", "coverage_r95_pct")}})
        # outage robustness: dead-reckoning part of the config + GPS.
        # (C4's dead-reckoning part is C3, so it is not repeated.)
        if cid == "C4":
            continue
        dr = tuple(s for s in sens if s != "gps")
        for sc in suite:
            ro = run_experiment(streams, dr + ("gps",), sc, ekf=ekf, proxy_sigma=proxy_sigma)
            om = ro.outage_table
            if len(om):
                orows.append({"config": cid, "dead_reckoning": "+".join(dr), **om.iloc[0].to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(orows)


def summarize_outages(otab: pd.DataFrame) -> pd.DataFrame:
    if otab.empty:
        return otab
    return (otab.groupby(["config", "dead_reckoning", "duration_s"])
            .agg(max_err_m=("max_err_m", "mean"), end_err_m=("end_err_m", "mean"),
                 pct_within_target=("pct_within_target", "mean"),
                 time_to_exceed_s=("time_to_exceed_s", "mean"), recovery_s=("recovery_s", "mean"),
                 n=("max_err_m", "size"))
            .reset_index())


def recommend(overall: pd.DataFrame, outage_summary: pd.DataFrame, target_m: float = 0.3,
              outage_s: float = 30.0) -> str:
    """Cheapest configuration whose mean max error during ``outage_s`` outages
    stays within the target. Returns an honest sentence either way."""
    if outage_summary.empty:
        return "No outage results available."
    o = outage_summary[outage_summary.duration_s == outage_s].copy()
    o["rank"] = o.config.map(lambda c: COST_TIERS[c]["rank"])
    ok = o[o.max_err_m <= target_m].sort_values("rank")
    if len(ok):
        c = ok.iloc[0]
        return (f"Cheapest dead-reckoning fallback meeting the {target_m} m target through {outage_s:.0f} s GPS "
                f"outages: {c.config} ({CONFIG_LABEL[c.config]}, cost tier {COST_TIERS[c.config]['tier']}) + GPS.")
    best = o.sort_values("max_err_m").iloc[0]
    return (f"No implemented configuration met the {target_m} m target through {outage_s:.0f} s outages. "
            f"Best was {best.config} ({best.dead_reckoning}) + GPS with mean peak error {best.max_err_m:.2f} m.")
