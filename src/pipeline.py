"""One call that runs an experiment end-to-end: GPS input -> outages -> EKF ->
reliability monitor -> metrics. Used by the dashboard and the batch scripts so
every number shown in the UI comes from the same code path."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from .data_loader import SensorStreams
from .evaluation import error_series, outage_metrics, summary_metrics, coverage
from .localization import initial_pose_from_track
from .outage_simulator import OutageScenario, apply_scenario, gps_proxy_from_reference
from .reliability_monitor import MonitorConfig, classify, evaluate_warnings
from .sensor_fusion import EKFConfig, run_ekf

SENSOR_STREAM = {"odom": "odom", "imu": "imu", "lidar": "lidar_odom", "gps": "gps"}


@dataclass
class ExperimentResult:
    estimate: pd.DataFrame
    error: pd.DataFrame
    states: pd.DataFrame
    gps_log: pd.DataFrame | None
    metrics: dict
    outage_table: pd.DataFrame
    warnings: dict
    gps_source: str
    sensors: tuple
    outages_abs: list = field(default_factory=list)
    gps_events: pd.DataFrame | None = None


def gps_input(streams: SensorStreams, proxy_sigma: float = 0.03, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """Live GPS stream for the estimator and a label describing where it came from."""
    if streams.gps is not None and len(streams.gps):
        src = str(streams.gps["source"].iloc[0]) if "source" in streams.gps else "raw"
        return streams.gps, src
    if streams.reference is None:
        raise ValueError("no GPS stream and no reference to build a proxy from")
    return gps_proxy_from_reference(streams.reference, sigma=proxy_sigma, seed=seed), "proxy_from_reference"


def missing_streams(streams: SensorStreams, sensors) -> list[str]:
    av = streams.available()
    return [s for s in sensors if s != "gps" and not av.get(SENSOR_STREAM[s], False)]


def run_experiment(streams: SensorStreams, sensors=("odom", "imu", "gps"), scenario: OutageScenario | None = None,
                   ekf: EKFConfig | None = None, monitor: MonitorConfig | None = None,
                   proxy_sigma: float = 0.03, seed: int = 0) -> ExperimentResult:
    missing = missing_streams(streams, sensors)
    if missing:
        raise ValueError(f"sensor stream(s) not available for {streams.name}: {missing}")
    scenario = scenario or OutageScenario()
    gps_all, gps_src = gps_input(streams, proxy_sigma, seed)
    t0 = float(streams.reference["t"].iloc[0])
    gps_used, gps_log = apply_scenario(gps_all, scenario, t0)
    # initial pose from the first GPS fixes (same for every configuration)
    init = initial_pose_from_track(gps_all, streams.reference_lever_arm, odom=streams.odom)
    cfg = replace(ekf or EKFConfig(), sensors=tuple(sensors), lever_arm=streams.reference_lever_arm)
    res = run_ekf(init, odom=streams.odom, imu=streams.imu, lidar_odom=streams.lidar_odom,
                  gps=gps_used if "gps" in sensors else None, config=cfg,
                  t_end=float(streams.reference["t"].iloc[-1]))
    states = classify(res.estimate, monitor)
    err = error_series(res.estimate, streams.reference)
    outs = scenario.absolute(t0)
    metrics = summary_metrics(err)
    metrics["coverage_r95_pct"] = 100 * coverage(err, res.estimate)
    metrics["gps_rejected"] = res.counts.get("gps_rejected", 0)
    return ExperimentResult(res.estimate, err, states, gps_log, metrics, outage_metrics(err, outs),
                            evaluate_warnings(states, err, outages_abs=outs), gps_src, tuple(sensors), outs,
                            res.events)
