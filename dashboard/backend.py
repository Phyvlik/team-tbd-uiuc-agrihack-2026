"""Read-only bridge to the team's backend repository.

The algorithms, scripts and datasets live in the teammates' repo
(``UIUC-AgriRobotics-2026``). This module puts that repo on ``sys.path`` and
calls its public functions. Nothing here re-implements estimation, and nothing
writes into the backend repo. Point ``AGRI_BACKEND_DIR`` at another checkout
to use it instead.

Every data source is tagged with a provenance so the UI can label it:

* ``real``      recorded CitrusFarm sensor streams
* ``batch``     numbers from the teammates' ``results/*.csv`` (offline batch runs)
* ``synthetic`` the backend's labelled synthetic orchard
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(os.environ.get("AGRI_BACKEND_DIR", APP_DIR.parent / "UIUC-AgriRobotics-2026")).resolve()
CACHE_DIR = APP_DIR / ".cache"

if not (BACKEND_DIR / "src" / "pipeline.py").exists():
    raise ImportError(
        f"Backend repo not found at {BACKEND_DIR}. Clone UIUC-AgriRobotics-2026 next to this folder "
        f"or set AGRI_BACKEND_DIR.")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --- teammate backend (imported, never modified) ----------------------------
from src.data_loader import (CITRUSFARM_GPS_LEVER_ARM, CITRUSFARM_SEQUENCES, PROCESSED_DIR, RAW_DIR,  # noqa: E402
                             DataUnavailableError,
                             SensorStreams, build_citrusfarm_sequence, list_processed_sequences, load_processed)
from src.evaluation import TARGET_M  # noqa: E402
from src.gps_risk import canopy_quality_relationship, quality_variation_check, risk_map  # noqa: E402
from src.outage_simulator import OUTAGE_DURATIONS_S, OutageScenario, standard_outage_suite  # noqa: E402
from src.pipeline import ExperimentResult, missing_streams, run_experiment  # noqa: E402
from src.reliability_monitor import (DEGRADED, STOP, TRUSTED, MonitorConfig, transitions)  # noqa: E402
from src.sensor_comparison import (CONFIG_LABEL, CONFIGS, COST_TIERS, compare_configs,  # noqa: E402
                                   summarize_outages)
from src.sensor_fusion import ellipse_points  # noqa: E402
from src.synthetic_data import SYNTHETIC_LABEL, make_synthetic_sequence  # noqa: E402
from src.worker_safety import (MESSAGES, SafetyConfig, evaluate_safety, place_workers_along_route,  # noqa: E402
                               score_safety)

RESULTS_DIR = BACKEND_DIR / "results"
SYNTHETIC = "SYNTHETIC"

SENSOR_NAMES = {"odom": "Wheel odometry", "imu": "IMU gyroscope", "lidar": "LiDAR odometry", "gps": "GPS-RTK"}
STREAM_OF = {"odom": "odom", "imu": "imu", "lidar": "lidar_odom", "gps": "gps"}

# Localization methods offered to the operator (all keep GPS when it is available).
METHODS = {
    "odom+gps": ("odom", "gps"),
    "odom+imu+gps": ("odom", "imu", "gps"),
    "odom+imu+lidar+gps": ("odom", "imu", "lidar", "gps"),
}
METHOD_LABEL = {
    "odom+gps": "Odometry + GPS",
    "odom+imu+gps": "Odometry + IMU + GPS",
    "odom+imu+lidar+gps": "Full fusion + LiDAR",
}
# Dead-reckoning cores used while GPS is out (C1–C3 of the team plan).
FALLBACK = {"C1": ("odom",), "C2": ("odom", "imu"), "C3": ("odom", "imu", "lidar")}


@dataclass(frozen=True)
class SourceInfo:
    key: str
    label: str
    provenance: str          # "real" | "synthetic"
    origin: str              # how it was loaded
    streams_text: str


# ------------------------------------------------------------------ sources --

def _raw_has_extracted(seq: str) -> bool:
    d = RAW_DIR / seq
    return any((d / f"{s}.csv").exists() for s in ("imu", "gps", "gps_quality", "lidar_odom", "canopy"))


def source_keys() -> list[str]:
    """Synthetic demo + every CitrusFarm sequence that has processed or raw-extracted data."""
    real = sorted(set(list_processed_sequences()) | {s for s in CITRUSFARM_SEQUENCES if _raw_has_extracted(s)})
    return [SYNTHETIC] + real


def default_source() -> str:
    keys = source_keys()
    return "06_14B_Jackal" if "06_14B_Jackal" in keys else keys[0]


@st.cache_resource(show_spinner="Loading sensor streams…")
def load_streams(key: str) -> SensorStreams:
    """Synthetic → backend generator. Real → raw extracted streams if present
    (via the backend's own builder), else the processed cache. Never mixes in
    synthetic data for a real sequence."""
    if key == SYNTHETIC:
        return make_synthetic_sequence()
    if _raw_has_extracted(key):
        try:
            s = build_citrusfarm_sequence(key)
            s.meta["ui_origin"] = "raw extracted streams (built read-only with data_loader.build_citrusfarm_sequence)"
            return s
        except Exception as e:  # fall back to processed cache, and say so
            if key not in list_processed_sequences():
                raise DataUnavailableError(f"{key}: raw build failed ({e}) and no processed cache") from e
            s = load_processed(key)
            s.meta["ui_origin"] = f"processed cache (raw build failed: {e})"
            return s
    s = load_processed(key)
    s.meta["ui_origin"] = "processed cache (data/processed)"
    return s


def source_info(key: str) -> SourceInfo:
    if key == SYNTHETIC:
        return SourceInfo(key, "Synthetic orchard demo", "synthetic", "backend synthetic_data generator",
                          "odometry, IMU, LiDAR, GPS, canopy — all simulated")
    s = load_streams(key)
    av = s.available()
    have = [SENSOR_NAMES[k] for k in ("odom", "imu", "lidar") if av.get(STREAM_OF[k])]
    gps = "raw RTK fixes" if av.get("gps") else "GPS proxy (reference + 3 cm noise, simulated)"
    return SourceInfo(key, f"CitrusFarm {key}", "real", s.meta.get("ui_origin", ""), ", ".join(have) + f" · {gps}")


def source_label(key: str) -> str:
    if key == SYNTHETIC:
        return "Synthetic orchard (demo)"
    short = key.replace("_Jackal", "")
    if _raw_has_extracted(key):
        return f"{short} · real · odom + IMU + GPS"
    return f"{short} · real · odom only"


def methods_for(streams: SensorStreams) -> list[str]:
    return [m for m, sens in METHODS.items() if not missing_streams(streams, sens)]


def time_span(key: str) -> tuple[float, float]:
    return load_streams(key).time_span()


# -------------------------------------------------------------- experiments --

@st.cache_data(show_spinner="Running the team's EKF + integrity monitor…", max_entries=96)
def experiment(key: str, sensors: tuple, outages: tuple = (), jumps: tuple = (),
               trusted_r95: float = 0.30, stop_r95: float = 1.00) -> ExperimentResult:
    """One run of the backend pipeline (src.pipeline.run_experiment)."""
    sc = OutageScenario(outages=[tuple(o) for o in outages], multipath_jumps=[tuple(j) for j in jumps])
    return run_experiment(load_streams(key), tuple(sensors), sc,
                          monitor=MonitorConfig(trusted_r95=trusted_r95, stop_r95=stop_r95))


def _disk_cache(name: str, parts: dict) -> Path:
    h = hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}_{h}.csv"


@st.cache_data(show_spinner=False, max_entries=16)
def live_config_comparison(key: str, n_starts: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1–C4 on the selected source with the backend's compare_configs.
    Persisted to .cache/ in THIS app folder so restarts are instant."""
    s = load_streams(key)
    stamp = {"key": key, "n": n_starts, "av": s.available(), "v": 2}
    p_over, p_out = _disk_cache("overall", stamp), _disk_cache("outage", stamp)
    if p_over.exists() and p_out.exists():
        return pd.read_csv(p_over), pd.read_csv(p_out)
    overall, otab = compare_configs(s, n_starts=n_starts)
    overall.to_csv(p_over, index=False)
    otab.to_csv(p_out, index=False)
    return overall, otab


def live_comparison_cached(key: str, n_starts: int = 2) -> bool:
    s = load_streams(key)
    stamp = {"key": key, "n": n_starts, "av": s.available(), "v": 2}
    return _disk_cache("overall", stamp).exists() and _disk_cache("outage", stamp).exists()


# ------------------------------------------------------------ batch results --

@st.cache_data(show_spinner=False)
def batch_results() -> dict[str, pd.DataFrame | None]:
    """The teammates' result tables (scripts/run_experiments.py). Missing → None."""
    out = {}
    for name in ("config_comparison", "outage_robustness", "warning_performance", "baseline_validation"):
        f = RESULTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(f) if f.exists() else None
    return out


def results_generated() -> str | None:
    r = batch_results().get("config_comparison")
    if r is None or "generated" not in r:
        return None
    return str(r["generated"].dropna().iloc[0]) if r["generated"].notna().any() else None


# ------------------------------------------------------------- calibration --
# The EKF's 95 % uncertainty has NOT been validated against ground truth.
# See docs/UNCERTAINTY_CALIBRATION.md and scripts/calibration_audit.py. Nothing in
# the app rescales or inflates the estimator's covariance.
CALIBRATION_STATUS = "Uncalibrated"
CALIBRATION_DOC = "docs/UNCERTAINTY_CALIBRATION.md"


def calibration_audit(key: str) -> dict | None:
    """Results of scripts/calibration_audit.py for this sequence, if it has been run."""
    p = CACHE_DIR / f"calibration_audit_{key}.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def reference_caveat(key: str, gps_source: str | None) -> str:
    """Known issue with errors measured against gt.csv for this run ('' if none known)."""
    if gps_source == "proxy_from_reference":
        return ("Live GPS here is a SIMULATED proxy built from the reference (+3 cm white noise), so errors and "
                "uncertainty coverage with GPS available are not real-world results.")
    a = calibration_audit(key)
    if a and "y_scale_fit" in a and gps_source not in (None, "synthetic"):
        f = a["y_scale_fit"]
        return (f"Recorded GPS fixes and gt.csv differ by a position-proportional {f['k_percent']:.2f} % scale along "
                f"gt's y axis (up to {f['max_offset_on_route_m']:.2f} m on this route; R² {f['r2']:.3f}). Errors vs gt "
                f"with GPS available include this reference-frame discrepancy until the data owner resolves it.")
    return ""


# ------------------------------------------------------------------ helpers --

def at_time(df: pd.DataFrame, t_abs: float) -> int:
    """Index of the last row with t <= t_abs (0 if before the start)."""
    return max(int(np.searchsorted(df["t"].to_numpy(), t_abs, side="right") - 1), 0)


def finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False
