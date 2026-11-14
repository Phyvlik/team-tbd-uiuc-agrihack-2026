"""Data model and loaders.

Every data source (synthetic demo, CitrusFarm bags, teammate CSVs) is
normalised into one :class:`SensorStreams` container of pandas DataFrames.
Column names, units and frames are fixed here and documented in
``docs/DATA_INTERFACE.md``.

Canonical streams (all ``t`` = seconds, float64, Unix/ROS epoch for real data,
seconds-from-start for synthetic data; strictly increasing):

=============  ==========================================================
reference      t, x, y            map-frame metres. EVALUATION ONLY.
odom           t, v, w            wheel odometry body twist (m/s, rad/s)
imu            t, wz              gyro yaw rate (rad/s)
gps            t, x, y, sigma     live GPS fixes in map frame, sigma = 1-σ
                                  horizontal std (m); optional fix_status
lidar_odom     t, v, w            scan-matching body twist (m/s, rad/s)
gps_quality    t, x, y, + metrics receiver quality along the route (SP6)
canopy         t, x, y, ndvi      canopy index along the route (SP6)
=============  ==========================================================

``reference`` is never passed to an estimator — see ``sensor_fusion.run_ekf``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .coordinate_utils import quat_to_yaw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CITRUSFARM_SEQUENCES = [
    "01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal", "04_13D_Jackal",
    "05_13D_Jackal", "06_14B_Jackal", "07_14B_Jackal",
]

# GPS antenna position in the robot body frame (base_link), from the official
# CitrusFarm calibration files (CAD-derived):
#   05-baselink-lidar-result.txt : base_link -> lidar  = (+0.0400, 0, +0.3787)
#   04-lidar-gps-result.txt      : lidar -> GPS-RTK    = (-0.220, 0, +0.153)
# => base_link -> GPS antenna = (-0.180, 0.000, +0.5317) m
CITRUSFARM_GPS_LEVER_ARM = (-0.180, 0.0)

STREAM_NAMES = ["reference", "odom", "imu", "gps", "lidar_odom", "gps_quality", "canopy"]
REQUIRED_COLUMNS = {
    "reference": ["t", "x", "y"],
    "odom": ["t", "v", "w"],
    "imu": ["t", "wz"],
    "gps": ["t", "x", "y", "sigma"],
    "lidar_odom": ["t", "v", "w"],
    "gps_quality": ["t"],
    "canopy": ["t", "ndvi"],
}


class DataUnavailableError(RuntimeError):
    """Raised when requested real data is missing. Never caught silently."""


@dataclass
class SensorStreams:
    name: str
    mode: str  # "synthetic" | "citrusfarm" | "imported"
    reference: pd.DataFrame | None = None
    odom: pd.DataFrame | None = None
    imu: pd.DataFrame | None = None
    gps: pd.DataFrame | None = None
    lidar_odom: pd.DataFrame | None = None
    gps_quality: pd.DataFrame | None = None
    canopy: pd.DataFrame | None = None
    # body-frame (dx, dy) of the point the reference/GPS positions refer to
    reference_lever_arm: tuple = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def available(self) -> dict[str, bool]:
        return {s: (getattr(self, s) is not None and len(getattr(self, s)) > 0) for s in STREAM_NAMES}

    def validate(self) -> list[str]:
        """Return a list of problems (empty list = OK)."""
        problems = []
        for s in STREAM_NAMES:
            df = getattr(self, s)
            if df is None:
                continue
            missing = [c for c in REQUIRED_COLUMNS[s] if c not in df.columns]
            if missing:
                problems.append(f"{s}: missing columns {missing}")
                continue
            t = df["t"].to_numpy()
            if len(t) > 1 and np.any(np.diff(t) <= 0):
                problems.append(f"{s}: timestamps not strictly increasing")
            if df[REQUIRED_COLUMNS[s]].isna().any().any():
                problems.append(f"{s}: NaN values in required columns")
        return problems

    def time_span(self) -> tuple[float, float]:
        ref = self.reference if self.reference is not None else self.odom
        return float(ref["t"].iloc[0]), float(ref["t"].iloc[-1])


def _clean_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop duplicate timestamps (keeps first)."""
    return df.sort_values("t").drop_duplicates("t", keep="first").reset_index(drop=True)


# ----------------------------------------------------------------------------
# CitrusFarm raw files
# ----------------------------------------------------------------------------

def load_gt_csv(path: str | Path) -> pd.DataFrame:
    """Load an official CitrusFarm ``gt.csv``.

    Verified format (all seven sequences, downloaded 2026-11-13)::

        # ground truth trajectory
        # file: '06_14B_Jackal/gt.bag'
        # timestamp,tx,ty,tz,qx,qy,qz,qw

    tx/ty/tz are local metres with the origin at the first sample. The
    quaternion is all zeros — **the reference carries no heading**.
    """
    df = pd.read_csv(path, comment="#", header=None,
                     names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
    df = _clean_time(df)
    return df[["t", "x", "y", "z"]]


def read_odom_bag(path: str | Path, topic: str = "/jackal_velocity_controller/odom") -> pd.DataFrame:
    """Extract wheel odometry (nav_msgs/Odometry) from a CitrusFarm ``odom_*.bag``.

    Returns t (header stamp, s), v (twist.linear.x), w (twist.angular.z), and the
    controller's own integrated pose (x_ctrl, y_ctrl, yaw_ctrl) for cross-checks.
    """
    from rosbags.highlevel import AnyReader

    rows = []
    with AnyReader([Path(path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise DataUnavailableError(f"topic {topic} not in {path}; topics: "
                                       f"{sorted({c.topic for c in reader.connections})}")
        for conn, _ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            st = m.header.stamp
            p, q = m.pose.pose.position, m.pose.pose.orientation
            rows.append((st.sec + st.nanosec * 1e-9, m.twist.twist.linear.x, m.twist.twist.linear.y,
                         m.twist.twist.angular.z, p.x, p.y, q.x, q.y, q.z, q.w))
    a = np.array(rows)
    df = pd.DataFrame({"t": a[:, 0], "v": a[:, 1], "vy": a[:, 2], "w": a[:, 3],
                       "x_ctrl": a[:, 4], "y_ctrl": a[:, 5],
                       "yaw_ctrl": quat_to_yaw(a[:, 6], a[:, 7], a[:, 8], a[:, 9])})
    return _clean_time(df)


def find_odom_bag(seq_dir: Path) -> Path | None:
    bags = sorted(seq_dir.glob("odom_*.bag"))
    return bags[0] if bags else None


# ----------------------------------------------------------------------------
# Processed cache  (data/processed/<sequence>/<stream>.csv + meta.json)
# ----------------------------------------------------------------------------

def save_processed(streams: SensorStreams, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or PROCESSED_DIR / streams.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in STREAM_NAMES:
        df = getattr(streams, s)
        if df is not None:
            df.to_csv(out_dir / f"{s}.csv", index=False, float_format="%.9f")
    meta = dict(streams.meta, name=streams.name, mode=streams.mode,
                reference_lever_arm=list(streams.reference_lever_arm))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def list_processed_sequences() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(p.name for p in PROCESSED_DIR.iterdir()
                  if (p / "meta.json").exists() and (p / "reference.csv").exists())


def load_processed(name: str, base: Path | None = None) -> SensorStreams:
    d = (base or PROCESSED_DIR) / name
    if not (d / "meta.json").exists():
        raise DataUnavailableError(
            f"No processed data for '{name}' in {d}. Run: python scripts/extract_citrusfarm.py {name}")
    meta = json.loads((d / "meta.json").read_text())
    kw = {}
    for s in STREAM_NAMES:
        f = d / f"{s}.csv"
        kw[s] = pd.read_csv(f) if f.exists() else None
    st = SensorStreams(name=meta.get("name", name), mode=meta.get("mode", "citrusfarm"),
                       reference_lever_arm=tuple(meta.get("reference_lever_arm", (0.0, 0.0))),
                       meta=meta, **kw)
    problems = st.validate()
    if problems:
        raise DataUnavailableError(f"Processed data for {name} failed validation: {problems}")
    return st


def build_citrusfarm_sequence(seq: str, raw_dir: Path | None = None) -> SensorStreams:
    """Build SensorStreams for one CitrusFarm sequence from raw files.

    Uses whatever verified raw inputs exist: gt.csv (required), odom bag
    (required), plus imu/gps/gps_quality CSVs produced by
    ``scripts/extract_base_bag.py`` when available. Missing optional streams
    stay ``None`` — they are never synthesised in real-data mode.
    """
    seq_dir = (raw_dir or RAW_DIR) / seq
    gt_path = seq_dir / "gt.csv"
    if not gt_path.exists():
        raise DataUnavailableError(f"missing {gt_path}")
    bag = find_odom_bag(seq_dir)
    if bag is None:
        raise DataUnavailableError(f"missing odom_*.bag in {seq_dir}")
    gt = load_gt_csv(gt_path)
    odom = read_odom_bag(bag)

    extra = {}
    for s in ["imu", "gps", "gps_quality", "lidar_odom", "canopy"]:
        f = seq_dir / f"{s}.csv"
        if f.exists():
            extra[s] = _clean_time(pd.read_csv(f))

    meta = {
        "source_files": [str(gt_path.relative_to(PROJECT_ROOT)), str(bag.relative_to(PROJECT_ROOT))]
        + [str((seq_dir / f"{s}.csv").relative_to(PROJECT_ROOT)) for s in extra],
        "reference_note": "gt.csv post-processed GPS-RTK at the GPS antenna; local metres; no heading.",
        "gt_rate_hz": float(1.0 / np.median(np.diff(gt["t"]))),
        "odom_rate_hz": float(1.0 / np.median(np.diff(odom["t"]))),
        "gt_t0": float(gt["t"].iloc[0]), "gt_t1": float(gt["t"].iloc[-1]),
        "odom_t0": float(odom["t"].iloc[0]), "odom_t1": float(odom["t"].iloc[-1]),
    }
    return SensorStreams(name=seq, mode="citrusfarm", reference=gt, odom=odom,
                         reference_lever_arm=CITRUSFARM_GPS_LEVER_ARM, meta=meta, **extra)
