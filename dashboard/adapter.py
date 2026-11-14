"""Data adapter: one UI-facing ``RunBundle`` for any localization run.

Sources
-------
* ``from_experiment`` — the backend pipeline (run_experiment) on a SensorStreams.
* ``from_csv``        — a teammate's run file (schema below). Drop files into
  ``team_runs/`` in this app folder, or ``<backend>/results/runs/``.

Teammate CSV schema (one row per estimate; only the first three are required)
---------------------------------------------------------------------------
    t                       timestamp (s; ns/us/ms epoch auto-detected) — REQUIRED
    est_x, est_y            estimated position, metres, SAME map frame as CitrusFarm gt.csv — REQUIRED
    sequence_id             e.g. 06_14B_Jackal (or put it in the file name: <SEQ>__<method>.csv)
    method / sensor_config  e.g. "odom+imu" (or file name)
    ref_x, ref_y            reference positions (if absent and the sequence is known, the recorded
                            reference is interpolated — labelled as such)
    var_x, var_y, cov_xy    position covariance (m²)            → uncertainty
    r95                     95 % radius (m) if no covariance
    speed, heading          m/s, rad (map frame, CCW from +x)
    gps_available           0/1          gps_quality            any numeric quality metric
    gps_rejected            0/1          gyro_bias              rad/s
    outage                  0/1 flag marking simulated/real GPS outage samples
Optional sidecar ``<file>.json``: {"sequence_id", "method", "sensors", "frame", "length_unit",
"time_unit", "outages_s": [[start_rel, end_rel], ...], "lidar_points": "<path>", "orchard_geometry": "<path>"}

Rules: units/frames/timestamps are checked and every assumption is reported
in ``checks``. Missing fields stay ``None`` (UI: "Not evaluated"); nothing is
inferred from other fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend as be
from src.teammate_import import LENGTH_FACTORS, guess_time_unit, verify  # teammate verification (read-only)

TEAM_DIRS = [be.APP_DIR / "team_runs", be.RESULTS_DIR / "runs"]
ALIASES = {
    "t": ["t", "time", "timestamp", "stamp", "time_s", "time_ns", "timestamp_ns"],
    # no bare "x"/"y" aliases: a column called x could be the reference — require an explicit name
    "est_x": ["est_x", "x_est", "estimated_x", "x_hat"], "est_y": ["est_y", "y_est", "estimated_y", "y_hat"],
    "ref_x": ["ref_x", "gt_x", "rtk_x", "reference_x"], "ref_y": ["ref_y", "gt_y", "rtk_y", "reference_y"],
    "var_x": ["var_x", "cov_xx", "p_xx"], "var_y": ["var_y", "cov_yy", "p_yy"], "cov_xy": ["cov_xy", "p_xy"],
    "r95": ["r95", "radius_95", "uncertainty_95_m"], "speed": ["speed", "v", "vel"], "heading": ["heading", "theta", "yaw"],
    "gps_available": ["gps_available", "gps_ok"], "gps_quality": ["gps_quality", "num_sat", "h_sigma_m"],
    "gps_rejected": ["gps_rejected", "rejected"], "gyro_bias": ["gyro_bias", "b_g", "bias_gz"], "outage": ["outage"],
    "sequence_id": ["sequence_id", "sequence", "seq"], "method": ["method", "sensor_config", "config"],
}


@dataclass
class RunBundle:
    sequence: str
    method: str
    provenance: str                       # computed | imported | synthetic
    t: np.ndarray                         # seconds since sequence start
    est_x: np.ndarray
    est_y: np.ndarray
    ref: pd.DataFrame | None = None       # t (rel s), x, y
    ref_origin: str = ""
    r95: np.ndarray | None = None
    cov: np.ndarray | None = None         # (n, 3): var_x, var_y, cov_xy
    speed: np.ndarray | None = None
    heading: np.ndarray | None = None
    gps_available: np.ndarray | None = None
    gps_quality: np.ndarray | None = None
    gps_rejected: np.ndarray | None = None
    gyro_bias: np.ndarray | None = None
    sensors: str = ""
    outages: list = field(default_factory=list)
    lidar_points: str | None = None
    orchard_geometry: str | None = None
    err: np.ndarray | None = None
    checks: list = field(default_factory=list)
    verify_report: dict | None = None
    source_file: str = ""

    def has(self, name: str) -> bool:
        return getattr(self, name) is not None

    def summary(self) -> dict:
        e = self.err
        return {"sequence": self.sequence, "method": self.method, "provenance": self.provenance,
                "samples": len(self.t), "duration_s": float(self.t[-1] - self.t[0]) if len(self.t) else 0.0,
                "mean_err_m": float(np.nanmean(e)) if e is not None else None,
                "max_err_m": float(np.nanmax(e)) if e is not None else None,
                "pct_within_0.3m": float(100 * np.nanmean(e <= 0.3)) if e is not None else None,
                "uncertainty": "covariance" if self.cov is not None else ("r95" if self.r95 is not None else "Not evaluated"),
                "gps_quality": "yes" if self.gps_quality is not None else "Not evaluated",
                "gyro_bias": "yes" if self.gyro_bias is not None else "Not evaluated",
                "file": self.source_file}


def _r95_from_cov(c):
    a, b, x = c[:, 0], c[:, 1], c[:, 2]
    lam = 0.5 * (a + b) + np.sqrt(0.25 * (a - b) ** 2 + x ** 2)
    return np.sqrt(5.991 * np.maximum(lam, 0))


def from_experiment(key: str, method_label: str, res, provenance: str | None = None) -> RunBundle:
    s = be.load_streams(key)
    t0 = s.time_span()[0]
    est = res.estimate
    ref = s.reference.assign(t=s.reference.t - t0)[["t", "x", "y"]]
    return RunBundle(
        sequence=key, method=method_label, provenance=provenance or ("synthetic" if s.mode == "synthetic" else "computed"),
        t=est.t.to_numpy() - t0, est_x=est.x_ant.to_numpy(), est_y=est.y_ant.to_numpy(), ref=ref,
        ref_origin="recorded RTK reference (gt.csv)" if s.mode != "synthetic" else "synthetic truth",
        r95=est.r95.to_numpy(), cov=est[["var_x", "var_y", "cov_xy"]].to_numpy(), speed=est.v.to_numpy(),
        heading=est.theta.to_numpy(), gps_available=(est.time_since_gps.to_numpy() <= 2.0),
        gps_rejected=est.gps_rejected.to_numpy().astype(bool),
        gyro_bias=est.b_g.to_numpy() if "imu" in res.sensors else None, sensors="+".join(res.sensors),
        outages=[(a - t0, b - t0) for a, b in res.outages_abs],
        err=np.interp(est.t.to_numpy(), res.error.t, res.error.err), checks=["computed by the backend pipeline"])


def _find(cols, target):
    normed = {re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"): c for c in cols}
    for a in ALIASES[target]:
        if a in normed:
            return normed[a]
    return None


def from_csv(path: str | Path, sequence: str | None = None, method: str | None = None) -> RunBundle:
    path = Path(path)
    df = pd.read_csv(path, comment="#")
    side = {}
    sj = path.with_suffix(".json")
    if sj.exists():
        side = json.loads(sj.read_text())
    checks = []
    col = {k: _find(df.columns, k) for k in ALIASES}
    for req in ("t", "est_x", "est_y"):
        if col[req] is None:
            raise ValueError(f"{path.name}: required column '{req}' not found (columns: {list(df.columns)})")
    stem = path.stem.split("__")
    seq = sequence or side.get("sequence_id") or (str(df[col["sequence_id"]].iloc[0]) if col["sequence_id"] else None) \
        or (stem[0] if len(stem) > 1 else None)
    meth = method or side.get("method") or (str(df[col["method"]].iloc[0]) if col["method"] else None) \
        or (stem[1] if len(stem) > 1 else path.stem)
    df = df.sort_values(col["t"])
    t = pd.to_numeric(df[col["t"]], errors="coerce").to_numpy(float)
    tu = side.get("time_unit") or guess_time_unit(t)
    t = t * {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[tu]
    checks.append(f"time unit: {tu}" + ("" if side.get("time_unit") else " (guessed from magnitude)"))
    lu = side.get("length_unit", "m")
    lf = LENGTH_FACTORS[lu]
    checks.append(f"length unit: {lu}" + ("" if "length_unit" in side else " (assumed)"))
    gin = side.get("gps_input")
    if gin == "proxy":
        checks.append("⚠ GPS input was a SIMULATED proxy (from the reference): with-GPS errors are not real-world results")
    elif gin == "recorded":
        checks.append("GPS input: recorded receiver fixes")
    else:
        checks.append("GPS input not declared (sidecar 'gps_input': recorded | proxy)")
    g = lambda k: None if col[k] is None else pd.to_numeric(df[col[k]], errors="coerce").to_numpy(float)  # noqa: E731
    ex, ey = g("est_x") * lf, g("est_y") * lf
    # which body point do the estimates describe? CitrusFarm gt.csv measures the GPS antenna,
    # which sits 0.18 m behind base_link. Convert only when the heading is provided; never guess it.
    point = side.get("point", "antenna")
    if point == "base_link":
        if col["heading"] is not None:
            lx, ly = be.CITRUSFARM_GPS_LEVER_ARM if seq and seq != be.SYNTHETIC else (0.0, 0.0)
            th = g("heading")
            ex, ey = ex + np.cos(th) * lx - np.sin(th) * ly, ey + np.sin(th) * lx + np.cos(th) * ly
            checks.append(f"estimates at base_link → shifted to the GPS antenna with lever arm {lx, ly} m using heading")
        else:
            checks.append("⚠ estimates are at base_link but no heading column: compared to the antenna-point "
                          "reference WITHOUT lever-arm correction (up to 0.18 m bias)")
    else:
        checks.append("estimates assumed to describe the GPS antenna point (sidecar 'point' to override)")
    # time alignment with the sequence
    ref, ref_origin = None, ""
    known = seq in be.source_keys() if seq else False
    if known:
        s = be.load_streams(seq)
        t0, t1 = s.time_span()
        if np.nanmedian(t) > 1e8 and t0 > 1e8:
            t = t - t0
            checks.append("timestamps are epoch → converted to seconds since sequence start")
        elif np.nanmedian(t) > 1e8:
            checks.append("⚠ epoch timestamps but the sequence uses relative time — left unchanged")
        else:
            checks.append("timestamps already relative to the sequence start (assumed)")
        overlap = (min(t1 - t0, np.nanmax(t)) - max(0, np.nanmin(t))) / max(np.nanmax(t) - np.nanmin(t), 1e-9)
        checks.append(f"time overlap with {seq}: {100 * overlap:.0f} %")
        if col["ref_x"] is not None:
            ref = pd.DataFrame({"t": t, "x": g("ref_x") * lf, "y": g("ref_y") * lf})
            ref_origin = "reference columns in the file"
        else:
            r = s.reference
            ref = pd.DataFrame({"t": r.t - t0, "x": r.x, "y": r.y})
            ref_origin = "recorded reference of the sequence (interpolated)"
    elif col["ref_x"] is not None:
        ref = pd.DataFrame({"t": t, "x": g("ref_x") * lf, "y": g("ref_y") * lf})
        ref_origin = "reference columns in the file"
        checks.append("⚠ unknown sequence — cannot check time alignment")
    else:
        checks.append("⚠ unknown sequence and no reference columns — error Not evaluated")
    err, rep = None, None
    if ref is not None:
        rx, ry = np.interp(t, ref.t, ref.x), np.interp(t, ref.t, ref.y)
        err = np.hypot(ex - rx, ey - ry)
        rep = verify(pd.DataFrame({"timestamp": t, "reference_x": rx, "reference_y": ry, "estimated_x": ex,
                                   "estimated_y": ey}).dropna())
        checks += [f"frame/units check: {f}" for f in rep["findings"]]
    cov = None
    if all(col[k] is not None for k in ("var_x", "var_y", "cov_xy")):
        cov = np.c_[g("var_x"), g("var_y"), g("cov_xy")] * lf ** 2
        if np.any(cov[:, 0] < 0) or np.any(cov[:, 1] < 0):
            checks.append("⚠ negative variances in covariance columns")
    r95 = g("r95") * lf if col["r95"] is not None else (_r95_from_cov(cov) if cov is not None else None)
    outages = side.get("outages_s") or []
    if not outages and col["outage"] is not None:
        o = g("outage") > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, o.astype(int), 0]))
        outages = [(float(t[a]), float(t[min(b, len(t) - 1)])) for a, b in zip(edges[::2], edges[1::2])]
    return RunBundle(
        sequence=seq or "unknown", method=str(meth), provenance="imported", t=t, est_x=ex, est_y=ey, ref=ref,
        ref_origin=ref_origin, r95=r95, cov=cov, speed=g("speed"), heading=g("heading"),
        gps_available=None if col["gps_available"] is None else g("gps_available") > 0.5,
        gps_quality=g("gps_quality"), gps_rejected=None if col["gps_rejected"] is None else g("gps_rejected") > 0.5,
        gyro_bias=g("gyro_bias"), sensors=side.get("sensors", str(meth)), outages=outages,
        lidar_points=side.get("lidar_points"), orchard_geometry=side.get("orchard_geometry"), err=err,
        checks=checks, verify_report=rep, source_file=str(path))


def discover() -> list[Path]:
    out = []
    for d in TEAM_DIRS:
        if d.exists():
            out += sorted(d.glob("*.csv"))
    return out


def load_team_runs(sequence: str | None = None) -> tuple[list[RunBundle], list[tuple[str, str]]]:
    runs, errors = [], []
    for p in discover():
        try:
            b = from_csv(p)
            if sequence is None or b.sequence == sequence:
                runs.append(b)
        except Exception as e:
            errors.append((p.name, str(e)))
    return runs, errors
