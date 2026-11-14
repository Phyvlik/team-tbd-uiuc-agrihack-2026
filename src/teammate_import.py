"""Import and verify a teammate's processed trajectory CSV.

Target schema (after normalisation)::

    timestamp     seconds (float)
    reference_x   metres, map frame (RTK reference)
    reference_y   metres
    estimated_x   metres, SAME frame as reference
    estimated_y   metres
    error_m       optional, the file's own error column (kept for cross-check)

The importer *inspects first*: it proposes a column mapping from known
aliases, guesses time/length units from magnitudes, and reports what it
assumed. Pass an explicit ``mapping``/units to override any guess.

Verification recomputes the error and runs the same frame diagnostics as
``baseline_validation`` (start offset, rigid-rotation offset, path-length
ratio) so large errors can be classified as drift vs frame artefact.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .coordinate_utils import fit_se2, path_length

ALIASES = {
    "timestamp": ["timestamp", "time", "t", "stamp", "time_s", "timestamp_s", "sec", "secs", "ros_time",
                  "header_stamp", "time_ns", "timestamp_ns", "%time"],
    "reference_x": ["reference_x", "ref_x", "gt_x", "rtk_x", "x_ref", "x_gt", "x_rtk", "true_x", "truth_x",
                    "groundtruth_x", "ground_truth_x", "tx"],
    "reference_y": ["reference_y", "ref_y", "gt_y", "rtk_y", "y_ref", "y_gt", "y_rtk", "true_y", "truth_y",
                    "groundtruth_y", "ground_truth_y", "ty"],
    "estimated_x": ["estimated_x", "est_x", "x_est", "odom_x", "x_odom", "pred_x", "x_pred", "b1_x", "x_b1",
                    "estimate_x", "wheel_x", "x_hat"],
    "estimated_y": ["estimated_y", "est_y", "y_est", "odom_y", "y_odom", "pred_y", "y_pred", "b1_y", "y_b1",
                    "estimate_y", "wheel_y", "y_hat"],
    "error_m": ["error_m", "error", "err", "position_error", "pos_err", "err_m", "localization_error",
                "error_meters", "abs_error"],
}
LENGTH_FACTORS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9%]+", "_", c.strip().lower()).strip("_")


def propose_mapping(columns) -> dict:
    normed = {_norm(c): c for c in columns}
    mapping = {}
    for target, aliases in ALIASES.items():
        for a in aliases:
            if a in normed:
                mapping[target] = normed[a]
                break
    return mapping


def guess_time_unit(t: np.ndarray) -> str:
    """Guess from magnitude of absolute epoch or from the sample spacing."""
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return "s"
    m = np.nanmedian(np.abs(t))
    if m > 1e17:
        return "ns"
    if m > 1e14:
        return "us"
    if m > 1e11:
        return "ms"
    return "s"


def normalize(df: pd.DataFrame, mapping: dict | None = None, time_unit: str | None = None,
              length_unit: str = "m") -> tuple[pd.DataFrame, list[str]]:
    """Return (normalised DataFrame, list of notes about every assumption)."""
    notes = []
    auto = propose_mapping(df.columns)
    mapping = {**auto, **(mapping or {})}
    required = ["timestamp", "reference_x", "reference_y", "estimated_x", "estimated_y"]
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(f"could not map required columns {missing}. Columns found: {list(df.columns)}. "
                         f"Pass mapping={{'{missing[0]}': '<column name>', ...}}")
    for k, v in mapping.items():
        notes.append(f"{k} <- '{v}'" + ("" if k in auto and auto[k] == v else " (user mapping)"))
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in mapping.items()})
    tu = time_unit or guess_time_unit(out["timestamp"].to_numpy(float))
    f = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[tu]
    out["timestamp"] = out["timestamp"] * f
    notes.append(f"time unit: {tu}" + (" (guessed from magnitude)" if time_unit is None else ""))
    lf = LENGTH_FACTORS[length_unit]
    for c in ["reference_x", "reference_y", "estimated_x", "estimated_y", "error_m"]:
        if c in out:
            out[c] = out[c] * lf
    notes.append(f"length unit: {length_unit} (assumed unless specified)")
    return out, notes


def verify(df: pd.DataFrame) -> dict:
    """Checks + diagnostics on a normalised trajectory table."""
    rep = {"rows": len(df)}
    t = df["timestamp"].to_numpy(float)
    rep["nan_rows"] = int(df[["timestamp", "reference_x", "reference_y", "estimated_x", "estimated_y"]]
                          .isna().any(axis=1).sum())
    d = df.dropna(subset=["timestamp", "reference_x", "reference_y", "estimated_x", "estimated_y"])
    t = d["timestamp"].to_numpy(float)
    dt = np.diff(t)
    rep["timestamps_strictly_increasing"] = bool(np.all(dt > 0))
    rep["duplicate_timestamps"] = int(np.sum(dt == 0))
    rep["backwards_steps"] = int(np.sum(dt < 0))
    rep["median_rate_hz"] = float(1 / np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan
    rep["duration_s"] = float(t.max() - t.min()) if len(t) else 0.0
    rx, ry = d["reference_x"].to_numpy(), d["reference_y"].to_numpy()
    ex, ey = d["estimated_x"].to_numpy(), d["estimated_y"].to_numpy()
    err = np.hypot(ex - rx, ey - ry)
    rep.update({"mean_err_m": float(err.mean()), "rmse_m": float(np.sqrt((err ** 2).mean())),
                "max_err_m": float(err.max()), "final_err_m": float(err[-1]),
                "pct_within_0.3m": float(100 * np.mean(err <= 0.3))})
    if "error_m" in d:
        rep["file_error_vs_recomputed_max_abs_diff_m"] = float(np.nanmax(np.abs(d["error_m"].to_numpy() - err)))
    # frame diagnostics
    rep["start_offset_m"] = float(err[0])
    lr, le = path_length(rx, ry), path_length(ex, ey)
    rep["path_length_ratio_est_over_ref"] = le / lr if lr > 0 else np.nan
    th, tt = fit_se2(np.column_stack([ex, ey]), np.column_stack([rx, ry]))
    rep["rigid_fit_rotation_deg"] = float(np.degrees(th))
    al = np.column_stack([ex, ey]) @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) + tt
    rep["max_err_after_rigid_fit_m_DIAGNOSTIC"] = float(np.max(np.hypot(al[:, 0] - rx, al[:, 1] - ry)))
    findings = []
    if not rep["timestamps_strictly_increasing"]:
        findings.append("Timestamps are not strictly increasing: sort/dedupe before computing drift.")
    if rep["start_offset_m"] > 0.5:
        findings.append(f"Estimate does not start at the reference ({rep['start_offset_m']:.1f} m apart): "
                        "initial position alignment is missing.")
    if abs(rep["rigid_fit_rotation_deg"]) > 10 and rep["max_err_after_rigid_fit_m_DIAGNOSTIC"] < 0.5 * rep["max_err_m"]:
        findings.append(f"A {rep['rigid_fit_rotation_deg']:.0f}° rigid rotation removes most of the error: likely an "
                        "initial-heading / frame mismatch rather than drift.")
    if abs(rep["path_length_ratio_est_over_ref"] - 1) > 0.05:
        findings.append(f"Estimated path is {100 * (rep['path_length_ratio_est_over_ref'] - 1):+.0f}% vs reference: "
                        "speed-scale error (or wrong length units / double integration).")
    if not findings:
        findings.append("No frame/units red flags; large errors would then be genuine drift.")
    rep["findings"] = findings
    return rep


def import_file(path: str | Path, **kw) -> tuple[pd.DataFrame, list[str], dict]:
    df = pd.read_csv(path, comment="#")
    norm, notes = normalize(df, **kw)
    return norm, notes, verify(norm)
