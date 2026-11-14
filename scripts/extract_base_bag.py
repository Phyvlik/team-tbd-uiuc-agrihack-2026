"""Extract IMU, raw GPS fixes and GPS receiver state from a CitrusFarm base_*.bag.

The bag is streamed (local file or the public S3 URL) and LiDAR is skipped, so
only a few MB are written to disk.

Usage:
    python scripts/extract_base_bag.py 06_14B_Jackal                 # stream all base bags from S3
    python scripts/extract_base_bag.py 06_14B_Jackal --max-mb 300    # partial: first 300 MB only (inspection)
    python scripts/extract_base_bag.py 06_14B_Jackal --local path/to/base_x.bag

Writes data/raw/<SEQ>/{imu.csv, gps_raw.csv, gps.csv, gps_quality.csv, base_extract_report.json}.
Afterwards re-run scripts/validate_baseline.py <SEQ> to rebuild data/processed/<SEQ>.

Frames:
  * imu.csv      t = header stamp (s); wz = angular_velocity.z (rad/s). The sign/axis is checked
                 against wheel yaw rate and reported (IMU mounting is not assumed).
  * gps_raw.csv  lat/lon/alt, NavSatFix status, covariance diagonal.
  * gps.csv      map-frame x,y in the gt.csv frame: WGS84 -> local ENU about the fix that coincides
                 with gt.csv's first sample, then rotated by GT_FRAME_ROT_DEG = -90 deg.
                 VERIFIED on 06_14B (2026-11-13): gt.csv is ENU rotated -90.00 deg (x = north,
                 y = west); after the rotation raw fixes match gt.csv to median 2.7 cm / max 7 cm.
                 The residual and a free rotation fit are re-reported for every sequence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coordinate_utils import fit_se2, interp_series, latlon_to_enu, rot2  # noqa: E402
from src.data_loader import RAW_DIR, find_odom_bag, load_gt_csv, read_odom_bag  # noqa: E402
from src.rosbag_stream import BagStream, Ros1Decoder  # noqa: E402

S3 = "https://ucr-robotics.s3.amazonaws.com"
GT_FRAME_ROT_DEG = -90.0
TOPICS = {"/microstrain/imu/data", "/piksi/navsatfix_best_fix", "/piksi/debug/receiver_state"}


class Limited:
    """File wrapper that stops after max_bytes (for partial inspection) and counts bytes."""

    def __init__(self, f, max_bytes=None):
        self.f, self.max, self.n, self.t0 = f, max_bytes, 0, time.time()

    def read(self, k):
        if self.max is not None:
            k = min(k, self.max - self.n)
            if k <= 0:
                return b""
        b = self.f.read(k)
        self.n += len(b)
        if self.n // (100 * 2 ** 20) != (self.n - len(b)) // (100 * 2 ** 20):
            print(f"  ... {self.n / 2 ** 20:.0f} MB ({self.n / 2 ** 20 / (time.time() - self.t0 + 1e-9):.1f} MB/s)",
                  flush=True)
        return b


def list_base_bags(seq: str) -> list[str]:
    xml = urllib.request.urlopen(f"{S3}/?list-type=2&prefix=citrus-farm-dataset/{seq}/base_").read().decode()
    return sorted(re.findall(r"<Key>(.*?)</Key>", xml))


def flat_numeric(msg, prefix=""):
    """Scalar numeric fields of a message (and mean/len of numeric arrays)."""
    out = {}
    for k in getattr(msg, "__dataclass_fields__", {}):
        v = getattr(msg, k)
        if k == "header" or k.isupper():
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[prefix + k] = float(v)
        elif isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.ndim == 1:
            out[prefix + k + "_len"] = len(v)
            if len(v):
                out[prefix + k + "_mean"] = float(np.mean(v))
        elif isinstance(v, str):
            out[prefix + k] = v
    return out


def extract(seq: str, sources: list, max_mb: float | None):
    dec = Ros1Decoder()
    imu, fix, rx = [], [], []
    report = {"sequence": seq, "sources": [], "partial": max_mb is not None}
    for src in sources:
        print(f"[stream] {src}")
        f = open(src, "rb") if Path(str(src)).exists() else urllib.request.urlopen(src)
        lim = Limited(f, None if max_mb is None else int(max_mb * 2 ** 20))
        for topic, mtype, _tr, raw, conn in BagStream(lim, TOPICS):
            m = dec.decode(raw, conn)
            st = m.header.stamp
            t = st.sec + st.nanosec * 1e-9
            if topic == "/microstrain/imu/data":
                imu.append((t, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
            elif topic == "/piksi/navsatfix_best_fix":
                c = np.asarray(m.position_covariance)
                fix.append((t, m.latitude, m.longitude, m.altitude, m.status.status, m.status.service,
                            c[0], c[4], c[8], m.position_covariance_type))
            else:
                rx.append({"t": t, **flat_numeric(m)})
        report["sources"].append({"source": str(src), "mb_read": lim.n / 2 ** 20})
        f.close()
    return imu, fix, rx, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--local", nargs="*", help="local base_*.bag path(s) instead of S3")
    ap.add_argument("--max-mb", type=float, default=None, help="stop after this many MB per bag (inspection)")
    ap.add_argument("--imu-rate", type=float, default=50.0, help="average IMU into bins of this rate (Hz)")
    a = ap.parse_args()
    seq_dir = RAW_DIR / a.seq
    # partial (inspection) runs must never be mistaken for complete streams by the loader
    out = seq_dir / "partial_inspection" if a.max_mb is not None else seq_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = a.local or [f"{S3}/{k}" for k in list_base_bags(a.seq)]
    imu, fix, rx, report = extract(a.seq, sources, a.max_mb)

    # ---- IMU --------------------------------------------------------------
    imu = pd.DataFrame(imu, columns=["t", "wx", "wy", "wz", "ax", "ay", "az"]).sort_values("t").drop_duplicates("t")
    report["imu_raw_rate_hz"] = float(1 / np.median(np.diff(imu.t))) if len(imu) > 1 else None
    if len(imu):
        b = np.floor((imu.t - imu.t.iloc[0]) * a.imu_rate)
        imu_b = imu.groupby(b).mean().reset_index(drop=True)
        # axis / sign check against wheel yaw rate
        odom = read_odom_bag(find_odom_bag(seq_dir))
        w_odom = interp_series(imu_b.t.to_numpy(), odom.t.to_numpy(), odom.w.to_numpy())
        m = np.isfinite(w_odom) & (np.abs(w_odom) > 0.05)
        corr = {ax: float(np.corrcoef(imu_b[ax][m], w_odom[m])[0, 1]) if m.sum() > 10 else None
                for ax in ["wx", "wy", "wz"]}
        report["imu_vs_wheel_yaw_corr"] = corr
        valid = {k: v for k, v in corr.items() if v is not None and np.isfinite(v)}
        best = max(valid, key=lambda k: abs(valid[k])) if valid else None
        if best is None or abs(valid[best]) < 0.5:
            # not enough turning to verify: keep the REP-103 default (+z) and say so
            best, sign = "wz", 1.0
            report["imu_yaw_axis_verified"] = False
        else:
            sign = float(np.sign(valid[best]))
            report["imu_yaw_axis_verified"] = True
        report["imu_yaw_axis_used"] = f"{'-' if sign < 0 else '+'}{best}"
        imu_b["wz"] = sign * imu_b[best]
        imu_b[["t", "wz", "ax", "ay", "az"]].to_csv(out / "imu.csv", index=False, float_format="%.9f")
        print(f"[ok] imu.csv {len(imu_b)} rows (raw {report['imu_raw_rate_hz']:.0f} Hz), yaw axis {report['imu_yaw_axis_used']} "
              f"({'verified' if report['imu_yaw_axis_verified'] else 'UNVERIFIED - too little turning'}), corr {corr}")

    # ---- GPS fixes ---------------------------------------------------------
    fix = pd.DataFrame(fix, columns=["t", "lat", "lon", "alt", "status", "service", "cov_e", "cov_n", "cov_u", "cov_type"])
    fix = fix.sort_values("t").drop_duplicates("t")
    fix.to_csv(out / "gps_raw.csv", index=False, float_format="%.10f")
    if len(fix):
        gt = load_gt_csv(seq_dir / "gt.csv")
        i0 = int(np.argmin(np.abs(fix.t.to_numpy() - gt.t.iloc[0])))
        e0, n0, _ = latlon_to_enu(fix.lat, fix.lon, fix.alt, fix.lat.iloc[i0], fix.lon.iloc[i0], fix.alt.iloc[i0])
        e, n = (np.column_stack([e0, n0]) @ rot2(np.radians(GT_FRAME_ROT_DEG)).T).T  # -> gt.csv frame
        gx = interp_series(fix.t.to_numpy(), gt.t.to_numpy(), gt.x.to_numpy())
        gy = interp_series(fix.t.to_numpy(), gt.t.to_numpy(), gt.y.to_numpy())
        ok = np.isfinite(gx)
        report["gt_origin_fix_time_offset_s"] = float(fix.t.iloc[i0] - gt.t.iloc[0])
        if ok.sum() > 10:
            th, tt = fit_se2(np.column_stack([e, n])[ok], np.column_stack([gx, gy])[ok])
            res_enu = np.hypot(e[ok] - gx[ok], n[ok] - gy[ok])
            report["raw_fix_vs_gt_after_frame_rotation"] = {
                "median_m": float(np.median(res_enu)), "p95_m": float(np.percentile(res_enu, 95)),
                "max_m": float(np.max(res_enu)),
                "remaining_rotation_deg_DIAGNOSTIC": float(np.degrees(th))}
        sigma = np.sqrt(np.maximum(0.5 * (fix.cov_e + fix.cov_n), 0))
        pd.DataFrame({"t": fix.t, "x": e, "y": n, "sigma": sigma, "fix_status": fix.status,
                      "source": "raw_navsatfix"}).to_csv(out / "gps.csv", index=False, float_format="%.6f")
        report["navsatfix_status_counts"] = {str(k): int(v) for k, v in fix.status.value_counts().items()}
        report["gps_rate_hz"] = float(1 / np.median(np.diff(fix.t))) if len(fix) > 1 else None
        print(f"[ok] gps.csv {len(fix)} fixes; status counts {report['navsatfix_status_counts']}; "
              f"raw vs gt {report.get('raw_fix_vs_gt_after_frame_rotation')}")

    # ---- receiver state ---------------------------------------------------------
    rx = pd.DataFrame(rx)
    if len(rx):
        rx = rx.sort_values("t").drop_duplicates("t")
        if len(fix):
            rx["x"] = interp_series(rx.t.to_numpy(), fix.t.to_numpy(), e)
            rx["y"] = interp_series(rx.t.to_numpy(), fix.t.to_numpy(), n)
        num = rx.select_dtypes("number")
        if "fix_mode" in rx:
            num = num.assign(fix_mode=rx["fix_mode"].astype(str))
        num.to_csv(out / "gps_quality.csv", index=False, float_format="%.6f")
        report["receiver_state_fields"] = list(rx.columns)
        report["receiver_state_variation"] = {c: {"min": float(num[c].min()), "max": float(num[c].max()),
                                                  "std": float(num[c].std())}
                                              for c in num.select_dtypes("number").columns if c not in ("t", "x", "y")}
        if "fix_mode" in rx:
            report["fix_mode_counts"] = {str(k): int(v) for k, v in rx["fix_mode"].value_counts().items()}
        print(f"[ok] gps_quality.csv {len(rx)} rows; fields {list(num.columns)}")
    (out / "base_extract_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "receiver_state_variation"}, indent=1, default=str))


if __name__ == "__main__":
    main()
