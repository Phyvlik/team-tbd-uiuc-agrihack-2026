"""Batch experiments for the Results page (reproducible).

Usage:  python scripts/run_experiments.py [SEQ ...] [--synthetic]

For every processed CitrusFarm sequence (default: all) runs
  * SP5 configuration comparison (C1–C4, whichever streams exist)
  * SP3 outage robustness suite (10/30/60/120 s × 3 start times, identical across configs)
  * SP4 warning performance over the same suite
and writes results/*.csv. EKF noise was tuned on 02_13B_Jackal, so that
sequence is flagged ``role = tuning`` and excluded from held-out summaries.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import list_processed_sequences, load_processed  # noqa: E402
from src.outage_simulator import standard_outage_suite  # noqa: E402
from src.pipeline import run_experiment  # noqa: E402
from src.sensor_comparison import compare_configs  # noqa: E402
from src.synthetic_data import make_synthetic_sequence  # noqa: E402

TUNING_SEQUENCES = {"02_13B_Jackal"}
RESULTS = ROOT / "results"


def warning_suite(streams, sensors):
    t0, t1 = streams.time_span()
    rows = []
    for sc in standard_outage_suite(t1 - t0):
        r = run_experiment(streams, sensors, sc)
        w = r.warnings
        po = w["per_outage"][0] if w["per_outage"] else {}
        rows.append({"duration_s": sc.outages[0][1], "start_s": sc.outages[0][0],
                     "recall_warn_when_bad": w["recall_warn_when_bad"], "false_alarm_rate": w["false_alarm_rate"],
                     "missed_hazard_rate": w["missed_hazard_rate"], **{k: v for k, v in po.items() if k != "outage_start"},
                     "coverage_r95_pct": r.metrics["coverage_r95_pct"]})
    return pd.DataFrame(rows)


def main(argv):
    RESULTS.mkdir(exist_ok=True)
    seqs = [a for a in argv if not a.startswith("--")] or list_processed_sequences()
    jobs = [(s, lambda s=s: load_processed(s)) for s in seqs]
    if "--synthetic" in argv:
        jobs.append(("SYNTHETIC_orchard_demo", make_synthetic_sequence))
    ov_all, ot_all, w_all = [], [], []
    for name, loader in jobs:
        tic = time.time()
        st = loader()
        role = "synthetic" if st.mode == "synthetic" else ("tuning" if name in TUNING_SEQUENCES else "held-out")
        overall, otab = compare_configs(st)
        sens = ("odom", "imu", "gps") if st.available().get("imu") else ("odom", "gps")
        w = warning_suite(st, sens)
        gps_src = "raw" if st.available().get("gps") else "proxy_from_reference"
        for df in (overall, otab, w):
            df.insert(0, "sequence", name)
            df.insert(1, "role", role)
            df["gps_source"] = st.gps["source"].iloc[0] if st.available().get("gps") and "source" in st.gps else gps_src
        w["sensors"] = "+".join(sens)
        ov_all.append(overall); ot_all.append(otab); w_all.append(w)
        print(f"[ok] {name} ({role}) in {time.time() - tic:.0f} s")
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    for fname, parts in [("config_comparison.csv", ov_all), ("outage_robustness.csv", ot_all),
                         ("warning_performance.csv", w_all)]:
        df = pd.concat(parts, ignore_index=True)
        df["generated"] = stamp
        df.to_csv(RESULTS / fname, index=False)
        print(f"wrote results/{fname} ({len(df)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
