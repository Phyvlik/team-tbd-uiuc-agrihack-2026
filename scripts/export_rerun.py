r"""Reproducible Rerun export of one scenario (no Streamlit needed).

    .venv\Scripts\python scripts\export_rerun.py --source 06_14B_Jackal --method odom+imu+gps --outage 100:60
    .venv\Scripts\python scripts\export_rerun.py --source SYNTHETIC --kind operator --crew
    .venv\Scripts\python scripts\export_rerun.py --results team --source 02_13B_Jackal --mode Safe

Writes recordings/<kind>_<source>_<method>_<outage>.rrd (open with `rerun <file>`).
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

from dashboard import backend as be  # noqa: E402
from dashboard import rerun_export as rx  # noqa: E402
from dashboard import scenario as scn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="06_14B_Jackal", help=f"one of {be.source_keys()}")
    ap.add_argument("--method", default="odom+imu+gps", choices=list(be.METHODS))
    ap.add_argument("--outage", default="100:60", help="start_s:duration_s (relative), or 'none'")
    ap.add_argument("--kind", default="engineer", choices=["engineer", "operator"])
    ap.add_argument("--crew", action="store_true", help="add the simulated demo crew + safety buffer")
    ap.add_argument("--trusted", type=float, default=0.3)
    ap.add_argument("--stop", type=float, default=1.0)
    ap.add_argument("--results", default="ekf", choices=["ekf", "team"],
                    help="ekf: backend EKF run (default). team: teammate slip-learning replay")
    ap.add_argument("--mode", default="Safe", choices=["Safe", "Productive"], help="team only: decision mode")
    ap.add_argument("--overwrite", action="store_true", help="team only: replace an existing recording")
    a = ap.parse_args()
    if a.results == "team":
        from dashboard import team_slip as ts
        p, created = rx.build_team_recording(ts.load_run(a.source), a.mode, overwrite=a.overwrite)
        print(p if created else f"{p} (exists; pass --overwrite to rebuild)")
        return
    outages = () if a.outage == "none" else (tuple(float(v) for v in a.outage.split(":")),)
    zones = scn.demo_crew(a.source, outages) if a.crew else None
    p = rx.build_recording(a.source, a.method, outages, (), a.trusted, a.stop, kind=a.kind, zones=zones)
    print(p)


if __name__ == "__main__":
    main()
