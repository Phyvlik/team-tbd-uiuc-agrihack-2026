"""Download the smallest useful CitrusFarm subset (public S3, no account needed).

Default: all 7 gt.csv (~6 MB) + all 7 odom_*.bag (~240 MB) + calibration text files.
These are enough for SP1 and for wheel-odometry + GPS-outage experiments.
base_*.bag (IMU/GPS/LiDAR, GBs) is NOT downloaded — use scripts/extract_base_bag.py to stream it.

Usage: python scripts/download_citrusfarm_subset.py [SEQ ...]
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
S3 = "https://ucr-robotics.s3.amazonaws.com"
PREFIX = "citrus-farm-dataset"
SEQS = ["01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal", "04_13D_Jackal", "05_13D_Jackal", "06_14B_Jackal",
        "07_14B_Jackal"]
CALIB = ["Calibration/results/04-lidar-gps-result.txt", "Calibration/results/05-baselink-lidar-result.txt",
         "Calibration/results/02-imu-intrinsics.yaml", "Calibration/config/microstrain_gx5.yaml"]


def get(key: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[have] {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {key}")
    urllib.request.urlretrieve(f"{S3}/{PREFIX}/{key}", dest)


def main(seqs):
    for c in CALIB:
        get(c, RAW / "calibration" / Path(c).name)
    for s in seqs:
        get(f"ground_truth/{s}/gt.csv", RAW / s / "gt.csv")
        xml = urllib.request.urlopen(f"{S3}/?list-type=2&prefix={PREFIX}/{s}/odom_").read().decode()
        for key in re.findall(r"<Key>(.*?)</Key>", xml):
            get(key.removeprefix(PREFIX + "/"), RAW / s / Path(key).name)


if __name__ == "__main__":
    main(sys.argv[1:] or SEQS)
