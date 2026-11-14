# data/

CitrusFarm recordings and files derived from them. **Nothing under `data/raw/` or
`data/processed/` is committed** (see `.gitignore`); every teammate rebuilds it locally
with the scripts below.

Dataset: CitrusFarm, Teng, Wang, Song and Karydis, UC Riverside, 2023, CC BY-SA 4.0.
Registry: https://registry.opendata.aws/citrus-farm/ (public S3 bucket `ucr-robotics`,
prefix `citrus-farm-dataset/`, no account needed).

## Layout

```text
data/
  raw/
    calibration/                 4 official calibration files (IMU intrinsics, LiDAR/GPS extrinsics)
    <SEQ>/gt.csv                 original RTK reference trajectory (gt.csv frame: ENU rotated -90 deg)
    <SEQ>/odom_*.bag             original ROS1 bag, wheel odometry (/jackal_velocity_controller/odom)
    <SEQ>/imu.csv                extracted from base_*.bag (/microstrain/imu/data)
    <SEQ>/gps_raw.csv            extracted raw fixes, lat/lon (/piksi/navsatfix_best_fix)
    <SEQ>/gps.csv                the same fixes converted to the gt.csv frame
    <SEQ>/gps_quality.csv        receiver state (/piksi/debug/receiver_state)
    <SEQ>/base_extract_report.json  extraction report (frame check, counts)
  processed/
    <SEQ>/reference.csv, odom.csv, meta.json   reformatted gt.csv and decoded odometry
```

## What exists per sequence (local copy, 2026-11-14)

| sequence | gt.csv | odom bag | imu.csv | gps.csv | gps_quality.csv | processed |
|---|---|---|---|---|---|---|
| 01_13B_Jackal | yes | yes | no | no | no | reference + odom |
| 02_13B_Jackal | yes | yes | no | no | no | reference + odom |
| 03_13B_Jackal | yes | yes | no | no | no | reference + odom |
| 04_13D_Jackal | yes | yes | no | no | no | reference + odom |
| 05_13D_Jackal | yes | yes | no | no | no | reference + odom |
| 06_14B_Jackal | yes | yes | **yes** | **yes** | **yes** | reference + odom (built before the IMU/GPS extraction) |
| 07_14B_Jackal | yes | yes | no | no | no | reference + odom |

No LiDAR odometry (`lidar_odom.csv`) or canopy index (`canopy.csv`) exists for any sequence.
`06_14B_Jackal/partial_inspection/` holds an earlier partial extraction (first 120 MB of the
base bag) kept for reference; nothing reads it.

## How the app loads a sequence (`dashboard/backend.py: load_streams`)

1. If `data/raw/<SEQ>/` has any extracted CSV (`imu`, `gps`, `gps_quality`, `lidar_odom`,
   `canopy`), the sequence is built from raw files with `src.data_loader.build_citrusfarm_sequence`.
   Today that is only 06_14B, which therefore loads odometry, IMU and **recorded** GPS.
2. Otherwise `data/processed/<SEQ>/` is loaded with `src.data_loader.load_processed`
   (reference + odometry only). Experiments that need GPS then use a **simulated GPS proxy**:
   the reference plus 3 cm white noise (`src.outage_simulator.gps_proxy_from_reference`).

## Rebuilding the data (Windows, from the project root)

```powershell
# 1. gt.csv + odom bags for all 7 sequences + calibration (~250 MB)
.venv\Scripts\python scripts\download_citrusfarm_subset.py

# 2. Optional: stream base_*.bag from S3 (GBs, not stored) and keep IMU + GPS (~3 MB)
.venv\Scripts\python scripts\extract_base_bag.py 06_14B_Jackal

# 3. Build data/processed and results/baseline_validation.* (overwrites tracked results files)
.venv\Scripts\python scripts\validate_baseline.py

# 4. Optional: batch experiments, rewrites results/config_comparison.csv etc. (tracked)
.venv\Scripts\python scripts\run_experiments.py
```

Steps 3 and 4 change files under `results/`, which are committed. Review the diff before
committing regenerated numbers.
