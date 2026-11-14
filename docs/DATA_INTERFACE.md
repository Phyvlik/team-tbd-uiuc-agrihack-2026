# Data interface

Every data source is normalised into one `SensorStreams` container (`src/data_loader.py`).
This file is the contract between teammates: if your processed data matches it, every page of
the dashboard and every script will use it unchanged.

## 1. Canonical streams (`data/processed/<SEQUENCE>/<stream>.csv`)

| stream | required columns | units | frame / meaning | used by |
|---|---|---|---|---|
| `reference.csv` | `t, x, y` (+ `z`) | s, m | map frame (= `gt.csv` frame). **Scoring only — never given to an estimator.** | evaluation |
| `odom.csv` | `t, v, w` | s, m/s, rad/s | body twist of base_link (x forward, z up); optional `x_ctrl, y_ctrl, yaw_ctrl` = controller's own pose (cross-check only) | EKF, B1 |
| `imu.csv` | `t, wz` | s, rad/s | gyro yaw rate about body z (axis/sign verified against wheel yaw rate) | EKF, B2 |
| `gps.csv` | `t, x, y, sigma` (+ `fix_status`, `source`) | s, m, m | live receiver fixes in the map frame; `sigma` = 1-σ horizontal | EKF |
| `lidar_odom.csv` | `t, v, w` | s, m/s, rad/s | scan-matching body twist | EKF (C3) |
| `gps_quality.csv` | `t` + any numeric metrics (+ `x, y`) | — | receiver state along the route | SP6 |
| `canopy.csv` | `t, ndvi` (+ `x, y`) | — | canopy index per camera frame | SP6, SP7 proactive mode |
| `meta.json` | `name, mode, reference_lever_arm` | m | lever arm = body-frame (dx, dy) of the point that `reference`/`gps` measure | all |

Rules
* `t` is float seconds. Real data uses ROS header stamps (Unix epoch); synthetic data uses seconds from start.
* `t` must be strictly increasing (loader sorts and drops duplicate stamps; `SensorStreams.validate()` checks).
* Missing optional streams are simply absent — **they are never synthesised in real-data mode**.

## 2. Verified CitrusFarm facts (checked 2026-11-13)

| item | finding | how verified |
|---|---|---|
| `gt.csv` format | `# timestamp,tx,ty,tz,qx,qy,qz,qw`, ~10 Hz, local metres, origin = first sample | read all 7 files |
| `gt.csv` orientation | quaternion is all zeros → **no heading in the reference** | read all 7 files |
| `gt.csv` frame | **ENU rotated −90° (x = north, y = west)**; origin = first RTK fix | raw `/piksi/navsatfix_best_fix` (06_14B, first 37 s) matches gt after −90° rotation: median 2.7 cm, max 7 cm |
| what gt measures | GPS antenna position | calibration: base_link→lidar (+0.04, 0, 0.379), lidar→GPS (−0.22, 0, 0.153) ⇒ antenna at base_link **(−0.18, 0, 0.53) m** |
| odom | `/jackal_velocity_controller/odom`, 50 Hz, `twist.linear.x`, `twist.angular.z` | all 7 bags |
| odom pose | controller pose does **not** start at 0 (e.g. 06_14B starts at (2.02, 1.10, 1.62 rad)) | all 7 bags |
| time sync | odom vs gt speed cross-correlation lag between −0.08 s and 0 s (where speed varies) | `scripts/validate_baseline.py` diagnostics |
| IMU | `/microstrain/imu/data`, 200 Hz, yaw = +z (corr. 0.92 with wheel yaw rate) | 06_14B partial stream |
| raw GPS | `/piksi/navsatfix_best_fix`, 10 Hz, status 2, reported σ = 0.07 m | 06_14B partial stream |
| receiver state | `/piksi/debug/receiver_state` (`piksi_rtk_msgs/ReceiverState_V2_4_1`), ~11 Hz: `num_sat`, per-constellation counts and C/N0, `fix_mode` | 06_14B partial stream |

## 3. File sizes (S3, `s3://ucr-robotics/citrus-farm-dataset/`, public, no account needed)

| file | per sequence | needed for |
|---|---|---|
| `ground_truth/<SEQ>/gt.csv` | 0.3–1.6 MB | reference (all SPs) |
| `<SEQ>/odom_*.bag` | 12–66 MB | wheel odometry |
| `<SEQ>/base_*.bag` | 2.5–12.8 GB (split in ≤4.2 GB parts) | IMU, raw GPS, receiver state, LiDAR |
| `<SEQ>/mapir_*.bag` | 8–51 GB | R-G-NIR canopy index |

Smallest useful full sequence: **06_14B_Jackal** (gt 0.3 MB, odom 12 MB, base 2.5 GB in one file).
`scripts/extract_base_bag.py` streams `base_*.bag` over HTTP and keeps only IMU/GPS/receiver state (a few MB).

## 4. Importing a teammate's trajectory CSV

In the dashboard, drop run files into `team_runs/` and open Mission Control, Engineer mode, tab
**Team data (adapter)** (format: "Teammate integration guide" in `README.md`). The adapter calls the
same checks as this lower-level helper, which you can also use directly:

```python
from src.teammate_import import import_file
norm, notes, report = import_file("her_baseline_06.csv")              # auto-detect
norm, notes, report = import_file("x.csv", mapping={"estimated_x": "odomX"}, time_unit="ns", length_unit="mm")
```

Target schema after normalisation:

| column | unit | meaning |
|---|---|---|
| `timestamp` | s | float seconds (ns/µs/ms auto-detected from magnitude, or pass `time_unit`) |
| `reference_x`, `reference_y` | m | RTK reference in its own frame |
| `estimated_x`, `estimated_y` | m | estimate **in the same frame as the reference** |
| `error_m` (optional) | m | the file's own error; compared with the recomputed error |

Recognised aliases include `time, t, stamp, time_ns`, `gt_x, rtk_x, ref_x, tx`, `odom_x, est_x, pred_x, b1_x`,
`error, err, position_error` (full list in `src/teammate_import.py`).

The verification report flags: non-monotonic timestamps, start offset (missing initial alignment), a large
rigid rotation that removes most of the error (heading/frame mismatch, e.g. the −90° ENU/gt rotation), and
path-length ratio ≠ 1 (speed-scale error, wrong units, or double integration).

**Please send numerical CSVs, not plot images** — metrics cannot be recovered from a picture.
