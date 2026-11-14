# Implementation status — 2026-11-13 (end of phase 1)

Legend: ✅ working on real CitrusFarm data · 🟡 implemented and tested, waiting on a data stream · ⛔ not started

| SP | status | what exists | what's missing |
|---|---|---|---|
| SP1 Trajectory estimation | ✅ | B1 on all 7 sequences, alignment from first 2 m, lever arm, artefact-vs-drift validation (`results/baseline_validation.*`) | B2 (odom + IMU heading) needs `imu.csv` |
| SP2 Sensor fusion | ✅ odom+GPS · 🟡 IMU/LiDAR | 7-state EKF with odometry, IMU, LiDAR and GPS updates; online odom-scale estimation; NIS gating | real IMU (extractor ready), LiDAR scan matching (not written) |
| SP3 GPS outages | ✅ | 10/30/60/120 s suites with identical windows, multipath jumps, recovery metrics, held-out results | rerun with raw GPS fixes once extracted |
| SP4 Degradation detection | ✅ | Trusted/Degraded/Stop from covariance + GPS staleness + NIS; latency, lead time, false alarms, coverage | real receiver-state degradation events |
| SP5 Reduced sensors | 🟡 | C1–C4 framework, qualitative cost tiers, recommendation logic; C2/C3 reported as NOT RUN on real data | IMU + LiDAR streams |
| SP6 GPS risk | 🟡 | go/no-go check, contiguous held-out test, observed/modelled maps (synthetic shows the pipeline works) | full receiver state (extractor ready); NIR canopy index (mapir bags are 8–51 GB) |
| SP7 Worker safety | ✅ (synthetic workers) | dynamic buffer, slow/stop rules, monitor coupling, status light, EN/ES alerts, offline safety scoring | Spanish review by a native speaker; domain-lead review of thresholds |

## Headline numbers (held-out sequences 01, 03–07; wheel odometry + GPS proxy; 18 runs per duration)

| outage | median peak error | range | % of outage within 0.3 m |
|---|---|---|---|
| 10 s | 0.36 m | 0.09–1.39 m | 79 % |
| 30 s | 2.99 m | 0.26–7.31 m | 35 % |
| 60 s | 8.32 m | 2.11–18.9 m | 15 % |
| 120 s | 18.7 m | 5.9–36.4 m | 8 % |

The integrity monitor flagged DEGRADED 2.0 s after GPS loss and STOP after 5.6 s (medians). The true error
crossed 0.3 m after a median of 6.7 s, so the warning came first in **all 72 held-out runs** (minimum lead
1.7 s). The 95% radius covered the truth 94–95% of the time.
Wheel-odometry-only B1 drifts 46–284 m (max) per sequence.

## Next technical step (in order)

1. `python scripts/extract_base_bag.py 06_14B_Jackal` (streams 2.5 GB, keeps ~5 MB) → real IMU, raw GPS and
   receiver state for the smallest sequence; then `validate_baseline.py 06_14B_Jackal` and `run_experiments.py`.
   This unlocks C2 (odom+IMU), B2, real GPS in the EKF, and the SP6 go/no-go check.
2. Repeat for one more held-out sequence (02_13B base bag is 3.5 GB) to keep the tuning/test split.
3. LiDAR odometry (C3): extend the extractor to save every Nth `/velodyne_points` scan (voxel-downsampled),
   2-D ICP between scans → `lidar_odom.csv` (t, v, w).
4. Teammate CSVs → Data Import page; compare with our B1 table.
5. Optional: Rerun 3D scene; demo-video script following the story on Mission Control.

## Open questions for the team

* Confirm the 1.0 m stop threshold and 2.0 m minimum worker buffer (domain lead).
* Review Spanish alert wording.
* Do we pull mapir bags at all (≥ 8 GB per sequence), or present SP6 as receiver-quality-only?
