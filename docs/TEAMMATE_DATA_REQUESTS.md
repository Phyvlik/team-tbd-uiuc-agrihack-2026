# Data and interfaces needed from teammates

## Export checklist: connecting the final localization models

One CSV per (sequence, method) → `team_runs/<SEQUENCE>__<method>.csv`
(e.g. `06_14B_Jackal__ekf-odom-imu-lidar.csv`), plus an optional `<same name>.json` sidecar.

- [ ] **`t`**: ROS header stamp of the estimate, Unix epoch (s or ns), *not* bag-receive time
- [ ] **`est_x`, `est_y`**: metres in the **`gt.csv` frame** (ENU rotated −90°: x = north, y = west, origin = first gt sample)
- [ ] **Point**: the **GPS antenna** (what `gt.csv` measures). If you export base_link, set `"point": "base_link"` in the
      sidecar **and** include `heading` (lever arm −0.18 m along body x is then applied; without heading it is flagged)
- [ ] **`var_x`, `var_y`, `cov_xy`** (m², antenna position), or `r95` (m). Omit if you don't have it → *Not evaluated*
- [ ] **`speed`** (m/s) and **`heading`** (rad, CCW from +x in the gt frame) if estimated
- [ ] **`gps_available`** (0/1, was a fix *used* at this step), **`gps_rejected`** (0/1), **`gps_quality`** (one numeric metric)
- [ ] **`gyro_bias`** (rad/s) if your filter estimates it
- [ ] **`outage`** (0/1) on every sample inside a simulated outage, or `"outages_s": [[start, end], …]` (s since the first gt sample)
- [ ] Sidecar: `"sequence_id"`, `"method"`, `"sensors"` (e.g. `"odom+imu+lidar+gps"`), `"gps_input"`
      (`"recorded"` or `"proxy"`), `"time_unit"`/`"length_unit"` if not s/m
- [ ] **Do not** include ground truth as an input, smooth with future data, or tune on held-out sequences (tune on 02_13B)
- [ ] Optional: `lidar_odom.csv` (`t, v, w`) per sequence → enables C3 in the backend; `<SEQ>__orchard_geometry.csv`
      (`x, y[, row]`, gt frame) → replaces illustrative trees; LiDAR scans (`.npz`/`.ply` + `t`) → Rerun point clouds
- [ ] **Blocking calibration item (data owner):** explain/resolve the 0.19 % y-scale mismatch between
      `extract_base_bag.py`'s raw-fix conversion and `gt.csv` (`docs/UNCERTAINTY_CALIBRATION.md`), then re-run
      `scripts/calibration_audit.py`

Check a file before the demo: Mission Control → Engineer → *Team data (adapter)* lists every assumption the adapter
made (time unit, epoch → relative, overlap %, point, frame/units red flags). Missing fields are shown as *Not evaluated*;
nothing is inferred.

The dashboard reads the backend's `SensorStreams` contract (`docs/DATA_INTERFACE.md`)
and `results/*.csv`. Drop files in the documented places and the UI picks them up; no UI change needed.
Status as of 2026-11-14.

## New: drop-in run files for the data adapter (Engineer → Team data, and the Rerun replay)

Put one CSV per run in `team_runs/` (or `results/runs/`), named
`<SEQUENCE>__<method>.csv`, e.g. `06_14B_Jackal__lidar-icp.csv`. Full schema: docstring of `dashboard/adapter.py`.

| column | required | unit / frame | if missing |
|---|---|---|---|
| `t` | yes | s (ns/µs/ms epoch auto-detected and converted to s since sequence start) | n/a |
| `est_x, est_y` | yes | m, **same frame as CitrusFarm gt.csv** (ENU rotated −90°) | n/a |
| `var_x, var_y, cov_xy` or `r95` | no | m², or m | uncertainty "Not evaluated" |
| `speed, heading` | no | m/s, rad CCW from +x | "Not evaluated" |
| `gps_available, gps_quality, gps_rejected` | no | 0/1, metric, 0/1 | "Not evaluated" |
| `gyro_bias` | no | rad/s | "Not evaluated" |
| `ref_x, ref_y` | no | m | recorded reference interpolated (labelled) |

Optional sidecar `<same name>.json`: `sequence_id, method, sensors, time_unit, length_unit, outages_s, lidar_points,
orchard_geometry`. The adapter reports every assumption (time unit, epoch→relative, overlap %, rigid-rotation / path-length
red flags from `teammate_import.verify`) and never fills in missing fields.

Also useful: `team_runs/<SEQ>__orchard_geometry.csv` with real tree positions (`x, y[, row]`, map frame) replaces the
illustrative trees in the 3D view and Rerun scene; LiDAR point clouds (`.ply`/`.npz` per scan + timestamps) would let the
Rerun replay show them (currently "Data unavailable").

## Blocking "Not evaluated" panels

| # | Needed | Format / location | Unlocks in the UI | Owner (suggested) |
|---|---|---|---|---|
| 1 | **Process the already-extracted 06_14B streams and re-run batch results**: `data/raw/06_14B_Jackal/` has `imu.csv`, `gps.csv`, `gps_quality.csv`, but `data/processed/06_14B_Jackal/` and `results/*.csv` predate them (batch reports C2 as NOT RUN, missing imu) | `python scripts/validate_baseline.py 06_14B_Jackal` then `python scripts/run_experiments.py` | C2 in **Team batch** evidence (today C2 is only measured in **Live · 06_14B**) | fusion / data |
| 2 | **IMU + raw GPS for ≥ 1 more held-out sequence** (plan: 02_13B, 3.5 GB stream) | `python scripts/extract_base_bag.py 02_13B_Jackal` → `imu.csv`, `gps.csv`, `gps_quality.csv` | C2 on a second sequence; real GPS instead of the proxy; risk-map go/no-go on another route | data |
| 3 | **LiDAR scan-matching odometry** | `data/processed/<SEQ>/lidar_odom.csv`: `t, v, w` (s, m/s, rad/s, body frame) | **C3** everywhere (currently Not evaluated on all real data); "Full fusion + LiDAR" method | perception |
| 4 | **Canopy index** from R-G-NIR camera (`mapir_*.bag`, 8–51 GB); team decision pending | `canopy.csv`: `t, ndvi` (+ `x, y` map frame) | predictive risk blocks and held-out canopy→GPS test on real data; Worker Safety proactive mode on real data | domain / CV |
| 5 | **Receiver quality for other sequences** | `gps_quality.csv`: `t` + numeric metrics (+ `x, y`), e.g. `num_sat`, `h_sigma_m`, `fix_mode` | Orchard Risk Map beyond 06_14B (other sequences show Not evaluated) | data |

## Decisions / reviews needed

| # | Item | Why it matters | Who |
|---|---|---|---|
| 6 | **Integrity monitor is very conservative with odom + IMU.** On real 06_14B, 60 s outage at 100 s: Degraded after 2.0 s, Stop after 7.5 s, but the true error stayed < 0.3 m until ~29 s (peak 1.6 m) while the EKF's 95 % radius grew to ~13 m. | Early stops cost uptime; judges may ask. Consider process-noise (`q_xy`, `q_theta`) tuning for the IMU case, or a time-based Degraded band before Stop. | fusion owner |
| 7 | **Uncertainty is uncalibrated**: 95 % coverage 53 % on 06_14B with recorded GPS. Audit: mostly a 0.19 % y-scale mismatch between recorded fixes and `gt.csv` (diagnostic correction → 97–99 %), plus correlated GPS errors fused as white noise. See `docs/UNCERTAINTY_CALIBRATION.md`. | Until resolved, all uncertainty displays carry an *Uncalibrated* tag; no values are rescaled. | data owner + fusion owner |
| 8 | Confirm thresholds: **1.0 m stop**, **0.3 m trusted**, **2.0 m minimum worker buffer** | All operator/crew colours depend on them (editable in Engineer → Integrity thresholds) | domain lead |
| 9 | Confirm **risk-map limits** I set for display: σ ≥ 0.05 m elevated / ≥ 0.10 m high (3σ hits the 0.3 m target); satellites < 20 elevated / < 12 high; fix mode FIXED / FLOAT / other | Labels on the risk map | domain lead |
| 10 | **Spanish alert wording** review by a native speaker (`src/worker_safety.py · MESSAGES`) | Crew-facing text | domain lead |
| 11 | Optional: **real orchard row geometry** (tree-line polylines for blocks 13B/13D/14B) | Draw rows on real maps (synthetic orchard has them) | domain |
| 12 | Optional: **sourced sensor prices** to replace qualitative tiers (`COST_TIERS[...]["price_note"]`) | Cost page shows tiers only, by design | business |

## Interfaces the UI calls (do not rename without telling UI)

`src.pipeline.run_experiment / ExperimentResult (estimate, error, states, gps_log, metrics, outage_table, warnings,
gps_source, sensors, outages_abs, gps_events)` · `src.reliability_monitor.MonitorConfig, transitions` ·
`src.sensor_comparison.CONFIGS, CONFIG_LABEL, COST_TIERS, compare_configs, summarize_outages, recommend` ·
`src.gps_risk.quality_variation_check, canopy_quality_relationship, risk_map` ·
`src.worker_safety.SafetyConfig, evaluate_safety, score_safety, place_workers_along_route, MESSAGES` ·
`src.data_loader.build_citrusfarm_sequence, load_processed, list_processed_sequences` ·
`src.synthetic_data.make_synthetic_sequence` · `src.sensor_fusion.ellipse_points` · estimate columns
`t, x_ant, y_ant, theta, v, r95, var_x, var_y, cov_xy, k_odom, b_g, time_since_gps, gps_rejected`.
