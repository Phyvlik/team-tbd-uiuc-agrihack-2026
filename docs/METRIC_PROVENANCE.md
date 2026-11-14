# Metric provenance audit (2026-11-14)

Every value the dashboard displays, traced to its source in code. Categories:

* **REC** recorded CitrusFarm data (sensor logs, `gt.csv` reference), loaded read-only by the backend `data_loader`
* **CMP** computed live in this app by the team backend (`pipeline.run_experiment` → EKF + integrity monitor) from REC
* **BAT** team batch results, `results/*.csv` (backend `scripts/run_experiments.py`, generated 2026-11-13 22:37)
* **TM** teammate-provided run files via `dashboard/adapter.py` (**none present yet**; `team_runs/` empty)
* **SIM** explicitly simulated (outage windows, multipath jumps, GPS proxy, crews, spray, synthetic orchard)
* **ILL** illustrative geometry (tree positions inferred from the recorded path)

Cross-cutting caveats: **U** = uses the EKF r95, which is *uncalibrated* (see `UNCERTAINTY_CALIBRATION.md`);
**F** = error vs `gt.csv` on 06_14B with recorded GPS includes a ≤ 0.16 m reference-frame discrepancy;
**P** = on 01–05, 07 the live GPS is a SIMULATED proxy built from `gt.csv`. All three are labelled in the UI.

## Mission Control: Operator

| display | source | category | code | caveat |
|---|---|---|---|---|
| Navigation state (Trusted/Degraded/Stop) + reason | `reliability_monitor.classify` on EKF output | CMP (+SIM outage) | `res.states` | U |
| GPS status Available/Degraded/Lost | time since last accepted fix, recent NIS rejection | CMP | `ops.gps_status_series` | P (tile says "simulated proxy") |
| Position uncertainty ± r95 + gauge | EKF covariance → `position_radius(0.95)` | CMP | `est.r95` | **U** (tagged Uncalibrated) |
| Robot speed | EKF velocity state from wheel odometry (+IMU) | CMP | `est.v` | n/a |
| 3D/2D trail colours, bubble, robot pose | EKF antenna-point estimate + monitor state | CMP | `est.x_ant/y_ant`, `states.level` | U (bubble) |
| GPS lost / restored markers, outage timing | scenario | SIM | `OutageScenario` | n/a |
| Trees | inferred along recorded RTK path (6.4 m rows, 4 m spacing assumed) | ILL | `orchard.build_orchard` | not surveyed |
| Spray: treated / missed / double | rule-based simulation: sprayer fires from estimate, lands at RTK pose | SIM | `orchard.simulate_spray` | **not agricultural performance** |
| Crew, safety ring, crew alerts | simulated crew + backend `worker_safety.evaluate_safety` | SIM + CMP | `scenario.safety` | U (buffer = min + k·r95) |
| What the robot can do now | rules on monitor state + simulated-crew decision | CMP + SIM | `ops.capabilities` | advisory |
| EN/ES alerts | rules on the same situation | CMP + SIM | `ops.active_alerts` | Spanish unreviewed |
| Event chips (times) | state transitions + scenario times | CMP + SIM | `ops.events` | n/a |

## Mission Control: Engineer

| display | source | category | caveat |
|---|---|---|---|
| Actual error now / error timeline / mean, RMSE, max, % within 0.3 m | estimate vs `gt.csv` | CMP vs REC | F, P |
| Warned/Stop after, error crosses 0.3 m, lead time, peak error, recovery | `evaluate_warnings`, `outage_metrics` | CMP (SIM outage) | U, F |
| Truth inside 95 % radius | `evaluation.coverage` | CMP | U (this *is* the calibration metric) |
| False-alarm rate | `evaluate_warnings` | CMP | U |
| C1/C2 trajectory comparison, B1 baseline | same run per sensor set | CMP | F, P |
| NIS, rejections | EKF GPS consistency gate | CMP | n/a |
| Gyro bias b_g, odom scale k_odom | EKF states | CMP | Not evaluated without IMU |
| Wheel − IMU yaw-rate disagreement | `odom.w − imu.wz`, 1 s mean | REC (derived) | 06_14B only |
| Receiver satellites / σ | `gps_quality.csv` | REC (06_14B) / SIM (synthetic) | n/a |
| Team data table | adapter | TM (none) + CMP | n/a |
| Rerun replay | same objects as above, `time` = s since `gt.csv` start | as above | verified by `tests/test_consistency.py` |

## GPS Failure Simulator

| display | category | caveat |
|---|---|---|
| Phase, outage timing | SIM | n/a |
| Robot's own uncertainty | CMP | U (label says uncalibrated) |
| Actual drift from truth, outage summary sentence | CMP vs REC | F, P (caveat appended) |
| Same outage × C1/C2/C3 table and bars | CMP (SIM outage) | C3 Not evaluated; F, P |
| Held-out robustness chart | BAT (C1 only, 6 sequences) | P (batch used the proxy) |

## Sensor Cost Comparison

| display | category | caveat |
|---|---|---|
| Cost tiers ($–$$$$) | plan's qualitative tiers (`COST_TIERS`) | not prices |
| C1–C3 peak drift (Team batch) | BAT: real odometry during SIM outages, GPS proxy outside | P |
| C4 RMSE (Team batch) | BAT vs a SIM GPS proxy | **badge "GPS PROXY (SIMULATED)"**, not real-world accuracy |
| C1–C4 (Live · 06_14B) | CMP with recorded GPS (+ IMU) | F; C3 Not evaluated |
| Recommendation sentence | backend `sensor_comparison.recommend` on the shown evidence | inherits the above |

## Orchard Risk Map

| display | category | caveat |
|---|---|---|
| Observed GPS quality, go/no-go (06_14B) | REC + backend `gps_risk.quality_variation_check` | low/elevated/high limits are display limits, not agreed |
| Predicted risk blocks, held-out R²/ρ | SIM (synthetic orchard only) | relationship built into the generator |

## Worker Safety

| display | category | caveat |
|---|---|---|
| Crews, light, EN/ES crew alerts, distance, buffer, clearance | SIM crews + CMP estimate | U |
| Safety evaluation vs RTK (inside margin, unsafe samples, missed) | CMP vs REC | U, F |

## Team Slip-Learning Results source (added 2026-11-14)

Category **SL**: computed by the team's slip-learning localization analysis and exported to
`team_runs/slip_learning/`; replayed as supplied by `dashboard/team_slip.py` (validated, never recomputed).

| display | source column / file | category | caveat |
|---|---|---|---|
| Robot pose, heading, trail | `x_est, y_est, yaw_est` | SL | n/a |
| RTK reference (Engineer) | `x_true, y_true` (= `gt.csv`, verified) | REC | n/a |
| Navigation confidence Trusted / Degraded / Lost | `state` (matches 2σ thresholds on every row) | SL | thresholds from the export's author |
| Position uncertainty ± 2σ, bubble | `2 * sigma` | SL | model estimate, not validated; very conservative in outages |
| Safety bubble ring | `bubble_m` (= 2 + 2σ) | SL | n/a |
| GPS outage status and elapsed time | `gps_out` | SIM | fixed 30 s every 90 s |
| GO / SLOW / STOP, Safe / Productive | documented rules on `state`, `bubble_m`, worker distance | SL + SIM | decision support only |
| Nearest worker distance | estimate to visible `crew` row (visible `t_outage_start` to `t + 15`) | SIM + SL | workers simulated, on the path |
| Spray outcome per tree | `trees.status` | SIM | timing approximate (02, 06) or final only (others) |
| Actual error (Engineer) | `error_m` (equals the position difference) | SL vs REC | retrospective |
| Learned slip scale | `slip_scale` | SL | not independently validated |
| Backend EKF overlay (Engineer) | `src` pipeline with the same outage windows | CMP | different sensors and GPS input |
| GPS Failure Simulator, Sensor Cost Comparison, Worker Safety tables | `test_results`, `spray_results`, `safety_results` | SL + SIM | aggregation basis stated on each page |
| Not available | gyro bias, NIS, GPS rejections, raw GPS, LiDAR, covariance, speed | n/a | shown as "Not available in this export" |

## Audit actions taken

* Uncertainty everywhere tagged **Uncalibrated**; Engineer "Run metrics" states coverage and that nothing is rescaled.
* GPS proxy now labelled on the GPS status tile and on C4 batch cards ("GPS PROXY (SIMULATED)").
* Reference-frame caveat (F) shown wherever errors vs `gt.csv` are quoted for recorded-GPS runs.
* Risk-map limits labelled as display limits. Rerun views/info panel carry the same labels.
* Adapter no longer guesses that columns named `x`/`y` are estimates, and handles base_link vs antenna explicitly.
* No estimator, threshold, or teammate file was changed (`tests/test_app.py::test_estimator_covariance_is_not_modified`).
