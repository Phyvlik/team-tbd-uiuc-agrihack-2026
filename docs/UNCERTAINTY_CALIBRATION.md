# Uncertainty calibration status: **UNCALIBRATED**

The EKF's 95 % position region (`r95`, shown as the uncertainty bubble/gauge/tiles, and used to size the worker
safety buffer) has **not been validated** against ground truth. The dashboard displays it exactly as the backend
computes it (nothing is rescaled, inflated, or re-thresholded) and labels it *Uncalibrated* wherever it appears.

Reproduce: `.venv\Scripts\python scripts\calibration_audit.py 06_14B_Jackal` (read-only; writes
`.cache/calibration_audit_06_14B_Jackal.json`). Sequence 06_14B, odometry + IMU + **recorded** GPS
(`/piksi/navsatfix_best_fix`, reported σ = 0.07 m), backend EKF defaults.

## Findings (06_14B, 2026-11-14)

| run | 95 % circle coverage (UI metric) | 95 % ellipse coverage | mean NEES (ideal 2) | median error / median r95 |
|---|---|---|---|---|
| recorded GPS, no outage (as in the dashboard) | **53.2 %** | 37.4 % | 16.8 | 0.073 / 0.078 m |
| same run, GPS available phase of a 60 s-outage run | 38.8 % | 27.2 % | 21.4 | 0.098 / 0.078 m |
| same run, **during** the 60 s outage | 99.7 % | 97.0 % | 0.68 | 0.29 / 6.7 m |
| same run, 30 s after GPS returns | 88.7 % | 57.0 % | 6.4 | 0.050 / 0.078 m |
| GPS **proxy** (reference + 3 cm white noise, simulated) | 92.4 % | 83.4 % | 3.6 | 0.022 / 0.046 m |
| DIAGNOSTIC ONLY: recorded GPS with gt-y scale mismatch removed (never used in the app) | 99.3 % | 96.8 % | 1.24 | 0.018 / 0.078 m |

**1 · Dominant cause: the recorded GPS fixes and `gt.csv` are not in exactly the same frame.**
Raw fix − reference residual: x mean 0.0 mm, std 0.5 mm; **y mean −7.7 cm, std 5.0 cm**, correlation time 26 s.
The y residual is a position-proportional scale difference: `Δy = 0.191 % × y` (R² = 0.996, residual std 3.3 mm),
reaching 0.16 m at the far end of the 82 m route. No time lag explains it (best lag 0.00 s). The EKF follows the raw
fixes (σ 7 cm), so its estimate sits up to 16 cm from `gt.csv` while its r95 is ≈ 8 cm → ~half the samples fall
outside. Removing the fitted scale (diagnostic only) restores ≈ 97–99 % coverage.
*Which source is right is unknown*: the backend's raw-fix conversion (`scripts/extract_base_bag.py`: exact WGS84→ENU,
then −90° rotation) or the dataset's `gt.csv` local frame. The earlier "median 2.7 cm" verification used only the first
37 s near the origin, where the scale difference is still small.

**2 · Secondary cause: GPS errors are treated as independent.** The EKF fuses every 10 Hz fix as white noise with the
reported σ. Residual correlation times are 4–26 s, so the filter averages away less noise than it believes. Even with
an ideal white-noise proxy, NEES is 3.6 (> 2) and ellipse coverage 83 % (mildly overconfident).

**3 · During outages the region is very conservative** (NEES 0.68; median r95 6.7 m vs median error 0.29 m with
odom + IMU). This is why the integrity monitor recommends *Stop* ~7.5 s into an outage while the true error only
passes 0.3 m after ~29 s. Process-noise tuning for the IMU case is a fusion-owner decision, not a UI change.

**4 · The UI's "coverage" uses the conservative r95 circle** (major axis). The proper 95 % ellipse test is stricter
(37 % vs 53 % above); both are reported by the audit.

## Consequences in the dashboard (all labelled)

* Uncertainty tiles, gauge, bubble, Rerun r95 plot → tagged **Uncalibrated**.
* "Truth inside 95 % radius", worker-buffer coverage, warning lead times → depend on the uncalibrated r95.
* Errors vs `gt.csv` on 06_14B with recorded GPS include the ≤ 0.16 m reference discrepancy (caveat shown on Mission
  Control, GPS Simulator, Sensor Cost Comparison live view, Rerun info panel).
* Sequences 01–05, 07 use a **simulated GPS proxy** built from `gt.csv` → their with-GPS errors/coverage are not
  real-world results (labelled "simulated proxy").

## To mark it calibrated (owner: data + fusion)

1. Resolve the 0.19 % y-scale discrepancy (compare the raw-fix conversion with how `gt.csv` was produced; check the
   dataset's documentation / origin / projection). Re-extract `gps.csv` if the conversion changes.
2. Re-run `scripts/calibration_audit.py` on 06_14B **and** at least one more held-out sequence with recorded GPS.
3. If coverage is still off, tune the measurement/process noise on the tuning sequence only (02_13B per the plan),
   then report coverage on held-out sequences. Accept when 95 % ellipse coverage is ≈ 90–97 % with GPS and during
   outages. Only then change `CALIBRATION_STATUS` in `dashboard/backend.py`.
