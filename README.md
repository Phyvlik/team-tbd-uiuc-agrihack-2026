# Agricultural Robot Localization Dashboard

UIUC Precision & Digital Agriculture Hackathon (Nov 13–15, 2026) · Track 3 · Challenge 1
*Maintain reliable agricultural-robot localization using multimodal sensing* · Dataset: **CitrusFarm**

> Working title only — team and product names are not final.

A farm-robot localization system that estimates where the robot is, **estimates how reliable that
position is**, and turns that reliability into operator and worker-facing guidance. It uses recorded
CitrusFarm data. GPS outages and worker positions are simulated and labelled as such.

## Quick start (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_citrusfarm_subset.py      # gt.csv + odom bags, ~240 MB
.venv\Scripts\python scripts\validate_baseline.py               # extract + SP1 baseline validation
.venv\Scripts\python scripts\run_experiments.py --synthetic     # SP3/SP4/SP5 result tables
.venv\Scripts\python -m pytest -q                               # 26 tests
.venv\Scripts\python -m streamlit run app.py                    # http://localhost:8501
```

Optional (unlocks real IMU, raw GPS and receiver quality; streams ~2.5 GB, writes a few MB):
`python scripts\extract_base_bag.py 06_14B_Jackal`, then `python scripts\validate_baseline.py 06_14B_Jackal`.

## Dashboard

| page | what it shows |
|---|---|
| **Mission Control** (`app.py`) | playback of a run: map with reference vs estimate, 95 % uncertainty ellipse, navigation state (Trusted / Degraded / Fallback-Stop), GPS status, sensor availability, operator recommendation, alert log |
| **GPS Outage Lab** | choose method, outage duration (10/30/60/120 s), start and count, multipath jumps; live error-vs-time, outage metrics, all methods compared under the identical outage |
| **Sensor Intelligence** | modality inventory, GPS consistency (NIS) checks, degradation events, C1–C4 cost-vs-accuracy comparison |
| **Orchard Risk Map** | go/no-go check on GPS-quality variation, observed vs modelled quality maps, held-out canopy→GPS test |
| **Worker Safety** | simulated crews, dynamic buffer = min + k·r95 (+ canopy margin), slow/stop rules, worker status light, EN/ES alerts, offline safety scoring |
| **Results** | reproducible tables from `results/*.csv`; limitations |
| **Data Import** | upload a teammate CSV → mapping, unit detection, verification and frame diagnostics |

The sidebar switches between **Synthetic Demo Mode** (a labelled synthetic orchard) and **Real CitrusFarm
Mode**. Real mode never falls back to synthetic data. If something is missing, the page says so.

## Architecture

```
sensors (bags/CSVs) ─► data_loader ─► SensorStreams ─┬─► sensor_fusion.run_ekf ─► estimate + covariance
                                                      │        ▲ GPS after outage_simulator (withheld rows removed)
                                                      │        │
                                                      │   reliability_monitor ─► Trusted/Degraded/Stop
                                                      │   worker_safety ─────► buffer, slow/stop, EN/ES
                                                      └─► reference (scoring only) ─► evaluation metrics
```

| module | sub-problem | content |
|---|---|---|
| `src/coordinate_utils.py` | all | ENU conversion, SE(2) fits, lever arm, interpolation |
| `src/data_loader.py` | all | canonical streams, gt.csv / odom bag readers, processed cache |
| `src/localization.py` | SP1 | dead reckoning (B1, B2), initial alignment from first 2 m |
| `src/baseline_validation.py` | SP1 | separates frame/alignment artefacts from genuine drift |
| `src/sensor_fusion.py` | SP2 | 7-state EKF `[x,y,θ,v,ω,b_g,k_odom]`; odom/IMU/LiDAR/GPS updates; NIS gating; lever arm |
| `src/outage_simulator.py` | SP3 | outage windows, multipath jumps, GPS proxy (labelled) |
| `src/reliability_monitor.py` | SP4 | integrity states from covariance + GPS staleness + NIS; warning scoring |
| `src/sensor_comparison.py` | SP5 | C1–C4 under identical outages; qualitative cost tiers |
| `src/gps_risk.py` | SP6 | go/no-go, contiguous held-out canopy→quality test, risk grid |
| `src/worker_safety.py` | SP7 | synthetic worker zones, dynamic buffer, decisions, bilingual messages |
| `src/pipeline.py` | — | single code path used by UI and scripts |
| `src/rosbag_stream.py` | — | streaming ROS1 bag reader (extract small topics from multi-GB bags) |
| `src/teammate_import.py` | — | CSV import / normalisation / verification |

## Key verified findings so far (real CitrusFarm data)

1. **The large wheel-odometry errors are mostly real.** An independently built B1 drifts 46–284 m (max) across
   the 7 sequences. Heading error reaches 100–180° within 5–20 min, wheel distance is ~12 % longer than the RTK path,
   and the wheel yaw rate under-reads turns (gain 0.90–0.98). Leave-one-out calibration doesn't fix it consistently,
   so random skid-steer slip dominates. See `results/baseline_validation.md`.
2. **Two frame pitfalls exist:** (a) the controller's odom pose doesn't start at 0, and using it unaligned inflates
   error (06_14B: 181 m vs 46 m); (b) `gt.csv` is **ENU rotated −90°** (x = north, y = west), verified against raw
   receiver fixes (median 2.7 cm after rotation).
3. **The GPS antenna is 0.18 m behind base_link.** That's not negligible against a 0.3 m target, so it's modelled.
4. With the EKF, the wheel-speed scale is learned online while GPS is available (k ≈ 1.12, matching finding 1).
5. On held-out 06_14B (wheel odometry + GPS proxy), during a 30 s outage the integrity monitor warned 2.0 s after
   GPS loss and recommended stop at 5.6 s. The true error crossed 0.3 m at 6.4 s.

## Honesty rules built into the code

* `run_ekf` has no reference argument, so ground truth cannot leak into estimates.
* Outages remove GPS rows before the estimator sees them (tested).
* Missing real streams are reported as **NOT RUN / not extracted** and never simulated in their place.
* Until raw fixes are extracted, live GPS on real sequences is a **labelled proxy** (reference + 3 cm noise, causal).
  Raw fixes were measured within 2.7 cm (median) of the reference, so the proxy is close, but results with GPS
  should still be marked as such.
* EKF noise was tuned on 02_13B only; the other sequences are held out.
* Costs are qualitative tiers, not prices. Worker zones are synthetic. This isn't a certified safety system.

## Status / next steps

See `docs/STATUS.md`.

## Data & licence

CitrusFarm Dataset — Teng, Wang, Song & Karydis (2023), *Multimodal Dataset for Localization, Mapping and Crop
Monitoring in Citrus Tree Farms*, ISVC 2023. https://ucr-robotics.github.io/Citrus-Farm-Dataset/ . Data are not
redistributed in this repository (`data/` is git-ignored).
