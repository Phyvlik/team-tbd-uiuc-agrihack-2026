# Two-minute demo script

## Primary: Team Slip-Learning Results, 02_13B, Safe mode

Setup: start the app, set **Localization results** to **Team Slip-Learning Results** (sequence 02_13B), and in
Engineer mode build the Rerun recording once (Mission Control, tab *Rerun replay*). Return to **Operator** mode.

| time | screen | action | say |
|---|---|---|---|
| 0:00 | Mission Control (Operator) | n/a | "Orchard canopy blocks GPS. This is a real CitrusFarm robot run, replayed with our slip-learning localization. Trees, workers and spraying are simulated, and labelled that way." |
| 0:10 | | click **▶ Run GPS Failure Demo** | "Big green GO: position trusted, uncertainty about 8 centimetres." |
| 0:20 | (playing, 60 s) | point at GPS tile | "We cut GPS for 30 seconds. The robot keeps estimating its position from wheels and the IMU, with a learned correction for wheel slip." |
| 0:30 | (63 s to 75 s) | point at bubble + status | "Confidence drops to Degraded, so SLOW. At 75 seconds the model uncertainty passes 1.5 metres: Lost. Safe mode says STOP." |
| 0:45 | (around 84 s) | point at worker + alerts | "A simulated worker is on the route. The worker is inside the safety bubble, so STOP, in English and Spanish." |
| 0:55 | (90 s) | point at GPS tile | "GPS is back, confidence recovers to Trusted." |
| 1:05 | Engineer mode | tabs *Error vs uncertainty*, *Wheel-slip learning* | "Actual error in this outage stays below one metre; the model uncertainty is deliberately conservative. The learned slip scale settles near 0.885: the filter scales the wheel speed down by about 11.5 % to account for slip." |
| 1:25 | GPS Failure Simulator | n/a | "Across six held-out runs, a 30 second outage: odometry + GPS drifts 8.9 m worst case, adding the IMU 6.1 m, adding slip learning 3.7 m." |
| 1:45 | Sensor Cost Comparison | n/a | "Slip learning is software only: same sensors as odometry + IMU + GPS." |
| 2:00 | n/a | n/a | "Decision support from computed results, not a certified safety system." |

**Do not claim:** real spraying, detected workers, robot commands, calibrated uncertainty, or independently
validated slip learning.

## Alternative: backend EKF, 06_14B

Setup: start the app (`.venv\Scripts\python -m streamlit run app.py --server.port 8510`). Before judges arrive, open
**Sensor Cost Comparison → Live · 06_14B** once (warms the cache) and, in Engineer mode, click **Build Engineer
recording** once (the .rrd is reused). Return to **Operator** mode, source **06_14B · real**. Browser ≥ 1366 px wide.

| time | screen | action | say |
|---|---|---|---|
| 0:00 | Mission Control (Operator) | n/a | "Orchard canopy blocks GPS. Rows are about 20 feet wide, so the robot needs about a foot of accuracy. This is a real robot run from the CitrusFarm dataset; the trees are drawn for context." |
| 0:10 | | click **▶ Run GPS Failure Demo** | "The robot is driving normally: green status, GPS available, the sprayer treating trees blue." |
| 0:15 | (playing) | point at status + GPS tile | "We cut GPS for 60 seconds, like under dense canopy. It switches to wheel and motion sensors and the status goes yellow." |
| 0:25 | (playing) | point at bubble + buffer ring | "The bubble is how unsure the robot thinks it is (its own estimate, not yet calibrated). It grows, the safety buffer around the crew grows with it, and at 7 seconds it says **Stop recommended**, before the real error reaches a foot." |
| 0:35 | | point at checklist + alerts | "Operators don't need robotics terms: what the robot can do now, and alerts in English and Spanish." |
| 0:45 | (GPS restored) | point at pale trees | "GPS comes back, green again. The pale trees are ones a drifting robot would have missed. This is why localization matters for spraying. All simulated, and labelled that way." |
| 0:55 | | click chip **■ Stop recommended · 107s** | "Every event is a timestamp you can jump back to." |
| 1:05 | | **Inspect this event in Engineer mode →**, tab **Rerun replay → Open embedded web viewer** | "Engineers get a full Rerun replay: every method, the RTK truth, error, uncertainty, GPS checks, gyro bias." |
| 1:25 | Sensor Cost Comparison | n/a | "Which sensors to buy? Same 30-second outage: wheel odometry alone drifts 4.6 m; adding a low-cost IMU cuts it to 1.0 m. LiDAR isn't evaluated yet, and we say so." |
| 1:45 | Orchard Risk Map | n/a | "GPS didn't actually degrade on this route: the receiver held RTK the whole time. We report that honestly; the canopy prediction is shown on a labelled synthetic orchard." |
| 2:00 | n/a | n/a | "Measurable, predictive, worker-aware decision support, not a certified safety system." |

**If asked about accuracy claims:** the bubble is the estimator's own 95 % region and is **not yet calibrated**
(53 % empirical coverage on 06_14B, traced mainly to a reference-frame scale mismatch under investigation; see
`docs/UNCERTAINTY_CALIBRATION.md`). Spray results are a simulated scenario showing *why* drift matters, not measured
agricultural performance. Sequences other than 06_14B use a simulated GPS proxy.
