# Two-minute demo script

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
