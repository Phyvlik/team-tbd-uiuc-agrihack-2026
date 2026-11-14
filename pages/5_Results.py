from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sensor_comparison import CONFIG_LABEL, COST_TIERS
from src.ui import C_REF, C_SERIES, page_setup

page_setup("Results and Evaluation")
RES = Path(__file__).resolve().parents[1] / "results"

st.markdown("All tables on this page are read from `results/*.csv`, produced by `scripts/validate_baseline.py` and "
            "`scripts/run_experiments.py` from the official CitrusFarm files. Re-run those scripts to reproduce them. "
            "Synthetic rows are labelled and never pooled with real-data rows.")


def load(name):
    p = RES / name
    if not p.exists():
        st.warning(f"`results/{name}` not found — run the scripts listed above.")
        return None
    return pd.read_csv(p)


# ---- SP1 ---------------------------------------------------------------------
st.header("1 · Wheel-odometry baseline (SP1) — validated on all 7 sequences")
bv = load("baseline_validation.csv")
if bv is not None:
    st.markdown(
        "B1 = integrate wheel speed and yaw rate; initial pose from the first 2 m of the reference; GPS-antenna "
        "lever arm (−0.18 m) applied. Variants isolate **processing artefacts** from **genuine drift**: V0 uses the "
        "controller's pose without frame alignment (a common pitfall), D is a full-trajectory rigid fit "
        "(**diagnostic only — uses future data**).")
    piv = bv.pivot(index="sequence", columns="variant", values="max_m")
    fig = go.Figure()
    order = ["V0_naive_ctrl_pose", "V2_B1_aligned", "V3_B1_scale_calibrated_LOO", "D_B1_oracle_SE2_diagnostic"]
    names = ["V0 naive controller pose", "V2 B1 aligned (headline)", "V3 B1 + LOO speed calibration",
             "D oracle rigid fit (diagnostic)"]
    for i, (v, n) in enumerate(zip(order, names)):
        if v in piv:
            fig.add_trace(go.Bar(x=piv.index, y=piv[v], name=n, marker=dict(color=C_SERIES[i], line=dict(width=0))))
    fig.update_layout(barmode="group", height=360, yaxis_title="max position error (m)", bargap=0.25, bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    head = bv[bv.variant == "V2_B1_aligned"][["sequence", "duration_s", "distance_m", "mean_m", "rmse_m", "max_m",
                                             "final_m", "final_drift_pct_of_distance", "speed_scale_ratio",
                                             "yaw_rate_gain"]]
    st.dataframe(head.round(3), hide_index=True, use_container_width=True)
    st.info("**Finding:** wheel odometry alone drifts tens to hundreds of metres on every sequence. Wheel distance is "
            f"consistently {100 * (bv.speed_scale_ratio.mean() - 1):.0f}% longer than the RTK path, and wheel yaw rate "
            f"under-reads turns (gain {bv.yaw_rate_gain.min():.2f}–{bv.yaw_rate_gain.max():.2f}). Calibrating both on "
            "other sequences does not fix it consistently, so the dominant error is random skid-steer slip. Using the "
            "raw controller pose without frame alignment adds a further artefact on some sequences (e.g. 06_14B: "
            "181 m vs 46 m).")
    with st.expander("All variants"):
        st.dataframe(bv.round(3), hide_index=True, use_container_width=True)

# ---- SP2 / SP5 --------------------------------------------------------------------
st.header("2 · Sensor fusion & reduced-sensor configurations (SP2, SP5)")
cc = load("config_comparison.csv")
if cc is not None:
    st.dataframe(cc.drop(columns=["generated"], errors="ignore").round(3), hide_index=True, use_container_width=True)
    st.caption("C1–C3 never receive GPS (pure dead reckoning for the whole run). 'NOT RUN' = the required stream has "
               "not been extracted yet; nothing is simulated in its place. gps_source = proxy_from_reference means "
               "live GPS was simulated from the reference + 3 cm noise (raw fixes not yet extracted).")

# ---- SP3 ----------------------------------------------------------------------
st.header("3 · GPS outage robustness (SP3)")
ot = load("outage_robustness.csv")
if ot is not None and len(ot):
    real = ot[ot.role != "synthetic"]
    for title, df in [("Real CitrusFarm — held-out sequences", real[real.role == "held-out"]),
                      ("Real CitrusFarm — tuning sequence (02_13B)", real[real.role == "tuning"]),
                      ("SYNTHETIC demo (not real-world evidence)", ot[ot.role == "synthetic"])]:
        if df.empty:
            continue
        st.subheader(title)
        g = (df.groupby(["config", "dead_reckoning", "duration_s"])
             .agg(peak_err_median_m=("max_err_m", "median"), peak_err_max_m=("max_err_m", "max"),
                  pct_within_target=("pct_within_target", "mean"), time_to_exceed_s=("time_to_exceed_s", "median"),
                  recovery_s=("recovery_s", "median"), runs=("max_err_m", "size"), gps_source=("gps_source", "first"))
             .reset_index())
        fig = go.Figure()
        for i, cid in enumerate(sorted(g.config.unique())):
            s = g[g.config == cid]
            fig.add_trace(go.Scatter(x=s.duration_s, y=s.peak_err_median_m, mode="lines+markers",
                                     name=f"{cid}: {CONFIG_LABEL[cid]} + GPS", line=dict(color=C_SERIES[i], width=2),
                                     marker=dict(size=9)))
        fig.add_hline(y=0.3, line=dict(color=C_REF, dash="dash"), annotation_text="0.3 m target")
        fig.update_layout(height=320, yaxis_type="log", xaxis_title="outage duration (s)",
                          yaxis_title="median peak error (m, log)", hovermode="x unified",
                          legend=dict(orientation="h", y=1.18), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g.round(3), hide_index=True, use_container_width=True)

# ---- SP4 -------------------------------------------------------------------------
st.header("4 · Do warnings match real errors? (SP4)")
wp = load("warning_performance.csv")
if wp is not None and len(wp):
    g = (wp.groupby(["role", "duration_s"])
         .agg(detect_latency_s=("detect_latency_s", "median"), stop_latency_s=("stop_latency_s", "median"),
              error_crosses_target_s=("error_crosses_target_s", "median"), lead_time_s=("lead_time_s", "median"),
              recall_warn_when_bad=("recall_warn_when_bad", "mean"), false_alarm_rate=("false_alarm_rate", "mean"),
              missed_hazard_rate=("missed_hazard_rate", "mean"), coverage_r95_pct=("coverage_r95_pct", "mean"),
              runs=("duration_s", "size"))
         .reset_index())
    st.dataframe(g.round(3), hide_index=True, use_container_width=True)
    st.caption("detect_latency = time from GPS loss to first DEGRADED; lead_time = how long before the true error "
               "crossed 0.3 m the warning appeared (positive = warned in advance). recall/false-alarm are per "
               "reference sample over the whole run with 'bad' = true error > 0.3 m.")

# ---- limitations ----------------------------------------------------------------
st.header("5 · Limitations (read before quoting any number)")
st.markdown("""
- **Live GPS is currently a simulated proxy** (reference + 3 cm white noise) for real sequences, because raw
  `/piksi/navsatfix_best_fix` lives in `base_*.bag` (2.5–4.2 GB each) which has not been extracted yet. Outage
  results measure dead-reckoning drift *between* GPS fixes, which does not depend on the proxy; results *with* GPS
  are optimistic until raw fixes are used.
- **IMU and LiDAR are not yet extracted** for real sequences → C2/C3 are reported as NOT RUN on real data.
- **Outages are simulated** (fixes withheld); real canopy degradation is not yet measured (needs receiver state).
- EKF noise parameters were tuned on 02_13B and checked on the other sequences; 7 sequences from one research
  station (UC Riverside) — results may not generalise to other orchards, seasons or robots.
- Worker zones are synthetic; there is no person detection. This is decision support, not a certified safety system.
- Synthetic-mode numbers come from invented error models and are not evidence of real-world performance.
""")
