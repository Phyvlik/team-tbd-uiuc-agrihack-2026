"""GPS Failure Simulator — choose an outage, replay how the location estimate
changes, and compare fallback sensor sets under the identical outage."""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import backend as be
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import opui, ops
from dashboard import state as S
from dashboard import team_ui
from dashboard.theme import GOOD, STATE, WARN

ss = st.session_state
if team_ui.active():          # Team Slip-Learning Results: teammate summary tables
    team_ui.gps_page()
    st.stop()
key = ss.source
streams = be.load_streams(key)
info = be.source_info(key)
t0, t1 = streams.time_span()
dur = t1 - t0
eng = S.is_engineer()
trusted, stop = S.thresholds()
methods = be.methods_for(streams)
if ss.method not in methods:
    ss.method = methods[-1]
if ss.outage_dur == 0:
    ss.outage_dur = 60
S.clamp("outage_start", 10.0, max(10.0, dur - 40.0))

ui.page_header(
    f"Scenario lab · {info.label}", "GPS Failure Simulator",
    "Pick how long GPS drops out under the canopy, then replay how the robot's location estimate and its "
    "confidence change. GPS fixes are removed <i>before</i> the estimator sees them.",
    [ui.provenance_badge(info.provenance), ui.provenance_badge("sim_gps")])

with st.container(border=True):
    c = st.columns([2.6, 2.0, 2.2, 1.3], vertical_alignment="bottom", gap="small")
    c[0].select_slider("GPS outage length (simulated)", [5, 10, 15, 20, 30, 45, 60, 90, 120, 180],
                       format_func=lambda v: f"{v} s",
                       help="The plan's standard suite is 10 / 30 / 60 / 120 s.", **S.bind("outage_dur", "sim"))
    c[1].slider("Outage starts at", 10.0, float(max(11.0, dur - 40.0)), step=5.0, format="%.0f s",
                **S.bind("outage_start", "sim"))
    c[2].selectbox("Robot's localization method", methods, format_func=lambda m: be.METHOD_LABEL[m],
                   **S.bind("method", "sim"))
    c[3].toggle("Multipath", help="Also inject a 2.5 m GPS jump for 4 s shortly before the outage "
                                  "(tree multipath). The EKF consistency gate should reject it.",
                **S.bind("multipath", "sim"))

outages, jumps = S.scenario_outages(), S.scenario_jumps()
res = be.experiment(key, be.METHODS[ss.method], outages, jumps, trusted, stop)
est, states = res.estimate, res.states
a_abs, b_abs = res.outages_abs[0]
a_rel, b_rel = a_abs - t0, b_abs - t0
lo, hi = max(0.0, a_rel - 12.0), min(float(est.t.iloc[-1] - t0), b_rel + 35.0)
if ss.sim_t is None or not (lo <= ss.sim_t <= hi):
    ss.sim_t = a_rel - 3.0
w = res.warnings["per_outage"][0] if res.warnings.get("per_outage") else {}
o = res.outage_table.iloc[0] if len(res.outage_table) else None
sim_events = ops.events(est, states, ops.gps_status_series(est), None, res.outages_abs, t0)


def replay():
    ui.advance("sim", "sim_t", hi, 0.25 * ss.speed)
    S.clamp("sim_t", lo, hi)
    t_now = t0 + float(ss.sim_t)
    i = be.at_time(est, t_now)
    row, srow = est.iloc[i], states.iloc[i]
    phase = ("Before outage", "#8a948e") if t_now < a_abs else (
        ("GPS lost", WARN) if t_now < b_abs else ("GPS restored", GOOD))
    since = t_now - a_abs

    ui.section("Replay", "Scrub or press play — the map zooms to the outage stretch of the route")
    st.slider("Replay time", lo, hi, step=0.2, format="%.1f s", label_visibility="collapsed",
              **S.bind("sim_t", "sim_slider"))
    bar = st.container(horizontal=True, gap="small", vertical_alignment="center")
    bc = [bar] * 4
    playing = ss.playing == "sim"
    if bc[0].button("⏸ Pause" if playing else "▶ Play", type="secondary" if playing else "primary",
                    key="sim_pp"):
        ss.playing = None if playing else "sim"
        if not playing and ss.sim_t >= hi - 0.5:
            ss.sim_t = lo
        st.rerun()
    if bc[1].button("⟲ Before drop", key="sim_b0"):
        ss.sim_t, ss.playing = a_rel - 3.0, None
        st.rerun()
    if bc[2].button("GPS returns", key="sim_b1"):
        ss.sim_t, ss.playing = b_rel - 1.0, None
        st.rerun()
    bc[3].segmented_control("Speed", [1, 2, 4, 10], format_func=lambda s: f"{s}×", required=True,
                            label_visibility="collapsed", **S.bind("speed", "sim"))
    bar.segmented_control("Map view", ["Follow robot", "Whole route"], required=True, label_visibility="collapsed",
                          **S.bind("map_view", "sim"))

    left, right = st.columns([1.9, 1], gap="medium")
    with left:
        follow = 14.0 if ss.map_view == "Follow robot" else None
        fig = ch.trajectory_map(streams, res, t_now, height=520, follow=follow,
                                uirevision=None if follow else f"sim-{key}")
        st.plotly_chart(fig, key="sim_map", theme=None, config={"displaylogo": False, "scrollZoom": True})
        ui.status_legend([("True path (RTK, scoring only)", "#c3c2b7", "dot"), ("GPS fixes received", "#3987e5", "dot")],
                         map_items=True)
    with right:
        ui.status_hero(srow.state, srow.reason, label="Integrity monitor says")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        e_now = res.error[res.error.t <= t_now]
        err_now = float(e_now.err.iloc[-1]) if len(e_now) else np.nan
        cav = be.reference_caveat(key, res.gps_source)
        cards = [
            ui.kpi_html("Phase", f"<span style='color:{phase[1]}'>{phase[0]}</span>",
                        (f"{since:.1f} s into a {ss.outage_dur:.0f} s outage" if a_abs <= t_now < b_abs else
                         (f"{-since:.1f} s until GPS drops" if t_now < a_abs else f"{t_now - b_abs:.1f} s since GPS returned"))),
            ui.kpi_html("Robot's own uncertainty · uncalibrated", f"{row.r95:.2f} m",
                        ui.uncertainty_gauge(float(row.r95), trusted, stop),
                        accent=STATE[srow.state]["color"]),
            ui.kpi_html("Actual drift from truth", ui.fmt(err_now, 2, " m", "—"),
                        "offline check against RTK" + (" (incl. reference-frame discrepancy)" if cav else "")
                        + (" · within 0.3 m ✓" if be.finite(err_now) and err_now <= 0.3
                                                       else " · outside 0.3 m ✕" if be.finite(err_now) else "")),
        ]
        st.markdown("".join(f"<div style='margin-bottom:8px'>{c}</div>" for c in cards), unsafe_allow_html=True)

    opui.event_chips(sim_events[(sim_events.t_rel >= lo) & (sim_events.t_rel <= hi)], "sim", "sim_t", hi)
    st.plotly_chart(ch.state_ribbon(res, t0, t_now, height=86, xrange=(lo, hi)), key="sim_ribbon", theme=None,
                    config={"displayModeBar": False})
    st.plotly_chart(ch.error_timeline(res, t0, t_now, trusted, stop, height=300, xrange=(lo, hi)), key="sim_err",
                    theme=None)


st.fragment(run_every=0.4 if ss.playing == "sim" else None)(replay)()

# --------------------------------------------------------------- outcome ---
st.divider()
ui.section("What happened during this outage")
if o is not None:
    within = o.pct_within_target
    stays = "stayed within" if o.max_err_m <= 0.3 else "drifted beyond"
    sentence = (f"With <b>{be.METHOD_LABEL[ss.method].lower()}</b>, the robot {stays} the 0.3 m target "
                f"(peak drift <b>{o.max_err_m:.2f} m</b>; within target for <b>{within:.0f} %</b> of the "
                f"{o.duration_s:.0f} s outage). The integrity monitor warned after "
                f"<b>{ui.fmt(w.get('detect_latency_s'), 1, ' s', 'never')}</b> and recommended stopping after "
                f"<b>{ui.fmt(w.get('stop_latency_s'), 1, ' s', 'never')}</b>")
    if be.finite(w.get("error_crosses_target_s")):
        sentence += (f"; the real error only crossed 0.3 m after <b>{w['error_crosses_target_s']:.1f} s</b> — "
                     f"the warning came first." if be.finite(w.get("lead_time_s")) and w["lead_time_s"] > 0 else
                     f"; the real error crossed 0.3 m after <b>{w['error_crosses_target_s']:.1f} s</b>.")
    else:
        sentence += "; the real error never crossed 0.3 m (the stop was conservative)."
    _cav = be.reference_caveat(key, res.gps_source)
    if _cav:
        sentence += f"<br><span class='small'>⚠ {ui.esc(_cav)}</span>"
    ui.notice(sentence, "real" if info.provenance == "real" else "syn")

# Same outage, every fallback sensor set
ui.section("Same outage, different fallback sensors",
           "Identical outage window for every configuration; GPS used before and after it")
rows, runs = [], {}
for cid, core in be.FALLBACK.items():
    sens = core + ("gps",)
    miss = be.missing_streams(streams, sens)
    label = f"{cid} · {be.CONFIG_LABEL[cid]}"
    if miss:
        rows.append({"config": cid, "fallback sensors": be.CONFIG_LABEL[cid], "peak drift (m)": np.nan,
                     "within 0.3 m (%)": np.nan, "exceeds 0.3 m after (s)": np.nan, "warned after (s)": np.nan,
                     "stop after (s)": np.nan, "status": f"Not evaluated — missing {', '.join(miss)}"})
        continue
    r = be.experiment(key, sens, outages, jumps, trusted, stop)
    runs[label] = (cid, r)
    oo = r.outage_table.iloc[0]
    ww = r.warnings["per_outage"][0] if r.warnings.get("per_outage") else {}
    rows.append({"config": cid, "fallback sensors": be.CONFIG_LABEL[cid], "peak drift (m)": oo.max_err_m,
                 "within 0.3 m (%)": oo.pct_within_target, "exceeds 0.3 m after (s)": oo.time_to_exceed_s,
                 "warned after (s)": ww.get("detect_latency_s"), "stop after (s)": ww.get("stop_latency_s"),
                 "status": "measured" if info.provenance == "real" else "synthetic"})
tab = pd.DataFrame(rows)
cc = st.columns([1, 1.5] if eng else [1, 1.2])
with cc[0]:
    st.plotly_chart(ch.config_bars(tab.rename(columns={"peak drift (m)": "peak"}), "peak",
                                   f"peak drift in {ss.outage_dur:.0f} s outage (m, log)", height=320),
                    theme=None, key="sim_bars")
with cc[1]:
    if eng:
        st.plotly_chart(ch.compare_errors(runs, t0, res.outages_abs, trusted, height=320, xrange=(lo, hi)),
                        theme=None, key="sim_cmp")
    else:
        show = tab[["config", "fallback sensors", "peak drift (m)", "within 0.3 m (%)", "status"]].copy()
        show["peak drift (m)"] = show["peak drift (m)"].map(lambda v: ui.fmt(v, 2))
        show["within 0.3 m (%)"] = show["within 0.3 m (%)"].map(lambda v: ui.fmt(v, 0))
        ui.df_table(show)
if eng:
    show = tab.copy()
    for col in show.columns[2:-1]:
        show[col] = show[col].map(lambda v: ui.fmt(v, 2))
    ui.df_table(show)

# Batch evidence from the team's experiments
ui.section("Measured across the team's held-out runs",
           "Peak error vs outage length · median with min–max range · from results/outage_robustness.csv")
br = be.batch_results().get("outage_robustness")
if br is None:
    ui.not_evaluated("batch robustness results", "results/outage_robustness.csv was not found in the backend repo.",
                     "run scripts/run_experiments.py in the backend repo")
else:
    real = br[br.role == "held-out"]
    summ = (real.groupby(["config", "duration_s"]).max_err_m
            .agg(median="median", min="min", max="max", n="size").reset_index())
    cc = st.columns([1.5, 1])
    with cc[0]:
        st.plotly_chart(ch.robustness_chart(summ, height=330), theme=None, key="sim_robust")
    with cc[1]:
        seqs = sorted(real.sequence.unique())
        gps_src = ", ".join(sorted(real.gps_source.unique()))
        st.markdown(ui.badge("■ TEAM BATCH RESULTS", "batch"), unsafe_allow_html=True)
        st.markdown(f"<div class='small' style='margin-top:6px'>{len(real)} runs on {len(seqs)} held-out CitrusFarm "
                    f"sequences · GPS: <code>{gps_src}</code> · generated {be.results_generated() or 'unknown'}"
                    f"</div>", unsafe_allow_html=True)
        missing = [c for c in ("C2", "C3") if c not in set(real.config)]
        if missing:
            ui.not_evaluated(" / ".join(missing) + " on real data in the batch results",
                             "the batch run had no IMU or LiDAR stream for these sequences. "
                             + ("This page's live table above does measure C2 on 06_14B from the extracted IMU."
                                if key == "06_14B_Jackal" else ""),
                             "imu.csv / lidar_odom.csv per sequence, then re-run scripts/run_experiments.py")
        if eng:
            t = summ.copy()
            t["config"] = t.config + " + GPS"
            ui.df_table(t.round(2))
