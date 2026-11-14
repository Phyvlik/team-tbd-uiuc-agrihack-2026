"""Mission Control — Farmer / Operator dashboard (3D orchard centerpiece) and
Engineer mode (Rerun replay + diagnostics).

Everything in the live panel is driven by ONE replay clock (session key
``mc_t``): 3D/2D view, navigation state, four metrics, checklist and alerts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import adapter as ad
from dashboard import backend as be
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import opui, ops
from dashboard import orchard as orc
from dashboard import replay
from dashboard import rerun_export as rx
from dashboard import scene3d
from dashboard import scenario as scn
from dashboard import state as S
from dashboard import team_ui
from dashboard.theme import STATE

ss = st.session_state
if team_ui.active():          # Team Slip-Learning Results: replay the teammate's exports
    team_ui.mission_control()
    st.stop()
key = ss.source
streams = be.load_streams(key)
info = be.source_info(key)
t0, t1 = streams.time_span()
eng = S.is_engineer()
trusted, stop = S.thresholds()

# ------------------------------------------------------------- header -----
kinds = (opui.kind("recorded") if info.provenance == "real" else opui.kind("simulated", "Synthetic data")) \
    + opui.kind("computed") + opui.kind("simulated", "Simulated outage · crew · spray") \
    + opui.kind("illustrative", "Illustrative trees")
ui.page_header(
    f"{'Engineer' if eng else 'Operator'} view · {info.label}", "Mission Control",
    "Where is the robot, can we trust that position, and what should happen next?",
    [ui.provenance_badge(info.provenance), kinds])

tb = opui.experiment_toolbar("mc", streams, demo_page="mc")
if tb["demo"]:
    opui.start_demo("mc", "mc_t", streams)
    st.rerun()
if tb["reset"]:
    opui.reset_demo("mc_t")
    st.rerun()

sens = be.METHODS[ss.method]
outages, jumps = S.scenario_outages(), S.scenario_jumps()
res = be.experiment(key, sens, outages, jumps, trusted, stop)
est, states = res.estimate, res.states
t_max = float(est.t.iloc[-1] - t0)
gps_series = ops.gps_status_series(est)
spray_ev, t_pass = scn.spray(key, sens, outages, jumps, trusted, stop)
orchard = scn.orchard(key)
dec, zones = (scn.safety(key, sens, outages, jumps, trusted, stop, float(ss.ws_min_buffer), float(ss.ws_k),
                         float(ss.ws_slow), 2.5) if ss.show_crew else (None, None))
events = ops.events(est, states, gps_series, dec, res.outages_abs, t0)
speed_label = "estimated by the EKF from wheel odometry" + (" + IMU" if "imu" in sens else "")
caveat = be.reference_caveat(key, res.gps_source)


def rerun_operator_embed(sit):
    ok, ver = rx.available()
    if not ok:
        ui.not_evaluated("Rerun 3D", f"rerun-sdk is not available in this environment ({ver}).",
                         "run the app with this folder's .venv (see README)")
        return
    st.markdown(f"<div class='small'>Rerun {ver} web viewer · simplified scene (engineering panels hidden). "
                f"<b>Independent playback:</b> Rerun has its own timeline, not linked to the replay slider — "
                f"the same moment is <b>time = {sit.t_rel:.0f} s</b> there.</div>", unsafe_allow_html=True)
    want = (key, ss.method, outages, jumps)
    if ss.get("rrd_op_for") != want or not ss.get("rrd_op_url"):
        if st.button("Load simplified Rerun scene", type="primary"):
            with st.spinner("Writing operator .rrd and starting Rerun's web viewer…"):
                p = rx.build_recording(key, ss.method, outages, jumps, trusted, stop, kind="operator", zones=zones,
                                       safety_cfg=be.SafetyConfig(min_buffer_m=float(ss.ws_min_buffer),
                                                                  k_uncertainty=float(ss.ws_k),
                                                                  slow_margin_m=float(ss.ws_slow)))
                ss.rrd_op, ss.rrd_op_for = str(p), want
                ss.rrd_op_url = rx.serve_web(p, 9091, 9878, slot="op")
            st.rerun()
        return
    st.iframe(ss.rrd_op_url, height=470)


def live_panel():
    if team_ui.active() or ss.get("_current_page", "Mission Control") != "Mission Control":
        # A late rerun of this fragment (a clock report from the player that was on screen before
        # a source or page switch) must not draw the EKF panel into another view: redraw the page.
        st.rerun()
    if ss.viz == "Rerun 3D":                # no client-side player in this view: server clock (time-based)
        ui.advance("mc", "mc_t", t_max, ss.speed)
    S.clamp("mc_t", 0.0, t_max)

    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        bar = st.container(horizontal=True, gap="small", vertical_alignment="center")
        bar.segmented_control("View", ["3D orchard", "2D map", "Rerun 3D"], required=True,
                              label_visibility="collapsed", **S.bind("viz", "mc"),
                              help="3D orchard and 2D map follow the replay slider. Rerun 3D is Rerun's own viewer "
                                   "with its own timeline (independent playback).")
        if ss.viz == "3D orchard":
            bar.segmented_control("Camera", ["Follow robot", "Whole orchard"], required=True,
                                  label_visibility="collapsed", **S.bind("view3d", "mc"))
        elif ss.viz == "2D map":
            bar.segmented_control("Map view", ["Follow robot", "Whole route"], required=True,
                                  label_visibility="collapsed", **S.bind("map_view", "mc"))
        if ss.viz != "Rerun 3D":
            bar.segmented_control("Speed", list(replay.SPEEDS), format_func=lambda v: f"{v}×", required=True,
                                  label_visibility="collapsed", **S.bind("speed", "mc"))
            # client-side player: animates in the browser and reports its time back (dashboard/replay.py)
            crew = (float(ss.ws_min_buffer), float(ss.ws_k), float(ss.ws_slow), 2.5) if ss.show_crew else None
            replay.player("mc", "mc_t", replay.ekf_scene(key, ss.method, outages, jumps, trusted, stop, crew, eng),
                          view="3d" if ss.viz == "3D orchard" else "2d",
                          follow=(ss.view3d if ss.viz == "3D orchard" else ss.map_view) == "Follow robot")
        t_now = t0 + float(ss.mc_t)             # the player's time: everything below uses the same clock
        i = be.at_time(est, t_now)
        sit = ops.situation(est, states, gps_series, dec, i, t0, res.outages_abs)
        if ss.viz == "3D orchard":
            status = orc.spray_status(len(orchard.trees), spray_ev, t_pass, t_now)
            counts = {k: int((status == k).sum()) for k in (orc.TREATED, orc.MISSED, orc.DOUBLE)}
            st.markdown(
                "<div class='legend'>"
                + "".join(f"<span class='l'><span class='sw' style='background:{STATE[s]['color']}'>"
                          f"{STATE[s]['icon']}</span>Trail · {STATE[s]['label']}</span>" for s in STATE)
                + "<span class='l'><span class='sw' style='background:rgba(238,242,239,.25)'></span>Bubble = 95 % "
                  "uncertainty</span><span class='l'>◇ Crew (simulated) · dashed ring = safety buffer</span>"
                + "".join(f"<span class='l'><span class='sw' style='background:{c}'></span>{lab} {n}</span>"
                          for lab, c, n in (("Treated", scene3d.SPRAY_COLOR[orc.TREATED], counts[orc.TREATED]),
                                            ("Missed", scene3d.SPRAY_COLOR[orc.MISSED], counts[orc.MISSED]),
                                            ("Double", scene3d.SPRAY_COLOR[orc.DOUBLE], counts[orc.DOUBLE])))
                + f"</div><div class='small'>{opui.kind('illustrative')} {ui.esc(orchard.note)} "
                  f"{opui.kind('simulated', 'Simulated spray')} sprayer follows the estimated position "
                  f"(rules in dashboard/orchard.py). Robot drawn 2× size.</div>", unsafe_allow_html=True)
        elif ss.viz == "2D map":
            ui.status_legend([("True path (Engineer only)", "#c3c2b7", "dot"), ("GPS fixes", "#3987e5", "dot")],
                             map_items=True)
        else:
            rerun_operator_embed(sit)
        ui.playback_controls("mc", "mc_t", t_max, label="Replay time",
                             jump_to=[("⏮ GPS loss", res.outages_abs[0][0] - t0 - 3)] if res.outages_abs else None)
        opui.event_chips(events, "mc", "mc_t", t_max)

    with right:
        lvl_arr = states.level.to_numpy()
        held = ops.time_in_state(np.where(lvl_arr == lvl_arr[i], 1, 0), est.t.to_numpy(), i, 1)
        ui.status_hero(sit.nav_state, sit.nav_reason, label=f"Navigation state · t = {sit.t_rel:.0f} s",
                       extra=f" · for {held:.0f} s" if held >= 1 else "")
        st.markdown(opui.operator_tiles(sit, trusted, stop, ui.uncertainty_gauge(sit.r95, trusted, stop), speed_label,
                                        res.gps_source), unsafe_allow_html=True)
        if eng:
            e_now = res.error[res.error.t <= t_now]
            st.caption(f"Actual error vs RTK at this moment (replay only): "
                       f"**{ui.fmt(float(e_now.err.iloc[-1]) if len(e_now) else np.nan, 2, ' m', '—')}**"
                       + (" · ⚠ includes the known reference-frame discrepancy" if caveat else ""))
        opui.explainer()

    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1:
        ui.section("What the robot can do now", "Advisory · updates with the replay")
        st.markdown(opui.checklist_html(ops.capabilities(sit)), unsafe_allow_html=True)
    with c2:
        h = st.container(horizontal=True, vertical_alignment="center")
        h.markdown("<div class='sec'>Alerts · Alertas</div>", unsafe_allow_html=True)
        h.segmented_control("Language", ["English", "Español", "English + Español"], required=True,
                            label_visibility="collapsed", **S.bind("lang", "mc"))
        st.markdown(opui.alerts_html(ops.active_alerts(sit), ss.lang), unsafe_allow_html=True)
        st.caption("Spanish drafted for the demo — pending native-speaker review.")


# 3D orchard / 2D map: reruns come from the player's clock reports. Rerun 3D keeps a timer.
st.fragment(run_every=0.5 if (ss.playing == "mc" and ss.viz == "Rerun 3D") else None)(live_panel)()

if not eng:
    st.markdown("<div class='small' style='margin-top:6px'>Switch to <b>Engineer</b> mode (sidebar) for the Rerun "
                "replay, error timelines, sensor-fusion diagnostics and method comparison.</div>",
                unsafe_allow_html=True)
    st.button("Inspect this event in Engineer mode →", on_click=lambda: ss.update(view_mode="Engineer"))
    st.stop()

# ============================================================ ENGINEER ===
st.divider()
t_now = t0 + float(ss.mc_t)
tabs = st.tabs(["Rerun replay", "Error vs uncertainty", "Trajectory comparison", "Sensor fusion & GPS", "Event log",
                "Team data (adapter)", "Run metrics & provenance"])

with tabs[0]:
    ok, ver = rx.available()
    if not ok:
        ui.not_evaluated("Rerun replay", f"rerun-sdk not importable ({ver}).", "run with this folder's .venv")
    else:
        team_runs, _ = ad.load_team_runs(key)
        st.markdown(f"Detailed **Rerun {ver}** recording of this scenario: RTK reference, every implemented method, "
                    f"wheel-odometry baseline, GPS outage markers, error and uncertainty timelines, GPS status/quality/"
                    f"NIS with rejections, gyro bias, odometry scale, odometry-vs-IMU disagreement and an event log, "
                    f"on one timeline (`time` = seconds since start, same clock as the slider: now **{ss.mc_t:.1f} s**).")
        st.caption("Rerun runs as a separate local viewer. The Streamlit slider does not move Rerun's cursor — "
                   "playback is independent. LiDAR point clouds: Data unavailable.")
        b = st.container(horizontal=True, gap="small")
        want = (key, ss.method, outages, jumps, len(team_runs))
        if b.button("Build Engineer recording (.rrd)", type="primary"):
            with st.spinner("Logging all methods and diagnostics to Rerun…"):
                p = rx.build_recording(key, ss.method, outages, jumps, trusted, stop, kind="engineer", zones=zones,
                                       imported=team_runs)
                ss.rrd_eng, ss.rrd_eng_for, ss.rrd_eng_url = str(p), want, None
        if ss.rrd_eng and ss.get("rrd_eng_for") != want:
            st.warning("The scenario changed since the last recording — rebuild to match.")
        if ss.rrd_eng and Path(ss.rrd_eng).exists():
            p = Path(ss.rrd_eng)
            b.download_button("⬇ Download .rrd", p.read_bytes(), file_name=p.name, mime="application/octet-stream")
            if b.button("Open embedded web viewer"):
                with st.spinner("Starting Rerun web viewer…"):
                    ss.rrd_eng_url = rx.serve_web(p, 9090, 9877, slot="eng")
            if b.button("Open native Rerun app"):
                rx.open_native(p)
                st.toast("Rerun desktop viewer launched")
            st.caption(f"Saved: `{p}` · reproduce: `python scripts/export_rerun.py --source {key} --method {ss.method}"
                       + (f" --outage {ss.outage_start:.0f}:{ss.outage_dur:.0f}" if ss.outage_dur else "") + "`")
        if ss.get("rrd_eng_url"):
            st.iframe(ss.rrd_eng_url, height=760)

with tabs[1]:
    xr = None
    if res.outages_abs:
        a, b_ = res.outages_abs[0]
        if st.toggle("Zoom to outage", value=True, key="mc_zoom_err"):
            xr = (a - t0 - 20, b_ - t0 + 40)
    st.plotly_chart(ch.state_ribbon(res, t0, t_now, xrange=xr), theme=None, key="mc_ribbon",
                    config={"displayModeBar": False})
    st.plotly_chart(ch.error_timeline(res, t0, t_now, trusted, stop, height=330, xrange=xr), theme=None, key="mc_err")
    st.caption("Blue = the estimator's own 95 % uncertainty (known in real time, UNCALIBRATED). Orange = actual "
               "error vs RTK (replay only). A calibrated estimator keeps blue above orange ~95 % of the time.")
    if caveat:
        ui.notice("⚠ " + ui.esc(caveat), "")
    if res.warnings.get("per_outage"):
        w = res.warnings["per_outage"][0]
        o = res.outage_table.iloc[0]
        ui.kpi_row([
            ui.kpi_html("Warned (Degraded) after", ui.fmt(w["detect_latency_s"], 1, " s"), "from GPS loss"),
            ui.kpi_html("Stop recommended after", ui.fmt(w["stop_latency_s"], 1, " s"), "from GPS loss"),
            ui.kpi_html("True error > 0.3 m after", ui.fmt(w["error_crosses_target_s"], 1, " s", "never"),
                        "from GPS loss"),
            ui.kpi_html("Warning lead time", ui.fmt(w["lead_time_s"], 1, " s", "—"), "warning before error crossed"),
            ui.kpi_html("Peak error in outage", ui.fmt(o.max_err_m, 2, " m"), f"{o.duration_s:.0f} s outage"),
            ui.kpi_html("Recovery after GPS returns", ui.fmt(o.recovery_s, 1, " s"), "back within target"),
        ])

with tabs[2]:
    cfg_of = {"odom+gps": "C1", "odom+imu+gps": "C2", "odom+imu+lidar+gps": "C3"}
    runs = {}
    for m in be.methods_for(streams):
        runs[f"{cfg_of[m]} core · {be.METHOD_LABEL[m]}"] = (cfg_of[m], be.experiment(key, be.METHODS[m], outages,
                                                                                      jumps, trusted, stop))
    if len(runs) < 2:
        ui.not_evaluated("multi-sensor comparison", "only wheel odometry is available for this sequence.",
                         "imu.csv / lidar_odom.csv — see docs/TEAMMATE_DATA_REQUESTS.md")
    win = ch.outage_window(res, streams) if res.outages_abs else None
    cc = st.columns([1.2, 1])
    cc[0].plotly_chart(ch.compare_trajectories(streams, runs, height=460, window=win), theme=None, key="mc_cmp_map")
    cc[1].plotly_chart(ch.compare_errors(runs, t0, res.outages_abs, trusted, height=460,
                                         xrange=(res.outages_abs[0][0] - t0 - 20, res.outages_abs[0][1] - t0 + 40)
                                         if res.outages_abs else None), theme=None, key="mc_cmp_err")
    b1 = be.experiment(key, ("odom",), (), (), trusted, stop)
    rows = []
    for label, (cid, r) in list(runs.items()) + [("B1 · wheel odometry only, no GPS (baseline)", ("B1", b1))]:
        o = r.outage_table.iloc[0] if len(r.outage_table) else None
        rows.append({"method": label, "mean error (m)": r.metrics["mean_m"], "RMSE (m)": r.metrics["rmse_m"],
                     "max error (m)": r.metrics["max_m"], "within 0.3 m (%)": r.metrics["pct_time_within_target"],
                     "outage peak (m)": None if o is None else o.max_err_m,
                     "truth inside 95 % radius (%)": r.metrics["coverage_r95_pct"],
                     "GPS fixes rejected": r.metrics["gps_rejected"]})
    ui.df_table(pd.DataFrame(rows).round(3))
    st.caption("C3 (LiDAR): Not evaluated — no lidar_odom.csv for any real sequence. Cost vs performance: see the "
               "Sensor Cost Comparison page.")

with tabs[3]:
    d = st.columns(2)
    with d[0]:
        ui.section("GPS consistency check (NIS)", "Fixes far from the prediction are rejected")
        st.plotly_chart(ch.gps_consistency(res, t0, height=250), theme=None, key="mc_nis")
    with d[1]:
        ui.section("GPS status", "1 = available · 0.5 = degraded (rejection) · 0 = lost")
        gnum = np.select([gps_series == ops.GPS_AVAILABLE, gps_series == ops.GPS_DEGRADED], [1.0, 0.5], 0.0)
        st.plotly_chart(ch.series_over_time(est.t - t0, gnum, "GPS status", height=250, color="#3987e5"), theme=None,
                        key="mc_gps")
    d = st.columns(2)
    with d[0]:
        if "imu" in sens:
            ui.section("Gyro bias estimate (b_g)", "rad/s · EKF state")
            st.plotly_chart(ch.series_over_time(est.t - t0, est.b_g, "b_g (rad/s)", height=230, color="#199e70"),
                            theme=None, key="mc_bg")
        else:
            ui.not_evaluated("gyro bias", "the IMU is not fused in the selected method.")
    with d[1]:
        if streams.imu is not None and len(streams.imu):
            ui.section("Sensor disagreement: wheel vs IMU yaw rate",
                       "rad/s, 1 s mean · recorded data · wheel slip shows as spikes")
            o_ = streams.odom
            wz = np.interp(o_.t, streams.imu.t, streams.imu.wz)
            dis = pd.Series(o_.w.to_numpy() - wz).rolling(50, center=True, min_periods=5).mean()
            st.plotly_chart(ch.series_over_time((o_.t - t0)[::10], dis[::10], "odom − IMU (rad/s)", height=230,
                                                color="#c98500", hline=0.0), theme=None, key="mc_dis")
        else:
            ui.not_evaluated("sensor disagreement", "needs both wheel odometry and IMU; no imu.csv for this sequence.")
    d = st.columns(2)
    with d[0]:
        ui.section("Online wheel-speed scale (k_odom)", "Learned while GPS is available")
        st.plotly_chart(ch.series_over_time(est.t - t0, est.k_odom, "k_odom", height=220, hline=1.0), theme=None,
                        key="mc_k")
    with d[1]:
        gq = streams.gps_quality
        if gq is not None and "num_sat" in gq:
            ui.section("Receiver: satellites tracked", "recorded /piksi receiver state")
            st.plotly_chart(ch.series_over_time(gq.t - t0, gq.num_sat, "satellites", height=220), theme=None,
                            key="mc_sat")
        elif gq is not None and "h_sigma_m" in gq:
            ui.section("Receiver horizontal σ", "synthetic")
            st.plotly_chart(ch.series_over_time(gq.t - t0, gq.h_sigma_m, "σ (m)", height=220), theme=None,
                            key="mc_sig")
        else:
            ui.not_evaluated("GPS receiver quality", "no gps_quality.csv for this sequence.")

with tabs[4]:
    log = events.copy()
    log["time (s)"] = log.t_rel.round(1)
    log["severity"] = log.state.map(lambda s_: f"{STATE[s_]['icon']} {STATE[s_]['label']}")
    tr = be.transitions(states)
    tr = tr.assign(**{"time (s)": (tr.t - t0).round(1), "detail": tr.reason, "r95 (m)": tr.r95.round(2)})
    full = log[["time (s)", "label", "kind", "severity"]].merge(tr[["time (s)", "detail", "r95 (m)"]], on="time (s)",
                                                                 how="left")
    ui.df_table(full, height=320)
    ev = res.gps_events
    if ev is not None and (~ev.accepted).any():
        rj = ev[~ev.accepted].assign(t=lambda d_: (d_.t - t0).round(1))
        st.markdown(f"**GPS rejections ({len(rj)})**")
        ui.df_table(rj.rename(columns={"t": "time (s)"}).round(3), height=200)
    else:
        st.caption("GPS rejections: none in this run (enable Multipath to test the consistency gate).")

with tabs[5]:
    st.markdown("Teammate runs are read from `team_runs/` or `results/runs/` in the project folder "
                "via the data adapter (`dashboard/adapter.py`, schema in its docstring). Nothing is inferred.")
    team_runs, errs = ad.load_team_runs(None)
    ours = ad.from_experiment(key, be.METHOD_LABEL[ss.method], res)
    ui.df_table(pd.DataFrame([ours.summary()] + [r.summary() for r in team_runs]))
    for name, e in errs:
        st.error(f"{name}: {e}")
    for r in team_runs:
        with st.expander(f"Checks · {r.sequence} · {r.method}"):
            st.markdown("\n".join(f"- {c}" for c in r.checks))
    if not team_runs:
        ui.not_evaluated("teammate runs", "no CSV files in team_runs/ or results/runs/ yet.",
                         "&lt;SEQ&gt;__&lt;method&gt;.csv with t, est_x, est_y [+ covariance, speed, GPS flags …]")

with tabs[6]:
    m = res.metrics
    ui.kpi_row([
        ui.kpi_html("Mean error", ui.fmt(m["mean_m"], 3, " m"), "full run vs RTK"),
        ui.kpi_html("RMSE", ui.fmt(m["rmse_m"], 3, " m"), "full run"),
        ui.kpi_html("Max error", ui.fmt(m["max_m"], 2, " m"), "full run"),
        ui.kpi_html("Time within 0.3 m", ui.fmt(m["pct_time_within_target"], 1, " %"), "full run"),
        ui.kpi_html("Truth inside 95 % radius", ui.fmt(m["coverage_r95_pct"], 1, " %"), "≈95 % = honest uncertainty"),
        ui.kpi_html("Warnings: false-alarm rate", ui.fmt(100 * res.warnings["false_alarm_rate"], 1, " %"),
                    "warning while true error ≤ 0.3 m"),
    ])
    ui.notice(f"<b>Uncertainty status: {be.CALIBRATION_STATUS}.</b> On this run the 95 % radius contained the "
              f"reference {ui.fmt(m['coverage_r95_pct'], 1, ' %')} of the time (a calibrated estimator: ≈95 %). "
              f"Values are shown exactly as the EKF reports them — nothing is rescaled. "
              + (ui.esc(caveat) + " " if caveat else "")
              + f"Details: <code>{be.CALIBRATION_DOC}</code>, <code>scripts/calibration_audit.py</code>.", "")
    sp = orc.spray_status(len(orchard.trees), spray_ev, t_pass, est.t.iloc[-1])
    src = pd.DataFrame([
        ("Data source", info.label), ("Loaded from", info.origin), ("Streams", info.streams_text),
        ("GPS input to EKF", res.gps_source), ("Sensors fused", " + ".join(be.SENSOR_NAMES[s_] for s_ in res.sensors)),
        ("Scenario (simulated)",
         f"{ss.outage_dur:.0f} s outage at {ss.outage_start:.0f} s" if ss.outage_dur else "no outage"),
        ("Integrity thresholds", f"trusted ≤ {trusted} m · stop > {stop} m"),
        ("Trees (illustrative)", f"{len(orchard.trees)} · {orchard.origin}"),
        ("Spray (simulated, full run)", f"{(sp == orc.TREATED).sum()} treated · {(sp == orc.MISSED).sum()} missed · "
                                        f"{(sp == orc.DOUBLE).sum()} double"),
        ("Backend code", "src/ in the project folder" if be.BACKEND_DIR == be.APP_DIR else str(be.BACKEND_DIR)),
    ], columns=["item", "value"])
    ui.df_table(src)
