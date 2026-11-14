"""Orchard Robot Localization — Mission Control dashboard (UI layer).

Run (from the project root):
    .venv\\Scripts\\python -m streamlit run app.py

All estimation, simulation and safety logic comes from the localization
backend in src/ (see dashboard/backend.py).
"""
import streamlit as st

st.set_page_config(page_title="Orchard Robot Mission Control", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import backend as be  # noqa: E402
from dashboard import state as S  # noqa: E402
from dashboard import theme  # noqa: E402
from dashboard import opui  # noqa: E402
from dashboard import team_ui  # noqa: E402

theme.inject()
S.init()

pages = [
    st.Page("views/mission_control.py", title="Mission Control", icon=":material/satellite_alt:", default=True,
            url_path="mission-control"),
    st.Page("views/gps_simulator.py", title="GPS Failure Simulator", icon=":material/signal_disconnected:",
            url_path="gps-simulator"),
    st.Page("views/sensor_comparison.py", title="Sensor Cost Comparison", icon=":material/sensors:",
            url_path="sensor-comparison"),
    st.Page("views/risk_map.py", title="Orchard Risk Map", icon=":material/forest:", url_path="risk-map"),
    st.Page("views/worker_safety.py", title="Worker Safety", icon=":material/engineering:", url_path="worker-safety"),
]
nav = st.navigation(pages, position="top")
st.session_state["_current_page"] = nav.title   # read by fragments to detect a late rerun after a page switch

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown(
        "<div style='display:flex;gap:10px;align-items:center;margin:-6px 0 6px 0'>"
        "<div style='width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#199e70,#3987e5);"
        "display:flex;align-items:center;justify-content:center;font-size:1.2rem'>🌿</div>"
        "<div><div style='font-weight:800;color:#eef2ef;line-height:1.1'>Orchard Robot<br>Mission Control</div>"
        "<div style='color:#8a948e;font-size:.72rem'>Team TBD · UIUC Ag Hackathon · Track 3</div></div></div>",
        unsafe_allow_html=True)

    st.segmented_control("User mode", ["Operator", "Engineer"], key="view_mode", required=True,
                         help="Operator: status, alerts, map and actions. Engineer: adds trajectory comparison, "
                              "fusion diagnostics, localization errors and charts.")
    st.segmented_control("Localization results", list(team_ui.SOURCES), key="results_src", required=True,
                         on_change=lambda: st.session_state.update(playing=None, demo_end=None),
                         help="Backend EKF: our estimator runs live on the selected CitrusFarm data. Team Slip-Learning "
                              "Results: exported trajectories from the team's slip-learning localization (odometry + IMU + GPS with learned "
                              "wheel-slip correction) replayed as supplied.")
    if team_ui.active():
        seqs = team_ui.sequences()
        if seqs:
            if st.session_state.team_seq not in seqs:
                st.session_state.team_seq = seqs[0]
            st.selectbox("Sequence", seqs, format_func=team_ui.seq_label, **S.bind("team_seq", "sb"),
                         help="CitrusFarm sequences with exports in team_runs/slip_learning/.")
        st.caption("🟪 Slip-learning estimates · RTK reference · simulated outages, workers and spray")
    else:
        keys = be.source_keys()
        st.selectbox("Data source", keys, key="source", format_func=be.source_label,
                     help="Real CitrusFarm sequences use recorded sensor streams. The synthetic orchard is a labelled "
                          "demo scenario for features that need data the team has not extracted yet.")
        info = be.source_info(st.session_state.source)
        st.caption(f"{'🟦 Real' if info.provenance == 'real' else '🟨 Synthetic'} · {info.streams_text}")

    if S.is_engineer() and not team_ui.active():
        with st.expander("Integrity thresholds", expanded=False):
            st.slider("Trusted if 95 % uncertainty ≤ (m)", 0.1, 1.0, step=0.05, **S.bind("trusted_r95", "sb"))
            st.slider("Stop if 95 % uncertainty > (m)", 0.3, 3.0, step=0.1, **S.bind("stop_r95", "sb"))
            st.caption("Defaults from the team plan: 0.3 m target, 1.0 m stop (to be confirmed by the domain lead).")
            if st.session_state.stop_r95 <= st.session_state.trusted_r95:
                st.warning("Stop limit must be above the trusted limit.")

    st.divider()
    st.markdown("<div class='card-label'>Navigation status</div>" + "".join(
        f"<div style='display:flex;gap:8px;align-items:center;margin:4px 0'><span class='sw' style='width:14px;height:14px;"
        f"border-radius:4px;display:inline-flex;align-items:center;justify-content:center;font-size:.6rem;color:#0b0b0b;"
        f"font-weight:900;background:{v['color']}'>{v['icon']}</span><div><b style='color:#eef2f5;font-size:.85rem'>"
        f"{v['label']}</b><div class='small'>{v['meaning']}</div></div></div>"
        for v in (team_ui.STATE_KEY if team_ui.active() else theme.STATE).values()),
        unsafe_allow_html=True)
    with st.expander("Data labels", expanded=False):
        st.markdown(opui.kind_legend_html(), unsafe_allow_html=True)
    st.divider()
    st.caption("Data: CitrusFarm (Teng et al., UC Riverside, 2023), CC BY-SA 4.0. Decision support only — "
               "not a certified safety system.")

if st.session_state.stop_r95 <= st.session_state.trusted_r95:
    st.session_state.stop_r95 = round(st.session_state.trusted_r95 + 0.2, 2)

nav.run()
