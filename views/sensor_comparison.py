"""Sensor Cost Comparison — C1–C4: relative cost tier vs measured performance.

Evidence comes from either the team's batch result tables (all real
sequences) or a live run of the backend's compare_configs on the selected
data source. Every value says where it came from; missing → Not evaluated.
"""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import backend as be
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import state as S
from dashboard.theme import CONFIG_COLOR, GOOD
from src.sensor_comparison import recommend  # teammate recommendation rule

ss = st.session_state
key = ss.source
streams = be.load_streams(key)
info = be.source_info(key)
eng = S.is_engineer()

BATCH, LIVE = "Team batch", "Live run"
if ss.cmp_basis not in (BATCH, LIVE):
    ss.cmp_basis = LIVE if (info.provenance == "synthetic" or streams.available().get("imu")) else BATCH

ui.page_header(
    "Planning & purchasing", "Sensor Cost Comparison",
    "Which is the cheapest sensor setup that keeps the robot within 0.3 m when GPS fails? "
    "Cost tiers are relative (qualitative, from the team plan) — not prices.",
    [ui.provenance_badge("batch" if ss.cmp_basis == BATCH else info.provenance)])

with st.container(border=True):
    c = st.columns([2.6, 2.4, 3])
    c[0].segmented_control("Evidence", [BATCH, LIVE], required=True,
                           format_func=lambda v: "Team batch (all seqs)" if v == BATCH else f"Live · {key.replace('_Jackal', '') if key != be.SYNTHETIC else 'synthetic'}",
                           help="Batch = results/*.csv from scripts/run_experiments.py over all real sequences. "
                                "Live = run the backend's compare_configs on the data source chosen in the sidebar.",
                           **S.bind("cmp_basis", "cmp"))
    c[1].segmented_control("GPS outage length", [10, 30, 60, 120], format_func=lambda v: f"{v} s", required=True,
                           **S.bind("cmp_duration", "cmp"))
    with c[2]:
        st.markdown("<div class='small' style='padding-top:1.9rem'>Target: stay within <b>0.3 m</b> "
                    "(≈ 1 ft, the accuracy needed in 20–22 ft orchard rows).</div>", unsafe_allow_html=True)

D = int(ss.cmp_duration)

# ---------------------------------------------------------- gather evidence --
overall, otab, prov, note = None, None, None, ""
if ss.cmp_basis == BATCH:
    br = be.batch_results()
    cc_, orob = br.get("config_comparison"), br.get("outage_robustness")
    if cc_ is None or orob is None:
        ui.not_evaluated("batch comparison", "results/config_comparison.csv or outage_robustness.csv is missing.",
                         "run scripts/run_experiments.py in the backend repo")
        st.stop()
    real_over = cc_[cc_.role == "held-out"]
    real_out = orob[orob.role == "held-out"]
    rows = []
    for cid in be.CONFIGS:
        d = real_over[real_over.config == cid]
        ok = d[d.rmse_m.notna()]
        rows.append({"config": cid, "label": be.CONFIG_LABEL[cid],
                     "sensors_used": ok.sensors_used.iloc[0] if len(ok) else "+".join(be.CONFIGS[cid]),
                     "status": (d.status.iloc[0] if len(d) else "no rows"),
                     "rmse_m": ok.rmse_m.mean() if len(ok) else np.nan, "max_m": ok.max_m.mean() if len(ok) else np.nan,
                     "pct_time_within_target": ok.pct_time_within_target.mean() if len(ok) else np.nan,
                     "n_seq": len(ok)})
    overall = pd.DataFrame(rows)
    otab = real_out
    prov = "batch"
    gps_is_proxy = set(real_over.gps_source.dropna().unique()) == {"proxy_from_reference"}
    note = (f"Mean over {real_over.sequence.nunique()} held-out real sequences · GPS "
            f"<code>{', '.join(sorted(real_over.gps_source.dropna().unique()))}</code> · generated "
            f"{be.results_generated() or 'unknown'}")
else:
    n_starts = 2
    if not be.live_comparison_cached(key, n_starts):
        ui.notice(f"Live evaluation runs the team's EKF {4 * 4 * n_starts + 4} times on <b>{info.label}</b> "
                  f"(≈ 20–60 s, then cached on disk in this app's <code>.cache/</code>).", "")
        if not st.button("Run C1–C4 evaluation now", type="primary"):
            st.stop()
        with st.spinner("Running C1–C4 under identical outages (team backend: compare_configs)…"):
            be.live_config_comparison(key, n_starts)
        st.rerun()
    overall, otab = be.live_config_comparison(key, n_starts)
    overall = overall.copy()
    for col in ("rmse_m", "max_m", "pct_time_within_target"):
        if col not in overall:
            overall[col] = np.nan
    prov = info.provenance
    gps_src = be.experiment(key, ("odom", "gps")).gps_source
    gps_is_proxy = gps_src == "proxy_from_reference"
    note = f"Live on {info.label} · GPS <code>{gps_src}</code> · {n_starts} outage start times per length"

osum = be.summarize_outages(otab) if otab is not None and len(otab) else pd.DataFrame()


def outage_val(cid: str, col: str):
    if osum.empty:
        return np.nan, 0
    d = osum[(osum.config == cid) & (osum.duration_s == D)]
    return (float(d[col].iloc[0]), int(d.n.iloc[0])) if len(d) else (np.nan, 0)


# recommendation (teammate rule: cheapest config whose mean peak error in D-s outages ≤ target)
rec_text = recommend(overall, osum, 0.3, float(D)) if not osum.empty else "No outage results available."
rec_cfg = None
if not osum.empty:
    o = osum[osum.duration_s == D].copy()
    o["rank"] = o.config.map(lambda c: be.COST_TIERS[c]["rank"])
    ok = o[o.max_err_m <= 0.3].sort_values("rank")
    rec_cfg = ok.config.iloc[0] if len(ok) else None
not_eval = [cid for cid in ("C1", "C2", "C3") if np.isnan(outage_val(cid, "max_err_m")[0])]

kind = "batch" if prov == "batch" else ("syn" if prov == "synthetic" else "real")
extra = ""
ref_rank = be.COST_TIERS[rec_cfg]["rank"] if rec_cfg else 99
cheaper_missing = [c for c in not_eval if be.COST_TIERS[c]["rank"] < ref_rank]
if cheaper_missing:
    extra = (f" <span class='small'>({', '.join(cheaper_missing)} not evaluated on this evidence, so a cheaper "
             f"qualifying setup may exist.)</span>")
elif not_eval:
    extra = f" <span class='small'>({', '.join(not_eval)} not evaluated on this evidence.)</span>"
ui.notice(f"<b>Recommendation ({D} s outages):</b> {ui.esc(rec_text)}{extra}", kind)

# ------------------------------------------------------------------- cards --
cols = st.columns(4, gap="small")
SENS = [("odom", "Odometry"), ("imu", "IMU"), ("lidar", "LiDAR"), ("gps", "GPS-RTK")]
for col, cid in zip(cols, be.CONFIGS):
    tier = be.COST_TIERS[cid]
    r = overall[overall.config == cid].iloc[0] if (overall.config == cid).any() else None
    status = str(r.status) if r is not None else "no result"
    used = set(str(r.sensors_used).split("+")) if r is not None and isinstance(r.sensors_used, str) else set(be.CONFIGS[cid])
    planned = set(be.CONFIGS[cid])
    chips = "".join(
        f"<span class='chip{'' if (s in planned and (s in used or status.startswith('NOT RUN'))) else ' off'}'"
        f" title='{'planned but not available in this run' if s in planned and s not in used else ''}'>{n}"
        f"{' · n/a' if s in planned and s not in used and not status.startswith('NOT RUN') else ''}</span>"
        for s, n in SENS)
    if cid != "C4":
        val, n = outage_val(cid, "max_err_m")
        pct, _ = outage_val(cid, "pct_within_target")
        head_lbl = f"peak drift · {D} s GPS outage"
        sub = (f"within 0.3 m for {pct:.0f} % of the outage · mean of {n} runs" if be.finite(val) else "")
    else:
        val = r.rmse_m if r is not None else np.nan
        head_lbl = "error with RTK GPS available (RMSE)"
        sub = (f"max {r.max_m:.2f} m · during an outage it falls back to C3" if r is not None and be.finite(val)
               else "")
        if r is not None and "without" in status:
            sub += f" · <b>{ui.esc(status)}</b>"
    if be.finite(val):
        meets = val <= 0.3
        verdict = (f"<span style='color:{GOOD}'>✓ meets 0.3 m</span>" if meets else
                   "<span style='color:#ec835a'>✕ exceeds 0.3 m</span>")
        metric = (f"<div class='c-metric'>{val:.2f} m</div><div class='c-mlabel'>{head_lbl}</div>"
                  f"<div class='small' style='margin-top:4px'>{verdict}</div><div class='small'>{sub}</div>")
        pb = {"batch": ui.badge("■ BATCH · REAL", "batch"), "real": ui.badge("● LIVE · REAL", "real"),
              "synthetic": ui.badge("▲ SYNTHETIC", "synthetic")}[prov]
        if cid == "C4" and gps_is_proxy and prov != "synthetic":
            pb = ui.badge("▲ GPS PROXY (SIMULATED)", "synthetic")
            metric += ("<div class='small' style='margin-top:4px'>Error against a GPS proxy built from the reference "
                       "itself (+3 cm noise) — not a real-world accuracy.</div>")
    else:
        why = status if status.startswith("NOT RUN") else "no result for this outage length"
        metric = (f"<div class='c-metric' style='color:#8a948e'>Not evaluated</div>"
                  f"<div class='c-mlabel'>{head_lbl}</div><div class='small' style='margin-top:4px'>"
                  f"{ui.esc(why)}</div>")
        pb = ui.badge("NOT EVALUATED", "ne")
    best = " best" if cid == rec_cfg else ""
    col.markdown(
        f"<div class='cfg{best}' style='border-top:4px solid {CONFIG_COLOR[cid]}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'><span class='c-id'>{cid}</span>{pb}</div>"
        f"<div class='c-name'>{ui.esc(be.CONFIG_LABEL[cid])}</div>"
        f"<div class='c-chips'>{chips}</div>"
        f"<div class='c-cost' style='color:{CONFIG_COLOR[cid]}'>{ui.cost_dots(tier['rank'])}</div>"
        f"<div class='c-tier'>relative cost · {ui.esc(tier['tier'])}</div>"
        f"<div class='small' style='margin:2px 0 10px 0'>{ui.esc(tier['complexity'])}</div>"
        f"{metric}{'<div class=small style=margin-top:8px;color:#0ca30c>★ cheapest setup meeting target</div>' if best else ''}"
        f"</div>", unsafe_allow_html=True)

st.markdown(f"<div class='small' style='margin-top:8px'>{note}. C1–C3: real wheel odometry / IMU during "
            f"<b>simulated</b> outage windows; GPS outside the outage is "
            f"{'a simulated proxy' if gps_is_proxy else 'recorded'}."
            + (f" ⚠ {ui.esc(be.reference_caveat(key, gps_src))}" if ss.cmp_basis != BATCH and not gps_is_proxy
               and be.reference_caveat(key, gps_src) else "") + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ charts --
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
cc = st.columns([1, 1.25], gap="medium")
with cc[0]:
    ui.section("Cost tier vs drift", f"C1 → C3 ordered by relative cost · peak drift in a {D} s outage")
    bars = pd.DataFrame([{"config": cid, "peak": outage_val(cid, "max_err_m")[0]} for cid in ("C1", "C2", "C3")])
    st.plotly_chart(ch.config_bars(bars, "peak", "peak drift (m, log)", height=330), theme=None, key="cmp_bars")
with cc[1]:
    ui.section("Drift grows with outage length", "Mean peak error per configuration (+ GPS outside the outage)")
    if osum.empty:
        ui.not_evaluated("outage robustness", "no outage results for this evidence.")
    else:
        rs = (otab.groupby(["config", "duration_s"]).max_err_m
              .agg(median="median", min="min", max="max", n="size").reset_index())
        st.plotly_chart(ch.robustness_chart(rs, height=330), theme=None, key="cmp_robust")

if eng:
    st.divider()
    ui.section("Engineer detail", "Raw tables from the backend's sensor_comparison module")
    t = st.tabs(["Outage summary", "Whole-run comparison", "Method notes"])
    with t[0]:
        if osum.empty:
            ui.not_evaluated("outage summary", "no rows.")
        else:
            ui.df_table(osum.round(3))
    with t[1]:
        show = overall.drop(columns=[c for c in ("generated",) if c in overall]).copy()
        ui.df_table(show.round(3))
        st.caption("Whole-run rows: C1–C3 never see GPS (pure dead reckoning for the entire run), so their error is "
                   "large by design; C4 uses GPS whenever available.")
    with t[2]:
        st.markdown(
            "- **Identical outages**: every configuration is tested on the same outage windows "
            "(`standard_outage_suite`).\n"
            "- **C4 during an outage** has only its dead-reckoning part left (= C3), so it is not re-run.\n"
            "- **Cost tiers** are the plan's qualitative tiers (`COST_TIERS`); no prices are implied.\n"
            "- **Recommendation** uses the backend's `recommend()` rule: cheapest config whose *mean* peak error "
            "meets the target for the chosen outage length.\n"
            "- **C3** needs LiDAR scan-matching odometry (`lidar_odom.csv`), which no real sequence has yet.")
