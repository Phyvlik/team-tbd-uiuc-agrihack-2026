"""Orchard Risk Map — where is GPS likely to fail? (SP6, stretch goal)

Follows the team plan's go/no-go rule via the backend's gps_risk module:
first check whether measured GPS quality varies along the route; only then
model canopy → GPS quality on a held-out contiguous split. Observed and
modelled layers are always labelled as such.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import backend as be
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import state as S
from dashboard.theme import CRIT, GOOD, INK, WARN

ss = st.session_state
key = ss.source
streams = be.load_streams(key)
info = be.source_info(key)
eng = S.is_engineer()
av = streams.available()
t0, _ = streams.time_span()

ui.page_header(
    "Planning · predictive", "Orchard Risk Map",
    "Where is GPS likely to degrade under the canopy? Observed receiver quality along the route, and — where "
    "canopy imagery exists — a held-out model that predicts GPS dead zones before the robot gets there.",
    [ui.provenance_badge(info.provenance)] + ([] if av.get("canopy") else [ui.provenance_badge("ne")]))

LEVEL_NAMES = {0: "Low risk", 1: "Elevated risk", 2: "High risk"}

# Risk rules per metric: (label, direction, elevated threshold, high threshold, unit)
RULES = {
    "fix_mode": ("RTK fix mode", "categorical", None, None, ""),
    "num_sat": ("Satellites tracked", "lower", 20, 12, ""),
    "num_sats": ("Satellites tracked", "lower", 18, 12, ""),
    "num_gps_sat": ("GPS satellites tracked", "lower", 10, 6, ""),
    # high once 3σ reaches the 0.3 m target; elevated from 0.05 m
    "h_sigma_m": ("Receiver horizontal σ", "higher", 0.05, 0.10, " m"),
    "cn0_mean": ("Mean signal strength (C/N0)", "lower", None, None, ""),
}
FIX_LEVEL = {"FIXED_RTK": 0, "FLOAT_RTK": 1, "FLOAT": 1}


def levels_for(values: pd.Series, metric: str) -> np.ndarray:
    rule = RULES.get(metric)
    if metric == "fix_mode":
        return values.map(lambda v: FIX_LEVEL.get(str(v), 2)).to_numpy()
    v = values.to_numpy(float)
    if rule is None or rule[2] is None:  # relative: worst 10 % high, next 20 % elevated
        worse_high = rule is None or rule[1] == "higher"
        q = np.nanpercentile(v, [70, 90] if worse_high else [30, 10])
        return (np.where(v >= q[1], 2, np.where(v >= q[0], 1, 0)) if worse_high
                else np.where(v <= q[1], 2, np.where(v <= q[0], 1, 0)))
    _, d, e, h, _ = rule
    return (np.where(v >= h, 2, np.where(v >= e, 1, 0)) if d == "higher"
            else np.where(v < h, 2, np.where(v < e, 1, 0)))


def rule_text(metric: str) -> str:
    r = RULES.get(metric)
    if metric == "fix_mode":
        return "● FIXED_RTK = low · ▲ FLOAT = elevated · ■ anything worse = high"
    if r is None or r[2] is None:
        return "Relative to this route: worst 10 % = high, next 20 % = elevated (no absolute limit agreed yet)"
    lab, d, e, h, u = r
    return (f"● {lab} {'≤' if d == 'higher' else '≥'} {e}{u} low · ▲ elevated · ■ "
            f"{'≥' if d == 'higher' else '<'} {h}{u} high")


def cell_polys(df: pd.DataFrame, levels: np.ndarray, cell: float, value: str, label: str) -> list:
    traces = []
    cols = {0: GOOD, 1: WARN, 2: CRIT}
    icons = {0: "●", 1: "▲", 2: "■"}
    for lv in (0, 1, 2):
        d = df[levels == lv]
        if not len(d):
            continue
        xs, ys = [], []
        h = cell / 2 * 0.94
        for cx, cy in zip(d.cx, d.cy):
            xs += [cx - h, cx + h, cx + h, cx - h, cx - h, None]
            ys += [cy - h, cy - h, cy + h, cy + h, cy - h, None]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=ch._rgba(cols[lv], 0.55),
                                 line=dict(color=cols[lv], width=0.5), name=f"{icons[lv]} {LEVEL_NAMES[lv]} (modelled)",
                                 hoverinfo="skip"))
        traces.append(go.Scatter(x=d.cx, y=d.cy, mode="markers", marker=dict(size=1, color=cols[lv], opacity=0),
                                 showlegend=False, customdata=d[[value]].to_numpy(),
                                 hovertemplate=f"{LEVEL_NAMES[lv]}<br>{label}: %{{customdata[0]:.3f}}<extra></extra>"))
    return traces


# ------------------------------------------------------------ no quality --
if not av.get("gps_quality"):
    ui.not_evaluated(
        "GPS risk for this sequence",
        "no receiver-quality stream has been extracted, and nothing is simulated in its place.",
        "<code>python scripts/extract_base_bag.py " + ui.esc(key) + "</code> in the backend repo "
        "(streams base_*.bag, keeps <code>gps_quality.csv</code>); for the predictive layer also a canopy index "
        "(<code>canopy.csv</code>) from mapir_*.bag. Real receiver quality exists for 06_14B; the synthetic orchard "
        "shows the full pipeline.")
    fig = go.Figure()
    for tr in ch.orchard_rows_traces(streams):
        fig.add_trace(tr)
    r = streams.reference.iloc[::5]
    fig.add_trace(go.Scatter(x=r.x, y=r.y, mode="lines", line=dict(color="#c3c2b7", width=1.5, dash="dot"),
                               name="Route (RTK reference)"))
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.06, yanchor="top"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, theme=None, key="risk_empty")
    st.stop()

gq = streams.gps_quality.copy()
if "x" not in gq or "y" not in gq:
    ref = streams.reference
    gq["x"] = np.interp(gq.t, ref.t, ref.x)
    gq["y"] = np.interp(gq.t, ref.t, ref.y)
numeric = [c for c in gq.columns if c not in ("t", "x", "y") and pd.api.types.is_numeric_dtype(gq[c])
           and gq[c].std() > 0]
preferred = [m for m in ("fix_mode", "num_sat", "num_sats", "h_sigma_m", "num_gps_sat", "cn0_mean") if m in gq]
options = preferred + ([m for m in numeric if m not in preferred] if eng else [])
default_metric = "h_sigma_m" if "h_sigma_m" in gq else ("num_sat" if "num_sat" in gq else options[0])
if ss.get("risk_metric") not in options:
    ss.risk_metric = default_metric

with st.container(border=True):
    c = st.columns([2.5, 1.6, 4])
    c[0].selectbox("GPS quality measure", options, key="risk_metric",
                   format_func=lambda m: RULES.get(m, (m,))[0] + ("" if m not in RULES else f"  ({m})"))
    cell = c[1].select_slider("Map block size", [3.0, 5.0, 8.0, 12.0], value=5.0, format_func=lambda v: f"{v:.0f} m",
                              key="risk_cell")
    c[2].markdown(f"<div class='small' style='padding-top:1.9rem'>Risk rule (display limits, not yet agreed with the "
                  f"domain lead): {ui.esc(rule_text(ss.risk_metric))}</div>",
                  unsafe_allow_html=True)
metric = ss.risk_metric
mlabel = RULES.get(metric, (metric,))[0]
lv_obs = levels_for(gq[metric], metric)

# ---------------------------------------------------- step 1: go / no-go --
if metric == "fix_mode":
    nonfixed = 100 * float(np.mean(lv_obs > 0))
    check = {"go": nonfixed >= 2.0, "reason": f"{nonfixed:.1f} % of samples were not RTK-fixed"}
else:
    rule = RULES.get(metric)
    worse_high = rule is None or rule[1] == "higher"
    thr = float(np.nanpercentile(gq[metric], 80 if worse_high else 20))
    check = be.quality_variation_check(gq, metric, degraded_if_above=thr if worse_high else None,
                                       degraded_if_below=None if worse_high else thr)
    if rule and rule[2] is not None:  # absolute rule: degraded share from the agreed limits
        check["abs_degraded_pct"] = 100 * float(np.mean(lv_obs > 0))

go_ = check["go"]
if info.provenance == "real" and not go_:
    ui.notice(f"<b>Finding — no-go on real data ({ui.esc(key)}):</b> {ui.esc(check['reason'])}. "
              f"The receiver held a fixed RTK solution along this whole route, so there is no measured GPS "
              f"degradation to predict here. Reported as a result, not hidden. The predictive canopy layer is "
              f"<b>Not evaluated</b> on real data.", "real")
elif info.provenance == "real":
    ui.notice(f"<b>GO:</b> {ui.esc(check['reason'])} — GPS quality varies on this route.", "real")
else:
    ui.source_notice(info)

# ---------------------------------------------------------- maps -----------
canopy_ok = av.get("canopy", False)
model = None
grid = None
if canopy_ok and metric != "fix_mode":
    model = be.canopy_quality_relationship(streams.canopy, gq, metric) if go_ else None
    worse_high = RULES.get(metric, (None, "higher"))[1] == "higher"
    grid = be.risk_map(streams.canopy, gq, metric, model, worse_high, cell)

left, right = st.columns(2, gap="medium")
with left:
    ui.section("Observed GPS quality", f"Measured along the route · {mlabel}")
    obs = gq.iloc[:: max(1, len(gq) // 4000)].reset_index(drop=True)
    lv = levels_for(obs[metric], metric)
    if metric == "fix_mode":
        obs = obs.assign(fix_level=lv)
        fig = ch.quality_map(obs, "fix_level", "fix level (0 = RTK fixed)", lv, height=500, ref=streams.reference,
                             extra=ch.orchard_rows_traces(streams))
    else:
        fig = ch.quality_map(obs, metric, mlabel, lv, height=500, ref=streams.reference,
                             extra=ch.orchard_rows_traces(streams))
    st.plotly_chart(fig, theme=None, key="risk_obs")
with right:
    if grid is not None and model and model.get("ok"):
        ui.section("Predicted GPS risk from canopy", "Modelled from the canopy index · held-out validated below")
        g = grid.dropna(subset=["modelled"]).reset_index(drop=True)
        lvg = levels_for(g["modelled"], metric)
        fig = go.Figure(cell_polys(g, lvg, cell, "modelled", f"modelled {mlabel}"))
        for tr in ch.orchard_rows_traces(streams):
            fig.add_trace(tr)
        r = streams.reference.iloc[::5]
        fig.add_trace(go.Scatter(x=r.x, y=r.y, mode="lines", name="Route", line=dict(color="#c3c2b7", width=1,
                                                                                         dash="dot"), hoverinfo="skip"))
        fig.update_layout(height=500, showlegend=False, margin=dict(l=8, r=8, t=8, b=8),
                          xaxis_title="East–west (m)", yaxis_title="North–south (m)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, theme=None, key="risk_model")
    else:
        ui.section("Predicted GPS risk from canopy", "Requires a canopy index stream and a GO on step 1")
        if not canopy_ok:
            ui.not_evaluated("predictive risk layer",
                             "no canopy index (<code>canopy.csv</code>) exists for this sequence.",
                             "R-G-NIR canopy index per frame from <code>mapir_*.bag</code> (8–51 GB per sequence) — "
                             "team decision pending on whether to download it.")
        elif metric == "fix_mode":
            ui.not_evaluated("predictive layer for fix mode", "fix mode is categorical; pick a numeric quality measure.")
        else:
            ui.not_evaluated("canopy → GPS model", "GPS quality does not vary enough to model (no-go).")

st.markdown(
    f"<div class='legend' style='margin-top:4px'>"
    f"<span class='l'><span class='sw' style='background:{GOOD}'>●</span><b style='color:{INK}'>Low risk</b>&nbsp;GPS expected to hold</span>"
    f"<span class='l'><span class='sw' style='background:{WARN}'>▲</span><b style='color:{INK}'>Elevated risk</b>&nbsp;robot slows, wider buffer</span>"
    f"<span class='l'><span class='sw' style='background:{CRIT}'>■</span><b style='color:{INK}'>High risk</b>&nbsp;plan crews and routes around it</span>"
    f"<span class='l'><span class='ln' style='border-top-color:#c3c2b7;border-top-style:dotted'></span>Route</span>"
    + ("<span class='l'><span class='sw' style='background:rgba(25,158,112,0.35)'></span>Tree rows (synthetic layout)</span>"
       if streams.mode == "synthetic" else "")
    + "<span class='l'>Left: points = measured · Right: blocks = modelled</span></div>", unsafe_allow_html=True)

# --------------------------------------------------------- operator summary --
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
k = st.columns(4)
share = {lvl: 100 * float(np.mean(lv_obs == lvl)) for lvl in (0, 1, 2)}
k[0].markdown(ui.kpi_html("Route with low risk", f"{share[0]:.0f} %", "observed", accent=GOOD, icon="●"),
              unsafe_allow_html=True)
k[1].markdown(ui.kpi_html("Elevated risk", f"{share[1]:.0f} %", "observed", accent=WARN, icon="▲"),
              unsafe_allow_html=True)
k[2].markdown(ui.kpi_html("High risk", f"{share[2]:.0f} %", "observed", accent=CRIT, icon="■"), unsafe_allow_html=True)
if model and model.get("ok"):
    k[3].markdown(ui.kpi_html("Canopy predicts GPS? (held-out)", f"ρ = {model['spearman_rho_test']:.2f}",
                              f"R² {model['r2_test']:.2f} on unseen part of route"), unsafe_allow_html=True)
else:
    k[3].markdown(ui.kpi_html("Canopy predicts GPS?", "Not evaluated",
                              "needs canopy index + measurable GPS variation"), unsafe_allow_html=True)

if not eng:
    with st.expander("How should I use this map?"):
        st.markdown("- **Before a shift:** schedule crews away from ■ high-risk blocks while the robot works there.\n"
                    "- **During a run:** the robot slows and widens its safety buffer in ▲/■ blocks (proactive mode on "
                    "the Worker Safety page).\n- Observed = measured by the receiver. Predicted = modelled from canopy "
                    "imagery and checked on a part of the route the model never saw.")
    st.stop()

# ---------------------------------------------------------- engineer view --
st.divider()
t = st.tabs(["Go / no-go statistics", "Held-out canopy model", "Quality over time"])
with t[0]:
    stats = {k_: v for k_, v in check.items() if k_ not in ("reason",)}
    ui.df_table(pd.DataFrame([stats]).round(3))
    st.caption("Rule (team plan, gps_risk.quality_variation_check): continue only if the coefficient of variation ≥ "
               "0.10 or ≥ 2 % of samples are degraded.")
with t[1]:
    if model and model.get("ok"):
        kk = st.columns(4)
        kk[0].metric("Held-out R²", f"{model['r2_test']:.2f}")
        kk[1].metric("Held-out MAE", f"{model['mae_test']:.3f}",
                     delta=f"{model['mae_test'] - model['mae_baseline_mean']:+.3f} vs mean baseline",
                     delta_color="inverse")
        kk[2].metric("Spearman ρ (held-out)", f"{model['spearman_rho_test']:.2f}")
        kk[3].metric("Train / test samples", f"{model['n_train']} / {model['n_test']}")
        st.caption(model["split"] + ". Adjacent samples are autocorrelated, so p-values overstate certainty.")
        if info.provenance == "synthetic":
            ui.notice("Synthetic: the canopy → GPS relationship was <b>built into the generator</b>, so recovering it "
                      "only shows the pipeline works. It is not evidence about real orchards.", "syn")
    else:
        ui.not_evaluated("held-out canopy model", "needs canopy.csv and a GO on the variation check.")
with t[2]:
    if metric == "fix_mode":
        st.plotly_chart(ch.series_over_time(gq.t - t0, lv_obs, "fix level (0 = RTK fixed)", height=240),
                        theme=None, key="risk_ts")
    else:
        st.plotly_chart(ch.series_over_time(gq.t - t0, gq[metric], mlabel, height=240), theme=None, key="risk_ts")
