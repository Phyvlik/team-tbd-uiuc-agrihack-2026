import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.gps_risk import canopy_quality_relationship, quality_variation_check, risk_map
from src.ui import C_REF, data_selector, page_setup

page_setup("Orchard Risk Map")
streams = data_selector()
av = streams.available()

st.markdown("**Question (SP6):** does canopy density predict where GPS quality drops? Following the team plan, we "
            "first check whether GPS quality varies along the route at all, and only then test a canopy → GPS "
            "relationship on a held-out, contiguous part of the route (no random frame splits).")

if not av.get("gps_quality"):
    st.warning(
        "**Not yet available for this sequence — nothing is simulated in its place.**\n\n"
        "This page needs, from the official CitrusFarm files:\n"
        "- GPS receiver quality along the route: `base_*.bag` → `/piksi/debug/receiver_state` "
        "(satellite count, fix mode) and `/piksi/navsatfix_best_fix` (status, covariance)\n"
        "- Canopy index: `mapir_*.bag` → `/mapir_cam/image_raw` (R-G-NIR frames → NDVI-like index)\n\n"
        "Run `python scripts/extract_base_bag.py <SEQ>` once a `base_*.bag` is downloaded. "
        "Note: the dataset authors state the RTK receiver stayed in fixed mode throughout, so quality may vary "
        "very little — in which case this page will report that result rather than draw a risk map.")
    fig = go.Figure(go.Scattergl(x=streams.reference.x, y=streams.reference.y, mode="lines",
                                 line=dict(color=C_REF, width=2), name="Route (reference)"))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (m)", yaxis_title="y (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

gq = streams.gps_quality
metrics = [c for c in gq.columns if c not in ("t", "x", "y")]
c = st.columns(3)
metric = c[0].selectbox("GPS quality metric", metrics)
worse_high = c[1].radio("Worse when", ["higher", "lower"], horizontal=True,
                        index=0 if ("sigma" in metric or "dop" in metric) else 1) == "higher"
cell = c[2].slider("Map cell size (m)", 2.0, 15.0, 5.0, 1.0)

v = gq[metric]
thr = float(np.percentile(v, 80 if worse_high else 20))
check = quality_variation_check(gq, metric, degraded_if_above=thr if worse_high else None,
                                degraded_if_below=None if worse_high else thr)
st.subheader("Step 1 — Does GPS quality vary? (go / no-go)")
(st.success if check["go"] else st.error)(
    f"{'GO' if check['go'] else 'NO-GO'}: {check['reason']}. {metric}: mean {check['mean']:.3f}, "
    f"range {check['min']:.3f}–{check['max']:.3f}, coefficient of variation {check['cv']:.2f}.")

colA, colB = st.columns(2)
with colA:
    st.markdown("**OBSERVED** GPS quality along the route (measured)")
    f = go.Figure(go.Scattergl(x=gq.x, y=gq.y, mode="markers",
                               marker=dict(size=5, color=v, colorscale="Blues" if not worse_high else "Oranges",
                                           reversescale=not worse_high, colorbar=dict(title=metric)),
                               hovertemplate=f"{metric} %{{marker.color:.3f}}<extra></extra>"))
    f.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (m)", yaxis_title="y (m)")
    f.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(f, use_container_width=True)

if not av.get("canopy"):
    with colB:
        st.warning("Canopy index not extracted (mapir_*.bag); relationship not tested.")
    st.stop()

model = canopy_quality_relationship(streams.canopy, gq, metric) if check["go"] else None
grid = risk_map(streams.canopy, gq, metric, model, worse_high, cell)
with colB:
    st.markdown("**MODELLED** GPS quality from canopy index (prediction)" if model else "**Canopy index** (observed)")
    col = grid["modelled"] if model else grid["ndvi"]
    f = go.Figure(go.Scatter(x=grid.cx, y=grid.cy, mode="markers",
                             marker=dict(symbol="square", size=11, color=col,
                                         colorscale="Oranges" if model else "Greens",
                                         colorbar=dict(title="modelled" if model else "NDVI")),
                             hovertemplate="cell %{x:.0f}, %{y:.0f}<br>value %{marker.color:.3f}<extra></extra>"))
    f.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (m)", yaxis_title="y (m)")
    f.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(f, use_container_width=True)

st.subheader("Step 2 — Canopy → GPS-quality relationship (held-out evaluation)")
if model is None:
    st.info("Skipped: quality does not vary enough (no-go). Reported as a finding, not modelled.")
elif not model.get("ok"):
    st.warning(model["reason"])
else:
    k = st.columns(4)
    k[0].metric("Held-out R²", f"{model['r2_test']:.2f}")
    k[1].metric("Held-out MAE", f"{model['mae_test']:.3f}", delta=f"{model['mae_test'] - model['mae_baseline_mean']:+.3f} vs mean baseline",
                delta_color="inverse")
    k[2].metric("Spearman ρ (held-out)", f"{model['spearman_rho_test']:.2f}")
    k[3].metric("Train / test samples", f"{model['n_train']} / {model['n_test']}")
    st.caption(model["split"] + ". Adjacent samples are strongly autocorrelated, so p-values overstate certainty.")
if streams.mode == "synthetic":
    st.warning("Synthetic mode: the canopy → GPS relationship was **built into the generator**, so finding it here "
               "only shows the analysis pipeline works. It is not evidence about real orchards.")
