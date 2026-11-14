import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.teammate_import import ALIASES, LENGTH_FACTORS, normalize, propose_mapping, verify
from src.ui import C_REF, C_SERIES, page_setup

page_setup("Data Import & Baseline Check")
st.markdown("Upload a processed trajectory CSV (e.g. a teammate's wheel-odometry baseline). The importer **inspects "
            "first**: it proposes a column mapping and units, which you can override, then recomputes the error and "
            "runs frame diagnostics to tell genuine drift from alignment/units problems. "
            "Schema: see `docs/DATA_INTERFACE.md`.")

up = st.file_uploader("Trajectory CSV", type=["csv"])
if up is None:
    st.info("Expected columns (any of the listed aliases work): " +
            "; ".join(f"**{k}** ← {', '.join(v[:4])}…" for k, v in ALIASES.items()))
    st.stop()

raw = pd.read_csv(up, comment="#")
st.subheader("1 · Raw file")
st.write(f"{len(raw)} rows · columns: {list(raw.columns)}")
st.dataframe(raw.head(8), use_container_width=True)

st.subheader("2 · Column mapping and units")
auto = propose_mapping(raw.columns)
cols = ["—"] + list(raw.columns)
mapping = {}
c = st.columns(3)
for i, target in enumerate(ALIASES):
    default = auto.get(target, "—")
    sel = c[i % 3].selectbox(target + (" (optional)" if target == "error_m" else ""), cols,
                             index=cols.index(default), key=f"map_{target}")
    if sel != "—":
        mapping[target] = sel
c2 = st.columns(2)
tu = c2[0].selectbox("Time unit", ["auto", "s", "ms", "us", "ns"])
lu = c2[1].selectbox("Length unit", list(LENGTH_FACTORS), index=0)
try:
    norm, notes = normalize(raw, mapping, None if tu == "auto" else tu, lu)
except ValueError as e:
    st.error(str(e))
    st.stop()
st.caption(" · ".join(notes))

st.subheader("3 · Verification")
rep = verify(norm)
k = st.columns(5)
k[0].metric("Mean error", f"{rep['mean_err_m']:.2f} m")
k[1].metric("Max error", f"{rep['max_err_m']:.2f} m")
k[2].metric("Start offset", f"{rep['start_offset_m']:.2f} m")
k[3].metric("Rigid-fit rotation", f"{rep['rigid_fit_rotation_deg']:.1f}°")
k[4].metric("Path length ratio (est/ref)", f"{rep['path_length_ratio_est_over_ref']:.3f}")
for f in rep["findings"]:
    st.warning(f)
with st.expander("Full report"):
    st.json({k: v for k, v in rep.items() if k != "findings"})

fig = go.Figure()
fig.add_trace(go.Scattergl(x=norm.reference_x, y=norm.reference_y, mode="lines", name="reference",
                           line=dict(color=C_REF, dash="dot", width=2)))
fig.add_trace(go.Scattergl(x=norm.estimated_x, y=norm.estimated_y, mode="lines", name="estimate",
                           line=dict(color=C_SERIES[0], width=2)))
fig.update_layout(height=520, xaxis_title="x (m)", yaxis_title="y (m)", legend=dict(orientation="h", y=1.08),
                  margin=dict(l=10, r=10, t=30, b=10))
fig.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(fig, use_container_width=True)
st.download_button("Download normalised CSV", norm.to_csv(index=False), "normalised_trajectory.csv", "text/csv")
