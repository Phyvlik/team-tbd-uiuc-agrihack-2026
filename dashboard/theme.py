"""Design tokens, global CSS and the Plotly template.

Palette follows the dataviz reference palette (dark steps), validated against
the chart surface #121a23: categorical slots 1–4 pass every adjacent check.
Status colours are reserved for navigation state and always ship with an icon
and a text label, never colour alone.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# surfaces & ink: charcoal navy with muted orchard-green accents
PAGE = "#0b1118"
SURFACE = "#121a23"
SURFACE_2 = "#18222d"
BORDER = "rgba(255,255,255,0.08)"
INK = "#eef2f5"
INK_2 = "#c2cad3"
MUTED = "#8b96a3"
GRID = "#1f2a35"
AXIS = "#334252"
ACCENT = "#6fa77a"        # muted orchard green (brand accent, never a status colour)

# reserved status palette (never used for a data series)
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
NEUTRAL = "#6b746f"

# categorical, fixed order (dark steps): blue, orange, aqua, yellow
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
REF = "#c3c2b7"           # reference trajectory: neutral ink, not a series hue
GPS_FIX = SERIES[0]
CONFIG_COLOR = {"C1": SERIES[0], "C2": SERIES[1], "C3": SERIES[2], "C4": SERIES[3]}

STATE = {
    "TRUSTED": {"color": GOOD, "icon": "●", "emoji": "🟢", "label": "Trusted", "short": "TRUSTED",
                "meaning": "Position is reliable"},
    "DEGRADED": {"color": WARN, "icon": "▲", "emoji": "🟡", "label": "Degraded", "short": "DEGRADED",
                 "meaning": "Position less certain — caution"},
    "STOP": {"color": CRIT, "icon": "■", "emoji": "🔴", "label": "Stop recommended", "short": "STOP",
             "meaning": "Position cannot be trusted"},
}
DECISION_STATE = {"CONTINUE": "TRUSTED", "SLOW": "DEGRADED", "STOP": "STOP"}

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=INK_2, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickcolor=AXIS, title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickcolor=AXIS, title_font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=AXIS, font=dict(color=INK, family=FONT, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    t.layout.xaxis.automargin = True
    t.layout.yaxis.automargin = True
    return t


pio.templates["agri_dark"] = _template()
pio.templates.default = "agri_dark"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
  --page:{PAGE}; --surface:{SURFACE}; --surface-2:{SURFACE_2}; --border:{BORDER};
  --ink:{INK}; --ink-2:{INK_2}; --muted:{MUTED};
  --good:{GOOD}; --warn:{WARN}; --crit:{CRIT};
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family:{FONT}; }}
.stApp {{ background: radial-gradient(1100px 420px at 12% -12%, rgba(111,167,122,0.07), transparent 60%), var(--page); }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 2rem; padding-left: 1.6rem; padding-right: 1.6rem; max-width: 1760px; }}
div[data-testid='stVerticalBlock'] {{ gap: 0.6rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: #0d141c; border-right: 1px solid var(--border); }}
div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}

/* page header */
.pg-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
           margin: 0 0 4px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
.pg-eyebrow {{ color: var(--muted); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight:600; }}
.pg-title {{ color: var(--ink); font-size: 1.45rem; font-weight: 750; line-height:1.15; margin: 0; }}
.pg-sub {{ color: var(--ink-2); font-size: .84rem; max-width: 900px; margin-top:2px; }}
.badges {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}

/* badges / pills */
.badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.74rem;
         font-weight:600; letter-spacing:.03em; border:1px solid var(--border); color: var(--ink-2); background: var(--surface); white-space:nowrap; }}
.badge.real {{ border-color: rgba(57,135,229,.55); color:#a9c9f2; background: rgba(57,135,229,.10); }}
.badge.synthetic {{ border-color: rgba(250,178,25,.65); color:#ffd27a; background: rgba(250,178,25,.10); }}
.badge.batch {{ border-color: rgba(25,158,112,.55); color:#8fdcbf; background: rgba(25,158,112,.10); }}
.badge.ne {{ border-style: dashed; color: var(--muted); }}

.state-pill {{ display:inline-flex; align-items:center; gap:7px; padding:4px 12px 4px 10px; border-radius:999px;
              font-weight:700; font-size:.8rem; letter-spacing:.04em; color: var(--ink); }}

/* cards */
.card {{ background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px 16px; height:100%; }}
.card-label {{ color: var(--muted); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; font-weight:600; margin-bottom:6px; }}
.card-value {{ color: var(--ink); font-size:1.55rem; font-weight:700; line-height:1.15; }}
.card-sub {{ color: var(--ink-2); font-size:.82rem; margin-top:4px; }}
.kpi-row {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin-bottom: 12px; }}

/* hero status */
.hero {{ border-radius:14px; padding:12px 14px; border:1px solid var(--border); position:relative; overflow:hidden; }}
.hero .h-top {{ display:flex; align-items:center; gap:12px; }}
.hero .h-icon {{ width:40px; height:40px; flex-shrink:0; border-radius:50%; display:flex; align-items:center; justify-content:center;
                font-size:1.25rem; font-weight:800; color:#0b0b0b; flex: 0 0 auto; }}
.hero .h-label {{ color: var(--muted); font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }}
.hero .h-state {{ color: var(--ink); font-size:clamp(1.0rem, 1.45vw, 1.5rem); font-weight:800; line-height:1.12; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hero .h-reason {{ color: var(--ink-2); font-size:.82rem; margin-top:6px; }}
.hero .h-top > div:last-child {{ min-width:0; }}
.pulse {{ animation: pulse 1.4s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(208,59,59,.55); }} 50% {{ box-shadow: 0 0 0 10px rgba(208,59,59,0); }} }}

/* recommendation */
.rec {{ border-radius:14px; padding:14px 16px; background: var(--surface); border:1px solid var(--border); }}
.rec .r-title {{ color: var(--ink); font-weight:750; font-size:1.05rem; margin: 2px 0 8px 0; }}
.rec ul {{ margin: 0; padding-left: 1.1rem; color: var(--ink-2); font-size:.88rem; }}
.rec li {{ margin: 3px 0; }}

/* sensors */
.sensors {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; }}
.sensor {{ display:flex; gap:10px; align-items:center; padding:9px 10px; border-radius:10px; background: var(--surface-2);
          border:1px solid var(--border); }}
.sensor .s-dot {{ width:24px; height:24px; border-radius:7px; display:flex; align-items:center; justify-content:center;
                 font-size:.78rem; font-weight:800; flex:0 0 auto; }}
.sensor .s-name {{ color: var(--ink); font-size:.84rem; font-weight:600; line-height:1.15; }}
.sensor .s-state {{ color: var(--muted); font-size:.74rem; }}

/* gauge */
.gauge {{ margin-top:8px; }}
.gauge .g-track {{ position:relative; height:10px; border-radius:6px; overflow:hidden; display:flex; }}
.gauge .g-seg {{ height:100%; }}
.gauge .g-marker {{ position:absolute; top:-5px; width:3px; height:20px; background: var(--ink); border-radius:2px;
                   box-shadow: 0 0 0 2px var(--surface); }}
.gauge .g-ticks {{ position:relative; height:16px; color: var(--muted); font-size:.68rem; margin-top:4px; }}
.gauge .g-ticks span {{ position:absolute; transform: translateX(-50%); white-space:nowrap; }}

/* alert log */
.alog {{ display:flex; flex-direction:column; gap:6px; max-height: 280px; overflow-y:auto; padding-right:4px; }}
.alog .a {{ display:grid; grid-template-columns: 58px 1fr; gap:10px; align-items:start; padding:8px 10px; border-radius:10px;
           background: var(--surface-2); border:1px solid var(--border); }}
.alog .a-t {{ color: var(--muted); font-size:.78rem; font-variant-numeric: tabular-nums; padding-top:2px; }}
.alog .a-m {{ color: var(--ink); font-size:.84rem; font-weight:600; }}
.alog .a-r {{ color: var(--ink-2); font-size:.78rem; font-weight:400; }}

/* legend */
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; color: var(--ink-2); font-size:.78rem; }}
.legend .l {{ display:inline-flex; gap:6px; align-items:center; white-space:nowrap; }}
.legend .sw {{ width:12px; height:12px; border-radius:3px; display:inline-flex; align-items:center; justify-content:center;
              font-size:.55rem; color:#0b0b0b; font-weight:900; }}
.legend .ln {{ width:18px; height:0; border-top:2px solid; display:inline-block; }}

/* not evaluated / notices */
.ne-box {{ border:1px dashed rgba(255,255,255,.22); border-radius:12px; padding:12px 14px; color: var(--ink-2);
          background: repeating-linear-gradient(135deg, rgba(255,255,255,.025) 0 8px, transparent 8px 16px); font-size:.86rem; }}
.ne-box b {{ color: var(--ink); }}
.notice {{ border-radius:10px; padding:9px 13px; font-size:.84rem; color: var(--ink-2); border:1px solid var(--border);
          background: var(--surface); margin-bottom: 10px; }}
.notice.syn {{ border-left: 4px solid var(--warn); background: rgba(250,178,25,.06); }}
.notice.real {{ border-left: 4px solid #3987e5; background: rgba(57,135,229,.06); }}
.notice.batch {{ border-left: 4px solid #199e70; background: rgba(25,158,112,.06); }}

/* config cards */
.cfg {{ background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px; height:100%; }}
.cfg .c-id {{ font-weight:800; font-size:1.25rem; color: var(--ink); }}
.cfg .c-name {{ color: var(--ink-2); font-size:.84rem; min-height: 2.3em; }}
.cfg .c-cost {{ font-size:1.05rem; letter-spacing:.1em; font-weight:800; margin-top:8px; }}
.cfg .c-cost .off {{ color: #2d3833; }}
.cfg .c-tier {{ color: var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }}
.cfg .c-chips {{ display:flex; gap:4px; flex-wrap:wrap; margin: 8px 0; }}
.cfg .chip {{ font-size:.7rem; padding:2px 7px; border-radius:6px; background: var(--surface-2); border:1px solid var(--border); color: var(--ink-2); }}
.cfg .chip.off {{ opacity:.35; text-decoration: line-through; }}
.cfg .c-metric {{ color: var(--ink); font-size:1.3rem; font-weight:750; }}
.cfg .c-mlabel {{ color: var(--muted); font-size:.72rem; }}
.cfg.best {{ border-color: rgba(12,163,12,.6); box-shadow: 0 0 0 1px rgba(12,163,12,.35) inset; }}

/* worker light */
.light {{ display:flex; flex-direction:column; gap:10px; align-items:center; padding:16px 10px; background:#0c110f;
         border-radius:18px; border:1px solid var(--border); }}
.light .lamp {{ width:62px; height:62px; border-radius:50%; display:flex; align-items:center; justify-content:center;
               color:#0b0b0b; font-weight:900; font-size:.72rem; letter-spacing:.06em; opacity:.16; }}
.light .lamp.on {{ opacity:1; }}
.msg {{ border-radius:12px; padding:12px 14px; border:1px solid var(--border); background: var(--surface); margin-bottom:8px; }}
.msg .m-lang {{ color: var(--muted); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }}
.msg .m-text {{ color: var(--ink); font-size:.98rem; font-weight:600; margin-top:3px; }}

.small {{ color: var(--muted); font-size:.78rem; }}
.sec {{ color: var(--ink); font-weight:700; font-size:1.02rem; margin: 6px 0 6px 0; }}
.sec-sub {{ color: var(--muted); font-size:.8rem; margin: -4px 0 8px 0; }}
div[data-testid="stPlotlyChart"] {{ border-radius: 14px; }}
/* data-kind tags */
.kind {{ display:inline-flex; align-items:center; gap:4px; padding:1px 7px; border-radius:6px; font-size:.66rem;
        font-weight:700; letter-spacing:.06em; text-transform:uppercase; border:1px solid; white-space:nowrap; }}
.kind.recorded {{ color:#a9c9f2; border-color:rgba(57,135,229,.55); background:rgba(57,135,229,.10); }}
.kind.computed {{ color:#9fd8c0; border-color:rgba(25,158,112,.55); background:rgba(25,158,112,.10); }}
.kind.predicted {{ color:#cdbcf5; border-color:rgba(144,133,233,.55); background:rgba(144,133,233,.10); }}
.kind.simulated {{ color:#ffd27a; border-color:rgba(250,178,25,.6); background:rgba(250,178,25,.08); border-style:dashed; }}
.kind.illustrative {{ color:#c2cad3; border-color:rgba(194,202,211,.4); border-style:dotted; }}
.kind.uncalibrated {{ color:#ffd27a; border-color:rgba(250,178,25,.75); background:rgba(250,178,25,.12); }}

/* operator metric tiles */
.mgrid {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; }}
.mtile {{ background: var(--surface); border:1px solid var(--border); border-radius:12px; padding:9px 12px; min-width:0; }}
.mtile .m-l {{ color: var(--muted); font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mtile .m-v {{ color: var(--ink); font-size:1.35rem; font-weight:800; line-height:1.2; margin-top:2px; white-space:nowrap; }}
.mtile .m-s {{ color: var(--ink-2); font-size:.74rem; margin-top:2px; }}
.mtile.wide {{ grid-column: span 2; }}

/* checklist */
.chk {{ display:flex; flex-direction:column; gap:5px; }}
.chk .c {{ display:grid; grid-template-columns: 26px 1fr; gap:8px; align-items:start; padding:7px 10px; border-radius:10px;
          background: var(--surface); border:1px solid var(--border); }}
.chk .c .ic {{ width:22px; height:22px; border-radius:6px; display:flex; align-items:center; justify-content:center;
              font-weight:900; font-size:.78rem; }}
.chk .c .t {{ color: var(--ink); font-size:.86rem; font-weight:650; }}
.chk .c .w {{ color: var(--muted); font-size:.74rem; }}
.chk .c.dim .t {{ color: var(--muted); font-weight:500; }}

/* bilingual alerts */
.alrt {{ border-radius:12px; padding:9px 12px; border:1px solid var(--border); background: var(--surface); margin-bottom:6px; }}
.alrt .a-h {{ display:flex; gap:8px; align-items:center; }}
.alrt .a-i {{ width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-size:.7rem; font-weight:900; color:#0b0b0b; flex-shrink:0; }}
.alrt .a-en {{ color: var(--ink); font-weight:700; font-size:.9rem; }}
.alrt .a-es {{ color: var(--ink-2); font-weight:600; font-size:.86rem; margin-top:4px; }}
.alrt .a-do {{ color: var(--ink-2); font-size:.78rem; margin-top:2px; }}
.alrt .lang {{ color: var(--muted); font-size:.62rem; letter-spacing:.12em; font-weight:800; margin-right:6px; }}

.tb-label {{ color: var(--muted); font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }}
@media (max-width: 1400px) {{
  .mtile .m-v {{ font-size:1.15rem; }}
  .pg-sub {{ display:none; }}
}}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
