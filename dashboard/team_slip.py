"""Teammate slip-learning results: discovery, schema validation and loading.

Source folder: ``team_runs/slip_learning/``. These CSVs are **computed results**
from a teammate's localization analysis of CitrusFarm (wheel odometry + IMU + GPS
with a learned wheel-slip correction), not raw sensor measurements. This module
reads them as the source of truth for the "Team Slip-Learning Results" replay:
it never recomputes the estimates, the uncertainty or the states. It only
validates them and reports what it finds.

Per sequence (``<SEQ>`` = ``01_13B_Jackal`` ... ``07_14B_Jackal``):

* ``<SEQ>_dashboard.csv``  one row per ~0.1 s: t, x_true, y_true, x_est, y_est,
  yaw_est, sigma, gps_out, state, bubble_m, slip_scale, error_m
* ``<SEQ>_trees.csv``      simulated spray outcome per tree: x, y, status, during_outage
* ``<SEQ>_crew.csv``       simulated worker per outage: t, t_outage_start, x, y

Summaries: ``test_results.csv``, ``spray_results.csv``, ``safety_results.csv``.

Verified on the delivered files (2026-11-14): ``t`` is seconds since the first
``gt.csv`` sample and ``x_true, y_true`` equal the CitrusFarm ``gt.csv`` reference
in the same frame (no rotation or time offset), so the replay shares the clock
and frame of the backend EKF views.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
TEAM_DIR = APP_DIR / "team_runs" / "slip_learning"
SOURCE_LABEL = "Team Slip-Learning Results"

KINDS = ("dashboard", "trees", "crew")
SUMMARIES = ("test_results", "spray_results", "safety_results")
DASHBOARD_COLUMNS = ("t", "x_true", "y_true", "x_est", "y_est", "yaw_est", "sigma", "gps_out", "state",
                     "bubble_m", "slip_scale", "error_m")
TREE_COLUMNS = ("x", "y", "status", "during_outage")
CREW_COLUMNS = ("t", "t_outage_start", "x", "y")
SUMMARY_COLUMNS = {
    "test_results": ("seq", "group", "outage_s", "setup", "n", "time_within", "err_end", "worst"),
    "spray_results": ("seq", "outage_s", "setup", "trees", "correct_pct", "wrong_tree_pct", "missed_pct",
                      "double_pct", "wrong_row_pct"),
    "safety_results": ("closest_unsafe_m", "seq", "outage_s", "setup", "encounters", "unsafe",
                       "median_stop_dist_m", "bubble_contains_truth_outage_pct", "time_green_pct",
                       "time_yellow_pct", "time_red_pct"),
}
# summary columns that may legitimately be empty (no unsafe encounter -> no closest distance)
SUMMARY_OPTIONAL = {"safety_results": ("closest_unsafe_m",)}

# Navigation states as defined by the teammate, on the model-estimated 2*sigma
TRUSTED_2SIGMA_M = 0.3
LOST_2SIGMA_M = 1.5
STATE_NAMES = {0: "Trusted", 1: "Degraded", 2: "Lost"}
STATE_THEME = {0: "TRUSTED", 1: "DEGRADED", 2: "STOP"}      # colour keys in dashboard/theme.py
SPRAY_STATUSES = ("correct", "wrong_tree", "missed", "double")
WORKER_RADIUS_M = 2.0
WORKER_VISIBLE_AFTER_S = 15.0           # a worker stays visible until crew.t + 15 s
REVEAL_MIN_AGREEMENT = 0.90             # see TeamRun.tree_reveal

PRIMARY_SETUPS = ("Odometry + GPS", "Odometry + IMU + GPS", "Odometry + IMU + GPS + slip learning")
EXCLUDED_SETUPS = ("Fixed slip from 02 (no learning)",)

_SEQ_FILE = re.compile(r"^(?P<seq>\d{2}_\d{2}[A-Z]_Jackal)_(?P<kind>dashboard|trees|crew)$")
_DUP_SUFFIX = re.compile(r"\s*\(\d+\)$")


class TeamDataError(ValueError):
    """A teammate file is missing, malformed, or fails validation."""


def canonical_name(stem: str) -> str:
    """``02_13B_Jackal_dashboard (2)`` -> ``02_13B_Jackal_dashboard`` (Windows duplicate-download suffix)."""
    return _DUP_SUFFIX.sub("", stem)


# ------------------------------------------------------------------ discovery --

@dataclass
class Inventory:
    folder: Path
    runs: dict = field(default_factory=dict)          # seq -> {kind: Path}
    summaries: dict = field(default_factory=dict)     # name -> Path
    noncanonical: list = field(default_factory=list)  # file names that carry a duplicate suffix
    conflicts: list = field(default_factory=list)     # canonical name claimed by more than one file
    unrecognized: list = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return sum(len(k) for k in self.runs.values()) + len(self.summaries)

    @property
    def sequences(self) -> list[str]:
        """Sequences with all three per-sequence files."""
        return sorted(s for s, k in self.runs.items() if all(x in k for x in KINDS))

    @property
    def complete(self) -> bool:
        return bool(self.sequences) and all(s in self.summaries for s in SUMMARIES) and not self.conflicts


def discover(folder: Path = TEAM_DIR) -> Inventory:
    """Find teammate files. Canonical names win over duplicate-suffixed copies; both are reported."""
    inv = Inventory(Path(folder))
    if not inv.folder.exists():
        return inv
    files = sorted(inv.folder.glob("*.csv"), key=lambda p: (canonical_name(p.stem) != p.stem, p.name))
    for p in files:
        canon = canonical_name(p.stem)
        if canon != p.stem:
            inv.noncanonical.append(p.name)
        m = _SEQ_FILE.match(canon)
        if m:
            slot = inv.runs.setdefault(m["seq"], {})
            if m["kind"] in slot:
                inv.conflicts.append((canon, slot[m["kind"]].name, p.name))
            else:
                slot[m["kind"]] = p
        elif canon in SUMMARIES:
            if canon in inv.summaries:
                inv.conflicts.append((canon, inv.summaries[canon].name, p.name))
            else:
                inv.summaries[canon] = p
        else:
            inv.unrecognized.append(p.name)
    return inv


# ------------------------------------------------------------------- parsing --

_TRUE, _FALSE = {"true", "1", "yes", "t"}, {"false", "0", "no", "f"}


def parse_bool(s: pd.Series, name: str = "column") -> np.ndarray:
    """Strict boolean parsing. Unknown or missing values raise instead of becoming False."""
    if s.dtype == bool:
        return s.to_numpy()
    v = s.astype("string").str.strip().str.lower()
    bad = v.isna() | ~v.isin(_TRUE | _FALSE)
    if bad.any():
        raise TeamDataError(f"{name}: {int(bad.sum())} value(s) are not booleans, e.g. {list(s[bad].head(3))}")
    return v.isin(_TRUE).to_numpy()


def _require(df: pd.DataFrame, cols, where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise TeamDataError(f"{where}: missing column(s) {missing}; found {list(df.columns)}")


def _numeric(df: pd.DataFrame, cols, where: str, optional=()) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        raw = df[c]
        num = pd.to_numeric(raw, errors="coerce")
        bad = num.isna() | ~np.isfinite(num.to_numpy(float, na_value=np.nan))
        if c in optional:
            bad &= raw.notna()
        if bad.any():
            raise TeamDataError(f"{where}: column '{c}' has {int(bad.sum())} missing or non-numeric value(s)")
        df[c] = num.astype(float)
    return df


def expected_state(sigma) -> np.ndarray:
    """Teammate's documented rule: 0 if 2s <= 0.3 m, 1 if <= 1.5 m, else 2."""
    s2 = 2.0 * np.asarray(sigma, float)
    return np.where(s2 <= TRUSTED_2SIGMA_M, 0, np.where(s2 <= LOST_2SIGMA_M, 1, 2))


def outage_intervals(t: np.ndarray, gps_out: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) of each run of gps_out == True. end = first sample with GPS back (or last sample)."""
    g = np.asarray(gps_out, bool).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, g, 0]))
    return [(float(t[a]), float(t[min(b, len(t) - 1)])) for a, b in zip(edges[::2], edges[1::2])]


def tree_pass_times(dash: pd.DataFrame, trees: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Time and distance at which the reference path comes closest to each tree."""
    if not len(trees):
        return np.zeros(0), np.zeros(0)
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(np.c_[dash.x_true, dash.y_true]).query(np.c_[trees.x, trees.y])
    return dash.t.to_numpy()[idx], dist


# ------------------------------------------------------------------ one run --

@dataclass
class TeamRun:
    sequence: str
    dash: pd.DataFrame
    trees: pd.DataFrame
    crew: pd.DataFrame
    outages: list                      # [(start_s, end_s)] from gps_out
    checks: list                       # human-readable validation results
    error_check_max_diff_m: float      # |error_m - hypot(est - true)|, max
    state_mismatches: int              # rows where state disagrees with the 2*sigma thresholds
    tree_t_pass: np.ndarray            # closest approach of the reference path to each tree (s)
    tree_timing_agreement: float       # share of trees whose during_outage matches gps_out at t_pass
    files: dict = field(default_factory=dict)

    @property
    def tree_reveal(self) -> str:
        """'progressive' when the closest-approach rule reproduces the teammate's own
        during_outage flags for >= 90 % of trees, otherwise 'final_only' (timing not reconstructable)."""
        return "progressive" if self.tree_timing_agreement >= REVEAL_MIN_AGREEMENT else "final_only"

    @property
    def t_max(self) -> float:
        return float(self.dash.t.iloc[-1])

    def index_at(self, t: float) -> int:
        return max(int(np.searchsorted(self.dash.t.to_numpy(), t, side="right") - 1), 0)


def _read(path: Path, cols, where: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:  # unreadable file
        raise TeamDataError(f"{where}: cannot read {path.name} ({e})") from e
    _require(df, cols, where)
    return df


def load_run(sequence: str, folder: Path = TEAM_DIR, inv: Inventory | None = None) -> TeamRun:
    inv = inv or discover(folder)
    files = inv.runs.get(sequence, {})
    missing = [k for k in KINDS if k not in files]
    if missing:
        raise TeamDataError(f"{sequence}: missing teammate file(s) {missing} in {inv.folder}")
    checks: list[str] = []

    d = _read(files["dashboard"], DASHBOARD_COLUMNS, f"{sequence} dashboard")
    num = [c for c in DASHBOARD_COLUMNS if c not in ("gps_out", "state")]
    d = _numeric(d, num + ["state"], f"{sequence} dashboard")
    d["gps_out"] = parse_bool(d["gps_out"], f"{sequence} dashboard gps_out")
    if not np.all(np.isin(d.state, (0, 1, 2))):
        raise TeamDataError(f"{sequence} dashboard: state must be 0, 1 or 2")
    d["state"] = d.state.astype(int)
    dt = np.diff(d.t.to_numpy())
    if np.any(dt <= 0):
        raise TeamDataError(f"{sequence} dashboard: t is not strictly increasing "
                            f"({int((dt <= 0).sum())} non-increasing step(s))")
    if (d.sigma < 0).any():
        raise TeamDataError(f"{sequence} dashboard: negative sigma")
    checks.append(f"{len(d)} rows, t {d.t.iloc[0]:.1f} to {d.t.iloc[-1]:.1f} s, median step "
                  f"{np.median(dt):.3f} s, strictly increasing")
    err = np.hypot(d.x_est - d.x_true, d.y_est - d.y_true).to_numpy()
    emax = float(np.max(np.abs(err - d.error_m.to_numpy())))
    checks.append(f"error_m matches the distance between estimate and reference (max difference {emax:.1e} m)"
                  if emax < 1e-6 else f"WARNING: error_m differs from the position columns by up to {emax:.3f} m "
                                      f"(shown as supplied)")
    mism = int((expected_state(d.sigma) != d.state.to_numpy()).sum())
    checks.append("state agrees with the documented 2*sigma thresholds on every row" if mism == 0 else
                  f"WARNING: state disagrees with the 2*sigma thresholds on {mism} row(s) (shown as supplied)")
    bub = float(np.max(np.abs(d.bubble_m - (2.0 + 2.0 * d.sigma))))
    checks.append(f"bubble_m = 2 m + 2*sigma (max deviation {bub:.1e} m)" if bub < 1e-6 else
                  "bubble_m is not 2 m + 2*sigma; shown as supplied")

    trees = _read(files["trees"], TREE_COLUMNS, f"{sequence} trees")
    trees = _numeric(trees, ("x", "y"), f"{sequence} trees")
    bad = ~trees.status.isin(SPRAY_STATUSES)
    if bad.any():
        raise TeamDataError(f"{sequence} trees: unknown status value(s) {sorted(trees.status[bad].unique())}")
    trees["during_outage"] = parse_bool(trees["during_outage"], f"{sequence} trees during_outage")

    crew = _read(files["crew"], CREW_COLUMNS, f"{sequence} crew")
    crew = _numeric(crew, CREW_COLUMNS, f"{sequence} crew").sort_values("t_outage_start").reset_index(drop=True)
    if (crew.t < crew.t_outage_start).any():
        raise TeamDataError(f"{sequence} crew: t is before t_outage_start")

    outs = outage_intervals(d.t.to_numpy(), d.gps_out.to_numpy())
    checks.append(f"{len(outs)} simulated GPS outage(s) from gps_out" +
                  (f", {np.mean([b - a for a, b in outs]):.0f} s each on average" if outs else ""))
    t_pass, _ = tree_pass_times(d, trees)
    if len(trees):
        k = np.clip(np.searchsorted(d.t.to_numpy(), t_pass), 0, len(d) - 1)
        agree = float(np.mean(d.gps_out.to_numpy()[k] == trees.during_outage.to_numpy()))
    else:
        agree = 0.0
    checks.append(f"tree timing: the closest-approach rule reproduces the teammate's during_outage flag for "
                  f"{100 * agree:.1f} % of {len(trees)} trees")
    return TeamRun(sequence, d.reset_index(drop=True), trees.reset_index(drop=True), crew, outs, checks, emax,
                   mism, t_pass, agree, files={k: str(v) for k, v in files.items()})


# ----------------------------------------------------------------- summaries --

def load_summaries(folder: Path = TEAM_DIR, inv: Inventory | None = None) -> dict[str, pd.DataFrame | None]:
    """The three summary tables, schema-checked. Missing file -> None."""
    inv = inv or discover(folder)
    out = {}
    for name, cols in SUMMARY_COLUMNS.items():
        p = inv.summaries.get(name)
        if p is None:
            out[name] = None
            continue
        df = _read(p, cols, name)
        numeric = [c for c in cols if c not in ("seq", "group", "setup")]
        out[name] = _numeric(df, numeric, name, optional=SUMMARY_OPTIONAL.get(name, ()))
    return out


def primary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the presentation comparison: excludes 'Fixed slip from 02 (no learning)'."""
    return df[~df.setup.isin(EXCLUDED_SETUPS)].reset_index(drop=True)
