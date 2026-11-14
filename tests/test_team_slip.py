"""Teammate slip-learning results (team_runs/slip_learning): discovery, validation, rules."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import team_slip as ts

SEQS = ["01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal", "04_13D_Jackal", "05_13D_Jackal", "06_14B_Jackal",
        "07_14B_Jackal"]


@pytest.fixture(scope="module")
def inv():
    return ts.discover()


@pytest.fixture(scope="module")
def run02():
    return ts.load_run("02_13B_Jackal")


def _copy_run(tmp_path: Path, seq: str = "02_13B_Jackal") -> Path:
    for k in ts.KINDS:
        shutil.copy(ts.TEAM_DIR / f"{seq}_{k}.csv", tmp_path / f"{seq}_{k}.csv")
    return tmp_path


# ------------------------------------------------------------- discovery --

def test_discovers_all_24_files(inv):
    assert inv.n_files == 24 and inv.sequences == SEQS
    assert sorted(inv.summaries) == sorted(ts.SUMMARIES)
    assert inv.complete and not inv.noncanonical and not inv.unrecognized and not inv.conflicts


@pytest.mark.parametrize("stem,canon", [("02_13B_Jackal_dashboard (2)", "02_13B_Jackal_dashboard"),
                                        ("test_results (12)", "test_results"),
                                        ("02_13B_Jackal_trees", "02_13B_Jackal_trees")])
def test_canonical_name(stem, canon):
    assert ts.canonical_name(stem) == canon


def test_duplicate_suffix_is_mapped_and_conflicts_reported(tmp_path):
    src = ts.TEAM_DIR / "02_13B_Jackal_crew.csv"
    shutil.copy(src, tmp_path / "02_13B_Jackal_crew (2).csv")
    inv = ts.discover(tmp_path)
    assert inv.noncanonical == ["02_13B_Jackal_crew (2).csv"]
    assert inv.runs["02_13B_Jackal"]["crew"].name == "02_13B_Jackal_crew (2).csv"
    shutil.copy(src, tmp_path / "02_13B_Jackal_crew.csv")
    inv = ts.discover(tmp_path)
    assert inv.runs["02_13B_Jackal"]["crew"].name == "02_13B_Jackal_crew.csv"      # canonical wins
    assert inv.conflicts == [("02_13B_Jackal_crew", "02_13B_Jackal_crew.csv", "02_13B_Jackal_crew (2).csv")]
    assert not inv.complete


# ---------------------------------------------------------------- parsing --

def test_parse_bool_strict():
    assert ts.parse_bool(pd.Series(["True", "false", "1", "0", " TRUE "])).tolist() == [True, False, True, False, True]
    assert ts.parse_bool(pd.Series([True, False])).tolist() == [True, False]
    for bad in (["True", "maybe"], ["True", None]):
        with pytest.raises(ts.TeamDataError):
            ts.parse_bool(pd.Series(bad))


@pytest.mark.parametrize("seq", SEQS)
def test_every_run_loads_and_is_consistent(seq):
    r = ts.load_run(seq)
    d = r.dash
    assert list(d.columns) == list(ts.DASHBOARD_COLUMNS) and d.gps_out.dtype == bool
    assert np.all(np.diff(d.t) > 0) and d.t.iloc[0] == 0.0
    assert r.error_check_max_diff_m < 1e-6          # error_m equals hypot(est - true)
    assert r.state_mismatches == 0                   # states follow the 2*sigma thresholds
    assert set(r.trees.status) <= set(ts.SPRAY_STATUSES) and len(r.crew) == len(r.outages)
    assert all(b > a for a, b in r.outages)


def test_outage_transitions_02(run02):
    assert len(run02.outages) == 4
    for (a, b), start in zip(run02.outages, (60, 150, 240, 330)):
        assert abs(a - start) < 0.5 and abs((b - a) - 30) < 0.5
    t, g = run02.dash.t.to_numpy(), run02.dash.gps_out.to_numpy()
    a, b = run02.outages[0]
    assert g[np.searchsorted(t, a)] and not g[np.searchsorted(t, a) - 1] and not g[np.searchsorted(t, b)]


def test_outage_intervals_edges():
    t = np.arange(6.0)
    assert ts.outage_intervals(t, [0, 1, 1, 0, 0, 1]) == [(1.0, 3.0), (5.0, 5.0)]
    assert ts.outage_intervals(t, [0] * 6) == []


def test_tree_reveal_rule():
    assert ts.load_run("02_13B_Jackal").tree_reveal == "progressive"      # 99 % agreement, single pass
    assert ts.load_run("07_14B_Jackal").tree_reveal == "final_only"       # rows revisited: timing unknown


def _mutate(tmp_path, kind, fn, seq="02_13B_Jackal"):
    _copy_run(tmp_path, seq)
    p = tmp_path / f"{seq}_{kind}.csv"
    df = fn(pd.read_csv(p))
    df.to_csv(p, index=False)
    return ts.load_run(seq, folder=tmp_path)


def test_missing_column_raises(tmp_path):
    with pytest.raises(ts.TeamDataError, match="missing column"):
        _mutate(tmp_path, "dashboard", lambda d: d.drop(columns=["sigma"]))


def test_invalid_numeric_and_missing_values_raise(tmp_path):
    def bad(d):
        d = d.astype({"x_est": object})
        d.loc[5, "x_est"] = "oops"
        return d
    with pytest.raises(ts.TeamDataError, match="x_est"):
        _mutate(tmp_path, "dashboard", bad)
    with pytest.raises(ts.TeamDataError, match="sigma"):
        _mutate(tmp_path, "dashboard", lambda d: d.assign(sigma=d.sigma.where(d.index != 3)))


def test_unordered_timestamps_raise(tmp_path):
    with pytest.raises(ts.TeamDataError, match="strictly increasing"):
        _mutate(tmp_path, "dashboard", lambda d: d.iloc[::-1])


def test_bad_tree_status_and_bool_raise(tmp_path):
    with pytest.raises(ts.TeamDataError, match="status"):
        _mutate(tmp_path, "trees", lambda d: d.assign(status=d.status.where(d.index != 0, "sprayed")))
    with pytest.raises(ts.TeamDataError, match="during_outage"):
        _mutate(tmp_path, "trees", lambda d: d.astype({"during_outage": object}).assign(
            during_outage=lambda x: x.during_outage.where(x.index != 0, "perhaps")))


def test_inconsistent_values_are_reported_not_changed(tmp_path):
    r = _mutate(tmp_path, "dashboard", lambda d: d.assign(state=d.state.where(d.index != 10, 2),
                                                         error_m=d.error_m + (d.index == 20) * 0.5))
    assert r.state_mismatches == 1 and r.dash.state.iloc[10] == 2           # kept as supplied
    assert abs(r.error_check_max_diff_m - 0.5) < 1e-9 and any("WARNING" in c for c in r.checks)


def test_summaries_schema_and_primary_filter():
    s = ts.load_summaries()
    assert all(s[k] is not None for k in ts.SUMMARIES)
    t = s["test_results"]
    assert set(ts.EXCLUDED_SETUPS) <= set(t.setup) and not set(ts.EXCLUDED_SETUPS) & set(ts.primary(t).setup)
    assert set(ts.primary(t).setup) == set(ts.PRIMARY_SETUPS)
    assert s["safety_results"].closest_unsafe_m.isna().any()                 # empty = no unsafe encounter
    assert set(s["spray_results"].seq) == set(SEQS)


@pytest.mark.parametrize("state,bubble,dist,safe,productive", [
    (0, 2.1, None, "GO", "GO"),            # nothing going on
    (0, 2.1, 10.0, "GO", "GO"),            # worker beyond bubble + 5
    (0, 2.1, 6.0, "SLOW", "SLOW"),         # worker within bubble + 5
    (0, 2.1, 2.0, "STOP", "STOP"),         # worker inside the bubble
    (1, 3.0, None, "SLOW", "SLOW"),        # degraded
    (2, 5.0, None, "STOP", "SLOW"),        # lost: Safe stops, Productive slows
    (2, 5.0, 4.9, "STOP", "STOP"),         # worker inside bubble stops in both modes
    (1, 3.0, 2.99, "STOP", "STOP"),
])
def test_decision_rules(state, bubble, dist, safe, productive):
    assert ts.decide(state, bubble, dist, ts.SAFE)[0] == safe
    assert ts.decide(state, bubble, dist, ts.PRODUCTIVE)[0] == productive


def test_missing_worker_is_not_zero_distance():
    assert ts.decide(0, 2.1, None, ts.SAFE) == ("GO", ts.R_CLEAR)
    assert ts.decide(0, 2.1, float("nan"), ts.PRODUCTIVE) == ("GO", ts.R_CLEAR)
    with pytest.raises(ValueError):
        ts.decide(0, 2.1, None, "Aggressive")


def test_worker_visibility_window(run02):
    w = run02.crew.iloc[0]
    t = np.array([w.t_outage_start - 0.1, w.t_outage_start, w.t, w.t + 15.0, w.t + 15.1])
    assert ts.worker_visible(run02.crew.iloc[[0]], t)[0].tolist() == [False, True, True, True, False]


def test_worker_distance_uses_estimate_and_nearest_visible(run02):
    dist, idx = ts.worker_distances(run02)
    d, w = run02.dash, run02.crew.iloc[0]
    k = int(np.searchsorted(d.t, w.t))
    assert idx[k] == 0 and dist[k] == pytest.approx(np.hypot(d.x_est[k] - w.x, d.y_est[k] - w.y))
    j = int(np.searchsorted(d.t, 30.0))
    assert np.isnan(dist[j]) and idx[j] == -1


def test_decision_series_matches_scalar_rules(run02):
    for mode in ts.MODES:
        dec = ts.decision_series(run02, mode)
        for i in range(0, len(dec), 97):
            dd = dec.dist_m.iloc[i]
            assert dec.action.iloc[i] == ts.decide(int(run02.dash.state[i]), float(run02.dash.bubble_m[i]),
                                                   None if np.isnan(dd) else float(dd), mode)[0]
    safe, prod = ts.decision_series(run02, ts.SAFE), ts.decision_series(run02, ts.PRODUCTIVE)
    assert (safe.level >= prod.level).all() and (safe.level > prod.level).any()


def test_capabilities_thresholds():
    assert [ok for _, ok, _ in ts.capabilities(0.25)] == [True, True, True, True]
    assert [ok for _, ok, _ in ts.capabilities(1.2)] == [False, True, True, False]
    assert [ok for _, ok, _ in ts.capabilities(2.5)] == [False, True, False, False]
    assert [ok for _, ok, _ in ts.capabilities(3.5)] == [False, False, False, False]


def test_tree_status_timing(run02):
    assert (ts.tree_status_at(run02, -1.0) == "unreached").all()
    assert (ts.tree_status_at(run02, run02.t_max) == run02.trees.status.to_numpy()).all()
    mid = ts.tree_status_at(run02, 100.0)
    assert ((mid == "unreached") == (run02.tree_t_pass > 100.0)).all()
    r7 = ts.load_run("07_14B_Jackal")
    assert (ts.tree_status_at(r7, r7.t_max) == "unreached").all()
    assert (ts.tree_status_at(r7, 0.0, show_final=True) == r7.trees.status.to_numpy()).all()


def test_events_and_situation_02(run02):
    dec = ts.decision_series(run02, ts.SAFE)
    ev = ts.events(run02, dec)
    assert (ev.label == "GPS lost").sum() == 4 and (ev.label == "GPS restored").sum() == 4
    assert (ev.label == "Worker appears (simulated)").sum() == 4 and (ev.label == "STOP recommended").any()
    assert ev.t_rel.is_monotonic_increasing
    a, b = run02.outages[0]
    s = ts.situation(run02, dec, a + 5.0, ts.SAFE)
    assert s.gps_out and abs(s.outage_elapsed - 5.0) < 0.2 and s.since_restored is None
    assert "gps_lost" in ts.alert_keys(s)
    s2 = ts.situation(run02, dec, b + 3.0, ts.SAFE)
    assert not s2.gps_out and s2.since_restored is not None and "restored" in ts.alert_keys(s2)
    s0 = ts.situation(run02, dec, 20.0, ts.SAFE)
    assert (s0.action, s0.dist, ts.alert_keys(s0)) == ("GO", None, ["ok"])
    w = run02.crew.iloc[0]
    sw = ts.situation(run02, dec, float(w.t), ts.SAFE)
    assert sw.action == "STOP" and sw.dist is not None and "stop_worker" in ts.alert_keys(sw)
    assert ts.next_time(ev[ev.kind == "gps"].t_rel, 0.0) == pytest.approx(a)


def test_reference_matches_citrusfarm_gt():
    """x_true, y_true are the gt.csv reference on the same clock (needs the local CitrusFarm data)."""
    from src.data_loader import PROCESSED_DIR, load_processed
    if not (PROCESSED_DIR / "02_13B_Jackal").exists():
        pytest.skip("CitrusFarm data not available locally")
    r, ref = ts.load_run("02_13B_Jackal"), load_processed("02_13B_Jackal").reference
    t0 = ref.t.iloc[0]
    d = np.hypot(np.interp(r.dash.t + t0, ref.t, ref.x) - r.dash.x_true,
                 np.interp(r.dash.t + t0, ref.t, ref.y) - r.dash.y_true)
    assert np.median(d) < 0.005 and len(ref) == len(r.dash)
