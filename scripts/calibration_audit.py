r"""Uncertainty-calibration audit (read-only; changes nothing in the estimator or the UI).

Why does the EKF's 95 % position region contain the RTK reference only ~53–56 % of the
time on 06_14B? This script measures, it does not tune.

    .venv\Scripts\python scripts\calibration_audit.py [SEQUENCE]

Writes .cache/calibration_audit_<SEQ>.json and prints a summary.
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from dashboard import backend as be  # noqa: E402
from src.coordinate_utils import interp_series  # noqa: E402
from src.outage_simulator import OutageScenario, gps_proxy_from_reference  # noqa: E402
from src.pipeline import run_experiment  # noqa: E402

CHI2_95 = 5.991


def ellipse_stats(res, mask_fn=None):
    """Coverage of the r95 circle (what the UI shows) and of the true 95 % ellipse, plus mean NEES (ideal 2)."""
    err, est = res.error, res.estimate
    t = err.t.to_numpy()
    ex, ey = err.est_x.to_numpy() - err.ref_x.to_numpy(), err.est_y.to_numpy() - err.ref_y.to_numpy()
    vx = interp_series(t, est.t.to_numpy(), est.var_x.to_numpy())
    vy = interp_series(t, est.t.to_numpy(), est.var_y.to_numpy())
    cxy = interp_series(t, est.t.to_numpy(), est.cov_xy.to_numpy())
    r95 = interp_series(t, est.t.to_numpy(), est.r95.to_numpy())
    det = vx * vy - cxy ** 2
    nees = (vy * ex ** 2 - 2 * cxy * ex * ey + vx * ey ** 2) / det
    m = np.ones(len(t), bool) if mask_fn is None else mask_fn(t)
    e = np.hypot(ex, ey)[m]
    return {"samples": int(m.sum()),
            "coverage_r95_circle_pct": float(100 * np.mean(e <= r95[m])),
            "coverage_95_ellipse_pct": float(100 * np.mean(nees[m] <= CHI2_95)),
            "mean_nees_ideal_2": float(np.mean(nees[m])),
            "median_err_m": float(np.median(e)), "median_r95_m": float(np.median(r95[m])),
            "inflation_needed_for_95pct_DIAGNOSTIC": float(np.percentile(e / r95[m], 95))}


def gps_residuals(gps: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Raw receiver fix minus post-processed reference, at fix times."""
    t = gps.t.to_numpy()
    m = (t >= ref.t.iloc[0]) & (t <= ref.t.iloc[-1])
    t, gx, gy = t[m], gps.x.to_numpy()[m], gps.y.to_numpy()[m]

    def resid(lag):
        rx = np.interp(t + lag, ref.t, ref.x)
        ry = np.interp(t + lag, ref.t, ref.y)
        return gx - rx, gy - ry

    dx, dy = resid(0.0)
    # along/cross-track using the reference heading
    hx, hy = np.gradient(np.interp(t, ref.t, ref.x)), np.gradient(np.interp(t, ref.t, ref.y))
    n = np.hypot(hx, hy)
    moving = n > 1e-3
    ux, uy = hx[moving] / n[moving], hy[moving] / n[moving]
    along = dx[moving] * ux + dy[moving] * uy
    cross = -dx[moving] * uy + dy[moving] * ux
    lags = np.arange(-0.30, 0.301, 0.01)
    rms = [float(np.sqrt(np.mean(np.square(resid(L)).sum(0)))) for L in lags]
    best = float(lags[int(np.argmin(rms))])
    bx, by = resid(best)

    def acf_time(x, dt):
        x = x - x.mean()
        c = np.correlate(x, x, "full")[len(x) - 1:]
        c = c / c[0]
        k = np.flatnonzero(c < np.exp(-1))
        return float(k[0] * dt) if len(k) else np.inf

    dt = float(np.median(np.diff(t)))
    rep_sigma = float(np.median(gps.sigma)) if "sigma" in gps else np.nan
    rad = np.hypot(dx, dy)
    return {
        "fixes": int(len(t)), "rate_hz": 1 / dt, "reported_sigma_m": rep_sigma,
        "mean_bias_x_m": float(dx.mean()), "mean_bias_y_m": float(dy.mean()),
        "std_x_m": float(dx.std()), "std_y_m": float(dy.std()),
        "median_radial_m": float(np.median(rad)), "p95_radial_m": float(np.percentile(rad, 95)),
        "frac_radial_within_reported_95pct_circle": float(np.mean(rad <= rep_sigma * np.sqrt(CHI2_95))),
        "mean_along_track_m": float(along.mean()), "std_along_track_m": float(along.std()),
        "mean_cross_track_m": float(cross.mean()), "std_cross_track_m": float(cross.std()),
        "best_time_lag_s": best, "rms_at_lag0_m": rms[int(np.argmin(np.abs(lags)))],
        "rms_at_best_lag_m": min(rms),
        "residual_correlation_time_s_x": acf_time(dx, dt), "residual_correlation_time_s_y": acf_time(dy, dt),
        "median_radial_after_best_lag_m": float(np.median(np.hypot(bx, by))),
    }


def main(seq="06_14B_Jackal"):
    s = be.load_streams(seq)
    t0 = s.time_span()[0]
    out = {"sequence": seq, "gps_source": None}
    sens = ("odom", "imu", "gps") if s.available().get("imu") else ("odom", "gps")
    out["sensors"] = "+".join(sens)
    # 1) as shown in the dashboard (recorded GPS if extracted, else proxy)
    r_rec = run_experiment(s, sens, OutageScenario())
    out["gps_source"] = r_rec.gps_source
    out["no_outage_recorded_gps"] = ellipse_stats(r_rec)
    # 2) same estimator, same sequence, but the white-noise GPS proxy (3 cm, labelled simulated)
    s_proxy = be.SensorStreams(**{**s.__dict__, "gps": gps_proxy_from_reference(s.reference, sigma=0.03)})
    r_px = run_experiment(s_proxy, sens, OutageScenario())
    out["no_outage_proxy_gps_SIMULATED"] = ellipse_stats(r_px)
    # 3) phase split with a 60 s outage
    r_out = run_experiment(s, sens, OutageScenario(outages=[(100.0, 60.0)]))
    a, b = r_out.outages_abs[0]
    out["outage60_phase_gps_available"] = ellipse_stats(r_out, lambda t: (t < a) | (t > b + 30))
    out["outage60_phase_during_outage"] = ellipse_stats(r_out, lambda t: (t >= a) & (t <= b))
    out["outage60_phase_30s_after"] = ellipse_stats(r_out, lambda t: (t > b) & (t <= b + 30))
    # 4) raw receiver vs reference
    if s.gps is not None and str(s.gps["source"].iloc[0]) != "proxy_from_reference":
        out["raw_fix_vs_reference"] = gps_residuals(s.gps, s.reference)
        # 5) is the residual a position-proportional scale difference? (fit, then a DIAGNOSTIC-ONLY rerun)
        g, r = s.gps, s.reference
        t = g.t.to_numpy()
        m = (t >= r.t.iloc[0]) & (t <= r.t.iloc[-1])
        ry = np.interp(t[m], r.t, r.y)
        dy = g.y.to_numpy()[m] - ry
        k = float(np.sum(dy * ry) / np.sum(ry * ry))
        out["y_scale_fit"] = {"k_fraction": k, "k_percent": 100 * k,
                              "residual_std_after_fit_m": float(np.std(dy - k * ry)),
                              "r2": float(1 - np.var(dy - k * ry) / np.var(dy)),
                              "max_offset_on_route_m": float(np.max(np.abs(k * ry)))}
        g2 = g.copy()
        g2["y"] = g2["y"] / (1 + k)        # DIAGNOSTIC ONLY — never used by the dashboard
        s_fix = be.SensorStreams(**{**s.__dict__, "gps": g2})
        out["DIAGNOSTIC_ONLY_gps_y_rescaled_no_outage"] = ellipse_stats(run_experiment(s_fix, sens, OutageScenario()))
    p = be.CACHE_DIR / f"calibration_audit_{seq}.json"
    be.CACHE_DIR.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main(*sys.argv[1:])
