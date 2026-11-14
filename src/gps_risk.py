"""SP6 — does canopy information predict GPS-quality degradation?

Follows the team plan's go/no-go rule: first check whether GPS quality
actually varies along the route. Only if it does, test the canopy -> quality
relationship on held-out *contiguous* data (no random frame splits).

Everything returned is labelled ``observed`` (measured) or ``modelled``
(predicted) so the dashboard can keep them visually distinct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .coordinate_utils import interp_series


def quality_variation_check(gq: pd.DataFrame, metric: str, degraded_if_above: float | None = None,
                            degraded_if_below: float | None = None, min_cv: float = 0.10,
                            min_degraded_pct: float = 2.0) -> dict:
    """Go/no-go: does ``metric`` vary enough along the route to be worth modelling?"""
    v = gq[metric].dropna().to_numpy(float)
    if len(v) == 0:
        return {"go": False, "reason": f"no values for {metric}"}
    cv = float(np.std(v) / abs(np.mean(v))) if np.mean(v) != 0 else np.inf
    deg = np.zeros(len(v), bool)
    if degraded_if_above is not None:
        deg |= v > degraded_if_above
    if degraded_if_below is not None:
        deg |= v < degraded_if_below
    pct = 100 * float(deg.mean())
    go = cv >= min_cv or pct >= min_degraded_pct
    return {"go": bool(go), "metric": metric, "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)), "cv": cv, "degraded_pct": pct,
            "reason": ("quality varies enough to investigate" if go else
                       f"quality nearly constant (CV {cv:.2f} < {min_cv}, degraded {pct:.1f}% < {min_degraded_pct}%): "
                       "canopy-risk prediction not supported by this data")}


def grid_aggregate(df: pd.DataFrame, value_cols, cell_m: float = 5.0) -> pd.DataFrame:
    g = df.copy()
    g["cx"] = (np.floor(g["x"] / cell_m) + 0.5) * cell_m
    g["cy"] = (np.floor(g["y"] / cell_m) + 0.5) * cell_m
    agg = g.groupby(["cx", "cy"])[list(value_cols)].mean()
    agg["n"] = g.groupby(["cx", "cy"]).size()
    return agg.reset_index()


def canopy_quality_relationship(canopy: pd.DataFrame, gq: pd.DataFrame, metric: str,
                                train_frac: float = 0.6) -> dict:
    """Relate canopy index to GPS quality with a contiguous time split.

    Fits quality ~ a + b * ndvi on the first ``train_frac`` of the route and
    reports held-out R², MAE vs a predict-the-training-mean baseline, plus
    Spearman correlation on the held-out block."""
    t = gq["t"].to_numpy()
    nd = interp_series(t, canopy["t"].to_numpy(), canopy["ndvi"].to_numpy())
    q = gq[metric].to_numpy(float)
    m = np.isfinite(nd) & np.isfinite(q)
    t, nd, q = t[m], nd[m], q[m]
    if len(t) < 30:
        return {"ok": False, "reason": "fewer than 30 matched canopy/GPS samples"}
    cut = t[0] + train_frac * (t[-1] - t[0])
    tr, te = t <= cut, t > cut
    b, a = np.polyfit(nd[tr], q[tr], 1)
    pred = a + b * nd[te]
    base = np.full(te.sum(), q[tr].mean())
    ss_res, ss_tot = np.sum((q[te] - pred) ** 2), np.sum((q[te] - q[te].mean()) ** 2)
    rho, p = stats.spearmanr(nd[te], q[te])
    return {"ok": True, "metric": metric, "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "slope": float(b), "intercept": float(a),
            "r2_test": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "mae_test": float(np.mean(np.abs(q[te] - pred))),
            "mae_baseline_mean": float(np.mean(np.abs(q[te] - base))),
            "spearman_rho_test": float(rho), "spearman_p_test": float(p),
            "split": f"contiguous time split: first {int(train_frac * 100)}% train, rest test"}


def risk_map(canopy: pd.DataFrame, gq: pd.DataFrame, metric: str, model: dict | None,
             higher_is_worse: bool = True, cell_m: float = 5.0) -> pd.DataFrame:
    """Grid with observed quality and (if a model exists) modelled quality."""
    obs = grid_aggregate(gq.rename(columns={metric: "observed"}), ["observed"], cell_m)
    can = grid_aggregate(canopy, ["ndvi"], cell_m)
    g = obs.merge(can[["cx", "cy", "ndvi"]], on=["cx", "cy"], how="outer")
    if model and model.get("ok"):
        g["modelled"] = model["intercept"] + model["slope"] * g["ndvi"]
    else:
        g["modelled"] = np.nan
    return g
