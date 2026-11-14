"""Illustrative orchard geometry and a SIMULATED spray-application scenario.

Neither CitrusFarm nor the team backend provides surveyed tree positions or
per-tree spray labels. Everything here is therefore explicitly illustrative /
simulated and labelled as such in the UI:

* Synthetic source → tree lines come from the synthetic generator's known row
  layout (``streams.meta["rows"]``).
* Real source → tree positions are *inferred* from the recorded RTK path: on
  straight driving segments, trees are placed either side of the path at half
  an assumed row width, every assumed tree spacing. Candidates that would sit
  on the path or duplicate a neighbour are dropped. These are NOT measured
  tree positions. If a teammate provides real orchard geometry
  (``orchard_geometry.csv`` with ``x, y`` [, ``row``]), it is used instead.

Spray simulation rules (documented, deterministic)
-------------------------------------------------
* The sprayer has a lateral reach of ``REACH_M``; its spray fan covers
  ±``WINDOW_M`` along the row around the nozzle (robot centre).
* The *controller* decides when to spray from the ESTIMATED pose: it fires once
  for each tree it believes is abreast of the nozzle (along-track |offset| ≤
  ``FIRE_M``, lateral ≤ reach), and never re-fires for a tree it already treated.
* The spray physically lands on trees beside the TRUE pose (RTK reference),
  on the same side as the intended tree, within the same window.
* Outcome per tree, from the start up to the replay time:
  passed by the true path and 0 deposits → MISSED; 1 → TREATED;
  ≥ 2 → DOUBLE-TREATED; not yet passed → NOT YET REACHED.
Localization drift along the row therefore shows up as misses and doubles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROW_WIDTH_M = 6.4        # plan: rows are 20–22 ft wide
TREE_SPACING_M = 4.0     # assumption (citrus in-row spacing), illustrative only
REACH_M = 4.5
WINDOW_M = 1.0           # physical spray fan: ±1 m along the row
FIRE_M = 0.3             # controller fires when it believes the tree is abreast (±0.3 m)

NOT_REACHED, TREATED, MISSED, DOUBLE = 0, 1, 2, 3
WRONG_TREE = 4           # only in teammate results: the tree received another tree's dose
SPRAY_LABEL = {NOT_REACHED: "Not yet reached", TREATED: "Correctly treated", MISSED: "Missed",
               DOUBLE: "Double-treated", WRONG_TREE: "Wrong tree"}


@dataclass
class Orchard:
    trees: pd.DataFrame      # x, y, row
    origin: str              # "synthetic layout" | "inferred from recorded path" | "teammate geometry"
    note: str


def _heading(x, y, step=5):
    hx = np.gradient(x, step) if len(x) > step else np.gradient(x)
    hy = np.gradient(y, step) if len(y) > step else np.gradient(y)
    return np.arctan2(hy, hx)


def resample_path(x, y, ds: float = 0.5):
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s = np.arange(0, d[-1], ds)
    return np.interp(s, d, x), np.interp(s, d, y)


def build_orchard(streams, geometry_file: Path | None = None) -> Orchard:
    if geometry_file is not None and geometry_file.exists():
        g = pd.read_csv(geometry_file)
        if {"x", "y"} <= set(g.columns):
            g = g.assign(row=g["row"] if "row" in g else -1)[["x", "y", "row"]]
            return Orchard(g, "teammate geometry", f"Tree positions from {geometry_file.name}.")
    if streams.mode == "synthetic" and streams.meta.get("rows"):
        sp = float(streams.meta.get("row_spacing", 6.5))
        pts = []
        for i, (x0, y0, x1, y1) in enumerate(streams.meta["rows"]):
            n = int(np.hypot(x1 - x0, y1 - y0) // TREE_SPACING_M) + 1
            for side, dx in enumerate((-sp / 2, sp / 2)):
                for k in range(n):
                    f = k / max(n - 1, 1)
                    pts.append((x0 + dx + f * (x1 - x0), y0 + f * (y1 - y0), 2 * i + side))
        t = pd.DataFrame(pts, columns=["x", "y", "row"]).drop_duplicates(["x", "y"]).reset_index(drop=True)
        return Orchard(t, "synthetic layout", "Synthetic orchard layout (simulated).")
    # real: infer illustrative rows from the recorded path
    ref = streams.reference
    x, y = resample_path(ref.x.to_numpy(), ref.y.to_numpy(), 0.5)
    th = _heading(x, y, 6)
    # straightness: heading change over ±3 m
    k = 6
    dth = np.abs(np.angle(np.exp(1j * (np.roll(th, -k) - np.roll(th, k)))))
    straight = dth < np.radians(10)
    straight[:k] = straight[-k:] = False
    s = np.arange(len(x)) * 0.5
    cand = []
    for i in np.flatnonzero(straight):
        if (s[i] % TREE_SPACING_M) >= 0.5:
            continue
        nx, ny = -np.sin(th[i]), np.cos(th[i])
        for side in (-1, 1):
            cand.append((x[i] + side * nx * ROW_WIDTH_M / 2, y[i] + side * ny * ROW_WIDTH_M / 2))
    if not cand:
        return Orchard(pd.DataFrame(columns=["x", "y", "row"]), "inferred from recorded path", "No straight rows found.")
    c = np.array(cand)
    path_tree = cKDTree(np.column_stack([x, y]))
    dpath, _ = path_tree.query(c)
    c = c[dpath > 2.4]                      # not on any driven path
    keep, taken = [], cKDTree(c)
    used = np.zeros(len(c), bool)
    for i in range(len(c)):                 # merge duplicates from adjacent passes
        if used[i]:
            continue
        nb = taken.query_ball_point(c[i], 2.5)
        used[nb] = True
        keep.append(c[nb].mean(0))
    t = pd.DataFrame(keep, columns=["x", "y"])
    t["row"] = -1
    return Orchard(t, "inferred from recorded path",
                   f"Illustrative tree rows inferred from the robot's recorded RTK path (assumed {ROW_WIDTH_M} m row "
                   f"width, {TREE_SPACING_M:.0f} m tree spacing). Not surveyed tree positions.")


def _frame(px, py, th, tx, ty):
    dx, dy = tx - px, ty - py
    along = dx * np.cos(th) + dy * np.sin(th)
    lat = -dx * np.sin(th) + dy * np.cos(th)
    return along, lat


def simulate_spray(trees: pd.DataFrame, est: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Return spray events (t, intended tree, hit trees) and per-tree first-pass time.

    Returns (events with columns t, intended, hit [tree index or -1 = spray hit no tree],
    t_pass = first time the TRUE pose passed each tree)."""
    if trees is None or not len(trees):
        return pd.DataFrame(columns=["t", "intended", "hit"]), np.array([])
    T = trees[["x", "y"]].to_numpy()
    tree_idx = cKDTree(T)
    t = est.t.to_numpy()
    ex, ey, eth = est.x_ant.to_numpy(), est.y_ant.to_numpy(), est.theta.to_numpy()
    rx = np.interp(t, ref.t, ref.x)
    ry = np.interp(t, ref.t, ref.y)
    rth = _heading(rx, ry, 5)
    moving = np.r_[np.hypot(np.diff(rx), np.diff(ry)) > 0.01, False]
    sprayed = np.zeros(len(T), bool)
    t_pass = np.full(len(T), np.nan)
    events = []
    for k in range(len(t)):
        # true pass bookkeeping
        for j in tree_idx.query_ball_point((rx[k], ry[k]), REACH_M + WINDOW_M):
            if np.isnan(t_pass[j]):
                a, l = _frame(rx[k], ry[k], rth[k], T[j, 0], T[j, 1])
                if abs(a) <= WINDOW_M and abs(l) <= REACH_M:
                    t_pass[j] = t[k]
        if not moving[k]:
            continue
        for j in tree_idx.query_ball_point((ex[k], ey[k]), REACH_M + WINDOW_M):
            if sprayed[j]:
                continue
            a, l = _frame(ex[k], ey[k], eth[k], T[j, 0], T[j, 1])
            if abs(a) <= FIRE_M and abs(l) <= REACH_M:
                sprayed[j] = True
                side = np.sign(l)
                hit, best = -1, np.inf
                for h in tree_idx.query_ball_point((rx[k], ry[k]), REACH_M + WINDOW_M):
                    ah, lh = _frame(rx[k], ry[k], rth[k], T[h, 0], T[h, 1])
                    if abs(ah) <= WINDOW_M and abs(lh) <= REACH_M and np.sign(lh) == side and abs(ah) < best:
                        hit, best = h, abs(ah)
                events.append((t[k], j, hit))
    return pd.DataFrame(events, columns=["t", "intended", "hit"]), t_pass


def spray_status(n_trees: int, events: pd.DataFrame, t_pass: np.ndarray, t_now: float) -> np.ndarray:
    status = np.full(n_trees, NOT_REACHED)
    if n_trees == 0:
        return status
    ev = events[(events.t <= t_now) & (events.hit >= 0)]
    counts = np.bincount(ev.hit.to_numpy(int), minlength=n_trees) if len(ev) else np.zeros(n_trees, int)
    passed = np.nan_to_num(t_pass, nan=np.inf) <= t_now
    status[passed & (counts == 0)] = MISSED
    status[counts == 1] = TREATED
    status[counts >= 2] = DOUBLE
    return status
