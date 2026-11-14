# team_runs/

Drop teammate localization runs here. The dashboard reads every
`<SEQUENCE>__<method>.csv` in this folder (and in `results/runs/`) through
`dashboard/adapter.py`. Each file can have a JSON sidecar with the same name.

* Format, columns, frames and units: see "Teammate integration guide" in the
  top-level `README.md`.
* Worked example (synthetic values, format only):
  `docs/examples/06_14B_Jackal__example-synthetic.csv` and its `.json` sidecar.
* `<SEQUENCE>__orchard_geometry.csv` (`x, y[, row]`, gt.csv frame) is also read
  from this folder and replaces the illustrative tree rows. It is not treated as a run.

Imported runs are shown next to the live EKF runs. They never replace them.

## slip_learning/

The team's slip-learning localization exports (24 CSV files: per sequence `_dashboard`, `_trees`, `_crew`,
plus `test_results`, `spray_results`, `safety_results`). They drive the **Team Slip-Learning Results** source
through `dashboard/team_slip.py`, not the adapter above. Schema, validation and meaning of every column: README
section "Team Slip-Learning Localization Results". These are computed results; `x_true, y_true` are the
CitrusFarm `gt.csv` reference (Teng et al., UC Riverside, CC BY-SA 4.0), redistributed here under that license
with attribution.
