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
