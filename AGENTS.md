# Working on this repository

Instructions for anyone (human or AI assistant) changing this project. `README.md` explains what the project
does; this file explains how to work on it.

## Project shape

* One project folder, one Python environment. Everything runs from the repository root.
* `src/`: localization backend (EKF in `sensor_fusion.py`, `pipeline.run_experiment`, outages, integrity
  monitor, evaluation, CitrusFarm loading). Pure Python, no Streamlit.
* `dashboard/`: UI layer. `dashboard/backend.py` is the only module that imports `src/`; pages go through it.
* `dashboard/team_slip.py` (data and rules, no Streamlit) and `dashboard/team_ui.py` (views): the Team
  Slip-Learning Results source, replaying `team_runs/slip_learning/`. Treat those CSVs as the source of truth:
  validate and report, never recompute or "fix" their values. Pages branch on `team_ui.active()` and must keep
  the backend EKF path unchanged.
* `views/`: the five Streamlit pages, registered in `app.py`.
* `scripts/`: data download/extraction, batch experiments, Rerun export, calibration audit.
* `data/` (not committed), `results/` (committed batch tables), `team_runs/` (teammate run CSVs),
  `recordings/` (generated `.rrd`, not committed), `docs/`.

## Commands (Windows, from the repository root)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\export_rerun.py
```

## Testing expectations

* Run the full suite before every commit that touches code. All tests must pass; report failures with their
  output instead of skipping them.
* UI changes: also start the app and look at the affected page in both Operator and Engineer mode.
* Add or update a test when you fix a bug or add behaviour (`tests/test_app.py` for UI and adapter,
  `tests/test_core.py` for the backend, `tests/test_consistency.py` for Rerun vs Streamlit).
* Paths must be project-relative (`pathlib`, `Path(__file__)`); never hard-code a personal folder.

## Scientific rules

* Do not change the estimator, noise values, thresholds or teammate algorithms as a side effect of other work.
  Scientific changes get their own commit, a reason, and agreement from the owner.
* Never feed the reference (`gt.csv`) to an estimator. Tune on 02_13B only; report on the other sequences.
* Never inflate, rescale or hide uncertainty to make results look better. The 95% region is **Uncalibrated**
  until `scripts/calibration_audit.py` shows otherwise (`docs/UNCERTAINTY_CALIBRATION.md`).
* Label every displayed value: recorded, computed, simulated, predicted, illustrative. The GPS proxy, outages,
  crews, spray and tree rows are simulated or illustrative and must be described that way.
* Missing data stays missing: show *Not evaluated* or *Data unavailable*, never a substitute value.
* Keep `docs/METRIC_PROVENANCE.md` current when you add a displayed metric.

## Git workflow

* Conventional Commits: `type(scope): imperative subject`, types `feat`, `fix`, `refactor`, `test`, `docs`,
  `chore`, `perf`, `style`. Subject under 50 characters where possible, never over 72, no trailing period, no
  emojis. Use a body to explain why when it is not obvious.
* Commit often, one logical change per commit. Stage only related files; review `git diff --cached` first.
* Never force-push, rewrite published history, reset shared branches, or discard someone else's changes.
  Pull and merge (or rebase your own unpublished work) instead.
* Never commit `data/raw`, `data/processed`, `.bag` files, point clouds, `.rrd` recordings, `.venv`, secrets, or
  the organizer PDFs. Check `git status` before every commit.
* Preserve teammates' files and commits. If a change would overwrite their work, stop and ask.

## Writing style

* No em dashes and no double hyphens as punctuation in prose, documentation, UI text you add, or commit
  messages. Use commas, colons, parentheses or separate sentences.
* Plain, specific language. State limitations honestly; do not describe incomplete work as finished.
