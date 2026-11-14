@AGENTS.md

## Claude Code specifics

* **No AI attribution.** Never add `Co-Authored-By` trailers, "Generated with Claude Code" footers, or any other AI
  attribution to commits or pull requests. `.claude/settings.json` sets `attribution.commit` and
  `attribution.pr` to empty strings; keep it that way. Commits use the developer's own configured Git identity.
* After committing, check the message with `git log -1 --format=full`.
* Personal permissions belong in `.claude/settings.local.json` (ignored by Git), not in the shared
  `.claude/settings.json`.
* On the maintainer's machine the original backend checkout (sibling folder `UIUC-AgriRobotics-2026`) is kept as
  a backup. Nothing in this project may depend on it; do not edit or delete it unless the owner asks.
