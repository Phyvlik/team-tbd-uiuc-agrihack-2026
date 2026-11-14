@AGENTS.md

## Claude Code specifics

* **No AI attribution.** Never add `Co-Authored-By` trailers, "Generated with Claude Code" footers, or any other AI
  attribution to commits or pull requests. Set `"attribution": {"commit": "", "pr": ""}` in your own Claude Code
  settings. Commits use the developer's own configured Git identity.
* After committing, check the message with `git log -1 --format=full`.
* `.claude/` is ignored by Git: Claude Code settings stay local and are never committed.
* On the maintainer's machine the original backend checkout (sibling folder `UIUC-AgriRobotics-2026`) is kept as
  a backup. Nothing in this project may depend on it; do not edit or delete it unless the owner asks.
