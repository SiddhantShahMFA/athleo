# Release Log

Use one line per shipped change.

| Date | Branch | Commit | Type | Summary | AI Tool | AI Usage | Human Check | Validation | Client Impact | Rollback | Defect Found |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-18 | feat/player-tracking-poc | uncommitted | feat | Refactored the desktop workflow into cached interactive playback with instant click-based player isolation and export-from-player; Snyk unavailable | codex | code | self-verified | Manual validation completed with `py_compile`, `--help`, click-hit/drawing assertions, and a full cached selected-player export smoke test to `/tmp/athleo_validation_selected.mp4` | Live-feeling local player that preprocesses once, reuses cached tracking data, and switches overlays immediately during playback | Restore the prior `main.py`, `README.md`, `requirements.txt`, and docs state | None |
| 2026-03-18 | feat/player-tracking-poc | uncommitted | feat | Added a local OpenCV desktop POC for YOLO26x soccer player tracking, preview, and selected-player export; Snyk unavailable | codex | code | self-verified | Manual validation completed with `py_compile`, `--help`, and a 30-frame smoke test covering cache reuse plus all-player and selected-player exports in the repo venv | New desktop player tracking workflow with cached metadata and export modes | Remove `main.py`, `requirements.txt`, and docs changes | None |
