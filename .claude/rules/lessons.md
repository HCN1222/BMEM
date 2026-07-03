# Lessons Log

Append-only. Format defined in `maintenance.md`. Newest at the bottom.

## 2026-07-03 — Python must run in conda env BMEM
- Trigger: any Python/script execution in this repo
- Wrong: bare `python ...` → ModuleNotFoundError → attempting pip install
- Right: `conda run -n BMEM python ...` from repo root; never install packages
  to fix a missing-module error here
- Promoted: CLAUDE.md (Environment) and harness-diagnosis.md §3

## 2026-07-03 — README test count is stale; verify counts against code
- Trigger: adversarial review of new rules files during institution setup
- Wrong: copying "4 regression tests" from README.md into rules files
- Right: src/test_pipeline.py actually runs 5 tests (feature_consistency,
  feature_computation, hmm_probability_replication, long_xgb_signals,
  short_xgb_signals) — when a doc and the code disagree, the code wins;
  README fix pending user approval
- Promoted: CLAUDE.md (Project facts), maintenance.md (pending items)
