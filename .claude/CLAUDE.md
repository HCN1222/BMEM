# BMEM — Claude Instructions (Index)

This file is a routing index. Deep rules live in `.claude/rules/` and are read
ON DEMAND when their trigger hits — do not preload them, and do NOT read the
full README at startup (routing table below replaces that old rule).

## Environment (always applies)

- Every Python command: `conda run -n BMEM python ...`, run from the repo root.
  `ModuleNotFoundError` = you forgot the env. Never pip/conda install to fix it.
- Windows 11; native shell is PowerShell (no `&&`); Bash tool available for
  POSIX syntax. Repo is under OneDrive — transient file locks possible: retry a
  failed write once, then report instead of looping.
- Never commit or push unless the user asks.

## Project facts (trust these; do not re-derive each session)

- Two-stage pipeline per broker: Gaussian HMM (hmmlearn) regime states →
  two XGBoost binary classifiers (long/short). Taiwan stocks, FinMind data.
- Broker namespace: every artifact lives under `{broker_id}/` in `data/brokers/`,
  `data/preprocessed_data/`, `outputs/`. Every pipeline script REQUIRES
  `--broker-id`. Trained brokers: 1440, 1470, 1650, 8440 (raw data exists for
  11 brokers).
- Production entry: `conda run -n BMEM python src/daily_update.py --broker-id <id>`
  → writes `outputs/{id}/daily/signals_{YYYY-MM-DD}.csv`.
- Regression gate: `conda run -n BMEM python src/test_pipeline.py --broker-id <id>`
  (5 tests; must pass after any `src/` or `script/` change. README says "4
  tests" — the script is authoritative).
- Signal thresholds: long ≥ 0.6, short ≥ 0.8. Labels: ±10% within 10 days.
  Never change thresholds, labels, or feature definitions without an explicit
  user request.
- NO LOOKAHEAD is the project's core invariant: anything computed for date t
  uses only data ≤ t; HMM inference runs on a rolling 120-day past window.

## README routing (read sections on demand, never the whole file)

`README.md` (~764 lines) is authoritative for detail. To use it:
`Grep '^## ' README.md` for current line numbers, then `Read` just the section.

| Task | Section |
|---|---|
| Run/debug daily pipeline | Production Pipeline; Quick Start |
| Retrain a broker from scratch | Full Research Workflow (Steps 1–8) |
| Touch features / HMM / XGBoost code | Core Algorithms & Features |
| Interpret signal CSVs | Output Format |
| Discuss performance / backtests | Backtest Results & Discussion |
| Deploy / regression-check | Testing |
| Understand data layout | Repository Structure |

## Rules routing (read the file when the trigger hits)

| Trigger | Read |
|---|---|
| Starting a multi-step task; repeated command failures | `.claude/rules/harness-diagnosis.md` |
| Delegating to a subagent; choosing a model; escalating | `.claude/rules/model-dispatch.md` |
| Unsure if done / whether to ask the user / stuck after 2 tries | `.claude/rules/judgment-rubrics.md` |
| Writing a subagent prompt | `.claude/rules/task-templates.md` |
| Editing anything in `.claude/`; recording a lesson | `.claude/rules/maintenance.md` |
| Why this institution exists; handoff context | `.claude/rules/letter-to-future-sessions.md` |

Custom subagents available: `scout` (haiku, read-only recon — use for any
exploration beyond 3 files) and `verifier` (sonnet, fresh-context adversarial
verification — use before declaring multi-file work done).

## Standing behaviors

- Check `.claude/rules/lessons.md` before long tasks; append to it when a
  command fails twice for an avoidable reason or the user corrects you
  (format in `maintenance.md`).
- Never `Read` tabular/binary data files under `data/` or `outputs/`;
  summarize them via pandas one-liners instead (`harness-diagnosis.md` §2).
  Exceptions: `*.md`, `*_meta.json`, and viewing a `*.png` chart with the Read
  tool when the task requires inspecting it.
- Delegate bulk reading/searching/verification to cheap subagents; keep the
  main conversation for judgment (`model-dispatch.md` Rule 1).
