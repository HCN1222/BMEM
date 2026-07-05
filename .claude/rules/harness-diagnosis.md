# Harness Diagnosis — Top 3 Failure Modes

Written 2026-07-03 by a Fable 5 session after auditing this repo and harness.
Audience: any Claude model working in this repo. Read this at the start of long or
multi-step tasks. Each item = the failure, why it happens here, and the exact fix.

## 1. Startup token tax: reading the full README every conversation

**Failure:** The old CLAUDE.md ordered "read README.md before answering any
question." README.md is 764 lines (~10k tokens). Most sessions (run a daily
update, fix one script, inspect one output) need none of it, but the rule burned
that context every single time, and pushed real work closer to compaction.

**Fix (do this instead):**
- CLAUDE.md now contains the operational facts directly. Trust them.
- When you genuinely need README detail, read ONLY the relevant section:
  1. `Grep` pattern `^## ` on `README.md` to get current section headers + line numbers.
  2. `Read` with `offset`/`limit` covering just that section.
- Task → section map:
  | You are doing | Read section |
  |---|---|
  | Running/debugging daily pipeline | "Production Pipeline", "Quick Start" |
  | Retraining a broker | "Full Research Workflow" (Steps 1–8) |
  | Touching features/HMM/XGBoost code | "Core Algorithms & Features" |
  | Interpreting outputs/signals | "Output Format" |
  | Discussing performance/results | "Backtest Results & Discussion" |
  | Deploying / regression checks | "Testing" |

## 2. Context flooding: dumping data artifacts and long logs into the main conversation

**Failure:** This repo is full of large artifacts (`data/**/*.parquet`,
`outputs/**/*.csv`, backtest logs, PNGs). Weak models tend to `Read` a CSV or
pipe a whole training log into the conversation. One 5k-row CSV read can cost
more context than the entire task. Parquet files are binary — reading them
directly produces garbage AND wastes tokens.

**Fix (hard rules):**
- Never `Read` any file under `data/` or `outputs/` directly, except
  `*_meta.json`, `*.md`, and `*.png` (the Read tool renders images — viewing a
  chart when the task requires it is fine). For tabular files, summarize via
  Python:
  `conda run -n BMEM python -c "import pandas as pd; df = pd.read_parquet(r'<path>'); print(df.shape); print(df.columns.tolist()); print(df.head(5))"`
  (use `read_csv` for CSVs).
- Long-running commands (training, backtest, daily update): run via Bash with
  `run_in_background: true`, or delegate to a subagent; report only the final
  ~20 lines and the verdict, never the full log.
- Exploration wider than 3 files → delegate to the `scout` agent (haiku) and
  accept only conclusions + `file:line` back (see `.claude/rules/model-dispatch.md`).

## 3. Environment retry spirals: wrong env, wrong cwd, wrong shell syntax

**Failure:** Three recurring Windows-specific traps: (a) running `python ...`
without the conda env → `ModuleNotFoundError` → model "fixes" it by trying to
pip install (wrong, and pollutes the base env); (b) running from a subdirectory
so relative paths like `data/brokers/...` break; (c) mixing bash syntax into
PowerShell (`&&`, `2>/dev/null`) or vice versa. Each spiral wastes 3–6 turns.

**Fix (exact templates):**
- Every Python invocation, from repo root:
  `conda run -n BMEM python <script> --broker-id <id>` or
  `conda run -n BMEM python -m src.experiments.<module> --broker-id <id>`
- `ModuleNotFoundError` means YOU FORGOT THE ENV. Never install packages to fix it.
- Two-strike rule: the same command failing twice = stop retrying. Re-check
  (env? cwd? shell syntax? file locked by OneDrive sync?) and if still stuck,
  follow `.claude/rules/judgment-rubrics.md` § Wrong-direction signals.
- This repo lives under OneDrive: transient file locks happen. Retry a failed
  file write once after a short pause; if it persists, report it — don't loop.
