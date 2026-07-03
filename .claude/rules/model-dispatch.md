# Model Dispatch Rules

Facts below were verified 2026-07-03 against official docs (code.claude.com/docs,
platform.claude.com/docs) and the live harness. If today is more than ~6 months
past that date, re-verify model names via a `claude-code-guide` subagent before
relying on them — model lineups change.

## Verified facts (do not fill from memory)

- Subagent `model` parameter accepts: `haiku` | `sonnet` | `opus` | `fable`
  (custom agent frontmatter also accepts full IDs and `inherit`).
- Current model IDs: Opus 4.8 = `claude-opus-4-8`, Sonnet 5 = `claude-sonnet-5`,
  Haiku 4.5 = `claude-haiku-4-5-20251001`, Fable 5 = `claude-fable-5`.
- Cost order (cheap → expensive): haiku < sonnet < opus < fable.
- Effort controls: per-session `/effort` or `--effort` (low…max);
  `settings.json` key `effortLevel` accepts only `low/medium/high/xhigh`;
  custom agents (`.claude/agents/*.md`) support an `effort` frontmatter field.
- Built-in subagent types: `general-purpose` (all tools), `Explore` (read-only
  search), `Plan` (architecture), `claude-code-guide` (docs questions).
- Project custom agents: `scout` (haiku recon), `verifier` (sonnet, fresh-context
  verification) — defined in `.claude/agents/`.
- UNVERIFIED: whether requests auto-rerouted between models for safety consume
  the origin model's quota. Check `/usage` or https://platform.claude.com/usage.

## Rule 1 — The commander does not descend

The main conversation does judgment, integration, and user communication ONLY.
Delegate to a subagent when ANY of these holds:
- You would read more than 3 files or ~300 lines just to find something.
- Repo-wide or multi-directory search where you don't know the location.
- Web research (docs, APIs, papers).
- Mechanical edits repeated across more than 3 files.
- Running a pipeline/training job mainly to capture its outcome.
- Verifying work that you (main conversation) produced.

Do NOT delegate: reading one known file section, a one-line fix, any decision
requiring judgment about scope or user intent.

## Rule 2 — Default assignments

| Task type | Send to |
|---|---|
| Locate code / map structure / summarize files | `scout` or `Explore`, model `haiku` |
| Bulk extraction, log triage, format conversion | `general-purpose`, model `haiku` |
| Implement, refactor, write tests, batch edits | `general-purpose`, model `sonnet` |
| Verify finished work | `verifier`, model `sonnet` |
| Architecture / plan for a risky change | `Plan`, model `opus` |
| Debugging that already defeated sonnet (see Rule 5) | `general-purpose`, model `opus` |

Start every subtask at the cheapest plausible tier. Never spawn `opus`/`fable`
"to be safe."

## Rule 3 — Every handoff carries three elements

Use the fill-in templates in `.claude/rules/task-templates.md`. A delegation
prompt without all three is malformed — rewrite it before sending:
1. **Goal + why** — what outcome, and the purpose, so the agent can resolve
   small ambiguities in the right direction.
2. **Acceptance criteria** — objectively checkable conditions (a command that
   must exit 0, a file that must exist with given columns, a question answered
   with `file:line` evidence).
3. **Report format** — structure and a line cap for the reply.

## Rule 4 — Report contract (what comes back)

- Subagents return conclusions, `file:line` references, and PASS/FAIL against
  each acceptance criterion — not raw file contents or full logs.
- Any product longer than ~30 lines: the subagent saves it to a file and
  returns the path.
- The commander relays to the user what matters; never paste a subagent's full
  report unedited if it exceeds ~30 lines.

## Rule 5 — Escalation and de-escalation ladder

- `haiku` wrong once on a subtask → resend to `sonnet`, including haiku's wrong
  answer and why it was wrong.
- `sonnet` fails the SAME subtask twice → escalate to `opus`, attaching the full
  failure trail: both attempts, exact errors/test output, what was already ruled
  out. Escalation without the trail just repeats the failure expensively.
- Once the hard part is solved (root cause found, pattern established), drop
  back to `sonnet`/`haiku` to batch-apply it. Don't keep `opus` for rote work.
- Maximum 2 retry rounds for the same approach at any tier. After that it is a
  wrong-direction signal — see `judgment-rubrics.md`, do not escalate further.
- Never escalate because output merely "feels" thin, or to redo a task that
  failed for environment reasons (fix the environment first — see
  `harness-diagnosis.md` §3).

## Rule 6 — Verification is never self-verification

- Whoever produced the work does not certify it.
- Files written → fresh-context read-back: spawn `verifier` with the spec and
  file list; it confirms existence, completeness, and spec match.
- Code changed → tests or a real run, executed (not reasoned about):
  `conda run -n BMEM python src/test_pipeline.py --broker-id <id>` for pipeline
  changes; otherwise actually execute the changed path.
- High-stakes judgment (signal thresholds, label definitions, anything touching
  the no-lookahead guarantee, deleting/overwriting model artifacts) → get a
  second opinion from a separately prompted agent, or produce 2–3 candidate
  answers and have a judge agent pick with reasons. If opinions disagree, stop
  and ask the user.
