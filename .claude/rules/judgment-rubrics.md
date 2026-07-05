# Judgment Rubrics

High-level judgment externalized into checkable rules. Each rubric has a rule,
one positive example (✓ do this) and one negative example (✗ not this).
Audience: Sonnet-class models. When in doubt, follow the rule literally.

## 1. When to escalate to a stronger model

**Rule:** Escalate only after a concrete, documented failure — and attach the
failure evidence. Mechanics and ladder: `model-dispatch.md` Rule 5.

- ✓ Sonnet twice produced rolling-HMM code whose window included the current
  day's future rows (test 3 of `test_pipeline.py` failed both times). Escalate
  to opus with both diffs and the test output.
- ✗ Haiku's file summary is shorter than expected but accurate. That is not a
  failure — ask haiku to expand the specific missing part; do not escalate.

## 2. When a task is actually DONE

**Rule:** "Done" requires ALL of:
1. Every acceptance criterion from the original request individually checked.
2. The code/command actually ran — you observed the exit status and output,
   not "this should work."
3. For any change under `src/` or `script/`:
   `conda run -n BMEM python src/test_pipeline.py --broker-id <id>` passes
   (all 5 tests), for at least one trained broker (1440, 1470, 1650, 8440).
4. `git diff` reviewed: no unrelated files touched, no leftover debug prints.
5. Produced artifacts verified at the file level (exists, expected
   columns/format). For multi-file or pipeline-affecting work this check is
   done by a fresh `verifier` agent, not the author; for a trivial
   single-file artifact, directly reading the file back yourself suffices.

- ✓ Edited `prepare_xgb_data.py`, reran Step 6 for broker 1440, test_pipeline
  5/5 pass, `git diff` shows only the intended file: done.
- ✗ "I've edited the function and the logic looks correct" with no run — not
  done. Also not done: tests pass because you skipped/weakened one.

## 3. When to stop and ask the user

**Rule:** Ask before: (a) destructive or hard-to-reverse actions — deleting or
overwriting anything under `data/`, `outputs/*/models/`, retrained model files,
`git push`, history rewrites; (b) changing signal semantics — thresholds
(long ≥ 0.6, short ≥ 0.8), label definitions (+10%/−10%, 10-day horizon),
feature definitions; (c) actions that spend external quota at scale (bulk
FinMind re-downloads); (d) instructions that contradict repo reality; (e) pure
taste calls with no objective criterion.

Do NOT ask for: reversible actions clearly implied by the request — creating
files, read-only runs, fixing a bug you introduced this session, adding tests.

- ✓ Your fix changes backtest numbers and the old result charts are now stale —
  ask whether to regenerate/overwrite `outputs/*/backtest/` before touching them.
- ✗ "May I read train_hmm.py to understand the flow?" — never ask this; just read.

## 4. Wrong-direction signals — change approach, don't retry

**Rule:** Any of these means your model of the problem is wrong. Stop patching;
go collect ground truth (minimal repro, targeted print, read the actual data)
or step back and restate the problem. Never spend a third attempt on the same
theory, and never "fix" by weakening a guard.

Signals:
- The same error persists after 2 genuinely different fixes.
- The fix on the table is to skip a test, broaden a `try/except`, relax the
  no-lookahead rule, or hardcode an expected value.
- The diff keeps growing well beyond the natural size of the task.
- You are editing outputs/artifacts so a check passes.

- ✓ After two failed fixes to a date-alignment bug, write a 10-line script that
  prints the actual merge keys for one stock and one week — then fix from facts.
- ✗ Test 3 keeps failing, so comment it out "temporarily" and proceed. Never.

## 5. Quality floor — verify before "done" on any pipeline change

**Rule:** These invariants define this project. Check each one that your change
could plausibly touch; record what you checked in your report.

1. **No lookahead:** every feature/probability at date t uses only data ≤ t.
   Rolling windows must not be centered; merges must not pull future rows;
   HMM inference stays on the rolling 120-day PAST window.
2. **Train/eval separation:** eval rows never participate in any fitting;
   validation is carved from train only.
3. **Broker namespace:** with `--broker-id X`, nothing is read from or written
   to another broker's directories.
4. **Reproducibility:** rerunning the same step on the same inputs gives the
   same outputs (fixed seeds where applicable).
5. **Regression gate:** `test_pipeline.py` all 5 tests pass.

- ✓ Added a new 20-day feature: confirmed `.rolling(20)` with default (trailing)
  window, reran test_pipeline, noted "invariants 1 and 5 checked" in the report.
- ✗ "The feature is standard, lookahead is unlikely" — unchecked invariant
  claims are worth nothing; check or say you didn't.

## 6. Report honesty (applies to every rubric)

**Rule:** Report what actually happened. Failures with output, skipped steps as
skipped, assumptions as assumptions. An honest FAIL is a good report; a padded
PASS is the worst possible outcome because it poisons every later decision.
