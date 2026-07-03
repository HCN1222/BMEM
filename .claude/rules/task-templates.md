# Subagent Task Templates

Fill every {SLOT}. A delegation missing goal+why, acceptance criteria, or
report format is malformed (see `model-dispatch.md` Rule 3). Append the
standard footer to every prompt. Default models are starting points — apply
the escalation ladder in `model-dispatch.md` Rule 5.

**Standard footer (append verbatim to every template):**
```
If blocked or the task turns out different from described, STOP and report
what you found — do not improvise beyond scope. Report failures honestly
with the actual error output.
```

---

## 1. SEARCH / RECON — agent: scout or Explore, model: haiku

```
GOAL: Find {WHAT} in {WHERE/repo}. This is needed because {WHY — what decision
the answer feeds}.
CONTEXT: {1-3 lines: what the project/area is, anything already known}.
SCOPE: Look in {DIRS/GLOBS}. Do not read files under data/ or outputs/.
ACCEPTANCE: The answer names exact file:line locations for {WHAT}, or states
"not found" after checking {LIST OF PLACES/naming variants}.
REPORT: ≤{N=30} lines. Bullets: conclusion first, then file:line evidence.
No raw file dumps.
```

## 2. IMPLEMENTATION — agent: general-purpose, model: sonnet

```
GOAL: Implement {FEATURE/FIX} in {FILES}. Purpose: {WHY — user-visible effect}.
CONTEXT: {relevant facts: entry points, conventions, related code file:line}.
Environment: run Python as `conda run -n BMEM python ...` from repo root.
CONSTRAINTS: {invariants — for pipeline code always include: "must preserve
no-lookahead (features at date t use only data ≤ t) and broker namespacing"}.
Do not modify {OUT-OF-SCOPE FILES}. Do not change signal thresholds or labels.
ACCEPTANCE:
- {SPECIFIC BEHAVIOR, e.g. "signals CSV for 2026-07-02 contains column X"}
- `conda run -n BMEM python src/test_pipeline.py --broker-id {ID}` → 5/5 pass
- git diff touches only {FILES}
REPORT: ≤{N=40} lines: what changed (file:line per change), each acceptance
criterion PASS/FAIL with the command you ran, anything you assumed.
```

## 3. REFACTOR — agent: general-purpose, model: sonnet

```
GOAL: Refactor {TARGET} to {DESIRED SHAPE}. Purpose: {WHY}.
HARD RULE: behavior-preserving. No functional changes, however tempting.
Note improvement ideas in the report instead of applying them.
CONTEXT: {current structure, file:line; callers/consumers to keep working}.
ACCEPTANCE:
- {EVIDENCE OF UNCHANGED BEHAVIOR: test_pipeline 5/5, or before/after output
  comparison on {SAMPLE} — state which}
- No public interface changes except {ALLOWED LIST}.
REPORT: ≤{N=40} lines: mapping old→new (file:line), behavior-preservation
evidence, list of deferred improvement ideas (not applied).
```

## 4. RESEARCH — agent: general-purpose (web) or claude-code-guide (Claude docs), model: sonnet

```
GOAL: Answer: {QUESTIONS, numbered}. This feeds {DECISION}.
SOURCES: Prefer {official docs/primary sources}. Cite URL per claim.
ACCEPTANCE: Every question answered with a cited source, or explicitly marked
UNVERIFIED. No answers from memory for version numbers, model names, prices,
or API parameters.
REPORT: ≤{N=40} lines. Numbered answers matching the questions, ≤4 lines each,
URL per answer. Save any long extract to {SCRATCHPAD PATH} and return the path.
```

## 5. REVIEW / VERIFICATION — agent: verifier, model: sonnet

```
GOAL: Independently verify the following claimed work. Be adversarial: try to
falsify each claim, don't confirm-and-move-on.
SPEC (what was supposed to happen): {ORIGINAL ACCEPTANCE CRITERIA}.
CLAIMED: {WHAT THE AUTHOR SAYS WAS DONE, files touched}.
CHECKS:
- Read-back: files exist and match spec ({LIST}).
- Execution: run {COMMANDS, e.g. test_pipeline} and record real output.
- Invariants: {e.g. no-lookahead spot check, broker namespace} —
  see .claude/rules/judgment-rubrics.md §5.
ACCEPTANCE: every check has a PASS/FAIL verdict backed by evidence you
gathered yourself (file:line or command output), not by the author's report.
REPORT: ≤{N=30} lines: verdict table first (check → PASS/FAIL), then evidence
for each FAIL. Do NOT fix anything — report only.
```

---

## Choosing N (report line cap)

Default 30–40. Raise only when the acceptance criteria genuinely require more
(e.g., per-file verdicts across many files). If a subagent needs to hand back
something long, it writes a file and returns the path.
