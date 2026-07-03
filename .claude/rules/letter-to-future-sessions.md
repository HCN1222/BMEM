# Letter to Future Sessions

Written 2026-07-03 by a Claude Fable 5 session — the only one this environment
will get. Everything else in `.claude/rules/` is procedure; this file is
context and honest caveats. Read it once when you first encounter this setup,
and when something about the institution seems not to make sense.

## Three things the user didn't ask about, but matter most

**1. The biggest project risk is backtest seduction, not code bugs.**
The headline result (+378% Top-1, 2025 eval year) comes from ONE continuous
out-of-sample window in ONE macro regime, with full single-stock concentration.
Future sessions will inevitably be asked to "improve the model." The failure
mode: iterating against the 2025 eval year until it is effectively a training
set. Treat any improvement that is only evidenced by a better 2025 backtest as
unproven. Push for walk-forward or a later holdout before believing it, and say
so plainly even if the number looks great. Never tune thresholds or features
against the eval year.

**2. This institution is durable only if it's versioned and its anchors hold.**
Resolved on 2026-07-03: the user narrowed `.gitignore` (it previously excluded
all of `.claude/`; now only `settings.local.json` is ignored) and committed
the institution. Two fragile anchors remain: (a) the auto-memory directory is
keyed to the repo's absolute folder path on this machine — if the repo folder
is moved or renamed, memory silently starts from scratch; (b) the repo lives
in OneDrive, which both syncs (good) and locks files mid-write (occasionally
bad).
If you notice memory is unexpectedly empty or CLAUDE.md reverted, say so to the
user instead of silently rebuilding.

**3. The quality of a weak-model session is set by the request, and the user
can exploit that.** These rules force subagent delegations to carry acceptance
criteria — but the top-level request has no such gate. When the user includes
"done means X" in their ask (a command that must pass, a file that must exist),
Sonnet-class sessions perform near their ceiling; when the ask is vague,
they wander. If a request is ambiguous in a way that changes the work, ask ONE
sharp clarifying question up front (see judgment-rubrics §3 for what's worth
asking) — that trade is almost always worth it.

## How this institution most likely degrades, and prevention

- **Silent non-use.** Under "just quickly do X" pressure, sessions skip the
  routing table and the verifier. Prevention: the user should occasionally ask
  "which rules files did you consult, and what did the verifier report?" — if
  the answer is hollow, the institution has already rotted. Cheap spot-audits
  beat more rules.
- **Cargo-culting the templates.** Filling {SLOTS} with vague words satisfies
  the letter, not the point. The tell: acceptance criteria that nothing could
  ever FAIL ("code is clean"). Prevention: reject any subagent report that has
  no falsifiable PASS/FAIL rows.
- **Rule accretion.** Every incident adds a rule until CLAUDE.md is 400 lines
  and nothing is read. Prevention: the caps in `maintenance.md` (CLAUDE.md
  ≤150 lines, lessons compaction) — enforce them, they are the immune system.
- **Fact rot.** Model names/IDs in `model-dispatch.md` will go stale.
  Prevention: the verified-date + re-verify clause; don't patch from memory.

## Where my confidence is lowest (honest list)

1. **Escalation-ladder thresholds** (haiku: 1 strike; sonnet: 2 strikes; max 2
   retry rounds) are judgment defaults, not tuned to this user's workload.
   If they misfire in practice, adjust via lessons.md — the STRUCTURE (fail →
   escalate with evidence → de-escalate for batch work) matters more than the
   exact numbers.
2. **The "top 3" harness diagnosis** ranks failure modes from one session's
   audit, not longitudinal data. #1 (README tax) is measured and certain; the
   ranking of #2 vs #3 is informed guess.
3. **Custom agents (`scout`, `verifier`) were written per docs but never
   test-spawned** — agent definitions register at session start, so this
   session cannot see them. First future session: invoke each once; if the
   harness rejects a frontmatter field, fix per current docs and note it in
   lessons.md.
4. **Unresolved quota question:** whether safety-rerouted requests consume the
   origin model's quota is undocumented. If it ever matters, measure via
   `/usage` / https://platform.claude.com/usage rather than assuming.
5. **Decisions initially applied as unconfirmed defaults** (files in English;
   uncommitted; global `effortLevel` untouched) were all confirmed or resolved
   by the user later on 2026-07-03: institution committed, `effortLevel` set
   to `"xhigh"`, README test count fixed. No open decisions remain from the
   founding session (see `maintenance.md` § Decision log).

Good luck. Follow the rules literally when unsure; deviate only when you can
say precisely why the rule doesn't fit — and then write the deviation down.
