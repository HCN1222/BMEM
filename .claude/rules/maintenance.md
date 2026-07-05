# Maintenance Protocol for .claude/ Institution Files

Governs how these rule files evolve. Audience: any model working here, and the
user. Established 2026-07-03.

## Edit permissions

**May edit WITHOUT asking the user:**
- `.claude/rules/lessons.md` — append-only, format below. Never rewrite or
  delete existing entries without user approval.
- Memory files + `MEMORY.md` index (the per-project memory directory).
- `.claude/settings.local.json` — only ADDING allow-rules for read-only
  commands; never add allow-rules for destructive commands.
- The session scratchpad.

**Ask the user FIRST (and back up before editing once approved):**
- `.claude/CLAUDE.md` and every file in `.claude/rules/` except `lessons.md`.
- `.claude/agents/*.md`.
- `README.md`, anything under `src/` or `script/` (normal code-review rules).
- Global `~/.claude/settings.json`.

**Backup convention:** before editing an ask-first file, copy it to
`.claude/archive/{filename}.{YYYY-MM-DD}.md`. Create `.claude/archive/` if absent.

## Recording lessons

When a command fails twice for the same avoidable reason, or the user corrects
your behavior, append to `.claude/rules/lessons.md`:

```
## YYYY-MM-DD — <short title>
- Trigger: <what situation produced the mistake>
- Wrong: <what was done>
- Right: <what to do instead, concretely>
- Promoted: <rule file updated with user approval, or "not yet">
```

Also add durable user preferences to memory (memory directory + MEMORY.md
pointer) so they survive outside this repo checkout.

## Compaction thresholds

- `lessons.md` > 120 lines → propose (to the user) promoting recurring lessons
  into the matching rules file and pruning one-offs.
- `MEMORY.md` > 30 entries → run the memory consolidation pass
  (`consolidate-memory` skill if available).
- `CLAUDE.md` must stay ≤ 150 lines and index-like; if a section grows into
  content, move it to a `.claude/rules/` file and leave a route.

## Re-verification of volatile facts

`model-dispatch.md` hardcodes model names/IDs verified on 2026-07-03. If more
than ~6 months have passed, re-verify via a `claude-code-guide` subagent before
trusting them, then update the file (with user approval + backup).

## Decision log (resolved user decisions)

All founding-session pending items were resolved with user approval on
2026-07-03:
- Global `~/.claude/settings.json` `effortLevel` changed `"max"` → `"xhigh"`
  (documented valid value; backup kept next to the file).
- `.gitignore` narrowed: the blanket `.claude` entry became
  `.claude/settings.local.json`, and the institution files were committed.
- `README.md` Testing section corrected: test_pipeline runs 5 regression
  tests (was "4").

Rule going forward: institution files must NOT contain machine-specific
absolute paths (they are now versioned and shared) — write repo-relative
paths; refer to locations outside the repo descriptively (e.g. "the global
`~/.claude/settings.json`"), never as hardcoded local paths.
