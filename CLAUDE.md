@AGENTS.md

# Claude Code

AGENTS.md is the project constitution. This file adds only
Claude-specific operating instructions. If they conflict, AGENTS.md wins.

## Session start

- Run the inspect-before-edit steps in AGENTS.md.
- Read `docs/PROJECT_STATE.md`, then verify it against Git.
- Treat Claude memory as the lowest-priority source of truth.

## Modes and agents

- Use Plan mode for cross-cutting or architectural changes, and for
  anything touching point-in-time semantics or data contracts.
- Use subagents when they add genuine domain expertise or protect
  context (broad searches). Use agent teams only for work that can
  proceed meaningfully in parallel.
- Superpowers (planning, TDD, debugging, review) and Graphify
  (architecture / cross-file questions) are optional. Use them only
  when installed. Verify Graphify conclusions against source and tests.

## Workflow commands

- Invoke `/project-verify` yourself at the end of every implementation
  task, before committing. Do not wait for the owner to ask. Act on its
  verdict as AGENTS.md "Autonomy" describes.
- Run gates only through `python3 scripts/verify.py`, never bare
  `pytest` or `ruff`. Focused tests use `.venv/bin/python -m pytest -q`.
- Invoke `/project-checkpoint` at milestones only, and commit the
  checkpoint in the milestone's own branch.
- When `gh` is unavailable, push the branch and give the owner the
  compare URL instead of a PR link.

## Permissions and guardrails

- `.claude/settings.json` allows routine Git and gate commands; the
  `guard_commands.py` PreToolUse hook denies protected-branch writes,
  merges, and destructive commands, and asks at approval gates.
- Edits to guardrail files (`.claude/`, `AGENTS.md`, `CLAUDE.md`,
  `scripts/verify.py`, CI, dependency manifests, `.env*`) prompt the
  owner. Never route around a prompt or denial (for example by writing
  the file another way). Ask instead.

## Reporting

- Never claim work was committed, pushed, or passing unless verified
  in this session from Git or tool output.
- Report the commit hashes, the pushed branch, and the PR or compare URL.
