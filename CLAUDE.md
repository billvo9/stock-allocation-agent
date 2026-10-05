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

## Reporting

- Never claim work was committed, pushed, or passing unless verified
  in this session from Git or tool output.
- At the end of a substantial session, propose an update to
  `docs/PROJECT_STATE.md`; write it only with owner approval.
