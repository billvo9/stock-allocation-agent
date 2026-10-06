---
name: project-verify
description: Run at the end of every implementation task in this repository, before committing - canonical quality gates via scripts/verify.py, diff review for leakage and approval-gated changes, scope/artifact/secret scan, and a commit-gate verdict. Read-only. Use it yourself without waiting to be asked; also when the owner asks to verify.
argument-hint: "[base-ref (default: origin/main)]"
allowed-tools: Read Grep Glob Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git rev-parse *) Bash(git branch --show-current) Bash(git merge-base *) Bash(git ls-files *) Bash(git grep *) Bash(python3 scripts/verify.py) Bash(.venv/bin/python -m pytest -q *) Bash(bash .claude/skills/project-verify/scripts/scan_changes.sh*)
---

# Project verification (read-only)

Verify the current work against the AGENTS.md quality gates and decide
whether it may be committed autonomously. Git and command output are the
only sources of truth.

## Hard rules

- Do not edit, create, delete, format, stage, commit, push, merge, or
  fetch anything during verification. Committing happens afterwards, per
  the verdict below.
- If a gate fails, report it with its output. Never change code, tests, or
  configuration to make verification pass, and never weaken a test.
- Run gates only through `python3 scripts/verify.py`. Never run bare
  `pytest` or `ruff`: they may resolve to a non-project interpreter (this
  happened with Anaconda). Focused tests use `.venv/bin/python -m pytest`.
- Run the gate script directly. Never pipe it through `tail`, `grep`,
  `head`, `tee`, or anything else; its exit status is the result
  (0 pass, 1 gate failed, 2 environment unusable).
- Quote each command exactly as run, with its key result lines.

## Steps

1. **Git state**
   - `git branch --show-current`, `git rev-parse HEAD`
   - `git rev-parse --abbrev-ref --symbolic-full-name @{u}` (report
     "no upstream" if it fails)
   - `git status -sb`, `git log --oneline -5`

   Remote-tracking refs are as of the last fetch. Do not fetch.

2. **Changed files**
   - Base = `$ARGUMENTS` if given, else `origin/main`.
   - `git merge-base HEAD <base>`
   - Changed = `git diff --name-status <merge-base>` (committed plus
     working tree) and untracked files from `git status --short`.

3. **Focused tests (optional, fast feedback only)**
   - Changed `tests/test_*.py` files, plus tests importing a changed
     module (`git grep -l "stock_agent.<dotted.module>" tests`).
   - `.venv/bin/python -m pytest -q <files>` for a clearly determined set;
     otherwise skip and say so. Never a substitute for step 4.

4. **Full gates (authoritative)**: `python3 scripts/verify.py`

5. **Diff review**
   - `git diff --stat <merge-base>` and `git diff <merge-base>`.
   - Point-in-time leakage: availability timestamps, as-of joins,
     feature/label construction, train/test boundaries.
   - Unrelated changes, dead code, debug output.
   - Removed or weakened tests (`-` lines in existing test files).
   - Approval-gated changes (AGENTS.md "Owner approval gates"). The scan in
     step 6 lists review-required paths mechanically; judge each:
     dependencies, breaking schema / data-contract changes, point-in-time
     or availability semantics, financial or accounting formulas,
     secrets / auth, cloud / IAM / deployment / CI, Claude guardrails
     (`.claude/`, `scripts/verify.py`, `AGENTS.md`, `CLAUDE.md`), deleted
     data, major restructuring. A gated change is cleared only by the
     owner's explicit approval in this session; quote it.

6. **Scope / artifact / secret scan**
   - `bash .claude/skills/project-verify/scripts/scan_changes.sh <base>`
   - Every listed item needs a decision. The scan is heuristic.

## Report

1. Git state (branch, HEAD, upstream, working tree, staged files).
2. Commands and results table: exact command, pass/fail, counts.
3. Findings grouped as **BLOCKER**, **SHOULD FIX**, **FOLLOW-UP**,
   **NO ISSUE**, each with file and evidence. Failing gates, leaked
   secrets, committed generated data, and leakage are BLOCKERs.
4. Approval-gated changes: each listed with "approved (quote)" or
   "NOT approved".
5. **Commit gate verdict** (exactly one):
   - `READY` - all gates pass, no BLOCKER, every gated change approved.
     The lead may stage the task's files, commit, push the branch, and
     open a PR (AGENTS.md "Autonomy"). Never merge.
   - `NEEDS OWNER` - an unapproved gated change; stop and ask before
     committing.
   - `BLOCKED` - failing gate or BLOCKER; fix and re-verify.

Never claim commit, push, or merge state that Git does not show.
