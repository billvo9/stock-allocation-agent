---
name: project-verify
description: End-of-implementation verification for this repository - Git state, focused and full tests, Ruff, diff review, scope/artifact/secret scan, and classified findings. Read-only. Never commits, pushes, merges, formats, or edits files.
disable-model-invocation: true
argument-hint: "[base-ref (default: origin/main)]"
allowed-tools: Read Grep Glob Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git rev-parse *) Bash(git branch --show-current) Bash(git merge-base *) Bash(git ls-files *) Bash(git grep *) Bash(pytest -q) Bash(pytest -q tests/*) Bash(python -m pytest -q) Bash(ruff format --check src tests scripts) Bash(ruff check src tests scripts) Bash(bash .claude/skills/project-verify/scripts/scan_changes.sh*)
---

# Project verification (read-only)

Verify the current work against the AGENTS.md quality gates and report
facts. Git and command output are the only sources of truth.

## Hard rules

- Do not edit, create, delete, format, stage, commit, push, merge, or
  fetch anything. This includes `ruff format` without `--check`,
  `ruff check --fix`, `git add`, `git commit`, `git push`, `git fetch`,
  `git stash`, `git reset`, `git checkout`, and any install command.
- If a gate fails, report it with its output. Never change code, tests, or
  configuration to make verification pass.
- Quote each command exactly as run, with its key result lines.
- The full gates in step 4 are authoritative. Focused tests are only an
  optimization and never replace them.

## Steps

1. **Git state**
   - `git branch --show-current`
   - `git rev-parse HEAD`
   - `git rev-parse --abbrev-ref --symbolic-full-name @{u}` (report
     "no upstream" if it fails)
   - `git status -sb`
   - `git log --oneline -5`

   Remote-tracking refs are as of the last fetch. Do not fetch.

2. **Changed files**
   - Base = `$ARGUMENTS` if given, else `origin/main`.
   - `git merge-base HEAD <base>`
   - Changed = `git diff --name-status <merge-base>` (committed plus
     working tree) and untracked files from `git status --short`.

3. **Focused tests (optional optimization)**
   - Include changed `tests/test_*.py` files directly.
   - For a changed `src/stock_agent/<path>.py`, find tests that import that
     exact module with `git grep -l "stock_agent.<dotted.module>" tests`.
   - If the mapping is uncertain or empty, skip focused tests and say so.
   - Run `pytest -q <files>` only for a clearly determined set.

4. **Full gates (authoritative)** - run all four, even if one fails:
   - `pytest -q`
   - `python -m pytest -q`
   - `ruff format --check src tests scripts`
   - `ruff check src tests scripts`

5. **Diff review**
   - `git diff --stat <merge-base>` and `git diff <merge-base>`.
   - Check for point-in-time leakage, unrelated changes, dead code, and
     removed or weakened tests (`-` lines in existing test files).
   - Flag approval-gated changes (never approve them yourself): data
     schemas or contracts, financial formulas, point-in-time or
     availability semantics, dependencies (`pyproject.toml`,
     `requirements*.txt`), CI (`.github/`), Claude configuration
     (`.claude/`), secrets or environment configuration.

6. **Scope / artifact / secret scan**
   - `bash .claude/skills/project-verify/scripts/scan_changes.sh <base>`
   - Treat every listed item as needing a decision. The scan is heuristic.

## Report

1. Git state (branch, HEAD, upstream, working tree, staged files).
2. Commands and results table: exact command, pass/fail, counts.
3. Findings grouped as **BLOCKER**, **SHOULD FIX**, **FOLLOW-UP**,
   **NO ISSUE**, each with file and evidence. Failing gates, leaked
   secrets, and committed generated data are BLOCKERs.
4. Approval-gated changes detected (listed, not approved).
5. One-line verdict. Never claim commit, push, or merge state that Git
   does not show.
