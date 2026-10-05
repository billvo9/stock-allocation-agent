---
name: project-checkpoint
description: Draft an update to docs/PROJECT_STATE.md from verified Git state and verification results, show the diff, and write it only after explicit owner approval.
disable-model-invocation: true
argument-hint: "[short note on what this checkpoint covers]"
allowed-tools: Read Grep Glob Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git rev-parse *) Bash(git branch --show-current) Bash(git merge-base *) Bash(git rev-list *) Bash(diff -u docs/PROJECT_STATE.md *) Bash(cmp docs/PROJECT_STATE.md *)
---

# Project-state checkpoint

Update `docs/PROJECT_STATE.md` so it matches reality. Context for this
checkpoint: $ARGUMENTS

## Hard rules

- Git is authoritative. Record only facts verified by commands run in this
  session, and date them. When `docs/PROJECT_STATE.md` disagrees with Git,
  Git wins: list each discrepancy explicitly.
- Never claim a commit, push, merge, or PR that Git does not show.
  "Pushed" requires `HEAD == @{u}`; say "as of last fetch". Do not fetch.
- Test and Ruff results come only from commands run in this session (or a
  `/project-verify` report from this session). Otherwise write
  "not re-verified" - never carry old numbers forward as current.
- `docs/PROJECT_STATE.md` holds changing state only. Permanent policy
  belongs in `AGENTS.md`: do not copy rules into the state file. If a
  policy change seems needed, propose it separately for owner approval.
- Do not write `docs/PROJECT_STATE.md` until the owner explicitly approves
  the exact diff shown. Do not stage or commit it.

## Steps

1. **Verify Git**
   - `git branch --show-current`, `git rev-parse HEAD`, HEAD subject
   - upstream: `git rev-parse --abbrev-ref --symbolic-full-name @{u}`;
     compare `git rev-parse HEAD` with `git rev-parse @{u}`
   - `git status --short`, staged files (`git diff --cached --name-only`)
   - `git log --oneline -10`, `git log --oneline -5 origin/main`
   - `git merge-base HEAD origin/main`
2. **Read** `docs/PROJECT_STATE.md` and list every stale or contradicted
   statement with the Git evidence.
3. **Verification evidence**: use gate results from this session. If there
   are none and the owner wants current numbers, run `/project-verify`
   first (or ask).
4. **Draft** the complete new file outside the repository (the system temp
   or scratch directory, never inside the repo). Keep the existing
   structure where it still fits:
   - current branch; last verified pushed commit; working tree
   - current objective / ticket
   - completed work (brief)
   - key semantics in effect (brief; point to code or AGENTS.md)
   - last verified quality baseline (with date and commands)
   - open items (short and prioritized, not a backlog dump)
   - next steps (approval-gated actions marked as such)
5. **Show** `diff -u docs/PROJECT_STATE.md <draft>` and a short summary of
   what changed and why. Stop and wait for approval.
6. **Only after explicit approval**: write the file exactly as shown,
   confirm with `cmp docs/PROJECT_STATE.md <draft>`, and show
   `git status --short`. Do not stage or commit.
