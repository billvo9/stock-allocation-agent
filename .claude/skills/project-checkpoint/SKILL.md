---
name: project-checkpoint
description: Update docs/PROJECT_STATE.md at a milestone (not after every ticket) from verified Git state and this session's /project-verify results, committed in the same branch as the milestone work. Use when a milestone trigger applies or the owner asks for a checkpoint.
argument-hint: "[short note on what this milestone covers]"
allowed-tools: Read Grep Glob Edit Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git rev-parse *) Bash(git branch --show-current) Bash(git merge-base *) Bash(git rev-list *)
---

# Milestone checkpoint

Bring `docs/PROJECT_STATE.md` up to date at a milestone. Context for this
checkpoint: $ARGUMENTS

## When (milestones only)

Checkpoint when the current branch:

- changes a data contract, point-in-time / availability semantics, a
  financial formula, or the architecture;
- completes a multi-ticket objective, or changes the current objective or
  next steps;
- changes the quality baseline materially (gate set, test count by more
  than routine additions, environment), or resolves / adds an open item;
- or the owner asks.

Ordinary tickets do not checkpoint: their pull request description is the
record. Never open a docs-only pull request just to checkpoint; the
checkpoint is a commit in the milestone's own branch.

## Hard rules

- Git is authoritative. Record only facts verified by commands run in this
  session, and date them. When `docs/PROJECT_STATE.md` disagrees with Git,
  Git wins: list each discrepancy in your report.
- Record milestone state, not live Git state. Do not record the current
  branch, working tree, or upstream as "current" facts: they go stale at
  merge and the SessionStart hook reports them live. Record the commit the
  checkpoint was verified on and, for the branch being checkpointed, say
  "merge pending"; the next checkpoint records the merge from Git.
- Never claim a commit, push, merge, or PR that Git does not show.
- Test and Ruff results come only from `python3 scripts/verify.py` run in
  this session. Otherwise write "not re-verified"; never carry old
  numbers forward as current.
- Owner decisions (approvals, rejected options) are recorded only when the
  owner stated them; quote or paraphrase with the date. Never infer one.
- `docs/PROJECT_STATE.md` holds changing state only. Policy belongs in
  `AGENTS.md`; propose policy changes separately (owner approval gate).

## Steps

1. **Verify Git**: `git branch --show-current`, `git rev-parse HEAD`,
   `git status --short`, `git log --oneline -10`,
   `git log --oneline -5 origin/main`, `git merge-base HEAD origin/main`.
2. **Read** `docs/PROJECT_STATE.md`; list stale or contradicted statements
   with the Git evidence.
3. **Evidence**: use this session's `/project-verify` result. If there is
   none, run `/project-verify` first.
4. **Edit** `docs/PROJECT_STATE.md` in place, keeping its structure:
   last checkpoint (date, verified commit, milestone), recently merged
   work, current objective, key semantics in effect (brief; point to code
   or AGENTS.md), last verified quality baseline (date, command, counts),
   open items (short, prioritized), next steps (approval-gated actions
   marked as such).
5. **Report** `git diff docs/PROJECT_STATE.md` with a short summary of
   what changed and why. The owner reviews it in the pull request.
6. Commit it with the milestone work (`docs: checkpoint ...`) under the
   normal autonomy rules. Do not commit if `/project-verify` was not
   `READY`.
