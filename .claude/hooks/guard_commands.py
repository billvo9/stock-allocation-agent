#!/usr/bin/env python3
"""
PreToolUse(Bash) hook: the guarded-autonomy Git and shell policy.

Claude may stage, commit, and push non-main branches and open pull requests
without asking (AGENTS.md "Autonomy and approval gates"). Native rules in
.claude/settings.json allow those commands; this hook keeps them off protected
branches, because prefix rules cannot see which branch a bare `git commit` or
`git push` targets. It also catches forms prefix rules miss: `git -C dir`,
`cd repo && ...`, env-prefixed and `bash -c` wrapped commands.

    deny  force push (incl. `+` refspecs), push --mirror / --all, any push
          whose destination is a protected branch, commit / cherry-pick /
          revert / am / merge while a protected branch is checked out,
          force-moving a protected branch, merging pull requests,
          reset --hard, git clean -f, branch -D, rm -rf
    ask   rebase, history filters, update-ref, discarding restore / checkout /
          switch, stash deletion, branch deletion / rename / force-move,
          push --delete / --tags / wildcard refspecs, merge, non-ff pull on a
          protected branch, Git commands whose repository or branch cannot be
          determined, dependency installs, and shell writes to protected files
          (dependency manifests, .env, CI, Claude configuration, AGENTS.md,
          CLAUDE.md, scripts/verify.py)

It never returns "allow", so it cannot loosen permissions. It reads Git state
(current branch, upstream) but never executes the inspected command. An
internal error fails closed to "ask". Unparseable hook input exits 0 with no
output (native rules still apply).

This is defense in depth, not a security boundary: deliberately obfuscated
commands (eval, scripting languages, variables) can evade text inspection.
Server-side branch protection on GitHub is the authoritative control.

Standard library only.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable

PROTECTED_BRANCHES = frozenset({"main", "master"})

# Repository-relative globs whose modification requires owner approval.
# Mirrors the Edit(...) ask rules in .claude/settings.json.
PROTECTED_PATHS = (
    "pyproject.toml",
    "requirements*.txt",
    "setup.py",
    "setup.cfg",
    ".env",
    ".env.*",
    ".github/*",
    ".claude/*",
    "AGENTS.md",
    "CLAUDE.md",
    "scripts/verify.py",
)

Decision = tuple[str, str]

# Characters that separate commands in a shell line.
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n]|\$\(|`|\)")

# Output redirections: `> f`, `>> f`, `>| f`, `&> f`; not `2>&1` / `>&2`.
_REDIRECTION = re.compile(r"(?:^|[^<>&0-9])(?:&?>{1,2}\|?)\s*([^\s;&|<>()]+)")

# git global options that take a separate argument.
_GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}

# git push options that take a separate argument.
_PUSH_OPTS_WITH_ARG = {"--repo", "-o", "--push-option", "--receive-pack", "--exec"}

_WRAPPERS = {"sudo", "command", "builtin", "exec", "time", "nohup", "env", "xargs"}

_GIT_REPO_ENV = ("GIT_DIR=", "GIT_WORK_TREE=")

_COMMITTING = {"commit", "cherry-pick", "revert", "am"}

# Programs that write the paths they are given. Value: which positional
# arguments are written ("all" or "last").
_WRITERS = {
    "tee": "all",
    "mv": "all",
    "rm": "all",
    "truncate": "all",
    "touch": "all",
    "chmod": "all",
    "chown": "all",
    "cp": "last",
    "ln": "last",
    "install": "last",
}


class GitState:
    """Reads branch information. Tests substitute a fake."""

    def _git(self, repo_dir: str, *args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", repo_dir, *args],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def current_branch(self, repo_dir: str) -> str | None:
        """Checked-out branch name; None if detached or not a repository."""

        return self._git(repo_dir, "symbolic-ref", "--quiet", "--short", "HEAD") or None

    def implicit_push_targets(self, repo_dir: str) -> set[str] | None:
        """
        Remote branches a refspec-less `git push` may update.

        Conservative union of the current branch name, its upstream branch,
        and its push destination. None if no branch is checked out.
        """

        branch = self.current_branch(repo_dir)
        if branch is None:
            return None

        targets = {branch}
        refs = self._git(
            repo_dir,
            "for-each-ref",
            "--format=%(upstream:remoteref)%00%(push:remoteref)",
            f"refs/heads/{branch}",
        )
        for ref in (refs or "").split("\0"):
            if ref.startswith("refs/heads/"):
                targets.add(ref.removeprefix("refs/heads/"))
        return targets


def _segments(command: str) -> list[str]:
    return [part.strip() for part in _SEPARATORS.split(command) if part.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _strip_prefixes(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Drop env assignments and simple wrappers; return (tokens, assignments)."""

    index = 0
    assignments: list[str] = []

    while index < len(tokens):
        token = tokens[index]

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            assignments.append(token)
            index += 1
            continue

        if os.path.basename(token) in _WRAPPERS:
            index += 1
            # Skip wrapper flags such as `env -i` or `sudo -u user`.
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue

        break

    return tokens[index:], assignments


def _short_flags(args: list[str]) -> set[str]:
    """Letters from short-option clusters, e.g. -fdx -> {f, d, x}."""

    letters: set[str] = set()

    for arg in args:
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            letters.update(arg[1:])

    return letters


def _resolve_dir(base: str, target: str) -> str:
    return os.path.normpath(os.path.join(base, os.path.expanduser(target)))


def _branch_name(ref: str) -> str:
    return ref.removeprefix("refs/heads/")


def is_protected_path(path: str) -> bool:
    """Whether a path (relative or absolute) names a protected file."""

    normalized = path.strip("'\"").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]

    return any(
        fnmatch.fnmatchcase(normalized, pattern) or fnmatch.fnmatchcase(normalized, "*/" + pattern)
        for pattern in PROTECTED_PATHS
    )


def _protected_write(program: str, args: list[str]) -> Decision | None:
    positional = [arg for arg in args if not arg.startswith("-")]

    sed_in_place = program == "sed" and any(a.startswith(("-i", "--in-place")) for a in args)
    perl_in_place = program == "perl" and "i" in _short_flags(args)

    if program in _WRITERS:
        written = positional[-1:] if _WRITERS[program] == "last" else positional
    elif sed_in_place or perl_in_place:
        written = positional
    elif program == "dd":
        written = [arg.removeprefix("of=") for arg in args if arg.startswith("of=")]
    else:
        return None

    for path in written:
        if is_protected_path(path):
            return ("ask", f"{program} modifies protected file {path}")

    return None


def _protected_redirections(command: str) -> list[Decision]:
    return [
        ("ask", f"shell redirection writes protected file {target}")
        for target in _REDIRECTION.findall(command)
        if is_protected_path(target)
    ]


def _classify_push(rest: list[str], repo_dir: str | None, git: GitState) -> Decision:
    flags = _short_flags(rest)

    if any(arg.startswith("--force") for arg in rest) or "f" in flags:
        return ("deny", "git push --force / --force-with-lease")
    if "--mirror" in rest:
        return ("deny", "git push --mirror can overwrite or delete protected branches")
    if "--all" in rest or "--branches" in rest:
        return ("deny", "git push --all includes protected branches")

    positional: list[str] = []
    index = 0
    while index < len(rest):
        arg = rest[index]
        if arg in _PUSH_OPTS_WITH_ARG:
            index += 2
            continue
        if not arg.startswith("-"):
            positional.append(arg)
        index += 1

    refspecs = positional[1:]
    deleting = "--delete" in rest or "d" in flags

    if any(spec.startswith("+") for spec in refspecs):
        return ("deny", "git push with a forced (+) refspec")

    if repo_dir is None:
        return ("ask", "git push in a repository the guard cannot determine")

    destinations: set[str] = set()

    if not refspecs:
        implicit = git.implicit_push_targets(repo_dir)
        if implicit is None:
            return ("ask", "git push without a checked-out branch")
        destinations |= implicit

    for spec in refspecs:
        if "*" in spec:
            return ("ask", f"git push with wildcard refspec {spec}")
        source, colon, destination = spec.partition(":")
        if not colon:
            destination = source
        elif not source:
            deleting = True
        elif not destination:
            destination = source
        if destination in {"HEAD", "@"}:
            current = git.current_branch(repo_dir)
            if current is None:
                return ("ask", "git push of HEAD without a checked-out branch")
            destination = current
        destinations.add(_branch_name(destination))

    protected = sorted(destinations & PROTECTED_BRANCHES)
    if protected:
        return ("deny", f"git push to protected branch {', '.join(protected)}")
    if deleting:
        return ("ask", "git push deleting a remote branch")
    if "--tags" in rest:
        return ("ask", "git push --tags publishes tags")

    return ("ok", "")


def _classify_git(
    args: list[str], cwd: str, git: GitState, repo_env_override: bool
) -> Decision | None:
    index = 0
    repo_dir: str | None = cwd

    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option == "-C" and index + 1 < len(args):
            repo_dir = _resolve_dir(repo_dir or cwd, args[index + 1])
        elif option in {"--git-dir", "--work-tree"} or option.startswith(
            ("--git-dir=", "--work-tree=")
        ):
            repo_dir = None
        index += 2 if option in _GIT_OPTS_WITH_ARG else 1

    if repo_env_override:
        repo_dir = None

    if index >= len(args):
        return None

    sub = args[index]
    rest = args[index + 1 :]
    flags = _short_flags(rest)

    if sub == "push":
        decision = _classify_push(rest, repo_dir, git)
        return None if decision[0] == "ok" else decision

    if sub in _COMMITTING or sub == "merge":
        if repo_dir is None:
            return ("ask", f"git {sub} in a repository the guard cannot determine")
        branch = git.current_branch(repo_dir)
        if branch is None:
            return ("ask", f"git {sub} without a checked-out branch")
        if branch in PROTECTED_BRANCHES:
            return ("deny", f"git {sub} on protected branch {branch}")
        return ("ask", "git merge") if sub == "merge" else None

    if sub == "pull":
        if "--ff-only" in rest or repo_dir is None:
            return None if "--ff-only" in rest else ("ask", "git pull in an unknown repository")
        branch = git.current_branch(repo_dir)
        if branch in PROTECTED_BRANCHES:
            return ("ask", f"git pull without --ff-only on protected branch {branch}")
        return None

    if sub == "reset":
        if "--hard" in rest:
            return ("deny", "git reset --hard")
        return None

    if sub == "clean":
        if "f" in flags or "--force" in rest:
            return ("deny", "git clean -f")
        return None

    if sub == "branch":
        if "D" in flags or ("--delete" in rest and "--force" in rest):
            return ("deny", "git branch -D")
        if "d" in flags or "--delete" in rest:
            return ("ask", "git branch -d")
        if {"m", "M"} & flags or "--move" in rest:
            return ("ask", "git branch rename")
        if "f" in flags or "--force" in rest:
            names = {_branch_name(a) for a in rest if not a.startswith("-")}
            if names & PROTECTED_BRANCHES:
                return ("deny", "git branch --force on a protected branch")
            return ("ask", "git branch --force")
        return None

    if sub in {"rebase", "filter-branch", "filter-repo", "update-ref"}:
        return ("ask", f"git {sub}")

    if sub == "restore":
        staged = "--staged" in rest or "S" in flags
        worktree = "--worktree" in rest or "W" in flags
        if staged and not worktree:
            return None
        return ("ask", "git restore discards working-tree changes")

    if sub == "checkout":
        if "--" in rest or "." in rest or "f" in flags or "--force" in rest:
            return ("ask", "git checkout that may discard changes")
        return None

    if sub == "switch" and ("--discard-changes" in rest or "f" in flags or "--force" in rest):
        return ("ask", "git switch that discards changes")

    if sub == "stash" and rest[:1] and rest[0] in {"drop", "clear"}:
        return ("ask", f"git stash {rest[0]}")

    return None


def _classify_gh(args: list[str]) -> Decision | None:
    if args[:2] == ["pr", "merge"]:
        return ("deny", "gh pr merge: merging pull requests is owner-only")
    if args[:1] == ["api"] and any(re.search(r"/merges?\b", arg) for arg in args[1:]):
        return ("deny", "gh api merge endpoint: merging is owner-only")
    return None


def _classify_tokens(
    tokens: list[str], cwd: str, git: GitState, depth: int
) -> Decision | list[Decision] | None:
    tokens, assignments = _strip_prefixes(tokens)

    if not tokens:
        return None

    program = os.path.basename(tokens[0])
    args = tokens[1:]

    # Nested shells: inspect the -c script too.
    if program in {"bash", "sh", "zsh", "dash"} and "-c" in args and depth < 3:
        script_index = args.index("-c") + 1
        if script_index < len(args):
            return classify_all(args[script_index], cwd=cwd, git=git, depth=depth + 1)

    if program == "git":
        repo_env_override = any(a.startswith(_GIT_REPO_ENV) for a in assignments)
        return _classify_git(args, cwd, git, repo_env_override)

    if program == "gh":
        return _classify_gh(args)

    if program == "rm":
        flags = _short_flags(args)
        recursive = bool({"r", "R"} & flags) or "--recursive" in args
        force = "f" in flags or "--force" in args
        if recursive and force:
            return ("deny", "rm -rf")

    if program in {"pip", "pip3"} and args[:1] and args[0] in {"install", "uninstall"}:
        return ("ask", f"{program} {args[0]}")

    if (
        re.match(r"^python(3(\.\d+)?)?$", program)
        and args[:2] == ["-m", "pip"]
        and len(args) > 2
        and args[2] in {"install", "uninstall"}
    ):
        return ("ask", f"python -m pip {args[2]}")

    if program == "uv" and args:
        if args[0] in {"add", "remove", "sync"}:
            return ("ask", f"uv {args[0]}")
        if args[0] == "pip" and len(args) > 1 and args[1] in {"install", "uninstall", "sync"}:
            return ("ask", f"uv pip {args[1]}")

    return _protected_write(program, args)


def classify_all(
    command: str, *, cwd: str, git: GitState | None = None, depth: int = 0
) -> list[Decision]:
    """Every (decision, reason) found in the command."""

    git = git or GitState()
    found: list[Decision] = []
    current_dir = cwd

    for segment in _segments(command):
        tokens = _tokens(segment)
        stripped, _ = _strip_prefixes(tokens)

        if stripped and stripped[0] in {"cd", "pushd"}:
            target = next((a for a in stripped[1:] if not a.startswith("-")), "~")
            current_dir = _resolve_dir(current_dir, target)
            continue

        result = _classify_tokens(tokens, current_dir, git, depth)
        if isinstance(result, list):
            found.extend(result)
        elif result is not None:
            found.append(result)

    found.extend(_protected_redirections(command))
    return found


def _strictest(found: Iterable[Decision]) -> Decision | None:
    found = list(found)
    for decision in ("deny", "ask"):
        for result in found:
            if result[0] == decision:
                return result
    return None


def classify(command: str, *, cwd: str, git: GitState | None = None) -> Decision | None:
    """Return the strictest (decision, reason) found in the command, if any."""

    return _strictest(classify_all(command, cwd=cwd, git=git))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    if not isinstance(payload, dict) or payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command")

    if not isinstance(command, str) or not command.strip():
        return 0

    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = os.getcwd()

    try:
        result = classify(command, cwd=cwd)
    # Fail closed: an unexpected guard error must never let the command through
    # unexamined, so any exception becomes "ask".
    except Exception as error:  # noqa: BLE001
        result = ("ask", f"command guard internal error ({type(error).__name__})")

    if result is None:
        return 0

    decision, reason = result
    verb = "Blocked" if decision == "deny" else "Owner approval required"

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": (
                    f"{verb} by project command guard: {reason}. "
                    "See AGENTS.md autonomy and approval gates."
                ),
            }
        },
        sys.stdout,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
