#!/usr/bin/env python3
"""
PreToolUse(Bash) hook: second safety net behind native permission rules.

Native `permissions.deny` / `permissions.ask` in .claude/settings.json are
the primary control. They match command prefixes, so they can miss forms
such as `git -C dir push`, `cd repo && git push`, `FOO=1 git commit`, or
commands wrapped in `bash -c "..."`. This hook inspects the full command
text and mirrors the same policy:

    deny  highly destructive operations (force push, reset --hard,
          git clean -f, branch -D, rm -rf)
    ask   consequential operations (commit, push, merge, rebase,
          discarding restore/checkout, stash deletion, dependency
          changes, PR creation/merge)

It never returns "allow", so it cannot loosen permissions. Unknown or
unparseable input exits 0 with no output (native rules still apply).
This is defense in depth, not a security boundary: deliberately
obfuscated commands can evade text inspection.

Standard library only. Never executes or evaluates the command.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

# Characters that separate commands in a shell line.
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n]|\$\(|`|\)")

# git global options that take a separate argument.
_GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}

_WRAPPERS = {"sudo", "command", "builtin", "exec", "time", "nohup", "env", "xargs"}


def _segments(command: str) -> list[str]:
    return [part.strip() for part in _SEPARATORS.split(command) if part.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _strip_prefixes(tokens: list[str]) -> list[str]:
    """Drop env assignments and simple wrappers (sudo, env, time, ...)."""

    index = 0

    while index < len(tokens):
        token = tokens[index]

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            index += 1
            continue

        if os.path.basename(token) in _WRAPPERS:
            index += 1
            # Skip wrapper flags such as `env -i` or `sudo -u user`.
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue

        break

    return tokens[index:]


def _short_flags(args: list[str]) -> set[str]:
    """Letters from short-option clusters, e.g. -fdx -> {f, d, x}."""

    letters: set[str] = set()

    for arg in args:
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            letters.update(arg[1:])

    return letters


def _classify_git(args: list[str]) -> tuple[str, str] | None:
    index = 0

    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        index += 2 if option in _GIT_OPTS_WITH_ARG else 1

    if index >= len(args):
        return None

    sub = args[index]
    rest = args[index + 1 :]
    flags = _short_flags(rest)

    if sub == "push":
        if any(a.startswith("--force") for a in rest) or "f" in flags:
            return ("deny", "git push --force / --force-with-lease")
        if any(a.startswith("+") for a in rest):
            return ("deny", "git push with a forced (+) refspec")
        if "--mirror" in rest or "--delete" in rest or "d" in flags:
            return ("ask", "git push --mirror/--delete")
        return ("ask", "git push")

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
        return None

    if sub in {"commit", "merge", "rebase", "restore"}:
        return ("ask", f"git {sub}")

    if sub == "checkout":
        if "--" in rest or "." in rest or "f" in flags or "--force" in rest:
            return ("ask", "git checkout that may discard changes")
        return None

    if sub == "switch" and ("--discard-changes" in rest or "f" in flags or "--force" in rest):
        return ("ask", "git switch that discards changes")

    if sub == "stash" and rest[:1] and rest[0] in {"drop", "clear"}:
        return ("ask", f"git stash {rest[0]}")

    if sub in {"filter-branch", "filter-repo"}:
        return ("ask", f"git {sub}")

    return None


def _classify_tokens(tokens: list[str], depth: int) -> tuple[str, str] | None:
    tokens = _strip_prefixes(tokens)

    if not tokens:
        return None

    program = os.path.basename(tokens[0])
    args = tokens[1:]

    # Nested shells: inspect the -c script too.
    if program in {"bash", "sh", "zsh", "dash"} and "-c" in args and depth < 3:
        script_index = args.index("-c") + 1
        if script_index < len(args):
            return classify(args[script_index], depth + 1)

    if program == "git":
        return _classify_git(args)

    if program == "rm":
        flags = _short_flags(args)
        recursive = bool({"r", "R"} & flags) or "--recursive" in args
        force = "f" in flags or "--force" in args
        if recursive and force:
            return ("deny", "rm -rf")
        return None

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

    if program == "gh" and args[:1] == ["pr"] and len(args) > 1 and args[1] in {"create", "merge"}:
        return ("ask", f"gh pr {args[1]}")

    return None


def classify(command: str, depth: int = 0) -> tuple[str, str] | None:
    """Return the strictest (decision, reason) found in the command, if any."""

    found: list[tuple[str, str]] = []

    for segment in _segments(command):
        result = _classify_tokens(_tokens(segment), depth)
        if result is not None:
            found.append(result)

    for decision in ("deny", "ask"):
        for result in found:
            if result[0] == decision:
                return result

    return None


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

    result = classify(command)

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
                    "See AGENTS.md owner approval gates."
                ),
            }
        },
        sys.stdout,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
