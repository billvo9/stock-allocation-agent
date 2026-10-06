"""
Tests for the PreToolUse(Bash) command guard (.claude/hooks/guard_commands.py).

The guard is the control that keeps autonomous commits and pushes off
protected branches, so every rule has a positive and a negative case. Unit
tests inject a fake Git state; integration tests run the hook as a subprocess
against throwaway local repositories (no network).
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / ".claude" / "hooks" / "guard_commands.py"


def _load_guard():
    name = "stock_agent_guard_commands_hook"
    spec = importlib.util.spec_from_file_location(name, GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()

REPO = "/work/repo"


class FakeGit(guard.GitState):
    """Branch state per directory; None means detached / not a repository."""

    def __init__(self, branches=None, upstreams=None):
        self.branches = branches if branches is not None else {REPO: "feature/x"}
        self.upstreams = upstreams or {}

    def current_branch(self, repo_dir):
        return self.branches.get(repo_dir)

    def implicit_push_targets(self, repo_dir):
        branch = self.current_branch(repo_dir)
        if branch is None:
            return None
        return {branch} | set(self.upstreams.get(repo_dir, ()))


ON_FEATURE = FakeGit()
ON_MAIN = FakeGit({REPO: "main"})
DETACHED = FakeGit({})
FEATURE_TRACKING_MAIN = FakeGit({REPO: "feature/x"}, {REPO: {"main"}})


def decide(command, git=ON_FEATURE, cwd=REPO):
    result = guard.classify(command, cwd=cwd, git=git)
    return None if result is None else result[0]


# --- autonomy that must stay allowed (no decision -> native rules apply) ---


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git add -A",
        "git add src/stock_agent/x.py tests/test_x.py",
        'git commit -m "feat: add x"',
        "git commit --amend --no-edit",
        "git push -u origin HEAD",
        "git push origin feature/x",
        "git push origin HEAD:feature/x",
        "git push",
        "git switch -c feature/new",
        "git checkout -b feature/new",
        "git restore --staged src/x.py",
        "git stash",
        "git reset HEAD~1",
        "git diff > /tmp/review.diff",
        "gh pr create --fill",
        "gh pr view 12",
        "python3 scripts/verify.py",
        ".venv/bin/ruff format src tests scripts .claude/hooks",
        "cat AGENTS.md",
        "cp AGENTS.md /tmp/agents-copy.md",
        "ls 2>&1",
        "rm -r build/tmp",
    ],
)
def test_routine_autonomous_commands_are_not_blocked(command):
    assert decide(command) is None


def test_explicit_push_of_new_branch_allowed_even_if_upstream_is_main():
    # `git switch -c x origin/main` sets upstream to main; an explicit HEAD
    # refspec still pushes to the same-named branch, which is safe.
    assert decide("git push -u origin HEAD", git=FEATURE_TRACKING_MAIN) is None


def test_ff_only_pull_on_main_is_allowed():
    assert decide("git pull --ff-only", git=ON_MAIN) is None


# --- hard protections: deny ---


@pytest.mark.parametrize(
    "command",
    [
        "git push --force",
        "git push -f origin feature/x",
        "git push --force-with-lease origin feature/x",
        "git push origin +feature/x",
        "git push --mirror",
        "git push --all origin",
        "git push origin main",
        "git push origin HEAD:main",
        "git push origin feature/x:refs/heads/main",
        "git push origin :main",
        "git push --delete origin main",
        "git push origin master",
        "git reset --hard",
        "git reset --hard origin/main",
        "git clean -fdx",
        "git clean --force",
        "git branch -D feature/x",
        "git branch --delete --force feature/x",
        "git branch -f main HEAD~1",
        "rm -rf build",
        "rm -fr /tmp/x",
        "sudo rm -r -f data",
        "gh pr merge 12 --squash",
        "gh pr merge --auto 12",
        "gh api -X PUT repos/o/r/pulls/12/merge",
        "gh api repos/o/r/merges -f base=main -f head=x",
        'bash -c "git push origin main"',
        "FOO=1 git push origin main",
        "git -C /elsewhere push origin main",
    ],
)
def test_hard_protections_are_denied(command):
    assert decide(command) == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git push",
        "git push origin",
        "git push -u origin HEAD",
        "git push origin @",
        'git commit -m "x"',
        "git commit --amend",
        "git cherry-pick abc123",
        "git revert abc123",
        "git am patch.mbox",
        "git merge feature/x",
    ],
)
def test_writes_while_on_main_are_denied(command):
    assert decide(command, git=ON_MAIN) == "deny"


def test_refspecless_push_tracking_main_is_denied():
    assert decide("git push", git=FEATURE_TRACKING_MAIN) == "deny"


def test_cd_into_repo_on_main_then_commit_is_denied():
    git = FakeGit({"/work/other": "main", REPO: "feature/x"})
    assert decide("cd ../other && git commit -m x", git=git) == "deny"
    assert decide("cd /work/repo && git commit -m x", git=git) is None


def test_git_dash_c_uses_that_repository():
    git = FakeGit({"/work/other": "main", REPO: "feature/x"})
    assert decide("git -C ../other commit -m x", git=git) == "deny"
    assert decide("git -C /work/repo commit -m x", git=git) is None


def test_strictest_decision_wins_across_segments():
    assert decide("git add -A && git commit -m x && git push origin main") == "deny"


# --- consequential operations: ask ---


@pytest.mark.parametrize(
    "command",
    [
        "git rebase main",
        "git rebase -i HEAD~3",
        "git merge origin/main",
        "git restore src/x.py",
        "git restore -SW src/x.py",
        "git checkout -- .",
        "git checkout -f feature/y",
        "git switch --discard-changes feature/y",
        "git stash drop",
        "git stash clear",
        "git branch -d feature/old",
        "git branch -m feature/old feature/new",
        "git branch -f feature/x HEAD~1",
        "git push --delete origin feature/old",
        "git push origin :feature/old",
        "git push --tags",
        "git push origin 'refs/heads/*:refs/heads/*'",
        "git filter-branch --tree-filter x",
        "git update-ref refs/heads/x HEAD",
        "pip install pandas",
        "python3 -m pip install -r requirements.txt",
        "uv add numpy",
    ],
)
def test_consequential_operations_ask(command):
    assert decide(command) == "ask"


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "x"',
        "git push",
        "git push origin HEAD",
        "git merge feature/x",
    ],
)
def test_unknown_branch_asks(command):
    assert decide(command, git=DETACHED) == "ask"


@pytest.mark.parametrize(
    "command",
    [
        "GIT_DIR=/tmp/x.git git commit -m x",
        "git --git-dir=/tmp/x.git commit -m x",
        "git --work-tree /tmp/x push",
    ],
)
def test_overridden_repository_asks(command):
    assert decide(command) == "ask"


def test_non_ff_pull_on_main_asks():
    assert decide("git pull", git=ON_MAIN) == "ask"
    assert decide("git pull") is None


# --- protected files: shell writes ask, reads do not ---


@pytest.mark.parametrize(
    "command",
    [
        "echo x > AGENTS.md",
        "echo x >> CLAUDE.md",
        "printf x >| ./pyproject.toml",
        "cat new.json > .claude/settings.json",
        "echo y | tee requirements.txt",
        "echo y | tee -a requirements-dev.txt",
        "sed -i '' s/a/b/ pyproject.toml",
        "sed --in-place s/a/b/ .github/workflows/ci.yml",
        "perl -pi -e s/a/b/ .claude/hooks/guard_commands.py",
        "cp /tmp/x scripts/verify.py",
        "mv .env .env.bak",
        "rm .claude/settings.json",
        "truncate -s0 .env.local",
        "dd if=/dev/null of=AGENTS.md",
        "echo x > /Users/me/repo/.claude/hooks/guard_commands.py",
    ],
)
def test_shell_writes_to_protected_files_ask(command):
    assert decide(command) == "ask"


@pytest.mark.parametrize(
    "path, protected",
    [
        ("pyproject.toml", True),
        ("./requirements.txt", True),
        ("requirements-dev.txt", True),
        (".env", True),
        (".env.production", True),
        (".github/workflows/ci.yml", True),
        (".claude/agents/architect.md", True),
        ("/abs/repo/.claude/settings.json", True),
        ("AGENTS.md", True),
        ("scripts/verify.py", True),
        ("scripts/run_baseline.py", False),
        ("src/stock_agent/config.py", False),
        ("docs/PROJECT_STATE.md", False),
        ("environment.yml", False),
    ],
)
def test_is_protected_path(path, protected):
    assert guard.is_protected_path(path) is protected


# --- hook entry point ---


def _run_main(monkeypatch, payload):
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    assert guard.main() == 0
    return stdout.getvalue()


def test_main_emits_pretooluse_deny(monkeypatch):
    monkeypatch.setattr(guard, "GitState", lambda: ON_FEATURE)
    output = _run_main(
        monkeypatch,
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 1"}, "cwd": REPO},
    )
    decision = json.loads(output)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert "gh pr merge" in decision["permissionDecisionReason"]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        {"tool_name": "Edit", "tool_input": {"file_path": "AGENTS.md"}},
        {"tool_name": "Bash", "tool_input": {"command": "   "}},
        {"tool_name": "Bash", "tool_input": {"command": "git status"}, "cwd": REPO},
    ],
)
def test_main_is_silent_without_a_decision(monkeypatch, payload):
    monkeypatch.setattr(guard, "GitState", lambda: ON_FEATURE)
    assert _run_main(monkeypatch, payload) == ""


def test_main_fails_closed_on_internal_error(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(guard, "classify", broken)
    output = _run_main(
        monkeypatch,
        {"tool_name": "Bash", "tool_input": {"command": "git push"}, "cwd": REPO},
    )
    assert json.loads(output)["hookSpecificOutput"]["permissionDecision"] == "ask"


# --- integration: real hook process, real local repositories ---


def _hermetic_env():
    env = dict(os.environ)
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="guard-test",
        GIT_AUTHOR_EMAIL="guard-test",
        GIT_COMMITTER_NAME="guard-test",
        GIT_COMMITTER_EMAIL="guard-test",
    )
    return env


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, env=_hermetic_env(), check=True, capture_output=True)


def _hook(command, cwd):
    completed = subprocess.run(
        [sys.executable, str(GUARD_PATH)],
        input=json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}
        ),
        capture_output=True,
        text=True,
        env=_hermetic_env(),
        check=True,
    )
    if not completed.stdout:
        return None
    return json.loads(completed.stdout)["hookSpecificOutput"]["permissionDecision"]


def test_hook_process_reads_the_real_checked_out_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    assert _hook("git commit -m x", repo) == "deny"
    assert _hook("git push -u origin HEAD", repo) == "deny"

    _git(repo, "symbolic-ref", "HEAD", "refs/heads/feature/x")

    assert _hook("git commit -m x", repo) is None
    assert _hook("git push -u origin HEAD", repo) is None
    assert _hook("git push origin main", repo) == "deny"


def test_hook_process_denies_refspecless_push_to_upstream_main(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "main")
    _git(repo, "switch", "-q", "-c", "feature/tracks-main", "--track", "origin/main")

    assert _hook("git push", repo) == "deny"
    assert _hook("git push -u origin HEAD", repo) is None
