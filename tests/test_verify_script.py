"""
Tests for the canonical gate runner (scripts/verify.py) and the contract that
CI and Claude run the same gates under the same guardrails.

The runner is exercised with fake process runners and a fake virtual
environment, so these tests never recurse into the real gates.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    # Registered before execution so dataclasses can resolve the module.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = _load("stock_agent_verify_script", ROOT / "scripts" / "verify.py")


def _fake_venv(root: Path, tools=("python", "pytest", "ruff")) -> Path:
    bin_dir = root / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    for name in tools:
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    return bin_dir


class RecordingRunner:
    def __init__(self, exit_codes=None):
        self.exit_codes = dict(exit_codes or {})
        self.calls = []

    def __call__(self, argv, *, cwd, env):
        self.calls.append((list(argv), cwd, env))
        return self.exit_codes.get(tuple(argv[1:]), 0)

    def gate_calls(self):
        return [argv for argv, _, _ in self.calls if "--version" not in argv]


def _run(root, runner, version=(3, 12)):
    out = io.StringIO()
    code = verify.run(
        root=root,
        runner=runner,
        probe=lambda python: version,
        base_env={"PATH": "/opt/anaconda3/bin:/usr/bin", "PYTHONHOME": "/opt/anaconda3"},
        out=out,
    )
    return code, out.getvalue()


def test_gates_are_the_agents_md_gates_using_venv_tools(tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    gates = verify.build_gates(bin_dir)

    assert [gate.argv for gate in gates] == [
        (str(bin_dir / "pytest"), "-q"),
        (str(bin_dir / "python"), "-m", "pytest", "-q"),
        (str(bin_dir / "ruff"), "format", "--check", "src", "tests", "scripts", ".claude/hooks"),
        (str(bin_dir / "ruff"), "check", "src", "tests", "scripts", ".claude/hooks"),
    ]


def test_all_gates_pass(tmp_path):
    bin_dir = _fake_venv(tmp_path)
    runner = RecordingRunner()

    code, output = _run(tmp_path, runner)

    assert code == verify.EXIT_OK
    assert len(runner.gate_calls()) == 4
    assert all(Path(argv[0]).parent == bin_dir for argv in runner.gate_calls())
    assert all(cwd == tmp_path for _, cwd, _ in runner.calls)
    assert "all gates passed" in output


def test_every_gate_runs_even_after_a_failure(tmp_path):
    _fake_venv(tmp_path)
    runner = RecordingRunner(exit_codes={("-q",): 1})

    code, output = _run(tmp_path, runner)

    assert code == verify.EXIT_GATE_FAILED
    assert len(runner.gate_calls()) == 4
    assert "FAIL  pytest -q  (exit 1)" in output
    assert "PASS  python -m pytest -q" in output
    assert "FAILED" in output


@pytest.mark.parametrize("missing", ["python", "pytest", "ruff"])
def test_missing_venv_tool_refuses_without_running_anything(tmp_path, missing):
    _fake_venv(tmp_path, tools=[t for t in ("python", "pytest", "ruff") if t != missing])
    runner = RecordingRunner()

    code, output = _run(tmp_path, runner)

    assert code == verify.EXIT_BAD_ENVIRONMENT
    assert runner.calls == []
    assert missing in output
    assert "refusing to fall back" in output


def test_absent_venv_refuses(tmp_path):
    runner = RecordingRunner()

    code, _ = _run(tmp_path, runner)

    assert code == verify.EXIT_BAD_ENVIRONMENT
    assert runner.calls == []


def test_wrong_python_version_refuses(tmp_path):
    _fake_venv(tmp_path)
    runner = RecordingRunner()

    code, output = _run(tmp_path, runner, version=(3, 11))

    assert code == verify.EXIT_BAD_ENVIRONMENT
    assert runner.calls == []
    assert "requires 3.12" in output


def test_gate_environment_behaves_like_an_activated_venv(tmp_path):
    bin_dir = _fake_venv(tmp_path)
    runner = RecordingRunner()

    _run(tmp_path, runner)

    for _, _, env in runner.calls:
        assert env["VIRTUAL_ENV"] == str(bin_dir.parent)
        assert env["PATH"].split(":")[0] == str(bin_dir)
        assert "PYTHONHOME" not in env


def test_main_rejects_arguments():
    assert verify.main(["--fast"]) == verify.EXIT_BAD_ENVIRONMENT


def test_repo_root_is_the_repository():
    assert (verify.repo_root() / "pyproject.toml").is_file()
    assert (verify.repo_root() / "AGENTS.md").is_file()


# --- CI and Claude run the same gates under the same guardrails ---


def _ci_steps():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_ci_runs_the_canonical_gate_script():
    runs = [step.get("run", "") for step in _ci_steps()]

    assert any(run.strip() == "python scripts/verify.py" for run in runs)


def test_ci_does_not_run_gates_outside_the_canonical_script():
    # A separate `pytest` or `ruff` step could drift from the local gates.
    for step in _ci_steps():
        for line in step.get("run", "").splitlines():
            command = line.strip()
            assert not command.startswith(("pytest", "ruff", "python -m pytest")), command


def _requirement_pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        name, separator, pinned = line.strip().partition("==")
        if separator:
            pins[name.strip().lower()] = pinned.strip()
    return pins


@pytest.mark.parametrize("tool", ["pytest", "ruff"])
def test_gate_tools_are_pinned_and_installed_at_the_pin(tool):
    # CI installs exactly the pin; this fails if a local .venv drifts from it.
    pins = _requirement_pins()

    assert tool in pins, f"{tool} must be pinned with == in requirements.txt"
    assert installed_version(tool) == pins[tool]


def test_ci_installs_into_the_project_venv():
    runs = "\n".join(step.get("run", "") for step in _ci_steps())

    assert "python -m venv .venv" in runs
    assert ".venv/bin/python -m pip install -r requirements.txt" in runs


def _settings():
    return json.loads((ROOT / ".claude" / "settings.json").read_text())


@pytest.mark.parametrize(
    "rule",
    [
        "Bash(git push --force *)",
        "Bash(git push * --force *)",
        "Bash(git push -f *)",
        "Bash(git reset --hard *)",
        "Bash(git clean -f *)",
        "Bash(git branch -D *)",
        "Bash(rm -rf *)",
        "Bash(gh pr merge *)",
        "Bash(git push origin main)",
        "Bash(git push origin HEAD:main)",
    ],
)
def test_settings_keep_hard_protections_denied(rule):
    assert rule in _settings()["permissions"]["deny"]


@pytest.mark.parametrize(
    "rule",
    [
        "Edit(/pyproject.toml)",
        "Edit(/requirements*.txt)",
        "Edit(/.env)",
        "Edit(/.env.*)",
        "Edit(/.github/**)",
        "Edit(/.claude/**)",
        "Edit(/AGENTS.md)",
        "Edit(/CLAUDE.md)",
        "Edit(/scripts/verify.py)",
        "Bash(git rebase *)",
        "Bash(pip install *)",
        "Bash(uv add *)",
    ],
)
def test_settings_keep_approval_gates(rule):
    assert rule in _settings()["permissions"]["ask"]


def test_settings_never_allow_merging_or_force_pushing():
    allowed = " ".join(_settings()["permissions"].get("allow", []))

    assert "merge" not in allowed
    assert "--force" not in allowed
    assert "reset --hard" not in allowed


def test_command_guard_hook_is_registered_for_bash():
    hooks = _settings()["hooks"]["PreToolUse"]
    commands = [
        hook["command"]
        for entry in hooks
        if entry.get("matcher") == "Bash"
        for hook in entry["hooks"]
    ]

    assert any(".claude/hooks/guard_commands.py" in command for command in commands)
