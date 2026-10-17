"""
Set up the private macOS schedule for the Model Health local monitor
(docs/operations/runbook.md, "Activate the scheduler"): reviewable, and
installed only by the owner. It writes nothing without --write, never
overwrites a file that differs, and never runs launchctl to load, start or
remove a job.

Modes
    (default)  dry run: detect the repository, its venv, gh and the private
               inputs, then print every path, the schedule, each health option
               it enables or omits with the reason, and the exact run.sh and
               launchd plist it would write. Nothing is written.
    --write    the same, then write ~/stock-agent-private/health/run.sh (mode
               700) and ~/Library/LaunchAgents/local.stock-agent.health.plist
               (mode 644), each only when it is absent or already identical. A
               file that differs (or is a symlink) is never replaced: the run
               shows the difference and writes neither file. It refuses a
               linked Git worktree (temporary), a missing venv or one that is
               not Python 3.12. It prints the launchctl commands for the owner.
    --status   read-only: whether the job is loaded (`launchctl print`, the only
               launchctl call this script makes), its run count and last exit
               code with its meaning (EXIT_MEANINGS), when the last summary was
               written and whether it is fresh (at most model.STALE_AFTER, 26
               hours, old: the Lab page uses the same limit), stale or missing,
               and its overall state by condition.

What the job runs. launchd starts `/bin/zsh -f run.sh`: -f reads none of the
user's zsh startup files, so nothing exported in ~/.zshenv reaches the job.
run.sh refuses to run anything unless the checkout is a clean main (exit 3,
with a dated reason on stderr), then runs `scripts/health_check.py local
--output-dir ~/stock-agent-private/health` under `env -i` with only HOME,
USER, LOGNAME, TMPDIR, an absolute PATH and PYTHONPATH (this checkout's src:
the venv's editable install may point at another checkout), so no variable
from a shell profile or `launchctl setenv` reaches the monitor, verify.py or
pytest. Its options:
    --run-verify            unless --no-verify (the full gates, several minutes)
    --github                when gh is on PATH, unless --no-github; gh's sign-in
                            is not checked here (that would contact GitHub)
    --ledger-head-backup F  the ETF ledger's kept head copy, when found or given
    --prospective-store D   the prospective store, when found or given
    --fund-store D          the F1 fund store, when found or given
    --anchors F             when given, or when STOCK_AGENT_ANCHORS is set here
    --ledger F              when given, or when STOCK_AGENT_PROSPECTIVE_LEDGER is set
                            (the job's environment holds neither variable)
A given path that does not exist yet is still passed (its check then says it
is missing); a default that does not exist is left out, and its checks stay
insufficient evidence. The inputs health_check reads from fixed places (the
registry, the experiment ledger, the producers' inputs with the Stock Lab's
monitor files, the RL runs, the ETF ledger) are listed with whether they
exist, and so are the Stock Lab's store, its kept ledger head and its own
daily agent (scripts/stock_lab_schedule.py).

Exit status: 0 done (the dry run printed; the files were written or already
identical; the status is loaded and fresh), 1 a file differs and nothing was
written, or the status needs attention, 2 refused (wrong place, no usable venv,
not macOS) or bad arguments.

Usage:
    .venv/bin/python scripts/setup_health_schedule.py [--at HH:MM] [--no-verify]
        [--no-github] [--fund-store DIR] [--prospective-store DIR]
        [--ledger-head-backup FILE] [--anchors FILE] [--ledger FILE] [--write]
    .venv/bin/python scripts/setup_health_schedule.py --status
"""

from __future__ import annotations

import argparse
import difflib
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TextIO

from stock_agent.forecasting.prospective.ledger import DEFAULT_LEDGER
from stock_agent.health import backtest, forecast, stock_lab, training
from stock_agent.health.model import (
    CONDITION_PRECEDENCE,
    STALE_AFTER,
    SummaryRefused,
    read_summary,
)
from stock_agent.stock_lab.store import DB_NAME, HOME_ENV

PROJECT = Path(__file__).resolve().parents[1]
LABEL = "local.stock-agent.health"
PRIVATE = "stock-agent-private"  # ~/stock-agent-private: every private store and output
DEFAULT_AT = "07:30"  # local time; after the 06:17 UTC daily workflow on most US clocks
SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
DEFAULT_GIT = "/usr/bin/git"
MAIN_BRANCH = "main"
REFUSED_EXIT = 3  # run.sh's exit status when the checkout is not a clean main
# What the monitor's environment holds (`env -i`): nothing else from launchd or a shell.
KEPT_VARIABLES = ("HOME", "USER", "LOGNAME", "TMPDIR")
# health_check.py's exit statuses, and run.sh's refusal, as --status explains them.
EXIT_MEANINGS = {
    "0": "ok",
    "1": "the summary's overall state is failed",
    "2": "health_check.py refused to run (see the stderr log)",
    str(REFUSED_EXIT): "refused: the checkout is not a clean main (see the stderr log)",
}
# Other modules' names, kept here so this script imports nothing from the prospective
# side or another script (tests check each against its source).
ANCHORS_ENV = "STOCK_AGENT_ANCHORS"
ANCHORS_DEFAULT = Path("data/raw/issuer_sources/anchors.csv")
PROSPECTIVE_LEDGER_ENV = "STOCK_AGENT_PROSPECTIVE_LEDGER"
STOCK_LAB_LABEL = "local.stock-lab.daily"
STOCK_LAB_KEPT_HEAD = "stock_lab_ledger_head"

# run.sh's guard (zsh; `repo` and `git` are set above it): refuse, run nothing, exit
# REFUSED_EXIT and say why on stderr, unless the checkout is a clean main.
GUARD = """\
refuse() {{
  print -u2 -r -- "$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ) health job refused: $1; nothing was run"
  exit {code}
}}
branch=$("$git" -C "$repo" symbolic-ref --quiet --short HEAD) \\
  || refuse "$repo has a detached HEAD, not {main}"
[[ "$branch" == {main} ]] || refuse "$repo is on branch $branch, not {main}"
changes=$("$git" -C "$repo" status --porcelain) || refuse "git status failed in $repo"
[[ -z "$changes" ]] || refuse "$repo has uncommitted or untracked changes (git status --porcelain)"
cd "$repo" || refuse "cannot enter $repo"
"""

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class Machine:
    """What the script reads about this Mac, injected so tests touch nothing real."""

    repo: Path
    home: Path
    env: Mapping[str, str]
    uid: int
    platform: str
    which: Callable[[str], str | None]
    runner: Runner
    now: Callable[[], datetime]


def this_machine() -> Machine:
    env = dict(os.environ)
    return Machine(
        repo=PROJECT,
        home=Path.home(),
        env=env,
        uid=os.getuid(),
        platform=sys.platform,
        which=lambda name: shutil.which(name, path=env.get("PATH")),
        runner=subprocess.run,
        now=lambda: datetime.now(UTC),
    )


@dataclass(frozen=True)
class Places:
    private: Path
    health: Path
    run_sh: Path
    plist: Path
    out_log: Path
    err_log: Path
    latest: Path


def places(machine: Machine) -> Places:
    private = machine.home / PRIVATE
    health = private / "health"
    return Places(
        private=private,
        health=health,
        run_sh=health / "run.sh",
        plist=machine.home / "Library" / "LaunchAgents" / f"{LABEL}.plist",
        out_log=health / "launchd.out.log",
        err_log=health / "launchd.err.log",
        latest=health / "latest.json",
    )


@dataclass(frozen=True)
class Option:
    """One health_check option: passed or not, and why."""

    flag: str
    value: str | None
    enabled: bool
    reason: str


@dataclass(frozen=True)
class Found:
    """An input read from a fixed place, or a Stock Lab item, and whether it exists."""

    name: str
    path: Path
    present: bool
    note: str


@dataclass(frozen=True)
class Plan:
    machine: Machine
    places: Places
    python: Path
    python_version: str | None
    worktree: bool
    gh: str | None
    git: str
    hour: int
    minute: int
    path_dirs: tuple[str, ...]
    options: tuple[Option, ...]
    defaults: tuple[Found, ...]
    stock_lab: tuple[Found, ...]
    warnings: tuple[str, ...]
    refusals: tuple[str, ...]  # why --write would refuse


# --- detection --------------------------------------------------------------------------


def venv_version(repo: Path) -> str | None:
    """The venv's Python version from its pyvenv.cfg (no interpreter is started)."""

    try:
        text = (repo / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in {"version", "version_info"} and value.strip():
            return value.strip()
    return None


def absolute_tool(found: str | None) -> str | None:
    """A tool `which` found, as an absolute path (a relative PATH entry is resolved now)."""

    if found is None:
        return None
    return found if os.path.isabs(found) else os.path.abspath(found)


def checked_out_branch(repo: Path) -> str | None:
    """The branch a main checkout has checked out, from .git/HEAD; None when detached."""

    try:
        head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head.removeprefix("ref: refs/heads/") if head.startswith("ref: refs/heads/") else None


def parse_at(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?[0-9]|2[0-3]):([0-5][0-9])", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"--at takes HH:MM (24-hour local time), not {text!r}")
    return int(match.group(1)), int(match.group(2))


def _absolute(path: Path, machine: Machine) -> Path:
    """`~` is this machine's home; a relative path is taken from the current directory."""

    text = str(path)
    if text == "~" or text.startswith("~/"):
        path = machine.home / text[2:]
    return path if path.is_absolute() else Path(os.path.abspath(path))


def _path_option(
    flag: str, given: Path | None, default: Path | None, checks: str, missing_means: str
) -> Option:
    if given is not None:
        if given.exists():
            return Option(flag, str(given), True, f"given; it exists, so {checks} measure it")
        return Option(
            flag,
            str(given),
            True,
            f"given, but {given} does not exist yet: {checks} report it missing "
            "(unavailable or insufficient evidence) until it does",
        )
    if default is not None and default.exists():
        return Option(flag, str(default), True, f"found at {default}, so {checks} measure it")
    where = f"not found at {default}" if default is not None else "not given"
    return Option(
        flag,
        None,
        False,
        f"{where}: {checks} report insufficient evidence ({missing_means}); once it exists, "
        "run this setup again",
    )


def _env_option(flag: str, given: Path | None, variable: str, machine: Machine, checks: str):
    if given is not None:
        return _path_option(flag, given, None, checks, "")
    value = machine.env.get(variable, "").strip()
    if value:
        path = _absolute(Path(value), machine)
        return Option(
            flag,
            str(path),
            True,
            f"{variable} is set here, but the job's environment holds only the variables run.sh "
            "names, so the path is passed explicitly"
            + ("" if path.exists() else f" ({path} does not exist yet)"),
        )
    return Option(flag, None, False, f"not given and {variable} is not set: the default is read")


def main_checkout(repo: Path) -> Path | None:
    """The main checkout of a linked worktree, from its `.git` file; None if unknown."""

    try:
        text = (repo / ".git").read_text(encoding="utf-8")
    except OSError:
        return None
    gitdir = Path(text.removeprefix("gitdir:").strip()) if text.startswith("gitdir:") else None
    if gitdir is None or gitdir.parent.name != "worktrees":
        return None
    return gitdir.parent.parent.parent


def _count(directory: Path) -> int:
    try:
        return sum(1 for item in directory.glob("*.json") if item.is_file())
    except OSError:
        return 0


def _defaults(machine: Machine, options: Sequence[Option]) -> tuple[Found, ...]:
    repo = machine.repo
    inputs = repo / "reports" / "health_inputs"
    passed = {option.flag for option in options if option.enabled}
    found = [
        Found(
            "model registry",
            repo / "reports" / "ml" / "registry",
            (repo / "reports" / "ml" / "registry").is_dir(),
            "registry.*",
        ),
        Found(
            "experiment ledger",
            repo / "reports" / "ml" / "ledger.jsonl",
            (repo / "reports" / "ml" / "ledger.jsonl").is_file(),
            "training.ml_runs",
        ),
        Found(
            "RL runs", repo / "models" / "rl", (repo / "models" / "rl").is_dir(), "policy checks"
        ),
    ]
    for artifact, checks in (
        (training.ARTIFACT, "training.health"),
        (forecast.ARTIFACT, "forecast.coverage, forecast.calibration"),
        (backtest.ARTIFACT, "portfolio.*"),
        (stock_lab.ARTIFACT, "the Stock Lab checks (written by its daily job)"),
    ):
        count = _count(inputs / artifact)
        found.append(
            Found(f"{artifact} inputs", inputs / artifact, count > 0, f"{checks}; {count} files")
        )
    if "--ledger" not in passed:
        path = repo / DEFAULT_LEDGER
        found.append(Found("ETF forecast ledger", path, path.is_file(), "forecast.ledger*"))
    if "--anchors" not in passed:
        path = repo / ANCHORS_DEFAULT
        found.append(Found("anchors file", path, path.is_file(), "data.anchors"))
    return tuple(found)


def _stock_lab(machine: Machine) -> tuple[tuple[Found, ...], list[str]]:
    configured = machine.env.get(HOME_ENV, "").strip()
    home = (
        _absolute(Path(configured), machine) if configured else machine.home / PRIVATE / "stock_lab"
    )
    store = home / DB_NAME
    kept = home / STOCK_LAB_KEPT_HEAD
    agent = machine.home / "Library" / "LaunchAgents" / f"{STOCK_LAB_LABEL}.plist"
    found = (
        Found("Stock Lab store", store, store.is_file(), "its daily job writes the monitor files"),
        Found(
            "Stock Lab kept ledger head",
            kept,
            kept.is_file(),
            "checked by the daily job (forecast.stock_lab_ledger); its first run starts it",
        ),
        Found(
            "Stock Lab daily agent",
            agent,
            agent.is_file(),
            "printed by scripts/stock_lab_schedule.py; without it the Stock Lab checks stay "
            "insufficient evidence, or turn failed (overdue) once a monitor file exists",
        ),
    )
    warnings = []
    if configured:
        warnings.append(
            f"{HOME_ENV} is set here, but the Stock Lab's launchd agent does not pass it: its "
            "scheduled job would use the default store unless that agent sets it too."
        )
    return found, warnings


def detect(machine: Machine, arguments: argparse.Namespace) -> Plan:
    where = places(machine)
    repo = machine.repo
    python = repo / ".venv" / "bin" / "python"
    version = venv_version(repo)
    worktree = (repo / ".git").is_file()
    gh = None if arguments.no_github else absolute_tool(machine.which("gh"))
    git = absolute_tool(machine.which("git")) or DEFAULT_GIT
    if gh is not None:
        github = (
            f"gh found at {gh}: ci.main_runs and ci.daily_health record the latest runs on main "
            "(gh run list, read-only, the run's only network call). Whether gh is signed in is "
            "not checked here, because that contacts GitHub: run `gh auth status` yourself"
        )
    else:
        github = (
            "--no-github" if arguments.no_github else "gh is not on PATH"
        ) + ": ci.main_runs and ci.daily_health stay insufficient evidence (not_collected)"

    def given(name: str) -> Path | None:
        value = getattr(arguments, name)
        return None if value is None else _absolute(value, machine)

    private = where.private
    options = [
        Option(
            "--run-verify",
            None,
            not arguments.no_verify,
            "runs scripts/verify.py (several minutes), so ci.verify and ci.tests measure the "
            "checked-out code each day"
            if not arguments.no_verify
            else "--no-verify: ci.verify and ci.tests stay insufficient evidence",
        ),
        Option("--github", None, gh is not None, github),
        _path_option(
            "--ledger-head-backup",
            given("ledger_head_backup"),
            private / "ledger_head.txt",
            "forecast.ledger_backup",
            "backup_not_configured: a removed last ledger line would go unnoticed",
        ),
        _path_option(
            "--prospective-store",
            given("prospective_store"),
            private / "prospective_store",
            "data.store_batches and data.window_coverage",
            "not configured",
        ),
        _path_option(
            "--fund-store",
            given("fund_store"),
            private / "fund_store",
            "data.freshness, data.coverage, data.missing_sessions and data.vendor_discrepancy",
            "input_not_configured",
        ),
        _env_option("--anchors", given("anchors"), ANCHORS_ENV, machine, "data.anchors"),
        _env_option(
            "--ledger", given("ledger"), PROSPECTIVE_LEDGER_ENV, machine, "forecast.ledger*"
        ),
    ]
    # Absolute directories only: the job runs after `cd` into the repository, where a
    # relative PATH entry would name a directory inside the checkout.
    dirs = [str(Path(tool).parent) for tool in (gh, git) if tool]
    path_dirs = tuple(
        entry for entry in dict.fromkeys([*dirs, *SYSTEM_PATH]) if os.path.isabs(entry)
    )
    stock_lab_found, warnings = _stock_lab(machine)
    branch = checked_out_branch(repo)
    if not worktree and branch != MAIN_BRANCH:
        warnings.append(
            f"{repo} is on {branch or 'a detached HEAD'}, not {MAIN_BRANCH}: the job refuses to "
            f"run (exit {REFUSED_EXIT}) until the checkout is a clean {MAIN_BRANCH}."
        )
    refusals = []
    if worktree:
        main = main_checkout(repo)
        refusals.append(
            f"{repo} is a linked Git worktree, which is temporary: schedule the monitor from the "
            "main checkout" + (f" ({main})" if main else "")
        )
    if not python.exists():
        refusals.append(f"{python} does not exist: create the project venv first")
    elif version is None or not version.startswith("3.12"):
        refusals.append(f"the venv is Python {version or 'of unknown version'}, not 3.12")
    if machine.platform != "darwin":
        refusals.append(f"launchd is macOS only (this is {machine.platform})")
    for directory in (where.private, where.health):
        if directory.is_dir() and directory.stat().st_mode & 0o077:
            warnings.append(
                f"{directory} is readable by others (mode "
                f"{directory.stat().st_mode & 0o777:o}); keep it private: chmod 700 {directory}"
            )
    return Plan(
        machine=machine,
        places=where,
        python=python,
        python_version=version,
        worktree=worktree,
        gh=gh,
        git=git,
        hour=arguments.at[0],
        minute=arguments.at[1],
        path_dirs=path_dirs,
        options=tuple(options),
        defaults=_defaults(machine, options),
        stock_lab=stock_lab_found,
        warnings=tuple(warnings),
        refusals=tuple(refusals),
    )


# --- the two files ----------------------------------------------------------------------


def render_run_sh(plan: Plan) -> str:
    q = shlex.quote
    repo = plan.machine.repo
    parts = [
        f"{q(str(plan.python))} scripts/health_check.py local",
        f"--output-dir {q(str(plan.places.health))}",
    ]
    for option in plan.options:
        if option.enabled:
            parts.append(
                option.flag if option.value is None else f"{option.flag} {q(option.value)}"
            )
    lines = [
        "#!/bin/zsh -f",
        f"# Model Health: the private local monitor, run daily by launchd ({LABEL}).",
        "# Written by scripts/setup_health_schedule.py, which never overwrites a different",
        "# file: to change it, unload the job, move this file aside and run the setup again.",
        "# Alerts are informational: nothing here trades, retrains, deletes or changes",
        "# configuration.",
    ]
    omitted = [option for option in plan.options if not option.enabled]
    if omitted:
        lines.append("# Options left out:")
        lines += [f"#   {option.flag}: {option.reason.split(':')[0]}" for option in omitted]
    kept = [f'{name}="${{{name}:-}}"' for name in KEPT_VARIABLES]
    environment = [
        "/usr/bin/env -i " + " ".join(kept),
        f"PATH={q(':'.join(plan.path_dirs))} PYTHONPATH={q(str(repo / 'src'))}",
    ]
    lines += [
        "# zsh -f reads none of the user's startup files (~/.zshenv and the others), and",
        "# env -i passes the monitor only the variables named below, so no secret exported",
        "# in a shell profile or with launchctl setenv reaches it, verify.py or pytest.",
        "set -u",
        "umask 077",
        f"repo={q(str(repo))}",
        f"git={q(plan.git)}",
        "# Only a clean main checkout runs: the job is unattended and uses gh's sign-in, so",
        "# it never executes a feature branch or uncommitted code. A refusal runs nothing,",
        f"# exits {REFUSED_EXIT} and writes the reason to stderr (launchd.err.log), which",
        "# scripts/setup_health_schedule.py --status shows.",
        GUARD.format(main=MAIN_BRANCH, code=REFUSED_EXIT).rstrip("\n"),
        "exec " + " \\\n  ".join([*environment, *parts]),
    ]
    return "\n".join(lines) + "\n"


def render_plist(plan: Plan) -> bytes:
    document = {
        "Label": LABEL,
        "ProgramArguments": ["/bin/zsh", "-f", str(plan.places.run_sh)],
        "StartCalendarInterval": {"Hour": plan.hour, "Minute": plan.minute},
        "RunAtLoad": False,
        "Umask": 0o077,
        "StandardOutPath": str(plan.places.out_log),
        "StandardErrorPath": str(plan.places.err_log),
    }
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)


def file_state(path: Path, content: bytes) -> str:
    """absent, identical, differs, symlink, not_a_file or unreadable."""

    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "absent"
    if not path.is_file():
        return "not_a_file"
    try:
        return "identical" if path.read_bytes() == content else "differs"
    except OSError:
        return "unreadable"


def _create(path: Path, content: bytes, mode: int) -> None:
    """Create `path` exclusively (never replaces anything, never follows a link)."""

    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
    os.chmod(path, mode)


# --- output -----------------------------------------------------------------------------

STATE_WORDS = {
    "absent": "absent: {verb}",
    "identical": "already identical: left as it is",
    "differs": "differs from the plan: never overwritten",
    "symlink": "is a symlink: never followed or replaced",
    "not_a_file": "is not a regular file: never replaced",
    "unreadable": "cannot be read: never replaced",
}


def owner_commands(plan: Plan) -> list[tuple[str, str]]:
    plist = shlex.quote(str(plan.places.plist))
    service = f"gui/{plan.machine.uid}/{LABEL}"
    script = "scripts/setup_health_schedule.py"
    return [
        ("check the plist", f"plutil -lint {plist}"),
        (
            "load the job (it then runs daily)",
            f"launchctl bootstrap gui/{plan.machine.uid} {plist}",
        ),
        (
            "run it once now (several minutes with --run-verify)",
            f"launchctl kickstart -k {service}",
        ),
        ("see its state, runs and last exit code", f"launchctl print {service}"),
        ("check the summary is fresh", f".venv/bin/python {script} --status"),
        ("view it", ".venv/bin/python scripts/run_portfolio_lab.py  (Model Health)"),
        ("remove it later", f"launchctl bootout {service}"),
    ]


def print_plan(plan: Plan, out: TextIO, *, writing: bool, run_sh: str, plist: bytes) -> None:
    where = plan.places
    say = partial(print, file=out)
    say(
        "Model Health scheduler setup: "
        + ("writing the files below" if writing else "dry run (nothing is written)")
    )
    say()
    checkout = "linked worktree (temporary)" if plan.worktree else "main checkout"
    say(f"Repository  {plan.machine.repo}  ({checkout})")
    venv = "missing" if not plan.python.exists() else f"Python {plan.python_version or 'unknown'}"
    say(f"Python      {plan.python}  ({venv})")
    say(f"gh          {plan.gh or 'not on PATH (or --no-github)'}")
    say(f"Schedule    every day at {plan.hour:02d}:{plan.minute:02d} local time (launchd runs a")
    say("            missed run when the Mac wakes)")
    say(f"Output      {where.health}  (latest.json, history/, evidence/)")
    say(f"Logs        {where.out_log}")
    say(f"            {where.err_log}")
    say(f"Job PATH    {':'.join(plan.path_dirs)}")
    say()
    say("Health options (+ passed, - left out)")
    for option in plan.options:
        value = f" {option.value}" if option.value else ""
        say(f"  {'+' if option.enabled else '-'} {option.flag}{value}")
        say(f"      {option.reason}")
    say()
    say("Read from fixed places (no option needed)")
    for item in plan.defaults:
        say(f"  {'present' if item.present else 'missing'}  {item.name}: {item.path}")
        say(f"           {item.note}")
    say()
    say("Stock Lab (its own daily job, read through its monitor files)")
    for item in plan.stock_lab:
        say(f"  {'present' if item.present else 'missing'}  {item.name}: {item.path}")
        say(f"           {item.note}")
    if plan.warnings:
        say()
        say("Warnings")
        for warning in plan.warnings:
            say(f"  ! {warning}")
    say()
    say("Files")
    verb = "will be written" if writing else "would be written with --write"
    for path, content, mode in (
        (where.run_sh, run_sh.encode(), "700"),
        (where.plist, plist, "644"),
    ):
        state = file_state(path, content)
        say(f"  {path} (mode {mode}): {STATE_WORDS[state].format(verb=verb)}")
    say()
    say(f"----- {where.run_sh} -----")
    out.write(run_sh)
    say(f"----- {where.plist} -----")
    out.write(plist.decode("utf-8"))
    say("-----")


def print_commands(plan: Plan, out: TextIO) -> None:
    print("\nOwner steps (nothing here runs them):", file=out)
    for number, (what, command) in enumerate(owner_commands(plan), start=1):
        print(f"  {number}. {what}:\n       {command}", file=out)


def _diffs(plan: Plan, files: Sequence[tuple[Path, bytes]], out: TextIO) -> None:
    for path, content in files:
        if file_state(path, content) != "differs":
            continue
        old = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        new = content.decode("utf-8").splitlines(keepends=True)
        out.writelines(difflib.unified_diff(old, new, f"{path} (on disk)", f"{path} (planned)"))


def write(plan: Plan, out: TextIO, *, run_sh: str, plist: bytes) -> int:
    if plan.refusals:
        for refusal in plan.refusals:
            print(f"setup_health_schedule: refused: {refusal}", file=sys.stderr)
        return 2
    where = plan.places
    files = ((where.run_sh, run_sh.encode("utf-8"), 0o700), (where.plist, plist, 0o644))
    blocked = [
        (path, state)
        for path, content, _ in files
        if (state := file_state(path, content)) not in {"absent", "identical"}
    ]
    if blocked:
        _diffs(plan, [(path, content) for path, content, _ in files], out)
        for path, state in blocked:
            print(f"setup_health_schedule: not written: {path} {STATE_WORDS[state]}", file=out)
        print(
            "Nothing was written. To replace a file: unload the job (launchctl bootout), move "
            "the old file aside yourself, and run --write again.",
            file=out,
        )
        return 1
    for directory in (where.private, where.health):
        if not directory.exists():
            directory.mkdir(mode=0o700, parents=True)
            os.chmod(directory, 0o700)
    where.plist.parent.mkdir(parents=True, exist_ok=True)
    for path, content, mode in files:
        if file_state(path, content) == "absent":
            _create(path, content, mode)
            print(f"wrote {path} (mode {mode:o})", file=out)
        else:
            print(f"unchanged {path}", file=out)
    print_commands(plan, out)
    return 0


# --- status -----------------------------------------------------------------------------


def _launchctl(machine: Machine) -> tuple[bool | None, dict[str, str]]:
    """Whether the job is loaded, with launchctl's state, path, runs and last exit code."""

    try:
        completed = machine.runner(
            ["launchctl", "print", f"gui/{machine.uid}/{LABEL}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, {}
    if completed.returncode != 0:
        return False, {}
    fields = {}
    for key in ("state", "path", "runs", "last exit code"):
        match = re.search(rf"^\s*{re.escape(key)} = (.+)$", completed.stdout or "", re.MULTILINE)
        if match:
            fields[key] = match.group(1).strip()
    return True, fields


def status(machine: Machine, out: TextIO) -> int:
    where = places(machine)
    say = partial(print, file=out)
    say("Model Health scheduler status (read-only)")
    loaded, fields = _launchctl(machine)
    if loaded is None:
        say(f"  job       {LABEL}: launchctl is not available here")
    elif not loaded:
        say(f"  job       {LABEL}: not loaded")
    else:
        say(f"  job       {LABEL}: loaded ({fields.get('state', 'state unknown')})")
        say(f"  runs      {fields.get('runs', 'unknown')}")
        code = fields.get("last exit code", "unknown")
        meaning = EXIT_MEANINGS.get(code)
        say(f"  last exit {code}" + (f": {meaning}" if meaning else ""))
        if "path" in fields:
            say(f"  plist     {fields['path']}")
    for label, path in (("run.sh", where.run_sh), ("plist", where.plist)):
        say(f"  {label:<9} {path}: {'present' if path.is_file() else 'absent'}")
    fresh = False
    if not where.latest.exists():
        say(f"  summary   missing: {where.latest} (no run has finished yet)")
    else:
        try:
            summary = read_summary(where.latest)
        except SummaryRefused as error:
            say(f"  summary   refused ({error.reason.value}): {error.detail}")
        except OSError as error:
            say(f"  summary   unreadable: {type(error).__name__}")
        else:
            age = machine.now() - summary.generated_at
            hours = age / timedelta(hours=1)
            fresh = age <= STALE_AFTER
            word = "fresh" if fresh else "stale"
            say(
                f"  last run  {summary.generated_at:%Y-%m-%d %H:%M} UTC, {hours:.1f} hours ago "
                f"(latest.json)"
            )
            limit = int(STALE_AFTER / timedelta(hours=1))
            say(f"  summary   {word} (a daily job's summary is stale after {limit} hours)")
            conditions = Counter(check.condition for check in summary.checks)
            counts = ", ".join(
                f"{conditions[item]} {item.value}"
                for item in CONDITION_PRECEDENCE
                if conditions[item]
            )
            say(f"  overall   {summary.overall_state.value}: {counts or 'no checks'}")
    try:
        size = where.err_log.stat().st_size
    except OSError:
        size = None
    if size:
        lines = where.err_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        say(f"  stderr    {where.err_log}: {size} bytes; last line: {lines[-1] if lines else ''}")
    elif size == 0:
        say(f"  stderr    {where.err_log}: empty")
    if not loaded:
        say("  next      run this script without --status to see the setup, then --write")
    return 0 if loaded and fresh else 1


# --- entry point ------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write run.sh and the plist")
    mode.add_argument(
        "--status", action="store_true", help="read-only: is the job loaded and fresh"
    )
    parser.add_argument("--at", type=parse_at, default=parse_at(DEFAULT_AT), metavar="HH:MM")
    parser.add_argument("--no-verify", action="store_true", help="leave out --run-verify")
    parser.add_argument("--no-github", action="store_true", help="leave out --github")
    for name, metavar in (
        ("--fund-store", "DIR"),
        ("--prospective-store", "DIR"),
        ("--ledger-head-backup", "FILE"),
        ("--anchors", "FILE"),
        ("--ledger", "FILE"),
    ):
        parser.add_argument(name, type=Path, metavar=metavar)
    return parser


def main(
    argv: Sequence[str] | None = None, *, machine: Machine | None = None, out: TextIO = sys.stdout
) -> int:
    arguments = _parser().parse_args(argv)
    machine = machine or this_machine()
    if arguments.status:
        return status(machine, out)
    plan = detect(machine, arguments)
    run_sh, plist = render_run_sh(plan), render_plist(plan)
    print_plan(plan, out, writing=arguments.write, run_sh=run_sh, plist=plist)
    if arguments.write:
        return write(plan, out, run_sh=run_sh, plist=plist)
    for refusal in plan.refusals:
        print(f"\n! --write would refuse: {refusal}", file=out)
    print("\nNothing was written. Review the files above, then run with --write.", file=out)
    print_commands(plan, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
