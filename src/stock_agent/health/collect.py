"""
The two health runs: `run_ci` (public-safe) and `run_local` (private).

run_ci
    Repository code and synthetic fixtures only: the verify result, the
    dependency checks, the artifact-contract checks and the monitor smoke
    checks. It takes no path to private data, so it cannot read any. The
    summary's scope is `synthetic`; evidence paths are relative to the
    summary file, local paths in the evidence are redacted, and the run
    refuses to write a summary or evidence that still names the repository's
    absolute path, the home directory or an email address. This is what the
    daily GitHub Actions workflow uploads (a public artifact).
run_local
    The same checks plus the private ones on the owner's machine: the GitHub
    run list (recorded with `gh run list` only when asked, `health.github`),
    the F1 fund store, the prospective anchors, store and window (through
    sources the private script passes in, `health.prospective`), the model
    registry and the latest ML experiment run, the prospective forecast
    ledger, the producers' health inputs (training health, the registry
    export, backtest summaries under `reports/health_inputs/` by default) and
    the newest RL evaluation report, each only when it exists (otherwise
    insufficient_evidence). The scope is `private`; evidence paths are
    absolute. It refuses to run inside GitHub Actions, and refuses an output
    directory inside the repository unless Git ignores it, so a private
    summary cannot be committed by accident. It keeps `latest.json`, a
    write-once `history/` and per-run evidence, and compares each run with
    the previous `latest.json` for deterioration alerts. It never deletes
    anything.

Clock, environment, subprocess runner and installed-version lookup are
injected, so tests run without touching the real machine.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.reader import read_store
from stock_agent.data.funds.schema import FundDataRefused
from stock_agent.health import (
    backtest,
    contracts,
    data_quality,
    dependencies,
    forecast,
    github,
    ledger,
    ml_runs,
    policy,
    prospective,
    registry,
    smoke,
    training,
    verification,
)
from stock_agent.health.context import code_state, in_github_actions, run_context
from stock_agent.health.ledger import RulesCalendar
from stock_agent.health.model import (
    Check,
    HealthState,
    HealthSummary,
    NotApplicable,
    Scope,
    SummaryRefused,
    dumps,
    format_timestamp,
    parse_timestamp,
    read_summary,
)
from stock_agent.health.prospective import ProspectiveSources
from stock_agent.health.summary import build_summary, write_summary

SUMMARY_FILE = "health_summary.json"
LATEST_FILE = "latest.json"
EVIDENCE_DIR = "evidence"
HISTORY_DIR = "history"
DEFAULT_PRIVATE_DIR = Path.home() / "stock-agent-private" / "health"

Clock = Callable[[], datetime]
Runner = Callable[..., subprocess.CompletedProcess]


class LocalRunRefused(Exception):
    """The private monitor refused to run (wrong place or unsafe output)."""


def utc_now() -> datetime:
    return datetime.now(UTC)


class PublicSummaryRefused(Exception):
    """A public (CI) summary or its evidence would carry something private."""


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class Environment:
    root: Path
    env: Mapping[str, str]
    now: Clock = utc_now
    runner: Runner = subprocess.run
    installed: Callable[[str], str | None] | None = None
    python: str = sys.executable
    home: Path = field(default_factory=Path.home)

    def version_of(self, name: str) -> str | None:
        if self.installed is not None:
            return self.installed(name)
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None


def public_text(environment: Environment, text: str) -> str:
    """Repository paths made relative and the home directory shown as `~`."""

    for root in {str(environment.root.resolve()), str(environment.root)}:
        text = text.replace(root + "/", "").replace(root, ".")
    return text.replace(str(environment.home), "~")


def check_public(environment: Environment, label: str, text: str) -> None:
    """Refuse text bound for a public artifact that still names a local path or an email."""

    problems = [
        name
        for name, found in (
            ("the repository's absolute path", str(environment.root.resolve()) in text),
            ("the home directory", str(environment.home) in text),
            ("an email address", EMAIL.search(text) is not None),
        )
        if found
    ]
    if problems:
        raise PublicSummaryRefused(f"{label} would publish {', '.join(problems)}")


# The first line of every verify log this script records: when verify.py
# finished. A log's age comes from this line, never from the file's mtime
# (a copied old log would look fresh).
VERIFY_STAMP = "health_check: verify.py finished at "


@dataclass(frozen=True)
class VerifyOutput:
    text: str
    measured_at: datetime | None  # None: the log records no run time


def run_verify(environment: Environment, *, public: bool) -> VerifyOutput:
    """
    Run scripts/verify.py once. The output starts with a `VERIFY_STAMP` line;
    for a public summary its local paths are redacted (`public_text`).
    Nothing is written here: the caller writes the evidence.
    """

    command = [environment.python, str(environment.root / "scripts" / "verify.py")]
    try:
        completed = environment.runner(
            command,
            cwd=environment.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
            check=False,
        )
        text = completed.stdout or ""
    except (OSError, subprocess.SubprocessError) as error:
        text = f"verify.py could not be run: {type(error).__name__}: {error}\n"
    if public:
        text = public_text(environment, text)
    finished = environment.now()
    return VerifyOutput(f"{VERIFY_STAMP}{format_timestamp(finished)}\n{text}", finished)


def read_verify_log(path: Path) -> VerifyOutput:
    """An existing verify log, dated by its `VERIFY_STAMP` line (None without one)."""

    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.split("\n", 1)[0]
    measured_at = None
    if first.startswith(VERIFY_STAMP):
        try:
            measured_at = parse_timestamp("verify stamp", first.removeprefix(VERIFY_STAMP))
        except SummaryRefused:
            measured_at = None
    return VerifyOutput(text, measured_at)


VERIFY_LOG = "verify.log"
PIP_LOG = "pip_check.txt"


def _shared_checks(
    environment: Environment,
    verify: VerifyOutput | None,
    *,
    show: Callable[[str], str],
    public: bool,
) -> tuple[list[Check], dict[str, str]]:
    """The shared checks and their evidence texts by file name (nothing written)."""

    now = environment.now
    interpreter = public_text(environment, environment.python) if public else environment.python
    evidence: dict[str, str] = {}
    if verify is not None:
        evidence[VERIFY_LOG] = public_text(environment, verify.text) if public else verify.text
    checks = list(
        verification.verify_checks(
            None if verify is None else verify.text,
            measured_at=None if verify is None else verify.measured_at,
            now=now(),
            evidence=None if verify is None else show(VERIFY_LOG),
        )
    )
    requirements = environment.root / "requirements.txt"
    checks.append(
        dependencies.pins_check(
            requirements.read_text(encoding="utf-8") if requirements.is_file() else None,
            installed=environment.version_of,
            interpreter=interpreter,
            measured_at=now(),
            evidence="requirements.txt",
        )
    )
    pip, output = dependencies.pip_check(
        python=environment.python,
        display=interpreter,
        runner=environment.runner,
        measured_at=now(),
        evidence=show(PIP_LOG),
    )
    evidence[PIP_LOG] = public_text(environment, output) if public else output
    checks.append(pip)
    checks.extend(contracts.contract_checks(now=now))
    checks.extend(smoke.smoke_checks(now=now))
    return checks, evidence


def _write_evidence(directory: Path, evidence: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in evidence.items():
        (directory / name).write_text(text, encoding="utf-8")


def run_ci(
    environment: Environment, output_dir: Path, *, verify: VerifyOutput | None
) -> tuple[HealthSummary, Path]:
    """
    The public-safe summary at `<output_dir>/health_summary.json`. The summary
    and every evidence text are checked before anything is written, so a
    refused run leaves nothing in `output_dir` for the workflow to upload.
    """

    checks, evidence = _shared_checks(
        environment, verify, show=lambda name: f"{EVIDENCE_DIR}/{name}", public=True
    )
    summary = build_summary(
        checks,
        generated_at=environment.now(),
        code=code_state(environment.root, environment.runner),
        scope=Scope.SYNTHETIC,
        run_context=run_context(environment.env),
    )
    check_public(environment, "the summary", dumps(summary.to_json_dict()))
    for name, text in evidence.items():
        check_public(environment, f"{EVIDENCE_DIR}/{name}", text)
    _write_evidence(output_dir / EVIDENCE_DIR, evidence)
    return summary, write_summary(output_dir / SUMMARY_FILE, summary)


@dataclass(frozen=True)
class PrivateInputs:
    ci_runs: Path | None = None
    # Record the GitHub run list with `gh run list` (the one network step).
    github: bool = False
    fund_store: Path | None = None
    # Producers' health inputs: <health_inputs>/<artifact>/<run_id>.json
    # (reports/health_inputs/ by agreement), the newest per artifact.
    health_inputs: Path | None = None
    registry: Path | None = None  # the registry root (reports/ml/registry)
    ml_artifacts: Path | None = None  # its artifacts root; default: the root's parent
    ml_root: Path | None = None  # the experiment runner's root (reports/ml): ledger.jsonl
    rl_runs: Path | None = None  # RL run directories (models/rl): evaluation reports
    ledger: Path | None = None  # the prospective forecast ledger
    ledger_head_backup: Path | None = None  # the owner's kept copy of its head hash
    # The rules calendar and the prospective sources come from the private
    # script: research code may not import the prospective side.
    calendar: RulesCalendar | None = None
    prospective: ProspectiveSources | None = None

    def artifact_directory(self, artifact: str) -> Path | None:
        return None if self.health_inputs is None else self.health_inputs / artifact


def check_private_output(environment: Environment, output_dir: Path) -> None:
    """Refuse inside GitHub Actions, or an output directory Git would track."""

    if in_github_actions(environment.env):
        raise LocalRunRefused(
            "the local monitor reads private stores on the owner's machine; GitHub Actions "
            "cannot see them, so it does not run there (use the ci command)"
        )
    target = output_dir.expanduser().resolve()
    root = environment.root.resolve()
    if not target.is_relative_to(root):
        return
    try:
        ignored = environment.runner(
            ["git", "check-ignore", "-q", str(target)], cwd=root, timeout=30, check=False
        ).returncode
    except (OSError, subprocess.SubprocessError) as error:
        raise LocalRunRefused(f"cannot confirm Git ignores {target}: {error}") from error
    if ignored != 0:
        raise LocalRunRefused(
            f"{target} is inside the repository and not ignored by Git; private summaries "
            "must stay out of version control (use reports/ or a path outside the repository)"
        )


def _fund_store_checks(path: Path | None, *, as_of: datetime) -> list[Check]:
    if path is None:
        return data_quality.not_collected_checks(
            reason="input_not_configured",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            detail="No --fund-store was given to this run.",
        )
    if not path.is_dir():
        return data_quality.not_collected_checks(
            reason="store_missing",
            state=HealthState.UNAVAILABLE,
            detail=f"{path} is not a directory.",
        )
    try:
        store = read_store(path)
    except FundDataRefused as error:
        return data_quality.not_collected_checks(
            reason="store_refused",
            state=HealthState.UNAVAILABLE,
            detail=f"read_store refused {path}: {error.reason.value}: {error.detail}",
        )
    return data_quality.data_quality_checks(
        store, as_of=pd.Timestamp(as_of), measured_at=as_of, evidence=str(path)
    )


def _run_list(
    environment: Environment, inputs: PrivateInputs, evidence_dir: Path
) -> verification.RunList:
    """The GitHub run list: recorded with gh when asked, else the owner's file, else none."""

    if inputs.github:
        return github.collect_runs(
            environment.runner, now=environment.now, evidence_dir=evidence_dir
        )
    path = inputs.ci_runs
    if path is not None and path.is_file():
        return verification.RunList(
            path.read_text(encoding="utf-8", errors="replace"),
            datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            str(path),
        )
    return verification.RunList(None, None, None if path is None else str(path))


def _previous(path: Path) -> HealthSummary | None:
    try:
        return read_summary(path)
    except SummaryRefused:
        return None


def run_local(
    environment: Environment,
    output_dir: Path,
    *,
    verify: VerifyOutput | None,
    inputs: PrivateInputs,
) -> tuple[HealthSummary, Path]:
    """The private summary; returns it and the path of `latest.json`."""

    check_private_output(environment, output_dir)
    output_dir = output_dir.expanduser().resolve()
    started = environment.now()
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    evidence_dir = output_dir / EVIDENCE_DIR / stamp
    checks, evidence = _shared_checks(
        environment, verify, show=lambda name: str(evidence_dir / name), public=False
    )
    _write_evidence(evidence_dir, evidence)

    checks.extend(
        verification.run_list_checks(
            _run_list(environment, inputs, evidence_dir), now=environment.now()
        )
    )
    checks.extend(_fund_store_checks(inputs.fund_store, as_of=environment.now()))
    checks.extend(prospective.prospective_checks(inputs.prospective, now=environment.now()))
    checks.extend(training.training_checks(inputs.artifact_directory(training.ARTIFACT)))
    checks.append(
        ml_runs.ml_runs_check(
            inputs.ml_root,
            data_classes=registry.data_classes(inputs.registry, inputs.ml_artifacts),
        )
    )
    checks.extend(
        registry.registry_checks(
            inputs.registry, artifacts_root=inputs.ml_artifacts, now=environment.now
        )
    )
    forecasts = forecast.forecast_checks(inputs.artifact_directory(forecast.ARTIFACT))
    checks.extend(item for item in forecasts if isinstance(item, Check))
    checks.extend(
        ledger.ledger_checks(
            inputs.ledger,
            backup=inputs.ledger_head_backup,
            calendar=inputs.calendar,
            now=environment.now(),
        )
    )
    checks.extend(backtest.backtest_checks(inputs.artifact_directory(backtest.ARTIFACT)))
    checks.extend(
        policy.policy_checks(inputs.artifact_directory(backtest.ARTIFACT), inputs.rl_runs)
    )

    latest = output_dir / LATEST_FILE
    summary = build_summary(
        checks,
        generated_at=environment.now(),
        code=code_state(environment.root, environment.runner),
        scope=Scope.PRIVATE,
        run_context=run_context(environment.env),
        previous=_previous(latest),
        not_applicable=[item for item in forecasts if isinstance(item, NotApplicable)],
    )
    history = output_dir / HISTORY_DIR / f"{stamp}.json"
    if history.exists():
        raise LocalRunRefused(f"{history} already exists; history files are write-once")
    write_summary(history, summary)
    return summary, write_summary(latest, summary)
