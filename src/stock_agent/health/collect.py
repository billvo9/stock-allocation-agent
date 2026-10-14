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
    The same checks plus the private ones on the owner's machine: the CI
    run list, the F1 fund store, the model registry, and the producers' health
    inputs (training health, the registry export, backtest summaries under
    `reports/health_inputs/` by default), each only when it exists (otherwise
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
    registry,
    smoke,
    training,
    verification,
)
from stock_agent.health.context import code_state, in_github_actions, run_context
from stock_agent.health.model import (
    Check,
    HealthState,
    HealthSummary,
    NotApplicable,
    Scope,
    SummaryRefused,
    dumps,
    read_summary,
)
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


@dataclass(frozen=True)
class VerifyOutput:
    text: str
    measured_at: datetime


def run_verify(environment: Environment, log: Path, *, public: bool = False) -> VerifyOutput:
    """
    Run scripts/verify.py once and keep its whole output as evidence. For a
    public summary the output's local paths are redacted (`public_text`).
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
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(text, encoding="utf-8")
    return VerifyOutput(text, environment.now())


def read_verify_log(path: Path) -> VerifyOutput:
    """An existing verify log; measured at the file's modification time."""

    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return VerifyOutput(path.read_text(encoding="utf-8", errors="replace"), modified)


def _shared_checks(
    environment: Environment,
    verify: VerifyOutput | None,
    *,
    evidence_dir: Path,
    show: Callable[[Path], str],
    public: bool,
) -> list[Check]:
    now = environment.now
    interpreter = public_text(environment, environment.python) if public else environment.python
    checks = list(
        verification.verify_checks(
            None if verify is None else verify.text,
            measured_at=None if verify is None else verify.measured_at,
            now=now(),
            evidence=None if verify is None else show(evidence_dir / "verify.log"),
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
    pip_log = evidence_dir / "pip_check.txt"
    pip, output = dependencies.pip_check(
        python=environment.python,
        display=interpreter,
        runner=environment.runner,
        measured_at=now(),
        evidence=show(pip_log),
    )
    if public:
        output = public_text(environment, output)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pip_log.write_text(output, encoding="utf-8")
    checks.append(pip)
    checks.extend(contracts.contract_checks(now=now))
    checks.extend(smoke.smoke_checks(now=now))
    return checks


def run_ci(
    environment: Environment, output_dir: Path, *, verify: VerifyOutput | None
) -> tuple[HealthSummary, Path]:
    """The public-safe summary at `<output_dir>/health_summary.json`."""

    evidence_dir = output_dir / EVIDENCE_DIR

    def relative(path: Path) -> str:
        return path.relative_to(output_dir).as_posix()

    checks = _shared_checks(
        environment, verify, evidence_dir=evidence_dir, show=relative, public=True
    )
    summary = build_summary(
        checks,
        generated_at=environment.now(),
        code=code_state(environment.root, environment.runner),
        scope=Scope.SYNTHETIC,
        run_context=run_context(environment.env),
    )
    check_public(environment, "the summary", dumps(summary.to_json_dict()))
    for evidence in sorted(evidence_dir.glob("*")):
        check_public(environment, f"evidence/{evidence.name}", evidence.read_text("utf-8"))
    return summary, write_summary(output_dir / SUMMARY_FILE, summary)


@dataclass(frozen=True)
class PrivateInputs:
    ci_runs: Path | None = None
    fund_store: Path | None = None
    # Producers' health inputs: <health_inputs>/<artifact>/<run_id>.json
    # (reports/health_inputs/ by agreement), the newest per artifact.
    health_inputs: Path | None = None
    registry: Path | None = None  # the registry root (reports/ml/registry)
    ml_artifacts: Path | None = None  # its artifacts root; default: the root's parent

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
    checks = _shared_checks(environment, verify, evidence_dir=evidence_dir, show=str, public=False)

    ci_runs = inputs.ci_runs
    runs_text, runs_written = None, None
    if ci_runs is not None and ci_runs.is_file():
        runs_text = ci_runs.read_text(encoding="utf-8", errors="replace")
        runs_written = datetime.fromtimestamp(ci_runs.stat().st_mtime, tz=UTC)
    checks.append(
        verification.main_runs_check(
            runs_text,
            measured_at=runs_written,
            now=environment.now(),
            evidence=None if ci_runs is None else str(ci_runs),
        )
    )
    checks.extend(_fund_store_checks(inputs.fund_store, as_of=environment.now()))
    checks.extend(training.training_checks(inputs.artifact_directory(training.ARTIFACT)))
    checks.extend(
        registry.registry_checks(
            inputs.registry, artifacts_root=inputs.ml_artifacts, now=environment.now
        )
    )
    forecasts = forecast.forecast_checks(inputs.artifact_directory(forecast.ARTIFACT))
    checks.extend(item for item in forecasts if isinstance(item, Check))
    checks.extend(backtest.backtest_checks(inputs.artifact_directory(backtest.ARTIFACT)))

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
