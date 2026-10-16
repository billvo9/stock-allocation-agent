"""
CI and test health: the latest `scripts/verify.py` result and the latest CI
runs on `main`.

A verify log's age comes from its first line, written by the health script
when it runs verify (`health_check: verify.py finished at <UTC>`); a log
without it is unavailable (`verify_time_unknown`), never dated by its file
time. `verify.py` prints a fixed summary block (`verify summary:` then one
`PASS|FAIL  <gate>  (exit N)` line per gate) and pytest prints one result
line per pass (`1845 passed in 70.12s`). This module parses that text; it
never runs anything itself (the health script runs verify and passes the
output in). A measurement older than `STALE_AFTER` is degraded: an old green
result cannot vouch for today's code.

CI runs on `main` come from a JSON run list: a file the owner writes with
the GitHub CLI (the runbook gives the exact command), or the list the local
monitor records itself with `gh run list` when asked to (`health.github`).
Nothing here contacts GitHub. Two checks read the list:

    ci.main_runs     the latest completed `CI` run on main
    ci.daily_health  the latest completed `Model health` run on main (the daily
                     smoke test); degraded when it is older than
                     `DAILY_OVERDUE_AFTER`, because the schedule should have
                     produced a newer one
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from stock_agent.health.model import Area, Better, Check, HealthState, parse_timestamp

STALE_AFTER = timedelta(days=7)

_GATE_LINE = re.compile(r"\s+(PASS|FAIL)  (.+?)  \(exit (-?[0-9]+)\)")
_PYTEST_LINE = re.compile(r"=*\s*((?:[0-9]+ [a-z]+(?:, )?)+) in ([0-9]+(?:\.[0-9]+)?)s\b.*")
_COUNT = re.compile(r"([0-9]+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_ENVIRONMENT_PROBLEMS = ("verify: project virtual environment not usable", "verify: .venv runs")


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    exit_code: int


@dataclass(frozen=True)
class PytestCounts:
    passed: int
    failed: int
    errors: int
    skipped: int
    xfailed: int
    xpassed: int
    seconds: float

    def describe(self) -> str:
        return (
            f"{self.passed} passed, {self.failed} failed, {self.errors} errors, "
            f"{self.skipped} skipped, {self.xfailed} xfailed, {self.xpassed} xpassed "
            f"in {self.seconds:.1f}s"
        )


@dataclass(frozen=True)
class VerifyResult:
    gates: tuple[Gate, ...]
    pytest_runs: tuple[PytestCounts, ...]
    environment_problem: str | None


def _pytest_counts(line: str) -> PytestCounts | None:
    match = _PYTEST_LINE.fullmatch(line.strip())
    if match is None:
        return None
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for number, word in _COUNT.findall(match.group(1)):
        key = "errors" if word.startswith("error") else word
        if key in counts:
            counts[key] += int(number)
    return PytestCounts(**counts, seconds=float(match.group(2)))


def parse_verify_output(text: str) -> VerifyResult | None:
    """The gates and pytest counts of one verify run; None if the text is not one."""

    lines = text.splitlines()
    problem = next(
        (line.strip() for line in lines if line.strip().startswith(_ENVIRONMENT_PROBLEMS)), None
    )
    if problem is not None:
        return VerifyResult(gates=(), pytest_runs=(), environment_problem=problem)
    if "verify summary:" not in lines:
        return None
    start = lines.index("verify summary:") + 1
    gates = []
    for line in lines[start:]:
        match = _GATE_LINE.fullmatch(line)
        if match is None:
            break
        gates.append(Gate(match.group(2), match.group(1) == "PASS", int(match.group(3))))
    if not gates:
        return None
    runs = tuple(
        counts for counts in (_pytest_counts(line) for line in lines[:start]) if counts is not None
    )
    return VerifyResult(gates=tuple(gates), pytest_runs=runs, environment_problem=None)


def _check(check_id: str, title: str, **fields: object) -> Check:
    return Check(check_id=check_id, area=Area.CI, title=title, **fields)


VERIFY_TITLE = "Quality gates (verify.py)"
TESTS_TITLE = "Test suite"


def verify_checks(
    text: str | None, *, measured_at: datetime | None, now: datetime, evidence: str | None
) -> tuple[Check, Check]:
    """`ci.verify` (gates) and `ci.tests` (pytest counts) from one verify output."""

    if text is not None and measured_at is None:
        undated = {
            "state": HealthState.UNAVAILABLE,
            "reason": "verify_time_unknown",
            "measured_at": None,
            "value": None,
            "detail": "The verify log has no 'health_check: verify.py finished at' first line, "
            "so its age is unknown; a file's modification time is not used (a copied old log "
            "would look fresh).",
            "evidence": evidence,
        }
        return (
            _check(
                "ci.verify",
                VERIFY_TITLE,
                unit=None,
                threshold="every gate passes",
                summary="The quality-gate log records no run time, so it cannot be judged.",
                **undated,
            ),
            _check(
                "ci.tests",
                TESTS_TITLE,
                unit="tests passed",
                threshold="0 failed and 0 errors in both pytest passes",
                summary="The test log records no run time, so it cannot be judged.",
                **undated,
            ),
        )
    if text is None or measured_at is None:
        missing = {
            "state": HealthState.INSUFFICIENT_EVIDENCE,
            "reason": "not_collected",
            "measured_at": None,
            "value": None,
            "detail": "No verify.py output was given to this health run.",
            "evidence": None,
        }
        return (
            _check(
                "ci.verify",
                VERIFY_TITLE,
                unit=None,
                threshold="every gate passes",
                summary="No quality-gate result was recorded, so the gates cannot be judged.",
                **missing,
            ),
            _check(
                "ci.tests",
                TESTS_TITLE,
                unit="tests passed",
                threshold="0 failed and 0 errors in both pytest passes",
                summary="No test result was recorded, so the tests cannot be judged.",
                **missing,
            ),
        )

    result = parse_verify_output(text)
    stale = now - measured_at > STALE_AFTER
    age = f"measured {(now - measured_at).days} days before this summary"
    common = {"measured_at": measured_at, "evidence": evidence}

    if result is None or result.environment_problem is not None:
        reason = "unrecognized_output" if result is None else "verify_environment_unusable"
        detail = (
            "The output has no 'verify summary:' block with gate lines."
            if result is None
            else f"verify.py refused to run: {result.environment_problem}"
        )
        unavailable = {"state": HealthState.UNAVAILABLE, "reason": reason, "value": None}
        return (
            _check(
                "ci.verify",
                VERIFY_TITLE,
                unit=None,
                threshold="every gate passes",
                summary="The quality gates did not produce a readable result.",
                detail=detail,
                **unavailable,
                **common,
            ),
            _check(
                "ci.tests",
                TESTS_TITLE,
                unit="tests passed",
                threshold="0 failed and 0 errors in both pytest passes",
                summary="No test counts could be read from the quality-gate output.",
                detail=detail,
                **unavailable,
                **common,
            ),
        )

    failed_gates = [gate.name for gate in result.gates if not gate.passed]
    passed = len(result.gates) - len(failed_gates)
    gate_detail = "; ".join(
        f"{'PASS' if gate.passed else 'FAIL'} {gate.name} (exit {gate.exit_code})"
        for gate in result.gates
    )
    if failed_gates:
        gate_state, gate_reason = HealthState.FAILED, "gate_failed"
        gate_summary = f"{len(failed_gates)} of {len(result.gates)} quality gates failed."
    elif stale:
        gate_state, gate_reason = HealthState.DEGRADED, "stale_evidence"
        gate_summary = f"Every quality gate passed, but the result is old ({age})."
    else:
        gate_state, gate_reason = HealthState.HEALTHY, None
        gate_summary = f"All {len(result.gates)} quality gates passed."
    verify = _check(
        "ci.verify",
        VERIFY_TITLE,
        state=gate_state,
        reason=gate_reason,
        value=passed,
        unit=f"of {len(result.gates)} gates passed",
        threshold=f"every gate passes; a result older than {STALE_AFTER.days} days is degraded",
        better=Better.HIGHER,
        tolerance=0,
        summary=gate_summary,
        detail=f"{gate_detail}; {age}.",
        **common,
    )

    runs = result.pytest_runs
    run_detail = "; ".join(f"pass {index + 1}: {run.describe()}" for index, run in enumerate(runs))
    if not runs:
        tests_state, tests_reason, value = HealthState.UNAVAILABLE, "no_test_counts", None
        tests_summary = "The quality gates ran, but no pytest result line was found."
    elif any(run.failed or run.errors for run in runs):
        tests_state, tests_reason, value = HealthState.FAILED, "tests_failed", runs[0].passed
        broken = max(run.failed + run.errors for run in runs)
        tests_summary = f"{broken} tests failed or errored."
    elif any(run.passed == 0 for run in runs):
        tests_state, tests_reason, value = HealthState.FAILED, "no_tests_ran", 0
        tests_summary = "A pytest pass ran no tests."
    elif len({(run.passed, run.skipped, run.xfailed, run.xpassed) for run in runs}) > 1:
        tests_state, tests_reason, value = HealthState.DEGRADED, "passes_disagree", runs[0].passed
        tests_summary = "Every test passed, but the two pytest passes report different counts."
    elif stale:
        tests_state, tests_reason, value = HealthState.DEGRADED, "stale_evidence", runs[0].passed
        tests_summary = f"{runs[0].passed:,} tests passed, but the result is old ({age})."
    else:
        tests_state, tests_reason, value = HealthState.HEALTHY, None, runs[0].passed
        tests_summary = f"{runs[0].passed:,} tests passed in each pytest pass; none failed."
    tests = _check(
        "ci.tests",
        TESTS_TITLE,
        state=tests_state,
        reason=tests_reason,
        value=value,
        unit="tests passed",
        threshold="0 failed and 0 errors in both pytest passes, with identical counts",
        better=Better.HIGHER,
        tolerance=0,
        summary=tests_summary,
        detail=(run_detail or "no pytest result line") + f"; {age}.",
        **common,
    )
    return verify, tests


# --- CI runs on main (from `gh run list --json ...`) ---------------------------

RUN_FIELDS = (
    "databaseId",
    "status",
    "conclusion",
    "headSha",
    "headBranch",
    "workflowName",
    "createdAt",
    "url",
    "event",
)
MAIN_BRANCH = "main"
CI_WORKFLOW = "CI"  # the `name:` of .github/workflows/ci.yml
HEALTH_WORKFLOW = "Model health"  # the `name:` of .github/workflows/health.yml
# The daily workflow runs once a day; two days without a completed run means
# the schedule stopped (GitHub disables schedules in inactive repositories).
DAILY_OVERDUE_AFTER = timedelta(hours=48)
_FAILED = {"failure", "startup_failure"}
_GITHUB_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


@dataclass(frozen=True)
class CiRun:
    run_id: int
    conclusion: str | None
    status: str
    head_sha: str
    head_branch: str
    workflow: str
    created_at: datetime


def parse_ci_runs(text: str) -> tuple[CiRun, ...] | None:
    """Runs from `gh run list --json <RUN_FIELDS>`; None when the text is not that."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    runs = []
    for item in data:
        if not isinstance(item, dict) or not set(RUN_FIELDS) <= set(item):
            return None
        created = item["createdAt"]
        if not isinstance(created, str) or not _GITHUB_TIME.fullmatch(created):
            return None
        if not isinstance(item["databaseId"], int) or not all(
            isinstance(item[key], str) for key in ("status", "headBranch", "workflowName")
        ):
            return None
        runs.append(
            CiRun(
                run_id=item["databaseId"],
                conclusion=item["conclusion"] or None,
                status=item["status"],
                head_sha=str(item["headSha"]),
                head_branch=item["headBranch"],
                workflow=item["workflowName"],
                created_at=parse_timestamp("createdAt", created.replace("Z", ".000000Z")),
            )
        )
    return tuple(sorted(runs, key=lambda run: run.created_at, reverse=True))


MAIN_RUNS_TITLE = "Latest CI run on main"
DAILY_TITLE = "Latest daily health run"
NOT_COLLECTED_DETAIL = (
    "Pass --github to record the run list with the GitHub CLI, or write it yourself "
    "(runbook) and pass --ci-runs."
)


@dataclass(frozen=True)
class RunList:
    """
    A CI run list as the local monitor received it. `text` is None when no
    list was collected; `reason` and `detail` then say why (for example the
    GitHub CLI is not installed or not signed in).
    """

    text: str | None
    measured_at: datetime | None
    evidence: str | None
    reason: str = "not_collected"
    detail: str = NOT_COLLECTED_DETAIL


def main_runs_check(
    text: str | None,
    *,
    measured_at: datetime | None,
    now: datetime,
    evidence: str | None,
    branch: str = MAIN_BRANCH,
    workflow: str = CI_WORKFLOW,
    missing_reason: str = "not_collected",
    missing_detail: str = NOT_COLLECTED_DETAIL,
) -> Check:
    """
    `ci.main_runs`: the conclusion of the latest completed run of `workflow` on
    `branch`. Runs of other workflows or branches in the list are ignored here,
    whatever command wrote it.
    """

    return _latest_run_check(
        text,
        check_id="ci.main_runs",
        title=MAIN_RUNS_TITLE,
        measured_at=measured_at,
        now=now,
        evidence=evidence,
        branch=branch,
        workflow=workflow,
        overdue_after=None,
        missing_reason=missing_reason,
        missing_detail=missing_detail,
    )


def daily_health_check(
    text: str | None,
    *,
    measured_at: datetime | None,
    now: datetime,
    evidence: str | None,
    missing_reason: str = "not_collected",
    missing_detail: str = NOT_COLLECTED_DETAIL,
) -> Check:
    """
    `ci.daily_health`: the latest completed run of the daily `Model health`
    workflow on main (synthetic fixtures only). A success older than
    `DAILY_OVERDUE_AFTER` is degraded: the schedule should have run since.
    """

    return _latest_run_check(
        text,
        check_id="ci.daily_health",
        title=DAILY_TITLE,
        measured_at=measured_at,
        now=now,
        evidence=evidence,
        branch=MAIN_BRANCH,
        workflow=HEALTH_WORKFLOW,
        overdue_after=DAILY_OVERDUE_AFTER,
        missing_reason=missing_reason,
        missing_detail=missing_detail,
    )


def run_list_checks(runs: RunList, *, now: datetime) -> list[Check]:
    """`ci.main_runs` and `ci.daily_health` from one run list."""

    common = {
        "measured_at": runs.measured_at,
        "now": now,
        "evidence": runs.evidence,
        "missing_reason": runs.reason,
        "missing_detail": runs.detail,
    }
    return [main_runs_check(runs.text, **common), daily_health_check(runs.text, **common)]


def _latest_run_check(
    text: str | None,
    *,
    check_id: str,
    title: str,
    measured_at: datetime | None,
    now: datetime,
    evidence: str | None,
    branch: str,
    workflow: str,
    overdue_after: timedelta | None,
    missing_reason: str,
    missing_detail: str,
) -> Check:
    common = {
        "check_id": check_id,
        "area": Area.CI,
        "title": title,
        "unit": None,
        "threshold": f"the latest completed {workflow} run on {branch} succeeded"
        + (
            f", less than {overdue_after.total_seconds() / 3600:g} hours ago"
            if overdue_after
            else ""
        ),
        "evidence": evidence,
    }
    if text is None or measured_at is None:
        return Check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason=missing_reason,
            measured_at=None,
            value=None,
            summary=f"No list of GitHub runs was collected, so the latest {workflow} result "
            "is not shown here.",
            detail=missing_detail,
            **common,
        )
    runs = parse_ci_runs(text)
    if runs is None:
        return Check(
            state=HealthState.UNAVAILABLE,
            reason="unrecognized_format",
            measured_at=measured_at,
            value=None,
            summary="The CI run list could not be read.",
            detail=f"Expected a JSON list of objects with {', '.join(RUN_FIELDS)}.",
            **common,
        )
    relevant = [run for run in runs if run.head_branch == branch and run.workflow == workflow]
    ignored = len(runs) - len(relevant)
    completed = [run for run in relevant if run.status == "completed"]
    if not completed:
        return Check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_completed_runs",
            measured_at=measured_at,
            value=None,
            summary=f"The CI run list holds no completed {workflow} run on {branch}.",
            detail=f"{len(runs)} runs listed; {len(relevant)} of {workflow} on {branch}, "
            "none completed.",
            **common,
        )
    latest = completed[0]
    failures = sum(run.conclusion in _FAILED for run in completed)
    stamp = latest.created_at.strftime("%Y-%m-%d %H:%M UTC")
    age = now - latest.created_at
    detail = (
        f"run {latest.run_id} on {latest.head_sha[:12]} created {stamp}: "
        f"{latest.conclusion}; {failures} of the last {len(completed)} completed {workflow} "
        f"runs on {branch} failed; {ignored} runs of other workflows or branches ignored; "
        f"list written {(now - measured_at).days} days before this summary."
    )
    if latest.conclusion in _FAILED:
        state, reason = HealthState.FAILED, "ci_failed"
        summary = f"The latest {workflow} run on {branch} failed ({stamp})."
    elif latest.conclusion != "success":
        state, reason = HealthState.DEGRADED, "ci_inconclusive"
        summary = f"The latest {workflow} run on {branch} ended '{latest.conclusion}' ({stamp})."
    elif now - measured_at > STALE_AFTER:
        state, reason = HealthState.DEGRADED, "stale_evidence"
        summary = f"The latest listed {workflow} run succeeded, but the list itself is old."
    elif overdue_after is not None and age > overdue_after:
        state, reason = HealthState.DEGRADED, "daily_run_overdue"
        summary = (
            f"The latest {workflow} run succeeded, but it is {age.days} days old: the daily "
            "schedule has not completed a run since."
        )
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"The latest {workflow} run on {branch} succeeded ({stamp})."
    return Check(
        state=state,
        reason=reason,
        measured_at=measured_at,
        value=latest.conclusion or latest.status,
        summary=summary,
        detail=detail,
        **common,
    )
