"""
The Model Health page's view model: pure, built only from a stored summary.

Progressive disclosure: a headline in plain words and one next action first;
then the facts that matter most (test counts, gates, the latest CI run) taken
from the summary's checks; then each area with its checks' plain summaries;
exact measurements, thresholds and technical detail last (the page puts them
in expanders). The scope note and the informational-only caveat are always
shown, never collapsed.

Every number on the page comes from the summary. A value the summary does not
hold is shown as "not measured", never as zero or a placeholder.

Standard library only (no Streamlit, no pandas), so it is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from stock_agent.health.model import (
    PRECEDENCE,
    AlertKind,
    Area,
    Check,
    HealthState,
    HealthSummary,
    Scope,
)

RUNBOOK_BASE_URL = "https://github.com/billvo9/stock-allocation-agent/blob/main/"
OLD_AFTER = timedelta(days=2)
NOT_MEASURED = "not measured"
CAVEAT = (
    "Alerts are informational. Nothing on this page trades, retrains, promotes a model, "
    "deletes data or changes configuration automatically."
)

STATE_WORDS = {
    HealthState.HEALTHY: "Healthy",
    HealthState.INSUFFICIENT_EVIDENCE: "Insufficient evidence",
    HealthState.DEGRADED: "Degraded",
    HealthState.UNAVAILABLE: "Unavailable",
    HealthState.FAILED: "Failed",
}
# The design system's status treatments, one per state; the state is always also
# written out.
STATE_LEVELS = {
    HealthState.HEALTHY: "VALID",
    HealthState.INSUFFICIENT_EVIDENCE: "NOTICE",
    HealthState.DEGRADED: "WARNING",
    HealthState.UNAVAILABLE: "UNAVAILABLE",
    HealthState.FAILED: "INVALID/BLOCKED",
}
NOTICE_LEVELS = {
    HealthState.HEALTHY: "success",
    HealthState.INSUFFICIENT_EVIDENCE: "notice",
    HealthState.DEGRADED: "warning",
    HealthState.UNAVAILABLE: "warning",
    HealthState.FAILED: "blocking",
}
HEADLINES = {
    HealthState.HEALTHY: "Everything measured is healthy",
    HealthState.INSUFFICIENT_EVIDENCE: "Nothing measured is wrong, but some areas have no "
    "evidence yet",
    HealthState.DEGRADED: "Working, with warnings",
    HealthState.UNAVAILABLE: "Some measurements could not be taken",
    HealthState.FAILED: "Something is broken",
}
AREA_TITLES = {
    Area.CI: "CI and tests",
    Area.DEPS: "Dependencies",
    Area.CONTRACT: "Artifact contracts",
    Area.SMOKE: "Monitor self-tests",
    Area.DATA: "Data quality",
    Area.TRAINING: "Training",
    Area.FORECAST: "Forecasts",
    Area.REGISTRY: "Model registry",
    Area.PORTFOLIO: "Portfolio (simulated backtests)",
}
ALERT_WORDS = {
    AlertKind.FAILURE: "Failure",
    AlertKind.CONSTRAINT_VIOLATION: "Constraint violation",
    AlertKind.UNAVAILABLE_DATA: "Unavailable data",
    AlertKind.DEGRADED: "Degraded",
    AlertKind.DETERIORATION: "Deterioration",
}
SCOPE_NOTES = {
    Scope.SYNTHETIC: (
        "Built from repository code and synthetic fixtures only. CI cannot see private data "
        "or local models on the owner's Mac, so data, training, registry and portfolio health "
        "appear only in the local monitor's summary."
    ),
    Scope.PRIVATE: (
        "Built on this machine from private stores. Keep this summary private: it can name "
        "local paths and describe licensed data."
    ),
}


def _stamp(value: datetime | None) -> str:
    return NOT_MEASURED if value is None else value.strftime("%Y-%m-%d %H:%M UTC")


def value_text(check: Check) -> str:
    """The measured value with its unit, as read; `not measured` when absent."""

    value = check.value
    if value is None:
        return NOT_MEASURED
    if isinstance(value, str):
        text = value
    elif isinstance(value, int):
        text = f"{value:,}"
    elif check.unit and check.unit.startswith("fraction") and 0 <= value <= 1:
        return f"{value:.2%} ({check.unit})"
    else:
        text = f"{value:,.6g}"
    return f"{text} {check.unit}" if check.unit else text


@dataclass(frozen=True)
class CheckView:
    check_id: str
    title: str
    state: HealthState
    state_word: str
    level: str
    summary: str
    value: str
    threshold: str
    measured: str
    reason: str
    detail: str
    runbook_url: str
    evidence: str


@dataclass(frozen=True)
class AreaView:
    area: Area
    title: str
    state_word: str
    level: str
    counts: str
    checks: tuple[CheckView, ...]


@dataclass(frozen=True)
class Fact:
    label: str
    value: str
    caption: str


@dataclass(frozen=True)
class AlertView:
    kind_word: str
    notice: str
    message: str


@dataclass(frozen=True)
class HealthView:
    state: HealthState
    state_word: str
    level: str
    notice: str
    headline: str
    summary: str
    next_action: str
    age_warning: str | None
    scope_note: str
    caveat: str
    facts: tuple[Fact, ...]
    alerts: tuple[AlertView, ...]
    areas: tuple[AreaView, ...]
    not_applicable: tuple[str, ...]
    provenance: tuple[str, ...]


def check_view(check: Check, *, base_url: str = RUNBOOK_BASE_URL) -> CheckView:
    return CheckView(
        check_id=check.check_id,
        title=check.title,
        state=check.state,
        state_word=STATE_WORDS[check.state],
        level=STATE_LEVELS[check.state],
        summary=check.summary,
        value=value_text(check),
        threshold=check.threshold or "none declared",
        measured=_stamp(check.measured_at),
        reason=check.reason or "none",
        detail=check.detail,
        runbook_url=base_url + check.runbook,
        evidence=check.evidence or "none",
    )


def _counts(checks: tuple[Check, ...]) -> str:
    parts = []
    for state in PRECEDENCE:
        count = sum(check.state is state for check in checks)
        if count:
            parts.append(f"{count} {STATE_WORDS[state].lower()}")
    return ", ".join(parts)


def _names(checks: list[Check], limit: int = 4) -> str:
    titles = [check.title for check in checks]
    shown = ", ".join(titles[:limit])
    return shown + (f" and {len(titles) - limit} more" if len(titles) > limit else "")


def _ordered(summary: HealthSummary) -> list[Check]:
    """Checks worst first, then in area order, then as stored."""

    areas = list(Area)
    rank = {state: index for index, state in enumerate(PRECEDENCE)}
    return sorted(summary.checks, key=lambda c: (rank[c.state], areas.index(c.area)))


def _next_action(summary: HealthSummary) -> str:
    state = summary.overall_state
    ordered = _ordered(summary)
    if not summary.checks:
        return "Run scripts/health_check.py to measure something; this summary holds no check."
    first = ordered[0]
    if state in (HealthState.FAILED, HealthState.UNAVAILABLE, HealthState.DEGRADED):
        return (
            f"Start with “{first.title}”: {first.summary} Open its runbook entry for the "
            "diagnostic commands and the safe retry. Nothing was changed automatically."
        )
    waiting = [check for check in ordered if check.state is HealthState.INSUFFICIENT_EVIDENCE]
    if waiting:
        return f"No action needed for what was measured. Waiting for evidence: {_names(waiting)}."
    return "No action needed."


def _fact(summary: HealthSummary, check_id: str, label: str, value_of) -> Fact | None:
    check = summary.check(check_id)
    if check is None:
        return None
    if check.value is None:
        return Fact(label, "—", f"{STATE_WORDS[check.state]}: {check.summary}")
    return Fact(
        label,
        value_of(check),
        f"{STATE_WORDS[check.state]}; measured {_stamp(check.measured_at)}",
    )


def _facts(summary: HealthSummary) -> tuple[Fact, ...]:
    total = len(summary.checks)
    healthy = sum(check.state is HealthState.HEALTHY for check in summary.checks)
    facts = [
        # A summary is producer-written: a value or unit of an unexpected shape is shown
        # as stored, never allowed to break the page.
        _fact(
            summary,
            "ci.tests",
            "Tests passed",
            lambda c: value_text(c).removesuffix(" tests passed"),
        ),
        _fact(
            summary,
            "ci.verify",
            "Quality gates passed",
            lambda c: f"{c.value} {(c.unit or '').removesuffix(' gates passed')}".strip(),
        ),
        _fact(summary, "ci.main_runs", "Latest CI on main", lambda c: str(c.value)),
        Fact("Checks healthy", f"{healthy} of {total}", _counts(summary.checks) or "no checks"),
        Fact(
            "Informational alerts",
            str(len(summary.alerts)),
            "none" if not summary.alerts else "listed below; nothing acts on them",
        ),
    ]
    return tuple(fact for fact in facts if fact is not None)


def _provenance(summary: HealthSummary) -> tuple[str, ...]:
    if summary.code_sha is None:
        code = "code unknown (git was not available)"
    else:
        tree = {True: "dirty tree", False: "clean tree", None: "tree state unknown"}
        code = f"code {summary.code_sha[:12]} ({tree[summary.code_dirty]})"
    lines = [
        (
            f"Generated {_stamp(summary.generated_at)}; {code}; summary schema "
            f"{summary.schema_version}; scope {summary.scope.value}."
        )
    ]
    context = summary.run_context
    if context.get("environment") == "github_actions":
        run = (
            f"GitHub Actions: workflow {context.get('workflow') or '?'}, run "
            f"{context.get('run_number') or '?'} (attempt {context.get('run_attempt') or '?'}), "
            f"event {context.get('event') or '?'}, ref {context.get('ref') or '?'}"
        )
        if context.get("run_url"):
            run += f": {context['run_url']}"
        lines.append(run + ".")
    else:
        lines.append("Produced on a local machine.")
    lines.append(
        "The code SHA and tree state are the producer's claims, not authenticated provenance."
    )
    return tuple(lines)


def build_view(
    summary: HealthSummary, *, now: datetime, base_url: str = RUNBOOK_BASE_URL
) -> HealthView:
    state = summary.overall_state
    age = now - summary.generated_at
    age_warning = (
        f"This summary is {age.days} days old (generated {_stamp(summary.generated_at)}); "
        "it describes the code and data as they were then."
        if age > OLD_AFTER
        else None
    )
    areas = []
    for area in Area:
        checks = tuple(check for check in summary.checks if check.area is area)
        if not checks:
            continue
        area_state = min((c.state for c in checks), key=PRECEDENCE.index)
        areas.append(
            AreaView(
                area=area,
                title=AREA_TITLES[area],
                state_word=STATE_WORDS[area_state],
                level=STATE_LEVELS[area_state],
                counts=_counts(checks),
                checks=tuple(check_view(check, base_url=base_url) for check in checks),
            )
        )
    alerts = tuple(
        AlertView(
            kind_word=ALERT_WORDS[alert.kind],
            notice=NOTICE_LEVELS[alert.state]
            if alert.kind is not AlertKind.DETERIORATION
            else "warning",
            message=alert.message,
        )
        for alert in summary.alerts
    )
    total = len(summary.checks)
    return HealthView(
        state=state,
        state_word=STATE_WORDS[state],
        level=STATE_LEVELS[state],
        notice=NOTICE_LEVELS[state],
        headline=HEADLINES[state],
        summary=(
            f"{total} checks: {_counts(summary.checks)}."
            if total
            else "This summary has no checks."
        ),
        next_action=_next_action(summary),
        age_warning=age_warning,
        scope_note=SCOPE_NOTES[summary.scope],
        caveat=CAVEAT,
        facts=_facts(summary),
        alerts=alerts,
        areas=tuple(areas),
        not_applicable=tuple(
            f"{item.title} ({AREA_TITLES[item.area]}): {item.detail}"
            for item in summary.not_applicable
        ),
        provenance=_provenance(summary),
    )
