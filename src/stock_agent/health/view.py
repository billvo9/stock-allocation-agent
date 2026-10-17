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

Words follow the operator condition (`model.Condition`), derived from each
check's state and reason, so "no evidence", "not activated", "stale" and
"failed" never share a word: stale data is degraded with a stale reason and
says "Stale"; a producer that is not authorized yet says "Not activated".

Standard library only (no Streamlit, no pandas), so it is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from stock_agent.health.model import (
    CONDITION_PRECEDENCE,
    AlertKind,
    Area,
    Check,
    Condition,
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
# One word per operator condition: the state's word, except stale data and a producer
# that is not activated yet, which have their own.
CONDITION_WORDS = {
    Condition.FAILED: STATE_WORDS[HealthState.FAILED],
    Condition.UNAVAILABLE: STATE_WORDS[HealthState.UNAVAILABLE],
    Condition.DEGRADED: STATE_WORDS[HealthState.DEGRADED],
    Condition.STALE: "Stale",
    Condition.NO_EVIDENCE: STATE_WORDS[HealthState.INSUFFICIENT_EVIDENCE],
    Condition.NOT_ACTIVATED: "Not activated",
    Condition.HEALTHY: STATE_WORDS[HealthState.HEALTHY],
}
CONDITION_LEVELS = {
    Condition.FAILED: STATE_LEVELS[HealthState.FAILED],
    Condition.UNAVAILABLE: STATE_LEVELS[HealthState.UNAVAILABLE],
    Condition.DEGRADED: STATE_LEVELS[HealthState.DEGRADED],
    Condition.STALE: STATE_LEVELS[HealthState.DEGRADED],
    Condition.NO_EVIDENCE: STATE_LEVELS[HealthState.INSUFFICIENT_EVIDENCE],
    Condition.NOT_ACTIVATED: STATE_LEVELS[HealthState.INSUFFICIENT_EVIDENCE],
    Condition.HEALTHY: STATE_LEVELS[HealthState.HEALTHY],
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
# The headline when the worst condition is one of these (otherwise the state's).
CONDITION_HEADLINES = {
    Condition.STALE: "Working, but some data or evidence is stale",
    Condition.NOT_ACTIVATED: "Nothing measured is wrong; some jobs are not activated yet",
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
    condition: Condition
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
        condition=check.condition,
        state_word=CONDITION_WORDS[check.condition],
        level=CONDITION_LEVELS[check.condition],
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
    """Counts per operator condition, worst first: stale, not activated and no evidence apart."""

    parts = []
    for item in CONDITION_PRECEDENCE:
        count = sum(check.condition is item for check in checks)
        if count:
            parts.append(f"{count} {CONDITION_WORDS[item].lower()}")
    return ", ".join(parts)


def worst_condition(checks) -> Condition:
    """The worst condition among checks (or check views); no evidence for none."""

    present = [check.condition for check in checks]
    if not present:
        return Condition.NO_EVIDENCE
    return min(present, key=CONDITION_PRECEDENCE.index)


def _names(checks: list[Check], limit: int = 4) -> str:
    titles = [check.title for check in checks]
    shown = ", ".join(titles[:limit])
    return shown + (f" and {len(titles) - limit} more" if len(titles) > limit else "")


def _ordered(summary: HealthSummary) -> list[Check]:
    """Checks worst first, then in area order, then as stored."""

    areas = list(Area)
    rank = {item: index for index, item in enumerate(CONDITION_PRECEDENCE)}
    return sorted(summary.checks, key=lambda c: (rank[c.condition], areas.index(c.area)))


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
    waiting = [check for check in ordered if check.condition is Condition.NO_EVIDENCE]
    inactive = [check for check in ordered if check.condition is Condition.NOT_ACTIVATED]
    parts = []
    if waiting:
        parts.append(f"Waiting for evidence: {_names(waiting)}.")
    if inactive:
        parts.append(f"Not activated yet: {_names(inactive)}.")
    if parts:
        return " ".join(["No action needed for what was measured.", *parts])
    return "No action needed."


def _fact(summary: HealthSummary, check_id: str, label: str, value_of) -> Fact | None:
    check = summary.check(check_id)
    if check is None:
        return None
    word = CONDITION_WORDS[check.condition]
    if check.value is None:
        return Fact(label, "—", f"{word}: {check.summary}")
    return Fact(label, value_of(check), f"{word}; measured {_stamp(check.measured_at)}")


def _facts(summary: HealthSummary) -> tuple[Fact, ...]:
    total = len(summary.checks)
    healthy = sum(check.condition is Condition.HEALTHY for check in summary.checks)
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
        area_condition = worst_condition(checks)
        areas.append(
            AreaView(
                area=area,
                title=AREA_TITLES[area],
                state_word=CONDITION_WORDS[area_condition],
                level=CONDITION_LEVELS[area_condition],
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
    overall = worst_condition(summary.checks)
    return HealthView(
        state=state,
        state_word=CONDITION_WORDS[overall],
        level=CONDITION_LEVELS[overall],
        notice=NOTICE_LEVELS[state],
        headline=CONDITION_HEADLINES.get(overall, HEADLINES[state]),
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
