"""
The Model Health page's inputs (pure; no Streamlit): the local monitor's
latest summary and the latest daily CI summary, each read with the Model
Health Center's strict reader (`stock_agent.health.model.read_summary`) and
its view model (`stock_agent.health.view.build_view`), and the Lab's
grouping of their checks into the questions the page answers.

Where the summaries are: the environment variables below, which
scripts/run_portfolio_lab.py sets (the local monitor's
`~/stock-agent-private/health/latest.json`, and the newest downloaded
`reports/health/ci/*/health_summary.json`). Unset or absent means "no
summary yet", never an assumed state. A summary the reader refuses is shown
as refused, with nothing from it. Nothing here makes a network call, writes
a file or computes a measurement: every state, value and sentence comes from
a summary.

Boundary (tested): the app imports only `stock_agent.health.model` and
`stock_agent.health.view` from the health center, both standard library
only; never a collector.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.health.model import (
    CONDITION_PRECEDENCE,
    Condition,
    HealthState,
    HealthSummary,
    SummaryRefused,
    read_summary,
)
from stock_agent.health.view import (
    CONDITION_LEVELS,
    CONDITION_WORDS,
    CheckView,
    HealthView,
    build_view,
    check_view,
    worst_condition,
)

LOCAL_VARIABLE = "STOCK_AGENT_HEALTH_LOCAL_SUMMARY"
CI_VARIABLE = "STOCK_AGENT_HEALTH_CI_SUMMARY"

# The Lab's sections: (key, title, check ids or id prefixes), in page order. A
# check goes to the first section that names it; any other check goes to OTHER,
# so nothing a summary holds is hidden.
SECTIONS = (
    ("code", "Code and tests", ("ci.", "deps.", "contract.", "smoke.")),
    (
        "runs",
        "Training and evaluation runs",
        ("training.", "forecast.coverage", "forecast.calibration"),
    ),
    ("models", "Models, versions and evidence stage", ("registry.",)),
    ("ledger", "Prospective forecast ledger", ("forecast.ledger",)),
    (
        "stock_lab",
        "Stock Lab daily job (experimental)",
        ("data.stock_lab_", "forecast.stock_lab_"),
    ),
    ("data", "Data quality and vendor basis", ("data.",)),
    (
        "policy",
        "Policy limits and fallbacks",
        (
            "portfolio.violations",
            "portfolio.fallbacks",
            "portfolio.turnover",
            "portfolio.costs",
            "portfolio.concentration",
            "portfolio.drawdown",
        ),
    ),
    ("diagnostics", "Risk, turnover and drawdown diagnostics", ("portfolio.diagnostics",)),
)
OTHER = ("other", "Other checks")


def section_of(check_id: str) -> str:
    for key, _, names in SECTIONS:
        if any(check_id == name or check_id.startswith(name) for name in names):
            return key
    return OTHER[0]


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    state: HealthState
    state_word: str
    level: str
    summary: str
    checks: tuple[CheckView, ...]


@dataclass(frozen=True)
class Loaded:
    """One summary as the page shows it: missing, refused, or read."""

    path: Path | None
    view: HealthView | None = None
    sections: tuple[Section, ...] = ()
    refused_reason: str | None = None
    refused_detail: str | None = None
    generated_at: datetime | None = None

    @property
    def missing(self) -> bool:
        return self.view is None and self.refused_reason is None


@dataclass(frozen=True)
class HealthPage:
    local: Loaded
    ci: Loaded


def _summary_text(checks: list[CheckView]) -> str:
    """The worst check's own sentence and how many of the area's checks are healthy."""

    worst = min(checks, key=lambda check: CONDITION_PRECEDENCE.index(check.condition))
    healthy = sum(check.condition is Condition.HEALTHY for check in checks)
    noun = "check" if len(checks) == 1 else "checks"
    if healthy == len(checks):
        return f"All {len(checks)} {noun} healthy."
    return f"{worst.summary} {healthy} of {len(checks)} {noun} healthy."


def sections(summary: HealthSummary) -> tuple[Section, ...]:
    """The summary's checks grouped into the Lab's sections, worst check first in each."""

    grouped: dict[str, list[CheckView]] = {}
    for check in summary.checks:
        grouped.setdefault(section_of(check.check_id), []).append(check_view(check))
    titles = {key: title for key, title, _ in SECTIONS} | {OTHER[0]: OTHER[1]}
    result = []
    for key in [*(key for key, _, _ in SECTIONS), OTHER[0]]:
        checks = grouped.get(key)
        if not checks:
            continue
        ordered = sorted(checks, key=lambda check: CONDITION_PRECEDENCE.index(check.condition))
        worst = worst_condition(ordered)
        result.append(
            Section(
                key=key,
                title=titles[key],
                state=ordered[0].state,
                state_word=CONDITION_WORDS[worst],
                level=CONDITION_LEVELS[worst],
                summary=_summary_text(ordered),
                checks=tuple(ordered),
            )
        )
    return tuple(result)


def load_summary(path: Path | None, *, now: datetime) -> Loaded:
    if path is None:
        return Loaded(path)
    try:
        if not path.exists():  # a folder this account may not open raises here
            return Loaded(path)
        summary = read_summary(path)
    except SummaryRefused as error:
        return Loaded(path, refused_reason=error.reason.value, refused_detail=error.detail)
    except (OSError, RecursionError, ValueError) as error:  # unreadable: refused, never a crash
        return Loaded(path, refused_reason="unreadable", refused_detail=type(error).__name__)
    return Loaded(
        path,
        view=build_view(summary, now=now),
        sections=sections(summary),
        generated_at=summary.generated_at,
    )


def _path(environment: Mapping[str, str], variable: str) -> Path | None:
    value = environment.get(variable, "").strip()
    return Path(value).expanduser() if value else None


def load(
    environment: Mapping[str, str] | None = None,
    now: Callable[[], datetime] | None = None,
) -> HealthPage:
    """Both summaries, from the paths in the environment (unset: no summary yet)."""

    environment = os.environ if environment is None else environment
    stamp = (now or (lambda: datetime.now(UTC)))()
    return HealthPage(
        local=load_summary(_path(environment, LOCAL_VARIABLE), now=stamp),
        ci=load_summary(_path(environment, CI_VARIABLE), now=stamp),
    )


__all__ = [
    "CI_VARIABLE",
    "LOCAL_VARIABLE",
    "SECTIONS",
    "HealthPage",
    "Loaded",
    "Section",
    "load",
    "load_summary",
    "section_of",
    "sections",
]
