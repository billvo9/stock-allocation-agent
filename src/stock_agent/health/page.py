"""
The Model Health page: layout only, from `view.build_view` and the shared
components (`stock_agent.ui.components`). It reads summary files with the
strict reader and computes nothing.

Each source (the latest CI summary, the local monitor's summary) gets a tab.
A missing summary is a first-class state with the exact command that creates
it; a refused summary says why and shows nothing from it. Never a zero or an
empty chart in place of a missing value.

`render(sources)` is the integration point for the Portfolio Lab's Model
Health destination; `scripts/run_model_health.py` runs it on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from stock_agent.health.model import HealthSummary, SummaryRefused, read_summary
from stock_agent.health.view import HealthView, build_view
from stock_agent.ui import components as ui

TITLE = "Model health"
EYEBROW = "Operations"
PURPOSE = (
    "Whether the code, the data contracts, the data and the models can be trusted today, in "
    "plain words first and exact measurements underneath. Read-only."
)


STATE_COLUMNS = (0.2, 0.8)  # room for the longest pill, "Insufficient evidence"
CI_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_CI_SUMMARY"
LOCAL_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_LOCAL_SUMMARY"
CREATE_CI = (
    "RUN=$(gh run list --repo billvo9/stock-allocation-agent --workflow health.yml "
    "--limit 1 --json databaseId --jq '.[0].databaseId')\n"
    'gh run download "$RUN" --repo billvo9/stock-allocation-agent --name model-health '
    '--dir "reports/health/ci/$RUN"'
)
CREATE_LOCAL = ".venv/bin/python scripts/health_check.py local --run-verify"


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    path: Path | None
    create_with: str


def sources_from_environment(environment: Mapping[str, str]) -> list[Source]:
    """The two standard sources, with paths from the environment variables."""

    def path(variable: str) -> Path | None:
        value = environment.get(variable)
        return Path(value) if value else None

    return [
        Source("ci", "Latest CI run", path(CI_SUMMARY_VARIABLE), CREATE_CI),
        Source("local", "Local monitor", path(LOCAL_SUMMARY_VARIABLE), CREATE_LOCAL),
    ]


# Every ASCII punctuation character markdown can give meaning to.
_MARKDOWN = re.compile(r"""([\\`*_{}\[\]()#+\-.!|<>~$&:/=^%@?'"])""")


def _esc(text: object) -> str:
    """
    Text from a summary (much of it written by producers) as literal
    characters: every markdown-significant character is backslash-escaped,
    so no text can form a link, an image, emphasis, HTML or LaTeX. Only this
    page's own templates add markdown.
    """

    return _MARKDOWN.sub(r"\\\1", str(text))


def _load(source: Source) -> HealthSummary | SummaryRefused | None:
    if source.path is None or not source.path.exists():
        return None
    try:
        return read_summary(source.path)
    except SummaryRefused as error:
        return error


def _missing(source: Source) -> None:
    with ui.state_card(f"health-{source.key}-missing"):
        ui.pill("NOTICE", "No summary yet")
        where = f" at {_esc(source.path)}" if source.path else ""
        st.markdown(f"No {_esc(source.label.lower())} health summary was found{where}.")
        st.caption("Create one with:")
        st.code(source.create_with, language="bash")


def _stated(level: str, state_word: str, markdown: str) -> None:
    """A state pill beside its (already escaped) markdown, stacked on narrow screens."""

    pill, words = st.columns(STATE_COLUMNS, vertical_alignment="center")
    ui.pill(level, state_word, where=pill)
    words.markdown(markdown)


def _render_view(source: Source, view: HealthView) -> None:
    ui.pills([(view.level, view.state_word)])
    st.subheader(_esc(view.headline))
    st.markdown(_esc(view.summary))
    ui.notice(view.notice, f"Next action: {_esc(view.next_action)}")
    if view.age_warning:
        ui.notice("warning", _esc(view.age_warning))
    ui.notice("notice", _esc(view.caveat))
    st.caption(_esc(view.scope_note))

    ui.stat_row([ui.Stat(fact.label, fact.value, _esc(fact.caption)) for fact in view.facts])

    ui.section("Alerts", "Informational only: nothing acts on an alert.")
    if not view.alerts:
        st.caption("No alerts.")
    for alert in view.alerts:
        ui.notice(alert.notice, f"{_esc(alert.kind_word)}: {_esc(alert.message)}")

    ui.section("Checks by area", "Plain summaries first; open a check for its measurements.")
    for area in view.areas:
        with ui.card(f"health-{source.key}-{area.area.value}"):
            _stated(area.level, area.state_word, f"#### {_esc(area.title)}\n{_esc(area.counts)}")
            for check in area.checks:
                _stated(
                    check.level,
                    check.state_word,
                    f"**{_esc(check.title)}**: {_esc(check.summary)}",
                )
                with st.expander(f"Measurements: {_esc(check.title)}"):
                    # Reason codes and check ids match strict patterns; the runbook
                    # URL is this page's base plus a validated anchor.
                    st.markdown(
                        f"- **Value:** {_esc(check.value)}\n"
                        f"- **Threshold:** {_esc(check.threshold)}\n"
                        f"- **Measured:** {_esc(check.measured)}\n"
                        f"- **Reason code:** `{check.reason}`\n"
                        f"- **Check id:** `{check.check_id}`\n"
                        f"- **Evidence:** {_esc(check.evidence)}\n"
                        f"- **Runbook:** [{check.check_id}]({check.runbook_url})"
                    )
                    st.code(check.detail, language=None, wrap_lines=True)

    if view.not_applicable:
        ui.section(
            "Not applicable",
            "Checks that do not apply to what was measured; they never count towards the state.",
        )
        for line in view.not_applicable:
            st.markdown(f"- {_esc(line)}")

    ui.section("Provenance")
    for line in view.provenance:
        st.caption(_esc(line))


def render_source(source: Source, *, now: datetime) -> None:
    loaded = _load(source)
    if loaded is None:
        _missing(source)
    elif isinstance(loaded, SummaryRefused):
        ui.notice(
            "blocking",
            f"The summary at {_esc(source.path)} was refused ({_esc(loaded.reason.value)}): "
            f"{_esc(loaded.detail)}. Nothing from it is shown.",
        )
    else:
        _render_view(source, build_view(loaded, now=now))


def render(sources: Sequence[Source], *, now: Callable[[], datetime] | None = None) -> None:
    ui.page_header(TITLE, eyebrow=EYEBROW, purpose=PURPOSE)
    clock = now or (lambda: datetime.now(UTC))
    for tab, source in zip(st.tabs([source.label for source in sources]), sources, strict=True):
        with tab:
            render_source(source, now=clock())
