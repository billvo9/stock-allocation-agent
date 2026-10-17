"""
Model Health: what the Model Health Center last measured, laid out for an
operator in the Lab's editorial style.

Progressive disclosure:
    1. the overall state in words, the headline and the one next step;
    2. each area (code and tests, training and evaluation runs, models and
       their evidence stage, the prospective forecast ledger, the Stock Lab
       daily job, data quality, policy limits and fallbacks, risk
       diagnostics) with its worst condition and one plain sentence;
    3. per area, every check's state and sentence, with the exact value,
       threshold, time, reason code, evidence and runbook link in an
       expander;
    4. informational alerts, checks that do not apply, and provenance;
    5. the latest daily GitHub workflow summary (synthetic fixtures only).

The page lays out what the summaries say (`health_summary`) and computes
nothing. With no summary it says "Not yet collected" and gives the command
that creates one; a refused or unreadable summary is shown as refused, with
nothing from it. A state is always written in words beside its colour, and a
check that was never measured is never shown as healthy: the summary reader
refuses such a check. The words follow each check's condition
(`health.view.CONDITION_WORDS`), so stale data ("Stale"), a job not
authorized yet ("Not activated"), missing evidence and a failure differ.

Text a summary carries (sentences, values, details, paths, alert messages)
is written by producers, so it never goes through markdown, which would
turn a URL into a link: it is rendered as escaped HTML (`status_list`,
`_text`, `_measurements`) or as a code block. Only the Lab's own copy and
the view's fixed phrases are markdown.
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape

import streamlit as st

from stock_agent.app import wording
from stock_agent.app.health_summary import HealthPage, Loaded, Section
from stock_agent.app.prose import say, section
from stock_agent.ui import components as ui

# The theme's tokens (ui.theme), so the measurement list matches the Lab's type.
LINK_STYLE = "color: var(--sa-accent)"
CODE_STYLE = "font-family: var(--sa-mono); font-size: 0.85em"


def _text(text: object, *, where=None) -> None:
    """One plain line of text as escaped HTML: never markdown, so never a link."""

    (where or st).html(f'<p class="sa-list__text">{escape(str(text))}</p>')


def _where(loaded: Loaded) -> str:
    if loaded.path is None:
        return wording.HEALTH_NOT_SET
    return wording.HEALTH_LOOKED.format(path=loaded.path)


def _refused(loaded: Loaded, label: str) -> None:
    ui.status_list(
        [
            ui.Item(
                "INVALID/BLOCKED",
                wording.HEALTH_REFUSED_WORD,
                wording.HEALTH_REFUSED_TITLE.format(label=label, reason=loaded.refused_reason),
                wording.HEALTH_REFUSED_TEXT.format(
                    path=loaded.path, detail=str(loaded.refused_detail).rstrip(".")
                ),
            )
        ]
    )


def _not_collected(loaded: Loaded) -> None:
    with ui.state_card("health-empty"):
        ui.pill("NOTICE", wording.HEALTH_EMPTY_BADGE)
        say(f"**{wording.HEALTH_EMPTY_TITLE}**")
        say(wording.HEALTH_EMPTY)
        _text(_where(loaded))
        say(wording.HEALTH_CREATE, "caption")
        st.code(wording.HEALTH_CREATE_COMMAND, language="bash", wrap_lines=True)
        say(wording.HEALTH_LATER, "caption")


def _status(loaded: Loaded) -> None:
    view = loaded.view
    with ui.card("health-status"):
        ui.pills([(view.level, view.state_word)])
        # The headline, counts, caveat, age warning and scope note are the view's
        # fixed phrases; the next step quotes a check's sentence, so it is escaped HTML.
        st.subheader(view.headline, anchor=False)
        st.markdown(view.summary)
        ui.status_list(
            [
                ui.Item(
                    view.level,
                    wording.HEALTH_NEXT_WORD,
                    wording.HEALTH_NEXT_TITLE,
                    view.next_action,
                )
            ]
        )
        if view.age_warning:
            ui.notice("warning", view.age_warning)
        ui.notice("notice", view.caveat)
        st.caption(view.scope_note)


def _glance(sections: tuple[Section, ...]) -> None:
    section(wording.HEALTH_GLANCE_TITLE, wording.HEALTH_GLANCE_CAPTION)
    with ui.card("health-glance"):
        ui.status_list(
            [ui.Item(item.level, item.state_word, item.title, item.summary) for item in sections]
        )


def _measurements(check) -> str:
    """One check's exact measurements as escaped HTML."""

    fields = wording.HEALTH_MEASURED_FIELDS
    # The only link is the runbook: the view's fixed base URL plus an anchor the
    # summary reader validated against a strict pattern.
    runbook = (
        f'<a href="{escape(check.runbook_url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer" style="{LINK_STYLE}">{escape(check.check_id)}</a>'
    )
    rows = (
        (fields["value"], escape(check.value)),
        (fields["threshold"], escape(check.threshold)),
        (fields["measured"], escape(check.measured)),
        (fields["reason"], f'<code style="{CODE_STYLE}">{escape(check.reason)}</code>'),
        (fields["check_id"], f'<code style="{CODE_STYLE}">{escape(check.check_id)}</code>'),
        (fields["evidence"], escape(check.evidence)),
        (fields["runbook"], runbook),
    )
    items = "".join(f"<li><strong>{escape(label)}:</strong> {value}</li>" for label, value in rows)
    return f'<p class="sa-list__title">{escape(check.title)}</p><ul>{items}</ul>'


def _area(item: Section) -> None:
    with ui.card(f"health-area-{item.key}"):
        ui.pill(item.level, item.state_word)
        st.markdown(f"#### {item.title}")  # the Lab's own area title
        ui.status_list(
            [
                ui.Item(check.level, check.state_word, check.title, check.summary)
                for check in item.checks
            ]
        )
        with st.expander(wording.HEALTH_MEASUREMENTS.format(title=item.title)):
            for check in item.checks:
                st.html(_measurements(check))
                st.code(check.detail, language=None, wrap_lines=True)


def _after(loaded: Loaded) -> None:
    view = loaded.view
    section(wording.HEALTH_ALERTS_TITLE, wording.HEALTH_ALERTS_CAPTION)
    if not view.alerts:
        say(wording.HEALTH_NO_ALERTS, "caption")
    if view.alerts:
        ui.status_list(
            [
                ui.Item(ui.NOTICES[alert.notice][2], alert.kind_word, alert.message)
                for alert in view.alerts
            ]
        )
    if view.not_applicable:
        section(wording.HEALTH_NOT_APPLICABLE_TITLE, wording.HEALTH_NOT_APPLICABLE_CAPTION)
        ui.status_list(
            [
                ui.Item("FUTURE", wording.HEALTH_NOT_APPLICABLE_WORD, line)
                for line in view.not_applicable
            ]
        )
    section(wording.HEALTH_PROVENANCE_TITLE)
    for line in view.provenance:
        _text(line)


def _daily(loaded: Loaded) -> None:
    section(wording.HEALTH_CI_TITLE, wording.HEALTH_CI_SCOPE)
    if loaded.refused_reason is not None:
        _refused(loaded, wording.HEALTH_CI_LABEL)
        return
    if loaded.missing:
        with ui.state_card("health-ci-empty"):
            ui.pill("NOTICE", wording.HEALTH_EMPTY_BADGE)
            say(wording.HEALTH_CI_MISSING)
            _text(_where(loaded))
            st.code(wording.HEALTH_CI_DOWNLOAD, language="bash", wrap_lines=True)
        return
    view = loaded.view
    generated = f"{loaded.generated_at:%Y-%m-%d %H:%M} UTC"
    with ui.card("health-ci"):
        ui.status_list(
            [
                ui.Item(
                    view.level,
                    view.state_word,
                    view.headline,
                    wording.HEALTH_CI_GENERATED.format(generated=generated, summary=view.summary),
                )
            ]
        )
        if view.age_warning:
            ui.notice("warning", view.age_warning)


def render(load: Callable[[], HealthPage] | None = None) -> None:
    ui.hero(
        wording.HEALTH_TITLE,
        eyebrow=wording.HEALTH_EYEBROW,
        lede=wording.HEALTH_LEDE,
        key="health",
    )
    page = None if load is None else load()
    local = None if page is None else page.local
    if local is None or local.missing:
        _not_collected(local or Loaded(None))
    elif local.refused_reason is not None:
        _refused(local, wording.HEALTH_LOCAL_LABEL)
    else:
        _status(local)
        if local.sections:
            _glance(local.sections)
            for item in local.sections:
                _area(item)
        _after(local)
    if page is not None:
        _daily(page.ci)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
