"""
Model Health: what the Model Health Center last measured, laid out for an
operator in the Lab's editorial style.

Progressive disclosure:
    1. the overall state in words, the headline and the one next step;
    2. each area (code and tests, training and evaluation runs, models and
       their evidence stage, the prospective forecast ledger, data quality,
       policy limits and fallbacks, risk diagnostics) with its worst state
       and one plain sentence;
    3. per area, every check's state and sentence, with the exact value,
       threshold, time, reason code, evidence and runbook link in an
       expander;
    4. informational alerts, checks that do not apply, and provenance;
    5. the latest daily GitHub workflow summary (synthetic fixtures only).

The page lays out what the summaries say (`health_summary`) and computes
nothing. With no summary it says "Not yet collected" and gives the command
that creates one; a refused summary is shown as refused, with nothing from
it. A state is always written in words beside its colour, and a check that
was never measured is never shown as healthy: the summary reader refuses
such a check. Text from a summary is shown literally (`prose.literal`).
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from stock_agent.app import wording
from stock_agent.app.health_summary import HealthPage, Loaded, Section
from stock_agent.app.prose import literal, say, section
from stock_agent.ui import components as ui


def _where(loaded: Loaded) -> str:
    if loaded.path is None:
        return wording.HEALTH_NOT_SET
    return wording.HEALTH_LOOKED.format(path=loaded.path)


def _refused(loaded: Loaded, label: str) -> None:
    ui.notice(
        "blocking",
        literal(
            wording.HEALTH_REFUSED.format(
                label=label,
                path=loaded.path,
                reason=loaded.refused_reason,
                detail=str(loaded.refused_detail).rstrip("."),
            )
        ),
    )


def _not_collected(loaded: Loaded) -> None:
    with ui.state_card("health-empty"):
        ui.pill("NOTICE", wording.HEALTH_EMPTY_BADGE)
        say(f"**{wording.HEALTH_EMPTY_TITLE}**")
        say(wording.HEALTH_EMPTY)
        st.caption(literal(_where(loaded)))
        say(wording.HEALTH_CREATE, "caption")
        st.code(wording.HEALTH_CREATE_COMMAND, language="bash", wrap_lines=True)
        say(wording.HEALTH_LATER, "caption")


def _status(loaded: Loaded) -> None:
    view = loaded.view
    with ui.card("health-status"):
        ui.pills([(view.level, view.state_word)])
        st.subheader(literal(view.headline), anchor=False)
        st.markdown(literal(view.summary))
        ui.notice(view.notice, literal(wording.HEALTH_NEXT.format(action=view.next_action)))
        if view.age_warning:
            ui.notice("warning", literal(view.age_warning))
        ui.notice("notice", literal(view.caveat))
        st.caption(literal(view.scope_note))


def _glance(sections: tuple[Section, ...]) -> None:
    section(wording.HEALTH_GLANCE_TITLE, wording.HEALTH_GLANCE_CAPTION)
    with ui.card("health-glance"):
        ui.status_list(
            [ui.Item(item.level, item.state_word, item.title, item.summary) for item in sections]
        )


def _measurements(check) -> str:
    fields = wording.HEALTH_MEASURED_FIELDS
    # Reason codes and check ids match strict patterns; the runbook URL is the
    # view's base plus a validated anchor.
    return (
        f"**{literal(check.title)}**\n\n"
        f"- **{fields['value']}:** {literal(check.value)}\n"
        f"- **{fields['threshold']}:** {literal(check.threshold)}\n"
        f"- **{fields['measured']}:** {literal(check.measured)}\n"
        f"- **{fields['reason']}:** `{check.reason}`\n"
        f"- **{fields['check_id']}:** `{check.check_id}`\n"
        f"- **{fields['evidence']}:** {literal(check.evidence)}\n"
        f"- **{fields['runbook']}:** [{check.check_id}]({check.runbook_url})"
    )


def _area(item: Section) -> None:
    with ui.card(f"health-area-{item.key}"):
        ui.pill(item.level, item.state_word)
        st.markdown(f"#### {literal(item.title)}")
        ui.status_list(
            [
                ui.Item(check.level, check.state_word, check.title, check.summary)
                for check in item.checks
            ]
        )
        with st.expander(wording.HEALTH_MEASUREMENTS.format(title=item.title)):
            for check in item.checks:
                st.markdown(_measurements(check))
                st.code(check.detail, language=None, wrap_lines=True)


def _after(loaded: Loaded) -> None:
    view = loaded.view
    section(wording.HEALTH_ALERTS_TITLE, wording.HEALTH_ALERTS_CAPTION)
    if not view.alerts:
        say(wording.HEALTH_NO_ALERTS, "caption")
    for alert in view.alerts:
        ui.notice(alert.notice, literal(f"{alert.kind_word}: {alert.message}"))
    if view.not_applicable:
        section(wording.HEALTH_NOT_APPLICABLE_TITLE, wording.HEALTH_NOT_APPLICABLE_CAPTION)
        st.markdown("\n".join(f"- {literal(line)}" for line in view.not_applicable))
    section(wording.HEALTH_PROVENANCE_TITLE)
    for line in view.provenance:
        st.caption(literal(line))


def _daily(loaded: Loaded) -> None:
    section(wording.HEALTH_CI_TITLE, wording.HEALTH_CI_SCOPE)
    if loaded.refused_reason is not None:
        _refused(loaded, wording.HEALTH_CI_LABEL)
        return
    if loaded.missing:
        with ui.state_card("health-ci-empty"):
            ui.pill("NOTICE", wording.HEALTH_EMPTY_BADGE)
            say(wording.HEALTH_CI_MISSING)
            st.caption(literal(_where(loaded)))
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
            ui.notice("warning", literal(view.age_warning))


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
