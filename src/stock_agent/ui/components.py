"""
Shared UI components for the research dashboard and the product app. Every
page builds from these instead of styling itself, so the design system
(theme.py) stays consistent:

    page_header     eyebrow, title, one-line purpose
    hero            an editorial page opening: eyebrow, serif title, lede
    lede            a short serif paragraph that says what a page is for
    status_list     rows of (status in words, title, sentence): what needs attention
    page_links      a row of links to other pages: a page's next actions
    status_banner   the run status in words, colour-coded by the theme
    pill            a compact labelled status or scope tag
    pills           several pills on one wrapping line (a page's badges)
    stat_row        a row of metric cards with optional captions
    estimate_card   a stored estimate with its interval, p-value and status
    card            a keyed surface container
    state_card      a dashed container for loading, pending or unavailable states
    section         a subsection title with an optional caption
    notice          blocking / warning / notice / success callouts
    colour_scheme   the viewer's active colour scheme, for the Lab's figures
    chart           a Plotly figure with the shared config
    download        a CSV download of stored rows
    future_state    a clearly disabled placeholder for work not yet defined

Severity is always written in words (and marked by an icon), never carried
by colour alone. Components lay out values; they compute nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st

from stock_agent.ui import style, theme

# Notice levels, their words, icons and theme status.
NOTICES = {
    "blocking": ("Blocking", ":material/block:", "INVALID/BLOCKED"),
    "warning": ("Warning", ":material/warning:", "WARNING"),
    "notice": ("Note", ":material/info:", "NOTICE"),
    "success": ("OK", ":material/check_circle:", "VALID"),
}
_WRITERS = {"blocking": "error", "warning": "warning", "notice": "info", "success": "success"}


# CSS layers by variant: the research dashboard's, and the Portfolio Lab's
# editorial layer (theme.py).
THEMES = {"research": theme.css, "lab": theme.lab_css}


def apply_theme(variant: str = "research") -> None:
    """Inject a CSS layer once per script run (style-only: takes no space)."""

    st.html(f"<style>{THEMES[variant]()}</style>")


def page_header(title: str, *, eyebrow: str | None = None, purpose: str | None = None) -> None:
    if eyebrow:
        st.html(f'<div class="sa-eyebrow">{escape(eyebrow)}</div>')
    st.title(title)
    if purpose:
        st.caption(purpose)


def hero(
    title: str, *, eyebrow: str | None = None, lede: str | None = None, key: str = "page"
) -> None:
    """An editorial page opening in a keyed container: eyebrow, title and lede."""

    with st.container(key=f"sa-hero-{key}"):
        if eyebrow:
            st.html(f'<div class="sa-eyebrow">{escape(eyebrow)}</div>')
        st.title(title, anchor=False)
        if lede:
            st.html(_lede_html(lede))


def _lede_html(text: str) -> str:
    return f'<p class="sa-lede">{escape(text)}</p>'


def lede(text: str, *, where=None) -> None:
    """A short serif paragraph (plain text, escaped) saying what a page is for."""

    (where or st).html(_lede_html(text))


@dataclass(frozen=True)
class Item:
    """One row of a status list: a status level, its word, a title and a sentence."""

    level: str
    word: str
    title: str
    text: str = ""


def status_list(items: Sequence[Item], *, where=None) -> None:
    """
    Rows that each lead with a status written in words (a pill), then a
    title and one plain sentence. Plain text only: everything is escaped.
    """

    rows = []
    for item in items:
        text = f'<p class="sa-list__text">{escape(item.text)}</p>' if item.text else ""
        rows.append(
            f'<li class="sa-list__item">{pill_html(item.level, item.word)}'
            f'<div><p class="sa-list__title">{escape(item.title)}</p>{text}</div></li>'
        )
    (where or st).html(f'<ul class="sa-list">{"".join(rows)}</ul>')


def page_links(links: Sequence[tuple[object, str]], *, key: str) -> None:
    """A wrapping row of links to other pages, each (page, label): a page's next actions."""

    with st.container(key=f"sa-actions-{key}", horizontal=True, gap="small"):
        for page, label in links:
            st.page_link(page, label=label)


def pill_html(level: str, text: str | None = None, *, live: bool = False) -> str:
    """HTML for a pill. `live` announces changes (use only for the run status)."""

    role = ' role="status"' if live else ""
    return (
        f'<span class="sa-pill sa-pill--{theme.status_slug(level)}"{role}>'
        f"{escape(text or level)}</span>"
    )


def pill(level: str, text: str | None = None, *, where=None, live: bool = False) -> None:
    (where or st).html(pill_html(level, text, live=live))


def pills(items: Sequence[tuple[str, str]], *, where=None) -> None:
    """Several (level, text) pills on one wrapping line, for a page's badges."""

    body = "".join(pill_html(level, text) for level, text in items)
    (where or st).html(f'<div class="sa-pills">{body}</div>')


def status_banner(level: str, reasons: Sequence[str]) -> None:
    message = f"**{level}**" + (": " + "; ".join(reasons) if reasons else "")
    {"VALID": st.success, "WARNING": st.warning}.get(level, st.error)(message)


@dataclass(frozen=True)
class Stat:
    label: str
    value: object
    caption: str | None = None
    help: str | None = None


def stat_row(stats: Sequence[Stat]) -> None:
    for column, stat in zip(st.columns(len(stats)), stats, strict=True):
        column.metric(stat.label, stat.value, help=stat.help)
        if stat.caption:
            column.caption(stat.caption)


def estimate_card(
    where,
    label: str,
    row: pd.Series | None,
    *,
    fmt: str = "{:.3f}",
    help: str | None = None,
    method: str | None = None,
    note: str | None = None,
) -> None:
    """
    One stored metric row in a card: the estimate, then its stored interval
    (in the same format), p-value and status, so uncertainty is read with the
    number. A missing or unavailable row shows a dash and its stored reason.
    `note` adds a caption under an available estimate (e.g. its coverage).
    """

    digest = hashlib.sha1(f"{label}|{method}".encode()).hexdigest()[:12]
    with where.container(key=f"sa-card-estimate-{digest}"):
        if method:
            st.caption(method)
        if row is None:
            st.metric(label, "—", help=help)
            st.caption("not stored for this model")
            return
        if row["status"] == "unavailable":
            st.metric(label, "—", help=help)
            st.caption(f"unavailable: {row['reason']}")
            return
        st.metric(label, fmt.format(row["estimate"]), help=help)
        details = []
        if not pd.isna(row["ci_low"]):
            low, high = fmt.format(row["ci_low"]), fmt.format(row["ci_high"])
            details.append(f"{row['ci_level']:.0%} interval [{low}, {high}]")
        if not pd.isna(row["p_value"]):
            details.append(f"p = {row['p_value']:.3f} vs {row['null_value']:g}")
        if row["status"] == "warning":
            details.append(f"warning: {row['reason']}")
        if details:
            st.caption("; ".join(details))
        if note:
            st.caption(note)


def card(key: str):
    """A surface container; use as `with card("name"):`."""

    return st.container(key=f"sa-card-{key}")


def state_card(key: str):
    """
    A dashed container for a state that is not a value (loading, pending,
    unavailable), so it never reads as a number; use as `with state_card(...)`.
    """

    return st.container(key=f"sa-future-{key}")


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def notice(level: str, text: str, *, where=None) -> None:
    """A callout whose severity is spelled out and marked by an icon."""

    word, icon, _ = NOTICES[level]
    getattr(where or st, _WRITERS[level])(f"**{word}.** {text}", icon=icon)


def colour_scheme() -> str:
    """
    The viewer's active colour scheme, "light" or "dark", for figures drawn
    in the Lab's tokens (`style.lab_layout`); "light" when Streamlit does not
    report one (outside a browser session). It is the scheme of the last
    rerun, so after a change in Settings, charts follow on the next
    interaction (Streamlit issue #11920); the CSS layer follows at once.
    """

    reported = st.context.theme.type
    return reported if reported in theme.SCHEMES else "light"


def chart(figure, key: str) -> None:
    st.plotly_chart(figure, theme=None, key=key, config=style.PLOTLY_CONFIG)


def download(frame: pd.DataFrame, name: str, run_id: str) -> None:
    st.download_button(
        f"Download {name} (CSV)",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_{name}.csv",
        mime="text/csv",
        key=f"download_{name}",
    )


def future_state(key: str, title: str, reason: str) -> None:
    """A disabled placeholder: says what is missing and which ticket defines it."""

    with st.container(key=f"sa-future-{key}"):
        pill("FUTURE", "Not available yet")
        st.markdown(f"**{title}**")
        st.caption(reason)
        st.button(title, key=f"future_{key}", disabled=True, help=reason)
