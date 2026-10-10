"""
Shared dashboard components. Every page, including T4's model pages, builds
from these instead of styling itself, so the design system (theme.py) stays
consistent:

    page_header     eyebrow, title, one-line purpose
    status_banner   the run status in words, colour-coded by the theme
    pill            a compact labelled status or scope tag
    stat_row        a row of metric cards with optional captions
    estimate_card   a stored estimate with its interval, p-value and status
    card            a keyed surface container
    section         a subsection title with an optional caption
    notice          blocking / warning / notice / success callouts
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

from stock_agent.dashboard import style, theme

# Notice levels, their words, icons and theme status.
NOTICES = {
    "blocking": ("Blocking", ":material/block:", "INVALID/BLOCKED"),
    "warning": ("Warning", ":material/warning:", "WARNING"),
    "notice": ("Note", ":material/info:", "NOTICE"),
    "success": ("OK", ":material/check_circle:", "VALID"),
}
_WRITERS = {"blocking": "error", "warning": "warning", "notice": "info", "success": "success"}


def apply_theme() -> None:
    """Inject the CSS layer once per script run (style-only: takes no space)."""

    st.html(f"<style>{theme.css()}</style>")


def page_header(title: str, *, eyebrow: str | None = None, purpose: str | None = None) -> None:
    if eyebrow:
        st.html(f'<div class="sa-eyebrow">{escape(eyebrow)}</div>')
    st.title(title)
    if purpose:
        st.caption(purpose)


def pill_html(level: str, text: str | None = None, *, live: bool = False) -> str:
    """HTML for a pill. `live` announces changes (use only for the run status)."""

    role = ' role="status"' if live else ""
    return (
        f'<span class="sa-pill sa-pill--{theme.status_slug(level)}"{role}>'
        f"{escape(text or level)}</span>"
    )


def pill(level: str, text: str | None = None, *, where=None, live: bool = False) -> None:
    (where or st).html(pill_html(level, text, live=live))


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
) -> None:
    """
    One stored metric row in a card: the estimate, then its stored interval,
    p-value and status, so uncertainty is read with the number. A missing or
    unavailable row shows a dash and its stored reason.
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
            details.append(
                f"{row['ci_level']:.0%} interval [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            )
        if not pd.isna(row["p_value"]):
            details.append(f"p = {row['p_value']:.3f} vs {row['null_value']:g}")
        if row["status"] == "warning":
            details.append(f"warning: {row['reason']}")
        if details:
            st.caption("; ".join(details))


def card(key: str):
    """A surface container; use as `with card("name"):`."""

    return st.container(key=f"sa-card-{key}")


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def notice(level: str, text: str, *, where=None) -> None:
    """A callout whose severity is spelled out and marked by an icon."""

    word, icon, _ = NOTICES[level]
    getattr(where or st, _WRITERS[level])(f"**{word}.** {text}", icon=icon)


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
