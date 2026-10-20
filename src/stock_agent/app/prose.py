"""
Markdown-safe writing for the Lab's pages (other than Explore, which keeps
its own helper): the one place these pages escape dollar signs, which
Streamlit would otherwise read in pairs as LaTeX.
"""

from __future__ import annotations

import re

import streamlit as st

from stock_agent.ui import components as ui


def escape(text: str) -> str:
    return text.replace("$", "\\$")


# Markdown's punctuation, escaped with a backslash (CommonMark), except "$", which `escape` takes.
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def literal(text: str) -> str:
    """
    Text read from a file, shown exactly as written: no markdown in it is
    rendered (no link, image or emphasis), for `say`, captions and labels.
    """

    return _MARKDOWN.sub(r"\\\1", text)


def say(text: str, as_: str = "markdown", where=None) -> None:
    """Write `text` as markdown, a caption, a subheader or a notice level, dollars escaped."""

    if as_ in ui.NOTICES:
        ui.notice(as_, escape(text), where=where)
    else:
        getattr(where or st, as_)(escape(text))


def section(title: str, caption: str | None = None) -> None:
    say(title, "subheader")
    if caption:
        say(caption, "caption")


def bullets(lines) -> None:
    say("\n".join(f"- {line}" for line in lines))
