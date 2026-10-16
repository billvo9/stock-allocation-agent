"""
Markdown-safe writing for the Lab's pages (other than Explore, which keeps
its own helper): the one place these pages escape dollar signs, which
Streamlit would otherwise read in pairs as LaTeX. `literal` escapes every
markdown-significant character, for text the Lab did not write (a health
summary's sentences, written by producers).
"""

from __future__ import annotations

import re

import streamlit as st

from stock_agent.ui import components as ui

# Every ASCII punctuation character markdown can give meaning to.
_MARKDOWN = re.compile(r"""([\\`*_{}\[\]()#+\-.!|<>~$&:/=^%@?'"])""")


def escape(text: str) -> str:
    return text.replace("$", "\\$")


def literal(text: object) -> str:
    """
    Text shown exactly as written: every markdown-significant character is
    backslash-escaped, so no link, image, emphasis, HTML or LaTeX can form.
    """

    return _MARKDOWN.sub(r"\\\1", str(text))


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
