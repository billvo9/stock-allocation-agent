"""
Choices that survive a visit to another page. Streamlit drops a widget's
state when its page is not drawn, so each page keeps its own copy of its
widget values under one session key (`save`) and puts them back before
drawing its widgets (`restore`). Restored widgets take their value from
Session State only, never from a `default` argument as well.

A range slider is the exception: its `value` argument is what makes it a
range, so it is skipped by `restore` and given `value(...)` instead, which
is its live value, else its saved value, else the fallback. Streamlit 1.64
identifies a select slider by its key alone, so passing its own value back
never resets it.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


def restore(saved_key: str, *, skip: Sequence[str] = ()) -> None:
    """Put back saved widget values that Streamlit dropped while the page was away."""

    for key, saved in st.session_state.get(saved_key, {}).items():
        if key not in st.session_state and not key.startswith(tuple(skip)):
            st.session_state[key] = saved


def save(saved_key: str, prefixes: Sequence[str]) -> None:
    """Keep a copy of every widget value whose key starts with one of `prefixes`."""

    st.session_state[saved_key] = {
        key: st.session_state[key]
        for key in list(st.session_state)
        if isinstance(key, str) and key.startswith(tuple(prefixes))
    }


def value(saved_key: str, key: str, fallback):
    """A widget's live value, else its saved value, else `fallback`: pass it as `value=`."""

    if key in st.session_state:
        return st.session_state[key]
    return st.session_state.get(saved_key, {}).get(key, fallback)
