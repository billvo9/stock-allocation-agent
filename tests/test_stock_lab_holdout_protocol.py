"""
The sealed holdout's protocol (stock_lab.holdout.protocol): the registered
rules as the owner approved them on 2026-10-26, pinned by their hash, and
their sessions on the NYSE rules calendar. No network, no data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest
from stock_lab_holdout_helpers import short_protocol, weekdays

from stock_agent.calendars.nyse_rules import sessions
from stock_agent.stock_lab.holdout.protocol import (
    PROTOCOL,
    HoldoutRefused,
    Refusal,
    record_id,
)

# The registered protocol's hash, written before any protected price was fetched. A change
# to any rule changes it: this test then fails, and the change needs the owner's approval.
REGISTERED_SHA256 = "820d154e5d3a31019be692b809a9e0a953046ad8de4e38fcb2689c3fc603856c"


def test_the_registered_protocol_is_the_one_the_owner_approved():
    assert PROTOCOL.sha256() == REGISTERED_SHA256
    assert PROTOCOL.model_id == "2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7"
    assert PROTOCOL.tickers == ("AMD", "MRVL", "MU", "NVDA")
    assert (PROTOCOL.holdout_start, PROTOCOL.holdout_end) == ("2025-01-02", "2026-10-14")
    assert (PROTOCOL.comparator, PROTOCOL.deciding_score, PROTOCOL.deciding_target) == (
        "B0",
        "rps",
        "daily_pooled",
    )
    assert PROTOCOL.alpha == 0.025 and PROTOCOL.window_sessions == 252
    assert [name for name, _ in PROTOCOL.blocks] == [
        "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"alpha": 0.05},
        {"comparator": "B1"},
        {"holdout_end": "2026-10-15"},
        {"blocks": PROTOCOL.blocks[:-1]},
        {"tickers": ("AMD", "MU", "NVDA")},
    ],
)
def test_any_changed_rule_changes_the_hash(change):
    assert replace(PROTOCOL, **change).sha256() != REGISTERED_SHA256


def test_its_sessions_on_the_nyse_calendar():
    fetched = PROTOCOL.fetch_sessions(sessions)
    assert (fetched[0], fetched[-1], len(fetched)) == (
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2026-10-14", tz="UTC"),
        699,
    )
    decisions = PROTOCOL.decisions(sessions)
    assert len(decisions) == 443
    assert decisions[0] == pd.Timestamp("2024-12-31", tz="UTC")
    assert decisions[-1] == pd.Timestamp("2026-10-07", tz="UTC")
    # the first decision's window is the 252 sessions from the fetch's first
    assert len(fetched[fetched <= decisions[0]]) == 252
    # every target session lies in the holdout
    position = {day: i for i, day in enumerate(fetched)}
    targets = [fetched[position[d] + h] for d in decisions for h in range(1, 6)]
    assert min(targets) == pd.Timestamp("2025-01-02", tz="UTC")
    assert max(targets) == pd.Timestamp("2026-10-14", tz="UTC")


def test_quarterly_blocks_give_six_degrees_of_freedom():
    decisions = PROTOCOL.decisions(sessions)
    counts = Counter(PROTOCOL.block_labels(decisions))
    assert counts == {
        "2025Q1": 61, "2025Q2": 62, "2025Q3": 64, "2025Q4": 64,
        "2026Q1": 61, "2026Q2": 62, "2026Q3": 69,
    }  # fmt: skip
    assert len(counts) - 1 == 6
    # 2024-12-31 joins the first block, the October 2026 decisions the last
    labels = PROTOCOL.block_labels(pd.DatetimeIndex(["2024-12-31", "2026-10-07"], tz="UTC"))
    assert list(labels) == ["2025Q1", "2026Q3"]


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"fetch_start": "2024-01-03"}, "is not one window"),
        ({"holdout_start": "2025-01-03"}, "is not the session after"),
        ({"holdout_end": "2026-10-17"}, "is not a session"),  # a Saturday
        ({"blocks": (*PROTOCOL.blocks, ("2027Q1", "2027-01-01"))}, "a block holds no decision"),
    ],
)
def test_a_protocol_that_does_not_line_up_with_the_calendar_is_refused(change, words):
    with pytest.raises(HoldoutRefused) as refused:
        replace(PROTOCOL, **change).decisions(sessions)
    assert refused.value.reason is Refusal.PROTOCOL and words in refused.value.detail


def test_the_test_protocol_keeps_every_rule_but_the_dates():
    protocol = short_protocol("f" * 64)
    assert len(protocol.decisions(weekdays)) == 29
    kept = ("comparator", "deciding_score", "deciding_target", "alpha", "window_sessions")
    assert all(getattr(protocol, name) == getattr(PROTOCOL, name) for name in kept)


def test_a_record_id_covers_everything_but_itself():
    record = {"a": 1, "b": [0.5, None]}
    sealed = {**record, "record_id": record_id(record)}
    assert record_id(sealed) == sealed["record_id"]
    assert record_id({**sealed, "a": 2}) != sealed["record_id"]
