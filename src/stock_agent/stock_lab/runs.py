"""
Directional runs (streaks), with an explicit rule for Stable sessions.

A run is a block of consecutive sessions read as one direction. Two rules
for Stable sessions are offered, and the page always says which is in use;
nothing is redefined silently:

    "breaks" (the default)
        A Stable session ends a rising or falling run and starts a Stable
        run. Runs are maximal blocks of one category: in
        Increase Increase Stable Increase the rising runs are 2 and 1.

    "pauses"
        Up to `max_pause` consecutive Stable sessions (default 1) neither
        end nor extend a rising or falling run; a longer Stable stretch ends
        it. Only directional sessions are counted, so in
        Increase Increase Stable Increase the rising run is 3 (with one pause);
        in Increase Stable Stable Increase (max_pause 1) the runs are 1 and 1.
        Stable runs are still reported on their own.

A session without a category (its return is unavailable) ends every run.
The run that is still open at the last session, or that a gap ended, is
right-censored: it is known to have lasted at least its length, not how
long it lasted. Duration estimates must treat it that way (never as a
completed run).

Per session `current_runs` reports the run in progress at that session's
close, using only that session and earlier ones, so it is a past-only
feature: run_direction, run_length, stable_run (consecutive Stable sessions
ending at the session) and paused (Stable sessions inside the current
directional run, "pauses" rule only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import Category

BREAKS = "breaks"
PAUSES = "pauses"
POLICIES = (BREAKS, PAUSES)
POLICY_WORDS = {
    BREAKS: "A Stable session ends a rising or falling streak.",
    PAUSES: "Up to {max_pause} Stable session(s) in a row pause a streak without ending it.",
}

_UP, _STABLE, _DOWN = Category.INCREASE.value, Category.STABLE.value, Category.DECREASE.value
_DIRECTIONAL = (_UP, _DOWN)


@dataclass(frozen=True)
class RunRule:
    policy: str = BREAKS
    max_pause: int = 1

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if isinstance(self.max_pause, bool) or not isinstance(self.max_pause, int):
            raise TypeError("max_pause must be an integer")
        if not 1 <= self.max_pause <= 5:
            raise ValueError("max_pause must be between 1 and 5")

    def words(self) -> str:
        return POLICY_WORDS[self.policy].format(max_pause=self.max_pause)


DEFAULT_RULE = RunRule()


def current_runs(categories: pd.Series, rule: RunRule = DEFAULT_RULE) -> pd.DataFrame:
    """
    The run in progress at each session (module docstring), indexed like
    `categories` (values "increase", "stable", "decrease" or None).
    Columns: run_direction (None after a gap), run_length, stable_run, paused.
    """

    values = categories.to_numpy(dtype=object)
    n = len(values)
    direction = np.full(n, None, dtype=object)
    length = np.zeros(n, dtype="int64")
    stable_run = np.zeros(n, dtype="int64")
    paused = np.zeros(n, dtype="int64")

    run_dir, run_len, run_paused, stable_len = None, 0, 0, 0
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            run_dir, run_len, run_paused, stable_len = None, 0, 0, 0
            continue
        if value == _STABLE:
            stable_len += 1
            if rule.policy == PAUSES and run_dir in _DIRECTIONAL and stable_len <= rule.max_pause:
                run_paused += 1  # the directional run continues, paused
            else:
                run_dir, run_len, run_paused = _STABLE, stable_len, 0
        else:
            stable_len = 0
            if run_dir == value:
                run_len += 1
            else:
                run_dir, run_len, run_paused = value, 1, 0
        direction[i], length[i], stable_run[i], paused[i] = run_dir, run_len, stable_len, run_paused
    return pd.DataFrame(
        {
            "run_direction": direction,
            "run_length": length,
            "stable_run": stable_run,
            "paused": paused,
        },
        index=categories.index,
    )


def run_table(
    sessions: pd.Series | pd.DatetimeIndex, categories: pd.Series, rule: RunRule = DEFAULT_RULE
) -> pd.DataFrame:
    """
    Every run as one row: direction, start and end session, length (sessions
    counted by the rule), paused (Stable sessions inside it), and censored
    (True for a run open at the last session or ended by a gap).

    Under "pauses", Stable sessions that paused a run belong to it; a Stable
    stretch longer than `max_pause`, or one followed by the opposite
    direction, is a Stable run of its own.
    """

    dates = pd.DatetimeIndex(sessions)
    values = categories.to_numpy(dtype=object)
    if len(dates) != len(values):
        raise ValueError("sessions and categories must have the same length")
    rows: list[dict] = []
    run: dict | None = None  # the open run
    pending: list[int] = []  # Stable positions pausing an open directional run

    def emit(row: dict | None, censored: bool) -> None:
        if row is not None:
            rows.append({**row, "censored": censored})

    def stable_row(positions: list[int]) -> dict:
        return {
            "direction": _STABLE,
            "start": dates[positions[0]],
            "end": dates[positions[-1]],
            "length": len(positions),
            "paused": 0,
        }

    def flush(censored: bool) -> None:
        nonlocal run, pending
        emit(run, censored)
        if pending:
            emit(stable_row(pending), censored)
        run, pending = None, []

    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            flush(censored=True)
            continue
        directional_open = run is not None and run["direction"] in _DIRECTIONAL
        if value == _STABLE:
            if rule.policy == PAUSES and directional_open:
                pending.append(i)
                if len(pending) > rule.max_pause:
                    emit(run, censored=False)
                    run, pending = stable_row(pending), []
                continue
            if run is not None and run["direction"] == _STABLE:
                run["end"], run["length"] = dates[i], run["length"] + 1
                continue
            emit(run, censored=False)
            run = stable_row([i])
            continue
        if run is not None and run["direction"] == value:
            run["end"], run["length"] = dates[i], run["length"] + 1
            run["paused"] += len(pending)
            pending = []
            continue
        flush(censored=False)
        run = {"direction": value, "start": dates[i], "end": dates[i], "length": 1, "paused": 0}
    flush(censored=True)
    columns = ["direction", "start", "end", "length", "paused", "censored"]
    table = pd.DataFrame(rows, columns=columns)
    return table.astype({"length": "int64", "paused": "int64", "censored": bool})
