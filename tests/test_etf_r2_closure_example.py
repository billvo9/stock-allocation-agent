"""
The numerical example of proposed rule ETF-R2 (unscheduled closures;
docs/research/prospective_data_path.md, section 1), reproduced date for date.

ETF-R2 is a design pending the owner, so nothing in `src` implements it: the
amended calendar below is built here, in the test, by removing the
hypothetical closure from the registered rules calendar. Session n is the
n-th session from session 1; a forecast at as-of s enters at s + 1 and exits
at entry + h sessions (`entry_lag: 1` in configs/prospective/etf_prospective_v1.yaml).
"""

import pandas as pd
import pytest

from stock_agent.calendars import nyse_rules

SESSION_1 = pd.Timestamp("2026-11-02", tz="UTC")  # a hypothetical first eligible close
CLOSURE = pd.Timestamp("2026-11-18", tz="UTC")  # hypothetical, announced 2026-11-13
ENTRY_LAG = 1
LOOKBACK_CLOSES = 64  # 63 daily returns
TRIAL_SESSIONS = 20
C12, C24 = 272, 524

REGISTERED = nyse_rules.calendar(nyse_rules.CALENDAR_ID_2030).sessions("2026-01-02", "2030-12-31")
AMENDED = REGISTERED[REGISTERED != CLOSURE]


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def numbered(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sessions from session 1 on: position n - 1 holds session n."""

    return calendar[calendar >= SESSION_1]


def window(calendar: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    end = calendar.get_loc(as_of)
    return calendar[end - LOOKBACK_CLOSES + 1 : end + 1]


def entry_and_exit(calendar: pd.DatetimeIndex, as_of: pd.Timestamp, horizon: int):
    position = calendar.get_loc(as_of)
    return calendar[position + ENTRY_LAG], calendar[position + ENTRY_LAG + horizon]


def text(days) -> list[str]:
    return [f"{d:%Y-%m-%d}" for d in days]


def test_the_hypothetical_closure_is_a_session_of_every_registered_calendar():
    # The example changes no code: the registered calendars still trade on 2026-11-18.
    for calendar_id in nyse_rules.CALENDARS:
        assert nyse_rules.calendar(calendar_id).is_session(CLOSURE)
    assert nyse_rules.is_session(CLOSURE)
    v2027 = nyse_rules.calendar(nyse_rules.CALENDAR_ID_2027).sessions("2026-01-02", "2027-12-31")
    assert v2027.equals(REGISTERED[REGISTERED <= day("2027-12-31")])
    assert len(AMENDED) == len(REGISTERED) - 1


def test_session_numbers_before_and_after_the_amendment():
    without, with_r2 = numbered(REGISTERED), numbered(AMENDED)
    assert text(without[:12]) == text(with_r2[:12])  # sessions 1-12 never move
    assert without[0] == with_r2[0] == SESSION_1
    assert without[9] == with_r2[9] == day("2026-11-13")  # session 10: the announcement
    assert without[12] == CLOSURE  # session 13 today, and expected
    assert CLOSURE not in with_r2  # not a session: nothing expected, recorded or priced
    assert text(without[12:20]) == [
        "2026-11-18",
        "2026-11-19",
        "2026-11-20",
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
        "2026-11-27",
        "2026-11-30",
    ]
    assert text(with_r2[12:20]) == [
        "2026-11-19",
        "2026-11-20",
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
        "2026-11-27",
        "2026-11-30",
        "2026-12-01",
    ]
    # Every session after the closure is numbered one lower; numbers never skip.
    for n in range(13, C24):
        assert with_r2[n - 1] == without[n]


@pytest.mark.parametrize(
    ("label", "number", "without", "with_r2"),
    [
        ("last trial session", TRIAL_SESSIONS, "2026-11-30", "2026-12-01"),
        ("session 21", TRIAL_SESSIONS + 1, "2026-12-01", "2026-12-02"),
        ("C12", C12, "2027-12-01", "2027-12-02"),
        ("C24", C24, "2028-11-30", "2028-12-01"),
    ],
)
def test_the_trial_and_checkpoints_keep_their_numbers_and_move_one_session(
    label, number, without, with_r2
):
    assert numbered(REGISTERED)[number - 1] == day(without), label
    assert numbered(AMENDED)[number - 1] == day(with_r2), label


def test_the_next_forecasts_window_is_64_actual_closes():
    as_of = day("2026-11-19")
    assert numbered(REGISTERED).get_loc(as_of) + 1 == 14
    assert numbered(AMENDED).get_loc(as_of) + 1 == 13
    today, amended = window(REGISTERED, as_of), window(AMENDED, as_of)
    assert len(today) == len(amended) == LOOKBACK_CLOSES
    # Today: the window holds the closed day, which has no close, so it is incomplete.
    assert (today[0], today[-1]) == (day("2026-08-21"), as_of) and CLOSURE in today
    # ETF-R2: 64 actual closes, reaching one session further back; no price is invented.
    assert (amended[0], amended[-1]) == (day("2026-08-20"), as_of) and CLOSURE not in amended
    assert amended[1:].equals(today[today != CLOSURE])


@pytest.mark.parametrize(
    ("as_of", "entries", "exits"),
    [
        (
            "2026-11-12",  # session 9, recorded before the announcement
            ("2026-11-13", "2026-11-13"),
            {
                5: ("2026-11-20", "2026-11-23"),
                20: ("2026-12-14", "2026-12-15"),
                30: ("2026-12-29", "2026-12-30"),
            },
        ),
        (
            "2026-11-17",  # session 12, recorded after the amendment
            ("2026-11-18", "2026-11-19"),
            {
                5: ("2026-11-25", "2026-11-27"),
                20: ("2026-12-17", "2026-12-18"),
                30: ("2027-01-04", "2027-01-05"),
            },
        ),
    ],
)
def test_open_forecasts_exit_h_trading_sessions_after_entry(as_of, entries, exits):
    for horizon, (without, with_r2) in exits.items():
        entry, exit_ = entry_and_exit(REGISTERED, day(as_of), horizon)
        assert (entry, exit_) == (day(entries[0]), day(without))
        entry, exit_ = entry_and_exit(AMENDED, day(as_of), horizon)
        assert (entry, exit_) == (day(entries[1]), day(with_r2))
        # The amended exit is still exactly h sessions after entry, none of them the closure.
        span = AMENDED[(AMENDED > entry) & (AMENDED <= exit_)]
        assert len(span) == horizon and CLOSURE not in span
    # 2026-11-27 is a published early close: a session in both calendars.
    assert day("2026-11-27") in AMENDED and day("2026-11-27") in REGISTERED
