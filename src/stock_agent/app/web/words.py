"""
The Lab app's words: number formats, dates and times in plain English.

Times are New York time, the Stock Lab's clock (the evening job, the close and
the open). " ET" is added only when this Mac's own zone is elsewhere, so a
reader in New York never sees a label they do not need. Percentages are
rounded for display only; the value itself is never changed.
"""

from __future__ import annotations

import math
from datetime import date, datetime, tzinfo
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
DASH = "—"
MINUS = "−"
EVENING_FROM = 17  # from 5 PM, a time today is "tonight" and one yesterday "last night"
RECENT_DAYS = 6  # within this many days, a time is named by its weekday

# Company names are not in the stored data; these are the watchlist's.
NAMES = {
    "AMD": "Advanced Micro Devices",
    "MRVL": "Marvell Technology",
    "MU": "Micron Technology",
    "NVDA": "NVIDIA",
}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def percent(p: float | None) -> str:
    """
    A probability as a whole percentage, for display: 0.4712 -> '47%'. A chance that
    rounds to 0% or 100% without being exactly that reads '<1%' or '>99%', so the
    app never shows a certainty that was not recorded.
    """

    if not _finite(p):
        return DASH
    shown = f"{p:.0%}"
    if shown == "0%" and p > 0:
        return "<1%"
    if shown == "100%" and p < 1:
        return ">99%"
    return shown


def signed_percent(r: float | None) -> str:
    """A price move with its sign: '+4.88%', '−0.32%', '0.00%'."""

    if not _finite(r):
        return DASH
    sign = "+" if r > 0 else MINUS if r < 0 else ""
    return f"{sign}{abs(r):.2%}"


def money(value: float | None) -> str:
    return f"${value:,.2f}" if _finite(value) else DASH


def _day(value: date | str) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


def short_day(value: date | str) -> str:
    """'Wed, Oct 21'."""

    day = _day(value)
    return f"{day:%a, %b} {day.day}"


def weekday(value: date | str) -> str:
    """'Wed'."""

    return f"{_day(value):%a}"


def long_weekday(value: date | str) -> str:
    """'Wednesday'."""

    return f"{_day(value):%A}"


def clock(moment: datetime) -> str:
    """'8:31 PM'."""

    return f"{moment:%I:%M %p}".lstrip("0")


def when(stamp: str, *, now: datetime, local_zone: tzinfo | None) -> str:
    """
    A stored UTC stamp as a reader would say it, in New York time: 'last night,
    8:31 PM', 'tonight, 8:31 PM', 'this morning, 6:02 AM', 'Friday, 8:31 PM' or
    'Oct 9, 8:31 PM'; with ' ET' when this Mac's zone is not New York's
    (`local_zone` None: this Mac's own zone, with the offset in force at `now`).
    """

    moment = pd.Timestamp(stamp).tz_convert(NEW_YORK).to_pydatetime()
    today = now.astimezone(NEW_YORK).date()
    days = (today - moment.date()).days
    evening = moment.hour >= EVENING_FROM
    if days == 0:
        lead = "tonight" if evening else "this morning" if moment.hour < 12 else "today"
    elif days == 1:
        lead = "last night" if evening else "yesterday"
    elif 1 < days <= RECENT_DAYS:
        lead = long_weekday(moment.date())
    else:
        lead = f"{moment:%b} {moment.day}"
    text = f"{lead}, {clock(moment)}"
    return text + (" ET" if _elsewhere(now, local_zone) else "")


def _elsewhere(now: datetime, local_zone: tzinfo | None) -> bool:
    """Whether this Mac's clock reads differently from New York's at `now`."""

    return now.astimezone(local_zone).utcoffset() != now.astimezone(NEW_YORK).utcoffset()


# --- Home: the status line ---

RECORD_UNREADABLE = "The forecast record can't be read."
LEDGER_FAILED = "The forecast record didn't pass its integrity check."
NOTHING_CHANGED = "No unverified number is shown, and nothing was changed."
TECHNICAL = "Technical detail: {detail}"
UPDATES_TITLE = "Updates"
SHEET_COLUMNS = ["What could happen", "Chance"]
LINES_NOTE = "Lines show the past month; the dot's colour shows whether that day closed up or down."


def record_failed(symbols: list[str]) -> str:
    """'NVDA's forecast record didn't pass its check.' (one stock, or several)."""

    if len(symbols) == 1:
        return f"{symbols[0]}'s forecast record didn't pass its integrity check."
    return f"{len(symbols)} stocks' forecast records didn't pass their integrity check."


def checked(now: datetime, local_zone: tzinfo | None) -> str:
    """When the app last read the records: 'Checked 7:40 AM'."""

    text = f"Checked {clock(now.astimezone(NEW_YORK))}"
    return text + (" ET" if _elsewhere(now, local_zone) else "")


NOT_STARTED = "Outlooks haven't started yet."
NO_RUN = "No update has run yet."
NO_SUCCESS = "No update has finished yet."
RUN_PROBLEM = {
    "failed": "The last update didn't finish ({when}).",
    "partial": "The last update was only partly done ({when}).",
    "skipped": "The last update recorded nothing ({when}).",
}
RUN_PROBLEM_OTHER = "The last update ended as {status} ({when})."
# The daily job's run statuses, in words, for the update details.
RUN_WORDS = {
    "ok": "finished",
    "partial": "only partly done",
    "failed": "didn't finish",
    "skipped": "recorded nothing",
}


def updated(when_words: str) -> str:
    return f"Updated {when_words}"


def evening_name(prior: date, now: datetime) -> str:
    """Whose update recorded the outlooks for the session after `prior`: 'Last night's'."""

    days = (now.astimezone(NEW_YORK).date() - prior).days
    if days == 0:
        return "Tonight's"
    if days == 1:
        return "Last night's"
    return f"{long_weekday(prior)}'s"


def missed(evening: str, *, ran: bool, missing: int, total: int) -> str:
    if not ran:
        return f"{evening} update didn't run."
    return f"{evening} update didn't record {missing} of {total} stocks."


def withheld(count: int) -> str:
    return f"{count} outlook was withheld." if count == 1 else f"{count} outlooks were withheld."


# --- Home: the outlook chip and its sheet ---

PENDING_CHIP = "{day} outlook after tonight's update"
STALE_CHIP = "{day} outlook not recorded"
NONE_CHIP = "No outlook yet"
REFUSED_CHIP = "No outlook: the model gave none"
UNVERIFIED_CHIP = "Outlook can't be verified"
WITHHELD_CHIP = "Outlook withheld"
WHOLE_NUMBERS = "Shown as whole numbers, so they may not add up to exactly 100%."
EXPERIMENTAL = "Experimental: not yet proven, and not advice."

NO_PRICES = {
    "no_store": "No prices stored yet.",
    "no_data": "No prices stored for {ticker} yet.",
}
PRICES_UNREADABLE = "Prices can't be read right now."
NO_CLOSE = "No closing price stored yet."

FOOTER = (
    "Outlooks are experimental chances from a simple model, recorded after each trading "
    "day's close. They are not advice, and nothing here can be bought or sold."
)
EXPLAIN_TITLE = "What do the numbers mean?"
DEMO = "Synthetic demo: made-up stocks and numbers, to try the app safely."


def prices_note(session: str) -> str:
    return f"Closing prices from {short_day(session)}."


def open_label(ticker: str, name: str | None, price: dict) -> str:
    """A watchlist row's link, for a screen reader: 'NVDA, NVIDIA: $127.65, ... Open details.'"""

    who = f"{ticker}, {name}" if name else ticker
    if not price["available"]:
        return f"{who}. {price['words']} Open details."
    return (
        f"{who}: {price['close_words']}, {price['change_words']} on {price['session_words']}. "
        "Open details."
    )


# --- Stock detail: the chart ---

# (key, label, the moves a range shows, its name): a range of n moves draws n + 1 closes.
RANGES = (
    ("1W", "1W", 5, "1W, one week"),
    ("1M", "1M", 21, "1M, one month"),
    ("3M", "3M", 63, "3M, three months"),
    ("1Y", "1Y", 251, "1Y, one year"),
)
DEFAULT_RANGE = "3M"
# The recent-prices reader's notes, in words.
NOTE_WORDS = {
    "gap": "no price served",
    "ex-dividend": "ex-dividend day",
    "split": "share split",
    "large move": "large move",
    "revised": "revised by the provider",
}
DRAG = "Drag across the chart, or use the arrow keys, to read any day."
NO_TOTAL_BUTTON = "Why no total for the period?"


def no_total(lately: bool) -> dict:
    """The chart's "Why no total?" sheet; `lately`: a 4-week change chip is shown below."""

    paragraphs = [
        (
            "This chart shows each day's close and its move from the day before, as the price "
            "reader gives them, and adds nothing up: there is no change over the week, month or "
            "year."
        ),
        (
            "To keep its research fair, the Lab works out nothing more from recent prices (2025 "
            "on) outside its evening job, which records what it works out with each outlook."
        ),
    ]
    if lately:
        paragraphs.append(
            'The 4-week change under "Lately" is one of those: the evening job recorded it with '
            "the outlook."
        )
    paragraphs.append(
        TECHNICAL.format(
            detail="decision A6. The recent-prices reader serves each day's close, its change "
            "from the session before, a split-consistent level for the line, and notes."
        )
    )
    return {"title": NO_TOTAL_BUTTON, "columns": [], "rows": [], "paragraphs": paragraphs}


def chart_label(ticker: str) -> str:
    return (
        f"Price chart of {ticker}'s daily closes. Use the left and right arrow keys to read "
        "each day."
    )


def range_note(first: str, last: str, *, shown: int, asked: int) -> str:
    """Which days a range shows: all it asked for, or every stored day when fewer are stored."""

    span = f"{short_day(first)} to {short_day(last)}"
    if shown < asked:
        return f"Only {shown} trading days are stored: {span}."
    return f"Daily closes from {span}."


def point_when(session: str, notes: list[str], *, latest: bool) -> str:
    """'on Wed, Oct 21 (latest close) · ex-dividend day'."""

    text = f"on {short_day(session)}" + (" (latest close)" if latest else "")
    return " · ".join([text, *notes])


def point_label(
    session: str, close: str, change: str, notes: list[str], *, latest: bool = False
) -> str:
    """A day of the chart, for a screen reader."""

    day = short_day(session) + (" (latest close)" if latest else "")
    if close == DASH:
        text = f"{day}: no close"
    elif change == DASH:
        text = f"{day}: closed at {close}"
    else:
        text = f"{day}: closed at {close}, {change} that day"
    return text + "".join(f"; {note}" for note in notes) + "."


# --- Stock detail: the outlook ---

EXPERIMENTAL_PILL = "Experimental"
EXPERIMENTAL_SHEET = {
    "title": "Experimental",
    "columns": [],
    "rows": [],
    "paragraphs": [
        (
            "This outlook isn't proven. Its model was tested on past data, and its live record "
            "is still short. It takes months of results before anyone can tell whether it is "
            "any good."
        ),
        "Treat it as a measurement experiment, not a tip. It is not advice.",
    ],
}
LEGEND = ("↑ Up", "→ Flat", "↓ Down")
NEWS = "Made from closing prices. It knows nothing about news since then."
FIVE_TITLE = "Next 5 trading days"
EACH_DAY = ("Each day on its own", "Hide each day")
NO_TOTAL_RECORDED = "No five-day total was recorded with this outlook."


def outlook_title(session: str) -> str:
    return f"{long_weekday(session)}'s outlook"


def outlook_sub(session: str, made: str) -> str:
    return f"For {short_day(session)}. Recorded {made}, after the close."


def bar_label(increase: float, stable: float, decrease: float) -> str:
    return f"Up {percent(increase)}, Flat {percent(stable)}, Down {percent(decrease)}"


def outlook_definition(ticker: str, as_of: str, band: str) -> str:
    return (
        f"Up means {ticker} closes more than {band} above {long_weekday(as_of)}'s close, and "
        f"Down more than {band} below it. Flat means it ends within {band} either way."
    )


def five_sub(ticker: str, end: str, as_of: str) -> str:
    return (
        f"Where {ticker} closes on {short_day(end)}, compared with the close on {short_day(as_of)}."
    )


def five_definition(band: str) -> str:
    return (
        f"Over five days the line between Up and Flat is {band}, because prices drift further "
        "in five days than in one."
    )


def each_day_note(band: str) -> str:
    return (
        f"Each date's own chance of an up, flat or down day: that day's close against the day "
        f"before, using the same {band} line. They are not a path to the five-day total, which "
        "is an outlook of its own."
    )


def day_and_date(session: str) -> str:
    """'Thu 22'."""

    day = _day(session)
    return f"{day:%a} {day.day}"


def day_label(session: str, increase: float, stable: float, decrease: float) -> str:
    day = _day(session)
    return f"{day:%A}, {day:%b} {day.day}: " + bar_label(increase, stable, decrease) + "."


# --- Stock detail: how reliable ---

RELIABLE_TITLE = "How reliable has this been?"
NOT_ENOUGH = "Not enough data yet"
READ_THIS = "How to read this"
JUDGE = "How to judge it"
DEVELOPMENT_LATER = "The past-test result appears next to an outlook."
DEVELOPMENT_NONE = "No past-test result is stored for this outlook."
DETAILS = "Details"
RELIABLE_UNREADABLE = "Can't be shown while the forecast record can't be read."
READ_THIS_TITLE = 'How to read "how reliable"'
READ_THIS_COLUMNS = ("When it said", "It happened")
CATEGORY_WORDS = {
    "increase": ("↑ Up", "went up"),
    "stable": ("→ Flat", "stayed flat"),
    "decrease": ("↓ Down", "went down"),
}


def about(low: float, high: float) -> str:
    """A probability range in whole percentages: 'about 40–50%'."""

    return f"about {percent(low).rstrip('%')}–{percent(high)}"


def stocks_words(stocks: list[str]) -> str:
    """'AMD, MRVL, MU and NVDA'; six or more are counted instead."""

    if len(stocks) > 5:
        return f"{len(stocks)} stocks"
    if len(stocks) == 1:
        return stocks[0]
    return ", ".join(stocks[:-1]) + f" and {stocks[-1]}"


def happened(observed: float, low: float, high: float) -> str:
    """'47% (likely 45–49%)'."""

    return f"{percent(observed)} (likely {percent(low).rstrip('%')}–{percent(high)})"


def development_scope(stocks: list[str], through: str | None, synthetic: bool) -> str:
    """Whose past tests: ' on AMD, MRVL, MU and NVDA together, through 2024'."""

    if synthetic:
        return " on the synthetic demo stocks"
    text = f" on {stocks_words(stocks)} together" if len(stocks) > 1 else ""
    if through:
        text += f", through {through[:4]}"
    return text


def development_headline(low: float, high: float, observed: float) -> str:
    return (
        f"When the model said {about(low, high)} for Up, the stock went up {percent(observed)} "
        "of the time."
    )


def development_meta(scope: str, n: int, dates: int, low: float, high: float) -> str:
    return (
        f"Past tests{scope}. {n:,} forecasts on {dates:,} dates. Likely range "
        f"{percent(low).rstrip('%')}–{percent(high)}."
    )


def development_short(low: float, high: float, n: int, dates: int, needed: int) -> str:
    return (
        f"Too few past tests gave {about(low, high)} for Up: {n:,} forecasts on {dates:,} dates, "
        f"and at least {needed} dates are needed."
    )


def development_sheet_paragraphs(
    scope: str, low: float, high: float, observed: float, n: int, dates: int, synthetic: bool
) -> list[str]:
    caution = (
        "These are synthetic demo stocks: the test checks the machinery and says nothing about "
        "any real stock."
        if synthetic
        else "Two reasons to stay cautious: these stocks were picked knowing they had done well, "
        "and past tests can flatter a model. Only the live record counts in the end."
    )
    return [
        (
            f"In past tests{scope}, the Lab gathered every time the model gave {about(low, high)} "
            f"for Up, and counted how often the stock really went up: {percent(observed)} of "
            f"{n:,} times."
        ),
        (
            "When a model's chances are honest, the things it gives about 45% happen about 45% "
            "of the time."
        ),
        (
            "The likely range shows how far that could be off by luck alone. It is a 90% range "
            f"that counts the forecasts made on one date only once ({dates:,} dates)."
        ),
        caution,
    ]


def live_words(
    status: str, sessions: int, targets: int, *, sessions_needed: int, targets_needed: int
) -> str:
    if status == "none":
        return (
            f"Live record: no scored days yet. Model Health needs {sessions_needed} trading "
            "days before it judges the outlooks."
        )
    if status == "ok":
        return (
            f"Live record: {sessions} trading days scored, past Model Health's minimum of "
            f"{sessions_needed}. The research page shows how it has done."
        )
    if sessions >= sessions_needed:
        return (
            f"Live record: {sessions} trading days scored, but only {targets:,} of the "
            f"{targets_needed} scored outlooks Model Health needs."
        )
    return (
        f"Live record: {sessions} of the {sessions_needed} trading days Model Health needs "
        "before it judges the outlooks."
    )


LIVE_SCOPE = "Every stock's scored outlooks, counted together."


def meter_label(sessions: int, needed: int) -> str:
    """The meter, for a screen reader: '12 of 50 trading days' (full at the minimum)."""

    return f"{min(sessions, needed)} of {needed} trading days"


def judge_sheet(sessions_needed: int, targets_needed: int, technical: str) -> dict:
    return {
        "title": "How to judge an outlook",
        "columns": [],
        "rows": [],
        "paragraphs": [
            (
                "One day tells you almost nothing. A 47% chance of Up, for example, isn't wrong "
                "when the stock goes down."
            ),
            (
                "A fair test looks at many days: when it said about 47%, did it happen about 47% "
                "of the time?"
            ),
            (
                f"Model Health starts judging after {sessions_needed} trading days with a result "
                f"(about ten weeks) and {targets_needed} scored outlooks."
            ),
            TECHNICAL.format(detail=technical),
        ],
    }


# --- Stock detail: lately (the forecast record's own context) ---

LATELY_TITLE = "Lately"
RUN_ADJECTIVES = {"increase": "rising", "decrease": "falling", "stable": "flat"}
REGIME_WORDS = {"calm": "Calm swings", "middle": "Normal swings", "volatile": "Bigger swings"}
NOTHING_LATELY = "Nothing is recorded for this stock yet."
NOTHING_SHOWN = "Nothing recorded with the latest outlook can be shown here."


def lately_sub(as_of: str) -> str:
    return (
        f"As of the close on {short_day(as_of)}, recorded with that outlook. Facts about the past, "
        "not predictions."
    )


def run_chip(direction: str | None, length: int | None) -> str:
    """'3 rising days in a row', '1 falling day', 'No streak after a gap'."""

    if direction is None or not length:
        return "No streak after a gap"
    adjective = RUN_ADJECTIVES.get(direction, direction)
    return f"1 {adjective} day" if length == 1 else f"{length} {adjective} days in a row"


def run_sheet(ticker: str, as_of: str, band: str) -> dict:
    return {
        "title": "Days in a row",
        "columns": [],
        "rows": [],
        "paragraphs": [
            (
                f"How many trading days in a row {ticker} closed more than {band} higher "
                f"(rising), more than {band} lower (falling) or within {band} (flat), up to "
                f"the close on {short_day(as_of)}. A day with no price ends a streak."
            ),
            "Recorded by the evening job with the outlook. A streak doesn't say what happens next.",
        ],
    }


def momentum_chip(ret_20: float) -> str:
    return f"Past 4 weeks {signed_percent(ret_20)}"


def momentum_sheet(ticker: str, as_of: str) -> dict:
    return {
        "title": "Past 4 weeks",
        "columns": [],
        "rows": [],
        "paragraphs": [
            (
                f"The change in {ticker}'s closing price over the 20 trading days up to "
                f"the close on {short_day(as_of)}, as the evening job recorded it with the outlook. "
                "Dividends are not included."
            ),
            "A fact about the past, not a prediction.",
        ],
    }


def regime_sheet(ticker: str, as_of: str) -> dict:
    return {
        "title": "Swings",
        "columns": [],
        "rows": [],
        "paragraphs": [
            (
                f"How big {ticker}'s daily moves were up to the close on {short_day(as_of)}, against "
                "the year of prices the outlook read: its calmest third, its middle third or its "
                "liveliest third."
            ),
            "It says nothing about the direction. Recorded by the evening job with the outlook.",
            TECHNICAL.format(
                detail="the EWMA volatility against its own terciles in the 252-session window."
            ),
        ],
    }


# --- Stock detail: for learning ---

LEARNING_TITLE = "For learning"
RESEARCH_PAGE = "stock-probability"  # the Stock Probability page's path (app.wording.NAV)
RESEARCH_ELSEWHERE = "The research pages open from the Portfolio Lab app in your Dock."


def research_link(ticker: str) -> str:
    return f"Open {ticker} in the research view"


def research_sub(ticker: str) -> str:
    return (
        "The Stock Probability page: every probability, the model's details and its full test "
        f"results. Choose {ticker} there."
    )


def explain(band: str | None) -> list[str]:
    """The footer's explanation; `band` is the recorded band in words ('0.5%'), if any."""

    edge = band or "a small band"
    return [
        (
            f"↑ is the recorded chance that the stock closes the next trading day more than "
            f"{edge} higher than its last close. ↓ is the chance it closes more than {edge} "
            f"lower. The rest is the chance it stays within {edge} (Flat)."
        ),
        (
            "They come from a deliberately simple model, frozen so it can't change, that mostly "
            "reflects how often each stock went up or down in the past. It is a starting point "
            "that better ideas must beat."
        ),
        (
            "Each outlook is made after the close from closing prices, so it knows nothing about "
            "later news."
        ),
        "Green and red only ever mean a price went up or down. Outlooks are never green or red.",
    ]
