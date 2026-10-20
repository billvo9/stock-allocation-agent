"""
The copy of the Lab's Premarket Briefing destination, in one data module so
tests assert and scan it (the same rules as `wording`: no advice, no trading
verbs, nothing ranked or preferred). The briefing's own sentences come from
the written briefing and its renderer; these are the page's frame around
them.
"""

from __future__ import annotations

EYEBROW = "Stock Probability Lab · Experimental"
TITLE = "Premarket briefing"
LEDE = (
    "The morning briefing as the briefing script wrote it on this computer: each stock's "
    "stored probabilities for the session, how reliable numbers like them have been, and what "
    "is missing, with its reason. This page only reads: it prepares nothing, runs no job, "
    "changes no file and sends nothing."
)
EXPERIMENTAL_BADGE = "Experimental: no validated forecast"
READ_ONLY_BADGE = "Read-only"
REAL_BADGE = "Real stocks: private, local use"
SYNTHETIC_BADGE = "Synthetic demo stocks: not real"
PREMARKET_BADGE = "Prepared before the open"
PREVIEW_BADGE = "Preview, not a delivery"
SYNTHETIC_NOTICE = (
    "Synthetic demo: generated data for demo stocks, not real companies. Nothing here is "
    "evidence about any real stock."
)

# --- choosing a briefing ---

CONTROLS_TITLE = "Choose a briefing"
KIND_LABEL = "Show briefings for"
KINDS = {"real": "Real stocks", "synthetic": "Synthetic demo"}
KIND_HELP = "A briefing is wholly real or wholly synthetic; the two are never mixed on this page."
CHOOSE_LABEL = "Briefing"
CHOOSE_HELP = (
    "The briefings written on this computer, the latest first: the newest market date, then "
    "the one prepared last."
)
PREMARKET_OPTION = "premarket briefing"
PREVIEW_OPTION = "preview"
OPTION = "{day} · {kind}, prepared {prepared}"
UNREADABLE_OPTION = "{day} · could not be read ({name})"
OLDER = (
    "1 older briefing file of this kind is in the folder but not listed here.",
    "{count} older briefing files of this kind are in the folder but not listed here.",
)
OTHER_UNREADABLE = (
    "1 listed briefing file could not be read: {names}.",
    "{count} listed briefing files could not be read: {names}.",
)
NEWER_UNREADABLE = (
    "The newest briefing file, for {day}, could not be read: {reason}. Shown below: the latest "
    "readable one, for {shown}."
)
NOT_TODAY = (
    "This is the briefing for {market}. Today is {today} in New York, and no later briefing "
    "has been written on this computer."
)

# --- states without a briefing ---

NONE_TITLE = "No briefing written yet"
NONE = {
    "real": (
        "No real briefing has been written in {directory}. The briefing script writes one with "
        "--preview. Until real forecasts are activated (decision A4), every real stock in it "
        "shows as unavailable, with its reason."
    ),
    "synthetic": (
        "No synthetic demo briefing has been written in {directory}. This command replays "
        "synthetic sessions in a temporary store and writes the demo briefing there; nothing "
        "real is read."
    ),
}
COMMANDS = {
    "real": ".venv/bin/python scripts/stock_lab_briefing.py --preview",
    "synthetic": ".venv/bin/python scripts/stock_lab_briefing.py --demo --preview",
}
REFUSED_TITLE = "The briefing folder was not read"
REFUSED = "Nothing in {directory} was read. The reason:"
UNREADABLE_TITLE = "This briefing could not be read"
UNREADABLE = "{name} was not shown, and the file is left exactly as it is. The reason:"

# --- the briefing ---

DATELINE = "For {market} · target 08:30 New York time · prepared {prepared}. {state}."
NOTICES_TITLE = "Notices"
GLANCE_TITLE = "At a glance"
GLANCE_CAPTION = (
    "The next session's probabilities exactly as stored, and the change since the prior "
    "forecast at the same stable band, in percentage points. A stock without a forecast for "
    "this session shows why instead of a number."
)
GLANCE_COLUMNS = ("Stock", "Status", "Change (points)")
NO_CHANGE = "No change"
NOT_COMPARABLE = "Not comparable"
EMPTY_WATCHLIST = "The watchlist was empty: there was nothing to brief."
STATUS_TITLE = "Data, model and checks"
ITEM_COLUMN = "Item"
FACT_COLUMN = "What the briefing records"
STOCKS_TITLE = "Each stock"
STOCKS_CAPTION = "Open a stock for its five sessions, its descriptive notes and its lineage."
NEXT_SESSION = "Next session {session}, stable band {band}; that day's own move."
CHANGE = "Change."
FIVE_CAPTION = "Each session's own move, stable band {band}; never read as a path."
FIVE_COLUMNS = ("Session", "Sessions ahead")
DETAIL_COLUMN = "What the briefing says"
TOTAL_LABEL = "Five-session total"
RELIABILITY_TITLE = "Reliability"
RELIABILITY_CAPTION = (
    "How often numbers like these came true: the development tests and the prospective record, "
    "kept apart and never pooled."
)
DEVELOPMENT_TITLE = "Development evidence"
DEVELOPMENT_TABLE = (
    "When {estimator} said Increase in these ranges, how often Increase happened (development "
    "test). Interval: {method}."
)
PROSPECTIVE_TITLE = "Prospective record"
PROSPECTIVE_TABLE = "{label}. Interval: {method}."
COMPARED = "Compared with B1."
LIMITATIONS_TITLE = "Limitations"
SOURCE = (
    "Shown as written: {name} in {directory}, briefing schema {schema} {version}. The page "
    "never prepares a briefing again; the briefing script is the only writer."
)

# --- the delivery log ---

DELIVERY_TITLE = "Delivery"
DELIVERY_CAPTION = "As the duplicate-send log recorded it when this page loaded."
CHANNELS = {"preview": "as local files"}
CHANNEL = "through the {channel} channel"
DELIVERED = "Delivered"
DELIVERED_TITLE = "Delivered once, {channel}"
DELIVERED_PREVIEW = (
    "Recorded {at} (attempt {attempts}). The briefing's files were written into this "
    "computer's private briefing folder; no email delivery is connected."
)
DELIVERED_OTHER = "Recorded {at} (attempt {attempts})."
PREVIEW_WORD = "Preview"
PREVIEW_TITLE = "A preview, not a delivery"
PREVIEW_TEXT = "It never counts as this market date's delivery and never uses it up."
PREVIEW_STATES = {
    "sent": " This market date's premarket briefing was delivered {at}.",
    "failed": " This market date's last delivery attempt failed, recorded {at}.",
    "in_progress": (
        " A delivery attempt for this market date started {at} and has not recorded an end."
    ),
    "not_recorded": " No delivery is recorded for this market date.",
    "unreadable": " The delivery log could not be read.",
}
FAILED_WORD = "Failed"
FAILED_TITLE = "The last delivery attempt failed"
FAILED_TEXT = (
    "Recorded {at} (attempt {attempts}): {detail}. A later run before 09:30 New York may try "
    "again; after that, nothing is delivered for this market date."
)
UNFINISHED_WORD = "Not finished"
UNFINISHED_TITLE = "A delivery attempt has not recorded an end"
UNFINISHED_TEXT = (
    "Started {at}; it is still running or it stopped. No run delivers this market date again "
    "until the delivery log is checked, so the briefing is never sent twice."
)
NOT_RECORDED_WORD = "Not recorded"
NOT_RECORDED_TITLE = "No delivery is recorded for this market date"
NOT_RECORDED_TEXT = (
    "The duplicate-send log has no entry for this market date and data class, although this "
    "premarket briefing's files exist: the log was moved or removed, or the files were copied "
    "here."
)
LOG_UNREADABLE_WORD = "Not read"
LOG_UNREADABLE_TITLE = "The delivery log could not be read"
