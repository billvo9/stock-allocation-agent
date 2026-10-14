"""
The Explore page's copy: every caveat, badge and state message in one data
module, so tests assert the exact wording and a later front end reuses it.

Wording rules: describe the past only; no forecast, no advice, no trading
verbs; gains and losses carried by sign and words, never by colour.
"""

from __future__ import annotations

PAGE_EYEBROW = "Explore"
PAGE_TITLE = "Fund history"
PAGE_PURPOSE = (
    "Facts about the past for a few broad funds: how a hypothetical $10,000 would have "
    "grown, each fund's worst decline, and its calendar-year returns."
)

EVIDENCE_BADGE = "Descriptive: facts about the past"
SYNTHETIC_BADGE = "Synthetic demo data"
STALE_BADGE = "Stale data"
SYNTHETIC_NOTICE = (
    "Synthetic demo data, not real fund performance. Every fund on this page comes from a "
    "seeded random generator and exists only to demonstrate the page."
)

GROWTH_TITLE = "Hypothetical growth of $10,000"
REINVESTMENT = (
    "Distributions reinvested on the ex-date: a modeling convention, not actual investor "
    "cash flows."
)
BEFORE_COSTS = "Before taxes and transaction costs."
EXPENSES = "Fund expenses are already reflected in fund prices and are not deducted again."
HONEST_LABELS = (REINVESTMENT, BEFORE_COSTS, EXPENSES)
FRACTIONAL_SHARES = "Fractional shares assumed."
LOG_SCALE = "Logarithmic scale: equal vertical distances are equal percentage changes."

DECLINE_TITLE = "Worst decline"
DECLINE_SCOPE = (
    "The largest percentage decline from a previous high within the period shown. A later "
    "decline can be larger in dollars, because later highs are higher."
)
DAILY_CLOSES = "Measured on daily closes; declines within a day may have been deeper."

CALENDAR_TITLE = "Calendar-year returns"
CALENDAR_SCOPE = (
    "Each calendar year's total return within the period shown. Partial years are labelled "
    "and never annualized. Whether a year lost money is decided on the unrounded return."
)

ABOUT_TITLE = "About these numbers"
CANNOT_TELL_TITLE = "What this page cannot tell you"
CANNOT_TELL = (
    "This page describes the past. It is not a forecast, and it does not say which fund "
    "anyone should hold. Past returns, real or synthetic, do not predict future returns."
)

LOADING = "Loading fund histories…"
EMPTY = "No fund selected. Choose one or more funds above to see their history."
NO_AVAILABLE_FUND = "None of the selected funds has available data. Choose another fund."
NO_SHARED_PERIOD = (
    "The selected funds have no period in common, so they cannot share one chart. "
    "Choose fewer funds."
)
OUTSIDE_COVERAGE = (
    "{period} is outside the selected funds' shared coverage ({start} to {end}). "
    "Choose another period."
)
NO_COMMON_START = (
    "No session in this period has data for every selected fund, so the funds cannot share "
    "a start date."
)
NO_COMMON_START_REASON = "no_common_start"
PENDING_BADGE = "Pending approval"
UNAVAILABLE_BADGE = "Unavailable"
PENDING_EXPLAINER = (
    "The formulas behind these figures (growth of $10,000, worst decline and calendar-year "
    "returns) are awaiting the project owner's approval. Nothing is estimated or filled in "
    "meanwhile."
)

# Machine reason codes shown on the page, in plain words. An unknown code is
# shown with its underscores as spaces, never hidden.
REASON_TEXT = {
    "missing_input": "no value from the source",
    "no_records": "no records from the source",
}


def reason_text(code: str) -> str:
    return REASON_TEXT.get(code, code.replace("_", " "))


LIMITATION_TEXT = {
    "synthetic_data": "Synthetic demo data, not real fund performance.",
    "synthetic_calendar": (
        "Sessions are weekdays, not an exchange calendar: market holidays are not removed."
    ),
}

ROLE_TEXT = {"equity": "Equity", "bonds": "Bonds", "cash proxy": "Cash proxy"}
