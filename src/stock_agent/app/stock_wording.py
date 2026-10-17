"""
The copy of the Lab's Stock Probability destination, in one data module so
tests assert and scan it (the same rules as `wording`: no advice, no trading
verbs, nothing ranked or preferred; every pending capability is named as
pending). Today the page describes stored history only: no probability or
forecast is shown until the estimators are evaluated.
"""

from __future__ import annotations

EYEBROW = "Experimental"
TITLE = "Stock Probability Lab"
LEDE = (
    "Save the US stocks you follow, then study how each one has moved day to day: how often it "
    "rose, stayed within a small band or fell, how long its streaks lasted, and how often a "
    "streak reversed. Probabilities for the next five sessions come later, once the models are "
    "tested against simple baselines."
)
EXPERIMENTAL_BADGE = "Experimental: no validated forecast"
DESCRIPTIVE_BADGE = "Describes the past only"
SYNTHETIC_BADGE = "Synthetic demo stocks: not real"
NOT_YET = (
    "No probability is shown yet. The probability models, their walk-forward tests against "
    "simple baselines and the forecast record are still being built, so this page shows only "
    "what each stock's stored history says about the past."
)

TABS = ("Watchlist", "Stock detail", "Model diagnostics")

# --- the watchlist ---

WATCHLIST_TITLE = "Your watchlist"
WATCHLIST_SCOPE = (
    "Saved privately on this computer, outside the project's files, so it is kept when the "
    "app restarts. Listed alphabetically; nothing is ranked."
)
ADD_LABEL = "Add tickers"
ADD_HELP = (
    "One ticker or a pasted list, separated by spaces, commas or new lines. US-listed stocks "
    "only, for example AAPL or BRK.B. A ticker is kept exactly as entered: the lab never "
    "swaps in another listing, share class or ADR."
)
ADD_PLACEHOLDER = "AAPL, MSFT, BRK.B"
ADD_BUTTON = "Add to watchlist"
ADDED = "Added: {tickers}."
ALREADY = "Already in the watchlist: {tickers}."
REWRITTEN = "{entered} is saved as {ticker}, the data provider's form of the same ticker."
REFUSED_TITLE = "Not added"
REFUSED_COLUMNS = ("Entered", "Why")
TOO_MANY = "Paste at most {limit} tickers at a time."
SEARCH_LABEL = "Search the watchlist"
SEARCH_PLACEHOLDER = "Ticker"
NO_MATCH = "No saved ticker matches “{query}”."
EMPTY_TITLE = "No stocks saved yet"
EMPTY = "Add a ticker above. The list is saved on this computer and kept across restarts."
REMOVE_LABEL = "Remove from the watchlist"
REMOVE_HELP = "The ticker's history of adds and removals is kept; only the list changes."
REMOVE_BUTTON = "Remove selected"
REMOVED = "Removed: {tickers}."
NOT_CHECKED_NOTE = (
    "Tickers are checked against the data provider's list of US-listed stocks when prices are "
    "first downloaded. Live data has not been switched on yet, so every ticker shows “Not yet "
    "checked” and “Missing data”."
)
WATCHLIST_BAND_NOTE = (
    "Stable shares here use the default band of {band}; the Stock detail tab lets you change it."
)
STATUS_GUIDE_TITLE = "What the statuses mean"
_UNSUPPORTED = (
    "Not a US-listed stock: not listed by the provider, listed abroad, over the counter, "
    "not priced in dollars, or not a stock (for example a fund)."
)
_DELISTED = (
    "The provider's prices ended weeks ago. The provider does not say whether the stock "
    "was delisted or its ticker changed."
)
_AMBIGUOUS = (
    "The provider lists more than one current security under this ticker; the lab does "
    "not guess which one is meant."
)
_UNCHECKED = "Prices are stored, but the daily update that checks they are current is not running."
STATUS_GUIDE = (
    ("Not yet checked", "The ticker's format is valid; it has not been checked with the provider."),
    ("Supported", "A stock listed on a US exchange, priced in US dollars, with recent prices."),
    ("Unsupported", _UNSUPPORTED),
    ("Inactive", "The provider has no recent prices for it."),
    ("Delisted or renamed", _DELISTED),
    ("Ambiguous", _AMBIGUOUS),
    ("Missing data", "No prices are stored for it yet."),
    ("Freshness not checked", _UNCHECKED),
)
DEMO_TITLE = "Synthetic demo stocks"
DEMO_SCOPE = (
    "Three made-up stocks generated from fixed seeds, so the page has something to show before "
    "any real data is loaded. They are not real stocks and say nothing about any real company."
)

# --- one stock ---

DETAIL_TITLE = "Stock detail"
DETAIL_EMPTY = (
    "No stored prices to show yet. Choose a synthetic demo stock, or wait for the first "
    "authorized data update."
)
CHOOSE_LABEL = "Stock"
CHOOSE_HELP = "Synthetic demo stocks, and saved stocks with stored prices."
BAND_LABEL = "Stable band"
BAND_HELP = (
    "A session counts as Stable when its price return is within this band either way, as "
    "Increase above it and as Decrease below it."
)
RULE_LABEL = "How a Stable session affects a streak"
RULE_OPTIONS = {
    "breaks": "A Stable session ends a streak",
    "pauses": "One Stable session pauses a streak",
}
RULE_HELP = (
    "How a Stable session inside a run of rises or falls is treated. The rule in use is always "
    "stated with the results."
)
SHOWN_THROUGH = "Shown through {day}."
REAL_CUTOFF = (
    "Real stocks are shown through 2024-12-31 only. Prices from 2025 on are held back for a "
    "final, one-time test of the models, so they are not shown or used here yet."
)
PRICE_TITLE = "Price, split-consistent"
PRICE_SERIES = "Split-consistent price"
PRICE_AXIS = "Price on the last session's share basis (log scale)"
PRICE_SUBTITLE = "Raw closes put on one share basis across splits · dividends not added back"
SYNTHETIC_CHART = "Synthetic demo data, not a real stock"
AS_TABLE = "{chart}: as a table"
RECENT_TABLE_TITLE = "The last sessions"
RECENT_TITLE = "The last {count} sessions"
RECENT_SUBTITLE = "Daily price return · dotted lines mark the stable band"
BAND_MEANING_TITLE = "What the stable band means for this stock"
BAND_MEANING = (
    "The same band is wide for a calm stock and narrow for a volatile one. This stock's "
    "typical daily move is {typical}, so at {band} about {stable} of its sessions were "
    "Stable. The table shows the shares at nearby bands."
)
RUNS_TITLE = "How long streaks lasted"
RUNS_CAPTION = (
    "Completed rising and falling streaks by length. A streak still open at the last session, "
    "or cut short by a missing session, has an unknown final length; it is counted in the "
    "last column, never as a finished streak."
)
FEATURES_TITLE = "Latest measures (from past sessions only)"
FEATURES_CAPTION = (
    "Each measure uses the latest session and earlier ones only. These are the inputs the "
    "probability models will use; on their own they predict nothing."
)
METHOD_TITLE = "How these numbers are computed"
METHOD = (
    "Price return of session t: close(t) × split factor(t) ÷ close(t−1) − 1, from raw "
    "closes. The split factor is the number of new shares per old share on a split's "
    "ex-date and 1 otherwise. Dividends are not added back, so an ex-dividend session "
    "includes the dividend's mechanical drop: this is price direction, not total return. "
    "Increase means a return above the band, Decrease below minus the band, Stable within "
    "it, edges included. A vendor's adjusted prices are never used, because they are "
    "recomputed after every later dividend or split."
)
LIMITATIONS_TITLE = "Data notes"

# --- diagnostics ---

DIAGNOSTICS_TITLE = "Model diagnostics"
DIAGNOSTICS_BADGE = "Not built yet"
DIAGNOSTICS_EMPTY_TITLE = "No probability model yet"
DIAGNOSTICS_EMPTY = (
    "Probabilities will come from fitted estimators, compared in walk-forward tests on "
    "history through 2024 with the most transparent baselines: the stock's long-run "
    "frequencies, recent-weighted frequencies, a streak-aware version of them, and a "
    "volatility-scaled baseline. A more flexible model is kept only if it beats them on "
    "calibration and probability scores; otherwise the baseline is kept and the page says so."
)
DIAGNOSTICS_EXPECTATIONS = (
    "What to expect: day-to-day direction of large US stocks is close to unpredictable, so "
    "most measurable skill is in telling Stable sessions from moving ones (a matter of "
    "volatility), not rises from falls. With a watchlist of a few dozen stocks that move "
    "together, a small edge in direction would take decades of data to detect."
)
