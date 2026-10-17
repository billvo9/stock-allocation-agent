"""
The copy of the Lab's Stock Probability destination, in one data module so
tests assert and scan it (the same rules as `wording`: no advice, no trading
verbs, nothing ranked or preferred; every pending capability is named as
pending). Probabilities appear only where a stored, checked forecast exists,
always with their evidence status; today that means the synthetic demo
stocks only.
"""

from __future__ import annotations

EYEBROW = "Experimental"
TITLE = "Stock Probability Lab"
LEDE = (
    "Save the US stocks you follow and study how each one has moved day to day: how often it "
    "rose, stayed within a small band or fell, how long its streaks lasted and how often they "
    "reversed. Where a tested model exists, see its probabilities for each of the next five "
    "sessions beside the stock's usual mix."
)
EXPERIMENTAL_BADGE = "Experimental: no validated forecast"
DESCRIPTIVE_BADGE = "Describes the past only"
SYNTHETIC_BADGE = "Synthetic demo stocks: not real"
NOT_YET = (
    "Probabilities are shown only for the synthetic demo stocks, from estimators tested "
    "walk-forward on synthetic data. Real stocks get probabilities only after live data and "
    "the way forecasts are recorded are approved; until then they show their history only."
)

TABS = (
    "Watchlist",
    "Next five sessions",
    "History",
    "Reversals",
    "Bull traps",
    "Model diagnostics",
)
CONTROLS_TITLE = "Choose a stock"

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

DETAIL_TITLE = "History"
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

# --- the next five sessions ---

FORECAST_TITLE = "Next five sessions"
FORECAST_BADGE = "Experimental, unvalidated"
DEVELOPMENT_BADGE = "Tested on data through 2024-12-31"
FORECAST_NONE_DEMO_TITLE = "No demo forecast stored yet"
FORECAST_NONE_DEMO = (
    "The synthetic demo forecasts are computed on this computer, with no network, by one "
    "command. It also records the walk-forward evaluation shown under Model diagnostics."
)
DEMO_COMMAND = ".venv/bin/python scripts/stock_lab_evaluate.py --demo"
FORECAST_NONE_REAL_TITLE = "No forecast for real stocks yet"
FORECAST_NONE_REAL = (
    "Probabilities for a real stock need three approvals first: live data, using recent "
    "prices as inputs only, and recording forecasts so they can be scored as sessions close. "
    "Until then the History, Reversals and Bull traps tabs describe its past."
)
FORECAST_BAND = (
    "These probabilities use the {band} band the models were tested with. The band chosen "
    "above changes the history views only."
)
FORECAST_REFUSED = "No probability was given for this stock: {detail}"
NEXT_TITLE = "The next session"
DAILY_NOTE = (
    "Each session's probabilities are for that day's own move, not for the move from today's "
    "close to that day. The total over all five sessions is shown separately below."
)
FIVE_TITLE = "Each of the next five sessions"
FIVE_SUBTITLE = "Dot: probability · bar: how often similar forecasts came true in tests"
USUAL_SERIES = "Usual rate"
FORECAST_SERIES = "Forecast"
FIVE_TABLE = "The five sessions"
CUMULATIVE_TITLE = "The five-session total"
CUMULATIVE_NOTE = (
    "The change from the last close to the close five sessions ahead, against a wider band "
    "of {band} (the daily band times the square root of five)."
)
RECORDED = (
    "As of the close of {as_of}; made {created}. Estimator: {estimator}. Model {version}, "
    "features {features}, evaluation {evaluation}."
)
SUPERSEDED = "{count} earlier forecast(s) for this stock are kept, each replaced by a newer one."
TRACK_NOTE = (
    "“Tests: 29–37% came true” means: in the walk-forward tests, forecasts in the same range "
    "for this estimator came true that often. It describes past calibration, not certainty."
)
SAME_AS_USUAL = "Same as usual"

# --- reversals ---

REVERSALS_TITLE = "Reversals"
REVERSALS_LEDE = (
    "What followed a streak, or a large one-session move, in this stock's own past. These are "
    "counts of past cases, with their uncertainty; they are not a forecast."
)
DIRECTION_LABEL = "After a streak of"
DIRECTION_OPTIONS = {"decrease": "Falling sessions", "increase": "Rising sessions"}
LENGTH_LABEL = "Streak length"
LENGTH_HELP = "Each streak is counted once, when it first reaches this length."
STREAK_CHART_TITLE = "How often the next session {reverse} after a {noun} streak"
STREAK_CHART_SUBTITLE = "Hatched bars: few cases · outlined bar: the current streak"
STREAK_TABLE = "By streak length"
REGIME_TITLE = "Calm and volatile spells"
REGIME_NOTE = (
    "Volatility at the time of each streak, against the stock's own earlier volatility (its "
    "lowest, middle and highest thirds so far)."
)
DURATION_TITLE = "How long reversals lasted"
DURATION_NOTE = (
    "The run in the other direction that began right after a streak of at least this length. "
    "Runs still open, or cut by a missing session, count as lasting at least as long as seen "
    "(Kaplan-Meier). Median: {median}. {runs} reversal(s), {censored} still open."
)
MOVE_TITLE = "After a large one-session move"
MOVE_LABEL = "Move larger than"
MOVE_SIGN_LABEL = "Direction of the move"
MOVE_SIGN = {"1": "A gain", "-1": "A fall"}
MOVE_NOTE = (
    "{definition}: {events} case(s), each counted once per five sessions, {pending} still "
    "pending. Continuation, stabilization and reversal compare the total change over the next "
    "sessions with a band that widens with the number of sessions."
)
INSUFFICIENT = "Insufficient observations"

# --- bull traps ---

TRAPS_TITLE = "Bull traps: failed breakouts"
TRAPS_LEDE = (
    "A bull trap is a rally that breaks above a recent high and then falls back below it. Here "
    "it is a fixed, written rule, checked against this stock's own past, beside how often any "
    "breakout failed. A pattern name is not evidence: the comparison is."
)
TRAP_STEPS = (
    (
        "NOTICE",
        "Step 1",
        "A decline",
        "The recent low was at least {decline} below the high of the previous {high} sessions.",
    ),
    ("NOTICE", "Step 2", "A rebound", "The close is at least {rebound} above that low."),
    (
        "NOTICE",
        "Step 3",
        "A breakout",
        "The close is above the highest close of the previous {window} sessions.",
    ),
    (
        "WARNING",
        "Failure",
        "Back below",
        "Within {failure} sessions, a close falls more than {margin} below that earlier high.",
    ),
)
TRAP_LATEST = "Latest setup: {session}. {status}"
TRAP_NONE = "No setup in this stock's history matches the rule."
TRAP_NOW = "A setup is open now; it stays unresolved until its {failure}-session window closes."
TRAP_RATES = "Setup failure rate {rate} over {n} resolved setup(s), against {base} for every breakout ({lift})."
TRAP_DETAILS = "Precision, recall and false alarms"
TRAP_DETAILS_TEXT = (
    "Treating “setup” as a warning that a breakout will fail: precision (warnings that came "
    "true) {precision}; recall (failures that were warned of) {recall}; false-alarm rate "
    "(breakouts that held but were warned of) {false_alarm}."
)
BEAR_TITLE = "Bear traps (the mirror rule)"
BEAR_TEXT = (
    "A fall below a recent low after a rally, then a recovery above it. Failure rate {rate} "
    "over {n} resolved setup(s), against {base} for every breakdown."
)

# --- diagnostics (PR 2) ---

DIAGNOSTICS_NONE = (
    "No evaluation is stored yet. Run the demo command to evaluate the estimators on the "
    "synthetic demo stocks."
)
DIAGNOSTICS_SYNTHETIC = (
    "This evaluation used synthetic data: it checks that the machinery works, and says nothing "
    "about any real stock."
)
SCORES_TITLE = "Scores against the baselines"
SCORES_NOTE = (
    "Lower is better. RPS, the ranked probability score, is the primary score: it charges more "
    "for predicting a fall when a rise happens than for predicting Stable. Averages over the "
    "five daily targets, on identical test sessions."
)
COMPARISON_TITLE = "Is the difference real?"
COMPARISON_NOTE = (
    "Each estimator's daily RPS minus the recent-weighted baseline's, averaged per date across "
    "stocks (stocks that move together are not independent evidence), with Newey-West "
    "standard errors and Holm's correction for the comparisons made. Newey-West decides; two "
    "other tests are shown beside it as checks: a t-test on the yearly averages (fold-block t) "
    "and a block bootstrap of whole dates."
)
SIZE_TITLE = "How often each test is fooled (simulated)"
SIZE_NOTE = (
    "Simulated panels shaped like this one ({names} stocks, {years} test years, returns "
    "correlated {correlation} across stocks and unpredictable over time), where the challenger "
    "has no skill: it is the recent-weighted baseline read from another stock's history. Each "
    "cell is the share of {reps} panels in which the test still called the challenger better "
    "at the {nominal} level, with its simulation error. Near {nominal}, the test's p-values can "
    "be taken at face value; well above it, they are too small. Rows are the half-lives the "
    "baseline can use. Reported only: never used to choose an estimator."
)
SIZE_SYMMETRIC = (
    "Against the baseline read from yet another stock's history: a fair coin between two "
    "equally uninformed estimators, so this table measures the tests themselves."
)
SIZE_OWN = (
    "Against the stock's own baseline, which keeps learning from the very sessions it is "
    "scored on while the challenger does not. The real comparisons have this imbalance too, "
    "because the estimators learn from a stock's past in different ways."
)
BAND_TITLE = "Sensitivity to the Stable band"
BAND_NOTE = (
    "The recent-weighted (B1) and volatility-scaled (B3) estimators re-run on the same test "
    "years with a narrower and a wider band; mean RPS over the five daily targets. Scores are "
    "not comparable between rows, because a wider band makes Stable more common: compare B3 "
    "with B1 within a row. Reported only: never used to choose the band or an estimator."
)
RELIABILITY_TITLE = "{category}: said versus happened"
RELIABILITY_SUBTITLE = "Filled: estimator in use · hollow: recent-weighted baseline (B1)"
DIAGONAL = "Perfect calibration"
COUNTS_TITLE = "Forecasts in each range"
COUNTS_SUBTITLE = "Development tests, the next session"
FOLDS_TITLE = "Walk-forward folds"
FOLDS_NOTE = (
    "Each year is tested with models fitted only on earlier, matured outcomes; every choice "
    "(the half-life, the logistic penalty) was made inside the training years."
)
CANDIDATES = "Candidates tried: {family}. Every one is reported here; none was dropped."

# --- the forecast record (PR 3) ---

HISTORY_TITLE = "Earlier forecasts and what happened"
HISTORY_NOTE = (
    "Each forecast is stored once and never changed; a newer one replaces it on the page but "
    "the old one stays. When its session closes, what happened is recorded beside it with the "
    "score of the stored probabilities (lower RPS is better). {summary}"
)
HISTORY_SUMMARY = "{targets} session targets scored so far, mean RPS {rps:.3f}."
HISTORY_NONE = "No session has closed since the first forecast, so nothing is scored yet."
JOB_STATUS = "Last update: {status} at {finished} (expected session {expected})."
REPLAY_COMMAND = ".venv/bin/python scripts/stock_lab_daily.py --demo-replay 20"
REPLAY_NOTE = (
    "To see forecasts scored as sessions close, run a synthetic replay: it reveals the demo "
    "stocks' last sessions one simulated day at a time, with models fitted only on earlier data."
)
