"""
The Stock Probability & Reversal Lab: an experimental, self-contained
feature for a saved list of US-listed stocks. It describes each stock's
daily direction (Increase, Stable or Decrease against a stable band), its
streaks and reversals, and (in later modules) estimates the probability of
each direction over the next five sessions.

It is independent of the ETF research track, the prospective ETF pipeline,
the RL environment and the portfolio simulator: none of them imports it and
it imports none of them. Nothing here is investment advice or validated
evidence; every output carries its evidence status.

Modules:
    contracts  categories, the stable band, symbol and data statuses,
               reason codes (standard library only)
    symbols    ticker parsing and normalization (offline), and the
               classification of a ticker against a provider listing
    prices     the daily bar schema, split-consistent returns and categories
    runs       directional runs (streaks) with an explicit stable-session rule
    features   deterministic, past-only features as of each session
    store      the private SQLite store: the watchlist and its event log,
               listing snapshots and write-once price batches
    synthetic  seeded synthetic demo stocks (visibly synthetic, no I/O)
    view       pure view logic for the Lab's Stock Probability page
    tiingo     the Tiingo end-of-day stock adapter (scripts only: the Lab
               never imports it, and nothing calls it without authorization)

This package's `__init__` imports nothing, so loading one module never
loads another (the Lab relies on this; tested).
"""
