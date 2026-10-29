# The vintage-aware rates store and the DGS3MO backfill (group C, C2)

**Status.** Implements C2 (a). The owner approved it on 2026-10-26: "Approve all C1–C9
(Recommended)", which included authorizing a one-time FRED download of 2003-2024 T-bill
history; the record is in `docs/PROJECT_STATE.md`. The cash returns (C1, C3, C4, C9) and the
Sharpe ratio (M2) come next, on top of this store.

## What it does

- **Every published value is kept** (`stock_agent.data.rates.vintages`). Each ALFRED vintage is
  its own row, keyed (date, rate_id, vintage_date). This replaces the legacy (date, rate_id) key,
  C2's approved breaking change.
  - A revision is a new row.
  - A merge refuses a key stored with another value, and it refuses a vintage earlier than a
    stored date's first release. Either would rewrite booked history.
  - Storage is one new read-only parquet file per retrieval that adds anything, in
    `~/stock-agent-private/rates/` (outside Git, folders created 0o700). It is written to a
    temporary name and linked into place, so a failed write leaves no partial file, and a
    repeat retrieval adds no file.
  - Rows carry the `rate_id` that existing callers use (`USD_TREASURY_3M`, from
    `config/rates.yaml`).
- **Availability:** `h15_or_alfred_later`.
  - Rule: a vintage may first be used as a decision input at the later of 16:15 New York on the
    next Federal Reserve business day after its date, and the end (midnight New York) of its
    ALFRED `realtime_start` day.
  - The calendar treats every federal holiday, and the days around a weekend holiday, as closed,
    including across a year boundary (a Saturday January 1 closes the Friday before). Marking
    extra days closed can only delay the H.15 term. Missing a real closure would make the H.15
    term early; tests pin the boundary cases, and after the backfill every weekday without a
    value is checked against the calendar.
  - Dates before ALFRED's first DGS3MO vintage (2005-06-28) carry `pre_alfred_first_vintage`,
    decided row by row and checked.
- **Two views:**
  - `first_release` is what booked wealth uses, so a revision never rewrites history.
    `legacy_rates` gives it in the old column contract, so existing callers are unchanged.
  - `as_of(cutoff)` is for decision inputs: per date, the latest vintage available by then.
- **The backfill:** `scripts/rates_backfill.py`, run by the owner with `--keychain`.
  - Request 1: the DGS3MO observations dated 2002-12-02 to 2024-12-30, with every vintage
    published through 2024-12-31.
  - Request 2: FRED's vintage-date list, bounded at 2024-12-31, as a cross-check.
  - Nothing from 2025 on is requested, and anything served from then refuses the run.
  - The key (Keychain: service `stock-agent-fred`, account `rates`) stays in memory and is never
    printed.

The legacy store (`scripts/download_rates.py`, `data/raw/rates.parquet`) is unchanged for now.
Moving it onto vintages is a follow-up.

## Learning Notes

- **Problem.** What did investors know about the T-bill rate, and when? The rate for a day is
  published the next business afternoon, and it can be revised later.
- **Concept: vintages and real time.**
  - A data vintage is the version of a series as it stood on some date.
  - ALFRED (FRED's archive) keeps every vintage with the date it first appeared
    (`realtime_start`).
  - A backtest that uses today's values for the past can use revisions nobody had yet. That is
    lookahead, even for something as dull as a T-bill rate.
- **Assumptions.**
  - The H.15 release posts about 16:15 New York on the next Fed business day.
  - `realtime_start` is a date without a time, so a vintage counts from the end of that day:
    the later, safer time.
- **Small example (Columbus Day 2024).**
  - Thursday's 5.00% is posted Friday and counts from Friday's end.
  - Friday's 5.10% is posted Tuesday, because Monday is a Fed holiday although the NYSE trades.
    A decision on Tuesday evening still sees 5.00%; Wednesday's sees 5.10%.
- **Leakage risks closed.**
  - Using a value on its own date.
  - Using a later revision.
  - Reading any vintage published inside the 2025+ holdout.
- **How it could mislead.**
  - Before 2005-06-28 ALFRED has no first release, so those values may include early
    revisions. They are labelled.
  - A missing value on a bond-market holiday is recorded as missing, never filled.
- **Practitioner validation.**
  - Compare the stored first releases with archived H.15 releases.
  - Count the revisions; DGS3MO revisions are rare and tiny.
  - Check that every weekday with no value is a known holiday.
- **What to learn next.** Real-time data in macro forecasting (Croushore and Stark's real-time
  data set), and why GDP is the textbook case of revisions.
