# F1 fund data: contract, timestamps and private store (foundations slice)

Status (2026-10-13): **synthetic fixtures only**, on `feature/f1-fund-data-foundations`; merge pending.
No real data, vendor adapter or return formula is part of this slice.

Code:
- `calendars/nyse.py` (the frozen NYSE full-day session calendar)
- `data/funds/schema.py` (enums, typed tables, the batch record, refusals)
- `data/funds/validation.py` (deterministic checks shared by ingest and the reader)
- `data/funds/adapter.py` (the `FundDataAdapter` seam and `SyntheticFundAdapter`)
- `data/funds/ingest.py` (the write-once private store)
- `data/funds/reader.py` (the fail-closed reader)

Pending owner decisions this slice does not take: OD9 (the production availability rule and `signal_cutoff_at`), OD8 group A (total-return formulas, `t5_portfolio_formulas.md` section 5), OD7's fund list, OD14 (the calendar) and the F1 data source.

## What is stored

The owner asked for an auditable private dataset that keeps the source's own observations, not only daily ratios. Each batch keeps:

| Table | Columns | Key within a batch |
|---|---|---|
| `prices` | `fund_id`, `session_date`, `close`, `adjustment_basis`, `volume` (nullable), `source_id`, `batch_id` | (`fund_id`, `session_date`) |
| `distributions` | `fund_id`, `ex_date`, `pay_date` (nullable), `record_date` (nullable), `amount`, `amount_basis`, `distribution_type`, `source_id`, `batch_id` | (`fund_id`, `ex_date`, `distribution_type`) |
| `splits` | `fund_id`, `effective_date`, `ratio_new_per_old`, `source_id`, `batch_id` | (`fund_id`, `effective_date`) |
| `missing_sessions` | `fund_id`, `session_date`, `status`, `reason`, `detail`, `batch_id` | (`fund_id`, `session_date`) |

Closed enums:
- `adjustment_basis`: `unadjusted`, `split_adjusted_as_of_retrieval`, `split_and_distribution_adjusted_as_of_retrieval`.
- `amount_basis`: `per_share_as_paid`, `split_adjusted_as_of_retrieval`.
- `distribution_type`: `income_dividend`, `capital_gain_short`, `capital_gain_long`, `return_of_capital`, `unknown`.

The batch record (`batch.json`) holds `batch_id`, `adapter_id`, `adapter_version`, the request (fund ids, `start`, `end_exclusive`), `retrieved_at`, `raw_payload_sha256`, `terms_note`, `vintage_note`, `assumed_availability_rule` (nullable), `calendar_id` and the schema version. The raw provider bytes are stored beside it as `raw_payload.bin`, so a batch can be re-parsed later without a new request. `read_store` returns all batches plus a `batches` table with one row per batch.

**Revisions are new observations.** A later retrieval of the same fund and session is a new batch with its own `batch_id` and `retrieved_at`. Nothing is overwritten, so keys are unique per batch, not across the store. Choosing among vintages at a decision time is a later, owner-approved rule (it depends on OD9).

**The store never computes.** It does not adjust a close, fill a gap, splice a proxy or compute a return. Returns and the growth index are OD8 group A (T5-A), pending approval.

## Four timestamps

1. **Observed session date** (`session_date`, `ex_date`, `effective_date`; also `pay_date` and `record_date`). When the market event happened, as an NYSE session date stored at 00:00 UTC. It is not the time the value became known.
2. **Assumed availability: pending OD9.** The proposed convention is 18:00 America/New_York on session d. It is stored only as a label, `assumed_availability_rule`, which is `None` by default; the one allowed label, `proposed_od9_session_1800_america_new_york`, names a modelled, unapproved convention. No `available_at` column exists, and nothing derives a timestamp from the label. It is not verified point-in-time availability.
3. **Actual retrieval time** (`retrieved_at`, UTC, microseconds, from an injected clock). The only timestamp the project observes: it had the data by then. As a minimum check, a close dated at or after `retrieved_at` is refused.
4. **Vintage uncertainty.** What the source's values reflected at retrieval (`vintage_note`, `adjustment_basis`, `amount_basis`). Vendors revise and backfill distributions, and adjusted levels change with every later action.

**Adjustment-as-of-retrieval leakage in price levels.** An adjusted close for an old date is computed with every split and distribution the source knew at retrieval, so its level carries later information. Example: a fund closes at $100 in 2010 and splits 2-for-1 in 2014. Retrieved in 2026 with `split_adjusted_as_of_retrieval`, the 2010 close is stored as $50. A 2010 rule such as "price below $60" would select the fund using a 2014 event. Returns over a window with no action are unaffected, but any level-based feature, threshold or price filter is not point-in-time. That is why `adjustment_basis` is recorded per row and raw (`unadjusted`) closes are preferred.

## The lockbox cap

- A request's `end_exclusive` may not pass `MODEL_HOLDOUT_START` (2025-01-01, imported from `model_validation/folds.py`, never restated). Adapters check the request before contacting any provider; ingest checks it again.
- A single row dated on or after the holdout start, in **any** date column (including `pay_date` and `record_date`), refuses the whole batch. Rows are never dropped to make a batch fit.
- Open point for the first real slice: a late-December 2024 ex-date with a January 2025 pay date refuses its batch under this rule. Whether such a pay date may be stored (it is usually declared in advance) is an owner decision; until then the batch fails closed.

## Replaceable adapters

- An adapter implements `fetch(fund_ids, start, end_exclusive) -> RawBatch`: retrieval and parsing only. It returns the raw bytes and the parsed rows without storage ids.
- It calls `validation.validate_request` before retrieving, so a lockbox request never reaches a provider.
- It never writes files, assigns batch ids, adjusts, fills or computes. Swapping a provider changes one adapter and nothing downstream.
- `SyntheticFundAdapter` is the only implementation: invented funds `SYN_EQ`, `SYN_BOND` and `SYN_CASH`, integer-cent closes from each session's position in the calendar (so values do not depend on the request window), a 2-for-1 split on 2014-06-09, and distributions whose ex, record and pay dates are NYSE sessions. No real tickers and no network.
- A vendor adapter needs the owner's market-data decision first. Tests stay offline and use fakes.

## Validation and diagnostics

Deterministic checks, each refusing the whole batch with a machine-readable `RefusalReason`:
- the exact column set and canonical types (`schema_mismatch`, `invalid_date`);
- known enum values (`unknown_enum`);
- finite, positive closes, amounts and split ratios, and whole non-negative volumes (`invalid_value`);
- unique keys (`duplicate_key`);
- every row's `batch_id` is the batch's own (`unknown_batch`); adapters may not set it;
- the lockbox (`lockbox_violation`, `request_after_cap`);
- requested funds and range (`unrequested_fund`, `outside_request`);
- price dates, ex-dates and split dates on NYSE sessions (`not_a_session`);
- no close dated at or after retrieval (`observed_after_retrieval`).

Sessions in the requested range with no close are not an error. They are reported in `missing_sessions` (status `unavailable`, reason `missing_session`) and never repaired.

## The private store

- The root is caller-given; there is no default. It must lie outside the repository or under an ignored path such as `data/raw/`. Vendor data is never committed.
- Each batch is staged in a hidden directory, written tables first and `manifest.json` last, then renamed to `<root>/<batch_id>` (mode 0o700). An existing batch id is refused; a failed write leaves nothing behind.
- `batch_id` is `<retrieved_at>-<12 hex>` from a hash of the record, so storing the same retrieval twice is refused.
- The reader checks the file set, symlinks, strict JSON, every file and table hash, the record's id and payload hash, then re-runs the full validation and requires the stored missing-session report to match a fresh one. Hashes prove integrity, not provenance: anyone who can write the root can rewrite a batch and its manifest, though not past the validation.

## The NYSE calendar

- `calendars/nyse.py` covers 2003-01-02 to 2024-12-31 and refuses any other date, including anything on or after the holdout start. API: `sessions(start, end)` (both inclusive, `datetime64[us, UTC]`), `is_session`, `session_date`, `CALENDAR_ID`, `COVERAGE`.
- The closures are a frozen tuple of dates. Tests rebuild them independently, for every year, from the holiday rules: an Easter computus for Good Friday, nth-weekday rules, Juneteenth from 2022, Sunday holidays observed on Monday and Saturday holidays on Friday unless that Friday ends a month or year (so 2004-12-31, 2010-12-31 and 2021-12-31 traded). Special closures: 2004-06-11, 2007-01-02, 2012-10-29, 2012-10-30 and 2018-12-05.
- The special-closure dates come from the ticket brief and published NYSE history; they were not re-checked against an external source in this slice.
- **Out of scope:** early closes (13:00 ET) are ordinary sessions here, with no open or close times; the SIFMA bond calendar is not represented. OD14 decides the production calendar.

## Learning notes

- **Problem.** Store fund prices and distributions so that any later return or feature can be rebuilt exactly and checked for lookahead.
- **Concept: basis and vintage.** A number is meaningful only with its basis (raw or adjusted, per share as paid or split-adjusted) and its vintage (what the source knew when we retrieved it). A total return computed from raw closes and dated distributions (`t5_portfolio_formulas.md`, F5.1) stays the same when later data arrives; a vendor's adjusted series is rescaled at every download.
- **Assumptions.** Session dates are NYSE full-day sessions. The synthetic data is invented and says nothing about markets.
- **Leakage risks.** Adjusted levels used as features; distributions backfilled after the fact; treating the retrieval date as the date information was available; a 2025 pay date slipping into the development store.
- **How practitioners validate.** Prefix stability (re-retrieve later and check that earlier observations are unchanged, or record each change as a new vintage); cross-checking raw closes plus actions against the vendor's adjusted series, as a flag only (F5.3); coverage reports from the missing-session table.
- **Next to learn.** How CRSP-style adjustment factors are built, and why the ex-date return P_ex/(P_{ex−1} − D) − 1 differs slightly from F5.1's (P_ex + D)/P_{ex−1} − 1.
