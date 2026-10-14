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

Pending owner decisions this slice does not take: OD9 (the production availability rule and `signal_cutoff_at`), OD8 group A (total-return formulas, `t5_portfolio_formulas.md` section 5), OD7's fund list, OD14 (the calendar), the F1 data source, and the lockbox treatment of 2025 pay dates (below).

## What is stored

The owner asked for an auditable private dataset that keeps the source's own observations, not only daily ratios. Each batch keeps:

| Table | Columns | Key within a batch |
|---|---|---|
| `prices` | `fund_id`, `session_date`, `close`, `adjustment_basis`, `volume` (nullable), `source_id`, `batch_id` | (`fund_id`, `session_date`, `adjustment_basis`) |
| `distributions` | `fund_id`, `ex_date`, `declaration_date` (nullable), `record_date` (nullable), `pay_date` (nullable), `amount`, `amount_basis`, `distribution_type`, `source_sequence`, `source_id`, `batch_id` | (`fund_id`, `ex_date`, `distribution_type`, `source_sequence`) |
| `splits` | `fund_id`, `effective_date`, `ratio_new_per_old`, `source_id`, `batch_id` | (`fund_id`, `effective_date`) |
| `missing_sessions` | `fund_id`, `session_date`, `status`, `reason`, `detail`, `batch_id` | (`fund_id`, `session_date`) |

Closed enums:
- `adjustment_basis`: `unadjusted`, `split_adjusted_as_of_retrieval`, `split_and_distribution_adjusted_as_of_retrieval`.
- `amount_basis`: `per_share_as_paid`, `split_adjusted_as_of_retrieval`.
- `distribution_type`: `income_dividend`, `capital_gain_short`, `capital_gain_long`, `return_of_capital`, `unknown`.

Keys and columns:
- **Prices** are keyed by basis as well as date, so a raw close and a vendor-adjusted close for one session can be stored side by side. Every close is a value the source provided, with its declared basis; the store never derives one basis from another.
- **`source_sequence`** is an integer, 0 unless the source lists several distributions of one type on one ex-date (for example a supplemental income dividend, sequence 1). The adapter sets it; the store never fills it.
- **`declaration_date`** (nullable) is when the distribution was announced. It may not be after the ex-date.
- **`ratio_new_per_old`** is new shares per old share: 2.0 for a 2-for-1 split, 0.25 for a 1-for-4 reverse split.

The batch record (`batch.json`) holds `batch_id`, `adapter_id`, `adapter_version`, the request (fund ids, `start`, `end_exclusive`), the instrument mapping, `retrieved_at`, `raw_payload_sha256`, `terms_note`, `vintage_note`, `assumed_availability_rule` (always `None` in version 1), `calendar_id` and the schema version.
- The **instrument mapping** lists `{fund_id, source_symbol, source_security_id}` for every requested fund; the security id is nullable, and there is no CUSIP field. It records which source identifier each fund was retrieved under. Every requested fund must be mapped exactly once, and every row's `fund_id` must be mapped.
- The raw provider bytes are stored beside the record as `raw_payload.bin`, so a batch can be re-parsed later without a new request.
- `read_store` returns all batches plus a `batches` table with one row per batch.

**Revisions are new observations.** A later retrieval of the same fund and session is a new batch with its own `batch_id` and `retrieved_at`. Nothing is overwritten, so keys are unique per batch, not across the store. Choosing among vintages at a decision time is a later, owner-approved rule (it depends on OD9).

**The store never computes.** It does not adjust a close, fill a gap, splice a proxy or compute a return. Returns and the growth index are OD8 group A (T5-A), pending approval.

## Four timestamps

1. **Observed session date** (`session_date`, `ex_date`, `effective_date`; also `declaration_date`, `record_date` and `pay_date`). When the market event happened, as an NYSE session date stored at 00:00 UTC. It is not the time the value became known.
2. **Assumed availability: pending OD9.** The proposed convention is 18:00 America/New_York on session d, but it is not approved. The batch record keeps a nullable `assumed_availability_rule` field for later. In version 1 it must be `None`: any value, including a label for the proposed rule, is refused with `pending_od9` on write and on read. No `available_at` column exists and nothing derives one. Nothing in the store is verified point-in-time availability.
3. **Actual retrieval time** (`retrieved_at`, UTC, microseconds, from an injected clock). The only timestamp the project observes: it had the data by then. A close whose session date (at 00:00 UTC) is at or after `retrieved_at` is refused. This is a sanity check on the record, since no source can return a close for a session that had not started. It is not an availability rule and models no close time.
4. **Vintage uncertainty.** What the source's values reflected at retrieval (`vintage_note`, `adjustment_basis`, `amount_basis`). Vendors revise and backfill distributions, and adjusted levels change with every later action.

**Adjustment-as-of-retrieval leakage in price levels.** An adjusted close for an old date is computed with every split and distribution the source knew at retrieval, so its level carries later information. Example: a fund closes at $100 in 2010 and splits 2-for-1 in 2014. Retrieved in 2026 with `split_adjusted_as_of_retrieval`, the 2010 close is stored as $50. A 2010 rule such as "price below $60" would select the fund using a 2014 event. Returns over a window with no action are unaffected, but any level-based feature, threshold or price filter is not point-in-time. That is why `adjustment_basis` is recorded per row and raw (`unadjusted`) closes are preferred.

## The lockbox cap

- A request's `end_exclusive` may not pass `MODEL_HOLDOUT_START` (2025-01-01, imported from `model_validation/folds.py`, never restated). Adapters check the request before contacting any provider; ingest checks it again.
- A single row dated on or after the holdout start, in **any** date column (including `declaration_date`, `record_date` and `pay_date`), refuses the whole batch. Rows are never dropped to make a batch fit.

**Open owner question: 2024 distributions paid in 2025.** A late-December 2024 ex-date with a January 2025 pay date (or record date) refuses its whole batch under the current rule. The options:
- **(a) Keep fail-closed (current).** No 2025 date of any kind enters the development store. Simple and strict, but a fund that pays its December distribution in January cannot be stored for that month, and the batch must be re-requested with an earlier end.
- **(b) Lock on the ex-date and the declaration date only.** Pay and record dates would be treated as attributes announced at or before declaration, so a 2025 pay date could be stored for a 2024 ex-date. This changes the lockbox semantics and needs explicit approval. It would also need a rule for sources that report no declaration date.

Until the owner decides, the store keeps (a).

## Replaceable adapters

- An adapter implements `fetch(fund_ids, start, end_exclusive) -> RawBatch`: retrieval and parsing only. It returns the raw bytes, its instrument mapping and the parsed rows without storage ids.
- It calls `validation.validate_request` before retrieving, so a lockbox request never reaches a provider.
- It never writes files, assigns batch ids, adjusts, fills or computes. Swapping a provider changes one adapter and nothing downstream.
- `SyntheticFundAdapter` is the only implementation: invented funds `SYN_EQ`, `SYN_BOND` and `SYN_CASH`, with integer-cent closes from each session's position in the calendar (so values do not depend on the request window). It covers the awkward cases a real source has:
  - a 2-for-1 split (`SYN_EQ`, 2014-06-09) and a 1-for-4 reverse split (`SYN_BOND`, 2019-03-11);
  - dividends paid the following month (`SYN_EQ` in March, June and September);
  - two types on one ex-date (December income and capital gains), and a same-type supplemental with `source_sequence` 1 (`SYN_BOND` in December);
  - distribution amounts per share as paid, which change at each split.

  Every ex, record, pay and declaration date is an NYSE session. December distributions are paid in December, so the fixture never needs a 2025 date. No real tickers and no network.
- A vendor adapter needs the owner's market-data decision first. Tests stay offline and use fakes.

## Validation and diagnostics

Deterministic checks, each refusing the whole batch with a machine-readable `RefusalReason`:
- the exact column set and canonical types (`schema_mismatch`, `invalid_date`);
- known enum values (`unknown_enum`);
- finite, positive closes, amounts and split ratios, and whole non-negative volumes and sequences (`invalid_value`);
- unique keys (`duplicate_key`);
- every row's `batch_id` is the batch's own (`unknown_batch`); adapters may not set it;
- the lockbox (`lockbox_violation`, `request_after_cap`);
- no availability rule until OD9 (`pending_od9`);
- the instrument mapping covers exactly the requested funds, and every row's fund is mapped (`unmapped_fund`);
- rows inside the requested range (`outside_request`);
- price dates, ex-dates and split dates on NYSE sessions (`not_a_session`);
- distribution dates in order: pay and record dates not before the ex-date, and the declaration date not after it (`invalid_date_order`); no optional date before the calendar starts (`outside_coverage`);
- no close dated at or after retrieval (`observed_after_retrieval`; a sanity check only);
- strict parsing: JSON containing `NaN`, `Infinity`, an overflowing number such as `1e999` or a duplicate key is refused (`invalid_json`). Dates, timestamps and ids must fully match ASCII-only patterns, so a trailing newline or a full-width digit is refused. This matters because pandas parses full-width digits.

Known limitations of the date-order rules (fail-closed by choice; revisit with real data):
- For a distribution of 25% or more of the share price, the exchange rule sets the ex-date after the payable date [from memory; not verified in this slice]. Such an event has pay and record dates before its ex-date and would refuse its batch. This is rare for broad funds.
- A January 2003 ex-date declared in December 2002 has a declaration date before the calendar starts and would refuse its batch.

Sessions in the requested range with no close of any basis are not an error. They are reported in `missing_sessions` (status `unavailable`, reason `missing_session`) and never repaired.

## The private store

- The root is caller-given; there is no default. It must lie outside the repository or under an ignored path such as `data/raw/`. Vendor data is never committed.
- Each batch is staged in a hidden directory, written tables first and `manifest.json` last, then renamed to `<root>/<batch_id>` (mode 0o700). An existing batch id is refused; a failed write leaves nothing behind.
- `batch_id` is `<retrieved_at>-<12 hex>` from a hash of the record, so storing the same retrieval twice is refused.
- The reader checks the file set, symlinks, strict JSON, every file and table hash, the record's id and payload hash, then re-runs the full validation and requires the stored missing-session report to match a fresh one. Any extra file in a batch directory, including an operating-system file such as `.DS_Store`, refuses that batch.
- `read_store` skips only a crashed write's staging directory (the `.staging-` prefix). Any other entry must be a valid batch, so a hidden or renamed batch can never silently drop out of the store, and a stray `.DS_Store` in the root refuses the store.
- **Integrity, not provenance.** The hashes detect inconsistent edits: a changed file whose manifest entry was not updated. A consistent rewrite that recomputes every hash and the batch id passes them, and passes validation too if its rows obey the contract. Nothing in the store proves who wrote a batch.
- **Follow-up (not implemented):** on read, re-parse `raw_payload.bin` with the adapter's parser and require the stored tables to match, so the tables are tied to the raw bytes and not only to the manifest.

## The NYSE calendar

- `calendars/nyse.py` covers 2003-01-02 to 2024-12-31 and refuses any other date, including anything on or after the holdout start. API: `sessions(start, end)` (both inclusive, `datetime64[us, UTC]`), `is_session`, `session_date`, `CALENDAR_ID`, `COVERAGE`.
- Inputs are refused rather than coerced: a string must be `YYYY-MM-DD` in ASCII digits and a real date (`2024-02-30` and `2024-13-01` are refused); a timestamp must be naive or UTC at midnight.
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
