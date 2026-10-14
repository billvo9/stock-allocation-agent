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
- The adapter's raw payload is stored beside the record as `raw_payload.bin`, so a batch can be re-parsed later without a new request. By default it is the provider's bytes exactly as retrieved. An adapter may instead define it, in its own documentation, as a canonical envelope of the allowlisted fields it uses. The Tiingo adapter does this (below), following the owner's 2026-10-14 requirement not to persist future-informed payloads or their hashes. This wording was amended additively; the schema and code are unchanged.
- `read_store` returns all batches plus a `batches` table with one row per batch.

**Revisions are new observations.** A later retrieval of the same fund and session is a new batch with its own `batch_id` and `retrieved_at`. Nothing is overwritten, so keys are unique per batch, not across the store. Choosing among vintages at a decision time is a later, owner-approved rule (it depends on OD9).

**The store never computes.** It does not adjust a close, fill a gap, splice a proxy or compute a return. Returns and the growth index belong to the OD8 group A descriptive calculations (T5-A), not the store.

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

- An adapter implements `fetch(fund_ids, start, end_exclusive) -> RawBatch`: retrieval and parsing only. It returns its raw payload (see above), its instrument mapping and the parsed rows without storage ids.
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
- distribution dates in order: pay and record dates not before the ex-date, and the declaration date not after it (`invalid_date_order`); no record or pay date before the calendar starts (`outside_coverage`); a declaration date may precede it;
- no close dated at or after retrieval (`observed_after_retrieval`; a sanity check only);
- strict parsing: JSON containing `NaN`, `Infinity`, an overflowing number such as `1e999` or a duplicate key is refused (`invalid_json`). Dates, timestamps and ids must fully match ASCII-only patterns, so a trailing newline or a full-width digit is refused. This matters because pandas parses full-width digits.

Known limitations of the date-order rules (fail-closed by choice; revisit with real data):
- For a distribution of 25% or more of the share price, the exchange rule sets the ex-date after the payable date [from memory; not verified in this slice]. Such an event has pay and record dates before its ex-date and would refuse its batch. This is rare for broad funds.

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

## Tiingo adapter

Status (2026-10-14): code and offline tests only. No request has been made to Tiingo, and no Tiingo data exists in this repository.

Owner approvals and requirements (2026-10-14), and the independent review that followed:
- Evaluate the Tiingo Starter (free) API for private, local research, subject to its terms and a coverage audit, for VTI, AGG and SHV. Data is never redistributed or committed, and credentials are never exposed. The real audit runs only after the owner confirms the key is configured and authorizes the run.
- No post-2024 metadata and no adjusted historical level in any stored artifact. Only allowlisted columns are requested and stored. Splits are stored only from a verified allowlist. No distribution is stored under an unverified share basis. There is no reconciliation against vendor-adjusted series in this code; an isolated procedure is proposed below.

Code:
- `data/vendors/tiingo.py`: `TiingoAdapter`, an F1 `FundDataAdapter`. It lives outside `data/funds`, whose boundary tests ban network imports, and uses only the standard library (`urllib.request`).
- `scripts/audit_fund_coverage.py`: the coverage audit (prepared, not run).
- `tests/test_tiingo_adapter.py`: offline tests with invented fixture values and a sentinel key.

### Why an allowlist (the lockbox information-content rule)

`docs/product/architecture.md`, section 4, rule 7, says that "dated" means information content.
- **Adjusted fields.** Tiingo's adjusted fields (`adjClose`, `adjHigh`, `adjLow`, `adjOpen`, `adjVolume`) are scaled by every split and distribution Tiingo knows at retrieval, including those after 2024. So an adjusted level for a 2024 session encodes holdout-period distributions.
- **Metadata.** The metadata endpoint describes the vendor at retrieval. Its `endDate` and `description` postdate the holdout start.
- **Volume.** Raw volume's split basis is unverified, and F1's `volume` column is optional, so volume is not requested. Stored prices carry NULL volume.
- **Raw closes.** A raw close is not rescaled for later events. It is still a vendor observation, subject to correction (see the list below).

The adjusted fields, the metadata and volume are never requested, parsed or stored.

### Two phases: nothing reaches the store unverified

1. **`TiingoAdapter.retrieve`** requests, checks and parses in memory and returns a `Retrieval`. Its repr never shows the tables. It covers the field allowlist, the values, the dates and the verified splits, but not the distribution basis.
2. **`verify_distribution_basis(retrieval)`** checks every fund (below).
3. **`verified_batch(retrieval)`** is the only way to an F1 `RawBatch`. It never trusts the retrieval's in-memory tables.
   - It rebuilds the tables from `retrieval.raw_payload` (the bytes F1 stores and hashes).
   - It refuses with `payload_mismatch` when the payload's tickers, start or end differ from the retrieval's request, or when an in-memory table differs from the rebuilt one.
   - It verifies the rebuilt tables, refuses the whole batch with `unverified_distribution_basis` unless every fund passes, and builds the batch from the rebuilt tables.
   - The batch's `vintage_note` ends with "distribution basis spot-checked against primary-source amounts (a sample, not a validation of every distribution): <funds>".
4. **`TiingoAdapter.fetch`** (the F1 seam) is `verified_batch(retrieve(...))`. So a caller that passes `fetch`'s result to `ingest_batch` can only store verified data.

### Request

- **One call per fund**, prices only; the metadata endpoint is never called. The request is `GET https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=…&endDate=…&format=json&resampleFreq=daily&columns=date,close,divCash,splitFactor` [endpoint and parameter names recalled from Tiingo's documentation; unverified].
- **Date bounds.** Tiingo's `endDate` is inclusive, so the request asks for `endDate = end_exclusive − 1 day`. `end_exclusive` may be at most 2025-01-01, so `endDate` is at most 2024-12-31. A stored payload whose `start_date` is after its `end_date`, or whose `end_date` passes the cap, is refused on re-parse.
- **Refusals before any request.** A request past the cap, with more than 20 funds, or with a fund id that is not an upper-case ticker is refused before the key is read or any request is made.
- **One response at a time.** Each response is fully checked before the next ticker is requested, so a refused response stops the batch at once.
- **Failures.** Any status other than 200, including 429 and 5xx, refuses the batch. No retry is made; the owner allows at most one bounded retry.
- **Retrieval time.** `retrieved_at` comes from the injected clock, read after the last response.

### Field allowlist and fail-closed parsing

- **Exact field set.** Each element must hold exactly `date`, `close`, `divCash` and `splitFactor`.
  - Any other field refuses the batch with `non_allowlisted_field`, whether it is `adjClose`, `volume`, `high` or a field nobody has heard of.
  - A missing allowlisted field is `schema_mismatch`.

  This is how the audit learns whether Tiingo honours `columns` [unverified].
- **If Tiingo ignores `columns`, every batch is refused.** In that case the endpoint necessarily exposes information outside the lockbox boundary. The owner must then approve an isolated audit procedure (proposed below) or choose another source. The lockbox is never reinterpreted to let the data through.
- **Refused responses.** A refused response is never stored or logged. Refusal details name fields, dates and element indices, never a value. No refusal chains an exception that holds the response body: F1's `loads_strict` now raises its `invalid_json` refusal after the except block, because a `JSONDecodeError` keeps the whole document.
- **What text may contain.** Refusal text and audit output may include counts, dates and derived ratios (2 decimals), never prices, amounts or the key.
- **Other refusals:**
  - a wrong type (a string, `null` or a boolean where a number belongs);
  - a non-finite number (`NaN`, `Infinity`, `1e999` or an integer beyond the float range);
  - a non-positive close, a negative `divCash` or a non-positive `splitFactor`;
  - a date that is not exactly `YYYY-MM-DDT00:00:00.000Z`, a repeated date, or a date outside the request;
  - any date on or after the holdout start (`lockbox_violation`).

  Vendor-specific reasons are `TiingoRefusal` values on a `FundDataRefused` subclass, so F1's own `RefusalReason` is unchanged.

### Mapping

| Tiingo | F1 |
|---|---|
| `date` | `session_date`, `ex_date` or `effective_date` |
| `close` | a `prices` row with basis `unadjusted` (the only basis this adapter writes); `volume` NULL |
| `divCash > 0` | a `distributions` row: `amount = divCash`, `per_share_as_paid` (stored only after the basis is verified), type `unknown`, `source_sequence` 0; declaration, record and pay dates NULL |
| `splitFactor != 1` | a `splits` row only for a verified share-count split (below); any other factor refuses the batch |
| the requested ticker | `fund_id` and `source_symbol`; `source_security_id` NULL |

### Splits: verified share-count changes only

Tiingo's `splitFactor` is not assumed to be a pure share-count change. A split is stored only if its (ticker, session) is in the code-reviewed `VERIFIED_SPLITS` allowlist, and every entry cites a primary source:

| Ticker | Session | New shares per old | Source |
|---|---|---|---|
| VTI | 2008-06-18 | 2.0 | SEC EDGAR 0000036405/000093247108001251, "split-adjusted trading from 2008-06-18" [owner-provided; not fetched here] |

Every check expected near 1 uses one fixed band, `BAND = log 1.25`.
- **`unverified_split_factor`:** `splitFactor != 1` on any session that is not in the allowlist.
- **`split_factor_not_discriminable`:** an allowlisted factor with |log f| < 3 × BAND (roughly 0.512 < f < 1.953) is refused, even though it is listed.
  - For a discriminable factor, a ratio within BAND of f lies at least 2 × BAND away from 1.
  - A test checks that every `VERIFIED_SPLITS` entry meets this rule. It claims nothing more.
- **`split_factor_mismatch`:** an allowlisted session inside the request whose vendor factor differs, is 1, or whose row is missing.
- **`split_price_mismatch`:** at an allowlisted split, r = (raw close × f) / the previous session's raw close fails |log r| < BAND. The raw close may already be split-adjusted. The same reason applies when the previous session's close is absent, including a request that starts on the split date.

### Distribution-basis verification (every fund must pass)

F1's `AmountBasis` has no "unknown" value, and adding one would be a contract change needing owner approval. So instead of labelling an unverified amount, the adapter refuses to store it.

**The gate (a fail-closed lead decision).** Storage requires primary-source amounts for every fund. The split ratio is only a consistency check: it can fail a fund but never pass one. Relaxing this gate would be an owner lockbox decision. A fund passes only when all four hold:
1. **Anchored amounts.** At least 2 entries of the code-reviewed `VERIFIED_DISTRIBUTIONS` allowlist fall in the window, each matching the vendor's `divCash` on the same ex-date within 1e-6, with no mismatch.
   - Entries are primary-source amounts per share as paid, keyed (fund, ex-date), each with a citation comment.
   - A listed date with no served distribution counts as a mismatch.
2. **Both sides of every split.** For every verified split in the window, at least one listed match lies strictly before the split session and one strictly after it. A vendor rescales every amount between one split and the next by the same factor, so matches on one side prove nothing about the other.
3. **The split ratio, when computable.** With at least 4 distributions on each side, m = median(last 4 before) / (f × median(first 4 after)) must satisfy |log m| < BAND.
   - Per-share-as-paid amounts fall by f at the split, so m ≈ 1.
   - Split-adjusted amounts give m ≈ 1/f.
   - Outside the band the fund fails, whatever the listed amounts say. With fewer than 4 on a side the check is "not computable": it neither passes nor fails the fund.
4. **No distribution on a split session.** A distribution whose ex-date is a verified split session fails the fund, because its share basis cannot be verified. The T5 specification treats an unknown basis on a split date as unavailable.

**What each check reports.** Each fund's `BasisCheck` gives `passed`, `route` ("primary_source_spot_check" when passed) and `split_check`:
- `no_split`: there is no verified split in the window.
- `not_computable`: fewer than 4 distributions on a side.
- `agreed`: every computable ratio is within the band.
- `failed`: a ratio is outside the band.

The detail holds counts and ratios only.

**Owner step.** `VERIFIED_DISTRIBUTIONS` starts empty: amounts are never invented. So nothing stores today, VTI included.
- For every fund, add at least two sourced per-share amounts in each window you want stored. Each comes from the fund sponsor's distribution history or an SEC filing, with its citation in a comment.
- For VTI across 2008-06-18, include at least one amount on each side of the split.
- A window with no distributions cannot be verified, so it is not stored.

**What neither route detects.** Neither route detects a later split applied uniformly to the whole closing-price history. Route (b) anchors amounts only. Closes are checked against an external fact only at verified split sessions. A uniform rescaling of closes alone would distort D/P, and it is undetectable without a primary-source close anchor. **Follow-up:** add primary-source close anchors (for example sponsor-published closing prices or NAVs on a few dates) as an owner-reviewed allowlist.

### Evidence levels: what a stored batch does and does not show

A stored batch passed checks of four different strengths. Only the first is a property of every row.

| Level | Check | Rows covered | Shows | Does not show |
|---|---|---|---|---|
| Enforced | Field allowlist, date bounds, value types, split-factor allowlist | Every element of every response | No adjusted field, metadata, volume, post-2024 row or unlisted split factor reached the store | That any served value is correct |
| Spot check | Primary-source amounts (rules 1 and 2) | The listed ex-dates only: at least 2 per fund and 1 on each side of each split | Those amounts equal the issuer's. Assuming the vendor uses one basis between consecutive splits, each split segment is per share as paid | That any other amount is correct; that no distribution is missing or extra; that nothing was revised |
| Consistency | Split-ratio check (rule 3) and the raw close at the split session | Up to 4 + 4 distributions, and 2 closes, around each verified split | The basis does not change across that split | Anything about a rescaling applied to both sides, such as a later split applied uniformly |
| Not checked | Closes on every other session; completeness of the distribution list; distribution types and pay dates | — | Nothing | — |

Two matched amounts do not certify 18 years of distributions, and one split-session price check cannot detect a uniform rescaling of the whole history. Reports and notes therefore say "spot-checked", never "validated". Stronger evidence would need more anchors, for example a full issuer distribution history compared row by row, or primary-source close anchors (the follow-up above).

### Raw payload and hashes

- **The payload: an adapter-defined canonical envelope.** F1's contract lets an adapter define its raw payload instead of storing the provider's bytes, provided the adapter documents it (see "What is stored"). Tiingo's payload is:
  - **Content:** `{"envelope_version": "4", "source_id": "tiingo", "start_date", "end_date" (inclusive, as requested), "fields": ["date", "close", "divCash", "splitFactor"], "prices": {ticker: [elements]}}`. Each element holds exactly the four allowlisted fields, in the order served.
  - **Serialization:** `schema.dumps_canonical`: JSON with sorted keys, separators `,` and `:`, no whitespace, UTF-8 (`ensure_ascii=False`), and non-finite numbers refused.
  - **Numbers are re-serialized.** Values pass through IEEE-754 doubles and are written in Python's shortest round-trip form. Each value is preserved exactly as a double, but the vendor's original digit strings, such as trailing zeros, are not. The payload therefore cannot show the vendor's reported decimal precision.
  - **What the hash means.** `raw_payload_sha256` attests that the stored envelope is unchanged since ingestion. It does not attest that the envelope matches the vendor's whole response, or that the values are a point-in-time vintage.
  - **Re-parsing.** `parse_payload` rebuilds the stored tables from the envelope alone, and `verified_batch` refuses when its in-memory tables differ.
  - **Versioning.** Any change to the content or serialization needs a new `envelope_version`, and `parse_payload` refuses other versions.
- **No original hashes.** No hash of an original vendor response is kept, because a hash of a future-informed payload must not be stored as if it were an admissible input.
- **The notes.** `vintage_note` records:
  - the fields that were not requested: the metadata endpoint and `adjClose`, `adjHigh`, `adjLow`, `adjOpen`, `adjVolume`, `high`, `low`, `open` and `volume`;
  - the vintage caveats below;
  - the funds whose distribution basis was spot-checked against primary-source amounts (a sample, not a validation of every distribution).
- **Re-parsing.** `parse_payload` re-runs every element check on the stored bytes, and the audit requires the stored tables to equal a re-parse.

### What Tiingo does not provide

- **Distribution dates and types.** Tiingo gives no declaration, record or pay date, and no distribution type: `divCash` is one number per ex-date, so income and capital gains are not separated. These stay NULL and `unknown`, and are never invented. The open 2025 pay-date question above cannot arise for Tiingo rows, but nothing downstream can use Tiingo for pay timing.
- **Security identifier.** No stable security id is requested, so instruments are mapped by ticker only.
- **Revision history.** Tiingo serves only its current view of history.

### What cannot be a true point-in-time vintage

Each batch's `retrieved_at` is the only observed timestamp, and a later retrieval is a new batch. None of the following is known as it stood on its session date:
- **Raw closes** are as served at retrieval, and vendor corrections are possible. They are checked against an external fact only at verified split sessions. A uniform rescaling of closes alone, which would distort D/P, is undetectable without a primary-source close anchor (a follow-up, above).
- **`divCash` amounts** may have been revised. They are stored only after the primary-source spot check (above), which checks the listed ex-dates only.
- **Split factors** are as the vendor reports them; only allowlisted ones are stored.
- **The ticker-to-security mapping and the set of sessions served** are Tiingo's view at retrieval.
- **A later split applied uniformly** to the whole closing-price history is detected by neither route; route (b) anchors amounts only.
- **No declaration, record or pay dates** are available.
- **No revision history** comes from Tiingo, so a correction cannot be dated.

### Secrets

- **Header authentication.** The key travels only in the `Authorization: Token <key>` header. It never appears in a URL, the raw payload, the batch record, an exception, a log or the audit output. A test scans every stored byte for a sentinel key.
- **Reading the key.** It comes from `TIINGO_API_KEY` at retrieval time (an injected provider in tests). It is wrapped in an object whose repr is redacted, and is never stored on the adapter. A missing or malformed key raises `TiingoRequestFailed` (`missing_token` or `invalid_token`) without echoing the value.
- **The variable `TIINGO_API_KEY`.**
  - The name is a project convention (`tiingo.TOKEN_ENV`), read only by `env_token`. Tiingo's API does not read environment variables: it takes the key in the `Authorization` header.
  - Set it per shell session, as in the audit example below. Do not put it in a committed file. `.gitignore` ignores `.env` and `.env.*`, and the repository has no example file holding a key.
  - The owner configures it. Code never writes, prints or logs it, and tests use a sentinel value.
- **Network failures.** The expected types (`URLError`, `HTTPException`, `OSError`, `TimeoutError`, `ValueError`) are reported by type name only, together with the reason, ticker and HTTP status. The error is raised outside the except block, so no chained exception, header or response body reaches a traceback.
  - A `ValueError` from an invalid header value would contain the key, which is why `ValueError` is in this list.
  - Any other exception type propagates unchanged, so a programming error is not hidden.
- **The default transport.**
  - It builds its opener with `ProxyHandler({})`, so environment proxies are ignored.
  - It follows no redirects; a 3xx becomes an `http_error` refusal with its status and an empty body.
  - It sends the key as an unredirected header, which is never copied onto another request.

### Running the coverage audit (after the owner configures the key and authorizes it)

```bash
read -rs TIINGO_API_KEY && export TIINGO_API_KEY    # typed, not echoed or kept in history
PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized    # read-only report
```

Persisting a real fund batch needs the owner's separate approval, and then `--store --root ~/stock-agent-private/fund_store` is added. Without `--store` the audit writes nothing anywhere. `--store` and `--root` go together, and either one alone is refused.

- **Defaults.** `--funds VTI AGG SHV`, `--start 2007-01-01`, and `--end-exclusive` at the holdout start (2025-01-01). That makes three requests. The default run is a read-only report: phase 1 only, ending with `NOT STORED read-only coverage report`, and exit code 0 when the retrieval succeeds.
- **Refusals before anything runs.**
  - Without `--authorized` the script exits with code 2.
  - It refuses a root inside any repository, unless the root lies strictly inside that repository's ignored `data/raw/`. It walks every existing ancestor of the root and matches each with `os.path.samefile`, which also catches a case-variant path on macOS's case-insensitive file system. Any ancestor that is this repository, or holds a `.git` (another clone or a worktree), counts as a repository.
  - It also refuses a root that contains the repository.
- **Root permissions** (with `--store` only).
  - A missing root is created with mode 0o700.
  - An existing root is never changed. It must be a directory whose mode grants no group or other permission (`mode & 0o077 == 0`); otherwise the script exits with code 2 and asks the owner to restrict it (for example `chmod 700 <root>`).
- **Phase 1 (in memory, aggregates only).** It prints:
  - the request and the allowlist confirmation;
  - row counts, and the first and last session;
  - missing NYSE sessions (before the first session, inside the range, after the last);
  - distributions per year;
  - verified splits, with the raw-close ratio the adapter enforced;
  - each fund's distribution-basis verification, with its split check (counts and ratios).

  It never prints a price, an amount or the key.
- **Phase 2 (store; only with `--store`, after the owner's separate approval).** It stores only when every fund passed. Otherwise it prints `NOT STORED unverified_distribution_basis: <funds>` and exits with 1. After storing, it reads the batch back and compares the stored tables with a re-parse of the stored payload; if they differ, it says so and exits with 1.
- **What it never does.** It requests no adjusted series and performs no vendor-adjusted reconciliation (it says so). It computes no return: returns belong to the OD8 group A descriptive calculations, not the audit.
- **Exit codes.** A refused retrieval prints `REFUSED <reason>` and exits with 1, with nothing stored. A re-run repeats the requests.

### Still unverified (the audit checks these)

- **Columns.** Whether Tiingo honours `columns`. If it does not, every batch is refused.
- **`divCash` basis.** Whether `divCash` is per share as paid. Primary-source amounts decide this; the split ratio at VTI's 2008-06-18 split can only contradict it.
- **`splitFactor` meaning.** Whether `splitFactor` is new shares per old, on the effective session. The audit requires 2.0 on 2008-06-18 for VTI.
- **Raw closes.** Whether the raw close is truly unadjusted. The split's raw-close ratio must be near 1; if Tiingo's `close` were split-adjusted, it would be near 2.
- **Date format.** The exact `date` format.

### Proposed, not implemented: an isolated vendor-adjusted reconciliation

Comparing the project's total return with a vendor's adjusted series (`t5_portfolio_formulas.md`, F5.3) needs adjusted fields. They carry post-2024 information, so this code stops before them. If the owner wants the comparison, a separate procedure could run as follows (each step needs the owner's explicit approval):
1. **A separate process.** It is not the adapter, the audit script or the research store, and it runs only with a recorded owner approval for that run.
2. **Temporary storage only.** It requests the adjusted fields into a temporary directory (mode 0o700) outside the research store and every repository.
3. **Aggregates only, up to 2024-12-31.** It computes only aggregate reconciliation statistics for sessions up to 2024-12-31, for example the count of sessions per fund and year where the vendor's daily adjusted ratio and the OD8 group A total return differ by more than a pre-registered tolerance.
4. **Deletion.** It deletes the response files before exiting, also on failure. It keeps no row-level output, adjusted value or response hash.
5. **What it records.** The run date, the request, the tolerance and the aggregates.

One caveat: a daily ratio of adjusted closes cancels later adjustment factors only if the vendor applies one consistent methodology to the whole history it serves. It is not claimed to remove all vintage leakage, which is why nothing from such a run would be persisted.

### Learning notes: the split and share-basis checks

- **Problem.** Make sure a vendor's split and dividend fields mean what the store's labels say, without using any adjusted series.
- **Raw close across a split.** An unadjusted price falls by the split factor f on the effective session, so r = (P_t × f) / P_{t−1} ≈ 1 + the day's market move.
  - Example: P_{t−1} = 120.00, a 2-for-1 split, P_t = 60.60, giving r = 60.60 × 2 / 120.00 = 1.01, so |log r| ≈ 0.01, which passes.
  - If the vendor's "raw" close were already split-adjusted, P_{t−1} would read 60.00. Then r = 2.02 and |log r| ≈ 0.70, which is refused.
- **The fixed band, BAND = log 1.25 ≈ 0.223.**
  - It allows a ±25% daily move; a broad ETF moving that much in a day would be extraordinary.
  - A split is usable only if |log f| ≥ 3 × BAND. For f = 2, |log 2| ≈ 0.693 ≥ 0.669. In log terms the two cases are then centred at least 3 band widths apart: "unadjusted" at 0 and "already adjusted" at log f. Their acceptance intervals, each one band wide on either side, leave a gap of at least one band width between them.
- **The split ratio for distributions, with worked examples** (f = 2; the tests run these numbers).
  - **Set-up.** After the split VTI pays D_a per new share, or 2 × D_a per old share. Before it, the issuer paid D_b per old share. Let R = D_b / (2 × D_a): R = 1 means an unchanged dividend per old share, and R > 1 means a cut at the split.
  - **As paid,** the vendor serves D_b and D_a, so m = D_b / (2 × D_a) = R. The check agrees when 0.8 < R < 1.25. A correct series therefore fails after a cut of more than 20% at the split, or a rise of more than 25%. Those failures are false refusals, and they fail closed.
  - **Split-adjusted,** the vendor serves D_b / 2 before the split, so m = R / 2. The check wrongly agrees when 0.8 < R / 2 < 1.25, that is 1.6 < R < 2.5. At that point the dividend per old share fell to between 40% and 62.5% of its pre-split level, **a cut of 37.5% to 60%**. A false agreement needs a large dividend *cut* at the split, never a rise.
  - **Example 1 (no change).** D_b = 0.60 and D_a = 0.30. As paid, m = 0.60 / 0.60 = 1.00, which agrees. Split-adjusted, m = 0.30 / 0.60 = 0.50 and |log m| = 0.69, which fails.
  - **Example 2 (a 40% cut).** D_b = 1.00 and D_a = 0.30, so after the split the dividend per old share is 0.60. Split-adjusted, m = 0.50 / 0.60 = 0.83 and |log m| = 0.18 < 0.223: a false agreement. As paid, m = 1.00 / 0.60 = 1.67, which fails: a false refusal. The spot check catches the false agreement, because the issuer's pre-split amount (1.00) does not match the vendor's 0.50. This is why rule 2 needs a listed amount on each side of every split.
- **Why relative checks are not enough.** A ratio test across a split compares one side with the other. A vendor that rescales everything after the fact (for a later split) moves both sides together and passes it. Only absolute anchors, meaning primary-source amounts or closes, catch that. This is why storage requires listed amounts on both sides of every split.
- **How it can mislead.**
  - A fund that changes its payout near the split can move m outside the band. That fails closed, and the owner can check the listed amounts.
  - Listed amounts are only as good as their sources; every entry must cite one.
- **Leakage.** Every input is a raw observation dated at or before 2024-12-31.
- **Validation.** Compare with the fund's own distribution notices and the SEC filing cited in the allowlist.
