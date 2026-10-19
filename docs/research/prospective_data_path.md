# Prospective data path: real ETF data for frozen forecasters

Status (2026-10-14): code, offline tests and owner commands only. No request has been made to Tiingo, no vendor or issuer data exists in this repository, and no private file was read. The live runs below are the owner's, in the owner's terminal.

This slice builds the data path that the prospective forecasting track needs: a calendar beyond 2024, the owner's private distribution anchors, a separate prospective store, a Tiingo prospective mode, and the one loader that frozen forecasters may use.

## Owner decisions (2026-10-14, `docs/PROJECT_STATE.md`, verbatim where quoted)

- Path: "Start prospective track". ETF forecasters at 5, 20 and 30 sessions are registered and frozen before the first forecast; each forecast is recorded once, timestamped, labelled "unvalidated, no demonstrated skill", and scored only as real outcomes arrive.
- H_end: "H_end today, lookback OK". "The sealed holdout is 2025-01-01 to 2026-10-14 (H_end), kept for one historical evaluation. A separate prospective store may use the last 63 sessions before H_end only as inputs to frozen models, never to train, tune or evaluate. A test enforces that research code cannot read the prospective store."
- Anchors: "Fill by hand". "The owner copies issuer amounts into the private, git-ignored anchors file (issuer terms: personal use, no automated access). The rows the research agents gathered are not used as anchors."
- Live data: "Audit + prospective". "First the read-only pre-2025 Tiingo audit (VTI, AGG, SHV), then collection into the separate prospective store. Each run happens in the owner's terminal, and storing anything still needs the owner's OK at that step." Lead implementation choice: "the first collection requests only the 63-session lookback before H_end and later sessions; the rest of the sealed period is not collected until the one-time historical evaluation."
- OD9 (descriptive display only): "use 18:00 America/New_York as an explicitly assumed close-availability convention. This is not proof of final vendor publication time and must not be used as evidence of historical trading executability or forecasting validity."

## What was added

| Module | Role |
|---|---|
| `calendars/nyse_rules.py` | Rule-based NYSE closures; equal to the frozen list through 2024; coverage through 2030-12-31 since 2026-10-18 (section 1) |
| `data/prospective/window.py` | `H_END`, `LOOKBACK_SESSIONS`, `WINDOW_START`, the complete-session request rule |
| `data/prospective/validation.py` | The store marker, the batch record and the batch checks (shared by writer and reader) |
| `data/prospective/store.py` | The write-once writer (the only module that writes into a prospective root) |
| `data/prospective/inference.py` | `load_inference_window`, the only reader of the prospective store |
| `data/vendors/anchors.py` | The loader for the owner's private anchors file (replaces the in-code `VERIFIED_DISTRIBUTIONS`) |
| `data/vendors/tiingo_prospective.py` | Tiingo prospective mode: `collect_prospective` and the prospective gate |
| `data/private_paths.py` | Root and path rules shared by the scripts and the anchors loader |
| `scripts/collect_prospective.py` | The owner's collection command (read-only by default; `--store`) |
| `scripts/check_anchors.py` | Validates the anchors file offline; prints counts, never amounts |
| `docs/templates/distribution_anchors_template.csv` | The header-only anchors template |

The research path is unchanged in what it accepts: the frozen calendar `calendars/nyse.py`, the F1 store and the research Tiingo adapter still refuse anything on or after 2025-01-01. The research adapter now takes its anchors explicitly (adapter version 5; the stored payload format is unchanged, envelope version 4).

## 1. The NYSE calendar beyond 2024

`calendars/nyse_rules.py` generates full-day closures from the exchange's holiday rules (NYSE Rule 7.2):
- New Year's Day; Martin Luther King Jr. Day (third Monday of January); Washington's Birthday (third Monday of February); Good Friday (two days before Western Easter, from the anonymous Gregorian computus); Memorial Day (last Monday of May); Juneteenth (June 19, from 2022); Independence Day; Labor Day (first Monday of September); Thanksgiving Day (fourth Thursday of November); Christmas Day.
- A Sunday holiday closes the Monday after. A Saturday holiday closes the Friday before, unless that Friday is the month's last weekday (it ends an accounting period). New Year's Day on a Saturday therefore closes nothing. NYSE's published page agrees for 2028 ("Because the holiday falls on Saturday, January 1, 2028, no New Year's Day holiday is observed"), so 2027-12-31 trades.

**Two calendar versions, each resolvable by its id** (`nyse_rules.calendar(id)`). Both come from the same rules and special closures, and are identical through 2027-12-31: a test shows the 2027 version's sessions are a byte-identical prefix of the 2030 version's. Only the two listed (id, last year) pairs can be constructed.

| Id | Coverage | Role |
|---|---|---|
| `nyse_full_day_rules_2003_2030_v1` | 2003-01-02 to 2030-12-31 | `CALENDAR_ID` since 2026-10-18: the calendar every consumer uses and records. Registered code needs no calendar change before C24, if session 1 is no later than 2028-11-29 |
| `nyse_full_day_rules_2003_2027_v1` | 2003-01-02 to 2027-12-31 | The earlier default. It stays resolvable by its id |

**Where each year's closures come from** (`nyse_rules.YEAR_SOURCES`, `year_source(year)`).
- "Verified" describes the holiday schedule, not the sessions: it equalled NYSE's published page as of `retrieved_on`.
- A session after that date stays an expected session, because an unscheduled closure can still occur.
- Every session in an unverified year is expected.
- `to_json_dict` says so (`schedule_verified`, `note`).

| Years | Status | Source |
|---|---|---|
| 2003-2024 | `frozen_research_list` | Equal to `nyse.FULL_DAY_CLOSURES`. Its special closures were not re-checked externally (`f1_fund_data.md`) |
| 2025 | `rule_generated` | The rules plus the special closure 2025-01-09. NYSE's 2025 page was not retrieved |
| 2026-2028 | `verified_published` | NYSE, "Holidays & Trading Hours", https://www.nyse.com/markets/hours-calendars, retrieved 2026-10-18 |
| 2029-2030 | `rule_generated` | The rules only. Not yet published when retrieved |

A verified source must carry an `https://` URL and a retrieval date, and any other source must carry neither.

`check_published(year, closures)` compares a published list with the rules.
- It reports closures the rules have but the list lacks (`unexpected`), and the reverse (`missing`).
- It refuses a malformed list (`invalid_date`) and a date listed more than once (`duplicate_date`).
- When NYSE publishes 2029, the year can be verified with it. Marking a year verified edits `YEAR_SOURCES` and the test's embedded published lists, which is a code change.

**The switch of `CALENDAR_ID` to the 2030 version (approved by the owner on 2026-10-18; done).** The switch was one line in `nyse_rules.py`. It is breaking under AGENTS.md, because two stores compare the recorded id exactly:
- The prospective store's marker and batch records hold `calendar_id`. A root or batch written under the 2027 id is refused (`not_prospective_store`).
- The Stock Lab inference window refuses a batch recorded under another id (`calendar_mismatch`).

No fund, prospective or Stock Lab inference store existed when it was made, so no real data was invalidated. The forecast ledger's `calendar` and `checked_against` fields are recorded, never compared. The switch is a change under `src`, so it lands before the freeze.

Tests:
- the default is the 2030 version, and the module-level calendar refuses dates after 2030-12-31;
- a prospective store marker under the 2027 id is refused;
- the generated 2003-2024 closures equal `nyse.FULL_DAY_CLOSURES` exactly, year by year, in both versions;
- the extended sessions through 2024 equal the frozen sessions byte for byte;
- the frozen calendar's contract is pinned: its id, coverage, session count (5,537) and the sha256 of its sessions and closures;
- 2026-2028 equal NYSE's published closures exactly. The test embeds the list with its citation and makes no request. The published early closes (2026-11-27, 2026-12-24, 2027-11-26, 2028-07-03, 2028-11-24) are sessions;
- 2025 matches a hand-entered list (as recalled) and is labelled rule-generated;
- 2029 and 2030 equal hand-computed expected closures (pinned, labelled rule-generated);
- each version's full session list is pinned by count and sha256 (2027 version: 6,289 sessions; 2030 version: 7,042);
- observance through 2030, including 2026-07-03, 2027-06-18, 2027-07-05, 2027-12-24 and 2027-12-31 (a session);
- Good Friday 2029-03-30 and 2030-04-19;
- 251 sessions in each year from 2026 to 2030, and 250 in 2025;
- source labels per year: exactly the years with an embedded published list are verified, each with its citation;
- negative checks:
  - a fabricated closure, a missing holiday and a wrong observance each fail `check_published`;
  - a duplicate date is refused;
  - a verified source without a citation is refused;
  - an unknown (id, last year) pair is refused.

**Unscheduled closures cannot be predicted.** National days of mourning, storms and emergencies are announced at short notice. `SPECIAL_CLOSURES` holds the five in the frozen list and 2025-01-09 (the national day of mourning for Jimmy Carter). A closure announced later is missing until someone adds it with a source; until then the prospective store reports that date as a missing session for every fund (never filled), and a session count across it is one too high. No network fetch is made.

### Unscheduled closures after registration: proposed rule ETF-R2 (design only; owner decision pending)

Nothing here is implemented. Until the owner approves ETF-R2, the code keeps today's semantics for the trial, session numbering, cadence and confirmation (`prospective_registration.md`; `forecasting/prospective/cadence.py`).

**Why a rule is needed.** Without one, an unscheduled closure remains a calendar session with no close for any fund.
- The window ending there is incomplete for every fund, and so are the next 63 windows.
- Every fund is refused. During the trial that is a void. After it, those 64 sessions (the closure and the next 63) are about 12% of the 524 sessions from session 1 to C24 (64/524 ≈ 12.2%), beyond C3's 5%. `prospective_registration.md`, section 12, rounds it to 13%.
- Adding the date to `SPECIAL_CLOSURES` changes `src`, so it would be a new registration.

**Proposed rule.**
1. **Evidence, never inference.** A closure enters only as an amendment.
   - An amendment is one line appended to a private, append-only, hash-chained amendments file. Each line holds the date, the occasion, the exchange notice's `https://` URL (NYSE or ICE), the retrieval time (UTC), the sha256 of the retrieved notice, the recording time (UTC) and the previous line's sha256.
   - The file lives outside every checkout. The forecast job runs from a checkout pinned at the registration commit, so a file inside it could never change.
   - Nothing creates an amendment from missing vendor prices. A date without a close for every fund stays a reported missing session until an amendment exists.
   - Validation refuses:
     - a date that is not a session of the registered calendar;
     - a duplicate date;
     - a recording time before the retrieval time;
     - a malformed URL or hash;
     - any change to an earlier line.
2. **A non-session from its recording time.** A record made at or after the amendment's recording time uses the registered calendar minus the amended dates. The lookup is as of the record's own decision time, backward only.
   - That date is not a model error, a missed or refused session, or a data gap.
   - No price is invented for it.
   - Inference windows stay the 64 actual closes ending at the as-of close. A window spanning the closure reaches one session further back.
3. **The ledger and every recorded forecast, refusal and score are never altered.** A record already made for the closure date stays in the ledger. It is reported, but it does not count as a session record.
4. **Late amendments excuse nothing else.** An amendment recorded after a session's recording deadline does not excuse that session's refusals or gaps. Only the closure date itself is reclassified.
   - The alternative is to also excuse refusals caused only by that date. It is more forgiving, but it lets gaps be explained after the fact.

**Effects** (per amended closure after session 1):
- **Session numbering.** Session n is the n-th session of the amended calendar from session 1. Every session after the closure is numbered one lower than on the unamended calendar. Numbers never skip, and session 1 does not move.
- **Trial status.** The trial is still sessions 1-20; it ends one session later in calendar time. The gap rule counts expected sessions only, so the closure cannot start or extend a gap.
- **Open forecasts.** Exit = entry + h sessions of the amended calendar (horizons count trading sessions). A forecast spanning the closure exits one session later in calendar time.
  - Scoring computes exits from the calendar as of the scoring time; exits are not stored at recording, so no forecast entry changes.
  - This uses an amendment recorded after the forecast, but only on the outcome side: it fixes which close is the h-th trading session after entry. It is never a model input or a feature, so it is not leakage. The forecast's inputs remain what was visible at its own decision time.
  - Before the amendment, the missing close keeps such a forecast pending (`outcome_inputs_unavailable`). It cannot be scored on the wrong session.
- **Checkpoints.** C12 (session 272) and C24 (session 524) keep their session numbers and registered outcome counts (section 10's table is unchanged). Their calendar dates move one session later.

**Code identity.** The amendments file is data outside the registered code identity: appending to it is not a new registration and does not fail C4.
- The code that reads and validates the file is registered. It must be built before the freeze, in the forecast job, scoring and Model Health.
- Every forecast, refusal and score entry records the head sha256 and line count of the amendments it used, so each record's calendar can be reproduced.
- A rewritten or removed amendment line is an integrity failure under C3, like a failed kept head.

**Numerical example** (hypothetical; `tests/test_etf_r2_closure_example.py` reproduces every date).
- Session 1 is Monday 2026-11-02.
- On Friday 2026-11-13 (session 10), NYSE announces a full-day closure on Wednesday 2026-11-18.
- The amendment is recorded that day at 21:00 UTC (16:00 New York), before session 10's recording window opens at 18:00 New York.
- Sessions come from `nyse_full_day_rules_2003_2030_v1`, which is identical to the 2027 version through 2027-12-31. C24 falls in 2028, beyond the 2027 version.

| | Without an amendment (today) | With ETF-R2 |
|---|---|---|
| 2026-11-18 | Session 13, expected. Every fund lacks its close, so every fund is refused: a trial void | Not a session: nothing is expected or recorded, and no price exists |
| Sessions 13-20 | 11-18, 11-19, 11-20, 11-23, 11-24, 11-25, 11-27, 11-30 | 11-19, 11-20, 11-23, 11-24, 11-25, 11-27, 11-30, 12-01 |
| Last trial session (20) | 2026-11-30 | 2026-12-01 |
| Session 21 (first confirmatory) | 2026-12-01 | 2026-12-02 |
| C12 (session 272) | 2027-12-01 | 2027-12-02 |
| C24 (session 524) | 2028-11-30 | 2028-12-01 |
| Window of the next forecast, as-of 2026-11-19 | Session 14: the 64 sessions 2026-08-21 to 2026-11-19, including 11-18 without a close, so incomplete and refused | Session 13: the 64 actual closes 2026-08-20 to 2026-11-19 |

Open forecasts (entry = the session after the as-of; exit = entry + h sessions). 2026-11-27 is an early close, so it is a session.

| As-of (session) | Entry without / with | h = 5 without / with | h = 20 without / with | h = 30 without / with |
|---|---|---|---|---|
| 2026-11-12 (9), recorded before the announcement | 11-13 / 11-13 | 11-20 / 11-23 | 12-14 / 12-15 | 12-29 / 12-30 |
| 2026-11-17 (12), recorded after the amendment | 11-18 / 11-19 | 11-25 / 11-27 | 12-17 / 12-18 | 2027-01-04 / 2027-01-05 |

- Without the amendment, these outcomes span the missing 11-18 close, so they stay pending (`outcome_inputs_unavailable`) and are never scored on a wrong session.
- With ETF-R2, they are scored at the later exits.
- In both cases the ledger entries are byte-identical, because exits are computed at scoring and never stored at recording.
- Session 1, sessions 1-12 and every earlier record are unchanged.

> **ETF-R2 (owner decision, pending): unscheduled NYSE closures during the prospective trial and confirmation.**
> No code may change trial, session-numbering or confirmation semantics for this rule before the owner approves ETF-R2. The amendment reader and validator are then a separate change, built and reviewed before the freeze.
>
> "I approve ETF-R2 for every prospective registration.
> (1) An unscheduled full-day NYSE closure enters the prospective calendar only through an amendment appended to the private, append-only, hash-chained amendments file. The amendment holds the date, the occasion, the exchange notice URL, the retrieval time (UTC), the sha256 of the retrieved notice and the recording time. It is never inferred from missing vendor prices.
> (2) From its recording time the date is not an exchange session. It is not a model error, a missed session, a refused session or a data gap, and no price is invented for it.
> (3) Inference windows remain the 64 actual closes ending at the as-of close.
> (4) The ledger and every recorded forecast, refusal and score stay unaltered. A record already made for that date is reported but not counted.
> (5) Sessions are numbered on the amended calendar. The trial (sessions 1–20), forecast exit sessions (entry + h sessions) and the checkpoints C12 (session 272) and C24 (session 524) keep their session counts and move later in calendar time by one session per closure. Exit sessions are computed with the calendar, amendments included, as of the scoring time. That is the outcome side, never a model input or a feature, so it is not leakage.
> (6) An amendment recorded after a session's recording deadline does not excuse that session's refusals or gaps.
> (7) The amendments file is outside the registered code identity, so an amendment is not a new registration and does not fail C4. The code that validates it is registered, and every ledger entry records the amendments head hash it used."

Research code may not import `nyse_rules` (it serves dates in the sealed holdout); `nyse.py` remains the research calendar.

## 2. The prospective window

- `H_END = 2026-10-14`, `LOOKBACK_SESSIONS = 63`.
- `WINDOW_START` is the session 63 sessions before H_END on the rules calendar: **2026-07-16**. [WINDOW_START, H_END] holds 64 sessions: H_END and the 63 before it. 64 closes give 63 daily returns, so a forecast made at the H_END close uses exactly the lookback.
- Everything from WINDOW_START on may be requested once complete. Earlier dates are refused: [2025-01-01, 2026-07-15] is the rest of the sealed period (`sealed_holdout`), and anything before 2025-01-01 belongs to the research store (`before_window`).
- **The request rule.** Under the OD9 descriptive convention a session's close is assumed available at 18:00 America/New_York. A request may end no later than the last session whose 18:00 has passed (`incomplete_session`). This only gates requests: no availability timestamp is stored, and it is not proof of vendor publication time or evidence of forecasting validity. Early-close days (13:00 ET) still wait for 18:00.

Example. At 17:59 New York time on 2026-10-14 the last complete session is 2026-10-13, so a default request is [2026-07-16, 2026-10-13] (63 sessions). From 18:00 it is [2026-07-16, 2026-10-14] (64 sessions). On Saturday 2026-10-17 it is [2026-07-16, 2026-10-16].

**An interpretation for the owner to confirm before the first `--store`.** The decision says "the last 63 sessions before H_end". Three readings differ by one session at either end:

| Reading | Sessions | Count |
|---|---|---|
| As implemented (the lead's brief): the session 63 sessions before H_end through H_end | 2026-07-16 to 2026-10-14 | 64 closes, 63 returns ending at the H_end close |
| 63 sessions ending at H_end | 2026-07-17 to 2026-10-14 | 63 closes (2026-07-16 would be an unauthorized sealed session) |
| 63 sessions strictly before H_end | 2026-07-16 to 2026-10-13 | 63 closes (the 2026-10-14 close would be sealed; the first forecast close would be 2026-10-15) |

The implementation is the union of the other two, so the owner's confirmation of 2026-07-16 and of 2026-10-14 is needed. Changing the reading is a one-line change to `WINDOW_START` (or to the earliest forecast close) and its tests.

## 3. Private distribution anchors

**Why.** Vanguard's and BlackRock's (iShares) website terms, read 2026-10-14, allow personal, non-commercial use and forbid redistribution and automated access. This repository is public. So no issuer amount lives in code or in a committed file, and nothing here fetches an issuer page.

**Where.** Default `data/raw/issuer_sources/anchors.csv` under the project root, ignored by `.gitignore`'s `data/raw/*` (verified with `git check-ignore`; a test repeats the check). `STOCK_AGENT_ANCHORS` overrides it, and `--anchors` overrides both.
- A path inside any repository (this one, another clone or a worktree) is loaded only if `git check-ignore` confirms it is ignored; Git missing counts as not ignored.
- Any path inside `issuer_sources/2026-10-14/`, where the research agents' gathered rows live, is refused before it is opened (also through a symlink, and in any letter case, since macOS's default file system ignores case). Those rows are not anchors.
- The file is opened once without following a final symlink, must be a regular file, and is read up to its 1 MiB limit only.
- A missing file is an empty allowlist: no fund passes the gate, so nothing is stored.

**Columns** (exactly these, in this order; the template is header-only):

| Column | Content | Example (invented) |
|---|---|---|
| `fund` | Upper-case ticker | `AGG` |
| `ex_date` | `YYYY-MM-DD`, an NYSE session | `2024-06-03` |
| `amount` | The issuer's string exactly as shown; an optional leading `$` | `0.1234` or `$0.1234` |
| `decimals` | Digits after the decimal point in `amount` | `4` |
| `basis` | `per_share_as_paid` (the only accepted value) | `per_share_as_paid` |
| `status` | `anchor` rows are used; another lower-case word (for example `draft`) is ignored and counted | `anchor` |
| `source_publisher` | Who published it | `BlackRock iShares` |
| `source_url` | The `https://` page you copied it from | `https://...` |
| `retrieved_on` | The day you copied it, `YYYY-MM-DD` | `2026-10-15` |
| `notes` | Optional free text | |

**Validation (any failure refuses the whole file, naming the row and column, never a value):** a wrong header or field count; a repeated (fund, ex_date) in any status; a malformed date, an ex-date that is not an NYSE session or is outside 2003-2030; a retrieval date after tomorrow (UTC); a non-positive or malformed amount (no sign, exponent, thousands separator or leading `.`); a `decimals` count that does not match the amount string (it catches a spreadsheet dropping trailing zeros); another basis; a malformed status, publisher or URL; control characters; invalid UTF-8 (a leading byte-order mark is fine).

**The anchors lockbox.**
- Research scope (the pre-2025 audit and the research store): ex-dates before 2025-01-01.
- Prospective scope (the prospective store): ex-dates on or after 2026-07-16.
- An anchor dated from 2025-01-01 to 2026-07-15 refuses the whole file in either scope (`sealed_period`), whatever its status: sealed-period issuer values may not be recorded at all.

**What is printed.** The file's sha256, row counts by status, per-fund anchor counts in scope, and the count outside scope. Never an amount. Each stored batch's vintage note records the sha256.

**The gate is unchanged.** Per fund, at least 2 anchors in the collected window, each matching Tiingo's `divCash` within 1e-6, with no mismatch; for every verified split in the window, a matching anchor on each side; the split-ratio check can only fail a fund. Every fund in a batch must pass, or nothing is stored.

## 4. The prospective store

Layout: `<root>/PROSPECTIVE_STORE.json` (the marker: store kind, schema, H_end, lookback, window start and calendar) and one directory per batch, `P<retrieved_at>-<12 hex>/`, holding `prospective_batch.json`, `raw_payload.bin`, F1's four Parquet tables and `manifest.json` (hashes, written last). Each batch is written once through a staging directory and an atomic rename, mode 0o700.

Contract checks (each refuses the whole batch): the request inside the window and ending no later than the last complete session at `retrieved_at`; the instrument mapping; F1's column types and values; only unadjusted closes and amounts per share as paid, with no declaration, record or pay date (Tiingo provides none); every row inside the request and on a rules-calendar session; the missing-session report.

Isolation from the research store:
- F1's `read_store` and `read_batch` refuse a prospective root (the marker is not a batch) and a prospective batch (the `P` prefix);
- a prospective batch is not an F1 `RawBatch`, and its dates fall after F1's cap, so F1's `ingest_batch` refuses it;
- the payload envelope is version `p1` and names the store, H_end and the window start, so neither Tiingo parser accepts the other's payload;
- the audit refuses a `--root` that is, or lies inside, a prospective store; the collection refuses a root that is, or lies inside, a research store (a directory holding an F1 batch), a non-empty root without the marker (for example one that contains the research store), and a root nested inside a prospective store;
- tests forbid research packages and scripts from importing the prospective package, `nyse_rules` or the Tiingo prospective mode (by import statement or by name), check that importing research entry points loads none of them, and run every research reader (F1, the model-health fund-store check, the T2 run reader, the forecast-artifact reader) against a prospective root to confirm each refuses it.

Roots: the collection's `--root` (and the audit's) must lie outside every repository, or strictly inside a repository's `data/raw/` where `git check-ignore` confirms it is ignored (for example `data/raw/prospective/`); a repository that tracks its `data/raw/`, or Git missing, refuses it. The collection checks the root before any request but creates it, with any missing parents at mode 0o700, only when a verified batch is about to be written; a refused run leaves no empty store, and a root it created is removed again if the write itself is refused. An existing directory is never changed, and an existing root must grant no group or other access.

## 5. The inference loader (for the forecast job)

```python
from stock_agent.data.prospective.inference import load_inference_window

window = load_inference_window(
    store_root, ("VTI", "AGG", "SHV"), "2026-10-14", lookback=63, retrieved_by=decision_time
)
```

- `forecast_close` must be on or after H_END (2026-10-14), whatever the lookback: the lookback sessions are inputs only, so no forecast is made, and later scored, inside them (`sealed_holdout`). The window's first session must also be on or after 2026-07-16.
- `retrieved_by` is required: the forecast's recorded decision time. Only batches retrieved at or before it are read, so a late run or a re-run sees exactly what the forecast could have seen.
- `window.sessions`: the `lookback + 1` sessions ending at `forecast_close`. Every returned date lies in [sessions[0], forecast_close].
- Tables: `prices` (fund_id, session_date, close, adjustment_basis, batch_id), `distributions` (fund_id, ex_date, amount, amount_basis, distribution_type, batch_id), `splits` (fund_id, effective_date, ratio_new_per_old, batch_id), `missing` (fund_id, session_date, reason `no_batch` or `missing_session`), `batches` (the lineage of the batches used) and `revisions` (fund_id, session_date, table, batch_id, earlier_batch_id, change `changed`, `removed` or `added`; no values). `window.complete` is True only when no close is missing; check `revisions` too, because a distribution a later batch no longer serves shows there, not in `missing`. Gaps are never filled.
- Vintage rule: for each (asset, session) the latest-retrieved eligible batch whose request covers it; what that batch lacks is missing, with no fallback to an older batch.
- It reads no clock and no network and computes nothing. It is the only reader of stored prospective rows. The forecast job's module must be added to `INFERENCE_CONSUMERS` in `tests/test_prospective_boundaries.py`, and may import only `inference` and `window`.

The full contract is the module docstring of `data/prospective/inference.py`.

## 6. Owner commands

Run these in your own terminal, from the repository root. Nothing below has been run by an agent.

### 6.0 The key (each terminal session)

```bash
read -rs TIINGO_API_KEY && export TIINGO_API_KEY
```

Typed, not echoed or kept in the shell history. The key is sent only in the `Authorization` header and never printed or stored.

### 6.1 The read-only pre-2025 audit (first)

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized --funds VTI AGG SHV --start 2005-01-01 --end-exclusive 2025-01-01
```

Three requests (one per fund), nothing written. **A good result** (exit code 0):
- `field allowlist: every element had exactly date, close, divCash, splitFactor`: Tiingo honours `columns`.
- The anchors lines: the file's path and sha256, or "is missing; empty allowlist" before you fill it.
- Per fund: the price-row count, the first and last session, and missing NYSE sessions. SHV started trading in January 2007, so its sessions "before first" are expected; "inside" should be 0 or a short, explainable list.
- `distributions per year`: about 4 a year for VTI and about 12 for AGG and SHV.
- `splits: 2008-06-18 x2 (verified; raw close x factor / previous raw close 1.00)` for VTI (the ratio varies with the day's move): a ratio near 1 confirms unadjusted closes.
- `distribution basis: NOT verified ...` until research-scope anchors exist; with them, `spot-checked against primary-source amounts ...` per fund.
- It ends with `NOT STORED read-only coverage report: ...`.

**Refusals before any request (exit code 2, nothing written):**

| Message starts with | Meaning |
|---|---|
| `Refused: this audit contacts Tiingo` | `--authorized` is missing |
| `Refused: --store and --root go together` | One without the other |
| `Refused: invalid_date` | A malformed `--start` or `--end-exclusive` |
| `Refused: anchors not_git_ignored` | The anchors file is inside a repository and Git does not confirm it is ignored |
| `Refused: anchors research_rows` | The path is inside `issuer_sources/2026-10-14/` |
| `Refused: anchors sealed_period` / `invalid_...` / `duplicate_anchor` / `decimals_mismatch` | A row breaks the format (the message names the row and column) |
| `Refused: --root ... inside the repository` / `contains the repository` / `is (or lies inside) a prospective store` / `has mode ...` | Only with `--store`: the root is unsafe |

**Refused retrievals (exit code 1, nothing written):** `REFUSED missing_token` (the key is not set); `REFUSED invalid_token`; `REFUSED rate_limited; ticker X; HTTP 429`, `server_error`, `http_error` or `transport_error` (no retry; run again later); `REFUSED request_after_cap` (an end after 2025-01-01); `REFUSED outside_coverage` (a start before 2003-01-02); `REFUSED non_allowlisted_field` (Tiingo returned a column outside the allowlist: report it, the lockbox is never relaxed to let it through); `REFUSED lockbox_violation` (a row dated 2025 or later); `REFUSED unverified_split_factor`, `split_factor_mismatch` or `split_price_mismatch` (a split that is not the verified VTI 2008 split, or a vendor disagreement about it).

Storing the research batch is a separate approval: add `--store --root ~/stock-agent-private/fund_store` only then.

### 6.2 Fill the anchors file by hand

1. Create the private file from the template (it stays ignored):
   ```bash
   mkdir -p data/raw/issuer_sources
   ```
   ```bash
   cp docs/templates/distribution_anchors_template.csv data/raw/issuer_sources/anchors.csv
   ```
   ```bash
   git check-ignore -v data/raw/issuer_sources/anchors.csv
   ```
   The last command must print the `data/raw/*` rule. To keep the file outside the repository instead, copy the template to, for example, `~/stock-agent-private/anchors.csv` and set `export STOCK_AGENT_ANCHORS=~/stock-agent-private/anchors.csv`.
2. Choose the rows.
   - Research scope: at least 2 ex-dates per fund between 2005-01-01 and 2024-12-31, and for VTI at least one before 2008-06-18 and one after.
   - Prospective scope: run 6.3 first; it prints each fund's served ex-dates. Add at least 2 per fund from that list.
   - Never add an ex-date from 2025-01-01 to 2026-07-15 (refused), and never copy from the files under `data/raw/issuer_sources/2026-10-14/`.
3. For each row, open the issuer's distribution history in your browser, find the ex-date and copy the amount per share exactly as shown, trailing zeros included. Count the digits after the decimal point for `decimals`. Use `per_share_as_paid`, `anchor` (or `draft` while unsure), the publisher, the page's `https://` URL and today's date.
   - **Per share as paid matters most before VTI's 2008 split.** A current issuer page may restate older amounts "adjusted to reflect a 2-for-1 share split". A restated amount is not per share as paid and must not be entered as such: matching it against a restated vendor series would wrongly confirm the basis. For pre-2008-06-18 VTI rows, use a source from that time (for example the fund's annual or semiannual report for that year, filed on SEC EDGAR) and note the source in `notes`.
4. Edit with a plain-text editor. A spreadsheet can reformat dates and drop trailing zeros; `decimals` catches the second, and dates must stay `YYYY-MM-DD`.
5. Check it offline (no network, nothing printed but counts and the sha256):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/check_anchors.py
   ```
   A good result prints both scopes with the per-fund anchor counts and ends with `valid: ...` (exit code 0). A refusal (exit code 2) names the row and column to fix.

### 6.3 The read-only prospective collection

After 18:00 New York time on or after 2026-10-14:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_prospective.py --authorized
```

Three requests for [2026-07-16, the last complete session], nothing written. **A good result** (exit code 0):
- `H_end 2026-10-14; lookback 63 sessions; window start 2026-07-16; calendar nyse_full_day_rules_2003_2030_v1` (the 2030 calendar since the owner's 2026-10-18 switch).
- `request: VTI AGG SHV [2026-07-16, 2026-10-14] (64 sessions)` when run that evening; later runs end at a later session.
- The request rule line, the allowlist line and the anchors lines (prospective scope).
- Per fund: `price rows (unadjusted): 64`, `missing NYSE sessions: 0 (none)`, `distributions served: N (ex-dates ...)` (expect about 3 for AGG and SHV and 1 for VTI) and the basis verification.
- It ends with `NOT STORED read-only collection report: ...`.

Expect VTI to show `NOT verified ... verified distributions in the window: 1 (needs 2)` even with its anchor filled: see "Limitations".

**Refusals before any request (exit code 2, nothing written):** the same `--authorized`, `--store`/`--root` and anchors refusals as 6.1, plus:

| Message starts with | Meaning |
|---|---|
| `Refused: sealed_holdout` | `--start` before 2026-07-16 (the sealed period) |
| `Refused: before_window` | `--start` before 2025-01-01 (the research store's period) |
| `Refused: incomplete_session` | `--end` after the last complete session (before 18:00 New York time, or a future date), or no session of the window is complete yet |
| `Refused: invalid_request` | Repeated, lower-case or malformed fund ids, or more than 20 |
| `Refused: outside_coverage` | After 2030-12-31: the calendar must be extended first (a new calendar id) |

**Refused retrievals (exit code 1, nothing written):** the token, HTTP, transport, allowlist and split refusals of 6.1, plus `REFUSED sealed_holdout` or `REFUSED before_window` (Tiingo served a row before the window) and `REFUSED outside_request` (a row after the requested end).

### 6.4 The store step (only after you approve storing)

Store only funds that pass. For the first collection that is AGG and SHV:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_prospective.py --authorized --funds AGG SHV --store --root ~/stock-agent-private/prospective_store
```

**A good result** (exit code 0): the 6.3 report, then `stored batch: P2026...`, `stored tables match a re-parse of the stored raw payload: True` and `stored under: ...`. The root (and any new parent, such as `~/stock-agent-private`) is created with mode 0o700 only at this point, and holds `PROSPECTIVE_STORE.json` and the batch. Use a different directory from the research store, and not one inside it.

**Refusals:** exit code 2 before any request, with nothing created, for `--root` problems (`inside the repository`, `does not confirm it is ignored`, `has mode ...` with a `chmod 700` hint, `is, or lies inside, the research store`, `is neither empty nor a prospective store ... never the research store`, `lies inside a prospective store`); exit code 1, with no root created, for `NOT STORED unverified_distribution_basis: <funds>` (a fund failed the gate: store the passing funds alone or add anchors) or a refused retrieval; exit code 1 for `FAILED the stored tables differ from a re-parse ...` (report it).

### 6.5 Later collections

Run 6.3 and 6.4 again after each session's 18:00. Each run is a new batch covering [2026-07-16, the last complete session]; nothing is overwritten. The loader uses the latest batch for each session.

## 7. Limitations and open owner decisions

- **VTI cannot be stored at first (owner decision needed).** The gate needs 2 matching anchors per fund inside the collected window. VTI pays quarterly, so [2026-07-16, 2026-10-14] holds one VTI ex-date (late September). VTI cannot pass until a collection window spans two VTI ex-dates (after its late-December 2026 ex-date), unless the owner changes the gate for the prospective store, for example to "every served distribution in the window anchored, at least one". The gate rules are unchanged here, as briefed; changing them is an owner lockbox decision.
- **A window with no distributions** for a fund cannot be verified, so it is not stored.
- **Unscheduled closures** after this code was written are unknown to the calendar (section 1).
- **Not a point-in-time vintage.** Values are as Tiingo served them at retrieval and may be revised later; a later batch is a new observation, never an overwrite. The 18:00 convention is assumed, not observed.
- **The spot check is a sample.** Two matching anchors do not validate every distribution or show that none is missing.
- **Early closes** wait for 18:00 like any session.
- **Not run.** Every live behaviour above (Tiingo's honouring of `columns`, its date format, its `divCash` basis) is still unverified until the owner's runs.
- **Holiday verification.** 2026-2028 equal NYSE's published page (retrieved 2026-10-18). 2025, 2029 and 2030 are rule-generated: expected sessions, not verified (`nyse_rules.year_source`). Check 2029 and 2030 with `check_published` once NYSE publishes them; a difference would need a new calendar id.

Follow-ups (not done here):
- The store checks a batch's tables, not its raw payload; the payload's dates are checked by the Tiingo prospective parser before storing and re-parsed by the collection script after storing. A store-level payload check would couple the store to one adapter.
- `git check-ignore` honours the owner's global ignore file; requiring the match to come from the repository's own `.gitignore` would be stricter.
- A parent directory swapped for a symlink between the root check and the write, or a hard link to a gathered file, needs a local writer as the same user and is not detected.
- The publication and registry readers are not exercised against a prospective root (they read other contracts and never import the prospective package).
- In a worktree whose `.venv` is a symlink, `.gitignore`'s `.venv/` does not match it; never stage it.

## 8. Learning notes

- **Problem.** Feed frozen forecasters real inputs without touching the sealed holdout, and make it impossible for research code to read those inputs.
- **Concepts.** A lockbox (sealed holdout) keeps one honest out-of-sample test; prospective evaluation (forecast now, score later) is the only evaluation that cannot be overfitted by hindsight. Anchors are verification data, not model inputs: they test what a vendor's numbers mean.
- **Assumptions.** NYSE holiday rules hold through 2030 apart from unscheduled closures (checked against the published schedule for 2026-2028 only); Tiingo's `close` is unadjusted and `divCash` per share as paid (checked, not assumed, where anchors exist); the 18:00 New York convention.
- **Formulation.** With closes P_0..P_63 for the 64 sessions ending at the forecast close, the 63 daily total returns are r_t = (P_t + D_t) / P_{t-1} - 1 (D_t the distribution with ex-date t), and a 63-session realized volatility is sqrt(mean(log(1 + r_t)^2)). These formulas belong to the forecasters (OD8 group A for returns); the loader only supplies P, D and splits.
- **Small example.** A forecast at 2026-10-14 uses [2026-07-16, 2026-10-14]; a forecast at 2026-10-15 uses [2026-07-17, 2026-10-15]: the window rolls by one session and never reaches back into the sealed period.
- **Leakage risks.** Collecting or anchoring sealed-period values (refused at the request, the element, the payload, the store and the anchors file); a forecast made inside the lookback with a short window and scored on later lookback closes (refused: forecast closes start at H_end); a re-run reading a later download (refused: `retrieved_by` is required); research code reading prospective rows (boundary tests); the extended calendar used in research (forbidden import); reading the 18:00 convention as evidence that a close was knowable then.
- **How it could mislead.** A stored batch passed a spot check, not a validation. A missing unscheduled closure looks like a data gap. VTI's absence at first changes which assets the first forecasts cover.
- **Practitioner validation.** Re-collect and compare batches (vintage differences show vendor revisions); cross-check closes against the issuer's published closing prices or NAVs; the read-back check after every store; and, for the forecasts themselves, score only on outcomes that arrive after each forecast was recorded.
- **Next to learn.** Vintage databases (ALFRED-style real-time data), why purged cross-validation differs from prospective evaluation, and forecast-comparison tests (for example Diebold-Mariano) for when outcomes accrue.
- **Exercise.** With `nyse_rules.session_offset`, find the first forecast close whose 63-session window contains two VTI ex-dates if VTI goes ex on the fourth-to-last session of each quarter, and check your answer with `nyse_rules.sessions`.
