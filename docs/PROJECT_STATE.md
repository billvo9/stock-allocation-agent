# Project State

Changing project state only. Permanent policy lives in `AGENTS.md`.
Git is authoritative: verify everything here before relying on it.

Last reviewed: 2026-10-08


## Last checkpoint

2026-10-06, milestone "ML validation foundation", verified on branch
`feature/ml-validation-foundation` based on `da21961` (merge pending). Live
Git state (current branch, working tree, upstream) comes from Git and the
SessionStart hook, not from this file.

Recently merged to `main` (`origin/main` = `da21961` as of last fetch,
verified 2026-10-06):

- PR #39 (`da21961`): `b2e443a` chore: adopt guarded high-autonomy Claude
  workflow; `763b224` docs checkpoint; `28aaff6` chore: pin ruff 0.16.2 and
  pytest 9.1.1. The development workflow (autonomy, approval gates,
  `scripts/verify.py`) is described in AGENTS.md.
- PR #38 (`44efae1`): `1cfc901` docs: checkpoint project state after
  versioned fundamentals merge.
- PR #37 (`668a37c`): `b014e28` feat: add versioned point-in-time
  fundamentals (10 source / test files plus this file).
- PR #36 (`2025c1f`): `60fc6e7` chore: add Claude specialist agent team
  (`.claude/agents/`, eight agents).
- PR #35 (`d866fd4`): `3258e72` chore: add Claude project workflow
  automation (`.claude/settings.json`, hooks, `project-verify` and
  `project-checkpoint` skills).
- PR #34 (`0b7b1b8`): `add55fd` feat: integrate point-in-time EDGAR
  fiscal reconciliation.


Three older stashes (from `feature/macro-ingestion`,
`feature/fundamental-data-foundation`, `feature/equal-weight-baseline`)
are owner-owned and untouched.


## Current objective

ML phase, ticket 1: leakage-safe model-validation foundation (this branch,
merge pending). No prediction model is selected or implemented. Design,
leakage register, evidence, and the next-ticket proposal are in
`docs/research/model_validation.md`.

Owner decisions (2026-10-06):

- ML labels enter at the close after the feature date:
  `MODEL_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1)` in
  `features/training.py`. `add_forward_return_target` keeps its default
  `entry_lag=0`, so existing callers and tests are unchanged.
- Data from 2025-01-01 onward is a lockbox. Development folds end by it and
  never score labels that mature in it; holdout folds run once per
  pre-registered model.

In effect on this branch (`src/stock_agent/model_validation/`): expanding
forward-only folds; strict global purge (train iff
`target_end_date < test_start`); no embargo, justified and asserted per fold;
fit-once, train-only transforms with a fresh estimator per fold; a feature
contract plus a content-based prefix-stability check; canonical fingerprints
and per-fold metadata.


## Current EDGAR architecture

Modules under `src/stock_agent/data/fundamentals/`:

| Module | Responsibility |
|---|---|
| `edgar_source.py` | List financial filings and metadata (form, period end, filing date, acceptance time, accession, XBRL flag) |
| `edgar_fiscal.py` | `infer_edgar_fiscal_identity`: authoritative fiscal year / period / quarter from XBRL reporting periods |
| `edgar_concepts.py` | `resolve_statement_concept`: map XBRL concepts to canonical metrics |
| `edgar_periods.py` | Parse statement period columns; get direct-quarter, YTD, FY, instant values |
| `edgar_quarterly.py` | `build_edgar_quarter`: one filing to one canonical row; availability rule (`resolve_edgar_available_at`); `extract_fiscal_flow_observations`: single-filing direct / YTD / FY observations |
| `edgar_reconciliation.py` | Current- and prior-observation selection and cross-filing reconciliation (`select_current_fiscal_observation`, `select_prior_fiscal_observation`, `reconcile_fiscal_flow`, `reconcile_fiscal_flow_from_candidates`, `find_superseding_prior_observations`); optional `as_of` knowledge time |
| `edgar_history.py` | `build_edgar_history`: collect filings, index raw observations once, emit period-centric versions; returns versioned canonical frame, reconciliation lineage, diagnostics |
| `quarterly_schema.py` | Quarterly fundamental schema and validation |
| `point_in_time.py` | `align_quarterly_fundamentals_asof`: current-state change log, then one as-of join to market dates |
| `features.py` | Quarterly ratios and YoY growth (availability-guarded prior-year match) |


## EDGAR reconciliation (merged in PR #34)

- Authoritative EDGAR fiscal identity (`ec006a4`)
  drives every observation; quarters never come from calendar months.
- Deterministic cross-filing fiscal-flow reconciliation; reconciliation
  is the single authority for flow values in the history and is
  independent of filing order.
- Prior-observation selection: filtered by availability before ranking;
  latest available wins; equal-value ties take the smallest accession;
  conflicting values are unavailable (`ambiguous_prior_observation`); a
  newer amendment without a usable value is skipped and recorded.
- Fiscal-Q1 direct-value fallback for Q2 derivation
  (`prior_value_source="q1_direct"`); conflicting Q1 direct/YTD values
  are rejected; never applied to Q2/Q3 priors.
- Corrected deferred SEC availability semantics (below), with DST-safe
  06:00 America/New_York construction.
- `not_yet_available` pre-load filtering in strict and non-strict modes.
- Real-shaped MU regression fixtures (offline) and controlled real MU
  validation (below).


## Versioned EDGAR history (merged in PR #37)

Design owner-approved 2026-10-05, including the contract changes below
and the AGENTS.md interpretation that per-metric carry-forward is not a
"substitution" (no AGENTS.md change). Described as verified 2026-10-05
against the pre-commit working tree; committed as `b014e28` and merged
as `668a37c`:

- Versions are period-centric. For each fiscal period, each DISTINCT
  `available_at` of its own filings or the immediately preceding fiscal
  quarter's filings is evaluated once (same-instant events coalesced),
  using only raw filings available at or before that time.
- Per metric, the source is the latest own-period raw filing with a
  usable value; omitted metrics carry forward; same-instant conflicting
  values are unavailable (`ambiguous_current_observation`).
- A version is emitted when an own-period filing becomes available
  (always visible) or when a metric's (value, method, reason) changes.
  Fixture example: Q3 revenue 9000 at Q3, 9200 at the late Q2/A, 9700 at
  Q3/A. Historical rows never change.
- Raw SEC observations are the only reconciliation inputs; the candidate
  index is read-only and versions are outputs only.
- Version row: anchor (newest own-period filing) supplies
  `sec_accession_number`, form, filing date; `available_at` = event time;
  `availability_source` = anchor's rule if the anchor became available
  then, else the smallest-accession trigger's rule.
- Contract changes (owner-approved): canonical duplicate key adds
  `available_at`; EDGAR history unique on
  `(symbol, period_end, available_at)`; lineage key
  `(symbol, sec_accession_number, available_at, metric_name)` with new
  columns `source_accession_number`, `source_available_at`,
  `version_trigger_accessions`.
- Filings whose `period_end` conflicts with the period's earliest
  filing are rejected (`conflicting_period_end` diagnostic) and excluded
  from all inputs; strict mode raises.
- `derived_quarter_input_superseded` diagnostic retained.
- As-of: latest `period_end` known at the market date, then that
  period's latest version; a late amendment of an older period never
  becomes current. Duplicate key `(symbol, period_end, available_at)`.
- YoY: prior-year rows must be available no later than the current row;
  unknown availability is never eligible (NaN growth). Matching and
  growth formulas unchanged.
- Reconciliation formulas unchanged (checked 2026-10-06 with
  `git diff 2025c1f b014e28`): no arithmetic lines removed; the
  prior-availability guard now compares against an `as_of` knowledge
  time that defaults to, and may never precede, `current.available_at`.
- Verification: two independent reviews (test-reviewer, data-engineer)
  found no blockers; 12 of 12 mutation checks caught (backdating,
  filing-centric re-derivation, availability-only as-of, latest-overall
  prior, no-op versions, erasing omitted metrics, wrong lineage source,
  no coalescing, YoY guard removed, future prior in lineage, rejected
  filing as input, suppressed re-versions).


## EDGAR availability semantics

```text
accepted_at   SEC acceptance time
filing_date   SEC-assigned filing date
available_at  earliest timestamp this pipeline permits the information
              to be consumed
```

- `accepted_at` missing: `filing_date + 1 day` (`sec_filing_date_plus_1d`).
- Deferred filing (SEC `filing_date` later than the America/New_York date
  of `accepted_at`): `filing_date` at 06:00 America/New_York, converted
  to UTC (`sec_deferred_filing_date_6am`).
- Otherwise: `accepted_at` (`sec_acceptance_datetime`).
- `available_at` is never earlier than `accepted_at`.

06:00 ET for deferred filings is a conservative modeling proxy based on
the SEC filing rule (submissions after 5:30 p.m. ET receive the next
business day's 6:00 a.m. ET filing date and are not disseminated until
that day). It is not a measured dissemination timestamp.


## Controlled real-data validation (MU)

Seven MU XBRL filings, FY2025 Q1 through FY2026 Q3, edgartools 5.58.0,
validation only (run before PR #37; not re-run on the versioned
history):

- Fiscal identity matched MU's 52/53-week calendar.
- Adapter `accepted_at` is true UTC (API `23:52:13Z` equals SEC header
  `ACCEPTANCE-DATETIME 18:52:13` EST).
- FY2025 operating cash flow 3,244 / 3,942 / 4,609 / 5,730 USD M sums to
  the reported FY 17,525.
- Six of seven filings were accepted after 5:30 p.m. ET; their
  `available_at` moved to 06:00 ET on the SEC filing date (about 11 hours
  later). Values, methods, and selected priors were unchanged.
- Real-data prefix stability held at every knowledge time; no artifacts
  entered the repository.


## Real XBRL amendment validation (SMCI)

Owner-reported 2026-10-06: run after the PR #37 merge on Super Micro
Computer (SMCI) FY2017 filings, including amendments. Validation only;
not re-run in this session. No artifacts entered the repository
(verified 2026-10-06: clean tree, no SMCI references in tracked files).

No point-in-time correctness defect was found:

- Original versions stayed historically visible; amended versions
  became visible only at their own `available_at`.
- Later filings did not alter earlier histories.
- Downstream quarter re-derivation after amendments was correct.
- Lineage source / prior timestamps and accessions were correct.
- As-of selection chose the newest fiscal period first, then that
  period's newest version.
- Manual operating cash-flow reconciliation matched the reported FY
  values.

Coverage limitations found are tracked as open item 2.


## Last verified quality baseline

Measured 2026-10-06 with `python3 scripts/verify.py` on
`feature/ml-validation-foundation` (base `da21961` plus this milestone's
changes); gates ran with `.venv` tools (Python 3.12.2, pytest 9.1.1,
ruff 0.16.2). Exit 0.

- `pytest -q`: 865 passed (696 before + 169 validation and label tests)
- `python -m pytest -q`: 865 passed
- `ruff format --check src tests scripts .claude/hooks`: 132 files
  already formatted
- `ruff check src tests scripts .claude/hooks`: all checks passed
- Mutation checks: 34 of 34 deliberate leakage violations killed by the
  suite. The list is in `docs/research/model_validation.md`.
- Real-data evidence (read-only, not committed):
  - production feature SQL prefix-stable at 5 real cutoffs;
  - 24 development folds, 84 rows purged per fold;
  - same-symbol memorizer canary: corr -0.014 purged vs +0.190 unpurged;
  - the harness refuses an unpurged fold.


## Open items

1. Decision timestamp vs midnight-UTC market dates in
   `align_quarterly_fundamentals_asof` (stale-by-a-day, not leaking;
   availability-semantics change, owner-gated).
2. Real-data coverage gaps from the SMCI validation (coverage and
   diagnostics; no point-in-time defect found), in suggested priority:
   - edgartools can mislabel 52/53-week fiscal calendars; confirm
     whether affected filings are rejected or accepted.
   - Later comparative restatements are not ingested.
   - One ambiguous concept mapping rejects the entire filing.
   - Part-III-only 10-K/As are skipped under the same reason code as
     genuine errors.
   - Current-period columns can sometimes be missing.
   - Synthetic amendment fixtures are cleaner and more staggered than
     some real filings.
3. Version rows copy non-flow columns from the anchor; extend
   per-metric carry-forward before EDGAR populates balance-sheet / EPS /
   capex columns.
4. Two fiscal periods sharing `(period_end, available_at)` raise even in
   non-strict mode (rejection rule is an owner decision).
5. YoY: no refresh when a prior-year amendment arrives later; nearest-
   match fallback when the nearest prior-year row is not yet available.
6. ML validation follow-ups:
   - The universe is 5 semiconductor names chosen with hindsight
     (selection and survivorship bias). Validation cannot fix this. A
     point-in-time universe (historical index membership or a rules-based
     liquidity universe) is the prerequisite for generalizable claims.
   - `run_rebalanced_backtest` executes at the same close; add a matching
     execution lag before portfolio-level model metrics (model ticket).
   - `build_model_dataset` still includes benchmark index rows
     (DOW_JONES, NASDAQ_COMPOSITE, SP500); validation excludes them via
     `restrict_to_symbols`. Removing them upstream is a data-contract
     change (owner-gated).
   - Legacy `split_temporal_dataset` and `select_training_rows_asof` are
     kept with warning docstrings: no purge, and intraday cutoffs could
     admit unmatured labels. Nothing outside tests calls them. Two gaps:
     - there is no runtime guard (a `DeprecationWarning` would be
       non-breaking);
     - the split's defaults (validation 2025, test 2026) lie entirely
       inside the lockbox.

     Deprecating them changes contracts and tests.
   - SNDK has no rows dated before the lockbox. In holdout folds its first
     trainable label matures in mid-April 2025, so early holdout folds
     score it with no SNDK training history.
   - Audit gaps found in the pre-merge review (2026-10-08). The manifest
     lacks:
     - a dirty-tree flag;
     - the estimator spec, hyperparameters, and seeds;
     - fitted transform parameters;
     - a variant and holdout-use ledger;
     - versions of duckdb, pyarrow, yfinance, and edgartools, plus platform;
     - raw-data retrieval times;
     - hashes of the fold table and the predictions;
     - a writer that saves it.

     Planned in T2 (`docs/research/diagnostics_and_visualization_plan.md`).
   - Labels are computed after `build_model_dataset` drops rows with
     missing required features. An interior gap would stretch a label's
     span. Not present today; purging stays correct because it uses the
     recorded `target_end_date`.
   - Macro `available_at` is date-level only (no time of day stored), so a
     row dated d includes releases published after that day's close. Times
     verified 2026-10-08 against Federal Reserve sources:
     - H.15 (fed funds) at 4:15 p.m. ET;
     - weekly H.6 (M2) on Thursdays at 4:30 p.m. until 2021-02-11.

     This is harmless for ML labels (entry at the close of d+1). It is still
     unsafe for `entry_lag=0` labels and the same-close backtest if they are
     combined with macro. Macro values are vintage-correct: ALFRED's full
     real-time range, with each revision kept as its own row.
7. Workflow follow-ups:
   - GitHub branch protection on `main` is not verified (`gh` is not
     installed locally). It is the authoritative control against direct
     pushes and merges; the local guard is defense in depth.
   - Confirm in a fresh session that the `Edit(...)` ask rules prompt for
     `.claude/**`, `AGENTS.md`, and dependency manifests. They did not
     prompt in the session that added them.
   - `numpy` and `pandas` are unpinned; tests use the legacy `RandomState`
     stream and canonical fingerprints so results do not depend on them.
   - `.gitignore` line 44 reads `*.logreports/`: a missing newline, so
     `*.log` files are not ignored (`reports/` is ignored by the next line).
8. Smaller items: lineage lacks current-side tie lists; as-of change log
   trusts `period_end`; mixed availability rules at one trigger instant;
   FY consistency checks (Q1+Q2+Q3+Q4 = FY); duplicate flow logic in
   `build_edgar_quarter`; unused `edgar_history._infer_fiscal_quarter`;
   lineage lacks numeric derivation inputs.


## Next steps

1. Owner reviews and merges the `feature/ml-validation-foundation` pull
   request (merging is owner-only); confirm CI passes.
2. ML roadmap (owner request 2026-10-08: diagnostics and visualization are
   part of the roadmap, not an afterthought). Tickets T2-T6 are in
   `docs/research/diagnostics_and_visualization_plan.md`:
   - T2: diagnostics layer, run outputs and ledger, null models;
   - T3: Plotly and Streamlit dashboard, run on the null models;
   - T4: Ridge, Lasso, and Elastic Net, with coefficient stability;
   - T5: portfolio readiness;
   - T6: holdout evaluation, once per pre-registered model.

   Owner decisions:
   - plotly and streamlit dependencies (before T3);
   - raw vs excess-over-equal-weight label (excess recommended);
   - scikit-learn and the learning-core split for the estimator core
     (before T4).
3. Owner decision from open item 7: enable branch protection on `main`.
4. Strategic: a point-in-time universe (open item 6) before any claim
   beyond "relative skill within this hindsight-selected universe".
