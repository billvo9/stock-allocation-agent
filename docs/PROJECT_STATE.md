# Project State

Changing project state only. Permanent policy lives in `AGENTS.md`.
Git is authoritative: verify everything here before relying on it.

Last reviewed: 2026-10-04


## Current branch

`feature/edgar-cross-filing-reconciliation`


## Last verified pushed commit

`ee003eb` feat: use authoritative fiscal identity in EDGAR history

Verified 2026-10-04: local HEAD and
`origin/feature/edgar-cross-filing-reconciliation` both point to
`ee003eb979c3a78fd617c912d6c45aba2c2aafa0`.

Branch commits since `main` merge of PR #33 (`b7f1a6e`):

- `ec006a4` feat: add authoritative EDGAR fiscal identity resolution
- `06bcfe8` feat: add deterministic fiscal flow reconciliation
- `ee003eb` feat: use authoritative fiscal identity in EDGAR history


## Current working tree

Uncommitted, intentional, unfinished owner work in
`src/stock_agent/data/fundamentals/edgar_history.py`:

Import-only change preparing reconciliation integration:

- `resolve_statement_concept` (edgar_concepts)
- `get_direct_quarter_value`, `get_ytd_value`, `get_fy_value`
  (edgar_periods; `find_period_columns` import moved into this block)
- `FiscalFlowObservation`, `FiscalFlowReconciliation`,
  `reconcile_fiscal_flow` (edgar_reconciliation)

None are used yet. Do not modify this file unless the owner asks.

Patch backup: `~/Desktop/stock-agent-edgar-wip.patch`
(verified 2026-10-04 to exist and to be identical to `git diff`).


## Current EDGAR architecture

Modules under `src/stock_agent/data/fundamentals/`:

| Module | Responsibility |
|---|---|
| `edgar_source.py` | List financial filings and metadata (form, period end, filing date, acceptance time, accession, XBRL flag) |
| `edgar_fiscal.py` | `infer_edgar_fiscal_identity`: authoritative fiscal year / period / quarter from XBRL reporting periods |
| `edgar_concepts.py` | `resolve_statement_concept`: map XBRL concepts to canonical metrics |
| `edgar_periods.py` | Parse statement period columns; get direct-quarter, YTD, FY, instant values |
| `edgar_quarterly.py` | `build_edgar_quarter`: one filing to one quarterly row; resolves `available_at` |
| `edgar_reconciliation.py` | `reconcile_fiscal_flow`: deterministic cross-filing standalone-quarter derivation with lineage |
| `edgar_history.py` | `build_edgar_history`: iterate filings, build quarters, emit diagnostics |
| `quarterly_schema.py` | Quarterly fundamental schema and validation |
| `point_in_time.py` | `align_quarterly_fundamentals_asof`: as-of join to market dates |

Current behavior:

- `build_edgar_history` uses `infer_edgar_fiscal_identity` (since `ee003eb`).
- `available_at` = SEC acceptance datetime, else filing date + 1 day.
- Per-filing extraction (`edgar_quarterly._extract_quarterly_flow`) returns
  direct standalone-quarter values, or Q1 from Q1 YTD. Q2/Q3/Q4 flows
  without a direct value are currently missing. Cross-filing derivation
  is deferred to `edgar_history`.
- `reconcile_fiscal_flow` exists and is unit-tested but is not yet called
  by `build_edgar_history`.
- Original and amended filings are both preserved as separate rows.

Known cleanup candidate (not yet approved):
`edgar_history._infer_fiscal_quarter` (label-based) has no callers since
`ee003eb`.


## Current reconciliation objective

Integrate `reconcile_fiscal_flow` into `build_edgar_history` so Q2, Q3,
and Q4 flow metrics without a direct value are derived across filings,
point-in-time safely, with lineage (method, reason, accession numbers).


## Reconciliation invariants

Enforced in `edgar_reconciliation.py` (unit-tested):

- Direct standalone-quarter value wins when present.
- Q1 = Q1 YTD; Q2 = Q2 YTD - Q1 YTD; Q3 = Q3 YTD - Q2 YTD;
  Q4 = FY - Q3 YTD.
- Prior must match symbol, metric, and fiscal year.
- Prior must be exactly the preceding fiscal quarter.
- Prior must come from a different accession.
- Prior period end must be strictly before current period end.
- Prior `available_at` must be <= current `available_at`.
- Missing inputs return `method="unavailable"` with a reason code
  (e.g. `missing_prior_fiscal_observation`, `missing_prior_ytd`,
  `missing_current_ytd`, `missing_current_fy`,
  `prior_available_after_current`).


## Required integration scenarios

Integration tests for `build_edgar_history` must cover:

1. Q3 derivation: Q3 10-Q with only YTD, prior Q2 10-Q YTD available
   earlier, gives Q3 = Q3 YTD - Q2 YTD with both accessions recorded.
2. Q4 derivation: 10-K with FY, prior Q3 10-Q YTD, gives
   Q4 = FY - Q3 YTD.
3. Amendment leakage: a Q2 10-Q/A accepted after the Q3 10-Q must not
   be used to derive the Q3 value available at the Q3 filing time.
4. Fiscal-year boundary: Q1 of a new fiscal year never uses the prior
   year's observations.
5. Non-calendar fiscal year: quarters are assigned from fiscal metadata,
   not calendar months.
6. Missing prior: Q2/Q3/Q4 without an eligible prior yields an
   unavailable result with a reason, not a guess.


## Open design questions (owner decision)

- When both an original and an amended prior are available before the
  current filing, which is used? (Candidate: latest available at or
  before current `available_at`.)
- Should a later amendment produce a new, later-available version of an
  already-derived quarter?
- Where is reconciliation lineage stored: schema columns, diagnostics,
  or a separate frame? (A schema change requires approval.)
- Where are YTD / FY values extracted: in `build_edgar_quarter` or in
  `edgar_history`?


## Last known quality baseline

Measured 2026-10-04 on the working tree (HEAD `ee003eb` plus the
uncommitted import change):

- pytest: 346 passed
- `ruff check .`: 8 errors, all in `edgar_history.py` from the
  unfinished imports (7 x F401 unused import, 1 x I001 import order).
  HEAD's version of that file passes.
- `ruff format --check .`: 119 files already formatted


## Next implementation stage

1. Resolve the open design questions above.
2. Write failing integration tests for the required scenarios.
3. Implement reconciliation in `build_edgar_history` using the
   prepared imports.
4. Restore a clean Ruff baseline.
5. Decide on removing `_infer_fiscal_quarter`.


## Future work

- Merge this branch to `main` via pull request (owner approval).
- Feed reconciled fundamentals into `point_in_time` alignment and
  fundamental features.
- Deterministic and statistical data-quality checks on EDGAR history.
- Data-quality dashboard built from diagnostics and reconciliation
  reasons.
