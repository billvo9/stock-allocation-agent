# Project State

Changing project state only. Permanent policy lives in `AGENTS.md`.
Git is authoritative: verify everything here before relying on it.

Last reviewed: 2026-10-05


## Current branch

`chore/claude-agent-team` (local only; no upstream configured).


## Last verified commit

`d866fd4` Merge pull request #35 from billvo9/chore/claude-project-workflow

Verified 2026-10-05: local HEAD, `origin/main` (as of last fetch), and
`git merge-base HEAD origin/main` all equal
`d866fd41d7dfb054537a387d9af63eb7f2a08f2f`. The current branch has no
commits of its own.

Recently merged to `main`:

- PR #34 (`0b7b1b8`): `add55fd` feat: integrate point-in-time EDGAR
  fiscal reconciliation (on top of `ec006a4`, `06bcfe8`, `ee003eb`,
  `17d3d4a`).
- PR #35 (`d866fd4`): `3258e72` chore: add Claude project workflow
  automation (`.claude/settings.json`, hooks, `project-verify` and
  `project-checkpoint` skills).


## Current working tree

Verified 2026-10-05:

Uncommitted and unstaged (this ticket):

- `.claude/agents/*.md` (8 new, untracked): the specialist agent team.
- `.claude/skills/project-verify/SKILL.md` (modified): gate commands must
  run unpiped; pass/fail comes from exit status.
- `docs/PROJECT_STATE.md` (modified): this checkpoint.

Nothing staged.


## Current objective

Ticket: project-scoped specialist agent team (`chore/claude-agent-team`).
Owner-approved design, 2026-10-05:

- Eight agents under `.claude/agents/`: quant-researcher, data-scientist,
  data-engineer, backend-engineer, frontend-engineer (implementers), and
  security-reviewer, test-reviewer, architect (read-only: no Edit/Write;
  Bash for inspection and verification only).
- `model: inherit` for all; no WebSearch/WebFetch.
- Learning-core ML/RL/statistical-model work: agents deliver design,
  tests, and a bounded skeleton; the owner writes the core.

Status: the eight agent definitions and the project-verify edit are
being added in the working tree, pending owner review. Not committed.


## Current EDGAR architecture

Modules under `src/stock_agent/data/fundamentals/`:

| Module | Responsibility |
|---|---|
| `edgar_source.py` | List financial filings and metadata (form, period end, filing date, acceptance time, accession, XBRL flag) |
| `edgar_fiscal.py` | `infer_edgar_fiscal_identity`: authoritative fiscal year / period / quarter from XBRL reporting periods |
| `edgar_concepts.py` | `resolve_statement_concept`: map XBRL concepts to canonical metrics |
| `edgar_periods.py` | Parse statement period columns; get direct-quarter, YTD, FY, instant values |
| `edgar_quarterly.py` | `build_edgar_quarter`: one filing to one canonical row; availability rule (`resolve_edgar_available_at`); `extract_fiscal_flow_observations`: single-filing direct / YTD / FY observations |
| `edgar_reconciliation.py` | Prior selection and cross-filing reconciliation (`select_prior_fiscal_observation`, `reconcile_fiscal_flow`, `reconcile_fiscal_flow_from_candidates`, `find_superseding_prior_observations`) |
| `edgar_history.py` | `build_edgar_history`: collect filings, then reconcile flows across filings; returns canonical frame, reconciliation lineage, diagnostics |
| `quarterly_schema.py` | Quarterly fundamental schema and validation |
| `point_in_time.py` | `align_quarterly_fundamentals_asof`: as-of join to market dates |


## EDGAR reconciliation (merged in PR #34)

- Authoritative EDGAR fiscal identity (`ec006a4`)
  drives every observation; quarters never come from calendar months.
- Deterministic cross-filing fiscal-flow reconciliation integrated into
  `build_edgar_history`; reconciliation is the single authority for flow
  values in the history and is independent of filing order.
- Prior-observation selection: filtered by availability before ranking;
  latest available wins; equal-value ties take the smallest accession;
  conflicting values are unavailable (`ambiguous_prior_observation`); a
  newer amendment without a usable value is skipped and recorded.
- Fiscal-Q1 direct-value fallback for Q2 derivation
  (`prior_value_source="q1_direct"`); conflicting Q1 direct/YTD values
  are rejected; never applied to Q2/Q3 priors.
- Reconciliation lineage in `EdgarHistoryResult.reconciliation`
  (additive field), key `(symbol, sec_accession_number, metric_name)`.
- Late prior amendments never rewrite derived values; they emit
  `derived_quarter_input_superseded`, attached to the superseding filing.
- P1 prefix-stability protection for canonical frame, lineage, and
  diagnostics; mutation checks (leaky selector, backdated availability,
  misplaced diagnostic) each made the tests fail.
- Corrected deferred SEC availability semantics (below), with DST-safe
  06:00 America/New_York construction.
- `not_yet_available` pre-load filtering: filings whose `available_at` is
  after the run time are skipped before loading, in strict and
  non-strict modes.
- Real-shaped MU regression fixtures (offline, values from the smoke
  test).
- Controlled real MU validation (below).
- Canonical fundamental schema unchanged: `quarterly_schema.py` has no
  diff against HEAD.


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
validation only:

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


## Last verified quality baseline

Measured 2026-10-05 via `/project-verify` on HEAD `d866fd4` plus the
uncommitted ticket changes (each gate run unpiped, judged by exit status):

- `pytest -q`: 452 passed
- `python -m pytest -q`: 452 passed
- `ruff format --check src tests scripts`: 116 files already formatted
- `ruff check src tests scripts`: all checks passed


## Open items

1. Versioned derived observations: a late prior amendment does not yet
   produce a new version of an already-derived quarter (flagged, never
   rewritten), so the history is prefix-stable but not as-of complete
   for those cells.
2. `align_quarterly_fundamentals_asof` daily-date semantics: midnight-UTC
   market dates versus decision time, and same-instant rows.
3. No real XBRL amendment validated yet (MU's amendments predate XBRL).
4. FY accounting-consistency checks (Q1+Q2+Q3+Q4 = FY).
5. Remove the duplicate flow logic in `build_edgar_quarter`.
6. Remove unused `edgar_history._infer_fiscal_quarter`.
7. Lineage does not yet store the numeric derivation inputs.


## Next steps

1. Owner reviews the agent definitions and this checkpoint.
2. Commit the agent team and the project-verify edit on this branch
   (owner approval).
3. Push and open a pull request to `main` (owner approval).
4. Next major data ticket: "Versioned derived observations and
   amendment-aware as-of fundamentals" (open items 1 and 2).
