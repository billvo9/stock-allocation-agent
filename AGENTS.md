# Stock Allocation Agent

This file is the stable, tool-neutral project constitution.
It contains permanent policy. Changing state belongs in
`docs/PROJECT_STATE.md`.


## Mission

Build a production-quality, user-centered portfolio allocation and investment
decision system.

The long-term system should combine:

- market data
- macroeconomic data
- point-in-time company fundamentals
- portfolio and risk analytics
- statistical models
- machine learning
- eventually reinforcement learning / policy optimization
- explainable portfolio recommendations
- data-quality and model-monitoring dashboards
- agent-assisted software development

The system is not intended simply to find whichever strategy produced the
highest historical return.

The intended question is closer to:

> Given these assets, market conditions, expected returns, volatility,
> correlations, uncertainty, transaction costs, investor risk tolerance,
> and investment horizon, what allocation is reasonable and why?

User trust, auditability, reproducibility, point-in-time correctness,
explainability, and robust risk management are first-class requirements.


## Project owner and learning objective

The project owner is actively learning data science, statistics, quantitative
finance, machine learning, software architecture, and production engineering.

Do not treat the owner merely as someone approving generated code.

For substantial quantitative, statistical, ML, or data-science work:

1. Explain the underlying idea before or alongside implementation.
2. State assumptions explicitly.
3. Show formulas when useful.
4. Explain point-in-time and leakage risks.
5. Give a small numerical example when appropriate.
6. Explain how practitioners would validate the approach.
7. Mention meaningful alternatives and tradeoffs.
8. Distinguish evidence from assumptions or intuition.
9. Prefer understandable baselines before complex models.
10. When practical, leave a small bounded exercise or test for the owner.

See "Learning notes" below for the required write-up format.


## Current technology

Primary language:

- Python 3.12

Important tooling includes:

- pandas
- NumPy
- DuckDB
- PyArrow
- pytest
- Ruff
- Git
- YAML configuration
- yfinance / market-data adapters
- FRED / macroeconomic-data adapters
- SEC EDGAR / XBRL fundamental-data adapters

Future infrastructure may include AWS.

Do not introduce a new dependency merely because it simplifies implementation.

Before adding one, explain:

- value
- alternatives
- maintenance cost
- computational cost
- security implications
- deployment implications


# Engineering principles

## Source-of-truth hierarchy

When information conflicts, prefer:

1. actual source code and schemas
2. validated tests and invariants
3. Git history
4. architecture / ADR documentation
5. `docs/PROJECT_STATE.md`
6. agent memory and conversation context

Never treat agent memory as more authoritative than the repository.

If `docs/PROJECT_STATE.md` disagrees with Git, Git wins. Report the
discrepancy to the owner.


## Inspect before editing

Before modifying application code:

1. Inspect the current Git branch.
2. Inspect `git status`.
3. Inspect the relevant diff.
4. Read `docs/PROJECT_STATE.md`.
5. Read relevant implementation files.
6. Read relevant tests.
7. Identify affected callers and data contracts.
8. Explain the intended change and important risks.

Typical commands:

```bash
git branch --show-current
git status
git diff
git log --oneline -5
```

Do not make broad speculative edits before understanding the current design.

Uncommitted work in the tree may be the owner's intentional unfinished work.
Do not modify, revert, stage, or reformat it unless asked.


## Architecture may be challenged

Existing architecture is not sacred.

An agent may recommend a materially better design.

For substantial redesigns:

1. Explain the weakness of the current design.
2. Separate observed facts from inference.
3. Present meaningful alternatives.
4. Compare tradeoffs.
5. Describe migration risk.
6. Explain quantitative / point-in-time consequences where relevant.
7. Wait for owner approval before major restructuring.

Prefer the simplest design that preserves correctness and extensibility.

Avoid premature abstraction.


# Data and quantitative safety

## Point-in-time correctness

No model feature may contain information unavailable at the decision timestamp.

This applies to:

- SEC filing acceptance times
- amendments
- macroeconomic release times
- revised historical economic data
- asset-universe membership
- company fundamentals
- features
- labels
- benchmark construction
- training / validation / test splits

Lookahead leakage is a correctness defect, not merely a modeling preference.

Whenever a timestamp represents availability, document its semantics.

Rules:

- Every derived value is available no earlier than the latest
  availability timestamp of every input used to derive it.
- Later information (amendments, restatements, revisions) is stored as a
  new observation with its own availability timestamp. It never
  overwrites the earlier observation.
- As-of joins select the latest observation available at or before the
  decision timestamp, never the latest observation overall.
- When availability is uncertain, choose the later (conservative)
  timestamp and record which rule was used.


## Temporal validation

For investment time-series systems:

- prefer temporal or walk-forward validation
- do not default to random train/test splitting
- fit transforms and scalers only on information available in training
- prevent future observations from entering features
- prevent hyperparameter selection from leaking test-period information
- account for label horizons overlapping the split boundary
  (purge / embargo where relevant)


## Fiscal accounting semantics

Never infer company fiscal quarter purely from calendar month.

Different companies use different fiscal calendars, including
52/53-week years whose period ends move from year to year.

Use authoritative EDGAR/XBRL fiscal metadata whenever available.

For example:

```text
calendar February != necessarily fiscal Q1
calendar May      != necessarily fiscal Q2
calendar December != necessarily fiscal Q4
```

Canonical semantics:

- Fiscal identity (fiscal year, fiscal period, fiscal quarter) comes from
  XBRL reporting-period metadata matched to the filing's period end.
- 10-Q / 10-Q/A filings carry fiscal periods Q1, Q2, or Q3.
- 10-K / 10-K/A filings carry fiscal period FY, normalized to canonical
  fiscal quarter 4.
- A 10-K reports the full fiscal year, not a standalone Q4.
- Conflicting fiscal identities for one filing are rejected, not guessed.
- Flow metrics (revenue, income, cash flow) cover a duration and may be
  reported as standalone-quarter, year-to-date (YTD), or full-year values.
- Instant metrics (balance-sheet items) are point-in-time values and are
  never reconciled by subtraction.
- Filing availability uses the SEC acceptance timestamp when present.
  Otherwise it uses a conservative filing-date-based fallback, and
  records which rule was used.
- Amendments (10-Q/A, 10-K/A) are separate observations with their own
  accession number and availability timestamp.


## Deterministic fiscal-flow reconciliation

Standalone quarterly flow values are derived deterministically:

```text
Direct standalone-quarter value, when reported: use it as reported.

Q1 = Q1 YTD
Q2 = Q2 YTD - Q1 YTD
Q3 = Q3 YTD - Q2 YTD
Q4 = FY     - Q3 YTD
```

Cross-filing subtraction is allowed only when the prior observation:

- has the same symbol
- has the same metric
- has the same fiscal year
- is the immediately preceding fiscal quarter
- comes from a different SEC accession
- has an earlier fiscal period end
- was available no later than the current observation

Rules:

- Missing or ineligible inputs produce a structured "unavailable" result
  with a machine-readable reason. Never guess, interpolate, or silently
  substitute another observation.
- Every reconciled value records its method and the accession numbers
  used, so its lineage can be explained.
- Do not use a prior observation from the previous fiscal year.
- Do not use an observation that became available after the decision
  timestamp, including later amendments.


## Anomaly-detection hierarchy

Detect data problems in this order. Each level is only as trustworthy as
the levels below it.

1. Deterministic checks
   - schema, types, required fields
   - fiscal-identity and accession consistency
   - accounting identities and sign conventions
   - point-in-time ordering
   - reconciliation reason codes
2. Statistical checks
   - robust outlier measures (median / MAD)
   - quarter-over-quarter and year-over-year change bounds
   - cross-source comparison (e.g. EDGAR vs. another provider)
3. Machine-learning checks
   - only after deterministic and statistical baselines exist
   - only with point-in-time features and temporal evaluation
   - only with a documented false-positive / false-negative tradeoff

Anomaly detectors flag data. They never silently modify or drop it.
Corrections are explicit, recorded, and reviewable.


## Quantitative research standards

- Start with simple, explainable baselines (equal weight, inverse
  volatility, momentum) before complex models.
- Compare every strategy against appropriate benchmarks after transaction
  costs.
- Report risk, drawdown, turnover, and uncertainty, not only return.
- Treat multiple testing and repeated backtest tuning as sources of
  overfitting. Record how many variants were tried.
- Results must be reproducible from code, configuration, and recorded
  data versions.
- Separate observed results from interpretation.
- A strategy that cannot be explained should not be recommended.


# Quality and process

## Testing and Ruff quality gates

Before proposing a commit, all of these must pass:

```bash
pytest -q
python -m pytest -q
ruff format --check src tests scripts
ruff check src tests scripts
```

If formatting is required during implementation, it may run:

```bash
ruff format src tests scripts
```

Do not silently change this established repository workflow.

- Ruff line length is 100 (`pyproject.toml`).
- New behavior requires tests. Bug fixes require a regression test.
- Prefer test-first development for accounting and point-in-time logic.
- Tests must not require network access. Inject fakes through the existing
  dependency seams (for example filing listers, filing loaders, clocks).
- Point-in-time and reconciliation invariants deserve explicit tests,
  including negative tests proving that leakage is rejected.
- Report failing tests honestly with their output. Never weaken a test to
  make it pass without owner approval.


## Git safeguards

- Work on feature branches, never directly on `main`.
- Check `git status` before and after every change.
- Stage only files related to the current task. Never sweep unrelated or
  owner-owned uncommitted work into a commit.
- Use conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`,
  `refactor:`).
- Never rewrite published history, force-push, `reset --hard`, or discard
  uncommitted work without explicit owner approval.
- Never claim work was committed, pushed, or merged unless verified
  from Git.


## Owner approval gates

Explicit owner approval is required before:

- pushing, merging, or opening pull requests
- deleting or renaming branches
- rewriting Git history
- adding, removing, or upgrading dependencies
- changing secrets, credentials, or environment configuration
- modifying production or cloud infrastructure
- changing data schemas or data contracts
- changing point-in-time or availability semantics
- deleting stored data
- major architectural restructuring


## Agent workflow

1. Inspect (see "Inspect before editing").
2. Restate the task and identify affected contracts.
3. Plan. For non-trivial work, share the plan before editing.
4. Write or update tests first where practical.
5. Implement the smallest change that satisfies the tests.
6. Run the quality gates.
7. Review the diff for leakage, unrelated changes, and dead code.
8. Report what changed, what was verified, and what remains.
9. Propose an update to `docs/PROJECT_STATE.md`.


## Learning notes

For substantial quantitative, statistical, ML, or data-science work,
include a `Learning Notes` section covering:

- problem being solved
- relevant statistical or financial concept
- assumptions
- mathematical formulation when useful
- small example when useful
- leakage risks
- ways results could mislead
- practitioner validation
- what the owner should learn next


## Session continuity

`docs/PROJECT_STATE.md` carries state between sessions and tools.

- Read it at the start of every session, then verify it against Git.
- It records facts that change: branch, last verified commit, working-tree
  state, current objective, quality baseline, next stage, open questions.
- It does not restate permanent policy from this file.
- Every factual claim should be verifiable or dated.
- Update it at the end of substantial work, with owner approval.


## Dashboard and observability direction

The system should become observable, not only correct.

Intended direction:

- data-quality dashboards built from structured diagnostics
  (status, reason, detail) rather than log text
- coverage of fundamentals by symbol, fiscal period, and metric
- distribution of reconciliation methods and unavailable reasons
- data freshness and availability-lag monitoring
- model and portfolio monitoring: drift, turnover, risk, attribution

Design implication: pipelines should emit machine-readable diagnostics and
lineage now, so dashboards can be built later without re-engineering.


## Computational efficiency and agent readiness

- Prefer vectorized pandas / NumPy / DuckDB operations over row loops
  where clarity is preserved.
- Avoid repeated network requests. Cache raw provider data and separate
  retrieval from transformation.
- Keep functions deterministic, with explicit inputs (inject clocks and
  I/O) so results are reproducible and testable.
- Return structured results with reason codes rather than raising or
  logging free text for expected data conditions.
- Keep modules small with clear contracts, so an agent can understand
  one area without loading the whole codebase.
- Measure before optimizing. Do not trade correctness or point-in-time
  safety for speed.
