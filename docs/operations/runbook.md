# Model Health runbook

One entry per health check: what it means, likely causes, exact diagnostic
commands, a safe retry and the fallback behaviour. The design, the summary
format and the local monitoring set-up are in
[`model_health.md`](model_health.md).

Rules that apply to every entry:

- **Alerts are informational.** Nothing in the health center trades,
  retrains, promotes or demotes a model, deletes data, or changes
  configuration. Every action below is taken by a person.
- **Never make a check pass by changing what it measures.** Do not edit
  stored data, delete a test, loosen a threshold or re-run a producer with
  different settings to turn a check green. Weakening a test or a threshold
  is an owner approval gate (AGENTS.md).
- **Fallback means hold.** When a check is not healthy, nothing new is
  decided from the affected area; existing published, write-once artifacts
  stay as they are and keep their own labels. Stale or missing data is
  never filled, and a decision is never made on stale prices (risk
  register, OPS-4).
- **A safe retry repeats the measurement, not the fix.** Re-run the health
  check once; if the result is the same, follow the diagnosis.

Commands run from the repository root with the project virtual environment.
`PRIVATE` stands for the private monitoring directory (default
`~/stock-agent-private/health`). Producers write their health inputs to
`reports/health_inputs/<artifact>/<run_id>.json` (ignored by Git); the local
monitor reads the newest file per artifact.

## States

| State | Meaning | Alert |
|---|---|---|
| failed | Measured, and a hard limit or contract is broken | failure, or constraint violation for a breached stored limit |
| unavailable | A measurement that should exist could not be taken | unavailable data |
| degraded | Measured, outside a warning threshold, still usable | degraded |
| insufficient_evidence | Nothing to judge yet (no input, or too few observations) | none |
| healthy | Measured, within its threshold | none |

A check that does not apply (coverage of a point forecast) is listed as
**not applicable**: shown, without a state, never counted.

The overall state is the worst state of any check, in the order of the
table. A check that was never measured is never healthy. A numeric value
that moves in its worse direction by more than its tolerance, while its
state stays the same, raises a deterioration alert in the local monitor.

## Getting the summaries

```bash
.venv/bin/python scripts/health_check.py ci --run-verify --output-dir reports/health/ci-local
```

```bash
RUN=$(gh run list --repo billvo9/stock-allocation-agent --workflow health.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

```bash
gh run download "$RUN" --repo billvo9/stock-allocation-agent --name model-health --dir "reports/health/ci/$RUN"
```

```bash
.venv/bin/python scripts/run_model_health.py
```

`reports/` is ignored by Git. The first command reproduces the daily
workflow's checks on this machine; the next two download the latest
workflow summary; the last opens the Model Health page.

## CI and tests

<a id="ci-verify"></a>
### Quality gates (`ci.verify`)

**What it means.** The result of `scripts/verify.py`: the two pytest passes,
the format check and the lint check. Healthy when every gate passed within
the last 7 days; degraded when the result is older; failed when any gate
failed; unavailable when verify could not run or its output is unreadable.

**Likely causes.** A code change broke a test or the formatting; an
unpinned dependency (pandas, numpy, pyarrow, ...) released a version that
changes behaviour (the daily workflow installs the newest allowed
versions); the `.venv` is missing or is not Python 3.12.

**Diagnose.**

```bash
python3 scripts/verify.py
```

```bash
.venv/bin/python -m pip list --format=freeze
```

Compare the second command's output with the last green run's evidence
(`evidence/verify.log` in the workflow artifact). For a CI failure:

```bash
gh run view "$RUN" --repo billvo9/stock-allocation-agent --log-failed
```

**Safe retry.** Run `python3 scripts/verify.py` once more. A gate that
passes on retry is a flaky test: record it as a defect; do not re-run until
green.

**Fallback.** Do not merge or publish from the failing commit. Published
artifacts from earlier commits are unaffected and stay readable.

<a id="ci-tests"></a>
### Test suite (`ci.tests`)

**What it means.** The pytest counts from both verify passes. Healthy when
both report the same counts with 0 failed and 0 errors; failed when any test
failed or errored, or a pass ran no tests; degraded when the passes
disagree (non-determinism) or the result is older than 7 days. A falling
number of passing tests raises a deterioration alert: tests were deleted,
skipped or deselected.

**Likely causes.** A regression; order- or time-dependent tests (the
AppTest timeouts use wall-clock time, so a machine sleeping mid-run can fail
them); a deleted or newly skipped test.

**Diagnose.**

```bash
.venv/bin/python -m pytest -q
```

```bash
.venv/bin/python -m pytest -q --lf
```

```bash
git log --oneline --diff-filter=D -- tests | head
```

The last command lists recent commits that deleted test files.

**Safe retry.** Re-run the failing tests alone with
`.venv/bin/python -m pytest -q <test path>`; a pass on retry is flaky, not
fixed.

**Fallback.** As for `ci.verify`. Deleting or weakening a test is an owner
approval gate.

<a id="ci-main-runs"></a>
### Latest CI run on main (`ci.main_runs`)

**What it means.** The conclusion of the latest completed `ci.yml` run on
`main`, from a run list the owner writes with the GitHub CLI (local monitor
only; nothing in this package contacts GitHub). Healthy on `success`;
failed on `failure` or `startup_failure`; degraded on any other conclusion
(`cancelled`, `timed_out`, ...) or when the list is older than 7 days.

**Likely causes.** A merge that broke `main`; a transient runner or network
failure; GitHub Actions disabled or out of minutes.

**Diagnose.**

```bash
gh run list --repo billvo9/stock-allocation-agent --workflow ci.yml --branch main --limit 20 --json databaseId,status,conclusion,headSha,createdAt,url,event > ~/stock-agent-private/health/ci_runs.json
```

```bash
gh run view <run id> --repo billvo9/stock-allocation-agent --log-failed
```

**Safe retry.** Re-write the run list (first command) and re-run the local
monitor. Re-running the CI run itself is the owner's decision
(`gh run rerun <run id> --failed`), and only for a failure that is clearly
transient.

**Fallback.** Treat `main` as broken: no merge, no publication from it,
until a run on a fix succeeds.

## Dependencies

<a id="deps-pins"></a>
### Pinned dependencies (`deps.pins`)

**What it means.** Every `==` pin in `requirements.txt` is installed at
exactly that version in the interpreter that ran the health check. Healthy
when all match; failed on any mismatch or missing package. Unpinned
requirements are listed and not judged: several are unpinned by owner
decision.

**Likely causes.** A manual `pip install --upgrade`; a stale `.venv` after a
pin changed on `main`; the check ran under another interpreter.

**Diagnose.**

```bash
.venv/bin/python -m pip show ruff pytest streamlit plotly scikit-learn edgartools
```

```bash
git log --oneline -5 -- requirements.txt
```

The check's detail names the interpreter it inspected.

**Safe retry.** Reinstall the pinned set, which changes nothing in the
repository: `.venv/bin/python -m pip install -r requirements.txt`. Changing
or adding a pin is an owner approval gate.

**Fallback.** Results from the mismatched environment are not reproducible
evidence; re-run them after reinstalling.

<a id="deps-pip-check"></a>
### Dependency consistency (`deps.pip_check`)

**What it means.** `python -m pip check` found no installed package whose
declared requirements are unmet. Failed when it reports any; unavailable
when pip could not run.

**Likely causes.** An unpinned dependency released a version outside
another package's range (for example a Streamlit release excluding a
pyarrow version); a partial install.

**Diagnose.**

```bash
.venv/bin/python -m pip check
```

```bash
.venv/bin/python -m pip show pyarrow streamlit
```

**Safe retry.** Run `pip check` again after
`.venv/bin/python -m pip install -r requirements.txt`.

**Fallback.** Do not upgrade anything to "fix" it without owner approval
(dependency changes are an approval gate); report the conflicting pair.

## Artifact contracts

<a id="contract-publication"></a>
### Publication contract (`contract.publication`)

**What it means.** Four probes of `stock_agent.publication` on a synthetic
fixture: a written publication reads back identical; an edited file is
refused (`hash_mismatch`); rows in the lockbox are refused
(`lockbox_violation`); a dirty-tree publication is refused by the default
reader (`dirty_tree`). Failed when any probe misbehaves.

**Likely causes.** A change to the writer, reader or `policy.validate`
that weakened a refusal; a pandas or pyarrow change in how parquet
round-trips.

**Diagnose.**

```bash
.venv/bin/python -m pytest -q tests/test_publication_contract.py tests/test_publication_writer.py tests/test_health_contracts.py
```

The check's detail names each broken probe and what happened instead.

**Safe retry.** Re-run the health check; the probes use a fresh temporary
directory each time.

**Fallback.** Write no new publication from this code. Existing
publications are still verified by the reader on every read.

<a id="contract-fund-data"></a>
### Fund-data contract (`contract.fund_data`)

**What it means.** Five probes of the F1 fund-data store on the synthetic
adapter: an ingested batch reads back complete with schema 1.0; the same
batch twice is refused (`already_exists`); a request past the lockbox is
refused (`request_after_cap`); an availability rule is refused before OD9
(`pending_od9`); an edited table is refused (`hash_mismatch`).

**Likely causes.** A change in `stock_agent.data.funds` that weakened a
refusal; a calendar change; a parquet round-trip change.

**Diagnose.**

```bash
.venv/bin/python -m pytest -q tests/test_fund_data_store.py tests/test_fund_data_boundaries.py tests/test_health_contracts.py
```

**Safe retry.** Re-run the health check.

**Fallback.** Ingest no new batch with this code. Stored batches are
re-verified by `read_store` on every read.

<a id="contract-generators"></a>
### Synthetic generators (`contract.generators`)

**What it means.** Each synthetic generator, run twice, reproduces the
SHA-256 recorded in `stock_agent.health.contracts.RECORDED_HASHES`. Failed
when two runs differ (non-determinism) or the output changed.

**Likely causes.** A generator or its parameters were edited; numpy's
legacy `RandomState` stream or pandas' hashing behaviour changed; a
calendar change moved sessions.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.health.contracts import generator_hashes, RECORDED_HASHES; print(generator_hashes()); print(RECORDED_HASHES)"
```

```bash
git log --oneline -5 -- src/stock_agent/publication/synthetic.py src/stock_agent/data/funds/adapter.py
```

**Safe retry.** Re-run the health check. Update a recorded hash only for an
intended, reviewed generator version change, in the same commit.

**Fallback.** Synthetic publications and the Explore demo are not
reproducible until resolved; do not compare results across the change.

## Monitor self-tests

<a id="smoke-data-monitors"></a>
### Data monitors on planted defects (`smoke.data_monitors`)

**What it means.** The four data monitors run on a clean synthetic store
(all healthy) and on one with planted defects (3 missing tail sessions, 2
closes moved 1 % in a second source), and must report exactly those
defects. Failed when a monitor misses a defect or flags a clean store.

**Likely causes.** A change in `stock_agent.health.data_quality` or its
thresholds; a change in the synthetic adapter's output.

**Diagnose.**

```bash
.venv/bin/python -m pytest -q tests/test_health_data_quality.py
```

**Safe retry.** Re-run the health check.

**Fallback.** Treat every `data.*` result in the local monitor as
unverified until this passes again.

<a id="smoke-input-readers"></a>
### Producer-file readers on fixtures (`smoke.input_readers`)

**What it means.** The readers of other tracks' files (training health,
registry, backtest) run on the example files in
`stock_agent.health.fixtures`: clean files are healthy, a planted cap breach
fails only its check, absent files are insufficient evidence and an
unreadable file is unavailable.

**Likely causes.** A reader or format change in `stock_agent.health`
without a matching fixture change.

**Diagnose.**

```bash
.venv/bin/python -m pytest -q tests/test_health_inputs.py
```

**Safe retry.** Re-run the health check.

**Fallback.** Treat `training.*`, `registry.*`, `forecast.*` and
`portfolio.*` results as unverified until this passes again.

## Data quality (local monitor)

These read a private F1 fund store (`--fund-store`) and never modify it.
Only batches retrieved at or before the run time count.

<a id="data-freshness"></a>
### Data freshness (`data.freshness`)

**What it means.** For each fund, the NYSE sessions between its newest
stored close and the latest permitted session: the last session before the
run date, the holdout start (2025-01-01) and the end of the frozen calendar
(2024-12-31). Healthy at 0 or 1 session behind; degraded beyond, or when a
requested fund has no close at all; unavailable with no closes in the store.

**Likely causes.** The newest batch requested a window ending earlier than
the cap; a vendor returned fewer sessions than requested.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('PATH_TO_STORE'); print(s.batches[['batch_id','request_start','request_end_exclusive','retrieved_at']]); print(s.prices.groupby('fund_id')['session_date'].agg(['min','max','size']))"
```

**Safe retry.** Re-run the local monitor. Fetching a new batch is an owner
action under the vendor terms and the lockbox cap.

**Fallback.** Pages built on this store show their stale and
unavailable states; no decision uses the missing sessions. Nothing is
filled.

<a id="data-coverage"></a>
### Session coverage (`data.coverage`)

**What it means.** The share of requested fund-sessions (every batch's
request, combined) that have a close. Healthy at 99.5 % or more; degraded
below; failed below 95 %.

**Likely causes.** Vendor gaps; a fund that did not trade for part of the
request (a later inception date); an instrument mapping error.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('PATH_TO_STORE'); print(s.missing_sessions.groupby('fund_id').size())"
```

**Safe retry.** Re-run the local monitor.

**Fallback.** Descriptive calculations already label incomplete histories
and sessions without a close; no gap is filled or interpolated.

<a id="data-missing-sessions"></a>
### Missing sessions (`data.missing_sessions`)

**What it means.** Requested fund-sessions without a close. Healthy at 0;
degraded with any; failed when one fund misses 5 or more consecutive
sessions (a broken feed rather than isolated gaps).

**Likely causes.** As for coverage; a feed that stopped mid-request.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('PATH_TO_STORE'); print(s.missing_sessions.sort_values(['fund_id','session_date']).to_string())"
```

**Safe retry.** Re-run the local monitor.

**Fallback.** As for coverage.

<a id="data-vendor-discrepancy"></a>
### Source discrepancies (`data.vendor_discrepancy`)

**What it means.** For fund-sessions held by two or more sources with the
same adjustment basis (each source's latest retrieval), the relative
difference (max − min) / min of their closes. Healthy when none exceeds
0.1 %; degraded otherwise; insufficient evidence with a single source or no
overlap.

**Likely causes.** Consolidated versus primary-exchange closes; a split or
distribution applied by one source; a wrong instrument mapping.

**Diagnose.** The check's detail lists the largest differences. Compare the
two sources' rows:

```bash
.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('PATH_TO_STORE'); p = s.prices; print(p[(p.fund_id=='FUND') & (p.session_date=='YYYY-MM-DD')].to_string())"
```

**Safe retry.** Re-run the local monitor.

**Fallback.** Never pick a source silently. Record the discrepancy; which
source a page uses is an owner decision.

## Training (local monitor)

<a id="training-health"></a>
### Training health (`training.health`)

**What it means.** The newest RL training-health file
(`reports/health_inputs/rl_training_health/<run_id>.json`, by
`generated_at`). The check takes its worst finding:

- failed: the run failed, or any producer check failed;
- degraded: a check warns, the run uses an engineering-only portfolio
  transition that is not yet approved (`unapproved_transition`, a visible
  limitation, not a failure of the run), or some input files are unreadable;
- insufficient evidence: the run is still in progress, one of the eight
  listed checks (`nan_or_inf`, `exploding_values`, `approx_kl`,
  `clip_fraction`, `entropy_collapse`, `value_loss`, `reward_trend`,
  `seed_spread`) did not run or is absent, or no file exists yet;
- healthy: a completed run whose eight listed checks all passed.

The producer's own statuses are used; nothing is recomputed. Extra check ids
are shown.

**Likely causes.** Diverging losses, entropy collapse, non-finite values,
too few seeds or a large spread between them, an environment bug; or the
transition is simply not approved yet.

**Diagnose.** Open the file named in the check's detail (run id, seeds,
scenario version) and read the failing producer checks; then the RL track's
own diagnostics.

```bash
ls -lt reports/health_inputs/rl_training_health/
```

**Safe retry.** Re-run the local monitor. Retraining is a separate,
deliberate run; nothing retrains automatically. Approving the transition is
an owner decision (T5a).

**Fallback.** Do not use the run's policy for anything; the previously
registered artifact, if any, stays as it was.

## Model registry and forecasts (local monitor)

These read the ML track's model registry (`--registry`, normally
`reports/ml/registry`; artifacts root `--ml-artifacts`, default its parent)
through its read API, `stock_agent.registry.open_registry`, which re-verifies
everything it reads (docs/research/ml_registry.md). Health never writes to
the registry and changes no stage. The diagnostic commands below use the same
read API; replace `ROOT` with the registry root.

<a id="registry-entries"></a>
### Registry entries and artifacts (`registry.entries`)

**What it means.** Every registry entry loads and hash-verifies, the
activation event chain verifies, and every entry's artifacts (experiment
runs, forecast artifacts) still match their SHA-256. Failed with
`registry_refused` when the reader refuses the registry (an edited entry or
event line), or `artifacts_unverified` when an entry's artifacts no longer
match; insufficient evidence with no entries.

**Likely causes.** A file under `reports/ml/` was edited, moved or deleted
(`reports/` is ignored by Git and easy to clean by mistake); a partial copy
of the registry; an artifacts root other than the one the entries were
registered with.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); print(r.summary().to_string())"
```

```bash
.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); print([str(e.ref) for e in r.entries() if not r.artifacts_verified(e)])"
```

**Safe retry.** Re-run the local monitor with the same `--registry` and
`--ml-artifacts`. Restoring a deleted artifact from a backup is the owner's
step; re-running an experiment registers a new version, it never repairs an
old one.

**Fallback.** Versions whose artifacts do not verify drop to the floor stage
by the registry's own rule; nothing compares against them or promotes them.

<a id="registry-stages"></a>
### Computed stages (`registry.stages`)

**What it means.** The stage the registry computes for each version from
verified evidence. The value is the number of versions above the floor
stage (`engineering_only`); a drop raises a deterioration alert in the local
monitor. The upper stages are fail-closed until the owner approves their
rules, so the floor is the expected stage today; the detail lists the most
common blockers.

**Likely causes.** A drop: an artifact altered or deleted; evidence that no
longer re-derives from its run record.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); print(r.assessment('MODEL@vN').frame().to_string())"
```

**Safe retry.** Re-run the local monitor.

**Fallback.** A stage is computed per read, so a dropped version is simply
treated at its lower stage; nothing is promoted or demoted by hand.

<a id="registry-selection-history"></a>
### Reference-version history (`registry.selection_history`)

**What it means.** The activation and rollback events that record each
research scope's reference version, and whether every currently active
version still verifies. Failed with `active_unverified` when an active
version's artifacts no longer verify; insufficient evidence before the first
activation. "Active" is research bookkeeping, never a deployment.

**Likely causes.** As for `registry.entries`, for the active version.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); [print(e) for e in r.events()]"
```

**Safe retry.** Re-run the local monitor. Activating or rolling back a
version is a deliberate research decision (`Registry.activate` /
`Registry.rollback` with a stated reason), never an automatic response.

**Fallback.** Comparisons against an unverified reference are not made.

<a id="registry-experiments"></a>
### Experiment counts (`registry.experiments`)

**What it means.** Every version records its multiple-testing counts
(candidates in its run, grid points, prior variants in its scope). Failed
when a version records no candidate count; degraded when a version's
`prior_variants_in_scope` is below the number of versions registered earlier
in its scope, which would understate how many variants were tried.

**Likely causes.** Variants tried outside the experiment runner; a counter
reset between runs; a scope renamed.

**Diagnose.**

```bash
.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); [print(e.ref, e.created_at, e.evidence['multiple_testing']) for e in r.entries()]"
```

**Safe retry.** Re-run the local monitor.

**Fallback.** Report results with the larger, conservative count until
resolved.

<a id="forecast-coverage"></a>
### Forecast interval coverage (`forecast.coverage`)

**What it means.** From the newest registry export
(`reports/health_inputs/model_registry/<run_id>.json`): for interval and
probabilistic forecasts with at least 50 observations and a reported
coverage, the largest gap between observed coverage and the stated level.
Healthy at 0.05 or less; degraded up to 0.10; failed beyond. A screening
band, not a statistical test (see the Learning Notes in `model_health.md`).
Point forecasts are named as "not applicable: point forecast" and never count;
an export of point forecasts only lists this check as not applicable.

**Likely causes.** Mis-specified uncertainty; a regime change; overlapping
horizons that make a short sample look better or worse than it is.

**Diagnose.** Read the export's `forecast_evaluation` rows named in the
detail, then the experiment run's calibration diagnostics.

```bash
ls -lt reports/health_inputs/model_registry/
```

**Safe retry.** Re-run the local monitor.

**Fallback.** Forecast intervals are not shown as calibrated; nothing is
promoted.

<a id="forecast-calibration"></a>
### Forecast calibration (`forecast.calibration`)

**What it means.** The producer's own calibration status for each interval
or probabilistic evaluation in the export: failed on any fail, degraded on
any warning, insufficient evidence when none ran. Point forecasts do not
apply, as for coverage.

**Likely causes.** As for coverage.

**Diagnose.** As for coverage.

**Safe retry.** Re-run the local monitor.

**Fallback.** As for coverage.

## Portfolio (simulated backtests, local monitor)

Backtest summaries are read from `reports/health_inputs/backtest_summary/`
(the newest by `generated_at`); each names its `policy`. A backtest is
simulated, never a track record. These checks compare stored measures (from
the approved T5a formulas only) with the backtest's own stored limits and
never recompute turnover, cost or drawdown (owner-gated, OD8). Each takes its
worst finding: a measure above its limit fails with `constraint_breached` and
raises an informational constraint-violation alert; an engineering-only
transition that is not yet approved is degraded (`unapproved_transition`);
a measure or limit that is not reported is insufficient evidence.

<a id="portfolio-turnover"></a>
### Portfolio turnover (`portfolio.turnover`)

**What it means.** Stored turnover against the declared `max_turnover`.

**Likely causes.** A rebalance rule trading more than designed; a missing
no-trade band; noisy weights.

**Diagnose.** Read the backtest's own record named in the evidence; compare
its rebalance log with its rule.

**Safe retry.** Re-run the local monitor.

**Fallback.** The construction is not eligible for promotion until reviewed;
nothing trades.

<a id="portfolio-costs"></a>
### Trading costs (`portfolio.costs`)

**What it means.** Stored estimated costs against the declared
`max_cost_bps`.

**Likely causes.** Higher turnover; a cost-model change.

**Diagnose.** As for turnover.

**Safe retry.** Re-run the local monitor.

**Fallback.** As for turnover.

<a id="portfolio-concentration"></a>
### Concentration (`portfolio.concentration`)

**What it means.** The stored largest single weight against the declared
`max_weight` cap. A breach means the construction violated its own cap.

**Likely causes.** A cap not enforced after drift between rebalances; a
rounding or normalization error.

**Diagnose.** As for turnover; check the weights on the dates in the
backtest's record.

**Safe retry.** Re-run the local monitor.

**Fallback.** As for turnover.

<a id="portfolio-drawdown"></a>
### Drawdown (`portfolio.drawdown`)

**What it means.** The stored maximum drawdown against the declared
`max_drawdown`.

**Likely causes.** A stress period in the sample; higher risk than
designed.

**Diagnose.** As for turnover.

**Safe retry.** Re-run the local monitor.

**Fallback.** As for turnover.
