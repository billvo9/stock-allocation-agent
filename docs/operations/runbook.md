# Model Health runbook

What to do when a health check is not healthy: where to look, how to
reproduce, when to retry, and when to disable a policy or rely on the
registered fallback. The design, the summary format and the scheduled local
run are in [`model_health.md`](model_health.md).

Rules for every entry:

- **Alerts are informational.** Nothing in the health center trades,
  retrains, promotes or demotes a model, deletes data, or changes
  configuration. Every action below is taken by a person.
- **Never make a check pass by changing what it measures.** Do not edit
  stored data, delete a test, loosen a threshold or re-run a producer with
  different settings to turn a check green. Weakening a test or a threshold
  is an owner approval gate (AGENTS.md).
- **Fallback means hold.** When a check is not healthy, nothing new is
  decided from the affected area; existing write-once artifacts keep their
  own labels. Stale or missing data is never filled, and nothing is decided
  on stale prices (risk register, OPS-4).
- **A safe retry repeats the measurement, not the fix.**

## Where to look

1. **Portfolio Lab, Model Health** (`.venv/bin/python
   scripts/run_portfolio_lab.py`): the overall state, the next step, each
   area's worst state, and each check's exact value, threshold, reason code,
   evidence and runbook link (under "Exact measurements").
2. **The private summary** (`PRIVATE` = `~/stock-agent-private/health`):
   `latest.json`, the write-once `history/` and per-run `evidence/<stamp>/`
   (verify log, pip check, GitHub run list).
3. **GitHub Actions**: the `CI` and `Model health` workflows. The daily
   summary is the `model-health` artifact (synthetic fixtures only).
4. **The technical page** (`.venv/bin/python scripts/run_model_health.py`):
   both summaries, every field, one tab each.

## States

| State | Meaning | Alert |
|---|---|---|
| failed | Measured, and a hard limit or contract is broken | failure, or constraint violation for a breached stored limit |
| unavailable | A measurement that should exist could not be taken | unavailable data |
| degraded | Measured, outside a warning threshold, still usable | degraded |
| insufficient_evidence | Nothing to judge yet (no input, or too few observations) | none |
| healthy | Measured, within its threshold | none |

The overall state is the worst check's. A check that was never measured is
never healthy. A check that does not apply is listed as **not applicable**
and never counts. In the local monitor a synthetic input is at most
insufficient evidence (`synthetic_input`). A value that moves in its worse
direction while its state stays the same raises a deterioration alert (a
falling ledger entry count is how a truncated ledger shows).

## Reproduce a failure

```bash
.venv/bin/python scripts/health_check.py ci --run-verify --output-dir reports/health/ci-local
```

```bash
.venv/bin/python scripts/health_check.py local --run-verify --github
```

The first repeats the daily workflow's checks; the second the private run
(add the inputs your scheduled run uses, for example `--prospective-store`
and `--ledger-head-backup`). Each entry below lists the narrower commands.
For a failed workflow run:

```bash
gh run view <run id> --repo billvo9/stock-allocation-agent --log-failed
```

## When to retry

Retry once, and only the measurement, when the cause can be transient: the
GitHub CLI was offline or not signed in (`gh_unavailable`), a workflow run
was `cancelled` or timed out, an AppTest timed out on a machine that slept,
a file was being written. A result that repeats is the diagnosis. Never
retry until green: a failed contract probe, a broken ledger chain, a kept
head missing from the ledger, a constraint violation, a refused registry or
store, and a failing test are not transient.

## When to disable a policy or rely on the fallback

- **The registered fallback is automatic, per decision, inside a guarded
  policy** (`stock_agent.rl.safety`): fixed balanced weights, or holding the
  current positions on stale inputs. Health only counts it
  (`portfolio.fallbacks`); it never switches anything.
- **Disable a policy** (a person stops using it; nothing does it
  automatically) when its `training.health` is failed or unavailable, when
  `portfolio.violations` is failed for it, when its registry version no
  longer verifies (`registry.entries`, `registry.selection_history`), or
  when `portfolio.fallbacks` shows it overruled by `model_unhealthy` or on
  a large share of decisions.
- **Rely on the fallback, and decide nothing new,** while its inputs are
  stale or missing (`data.freshness`, `data.window_coverage`,
  `data.missing_sessions`) or while the forecast ledger is refused
  (`forecast.ledger`).
- Every policy is engineering-only until its transition is approved
  (`unapproved_transition`): it is never used for a real decision.

The scheduled private run (launchd steps, outputs, keeping it private) is in
[`model_health.md`](model_health.md#scheduling-it-owner-steps-nothing-here-installs-them).

## CI and tests

<a id="ci-verify"></a>
### Quality gates (`ci.verify`)

- **What it means.** `scripts/verify.py`'s four gates. Healthy within 7 days; degraded when older; failed on any failing gate; unavailable when verify could not run or its log has no `health_check: verify.py finished at` first line (a log is never dated by its file time).
- **Likely causes.** A code change; a new release of an unpinned dependency (the daily workflow installs the newest allowed); a missing or wrong `.venv`.
- **Diagnose.** `python3 scripts/verify.py`, then compare `.venv/bin/python -m pip list --format=freeze` with the last green run's `evidence/verify.log`.
- **Safe retry.** Once. A pass on retry is a flaky test: record it as a defect.
- **Fallback.** No merge or publication from the failing commit; earlier artifacts stay readable.

<a id="ci-tests"></a>
### Test suite (`ci.tests`)

- **What it means.** Both pytest passes report the same counts with 0 failed and 0 errors. Failed on any failure or a pass that ran nothing; degraded when the passes disagree or the result is older than 7 days. Fewer passing tests raise a deterioration alert.
- **Likely causes.** A regression; order- or clock-dependent tests; a deleted or newly skipped test.
- **Diagnose.** `.venv/bin/python -m pytest -q --lf`; `git log --oneline --diff-filter=D -- tests | head`.
- **Safe retry.** The failing test alone: `.venv/bin/python -m pytest -q <test path>`.
- **Fallback.** As for `ci.verify`. Deleting or weakening a test is an owner approval gate.

<a id="ci-main-runs"></a>
### Latest CI run on main (`ci.main_runs`)

- **What it means.** The latest completed `CI` workflow run on `main`, from the run list the local monitor records with `--github` (or a list passed with `--ci-runs`). Healthy on `success`; failed on `failure`; degraded on another conclusion or a list older than 7 days; insufficient evidence without a list (`not_collected`, or `gh_unavailable` when gh could not run).
- **Likely causes.** A merge that broke `main`; a runner or network failure; Actions disabled.
- **Diagnose.** `gh run list --repo billvo9/stock-allocation-agent --workflow ci.yml --branch main --limit 5`, then `gh run view <run id> --log-failed`.
- **Safe retry.** Re-run the local monitor. Re-running the workflow (`gh run rerun <run id> --failed`) is the owner's call, for a clearly transient failure.
- **Fallback.** Treat `main` as broken: no merge or publication until a fix's run succeeds.

<a id="ci-daily-health"></a>
### Latest daily health run (`ci.daily_health`)

- **What it means.** The latest completed `Model health` workflow run on `main` (the daily smoke test on synthetic fixtures). Healthy on a success less than 48 hours old; degraded when older (`daily_run_overdue`) or inconclusive; failed on `failure`; insufficient evidence without a run list.
- **Likely causes.** A public check failed (open the run's summary); GitHub disabled the schedule after 60 days without activity; Actions minutes ran out.
- **Diagnose.** `gh run list --repo billvo9/stock-allocation-agent --workflow health.yml --limit 5`; download the summary with the commands on the Model Health page.
- **Safe retry.** `gh workflow run health.yml --repo billvo9/stock-allocation-agent` (owner) and re-run the local monitor.
- **Fallback.** Rely on the local monitor's own `ci.verify` until the daily run is green again.

## Dependencies

<a id="deps-pins"></a>
### Pinned dependencies (`deps.pins`)

- **What it means.** Every `==` pin in `requirements.txt` (with its environment marker) is installed at exactly that version in the interpreter that ran the check. Unpinned requirements are listed, not judged.
- **Likely causes.** A manual upgrade; a stale `.venv` after a pin changed; another interpreter.
- **Diagnose.** `git log --oneline -5 -- requirements.txt`; the check's detail names the interpreter.
- **Safe retry.** `.venv/bin/python -m pip install -r requirements.txt` (changes nothing in the repository), then re-run.
- **Fallback.** Results from the mismatched environment are not reproducible evidence. Changing a pin is an owner approval gate.

<a id="deps-pip-check"></a>
### Dependency consistency (`deps.pip_check`)

- **What it means.** `pip check` finds no package with unmet requirements. Failed when it reports any; unavailable when pip could not run.
- **Likely causes.** An unpinned dependency released outside another's range; a partial install.
- **Diagnose.** `.venv/bin/python -m pip check`.
- **Safe retry.** After `.venv/bin/python -m pip install -r requirements.txt`.
- **Fallback.** Upgrade nothing without owner approval; report the conflicting pair.

## Artifact contracts

<a id="contract-publication"></a>
### Publication contract (`contract.publication`)

- **What it means.** Probes on a synthetic publication: it reads back identical; an edited file, rows in the lockbox and a dirty-tree publication are refused.
- **Likely causes.** A change that weakened a refusal in `stock_agent.publication`; a parquet round-trip change.
- **Diagnose.** `.venv/bin/python -m pytest -q tests/test_publication_contract.py tests/test_health_contracts.py`.
- **Safe retry.** Re-run the health check (fresh temporary directory each time).
- **Fallback.** Write no new publication from this code; existing ones are re-verified on every read.

<a id="contract-fund-data"></a>
### Fund-data contract (`contract.fund_data`)

- **What it means.** Probes of the F1 fund store on the synthetic adapter: a batch reads back; a duplicate, a request past the lockbox, an availability rule before OD9 and an edited table are refused.
- **Likely causes.** A weakened refusal in `stock_agent.data.funds`; a calendar change.
- **Diagnose.** `.venv/bin/python -m pytest -q tests/test_fund_data_store.py tests/test_health_contracts.py`.
- **Safe retry.** Re-run the health check.
- **Fallback.** Ingest no new batch with this code.

<a id="contract-generators"></a>
### Synthetic generators (`contract.generators`)

- **What it means.** Each synthetic generator, run twice, reproduces its recorded SHA-256.
- **Likely causes.** An edited generator; a numpy, pandas or calendar change.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.health.contracts import generator_hashes, RECORDED_HASHES; print(generator_hashes()); print(RECORDED_HASHES)"`.
- **Safe retry.** Re-run. Update a recorded hash only for an intended, reviewed generator change.
- **Fallback.** Do not compare synthetic results across the change.

## Monitor self-tests

<a id="smoke-data-monitors"></a>
### Data monitors on planted defects (`smoke.data_monitors`)

- **What it means.** The fund-store monitors find exactly the defects planted in a synthetic store and nothing in a clean one.
- **Likely causes.** A change in `stock_agent.health.data_quality` or the synthetic adapter.
- **Diagnose.** `.venv/bin/python -m pytest -q tests/test_health_data_quality.py`.
- **Safe retry.** Re-run the health check.
- **Fallback.** Treat every `data.*` fund-store result as unverified until it passes.

<a id="smoke-input-readers"></a>
### Producer-file readers on fixtures (`smoke.input_readers`)

- **What it means.** The training, registry and backtest readers judge their example files correctly (clean, planted breach, absent, unreadable).
- **Likely causes.** A reader or format change without a matching fixture change.
- **Diagnose.** `.venv/bin/python -m pytest -q tests/test_health_inputs.py`.
- **Safe retry.** Re-run the health check.
- **Fallback.** Treat `training.*`, `registry.*`, `forecast.*` and `portfolio.*` as unverified until it passes.

## Data quality and vendor basis (local monitor)

<a id="data-freshness"></a>
### Data freshness (`data.freshness`)

- **What it means.** Sessions between each fund's newest close in the F1 store (`--fund-store`) and the latest permitted session (capped at the lockbox). Healthy at 0 or 1; degraded beyond.
- **Likely causes.** A batch requested an earlier end; the vendor served fewer sessions.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('STORE'); print(s.prices.groupby('fund_id')['session_date'].agg(['min','max','size']))"`.
- **Safe retry.** Re-run the local monitor. Fetching data is an owner action.
- **Fallback.** Pages show their stale states; nothing is filled or decided on the missing sessions.

<a id="data-coverage"></a>
### Session coverage (`data.coverage`)

- **What it means.** Requested fund-sessions with a close, over all requested. Healthy at 99.5 % or more; degraded below; failed below 95 %.
- **Likely causes.** Vendor gaps; a later inception date; a mapping error.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.data.funds.reader import read_store; s = read_store('STORE'); print(s.missing_sessions.groupby('fund_id').size())"`.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** Incomplete histories stay labelled; no gap is filled.

<a id="data-missing-sessions"></a>
### Missing sessions (`data.missing_sessions`)

- **What it means.** Requested fund-sessions without a close. Healthy at 0; degraded with any; failed with 5 or more consecutive for one fund.
- **Likely causes.** As for coverage; a feed that stopped.
- **Diagnose.** As for coverage, listing `s.missing_sessions` in full.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** As for coverage.

<a id="data-vendor-discrepancy"></a>
### Source discrepancies (`data.vendor_discrepancy`)

- **What it means.** The largest relative difference between two sources' closes for one fund, session and basis. Healthy at 0.1 % or less; insufficient evidence with one source.
- **Likely causes.** Consolidated versus primary closes; an applied corporate action; a mapping error.
- **Diagnose.** The detail lists the largest differences; compare the two sources' rows in the store.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** Never pick a source silently; which source a page uses is an owner decision.

<a id="data-anchors"></a>
### Distribution anchors file (`data.anchors`)

- **What it means.** The owner's hand-filled anchors file in the prospective scope: its sha256, rows by status, anchors per fund. Failed when the loader refuses it (the reason names a row and column, never a value); degraded when a fund has no anchor; insufficient evidence with no file (nothing can pass the spot check).
- **Likely causes.** A typing error (decimals, date, basis); a fund not yet filled in.
- **Diagnose.** `.venv/bin/python scripts/check_anchors.py` (counts and sha256 only).
- **Safe retry.** Fix the file by hand, then re-run the local monitor.
- **Fallback.** No batch is stored for a fund whose distributions cannot be spot-checked.

<a id="data-store-batches"></a>
### Prospective batches and spot checks (`data.store_batches`)

- **What it means.** The batches the prospective store's reader uses for the window, each with its retrieval time, request and the anchors file its spot check used. Degraded when a batch records no anchors file (`spot_check_not_recorded`) or a later batch serves a different value for the same fund and date (`vendor_revisions`, values never shown); unavailable when the reader refuses the store; insufficient evidence before anything is stored or with no fund named (`no_assets`).
- **Likely causes.** A vendor revision; an edited or foreign file in the store (`store_refused`).
- **Diagnose.** The detail names the batches and the revised fund-dates; `ls -la` the store root (`--prospective-store`).
- **Safe retry.** Re-run the local monitor. Nothing is ever deleted or repaired in the store.
- **Fallback.** The reader uses the newest batch and lists every disagreement; a refused store is not read at all.

<a id="data-window-coverage"></a>
### Prospective window coverage (`data.window_coverage`)

- **What it means.** Closes present over fund-sessions expected from the window start to the last complete session (18:00 New York, rules calendar), per fund, with each missing close's reason (`no_batch`, `missing_session`). Healthy when complete; degraded otherwise.
- **Likely causes.** The latest sessions are not collected yet; a vendor gap.
- **Diagnose.** The detail lists the missing dates per fund.
- **Safe retry.** Re-run after the owner's next approved collection.
- **Fallback.** A forecast for a fund with a missing close is refused, never filled.

## Training and evaluation runs (local monitor)

<a id="training-health"></a>
### Training health (`training.health`)

- **What it means.** The newest RL training-health file (`reports/health_inputs/rl_training_health/`): failed when the run or a producer check failed; degraded on a warning, an unapproved transition or unreadable files; insufficient evidence while running, with an unrun listed check or no file; healthy when all eight listed checks passed.
- **Likely causes.** Diverging losses, entropy collapse, non-finite values, seed spread; or the transition is not approved yet.
- **Diagnose.** `ls -lt reports/health_inputs/rl_training_health/`; read the failing producer checks in the file named in the detail.
- **Safe retry.** Re-run the local monitor. Retraining is a separate, deliberate run.
- **Fallback.** Do not use the run's policy (see "When to disable a policy").

<a id="training-ml-runs"></a>
### Latest ML experiment run (`training.ml_runs`)

- **What it means.** The newest attempt in the experiment runner's ledger (`reports/ml/ledger.jsonl`): healthy when completed from a clean tree; degraded from a dirty tree; failed when it failed; insufficient evidence when the attempt does not record its tree state (`tree_state_unknown`). In the local monitor it is healthy only when the registry shows every version the run registered as non-synthetic: synthetic versions (`synthetic_input`) or versions the registry cannot show (`data_class_unknown`) are insufficient evidence.
- **Likely causes.** A config or data error (the detail quotes the error); uncommitted changes when it ran; the registry missing or refused (`registry.entries` says which).
- **Diagnose.** `tail -3 reports/ml/ledger.jsonl`, then re-run the experiment's config locally.
- **Safe retry.** Re-run the experiment from a clean tree: a new run is a new, counted variant.
- **Fallback.** Nothing is registered from a failed run; earlier versions keep their stages.

<a id="forecast-coverage"></a>
### Forecast interval coverage (`forecast.coverage`)

- **What it means.** From the newest registry export: the largest gap between observed and stated interval coverage (n of 50 or more). Healthy at 0.05 or less; degraded to 0.10; failed beyond. A screening band, not a test. Point forecasts do not apply.
- **Likely causes.** Mis-specified uncertainty; a regime change; overlapping horizons.
- **Diagnose.** `ls -lt reports/health_inputs/model_registry/`; read the rows named in the detail.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** Intervals are not shown as calibrated; nothing is promoted.

<a id="forecast-calibration"></a>
### Forecast calibration (`forecast.calibration`)

- **What it means.** The producer's own calibration status per interval evaluation: failed on any fail, degraded on a warning, insufficient evidence when none ran.
- **Likely causes.** As for coverage.
- **Diagnose.** As for coverage.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** As for coverage.

## Models, versions and evidence stage (local monitor)

<a id="registry-entries"></a>
### Registry entries and artifacts (`registry.entries`)

- **What it means.** Every registry entry, the activation chain and every entry's artifacts verify through `open_registry` (`--registry`, default `reports/ml/registry`). Failed when the reader refuses the registry or an artifact no longer matches.
- **Likely causes.** A file under `reports/ml/` edited, moved or deleted; another artifacts root.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); print([str(e.ref) for e in r.entries() if not r.artifacts_verified(e)])"`.
- **Safe retry.** Re-run with the same `--registry` and `--ml-artifacts`. A re-run experiment registers a new version; it never repairs an old one.
- **Fallback.** Unverified versions drop to the floor stage; nothing compares against them.

<a id="registry-stages"></a>
### Computed stages (`registry.stages`)

- **What it means.** Each version's stage, computed from verified evidence, with the reasons that block its next stage (one line per version in the detail). The value counts versions above `engineering_only`; a drop raises a deterioration alert. Upper stages stay closed until the owner approves their rules.
- **Likely causes.** A drop: an altered or deleted artifact; evidence that no longer re-derives.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); print(r.assessment('MODEL@vN').frame().to_string())"`.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** A version is treated at its computed stage; nothing is promoted or demoted by hand.

<a id="registry-selection-history"></a>
### Reference-version history (`registry.selection_history`)

- **What it means.** Activation and rollback events per research scope; failed when an active version no longer verifies. "Active" is research bookkeeping, never a deployment.
- **Likely causes.** As for `registry.entries`, for the active version.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); [print(e) for e in r.events()]"`.
- **Safe retry.** Re-run. Activating or rolling back is a deliberate decision with a stated reason.
- **Fallback.** No comparison against an unverified reference.

<a id="registry-experiments"></a>
### Experiment counts (`registry.experiments`)

- **What it means.** Every version records its multiple-testing counts; failed without a candidate count; degraded when prior variants are below the versions registered earlier.
- **Likely causes.** Variants tried outside the runner; a reset counter; a renamed scope.
- **Diagnose.** `.venv/bin/python -c "from stock_agent.registry import open_registry; r = open_registry('ROOT'); [print(e.ref, e.evidence['multiple_testing']) for e in r.entries()]"`.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** Report results with the larger, conservative count.

## Prospective forecast ledger (local monitor)

<a id="forecast-ledger"></a>
### Forecast ledger integrity (`forecast.ledger`)

- **What it means.** The hash chain of `data/raw/prospective_forecasts/ledger.jsonl` (`--ledger`) verifies with the forecast job's own reader. The detail names the entry counts, the head hash and the as-of sessions. Failed on an edited, inserted, removed or reordered line, an incomplete last line or a symlink, and when the file is missing although the kept head shows it held records (`ledger_deleted`); unavailable when unreadable; insufficient evidence before the first record. Fewer entries than the previous summary raise a deterioration alert.
- **Likely causes.** A hand edit; a copy that cut the file; a crash mid-write.
- **Diagnose.** `.venv/bin/python scripts/forecast_prospective.py report`; the error names the first bad line.
- **Safe retry.** Re-run once (a write may have been in progress). Never edit the ledger to repair it; restoring it from a backup is the owner's step.
- **Fallback.** No record from a refused ledger is shown or scored; record nothing new until it verifies.

<a id="forecast-ledger-backup"></a>
### Kept copy of the ledger head (`forecast.ledger_backup`)

- **What it means.** The head hash the owner keeps outside the ledger (`--ledger-head-backup`, first line of the file) against the ledger. Healthy when it is the head (the genesis hash, all zeros, for an empty ledger); degraded when an earlier entry (copy the new head); failed when the ledger no longer holds it (lines removed, the file rewritten, or the file deleted); insufficient evidence when not configured.
- **Likely causes.** The kept copy was not refreshed after the last record; the ledger lost its tail or was deleted or moved.
- **Diagnose.** `.venv/bin/python scripts/forecast_prospective.py report --expect-head <kept hash>`.
- **Safe retry.** After copying the head printed by the job's last append.
- **Fallback.** As for `forecast.ledger` when failed.

<a id="forecast-ledger-sessions"></a>
### Sessions with a forecast record (`forecast.ledger_sessions`)

- **What it means.** Rules-calendar sessions from the first recorded as-of session to the last one whose recording deadline (09:30 New York, next weekday) has passed, without a forecast or refusal record. No cadence is pre-registered, so every session is expected (operator default): gaps are degraded and listed.
- **Likely causes.** A missed scheduled run; the job refused to record (a refusal counts as a record).
- **Diagnose.** The detail lists the sessions without a record.
- **Safe retry.** None for a past session: a forecast is never backdated.
- **Fallback.** The gap stays visible in the record; it is never filled.

## Policy limits and fallbacks (local monitor)

Backtest summaries are read from `reports/health_inputs/backtest_summary/`,
the newest per `policy` (so every policy a run writes is judged), and RL
evaluation reports from `models/rl/<run_id>/evaluation_*.json`
(`--rl-runs`). Each check takes the worst policy's state; its detail keeps
one line per policy with that policy's own state. Backtests are simulated,
never a track record. Measures come from the approved T5a formulas only and
are never recomputed here.

<a id="data-stock-lab-freshness"></a>
### Stock Lab data freshness (`data.stock_lab_freshness`)

- **What it means.** Whether every watched stock in the Stock Lab has the session the daily job expected (the last NYSE session whose close is at least 20:00 New York time old). Degraded (`stale_prices`) when any stock is behind: its last valid forecast is kept and shown as stale, never replaced by an invented one. Insufficient evidence without a watchlist, before the job has run, or while the price update is not authorized yet (`not_authorized`).
- **Likely causes.** The price update is not authorized yet (decisions A1 and A2); a provider outage; a ticker that stopped trading; the job ran before the close was published.
- **Diagnose.** The detail lists each stock's last stored session and how many sessions it is behind; the job run in the private store says what was skipped and why.
- **Safe retry.** Run `scripts/stock_lab_daily.py` again after the availability time; it is idempotent (scores and runs are write-once).
- **Fallback.** Leave the stale forecasts labelled stale; never edit a stored forecast.

<a id="forecast-stock-lab-job"></a>
### Stock Lab daily update (`forecast.stock_lab_job`)

- **What it means.** The last daily-job run's status as recorded by the job: ok is healthy, partial or skipped degraded, failed failed. Until the owner records decisions A1, A2 and A4, a real run records itself as skipped and not yet authorized, with the reason. That reads as not activated (`not_authorized`, insufficient evidence), so the rest of the private Model Health is not shown as degraded. Failed (`job_overdue`) when the monitor runs more than 6 hours past the run the job said was next, authorized or not: the scheduled job may have stopped, and its last file would otherwise stay green.
- **Likely causes.** Not activated: approvals pending. Skipped (degraded): a step the job could not run. Failed: a provider error, an unreadable store, a ledger that does not verify.
- **Diagnose.** The detail gives the last success and the next expected run; the private store's job runs keep every run's reason.
- **Safe retry.** Re-run the job after fixing the cause.
- **Fallback.** The last valid forecasts stay as they are, labelled stale.

<a id="forecast-stock-lab-scores"></a>
### Stock Lab rolling forecast scores (`forecast.stock_lab_scores`)

- **What it means.** The rolling ranked probability score of scored daily targets beside the recent-weighted baseline's development-test score. Insufficient evidence below 250 scored targets; degraded (`worse_than_baseline`) when worse than the baseline. This is tracking, never evidence of skill: the forecaster is experimental and unvalidated.
- **Likely causes.** Too few sessions since forecasting began; a regime the development data did not contain; drifted inputs.
- **Diagnose.** Compare with `forecast.stock_lab_drift`; the Model diagnostics tab shows the development tests.
- **Safe retry.** Re-run the job; scores accrue as sessions close.
- **Fallback.** Read the baseline's probabilities instead; the model is never promoted on these scores.

<a id="forecast-stock-lab-drift"></a>
### Stock Lab input drift (`forecast.stock_lab_drift`)

- **What it means.** The largest population stability index among a few model inputs, recent sessions against the development distribution. Degraded (`input_drift`) above 0.25, conventionally a material shift.
- **Likely causes.** A change of volatility regime; a corporate event; a data problem (check freshness and revisions first).
- **Diagnose.** The detail names the feature; look at the stock's History tab.
- **Safe retry.** Nothing to retry: drift is measured, not fixed.
- **Fallback.** Treat the probabilities with extra caution while drift persists; never refit on sealed data to make it pass.

<a id="forecast-stock-lab-ledger"></a>
### Stock Lab forecast ledger (`forecast.stock_lab_ledger`)

- **What it means.** Whether the Stock Lab ledger's hash chain verifies: every forecast, score and job run has one entry, each naming the previous entry's hash and matching the record it covers. The check also compares the chain with the head hash the daily job keeps outside the database (`--ledger-head-backup`; by default `stock_lab_ledger_head` beside the store, and a copy kept elsewhere is stronger).
  - Failed (`ledger_broken`) at the first entry that does not verify.
  - Failed (`head_not_in_ledger`) when the chain no longer holds the kept hash. Entries were removed from the end, or the store was replaced, which the chain alone cannot show.
  - Failed (`ledger_deleted`) when the store is gone but the kept hash shows it held entries.
  - Unavailable (`head_unusable`) when the kept file is not a 64-hex hash.
  - Insufficient evidence when no head is kept yet (`head_started` on the first run, `head_not_kept` without a file).
  - A mismatch never overwrites the kept file, so the failure stays until someone looks.
- **Likely causes.** The private store was edited outside the Lab; a copy was restored from an older backup; the kept head file was edited or replaced.
- **Diagnose.** The detail names the first failing entry; compare the kept hash with a backup of the store.
- **Safe retry.** None: a broken chain is never repaired in place.
- **Fallback.** Stop recording forecasts, keep the store for inspection, and start a new store only with the owner's decision.

<a id="portfolio-turnover"></a>
### Portfolio turnover (`portfolio.turnover`)

- **What it means.** Stored turnover against the backtest's declared `max_turnover`; above it fails with `constraint_breached`.
- **Likely causes.** A rebalance rule trading more than designed; noisy weights.
- **Diagnose.** Read the backtest record named in the evidence; compare its rebalances with its rule.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** The construction is not eligible for promotion until reviewed.

<a id="portfolio-costs"></a>
### Trading costs (`portfolio.costs`)

- **What it means.** Stored costs against the declared `max_cost_bps`.
- **Likely causes.** Higher turnover; a cost-assumption change.
- **Diagnose.** As for turnover.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** As for turnover.

<a id="portfolio-concentration"></a>
### Concentration (`portfolio.concentration`)

- **What it means.** The stored largest weight against the declared `max_weight` cap.
- **Likely causes.** A cap not enforced after drift; a normalization error.
- **Diagnose.** As for turnover, on the weights in the backtest's record.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** As for turnover.

<a id="portfolio-drawdown"></a>
### Drawdown (`portfolio.drawdown`)

- **What it means.** The stored maximum drawdown against the declared `max_drawdown`.
- **Likely causes.** A stress period; more risk than designed.
- **Diagnose.** As for turnover.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** As for turnover.

<a id="portfolio-violations"></a>
### Policy constraint violations (`portfolio.violations`)

- **What it means.** Decisions that broke a hard constraint: the backtest's `constraint_violations` plus the RL report's `invalid_targets`. Any is failed (`constraint_breached`); none is healthy; insufficient evidence when nothing reports them.
- **Likely causes.** A policy producing targets outside the simplex or the caps; a constraint not applied.
- **Diagnose.** The detail names the backtest and the methods with invalid targets.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** Disable the policy until the cause is fixed (see "When to disable a policy").

<a id="portfolio-fallbacks"></a>
### Fallback decisions (`portfolio.fallbacks`)

- **What it means.** Decisions the registered fallback took instead of the policy: the backtest's `fallback_events` plus the guarded policies' `fallback_decisions`, with their reasons. Any is degraded; none is healthy.
- **Likely causes.** `stale_inputs`, `out_of_distribution`, `model_unhealthy`, `invalid_action`.
- **Diagnose.** The detail gives the share per guarded policy and the reasons.
- **Safe retry.** Re-run the local monitor.
- **Fallback.** The fallback already decided; disable the policy if `model_unhealthy` or a large share recurs.

<a id="portfolio-diagnostics"></a>
### Risk, turnover and drawdown diagnostics (`portfolio.diagnostics`)

- **What it means.** Which of risk (`volatility`), turnover and drawdown (`max_drawdown`) the newest backtest summary of each policy reports. Healthy when every policy reports all three; insufficient evidence until the T5a engine writes summaries, or while a measure is not reported (the risk formula awaits approval).
- **Likely causes.** No backtest summary yet; a formula not yet approved.
- **Diagnose.** `ls -lt reports/health_inputs/backtest_summary/`.
- **Safe retry.** Re-run the local monitor after the next backtest.
- **Fallback.** No risk statement is shown without its approved measure.
