# Model Health Center

Whether the code, the data contracts, the data and the models can be trusted
today, as machine-readable summaries and a page that reads them. The
operator guide for each check is the [runbook](runbook.md).

Code: `src/stock_agent/health/`. Scripts: `scripts/health_check.py` (build a
summary), `scripts/run_model_health.py` (view it). Workflow:
`.github/workflows/health.yml`. Tests: `tests/test_health_*.py`.

## Principles

- **Never healthy without a measurement.** A check is healthy only with a
  measured value and a measurement time; a summary without checks is
  `insufficient_evidence`.
- **Informational only.** Alerts are messages. Nothing in the health center
  trades, retrains, promotes, deletes or changes configuration, and the
  summary format itself refuses any other `automatic_action` than `none`.
- **Read-only towards what it monitors.** Monitors flag; they never modify,
  fill, drop or adjust data (AGENTS.md, anomaly-detection hierarchy).
- **Public and private stay apart.** GitHub Actions sees only repository
  code and synthetic fixtures. Private stores, licensed data and local
  models are read only by the local monitor on the owner's Mac, whose
  summaries never leave it. CI cannot inspect the owner's Mac and nothing
  here pretends otherwise.
- **No network.** Nothing in `stock_agent.health` makes a request. The CI
  run list is a file the owner writes with the GitHub CLI.
- **Point-in-time.** Data monitors judge a store as of the run time and
  count only batches retrieved by then; freshness is measured against the
  lockbox cap, never against today.

## States and precedence

Worst first; the overall state is the first state any check has:

| State | Meaning |
|---|---|
| `failed` | Measured, and a hard limit or contract is broken |
| `unavailable` | A measurement that should exist could not be taken (unreadable input, a tool that did not run). It can hide a failure, so it outranks `degraded` |
| `degraded` | Measured, outside a warning threshold, still usable |
| `insufficient_evidence` | Nothing to judge yet: the input does not exist (another track has not delivered), or there are too few observations |
| `healthy` | Measured, within its threshold |

A check that does not apply to what was measured (interval coverage of a
point forecast) is not given a state: it is listed under `not_applicable`,
shown on the page, and never counts towards the overall state.

`insufficient_evidence` ranks above `healthy` on purpose: while any area has
no evidence, the overall state cannot be healthy. The page says so in plain
words ("nothing measured is wrong, but some areas have no evidence yet") and
lists what it is waiting for.

## The summary (schema 1.0)

One strict-JSON file (no NaN, no duplicate keys), refused as a whole by the
reader (`stock_agent.health.model.read_summary`) on any defect:

```text
format          "stock_agent.health_summary"
schema_version  "1.0"
generated_at    UTC, microseconds, "Z"
code_sha        40-hex or null        the producer's claim, not authenticated
code_dirty      true/false or null    any change or untracked file except .venv
scope           "synthetic" (public-safe) | "private" (owner's machine)
run_context     {"environment": "local"} or allowlisted GitHub Actions metadata
precedence      the five states, worst first
overall_state   recomputed by the reader and compared
state_counts    recomputed by the reader and compared
alert_policy    "informational_only"
checks          [check, ...]
not_applicable  [{check_id, area, title, reason, detail}, ...]  no state; never counted
alerts          [alert, ...]
```

Each check: `check_id` (`<area>.<name>`), `area`, `title`, `state`, `reason`
(machine-readable; required unless healthy), `measured_at`, `value`, `unit`,
`threshold`, `better` and `tolerance` (for deterioration), `summary` (plain
words), `detail` (technical), `runbook` (an anchor in the runbook) and
`evidence` (a path relative to the summary, an absolute path in private
summaries, or a `fixture:` / `module:` reference).

Each alert: `alert_id` (`<kind>:<check_id>`), `kind` (`failure`,
`constraint_violation`, `unavailable_data`, `degraded`, `deterioration`),
`check_id`, `state`, `message`, `raised_at`, `automatic_action: "none"`.

`run_context` in GitHub Actions copies only `GITHUB_WORKFLOW`,
`GITHUB_RUN_ID`, `GITHUB_RUN_NUMBER`, `GITHUB_RUN_ATTEMPT`,
`GITHUB_EVENT_NAME`, `GITHUB_REF_NAME`, `GITHUB_SHA`, `GITHUB_REPOSITORY` and
`RUNNER_OS`, each checked against a strict pattern, plus the run URL. No
token, secret or other variable is read.

## Checks

| Area | Check | Where | Source |
|---|---|---|---|
| CI and tests | `ci.verify`, `ci.tests` | CI and local | `scripts/verify.py` output (run, or an existing log) |
| | `ci.main_runs` | local | `gh run list` output written by the owner |
| Dependencies | `deps.pins`, `deps.pip_check` | CI and local | `requirements.txt`, the running interpreter, `pip check` |
| Contracts | `contract.publication`, `contract.fund_data`, `contract.generators` | CI and local | probes on synthetic fixtures in a temporary directory |
| Self-tests | `smoke.data_monitors`, `smoke.input_readers` | CI and local | the monitors and readers on planted synthetic defects |
| Data quality | `data.freshness`, `data.coverage`, `data.missing_sessions`, `data.vendor_discrepancy` | local | a private F1 fund store |
| Training | `training.health` | local | the newest training-health input |
| Registry | `registry.entries`, `registry.stages`, `registry.selection_history`, `registry.experiments` | local | the ML track's model registry, through its read API |
| Forecasts | `forecast.coverage`, `forecast.calibration` | local | the newest registry export's forecast evaluations |
| Portfolio | `portfolio.turnover`, `portfolio.costs`, `portfolio.concentration`, `portfolio.drawdown` | local | the newest backtest-summary input |

Thresholds are operator defaults written in the code next to each check and
in the runbook; they are not owner-approved policy and change together. The
monitor self-tests exist because a green data check means nothing unless the
monitor would have turned red on a real defect: in CI there is no real data,
so CI checks the monitors instead of the data.

## Inputs from other tracks (agreed formats v1.0, 2026-10-14)

Agreed with the lead on 2026-10-14. Producers own their check statuses;
health recomputes nothing, never shows healthy without a measurement, and
reports `insufficient_evidence` until an input exists. Worked examples:
`stock_agent.health.fixtures`.

**Location.** Producers write `reports/health_inputs/<artifact>/<run_id>.json`
(ignored by Git). The local monitor reads the newest file per artifact by
`generated_at`; unreadable files beside a readable one are named and make the
check at least degraded. CI reads only the synthetic fixtures.

**Envelope** (one strict-JSON object; extra fields are ignored):

```text
artifact        "rl_training_health" | "model_registry" | "backtest_summary"
schema_version  "1.0"
generated_at    "YYYY-MM-DDTHH:MM:SS.ffffffZ"
code_sha        40-hex or null
synthetic       true | false
run_id          text, equal to the file name
```

**Transition status.** Every artifact that depends on portfolio economics
(training health, backtest summaries) carries `transition_status`:
`"approved"` or `"engineering_only_unapproved_transition"`. The second is
shown as a visible limitation (`degraded`, reason `unapproved_transition`),
never as a failure of the run itself.

- **Training health** (RL track): `algorithm`, `status` (`completed` |
  `running` | `failed`), `seeds`, `scenario_version`, `transition_status`,
  `checks` `[{check_id, status: pass|warn|fail|not_run, value, unit,
  threshold, detail}]`. The listed check ids are `nan_or_inf`,
  `exploding_values`, `approx_kl`, `clip_fraction`, `entropy_collapse`,
  `value_loss`, `reward_trend` and `seed_spread`; a missing listed id counts
  as `not_run`, and extra ids are shown.
- **Model registry export** (ML track): `models` `[{model_id, version, stage,
  stage_basis, registered_at}]`, `selection_events`, `experiments`,
  `forecast_evaluation` `[{model_id, version, horizon, forecast_kind:
  point|interval|probabilistic, interval_level, observed_coverage, n,
  calibration_status, calibration_detail}]`. The registry stays the source of
  truth: the `registry.*` checks read it through
  `stock_agent.registry.open_registry` (docs/research/ml_registry.md), which
  re-verifies entries, events and artifacts and computes stages; health only
  reads the export's forecast evaluations. Point forecasts carry null
  interval fields and `not_run`; they are named "not applicable: point
  forecast" and never count, and an export of point forecasts only lists the
  forecast checks as **not applicable** (the summary's `not_applicable`
  list) rather than giving them a state.
- **Backtest summary** (the T5a engine or the RL comparison harness):
  `evidence_status`, `policy`, `transition_status`, `constraints`
  `{max_turnover, max_cost_bps, max_weight, max_drawdown}`, `measures`
  `{turnover, costs_bps, max_weight, max_drawdown}`, optional `units`.
  Measures come from the approved T5a formulas only; health compares stored
  numbers and never recomputes turnover, cost or drawdown (OD8).

## The daily workflow

`.github/workflows/health.yml` runs at 06:17 UTC every day and on demand
(`workflow_dispatch`), with `permissions: contents: read`, no secrets and
`persist-credentials: false`. It installs the project exactly as `ci.yml`
does, then runs

```bash
.venv/bin/python scripts/health_check.py ci --run-verify --output-dir "$RUNNER_TEMP/model-health"
```

and uploads `health_summary.json` with `evidence/verify.log` and
`evidence/pip_check.txt` as the `model-health` artifact (30-day retention).
The job fails when the overall state is `failed`, so GitHub's own
notification reaches the owner; nothing else happens.

Why run it daily when `ci.yml` runs on every push: most requirements are
unpinned, so a quiet repository can still break when pandas, numpy or
pyarrow release. The daily run on `main` catches that.

Limitations: scheduled workflows run only on the default branch, so the
workflow is active once merged; GitHub disables schedules after 60 days
without repository activity; the CI summary is not compared with the
previous day's (that would need `actions: read` and artifact downloads,
deferred), so deterioration alerts come from the local monitor only.

## The private local monitoring path

`scripts/health_check.py local` runs the CI checks plus every private check
whose input is given, on the owner's Mac:

```bash
.venv/bin/python scripts/health_check.py local --run-verify --ci-runs ~/stock-agent-private/health/ci_runs.json --fund-store PATH_TO_FUND_STORE
```

- Inputs: the model registry (`--registry`, default `reports/ml/registry`)
  and the producers' health inputs (`--health-inputs`, default
  `reports/health_inputs`) are read from their default places; the CI run
  list and the fund store only when given.
- Output (default `~/stock-agent-private/health`): `latest.json` (replaced
  atomically), `history/<stamp>.json` (write-once) and `evidence/<stamp>/`.
  Nothing is ever deleted; pruning old history is the owner's choice.
- It refuses to run inside GitHub Actions, and refuses an output directory
  inside the repository unless Git ignores it (`reports/` is ignored), so a
  private summary cannot be committed by accident. Keep the directory
  private (`chmod 700`).
- Each run compares with the previous `latest.json` and raises
  deterioration alerts.
- `--run-verify` runs the full gates (about three minutes). Without it, or
  `--verify-log`, `ci.verify` and `ci.tests` are `insufficient_evidence`.

### Scheduling it (owner steps; nothing here installs them)

Installing a launchd job changes the owner's system, so it is the owner's
step. Replace `YOU` and the paths, then:

1. Create the private directory and a wrapper script
   `~/stock-agent-private/health/run.sh`:

   ```bash
   mkdir -p ~/stock-agent-private/health && chmod 700 ~/stock-agent-private/health
   ```

   ```zsh
   #!/bin/zsh
   set -u
   cd /Users/YOU/Desktop/OMSA/stock-allocation-agent || exit 2
   gh run list --repo billvo9/stock-allocation-agent --workflow ci.yml --branch main --limit 20 \
     --json databaseId,status,conclusion,headSha,createdAt,url,event \
     > ~/stock-agent-private/health/ci_runs.json
   .venv/bin/python scripts/health_check.py local --run-verify \
     --ci-runs ~/stock-agent-private/health/ci_runs.json \
     --fund-store /Users/YOU/stock-agent-private/funds
   ```

   ```bash
   chmod 700 ~/stock-agent-private/health/run.sh
   ```

2. Save this as `~/Library/LaunchAgents/local.stock-agent.health.plist`:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>Label</key><string>local.stock-agent.health</string>
     <key>ProgramArguments</key>
     <array>
       <string>/bin/zsh</string>
       <string>/Users/YOU/stock-agent-private/health/run.sh</string>
     </array>
     <key>StartCalendarInterval</key>
     <dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
     <key>StandardOutPath</key><string>/Users/YOU/stock-agent-private/health/launchd.out.log</string>
     <key>StandardErrorPath</key><string>/Users/YOU/stock-agent-private/health/launchd.err.log</string>
   </dict>
   </plist>
   ```

3. Check, load, run once, inspect:

   ```bash
   plutil -lint ~/Library/LaunchAgents/local.stock-agent.health.plist
   ```

   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.stock-agent.health.plist
   ```

   ```bash
   launchctl kickstart -k gui/$(id -u)/local.stock-agent.health
   ```

   ```bash
   launchctl print gui/$(id -u)/local.stock-agent.health
   ```

4. To remove it:

   ```bash
   launchctl bootout gui/$(id -u)/local.stock-agent.health
   ```

Notes: launchd runs a missed calendar job when the Mac wakes. The monitor
checks whatever is checked out in that directory; the summary records the
commit and whether the tree had uncommitted changes. `gh` uses the owner's own
login; without it, `ci.main_runs` reports `unavailable` and the rest runs.

## Viewing

```bash
.venv/bin/python scripts/run_model_health.py
```

The page binds to 127.0.0.1 with usage statistics off and has two tabs: the
newest downloaded CI summary (`reports/health/ci/*/health_summary.json`, or
`--ci-summary`) and the local monitor's `latest.json` (or
`--local-summary`). It leads with the state in words and one next action,
then the real test counts, gate results and latest CI run read from the
summary, then each area's checks in plain words; exact values, thresholds,
reason codes, evidence and runbook links are inside each check's expander.
The scope note and the informational-only caveat are always visible. A
missing summary is shown with the exact command that creates it.

![The CI tab: healthy, with the real test count and gates](img/model_health_ci.jpg)

![The local monitor: ten checks healthy, fifteen waiting for evidence](img/model_health_local.jpg)

Both captured on 2026-10-14 from real summaries of this branch (before its
first commit): the CI-mode summary from that tree's own `verify.py` output,
and the local monitor with the `gh run list` of `ci.yml` on `main`, the
default registry and health-inputs locations (nothing written there yet) and
no private store configured.

Integration: the Portfolio Lab's Model Health destination calls
`stock_agent.health.page.render(page.sources_from_environment(os.environ))`
(or passes its own `page.Source` list) once the Lab's navigation is on
`main`.

## Learning Notes

**Problem.** A research system can be green in CI and still wrong: tests
cover code paths, not whether today's data is complete, whether a model's
intervals still cover what they claim, or whether a backtest respected its
own caps. Model health is the operational layer that measures those things
continuously and says plainly what it cannot see.

**Concepts.**

- *Ordinal severity and a worst-of rule.* States form a total order; the
  overall state is the minimum (worst) over checks. This is conservative by
  construction: one failure cannot be averaged away by many passes.
- *Absence of evidence is not evidence of absence.* "Never collected" is a
  distinct state from "measured and fine". Defaulting a missing metric to
  healthy is the most common monitoring failure.
- *Monitors need monitoring.* A check that never fires is indistinguishable
  from one that cannot fire. Planting known defects (the `smoke.*` checks)
  is the monitoring analogue of mutation testing and of fire drills.
- *Screening bands versus tests.* `forecast.coverage` uses a fixed band
  around the stated coverage level to screen for large gaps; it is not a
  hypothesis test (example below). Point forecasts have no interval, so the
  check does not apply to them.

**Assumptions.** Thresholds are operator defaults (for example 99.5 %
session coverage, 0.1 % source discrepancy, 5 consecutive missing sessions);
they encode
judgement, not a derived optimum. Producers' own statuses are trusted as
reported; health checks their consistency, not their statistics.

**Formulas.**

```text
coverage            = fund-sessions with a close / requested fund-sessions
source discrepancy  = (max close - min close) / min close, per fund-session and basis
coverage gap        = |observed interval coverage - stated level|, for n >= 50
freshness lag       = NYSE sessions in (newest close, latest permitted session]
```

**Small example.** A 90 % interval evaluated on n = 240 daily forecasts of
a 20-session return covers 88 %: gap 0.02, healthy. A naive binomial
standard error is sqrt(0.9 × 0.1 / 240) ≈ 0.019, which would make a 0.05 gap
look like a 2.6-sigma event. But consecutive 20-session returns overlap, so
the effective sample is closer to 240 / 20 = 12 independent windows, giving
sqrt(0.09 / 12) ≈ 0.087. With that uncertainty no fixed band is a test; the
band only screens for large, persistent gaps, and the producer's own
dependence-aware calibration check (`forecast.calibration`) is the evidence.

**Leakage risks.** A monitor that judges "as of" a time must not use data
retrieved after it: the data monitors drop batches retrieved after the as-of
time (a test pins this). Freshness against today would invite fetching data
past the lockbox; it is measured against the lockbox cap instead.
Deterioration compares only with an earlier stored summary.

**Ways results could mislead.**

- A healthy CI summary describes synthetic fixtures and repository code,
  not the private data or any model.
- A healthy backtest check means the simulation respected its own declared
  limits, not that the strategy is good, and never that it has a track
  record.
- Thresholds that are too tight cause alert fatigue; too loose, and real
  deterioration hides inside the healthy band (hence the tolerance-based
  deterioration alerts).
- The code SHA is a claim by the producer; a dirty or unknown tree weakens
  every result in the summary.

**How practitioners validate monitoring.** Planted-defect tests for every
monitor (here: clean and planted synthetic stores, a planted cap breach),
mutation probes of contracts (here: a reader that stops refusing edits, a
changed generator), tracking false-alarm and missed-alarm rates once real
history exists, and periodic review of thresholds against incidents.

**What to learn next.** Statistical process control for drift (CUSUM and
EWMA charts, which detect small persistent shifts that fixed thresholds
miss); the population stability index for feature drift; calibration tests
under serial dependence (block bootstrap of coverage indicators); and the
Google SRE material on alerting on symptoms rather than causes.

**Exercise.** Replace the fixed 0.05 coverage band with one that widens with
the effective sample size, `k × sqrt(p(1 − p) / (n / h))` for stated level
`p`, `n` forecasts and horizon `h`, as a pure function next to
`forecast.COVERAGE_DEGRADED`, and write a test that a 20-session horizon with
n = 240 tolerates a wider gap than a 1-session horizon with the same n. Then
decide, and write down in the runbook entry, what `k` means and why a band is
still not a test.
