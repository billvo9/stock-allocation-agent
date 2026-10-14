# Model registry

Status (2026-10-14): implemented on `feature/ml-registry-forecast-contracts`; merge pending.

Code: `src/stock_agent/registry/` (`entries.py`, `evidence.py`, `stages.py`, `store.py`).
Tests: `tests/test_registry.py`.

## Purpose

The registry records every evaluated model version so that versions can be compared, a research scope's reference version can be rolled back, and anyone can see how far a model has been validated, and why it has gone no further.

It is a local research registry under the ignored `reports/` tree. It is **not** the Git-tracked G1 registry of pre-registrations and holdout use (OD12, pending). The stages above `development_evaluated` depend on G1, which is one reason they are fail-closed today.

## Layout

```text
reports/ml/registry/
  entries/<model_id>/v0001.json   one immutable entry per version (exclusive create)
  events.jsonl                    activation and rollback events, hash-chained
reports/ml/                       artifacts root: runs/<run_id>/, forecasts/<artifact_id>/
```

- **Entries are write-once.** A version number is assigned once (`max + 1`) and its file is created with exclusive create. An entry is never edited in place.
- **Integrity.** Each entry carries `config_sha256` and `entry_sha256` (canonical JSON of every other field). Loading an edited entry raises.
- **Events.** Each line holds its sequence number, the previous line's SHA-256 and its own SHA-256, so an edited, dropped or reordered line is refused.

## Entry fields

| Field | Meaning |
|---|---|
| `model_id`, `version` | Path-safe id (`[a-z0-9][a-z0-9_.-]*`) and a positive integer |
| `research_scope` | The comparison and rollback unit, e.g. `equity_xs:synthetic:h20` |
| `parent` | The version this one descends from (registered, and earlier when it is the same model) |
| `created_at` | From an injected clock, UTC |
| `config`, `config_sha256` | Model specification and experiment settings |
| `training_data` | `data_class` (`synthetic`, `engineering_sandbox`, `research_panel`) and fingerprints of the labeled frame, fold table and universe |
| `metrics` | Summary metrics (numbers or null), descriptive |
| `artifacts` | `{kind, path, sha256}`: the experiment run and forecast artifacts it was evaluated from, relative to the artifacts root |
| `evidence` | Facts derived from the run record by `derive_evidence` (below) |
| `code` | Git SHA and dirty flag |

There is **no stage field**. An entry that contains one is refused.

## Stages are computed, never set

`assess(entry, artifacts_verified, rederived_evidence)` returns the highest stage whose criteria, and every lower stage's criteria, are all met:

1. **Artifacts.** The store re-verifies every artifact against its SHA-256. For a directory, the hash is that of its `manifest.json`, and every file the manifest lists must match.
2. **Evidence.** The store re-derives the evidence from the verified run record (`derive_evidence(record, model)`). The stored evidence must equal it, so evidence that does not follow from a verified run never counts.
3. **Criteria.** The stage criteria are then evaluated.

Registration refuses an entry whose artifacts do not verify, or whose evidence does not follow from its run.

| Stage | Rule status | Reachable today |
|---|---|---|
| `engineering_only` | The floor: the entry loads and its hashes verify | Yes |
| `development_evaluated` | Implemented from documented rules (below) | Only with a `research_panel` run, which no data source produces yet |
| `historically_checked` | **Proposal; not approved** | No (fail closed) |
| `prospectively_validated` | **Proposal; not approved** | No (fail closed) |
| `approved_for_scope` | **Proposal; not approved** | No (fail closed) |

**Fail closed.** `stages.RULES` marks the three upper rules `approved=False`, so their stages are unattainable whatever an entry records. Even if a rule were marked approved without criteria being implemented, the stage stays unattainable (`criteria_not_implemented`; tested). Approving a rule is an owner decision; it changes `RULES` and adds the criteria in the same reviewed change.

### `development_evaluated` criteria

These criteria come from the product direction (`docs/product/vision_and_prd.md`, section 6, "Evidence required for every rung above L0") and the T2/T4 protocol. They add no new evidence standard, but they are provisional like the PRD rules they come from. Owner review is requested.

`prior_variants_in_scope`, in the evidence's multiple-testing block, counts every earlier **registration** in the run's research scopes, not distinct variant ids: a rerun of one variant on new data or with a new seed is another trial. Counts cover one output root and exclude failed runs; the health export reports failed runs from the ledger.

| Criterion | Reason code when unmet |
|---|---|
| Artifacts verify | `artifacts_unverified` |
| Evidence re-derives from the verified run | `evidence_not_reproducible` |
| Role is `candidate` | `not_candidate` |
| Data class `research_panel` (synthetic and sandbox runs verify the pipeline only) | `data_class_engineering_only` |
| Universe point-in-time and survivorship-free | `universe_not_point_in_time` |
| Development mode; no holdout values; every data and label date before the lockbox | `lockbox_not_proven_closed` |
| Every leakage check passed (none undetermined) | `leakage_checks` |
| Every instrument check passed (none undetermined) | `instrument_checks` |
| Decision method fold-block t | `decision_method` |
| At least one registered null, and at least 199 permutation draws | `nulls_insufficient` |
| Multiple-testing count recorded | `multiple_testing_not_recorded` |

Fund-level claims have their own evidence rules, still to be set "with the first fund-level forecast ticket" (PRD section 6). Until then, the universe criterion above applies to every scope, which keeps fund scopes fail-closed too.

### Proposed: `historically_checked`

For owner decision. This corresponds to stage S1 and to T6.

1. `development_evaluated`.
2. OD12 approved, and a G1 pre-registration committed in Git **before** the holdout run. It names the claim, rung, horizon, universe, MDE, test family, α and the collapse rule.
3. Exactly one T6 run for the lineage, recorded in a Git-tracked holdout-use record, from a clean tree at the registered code SHA and config hash.
4. The pre-registered collapse rule did not fire.
5. The entry carries the `non_confirmatory` and `designed_after_holdout_start` flags (PRD section 6, item 4).

### Proposed: `prospectively_validated`

For owner decision. This corresponds to a model claim at stage S2 to S3.

1. `development_evaluated`.
2. OD16 activated: the prospective pipeline is live, with H_end and the transition rules approved.
3. A registered confirmatory plan (one-sided; non-overlapping blocks; α-spending at fixed look dates) committed before the burn-in starts.
4. The test passes at a look date, on post-burn-in dates only.
5. No change in the lineage other than registered refits.

**Open question.** The PRD says S1 is not a gate for S2. The code ladder requires every lower stage, so `prospectively_validated` would need `historically_checked`. If the owner keeps the PRD rule, this stage should require `development_evaluated` only. That is a decision to make with the approval.

### Proposed: `approved_for_scope`

For owner decision.

1. `prospectively_validated` in the same research scope.
2. An owner approval event, committed in Git, naming the entry, the scope, the claim level and a review date.
3. The S3 operational and risk-control requirements of PRD section 15, for whatever the approval allows.

## Activation and rollback

- **Active version.** `activate(ref, reason)` makes a version the active one of its research scope. `rollback(scope, to, reason)` re-activates a version that was active earlier in that scope.
- **History.** History is append-only: `activation_history(scope)` lists every change with its reason, time and the version it replaced.
- **Not a deployment.** "Active" is a research bookkeeping pointer, the reference version of a scope. It never changes a stage, and it is not a deployment.

## Read API (Model Health Center)

```python
from stock_agent.registry import open_registry

reader = open_registry("reports/ml/registry")  # artifacts root defaults to reports/ml
reader.summary()                 # one row per version: ref, model_id, version, research_scope,
                                 #   parent, created_at, data_class, run_id, stage, next_stage,
                                 #   n_blockers, active
reader.metrics_frame()           # ref, metric, value (long)
reader.assessment("m@v2")        # .stage, .criteria, .blockers, .frame() (one row per criterion)
reader.active(scope); reader.activation_history(scope)
reader.compare("m@v1", "m@v2")   # .table (metric, a, b, b_minus_a), .same_training_data, .same_folds
reader.entries(scope); reader.entry("m@v2"); reader.refs()
```

- The reader never writes (tested).
- Every call re-verifies what it reads. A stage is computed per call, so it can only drop when an artifact goes missing or is altered.
- `compare` is descriptive. A credible difference needs paired predictions on identical rows, which the experiment runner reports with uncertainty.

## Health export

The Model Health Center reads a snapshot of the registry in the agreed input format v1.0 (artifact `model_registry`). The registry stays the source of truth: the export is derived through the read API, so every stage in it is computed from re-verified evidence at export time.

**Writing it.**
- `scripts/run_experiment.py` refreshes the export after every run. Pass `--no-health-export` to skip it.
- `scripts/export_registry_health.py` writes one on demand, for example after an activation or a rollback.
- Code: `registry/export.py`. Tests: `tests/test_registry_export.py`, plus integration tests in the runner and ETF test files.

**File.**
- Location: `reports/health_inputs/model_registry/<run_id>.json` (ignored by Git).
- Encoding: strict JSON (no NaN or Infinity), written once.
- `run_id`: the UTC time plus a hash of the content.

| Field | Contents |
|---|---|
| envelope | `artifact`, `schema_version` "1.0", `generated_at` (UTC, microseconds, `Z`), `code_sha` (40-hex or null), `synthetic` (true only if every entry is synthetic), `run_id` |
| `models` | The latest version of each model id **in each research scope**, plus each scope's active version: `model_id`, `version`, `stage` (computed), `stage_basis`, `registered_at`. Extra: `research_scope`, `data_class`, `next_stage`, `blockers` (reason codes) |
| `selection_events` | Every activation and rollback: `event_id`, `decided_at`, `candidates` (versions registered in the scope by then), `selected`. Extra: `event`, `research_scope`, `previous`, `reason` |
| `experiments` | `variants_tried` (registered versions: every registration is a trial), `runs` (distinct experiment runs that registered something). Extra: `distinct_variants`; `ledger_attempts` and `failed_runs` from the output root's ledger (null without one). Every count covers one output root |
| `forecast_evaluation` | One row per forecast artifact of an exported model, and one per measured interval level for interval artifacts: `model_id`, `version`, `horizon`, `forecast_kind`, `interval_level`, `observed_coverage`, `n`, `calibration_status`, `calibration_detail` |

**Evidence ids in `stage_basis`.** Each id is a record the computed stage rests on:
- `entry:<ref>:<entry_sha256[:16]>`, the hash-verified entry;
- `artifact:<kind>:<sha256[:16]>` for every artifact that still verifies;
- `run:<run_id>` when the evidence re-derives from that run's verified record.

A tampered or missing artifact therefore shortens the basis and lowers the stage in the next export.

**Forecast rows.**
- **Point forecasts** carry null interval fields, `n` (rows in the artifact), `not_run` and "not applicable: point forecast".
- **Interval artifacts** report every level the run measured. For the ETF track that is 50%, 80% and 90%, pooled over assets:
  - `observed_coverage` and the status use **one estimand**: the fold-block estimate (the mean of the outer folds' coverage shares) and its fold-block t test of "coverage = nominal";
  - the detail names the worst asset when the run reports per-asset coverage, because a pooled status can hide one asset's gap.

**Calibration status.** This is a producer-owned monitoring status: not an evidence rule, and no stage uses it. With gap = |observed − nominal| and p the fold-block two-sided p-value:

| Status | Rule |
|---|---|
| `fail` | p < 0.01 **and** gap > 0.05 |
| `warn` | p < 0.05 **or** gap > 0.05 |
| `pass` | otherwise |
| `not_run` | A point forecast; no verified coverage measurement; or no test is possible (p unavailable, for example fewer than 3 folds) and gap ≤ 0.05. Without a test nothing passes, though a gap beyond 0.05 still warns |

Gaps are compared with a 1e-9 tolerance: 0.75 against 0.80 is a gap of exactly 0.05, which does not exceed the band.

**Example (measured, synthetic, the ETF runs in `ml_etf_track.md`).**
- The planted GARCH panel's return intervals (fold-block estimates) warn at all three levels: 50% → 45.7%, 80% → 75.3% (p 0.013), 90% → 85.8% (p 0.005).
  - The detail names the worst asset: at 80% that is EQ_US, at 71.6% (p 0.057). A pooled status alone would hide it.
  - The same export lists `ledger_attempts` 4 and `failed_runs` 2: the two first ETF attempts failed on a fold-start configuration error before registering anything.
  - The under-coverage is expected from a plug-in interval that ignores forecast error (`ml_etf_track.md`, Results).
- The null panel's intervals pass (80% → 78.3%, p 0.30).

## Limitations

- **Integrity is not provenance.** Anyone with write access to `reports/` can forge a run and its hashes. The registry detects edits and inconsistencies, not a deliberate forgery. The same caveat is recorded for publication P1-1a.
- **`reports/` is ignored by Git and can be deleted.** Durable research records belong in G1 (OD12).
- **Single writer.** Version assignment uses exclusive create, so a race fails loudly instead of overwriting, but there is no lock.
