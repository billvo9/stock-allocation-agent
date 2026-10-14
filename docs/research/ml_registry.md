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

## Limitations

- **Integrity is not provenance.** Anyone with write access to `reports/` can forge a run and its hashes. The registry detects edits and inconsistencies, not a deliberate forgery. The same caveat is recorded for publication P1-1a.
- **`reports/` is ignored by Git and can be deleted.** Durable research records belong in G1 (OD12).
- **Single writer.** Version assignment uses exclusive create, so a race fails loudly instead of overwriting, but there is no lock.
