# Research dashboard over T2 outputs (roadmap ticket T3)

Status: implemented on `feature/ml-diagnostics-dashboard` (2026-10-09).
Code: `src/stock_agent/dashboard/`, launcher `scripts/run_dashboard.py`.

The dashboard makes T2 results understandable and auditable. It is a
presentation layer only.
- It reads saved, versioned T2 run directories: `record.json`,
  `manifest.json`, and the contract tables.
- It never re-runs a model, fits anything, recomputes inference, reads raw
  data, or shows the lockbox.
- An unavailable or degenerate result is shown with its stored `status` and
  `reason`, never dropped or replaced by zero.


## Run it

```bash
.venv/bin/python scripts/run_null_diagnostics.py
.venv/bin/python scripts/run_dashboard.py
```

The launcher reads runs from `reports/runs/` (override with `--root`). It
binds Streamlit to `127.0.0.1`: Streamlit otherwise listens on every
network interface. Usage statistics are off. `.streamlit/config.toml`
repeats these settings for launches from the repository root.


## Guarantees and how they are enforced

| Guarantee | Enforcement | Test |
|---|---|---|
| Reads only T2 artifacts | `loader.load_run` goes through `artifacts.read_run`, which reads only `record.json` and one `<table>.parquet` per contract table inside the run directory, after checking the manifest's schema version, file hashes and content hashes | `test_only_files_inside_the_run_directory_are_read`, `test_reader_reads_only_the_files_a_run_may_contain` |
| No statistical or raw-data code | The reader no longer imports the T2 runner. Rendering every page in a fresh process loads no runner, scoring, inference, controls, harness, feature build, data adapter, DuckDB or yfinance module | `test_rendering_every_page_imports_no_model_statistics_or_raw_data_module` |
| Lockbox stays closed | A run is refused unless its record says development mode and `holdout_values_used = False`, and no stored date reaches the lockbox. The lockbox date is the earliest of the record's two copies and `MODEL_HOLDOUT_START`, so a run cannot move its own boundary | `test_runs_that_could_expose_the_lockbox_are_refused` (6 cases) |
| Exact values | Figure builders plot stored columns unchanged; intervals come only from stored `ci_low`/`ci_high`; dates are the stored session dates | `test_dashboard_figures.py` |
| Nothing dropped | Unavailable rows appear as labelled gaps or "unavailable: reason" | `test_unavailable_metrics_show_their_stored_reason`, `test_constant_forecasts_get_a_labelled_gap_not_zeros` |
| No new judgments | Models are grouped by role, never ranked. Colour encodes identity (model, symbol, method), never significance. Reference lines use stored values. The unsafe canary reference is drawn only in its own panel and never appears in a selector, model table or metric table (stored check rows that test it may name it) | `test_forest_uses_stored_estimates_and_intervals_and_never_ranks`, `test_unsafe_canary_reference_is_confined_to_its_panel`, `test_notices_never_quote_the_unsafe_canary_reference` |
| Summary and record agree with the rows | The status cross-checks the stored summary per check family (`passed + failed + undetermined = total`, each count equal to the rows) and the recorded decision method against the check names; any disagreement blocks the run. A record that marks a non-candidate eligible is refused | `test_any_summary_row_disagreement_blocks_the_run` (10 cases), `test_a_1_1_record_missing_or_contradicting_a_field_is_blocked`, `test_a_record_that_could_present_a_non_candidate_as_a_strategy_is_refused` (8 cases) |
| No assumed values for old runs | A schema 1.0 run's missing fields stay empty or "not recorded"; derived ones are labelled as derived | `test_a_1_0_run_shows_what_it_never_recorded_without_inventing_it`, `test_every_page_renders_a_schema_1_0_run_and_labels_its_gaps` |
| Re-verification | The run cache is keyed on the manifest and every file's size and modification time, so a changed file is verified again before display | `test_a_run_changed_after_loading_is_verified_again` |
| No widget chooses data | The output root comes only from the launcher's environment; no text input or file uploader exists | `test_no_widget_can_choose_the_output_root` |

The only derived display is a trailing 63-session mean of the stored daily
rank IC. It is labelled descriptive and has no interval.


## Run status

T2 defines check severities and stores each check's `passed` value, but it
defines no run-level status. The dashboard's mapping uses only the stored
check rows and is published on the overview page.

| Status | Rule |
|---|---|
| INVALID/BLOCKED | A stored leakage or instrument check failed, or has no stored result; or the stored summary or record disagrees with the check rows (below) |
| WARNING | No blocking check; a stored research warning is active |
| VALID | No T2 check failures. This certifies the pipeline, not the results |

How each severity reads `passed`:
- `research_warning`: `passed = True` means the warning is active.
- `info` rows (for example a corroborating method's simulated size), metric
  rows with status `warning`, and unavailable metrics never change the
  status.

The stored summary and record are cross-checked against the check rows,
and any disagreement blocks the run (fail closed):
- the 1.0 summary keys must equal the counts from the rows;
- for a 1.1 summary, `families` must cover exactly the severities in the
  rows, each family must satisfy `passed + failed + undetermined = total`,
  and every count must equal the count from the rows;
- a missing summary, or a 1.1 record without `families` or
  `decision_method`, blocks the run;
- a recorded `decision_method` that disagrees with the stored check names
  blocks the run.

A 1.1 record that could let a reader treat a non-candidate as a strategy
is refused (`malformed_run`): a non-candidate role marked eligible, a
reference predictor not marked ineligible, a missing eligibility flag, or
stored unsafe predictions with no `reference_predictors` entry.

## Schema 1.0 runs

Runs written before output schema 1.1 stay readable. The dashboard never
fills a field 1.0 did not record with an assumed value:
- curve interval method, level and lag are empty, and charts say
  "interval method not recorded";
- the decision method is derived from the stored check names and shown as
  "derived from stored check names (legacy run)" (or "unavailable");
- candidate eligibility is derived from the stored role; the unsafe
  reference's output kind is "not recorded";
- check-family counts are derived from the stored check rows.

The overview lists each of these as a labelled gap, and the run status is
computed from the same rows as for a 1.1 run.


## Pages

1. **Run overview.**
   - The status banner.
   - Identity: run id, spec hash, git SHA and dirty flag, ledger attempts.
   - The label and its signal-timing convention.
   - Universe and exclusions, the lockbox evidence, and inference settings.
   - The registered models, the stored checks with their outcomes, and the
     artifact hashes.
2. **Fold construction.** A per-fold timeline drawn from the stored fold
   table:
   - expanding training rows;
   - the purge gap;
   - the test window;
   - when the test labels mature;
   - the lockbox boundary.

   The fold drill-down shows dates, counts, rows by symbol and fingerprints.
   The purge band's exact dates are not stored, so it is drawn as a
   labelled gap.
3. **Data and feature diagnostics.** Two labelled sections:
   - TRAINING-SCOPE: quantiles, missing rate, extremes, date-variance share,
     VIF, design statistics and correlations.
   - EVALUATION-ONLY: PSI, KS and standardized mean difference as neutral
     heatmaps, captioned "distances only, never a removal rule".
4. **Nulls and controls.**
   - Role definitions, and a rank-IC forest grouped by role (decision method
     filled, corroborating methods hollow).
   - Static orderings, shown as an exact share rather than a p-value.
   - Rank-position occupancy, which shows the per-symbol mean's selection
     effect: NVDA is first on every date.
   - Permutation draws, with the stored p-value and its resolution.
   - Stale-feature differences.
   - The canary panel, which shows the purged canary against the
     hatched, labelled unsafe reference.
5. **Validation results.**
   - Decision-method cards with the minimum detectable effect and the
     no-skill reference share.
   - Daily rank IC with a trailing mean; IC by fold against the stored
     pooled interval.
   - Hit rate against the independence benchmark and the base rate.
   - OOS R² against a chosen baseline, beside the MSE-improvement decision
     test.
   - Same-date rank positions 1..N, with occupancy.
   - Pooled buckets, labelled as pooled.
   - Residual tables, ACF with the overlap band, and Goyal-Welch curves.
6. **Calibration.**
   - States that ranking and calibration are different questions.
   - Mincer-Zarnowitz rows (Driscoll-Kraay errors, normal critical values,
     size not simulated).
   - Prediction against realized with a 45° line and the stored MZ line.
     There is no fitted band and no recomputed regression.
7. **Statistical uncertainty.**
   - Each stored method side by side for any stored statistic, with the
     decision method first.
   - The full inference row: n_eff, df, lag, block and replicates.
   - Simulated false-positive rates against the nominal 5% and the stored
     threshold. Newey-West's over-rejection is stated, not hidden.
8. **Research warnings.** Stored facts quoted by level:
   - blocking (only failed or undetermined T2 checks), warning (active
     research warnings), and notices;
   - the notices cover the hindsight universe, small effective sample,
     undefined metrics, flagged metrics, drift, corroborating-method size,
     lockbox evidence and provenance.

A glossary of definitions is in the sidebar and on its own page.


## T2 findings surfaced by T3 (classified, not patched in the UI)

**Fixed in T2's reader.** These are non-statistical changes needed for
T3's read-only guarantee:
- `artifacts.py` imported the T2 runner at module level just for a type
  hint. It is now a `TYPE_CHECKING` import, so reading a run loads no
  statistical code.
- `read_run` built paths from manifest keys, so a crafted manifest could
  point outside the run directory or leave `record.json` unverified. It now:
  - accepts only `record.json` plus one `<table>.parquet` per contract
    table;
  - requires the manifest, the directory and the record to name the same
    run;
  - rejects symlinks.

  Runs written by `write_run` are unaffected. A run directory that was
  renamed, symlinked or hand-edited is now refused. The owner approved
  keeping this in the T2 reader on 2026-10-10.

**Fixed in output schema 1.1** (branch `fix/t2-diagnostics-record-contract`):
- Defect: the checks summary counted only checks with `passed == False`, so
  a leakage or instrument check with no result (for example an undefined
  canary IC) was missing from it. 1.1 adds per-family `undetermined` counts.
  The dashboard had never depended on the summary for this: it reads the
  check rows and shows such a run as INVALID/BLOCKED.
- Gap: `curves` rows did not record their interval method. They now store
  `ci_method`, `ci_level` and `hac_lag`, and charts label intervals from
  them.
- Gap: the decision method was not a record field. It is now
  `spec.decision_method`.
- Gap: `canary_unsafe_reference` was not described in the record. It is now
  in `spec.reference_predictors` with its output kind and
  `eligible_as_candidate: false`.

**Output gaps, documented for a later T2 minor version (additive):**
- Per-fold IC rows give the reason `undefined` instead of
  `constant_prediction`.
- Per-fold hit-rate rows of constant-sign models are marked `ok`.
- Score-model decile rows are marked `ok` although their x-axis is in score
  units.
- The intercept-slope covariance of the Mincer-Zarnowitz fit is not stored,
  so no calibration band can be drawn.
- Per-date hit-rate, spread and MSE-improvement series are not stored.
- `artifacts.write_run(extra_record=...)` can overwrite top-level record
  keys.
- The purge band's dates are not stored, only its row count.
- PSI is dominated by empty test bins through its 1e-6 floor.
- `cross_sectional_r2` is stored as `ok` 0.000 with interval [0, 0] for
  forecasts that are constant within each date (zero, pooled mean). It is
  degenerate by construction, not a measured zero.
- Each pooled-decile point is a mean over rows, but its stored interval is
  Newey-West around the mean over dates in that bucket, a slightly
  different quantity. The dashboard labels the interval as such.

The T2 follow-up list in `docs/research/model_diagnostics.md` is unchanged.
T3 does not decide whether T4 candidates must beat the per-symbol-mean
selection control; that is still an owner decision.


## Dependencies

Owner-approved 2026-10-09:
- `plotly==7.1.0` and `streamlit==1.64.0` are pinned in `requirements.txt`.
- Streamlit 1.64 requires `pyarrow!=25.0.0,<26`, so the environment's
  pyarrow moved from 25.0.0 to 25.0.1 and stays unpinned. CI will keep
  pyarrow below 26 while this Streamlit version is pinned.
- Streamlit's web stack (starlette, uvicorn, websockets) and Plotly's
  narwhals come in unpinned, like pandas and numpy.


## Tests and limits

- **Tests.** 102 tests in `tests/test_dashboard_*.py`, 20 schema 1.0 / 1.1
  compatibility tests in `tests/test_diagnostics_schema_compat.py`, and
  reader tests in `tests/test_model_diagnostics_artifacts.py`. They use one
  compact fixture run, generated per test session by the real writer, plus
  variants rewritten by the writer (so their hashes verify) for each
  refusal and status case. The schema 1.0 fixture is the same run written
  in the 1.0 layout (`write_schema_1_0_run` in `tests/conftest.py`); a real
  1.0 run from the 2026-10-09 development run also loads, with every gap
  labelled.
- **Mutation probes.** 20 deliberate violations, each in an isolated copy,
  all caught. They covered:
  - each lockbox guard: fold dates, the evidence copy, curve dates, fold
    mode, `holdout_values_used`, date coercion, and the code-constant floor;
  - malformed-run mapping;
  - intervals from the wrong column;
  - rank-position and occupancy values;
  - the unsafe reference leaking into charts, selectors or tables;
  - the root taken from a widget;
  - a hard-coded banner;
  - the cache ignoring changed files;
  - inverted or optimistic reading of `passed`;
  - the reader following paths outside a run.

  The output-contract fix (schema 1.1) adds 21 more, also all caught:
  ignoring the family identity, the family set, family values, record
  issues or a missing summary; a summary hiding undetermined checks;
  treating 1.1 runs as legacy; filling a legacy interval level; dropping
  the non-candidate eligibility guard, the decision-method cross-check, the
  missing-eligibility refusal or the undescribed-reference refusal; a
  hard-coded interval label or level; an eligible reference predictor;
  dropping the record/manifest version check or making it optional;
  validating old runs against the current schema; accepting a newer minor
  version; and hiding the legacy notice.- **AppTest smoke tests** cover every page, every model, baseline and fold
  selection, refused runs, an empty root, and the launcher's binding.
- **Chart display** in a browser is reviewed by eye, not by automated tests.
  Plotly charts are not screen-reader friendly, so key values also appear
  in tables and captions.
