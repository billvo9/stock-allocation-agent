# Research dashboard over T2 outputs (roadmap ticket T3)

Status: T3 merged in PR #44 (2026-10-10); the T3.1 refinement (research
workspace, Goyal-Welch views, maturity, filters, design system) is on
`feature/dashboard-research-workspace`.
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
| Only mature labels count | Predictions whose label ended after the evaluation as-of are pending: listed, excluded from derived series, and blocking when stored (T2 scored them) | `test_pending_forecasts_never_contribute_and_stay_visible`, `test_pending_predictions_block_the_run_and_stay_listed` |
| Derived series match T2 | The date-normalized Goyal-Welch series is drawn only if it reproduces T2's stored mean, fold-mean and cumulative curve | `test_the_derived_series_reproduces_t2_for_every_forecast_pair`, `test_a_comparator_that_does_not_reproduce_t2_is_not_drawn` |
| Filters never create inference | Filtered samples show counts and plain averages with a "not precomputed" notice; no interval or p-value | `test_a_filter_shows_descriptive_numbers_and_never_inference` |
| Point-in-time regimes | Built from stored features and earlier-date thresholds; retrospective episodes are flagged ex post | `test_point_in_time_regimes_never_change_when_later_data_arrive`, `test_retrospective_regimes_are_flagged_and_unavailable_ones_explained` |
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
3. **Data and feature diagnostics.** Two sections, each marked with a scope
   pill:
   - Training scope: quantiles, missing rate, extremes, date-variance
     share, VIF, design statistics and correlations.
   - Evaluation only: PSI, KS and standardized mean difference as neutral
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
   - Residual tables and ACF with the overlap band; a link to the
     Goyal-Welch page.
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
8. **Forecast error (Goyal-Welch).** See "Goyal-Welch views" below.
9. **Research warnings.** Stored facts quoted by level:
   - blocking (only failed or undetermined T2 checks), warning (active
     research warnings), and notices;
   - the notices cover the hindsight universe, small effective sample,
     undefined metrics, flagged metrics, drift, corroborating-method size,
     lockbox evidence and provenance.

A glossary of definitions is in the sidebar and on its own page. The
navigation groups the pages as Run, Construction, Evidence and Reference.


## Maturity: only realized labels count

`evaluation_asof` is the last session of the run's truncated development
frame (`record.lockbox_evidence.max_date`, written by T2 after truncating at
the lockbox and before labelling). It is not the raw data's last date: the
feature build may load later rows (`holdout_rows_loaded`), which T2 drops
first. A stored prediction is **mature** when
`target_end_date <= evaluation_asof` (a label is known after the close of
its end date) and its label is stored; otherwise it is **pending**.
- Only mature predictions enter the series the dashboard derives
  (Goyal-Welch, filtered summaries). Pending predictions of the registered
  models are counted on the overview and listed, never dropped silently.
- T2 scores every stored prediction, so a pending stored prediction means
  T2's own results include an unrealized label. The run is then
  INVALID/BLOCKED, every page of stored T2 results says so, and no derived
  Goyal-Welch line is drawn (it cannot reproduce T2's values). The same
  holds when a run does not record its as-of.
- Development runs have no pending rows by construction: labels are
  computed after truncation, so every label ends on or before `max_date`.
  The rule matters for later holdout and live runs.

Code: `dashboard/evaluation.py` (`evaluation_asof`, `maturity`), applied in
`loader.load_run`.


## Goyal-Welch views

The page compares a **forecast** model's squared errors with recorded
comparators on **identical observations**: the same fold, date, symbol and
label, or the page refuses the pair. Score models have no error scale and
are not offered.

**Primary: date-normalized advantage.** For each date t with N_t names,

    d_t = (1/N_t) * sum_i [ (y_it - b_it)^2 - (y_it - f_it)^2 ]

and the chart shows the running sum of d_t (rising = the model's errors
are smaller), plotted by forecast date: each value is realized when its
label ends (horizon + entry lag sessions later). Every date weighs the
same, so a growing universe cannot steepen the curve by itself. A per-date
panel shows d_t with N_t beneath it.

**Secondary: observation-weighted cumulative.** T2's stored
`cumulative_sse_improvement` (sum over all name-dates), in a labelled
expander. Dates with more names weigh more.

**Comparators keep their identity.** The page offers the comparators T2
recorded (`spec.config.baselines`), each labelled with its kind and its
stored description:
- `zero`: null baseline (natural for an excess-return label);
- `pooled_mean`: null baseline, the expanding mean of purged training
  labels (natural for raw returns);
- `per_symbol_mean`: **selection control (not a null baseline)**, drawn
  dashed and never selected by default.

No single comparator is privileged: the default shows every recorded null
baseline, and nothing at all when only the selection control is recorded.
The page shows the label so the natural comparator is visible (expanding
mean for raw returns, zero for excess returns). Every comparator is a
per-fold forecast from purged training rows only (point-in-time; the
expanding mean is refit once per fold, not every date). The comparator
table keeps T2's recorded note apart from this dashboard statement, and
lists any recorded baseline it cannot offer with the reason.

**Trusted inference only.** Before a line is drawn, the derived series must
reproduce T2's stored values for the same pair: the Newey-West estimate
(the mean of d_t over dates), the fold-block estimate (the mean of per-fold
means of d_t, which differs slightly when folds hold different numbers of
dates) and the stored observation-weighted curve. Both sides sum the same
numbers in a different order, so the check allows a relative 1e-9 plus an
absolute 1e-12 scaled by the losses summed so far. Otherwise the line is
not drawn and the reason is shown (fail closed). Intervals and tests on the
page are T2's stored rows for the full sample, each labelled with its
estimand: the decision method's estimate of mean d_t, and OOS R² with its
uncalibrated percentile interval (OOS R² is observation-weighted).

**Descriptive filters.** A forecast-date range and a regime can restrict
the sample. A filtered view re-accumulates the curve over the chosen dates
(excluded dates are gaps) and shows a table of counts and plain averages
(dates, observations, mean d_t, share of dates the model was better, mean
stored daily rank IC), with the notice that filtered inference is not
precomputed. No interval, standard error or p-value is computed for a
filtered sample. A caption explains that overlapping labels make a slice of
k dates hold far fewer independent outcomes than k, quoting T2's stored
full-sample effective sample size.

**Regimes** (`dashboard/regimes.py`), in two kinds that are never mixed:
- Point-in-time, built from stored features of each date and thresholds
  from earlier dates only (prefix-stability tests):
  - universe volatility: the cross-sectional median of `volatility_20d`,
    "high" above its median over the previous 252 dates;
  - universe trend: the sign of the cross-sectional mean of
    `momentum_20d`.

  These describe the run's own universe, not the market. They are
  unavailable, with the reason shown, when the stored inputs lack the
  feature.
- Retrospective (ex post): S&P 500 closing peak-to-trough windows (COVID-19
  crash 2020-02-19 to 2020-03-23; 2022 bear market 2022-01-03 to
  2022-10-12). Choosing one shows a warning that its boundaries were known
  only afterwards and that it is for descriptive slicing, never a signal.

**Portfolio equity** appears only as a disabled "not available yet" card:
execution timing, allocation, transaction costs and risk constraints are
defined in T5. Forecast error is not portfolio performance.

On the 2026-10-10 development run (`20261010T192511Z-d967ec9c0ff2`, schema
1.1), all nine forecast model-comparator pairs reproduce T2's stored values
(largest estimate difference 2e-18, largest curve difference 7e-15), and no
prediction is pending. That run has 4 names on every date, so date and
observation weighting coincide there; a test fixture whose universe grows
during the test period shows the two curves differ while both still
reproduce T2.


## Design system

One token module (`dashboard/theme.py`) drives the look:
- **Streamlit theme** (`.streamlit/config.toml`, kept identical to
  `streamlit_theme()` by a test): warm neutral canvas and surfaces, one
  restrained slate-teal accent, status colours for alerts, the platform UI
  font (no web font is fetched), a calmer heading scale, and radii.
- **A thin CSS layer** (`theme.css()`, injected once per page): cards,
  status pills, opaque captions, keyboard focus rings, tabular figures, chart
  surfaces. It uses only `data-testid` hooks and keyed containers
  (`st-key-sa-*`).
- **Plotly layouts** (`style.base_layout`): token fonts, surfaces, grid,
  subtitles, reference lines, and dotted fold boundaries distinct from the
  grid. Model colours keep their identity palette (Okabe-Ito).

Accessibility. Every text token meets WCAG AA (4.5:1) on every surface, and
so does every status foreground on its background. Streamlit draws
captions at 60% opacity, which falls just below AA on these surfaces, so the
CSS layer renders them opaque in the muted text token (at least 6.4:1).
Keyboard focus shows a 2px accent outline. Severity is always written in
words: the run banner and pills name their level, and every notice starts
with "Blocking", "Warning" or "Note" and carries an icon. Only the sidebar's
run status is a live region. All of this is tested.

Motion marks state changes only and never touches a value. A chart fades in
(opacity, 200 ms) when it appears, and an expander summary shades on hover.
Plotly's own transitions are off: they would tween traces through values
T2 never stored. Under `prefers-reduced-motion` the dashboard's CSS motion
is off; Streamlit's built-in widget animations (for example its expander
height animation) follow Streamlit's own handling.

**Shared components** (`dashboard/components.py`): `page_header`,
`status_banner`, `pill`, `stat_row`, `estimate_card` (a stored estimate
with its interval, p-value and status in one card), `card`, `section`,
`notice`, `chart`, `download` and `future_state`. A test checks that
`views.py` uses no raw titles, alerts, bordered containers, HTML or colour
literals. T4's Ridge, Lasso and Elastic Net pages should build from the
same components and tokens.


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

- **Tests.** 154 tests in `tests/test_dashboard_*.py` (52 added by T3.1 in
  `test_dashboard_evaluation.py`, `test_dashboard_regimes.py` and
  `test_dashboard_workspace.py`), 20 schema 1.0 / 1.1 compatibility tests
  in `tests/test_diagnostics_schema_compat.py`, and reader tests in
  `tests/test_model_diagnostics_artifacts.py`. They use compact fixture
  runs generated per test session by the real writer (a fixed 4-name
  universe, and a universe that grows during the test period and overlaps
  the 2020 episode), plus variants rewritten by the writer (so their hashes
  verify) for each refusal and status case. The schema 1.0 fixture is the same run written
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
  version; and hiding the legacy notice.

  T3.1 adds 22 more, also all caught: a fixed comparator list; an
  observation-weighted mean; a regime threshold that includes its own date;
  inference columns in the filtered table; a constant contrast function;
  unchecked label windows, fold membership or duplicate keys; comparing
  estimates T2 could not form; unlisted pending rows; a dropped
  retrospective label; the selection control as a fallback default; the
  caption contrast rule or focus ring removed; Plotly tweening restored; a
  quiet stored-results notice; the unsafe reference in the pending list; an
  unscaled curve tolerance; notices without their level word; every pill a
  live region; a filtered curve drawn through excluded dates; and omitted
  comparators without a reason.- **AppTest smoke tests** cover every page, every model, baseline and fold
  selection, refused runs, an empty root, and the launcher's binding.
- **Chart display** in a browser is reviewed by eye, not by automated tests.
  Plotly charts are not screen-reader friendly, so key values also appear
  in tables and captions.


## Learning Notes (T3.1: Goyal-Welch views and regimes)

- **Problem.** A cumulative squared-error plot shows *when* a forecast beat
  its comparator, not just whether it did on average. Summed over all
  name-dates, it also grows with the number of names, so a universe that
  expands (new listings, a point-in-time universe in the U track) would
  steepen it with no change in skill.
- **Concept.** Goyal and Welch (2008) plot the cumulative difference
  SSE(benchmark) - SSE(model) over time; Campbell and Thompson (2008)
  measure OOS R² = 1 - SSE_model / SSE_benchmark against the historical
  mean. Both are observation-weighted.
- **Formulation.** The date-normalized advantage d_t (above) gives each
  date one vote. T2 tests the same per-date series as
  `mse_improvement_vs_<comparator>`: its Newey-West and bootstrap estimates
  are the mean of d_t over dates, and its fold-block estimate (the decision
  method) is the mean of per-fold means, which differs slightly when folds
  hold different numbers of dates. That is why the page can check its
  derived series against T2. OOS R² stays observation-weighted; the two
  answer different questions ("typical date" versus "pooled error"). In
  Goyal and Welch's single-series setting (one return per date) the two
  weightings coincide; on a panel they do not.
- **Small example.** A model gains 0.01 in squared error per name on every
  date. With 2 names, then 4, then 8: d_t = 0.01 each date, so the primary
  curve rises in equal steps (0.01, 0.02, 0.03). The observation-weighted
  curve rises 0.02, 0.06, 0.14: it accelerates only because N grows (this
  is `test_a_growing_universe_does_not_steepen_the_primary_curve`).
- **Assumptions.** Squared-error loss; identical observations; comparators
  formed per fold from purged training rows only. The expanding mean here
  is refit once per fold (with a purge lag), not every period as in Goyal
  and Welch: still point-in-time, just staler. A comparator fitted on the
  full sample (for example a full-sample mean) would leak future labels
  into the benchmark.
- **Leakage risks.**
  - An unrealized label scored as if known: hence the maturity rule.
  - A regime threshold estimated on the full sample: a "high volatility"
    cut-off that uses future volatility. The point-in-time regimes use only
    earlier dates, and tests prove that later data cannot change them.
  - Ex-post episodes are known only afterwards. They describe history and
    must never become a signal.
- **Ways results could mislead.**
  - Slicing by many regimes or date ranges and keeping the slice that looks
    good is a multiple-testing problem; that is why filtered views carry no
    p-values or intervals.
  - Labels overlap: with a 20-session horizon, consecutive dates share most
    of their outcome window (on the development run d_t has lag-1
    autocorrelation near 0.96, and T2's stored effective sample size is
    about 92 of 1489 dates). A 24-date slice holds only one or two
    independent outcomes.
  - Curves are plotted by forecast date, but each value is realized up to
    21 sessions later. Forecasts dated inside the COVID-19 crash window are
    scored mostly on the rebound after the trough.
  - Date normalization gives a date with few names (a noisier d_t) the same
    weight as a full date: it trades some precision for an estimand that
    does not drift with universe size.
  - A cumulative curve can look trending from a few large dates; read it
    with the per-date panel and the stored fold-block result.
  - Four hindsight-selected names make every number here a pipeline check,
    not product evidence.
- **Practitioner validation.** Prefer the pre-registered full-sample test
  (here fold-block t on mean d_t), check sign stability across folds, and
  for nested models (a candidate that contains the benchmark) use a
  Clark-West adjusted comparison rather than a plain MSE difference.
- **Learn next.** Diebold-Mariano tests with HAC errors, Clark-West for
  nested forecasts, and how date-weighting versus observation-weighting
  changes the estimand.
- **Exercise.** By hand, compute d_t and both cumulative curves for the two
  dates in `test_the_advantage_is_the_mean_squared_error_difference_per_date`
  (2 names, then 4), then explain why the fold-block estimate can differ
  from the Newey-West estimate when folds hold different numbers of dates.
