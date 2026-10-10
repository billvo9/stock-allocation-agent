# Model diagnostics, null models and run outputs (roadmap ticket T2)

Status: implemented on `feature/ml-diagnostics-null-models` (2026-10-09).
Code: `src/stock_agent/model_diagnostics/`, runner script
`scripts/run_null_diagnostics.py`. Builds on
`src/stock_agent/model_validation/` without changing its contracts (one
additive function, `audit.table_sha256`).

The question T2 answers: when a model eventually produces a result, can its
predictive skill, uncertainty, data quality and failure modes be measured
without leaking future information or silently changing the experiment?
T2 builds the instruments and checks them on predictors whose correct
answer is known. It fits no real model.


## What runs

Every predictor runs through `model_validation.harness.run_walk_forward`
with the same folds as any future model: global purge, lockbox, a fresh
estimator per fold, and one predict call per date. None has a special path.

| Name | Role | What it answers |
|---|---|---|
| `zero` | null | Baseline for scale metrics. Constant, so rank IC, hit rate and calibration are undefined (reported as unavailable, never 0). |
| `pooled_mean` | null | Expanding training-label mean. Campbell-Thompson baseline for out-of-sample R² under a raw-return label. Constant within each date. |
| `random_noise` | null | iid noise. Exercises rank and sign metrics under no skill; seeded per fold. |
| `per_symbol_mean` | control (selection) | How much apparent skill symbol identity alone carries on this universe. **Not a null here** (see below). |
| `momentum_20d` | control (benchmark) | Unfitted score, sign +1 pre-registered. A score, not a calibrated forecast: scale metrics unavailable. |
| `memorizer` | canary | Nearest same-symbol training label. Purged, it can only show the skill of a lagged label. |
| `canary_unsafe_reference` | canary positive control | The memorizer trained **without** purging, outside the harness. It must show spurious skill. |

Placebos run as feature transforms through the same harness:

- **Block-permutation null.** Within each date, every row receives another
  symbol's feature vector from the same date. The mapping is held for
  63-session blocks anchored at the first test window, and drawn from
  SHA-256 keys of (seed, draw, block, symbol). Each date's cross-section and
  the series' persistence stay intact; only the symbol-to-future-return link
  is broken.
- **Static orderings** (labels only). The mean rank IC of all 24 fixed
  rankings of the 4 names: the reference distribution for a tilt.
- **Stale-feature timeliness control.** Each row gets its own symbol's
  features from 260, 290 or 330 sessions earlier (backward only, never
  wrapped; multiples of 63 avoided). IC(model) − IC(stale) asks whether a
  signal is timely.


## Metrics and conventions

All on out-of-sample rows only, after all predictions of a run are complete.
The label is explicit (`label_id`, `target_column` in the record); T2 does
not change it.

| Metric | Convention |
|---|---|
| Rank IC | Spearman within each date (average ranks for ties), then averaged over dates. Needs ≥ 3 names and non-constant predictions and labels; otherwise the date stays in the table as unavailable with a reason. With 4 names IC takes 11 values in steps of 0.2. |
| Hit rate | Rows with zero prediction or zero label excluded and counted. Tested against the independence benchmark p·q + (1−p)·(1−q), not 50% and not the base rate alone. |
| OOS R² | 1 − ΣSSE_model / ΣSSE_baseline on identical rows (keys and labels checked), never an average of per-fold R². Baselines: zero, pooled mean, per-symbol mean. Plus the per-date MSE improvement with full inference, and a cross-sectional R² on date-demeaned values. |
| Mincer-Zarnowitz | y = a + b·ŷ pooled and within date, Driscoll-Kraay errors. Unavailable when the forecast is constant within every fold. |
| Rank positions | 1..N per date (N = names on that date). A tied group gives every position it spans the group's mean label. Top-minus-bottom spread, position occupancy by symbol. |
| Pooled deciles | Labelled pooled. Flagged when predictions take fewer distinct values than bins. |
| Residuals | Mean, dispersion, MAE and RMSE by fold. ACF per symbol and of the date mean, with lags < horizon marked `overlap_expected`. |
| Goyal-Welch | Cumulative per-date SSE improvement over each baseline. |


## Inference

Every statistic is reduced to one value per scored date before inference.
Three estimates are reported side by side:

- **Newey-West**: Bartlett kernel, explicit lag (default 40), normal critical
  values.
- **Fold-block t**: per-fold means with t(B−1) critical values.
- **Circular block bootstrap**: whole dates, 63-session blocks, seed and
  replications recorded. Mean statistics use the bootstrap SE with
  t(k−1) critical values. Ratio statistics (R²) use percentile intervals
  labelled `percentile_uncalibrated`; their decision test is the per-date
  MSE improvement.

`n_eff` is filled for Newey-West only: n·γ₀/LRV, the number of independent
dates carrying the same information. It can exceed n under negative
dependence. The fold-block row carries its degrees of freedom in `df`.

The minimum detectable rank IC follows the decision test:
(t₀.₉₇₅,df + t₀.₈₀,df) × SE_fold-block, i.e. 2.93 × SE at 23 df (alpha 5%,
power 80%).

No single p-value decides. The run's checks use the fold-block t, the
best-calibrated method in simulation. Its simulated size must stay at or
below 6.5%: about 3 Monte Carlo SEs above 5% at 2,000 replicates.
Newey-West and the bootstrap are reported beside it, and their simulated
sizes are recorded as `info` checks for the run's own sample size.
Normal-based intervals on rank-position and decile curves inherit
Newey-West's over-rejection: they cover about 91-94%, not 95%.

**Simulated false-positive rates at nominal 5%** (development run: 1,489
dates, 4 names, 20-session labels, 2,000 no-skill replicates per signal;
seed-derived, recorded in `record.json`):

| Signal | iid SE | Newey-West lag 40 | Fold-block t | Bootstrap |
|---|---|---|---|---|
| iid noise | 5.3% | 6.0% | 5.6% | 5.8% |
| momentum (trailing 20-day return) | 54.6% | 7.6% | 5.3% | 5.8% |
| ordering constant per 63-session block | 60.5% | 9.2% | 5.5% | 7.8% |

Newey-West at lag 40 is better than lag 20 (8.6-11.4% in the design review)
but still over-rejects for persistent signals. The reason is the critical
values, not the lag: normal critical values ignore the noise in the
estimated long-run variance. Fixed-b critical values are the follow-up (and
a good owner exercise).


## Lockbox

- `prepare_development_frame` restricts to the universe, truncates before
  2025-01-01, and **then** labels. The last 21 rows per symbol become
  unlabeled instead of carrying labels built from holdout prices.
  - Features and matured labels are bit-identical to the untruncated panel
    (verified on the real panel by the data-engineer review).
  - SNDK has no pre-lockbox rows, so it is excluded with reason
    `no_pre_lockbox_rows`.
- `run_diagnostics` refuses:
  - any row dated in the lockbox, labeled or not;
  - any finite label entering or ending in the lockbox;
  - any fold with held-out rows (a sign that truncation was bypassed);
  - holdout-mode folds.
- The record does not claim that holdout data was never touched:
  - `holdout_rows_loaded = true`: the feature SQL scans the whole price
    file.
  - `holdout_values_used = false`: derived from the data, never passed in.
  - An adjusted-price disclosure: pre-2025 `adjusted_close` levels embed
    later dividend factors. Same-symbol ratios cancel the factor.
- A poison test sets every holdout-period price and feature to 1e9 and
  requires identical outputs.


## Saved outputs

Layout: `reports/runs/<run_id>/`, with an append-only `reports/ledger.jsonl`
beside it. Both are ignored by Git.

- `record.json` holds:
  - the spec: code state, environment versions, input hashes, label,
    lockbox truncation facts, universe and symbol codes, fold parameters,
    models with variant ids, config, and the seed rule;
  - lockbox evidence;
  - estimator audit (fallback counts);
  - simulated inference calibration;
  - output hashes: label-free prediction hashes, labels hash, table hashes,
    and the results hash;
  - the checks summary.
- Tables (Parquet, schema version 1.0):
  - `inputs`: the truncated labeled frame used;
  - `folds`;
  - `predictions`;
  - `metrics`;
  - `curves`;
  - `null_draws`;
  - `feature_stats`, scope `training` or `evaluation`;
  - `feature_pairs`;
  - `checks`.
- `manifest.json` is written last as the completion marker. It holds
  per-file SHA-256 hashes and per-table content hashes.
- `run_id` = start time + `spec_sha256[:12]`.
  - Every attempt, including a failure, gets its own ledger line, so trials
    are never undercounted.
  - The same spec giving a different `results_sha256` reveals
    nondeterminism.
- The reader refuses:
  - an unknown major version;
  - any file or table whose hash differs from the manifest;
  - a run without a manifest.

T3 can build every planned page from these tables without re-running a
model or reading raw data.


## Evidence

Development run `20261009T234504Z-f5699748abfd`: 24 folds of 63 sessions,
2019-2024, 5,956 scored rows over 1,489 dates. Mean rank IC is shown with
fold-block t standard errors:

| Model | Mean rank IC (SE) | Reading |
|---|---|---|
| random_noise | 0.015 (0.017) | No skill, as required |
| memorizer (purged) | 0.014 (0.069) | No skill, as required |
| canary_unsafe_reference | 0.262 (0.048) | Spurious skill without purge: the canary is sensitive |
| momentum_20d | 0.024 (0.055) | No evidence. The stale control is as good (IC − stale IC −0.005 to −0.020) |
| per_symbol_mean | 0.126 (0.055), t = 2.3 | Research warning: selection, not foresight |

Other observations:
- **zero / pooled_mean:** IC unavailable on every date
  (`constant_prediction`).
  - The pooled-mean hit rate is 59.1%, flagged `constant_sign_prediction`:
    with one sign it equals the base rate by construction.
  - Its pooled deciles are flagged `constant_within_date`: they group whole
    folds by time and are not a calibration curve.
- **R²:**
  - pooled mean vs zero: +3.8% [−1.3%, 8.3%];
  - per-symbol mean vs zero: +5.9%;
  - per-symbol mean vs pooled mean: +2.2%.

  All of this is drift and tilt of a hindsight-selected universe.
- **Per-symbol mean against the placebos:**
  - block-permutation p = 0.04;
  - static orderings: 4 of the 24 fixed rankings (share 0.17) reach at
    least its IC. That is an exact share of a complete enumeration, not a
    p-value.
- **Minimum detectable IC** with fold-block t at 80% power:
  - momentum 0.160;
  - per-symbol mean 0.161;
  - purged memorizer 0.203.
- **Share of dates with positive IC:** random noise 0.478, against an exact
  no-skill expectation of 0.458 (with 4 names, IC = 0 on 1/12 of
  orderings).

**Tests and mutation probes:**
- **Suite:** 140 new tests (1,005 in the suite).
- **Mutation probes:** 47 deliberate violations, each run in an isolated
  copy of the source against the new tests. 46 were killed.
  - The survivor, removing the runner's early frame-binding check, is
    equivalent: the harness refuses the same frame on every fold before
    anything else runs.
  - The violations cover:
    - the pooled mean computed outside the harness;
    - IC across dates;
    - ordinal ties;
    - IC = 0 for constant forecasts;
    - bootstrap block length 1 or no wrap;
    - lag 20 as the default;
    - a wrong Bartlett weight;
    - an iid SE;
    - diagnostics rescaling features;
    - each lockbox guard removed;
    - a scorer dropping its worst rows;
    - stale shifts that wrap or look forward;
    - permutations keyed on labels, counted from the last date, redrawn per
      date, or crossing dates;
    - a prediction hash that includes labels;
    - a caller-supplied holdout flag;
    - symbol codes assigned by first appearance;
    - OLS errors in Driscoll-Kraay;
    - R² against the test mean;
    - labeling before truncation;
    - scoring inside the prediction loop;
    - the fold-block test computed over rows;
    - training diagnostics reading test rows;
    - drift reading labels;
    - a missing VIF singularity guard;
    - the completeness check removed;
    - zero predictions counted in the hit rate;
    - ties broken by order;
    - a candidate model using control columns;
    - a writer that overwrites;
    - a reader that skips the hash check;
    - an ignored permutation anchor;
    - scoring before the placebo and canary predictions;
    - an unsafe canary that is actually purged;
    - cross-sectional R² without demeaning;
    - a flipped top-minus-bottom sign;
    - a reversed stale difference;
    - a minimum detectable effect from normal quantiles;
    - holdout-mode folds accepted.


## Design decisions and disagreements resolved

- **Per-symbol mean is a selection control, not a null.** Both the
  quantitative and statistical reviews measured IC ≈ 0.12-0.13 from
  NVDA's history alone.
  - The plan's acceptance rule "nulls show no skill" holds for the nulls.
  - The control's apparent skill is a recorded research warning.
  - Whether T4 must beat it on paired per-date differences is a T4 decision.
- **Which placebo is the null.** The reviews disagreed:
  - the statistical review favoured the within-date permutation for rank
    IC;
  - the quantitative review showed it is lenient toward a static tilt and
    favoured the backward date shift.

  Resolved by reporting all three with their distinct questions:
  - the permutation null asks about the symbol link;
  - the static orderings ask about tilt;
  - the stale control asks about timeliness.

  None is an acceptance gate on its own.
- **Primary inference method.** Fold-block t decides checks. The owner's
  reporting contract (point estimate, HAC, bootstrap, null comparison, fold
  stability) is met by showing all of them.
- **run_id.** The design reviews differed:
  - the architect wanted a pure spec hash;
  - the data engineer wanted the start time plus a spec hash.

  Chosen: the start time plus the spec hash, so every attempt is ledgered.
  The spec hash and results hash are both recorded.
- **Newey-West is implemented here,** although the roadmap had suggested it
  as an owner exercise. The T2 request made "HAC and bootstrap implemented
  and tested" a completion criterion. The owner exercise moves to fixed-b
  critical values (below).
- **JSONL ledger, not Parquet.** It is append-only and a crash cannot
  corrupt earlier lines.


## Limitations and follow-ups

- The universe is 4 hindsight-selected names, so every number above checks
  the pipeline. None is product evidence (open item 6, U track).
- Newey-West with normal critical values over-rejects (7.6-9.2%) for
  persistent signals. Add fixed-b (Kiefer-Vogelsang) critical values, or
  keep fold-block t as the decision method.
- Nothing yet stops a candidate model's feature list from using
  `control_*` columns through the harness directly (outside `ModelSpec`).
  Rejecting them in `validate_feature_columns` would change a contract, so
  it waits for T4.
- The ledger lives under ignored `reports/`, so it can be deleted. Holdout
  once-only enforcement and multiple-testing counts need a Git-tracked
  record (T6, owner decision).
- Raw-data retrieval times are not recorded by the downloaders (`null` in
  the record).
- The canary checks the mean IC over whole folds. A partial purge error
  (a few overlapping sessions) would be diluted. A sharper check compares IC
  in the first `horizon` sessions of each window with the rest; the leaky
  reference's IC decays from 0.87 to 0.42 across the first 20 sessions.
- The purged memorizer behaves like a lagged-label signal (IC ≈ 0.12 in
  window sessions 20-39 on real data). On another universe its "no skill"
  check could fail for legitimate reasons; read a failure with the
  per-session profile.
- Driscoll-Kraay intervals (Mincer-Zarnowitz) use normal critical values
  and have no simulated size yet; expect over-rejection similar to
  Newey-West's.
- Pooled hit rate mixes market timing with cross-sectional skill. A
  rank-based hit ("top name beats the median") would isolate the latter.
- Random-noise seeds are derived from the fold call order, not the fold id.
  Changing the first development window changes them; this is not leakage.
- `scripts/run_null_diagnostics.py` has no offline test. Its pieces
  (`prepare_development_frame`, `run_diagnostics`, `artifacts`) are tested.
- T2 adds about 5 s to each pytest run (about 10 s to `scripts/verify.py`,
  which runs the suite twice).


## Learning Notes

**1. What a null model is.**
- **Problem.** A model's score means nothing until we know what a
  predictor with no information scores under the same procedure.
- **Concept.**
  - A null model has a known expected skill of zero by construction.
  - A control has a known non-zero source of score (a benchmark, a
    selection effect).
  - A canary has no possible skill except leakage.
- **Assumption.** The procedure is identical. A null run on an easier path
  calibrates nothing.
- **Project example.** The zero forecast's R² against itself is 0. The
  pooled mean beats it by 3.8% only because 59% of development labels are
  positive (the universe went up).
- **"A null producing an exciting result is a bug or a research warning."**
  Here it was a warning: the per-symbol mean.

**2. Why the controls answer different questions.**
- **Zero:** "is there any signal in levels?"
- **Pooled mean:** "is there anything beyond the average return?"
- **Block permutation:** "does matching a forecast to the right stock on
  the right date matter?" It assumes the stocks are interchangeable.
- **Static orderings:** "could a ranking that never changes do as well?"
  On a hindsight universe, yes: always ranking NVDA first earns IC ≈ 0.12.
- **Memorizer:**
  - Purged, "does the harness let label information through?";
  - unpurged, "would we notice if it did?" (IC 0.26).

**3. Rank IC, and why 4-name daily IC is coarse.**
- **Formula.** IC_t = corr(rank ŷ_t, rank y_t) across the names on date t;
  the reported value is the mean over dates.
- **Coarseness.** With 4 names each IC_t takes one of 11 values (−1 to 1
  in steps of 0.2). One swapped pair moves it by 0.2 or more.
- **Small example.** Predictions [1, 1, 2, 3] vs labels [1, 2, 3, 4]: the
  tie gets average rank 1.5, so IC = √0.9 ≈ 0.949.
- **Misleading.** Never pool across dates. If market-wide moves line up
  with forecast levels, the pooled correlation can be +0.9 while every
  daily IC is −1. That case is a test.

**4. Autocorrelation and overlapping labels.**
- **Mechanism.** Adjacent rows' 20-session labels share 19 returns, so
  daily ICs are strongly autocorrelated. A persistent signal (momentum, a
  fixed tilt) stretches the dependence further.
- **Consequence.** The naive standard error, sd/√n, is about √(design
  effect) times too small. In simulation it rejected 55-60% of the time
  under no skill, against a nominal 5%.

**5. HAC / Newey-West.**
- **Formula.**
  - LRV = γ₀ + 2·Σ_{l=1..L} (1 − l/(L+1))·γ_l
  - SE(mean) = √(LRV/n)
- **Hand example.** x = 1..5 with L = 1: γ₀ = 2, γ₁ = 0.8, LRV = 2.8, so
  Var(mean) = 0.56. The iid version would give 0.4.
- **Why the weights.** The Bartlett weights keep LRV ≥ 0.
- **Lag choice.** It must cover the dependence (≥ 40 here). Even then,
  normal critical values over-reject with this few independent blocks.
- **Validation.** Simulate the null and count rejections. That is what
  `null_rejection_rates` records in every run.

**6. Block bootstrap.**
- **Method.** Resample whole dates in contiguous circular blocks (63
  sessions ≈ 3× the horizon), so each block keeps its internal dependence.
  Each date keeps its cross-section.
- **Failure mode.** A row-level or date-level iid bootstrap destroys
  exactly the dependence that matters.
- **Overlap with Newey-West.** A block of length b behaves like Bartlett
  with lag ≈ b, so the bootstrap and Newey-West are not independent
  checks.

**7. Effective sample size.**
- **Formula.** n_eff = n·γ₀/LRV: how many independent dates carry the same
  information about the mean.
- **Project example.** The per-symbol mean IC has n = 1,489 dates but
  n_eff ≈ 119.
  - Its fold-block SE is 0.055, so with 23 degrees of freedom the minimum
    detectable IC is (2.07 + 0.86) × 0.055 ≈ 0.16.
  - "No evidence" here is not "no effect".

**8. Calibration.**
- **Mincer-Zarnowitz.** Regress y on ŷ. Ideal a = 0, b = 1.
  - b < 1 can mean an overconfident forecast or just a noisy one.
  - With a raw label, the pooled slope mostly reflects market timing; the
    within-date slope removes it.
  - Example: the per-symbol mean's slopes are 0.97 pooled and 1.18 within
    date, with SE ≈ 0.35. "Calibrated" here means "too imprecise to tell".
- **Reliability.** The pooled deciles show mean realized against mean
  predicted.

**9. Multicollinearity, VIF and condition number.**
- **Formulas.**
  - VIF_j = 1/(1 − R²_j) = (R⁻¹)_jj on standardized training data.
  - κ = σ_max/σ_min of the standardized design.
- **Two features with correlation 0.8** give VIF = 2.78 and κ = 3 (both
  are tests).
- **Undefined cases.** VIF is undefined, and reported with a reason, for
  one feature, n ≤ p + 1, a constant column, or a singular design.
- **On the real panel.** The three price features have VIF ≈ 1.05 and
  κ ≈ 1.25: no collinearity yet. That changes when macro features join.
  Macro features are pure date components, with date-variance share
  exactly 1.

**10. Diagnostics leakage.**
- **The trap.** A diagnostic computed on test rows that changes the
  experiment, for example dropping a feature that drifted, or
  standardizing with full-sample statistics, leaks the test period into
  the model.
- **How T2 prevents it.**
  - Training diagnostics read only `fold.train_rows`.
  - Evaluation diagnostics read test features, never labels, and nothing
    feeds them back.
  - Tests prove that predictions are identical whatever diagnostics are
    computed, and a mutant that rescales features from the diagnostic set
    is caught.

**What to learn next.**
- Fixed-b asymptotics (Kiefer-Vogelsang).
- Diebold-Mariano and Clark-West tests for nested forecast comparison.
- Politis-White automatic block length.
- Why permutation tests need exchangeability.

**Owner exercise (bounded).** Implement fixed-b critical values for the
Bartlett kernel (b = (L+1)/n), apply them to `newey_west_mean`, and check
with `null_rejection_rates` that the simulated size for the momentum and
block-constant signals moves toward 5%. The calibration tests in
`tests/test_model_diagnostics_inference.py` already provide the harness.
