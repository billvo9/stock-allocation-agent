# Model-validation foundation

Status: implemented on `feature/ml-validation-foundation` (2026-10-06).
Code: `src/stock_agent/model_validation/`, labels in
`src/stock_agent/features/training.py`.

This note records why the validation foundation is built the way it is, so
future model results can be judged against it. Permanent policy stays in
`AGENTS.md`; changing state stays in `docs/PROJECT_STATE.md`.


## Owner decisions (2026-10-06)

- **Execution timing.** ML labels enter at the close after the feature date:
  `MODEL_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1)`, so
  `y(t) = A(t+21) / A(t+1) - 1` on `adjusted_close`. End-of-day data arrives
  after the close (same-close execution is infeasible), and some macro series
  are published after the close on their release dates.
  `add_forward_return_target` keeps its historical default `entry_lag=0`.
- **Lockbox.** Data from 2025-01-01 onward is a final holdout. Development
  folds must end by it and never score labels that mature in it; holdout folds
  (`mode="holdout"`) run once per pre-registered model.


## Design

| Concern | Rule | Where |
|---|---|---|
| Fold scheme | Expanding, forward-only walk-forward over half-open `[test_start, test_end)` session windows | `folds.make_expanding_folds`, `make_test_windows` |
| Knowledge cutoff | A row trains iff `target_end_date < test_start` (strict: a label is known only after the close of its end date) | `folds` |
| Purge | Global across symbols: rows dated before `test_start` whose labels end on or after it are removed for every symbol | `folds` |
| Embargo | None (see below); the premise is asserted per fold | `checks.assert_fold_is_leakage_safe` |
| Lockbox | `MODEL_HOLDOUT_START` bounds every development fold; windows end by it; window rows whose labels mature in it are withheld | `folds`, `checks`, `harness` |
| Boundaries | Session dates at 00:00 UTC only; intraday boundaries are rejected | `folds` |
| Universe | Config assets only; benchmark index rows excluded | `folds.restrict_to_symbols` |
| Learned transforms | Fit on the fold's training rows only; fit-once objects; no estimator or component shared across folds | `preprocessing`, `harness` |
| Prediction batches | `predict` is called once per test date, so no estimator sees later dates' features | `harness` |
| Entry timing | Labels must enter after the feature date (`target_start_date > date`) | `folds`, `checks` |
| Features | Explicit allowlist; labels, keys, timestamps, price and volume levels rejected | `checks.validate_feature_columns` |
| Disguised leaks | Content check: features dated before T must not change when data at or after T is added | `checks.assert_prefix_stable` |
| Audit | Canonical SHA-256 fingerprints, per-fold metadata, run manifest | `audit` |

**Why no embargo.** An embargo removes training rows dated *after* a test
window whose backward-looking features overlap the test labels. Expanding
forward folds never train on rows after the window, and that is asserted per
fold. Any scheme that does train on later blocks (K-fold, CPCV, inner CV with
later training blocks) needs an embargo of at least the longest feature
lookback plus the entry lag. That is 21 sessions today, because
`momentum_20d` at row s equals the label of row s - 21 (pinned by
`test_trailing_return_feature_equals_an_earlier_label`).

**Why the purge is global.** The four long-history equities have a mean
pairwise correlation of about 0.46 over non-overlapping 20-day periods.
Another symbol's label over the test window carries information about the
test labels.


## Leakage register

| Source | Status |
|---|---|
| Overlapping labels crossing into the test window | Mitigated: strict global purge |
| Legacy `split_temporal_dataset` (no purge) | Documented: its docstring warns. With entry lag 1 it puts 84 training rows with 2025+ labels into training |
| Same-close execution | Mitigated for ML labels (entry lag 1). The backtest engine still executes at the same close; it gets a matching lag in the model ticket |
| Macro releases after the close (fed funds, pre-2021 weekly M2) | Removed by entry lag 1. Macro `available_at` is date-level |
| Intraday cutoffs admitting unmatured labels | Mitigated in the new folds. Legacy `select_training_rows_asof` documents it |
| Price and volume levels (vendor back-adjustment) | Mitigated: rejected by the feature contract |
| Benchmark index rows in the asset panel | Mitigated in validation (`restrict_to_symbols`). `build_model_dataset` still includes them (data-contract follow-up) |
| Full-sample scalers, imputers, selection | Mitigated: fit-in-fold, fit-once, fresh per fold |
| Test-batch statistics | Mitigated: tested invariant that each prediction depends only on training rows and its own features |
| Lockbox contamination via late-2024 labels | Mitigated: withheld in development mode |
| Selection and survivorship of the universe (hindsight-chosen semiconductors) | **Open.** Validation cannot fix it; every report must state it |
| Hyperparameter selection on unmatured scores | **Open (model ticket).** Nested folds use `holdout_start = outer test_start` |
| Multiple testing across variants | **Open (model ticket).** Needs a variant ledger and a deflated Sharpe ratio |


## Evidence (2026-10-06)

- **Mutation testing:** 34 of 34 deliberate violations were killed by the
  suite. Each mutant ran in an isolated copy of the source, against the new
  tests plus the existing label tests. They cover:
  - purge boundary, basis, and scope;
  - lockbox breaches in the builder, checker, and harness;
  - same-close labels;
  - non-UTC or intraday boundaries;
  - estimator and component reuse;
  - fitting on test rows;
  - test-batch statistics, and whole-window prediction;
  - outcome-selected test sets;
  - fingerprint and binding weaknesses;
  - per-symbol label shifts.
- **Synthetic canary:** a nearest-key memorizer on independent random walks
  (5 seeds) has |corr| < 0.15 through purged folds and corr > 0.5 without
  purging. The harness refuses the unpurged fold.
- **Real-data canary:** same-symbol memorizer, 24 development folds of 63
  sessions, 2019 through 2024. Purged: corr -0.014, hit 47.8%. Unpurged:
  corr +0.190, hit 60.1%.
- **Prefix stability:** the production feature SQL is prefix-stable at 5
  real cutoffs; the macro builder is prefix-stable by `available_at` on
  synthetic vintages.


## Next ticket: Ridge vs Lasso vs Elastic Net

**Setup.** Identical outer folds, identical in-fold preprocessing (median
impute, then standardize). Nested selection runs inner purged forward folds
built from each outer training set with `holdout_start = outer test_start`.
A short, pre-registered hyperparameter grid; every variant recorded.

**What each baseline teaches.**
- Ridge: whether there is diffuse, dense signal, and how stable the
  coefficients are under correlated features.
- Lasso: which features survive. Selection frequency across folds is the
  evidence; an empty model at the chosen penalty is a valid null result.
- Elastic Net: grouped selection among correlated features, at the cost of
  a second hyperparameter and more selection noise.
- Expect the three to be statistically indistinguishable. Judge each against
  the null models and on stability, not against each other.

**Null models.** Zero forecast, per-symbol expanding mean, unfitted momentum
rank, and a block-permutation null.

**Predictive metrics.**
- Out-of-sample R² against the zero forecast.
- Mean per-date rank IC with Newey-West standard errors (lag at least 21) or
  non-overlapping subsamples.
- Calibration slope.
- Clark-West test against the null.
- Fraction of folds with positive IC.

**Portfolio metrics.**
- One pre-declared forecast-to-weight rule, the same for all models.
- 20-session rebalancing, entry lag 1 in the backtest, 10 bp costs.
- Net return, volatility, Sharpe, maximum drawdown, turnover, cost drag.
- Information ratio against equal weight of the same universe, plus the
  inverse-volatility and momentum baselines.

**Decision rule.** A model "adds value" only if it beats the nulls on
HAC-adjusted IC and net information ratio across development folds.
Multiple-testing control comes from the variant count. Only then does it
earn one holdout evaluation.

**Power caveat.** The minimum detectable rank IC is about 0.12 (quant
review). Plausible true ICs are 0.02-0.05, so "no evidence" must not be
reported as "no effect".

**Owner decisions needed.**
- Raw vs excess-over-equal-weight label (excess recommended).
- The scikit-learn dependency.
- The learning-core split: the owner implements the estimator core; Claude
  provides the harness wiring, metrics, and tests.
