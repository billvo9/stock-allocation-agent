# Diagnostics and visualization plan (ML phase)

Status: plan (2026-10-08), part of the ML roadmap.

Scope: the baseline linear models (Ridge, Lasso, Elastic Net), their null
models, and the dashboard that presents them.

It builds on `src/stock_agent/model_validation/` and changes none of its
contracts: folds, purge, lockbox, label timing, and the harness stay as they
are (`docs/research/model_validation.md`).


## Principles

1. **Data first, pictures second.**
   - Every diagnostic is a pure, tested function that returns a tidy table:
     `fold_id`, `model`, `scope`, `metric`, `value`, `ci_low`, `ci_high`,
     `n`, `n_eff`, `status`, `reason`.
   - Figures are pure functions of those tables.
   - The dashboard only reads saved run outputs; it never computes or fits
     anything.
   - This follows the AGENTS.md dashboard direction: structured diagnostics,
     not log text.
2. **Every diagnostic says which data it used.** Each one carries a `scope`:
   - `train`: one fold's training rows. These are the only diagnostics
     allowed to inform choices inside that fold.
   - `test`: out-of-sample rows, for evaluation only.
   - `development`: exploration across the development period
     (before 2025-01-01).

   Nothing reads the lockbox except a recorded holdout run.
3. **No hidden feedback loop.** A diagnostic may change a fold's fit only if
   it is computed from that fold's training rows. For example, dropping a
   collinear feature is decided per fold on training rows, or decided once
   from development exploration and recorded as a new variant.
4. **Always show uncertainty.**
   - Use Newey-West standard errors with lag ≥ 40, or a block bootstrap with
     63-session blocks. Check the false-positive rate by simulation before
     reporting: under a true null, lag 20 rejected 7-13% of the time at a
     nominal 5% (`docs/research/universe_and_market_context.md`).
   - Show the effective sample size beside every estimate. Adjacent
     overlapping labels share 19 of 20 returns, so naive standard errors are
     about √20 too small.
5. **Null models first.** The same diagnostics run on:
   - a zero forecast;
   - a per-symbol expanding mean;
   - an unfitted momentum rank;
   - a block-permutation null;
   - the memorizer canary.

   They must show no skill on these, and must recover planted signal in
   synthetic data, *before* any learned model is judged.
6. **Be honest about the small universe.** Development folds have only 4
   symbols, so per-date cross-sectional statistics are coarse:
   - A Spearman correlation across 4 names can only take 11 values, from −1
     to 1 in steps of 0.2.
   - Quintile buckets across names are impossible. Use rank positions 1–4
     per date, and pooled prediction deciles.


## Foundation: run outputs and a run ledger

Every evaluation run writes to `reports/runs/<run_id>/` (already ignored by
Git):

| File | Contents |
|---|---|
| `manifest.json` | `run_manifest` plus the audit gaps found in the pre-merge review: dirty-tree flag; estimator spec, hyperparameters, and seeds; variant id; lockbox mode; versions of duckdb, pyarrow, yfinance, and edgartools; platform; raw-data hashes and retrieval times |
| `folds.parquet` | `describe_folds` output |
| `predictions.parquet` | harness predictions (fold, date, symbol, label timing, prediction) |
| `fit_params.parquet` | fitted imputer medians and scaler means and scales, per fold |
| `coefficients.parquet` | standardized coefficients per fold, feature, and penalty |
| `diagnostics/*.parquet` | the tables from sections 1–5 |

- `run_id` is the SHA-256 of the manifest, so identical inputs give the same
  id.
- A separate append-only ledger (`reports/ledger.parquet`) records:
  - every run (run id, variant, mode, git SHA, and an injected timestamp);
  - the variant count, which feeds multiple-testing corrections;
  - holdout runs. A second holdout run of the same pre-registered model is
    refused.

**Code layout.**
- `src/stock_agent/model_diagnostics/` holds the computations (numpy and
  pandas only): `data`, `collinearity`, `coefficients`, `residuals`,
  `ranking`, `inference`, `artifacts`.
- Figures and the app live in `src/stock_agent/dashboard/`.
- Diagnostics never import the dashboard.


## 1. Feature distributions and missingness

**What is computed.** Per feature, fold, and scope:
- count and missing rate;
- mean, standard deviation, skew, excess kurtosis;
- quantiles at 1, 5, 25, 50, 75, 95, 99;
- the share of values that were imputed.

Also, per symbol and per year:
- **Train-vs-test drift:** population stability index (PSI) and
  Kolmogorov-Smirnov distance on values standardized with the fold's
  training statistics.
- **Missingness maps** (feature × date and feature × symbol), each gap
  labeled with a reason code:
  - before listing;
  - lookback warm-up;
  - macro not yet released;
  - EDGAR filing not yet available;
  - provider gap.

**Why it matters.**
- Training-scope statistics are exactly what the fold's transforms learn.
- Test-scope statistics are for drift monitoring only.
- Missingness here is often informative rather than random (release lags,
  listings), which argues for missing-value indicator features rather than
  silent imputation.

**Charts.** Train vs test distributions per fold (small multiples), a
missingness heatmap, drift bars with PSI guide lines at 0.1 and 0.25, and an
availability timeline per feature.

**Tests.** Synthetic panels with known moments and planted missing blocks.
Assert the reason codes. Assert that no lockbox rows enter any non-holdout
scope.


## 2. Multicollinearity

**What is computed.** Per fold, on training rows *after* imputation and
standardization (exactly what the model sees):
- Pearson and Spearman correlation matrices.
- Variance inflation factors, VIF_j = 1 / (1 − R_j²), computed with numpy
  least squares.
- The condition number κ = sqrt(λ_max / λ_min) of the standardized design
  matrix, and its eigenvalue spectrum.
- Hierarchical clustering of features by |correlation|.
- A **date-vs-within split** of each feature's variance:
  - the date component (cross-sectional mean);
  - the within-date component.

**What to expect.**
- `momentum_20d` already contains the same day's return, so it overlaps
  `daily_return`.
- `volatility_20d` co-moves with return magnitude.
- **Macro features are pure date components:** every symbol has the same
  value on a given date.
  - They can only explain market timing.
  - With an excess-over-equal-weight label they carry no information at
    all, unless interacted with a symbol characteristic.

**Guide lines, not rules.** VIF > 10, κ > 30.

**Why it matters.** Collinearity explains:
- unstable coefficients (section 3);
- Lasso picking arbitrarily among correlated features;
- Ridge shrinking correlated groups together.

**Charts.**
- Clustered correlation heatmap with a fold slider.
- VIF bars per fold.
- Scree plot of the eigenvalue spectrum.
- Date-vs-within variance shares.

**Tests.**
- VIF matches the closed form on planted collinear data.
- κ = 1 for an orthogonal design.
- The date and within shares sum to the total.


## 3. Coefficient stability across folds

**What is computed.**
- Standardized coefficients per fold and feature. They are comparable across
  folds because each fold's own training standardization is applied.
- The chosen penalty (α, and the L1 ratio for Elastic Net) per fold over time.
- Sign consistency: the share of folds with the most common sign.
- Coefficient variation across folds.
- Lasso and Elastic Net selection frequency: the share of folds in which the
  coefficient is nonzero.
- Regularization paths per fold (coefficient vs log α), with the chosen α
  marked.
- Within-fold block-bootstrap intervals (63-session date blocks), and
  bootstrap selection probability (stability selection).
- Ridge effective degrees of freedom: df(α) = Σ dᵢ² / (dᵢ² + α), where dᵢ are
  the design's singular values.

**How to read it.**
- Stable small coefficients mean diffuse signal.
- Sign flips and low selection frequency mean no reliable signal, whatever
  the backtest shows.

**Leakage note.**
- Penalties are chosen by inner purged folds built from the outer training
  set, with `holdout_start` set to the outer test start.
- Stability is measured across outer folds, is reported only, and is never
  fed back into fitting.

**Charts.**
- Coefficient trajectories across folds, with confidence bands.
- A feature × fold selection heatmap.
- Small-multiple regularization paths.
- The chosen penalty over time.

**Tests.** On synthetic data with one planted stable feature and pure-noise
features, the planted feature keeps its sign and the noise features are
rarely selected.


## 4. Residual and calibration diagnostics

**What is computed.** On out-of-sample test rows, per fold and pooled:
- **Residuals** e = y − ŷ:
  - QQ plot against a normal (fat tails);
  - residual vs prediction;
  - residuals by symbol, by fold, and by volatility regime (terciles
    defined on training rows);
  - residual autocorrelation per symbol at lags 1–40. Overlapping 20-session
    labels make lags up to 19 correlated by design, so show that expected
    band.
  - heteroskedasticity against `volatility_20d`.
- **Mincer-Zarnowitz calibration**: regress y = a + b·ŷ, with Newey-West
  confidence intervals. Ideal is a = 0, b = 1. A slope b < 1 means
  overconfident forecasts; b ≈ 0 means no signal.
- **Reliability diagram**: pooled prediction deciles, mean realized vs mean
  predicted, with confidence intervals.
- **Out-of-sample R²** against a zero forecast and against the expanding
  mean (Campbell-Thompson), per fold and cumulative.
- **Clark-West test** against the null models.
- **Goyal-Welch plot**: the cumulative difference in squared error between
  the null and the model over time. It shows *when* a model adds value, not
  just whether it does on average.

**Tests.**
- The slope is recovered on synthetic forecasts of known quality.
- The reliability bins are exact.
- The Newey-West standard error matches a Monte Carlo SD under simulated
  overlapping-label errors.


## 5. Rank IC, hit rate, and bucket performance

**What is computed.**
- **Cross-sectional rank IC per date** (Spearman across the eligible names):
  - mean IC, its Newey-West t-statistic (lag ≥ 40, calibrated), and IC information
    ratio;
  - a rolling 63-session mean, cumulative IC, and the share of dates with
    IC > 0.
  - This is the primary metric for an excess-return label.
- **Pooled time-series IC**, reported separately. When the names move
  together, it mostly measures market timing.
- **Hit rate:**
  - sign agreement, compared with the base rate (the share of positive
    labels), not with 50%;
  - "rank hit": whether the top-ranked name beats the median name.
- **Buckets:**
  - mean realized 20-session return by per-date rank position (1–4), with
    confidence intervals;
  - the top-minus-bottom spread and a monotonicity check;
  - pooled prediction deciles.
- **Overlap-free view:** sample every 21st session and average over the 21
  possible offsets.
- **Breakdowns** by fold, by symbol, and by volatility regime.

**Charts.**
- IC time series with rolling mean and zero line.
- IC distribution per fold.
- Cumulative IC.
- Bucket bars with confidence intervals.
- Decile monotonicity.
- Hit rate vs base rate.

**Tests.**
- A perfect forecast gives IC = 1; a reversed forecast gives −1.
- Random forecasts give IC ≈ 0 within tolerance.
- The Newey-West standard error is right under a simulated overlap
  (MA(19)) error.


## 6. Interactive dashboard (Plotly + Streamlit)

**Architecture.**
- A thin, read-only Streamlit app over run outputs.
- It never fits a model and never reads raw data.
- It shows lockbox results only for runs whose manifest mode is `holdout`.
- Plotly figure builders are pure and unit-tested; the app code only picks
  runs and folds and lays out figures.
- It runs on localhost only.

**Pages.** Each is built in the ticket that produces its data, not at the end.

| Page | Content |
|---|---|
| 0. Run overview | Manifest (git SHA, dirty flag, label spec, lockbox, universe, features, data hashes), gate status, variant count and ledger. A "leakage health" banner turns red if the canary or null models show skill. |
| 1. Fold construction | A timeline per fold showing the training span, the 21-session purge band, the test window, held-out rows, and lockbox shading. Rows per symbol per fold, knowledge cutoffs, purged counts, listing events (SNDK). |
| 2. Data diagnostics | Section 1. |
| 3. Multicollinearity | Section 2. |
| 4. Training diagnostics | Section 3: coefficient stability, regularization paths, chosen penalties, fitted transform parameters. |
| 5. Validation results | Sections 4–5, each model beside the null models; per fold and pooled; breakdowns by symbol and regime. |
| 6. Model comparison | Ridge vs Lasso vs Elastic Net vs nulls: IC, out-of-sample R², Clark-West, calibration slope, stability. Holm correction across model families; deflated Sharpe ratio using the ledger's variant count. |
| 7. Portfolio readiness | Pre-declared forecast-to-weight rule; net equity vs equal weight, inverse volatility, and momentum; drawdown, turnover, and cost drag; concentration. Checks that the label's entry lag equals the backtest's execution lag. A ready / not-ready checklist with explicit criteria. |
| 8. Holdout | Locked until a recorded holdout run exists. Then one evaluation against the pre-registered decision rule. |

**Presentation rules** (frontend-engineer standards):
- Every chart states its scope and knowledge time.
- Units and dates are labelled.
- Uncertainty bands are shown wherever an estimate is.
- No dual axes.
- A colorblind-safe palette, with each model in a consistent color.
- Cached reads of run outputs.

**Tests.**
- Figure builders: assert the traces, axis titles, and data counts.
- App smoke test: Streamlit's `AppTest` loads every page against a fixture
  run directory.
- No network access.


## Dependencies (owner approval required)

| Package | Value | Alternatives | Cost and risk |
|---|---|---|---|
| plotly | Interactive, publication-quality charts that also export to standalone HTML | matplotlib (present, static), altair | Moderate size, stable API; no server |
| streamlit | Fast multi-page interactive app in Python | Dash (more code), Panel, static Plotly HTML reports (no server) | Large dependency tree (tornado, protobuf, altair, ...), frequent releases. Runs a local web server: bind to localhost and never expose it without authentication. Later AWS hosting would need an authentication proxy. |
| scikit-learn | Ridge, Lasso, Elastic Net, coordinate descent | Owner-implemented closed-form Ridge and coordinate descent (a learning exercise) | Separate decision, made in the linear-model ticket |

**Recommendation.** Pin plotly and streamlit in `requirements.txt` so that
`scripts/verify.py` (local and CI) tests the figure builders and runs the
app smoke test. Splitting them into an optional install would leave the
dashboard untested in CI. A fallback with no server is static Plotly HTML
reports, adding Streamlit later.


## Roadmap

This replaces "Next ticket: Ridge vs Lasso vs Elastic Net" in
`docs/research/model_validation.md`. Each ticket ships its dashboard views.

| Ticket | Delivers | Dashboard | Gates |
|---|---|---|---|
| T1 (pending merge) | Validation foundation | — | — |
| T2 | `model_diagnostics` (sections 1, 2, 4, 5 and inference), run outputs and ledger, null models and canary through the harness. Accepted when the nulls show no skill and planted signal is recovered. | — | None. numpy and pandas only. Owner exercise: Newey-West standard errors. |
| T3 | Plotly figure builders and the Streamlit app, run on null models | Pages 0, 1, 2, 3, 5 | Dependency approval |
| T4 | Ridge, Lasso, Elastic Net with nested purged selection; coefficient outputs; section 3 | Pages 4, 6 | scikit-learn decision; label choice; learning-core split (owner implements the estimator core) |
| T5 | Portfolio readiness: backtest execution lag 1, forecast-to-weight rule, cost-aware evaluation | Page 7 | Financial-formula review |
| T6 | Holdout evaluation, once per pre-registered model | Page 8 | Pre-registration recorded |
| Parallel: U and C tracks | Point-in-time universe (about 100 large caps from 2010) and market-context layer V1, per `docs/research/universe_and_market_context.md` | All pages gain cross-sectional power; context ablation views | Data source and licence, `decision_at`, data-contract decisions. Both gate whether T4 results count as product-grade evidence |

**Why diagnostics come before models (T2 and T3 before T4).**
- Every diagnostic and chart is validated on models whose correct answer is
  known (no skill).
- Fold construction and the data become visible before anything is fitted.
- When Ridge arrives, its results are read through instruments already
  trusted.


## Decisions for the owner

1. Add plotly and streamlit (pinned, in `requirements.txt`), or begin with
   static Plotly HTML reports.
2. Raw vs excess-over-equal-weight label. This sets the primary metric in
   section 5 and decides whether macro features can matter at all (section
   2's date component).
3. scikit-learn and the learning-core split (T4).
4. Default run-output location `reports/runs/` (ignored by Git) and a
   localhost-only dashboard.
