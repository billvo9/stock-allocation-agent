# Linear models with nested purged selection (roadmap ticket T4)

Status (2026-10-10): **phase A done** on `feature/linear-model-baselines`. That covers the plumbing, the excess label, contract 1.2, the dashboard, and a bounded skeleton of the estimator core with its tests.

**Next: the owner exercise below.** After it, phase C:
- the real development run;
- independent reviews;
- results and a checkpoint;
- the pull request.

Code:
- `features/training.py` (excess label)
- `model_diagnostics/nested.py` (selection plumbing)
- `model_diagnostics/linear.py` (owner core)
- `model_diagnostics/scoring.py` (Clark-West)
- `model_diagnostics/runner.py` (integration and checks)
- `dashboard/` (Linear models page)

## Owner decisions (2026-10-10)

- **Primary target.** The point-in-time cross-sectional equal-weight excess forward return. The raw forward return is stored beside it, in `inputs.target_return_raw`; every metric is computed on the excess label.
- **Dependency.** scikit-learn, pinned.
- **Models.** Ridge, Lasso and Elastic Net, with hyperparameters chosen only by nested purged temporal validation inside each outer training fold. OLS is an optional diagnostic comparator.
- **Evaluation.** Reuse T2's metrics, inference, nulls and controls, and the T3 dashboard. No lockbox. Four names are not product evidence.
- **The per-symbol mean** is reported as a research warning and control, never a hard gate.
- **Clark-West** only for truly nested comparisons.
- **Implementation split.** The owner implements the estimator core.

## Label: equal-weight excess return

For date t, C_t is the set of universe names present on t:

    target_return     = raw_i,t - (1/|C_t|) * sum over j in C_t of raw_j,t
    target_return_raw = raw_i,t

- **Availability (point-in-time).** The excess depends on every member's label, so every row on t takes:
  - `target_end_date` = the latest member end;
  - `target_start_date` = the earliest member start.

  Purge (train iff `target_end_date < test_start`), lockbox evidence, `score_available_at` and the dashboard's maturity rule then hold unchanged.
- **Fail closed.**
  - If any present name has no finite raw label (for example a missing exit price), or fewer than `min_cross_section` (2) names are present, every row on t is unlabeled. A missing price can never shrink the benchmark, and truncation never changes a matured label (tested).
  - A symbol outside the universe is refused, so index rows can never enter the benchmark.
- **Consequences.**
  - Rank IC, hit counts and rank positions use within-date order, so rank IC equals the raw label's.
  - On the real development frame (excess-label run of 2026-10-10, scratch output, not committed) per_symbol_mean's rank IC is +0.1263 with fold-block t 2.29, exactly as under the raw label.
  - Squared errors refer to excess returns.
  - Under this label `pooled_mean` is identically zero: complete cross-sections sum to zero. On the real frame its largest absolute prediction is 6.7e-19. So zero is the null comparator, and pooled_mean is dropped as a comparator (the runner refuses it).
- **Pre-training cross-check** (also on the excess frame): every fold's latest test-label end must be on or before the evaluation as-of, `lockbox_evidence.max_date`. Otherwise `run_diagnostics` raises before any estimator is built.
  - Real frame: as-of 2024-12-31 = max `score_available_at`, with all 24 folds equal to their latest test-label end.

## Nested purged selection

**Inner folds.** For each outer fold, built once and reused for every grid point, permutation draw and stale-feature run:
- `make_expanding_folds(frame, windows, holdout_start = outer test_start, train_start = outer train_start)`.
- Windows: the last 4 blocks of 63 sessions before the outer test start, with at least 252 training sessions.
- An inner training row's label ends before its inner test start. An inner test row whose label ends at or after the outer test start is held out.
- Every inner row is asserted to be an outer training row whose label ends before the outer test.
- Forward-only folds need no embargo (`folds.py`).

**Selection.**
- Each grid point runs once through the validation harness on the inner folds.
- The owner's `select_hyperparameters` scores every point and picks one.
- Then ONE harness call refits all outer folds; its factory hands each fold its chosen scalars, so the harness's cross-fold guards stay in force.
- Estimators receive scalars only, never a grid.
- Selection is rerun inside every permutation and stale-feature draw, so those nulls include the tuning.

**Pre-registered grids.** These live in each model's params, so any change is a new ledger variant, and they are dimensionless:

| Model | Grid | Penalty used |
|---|---|---|
| Ridge | c in {1e-3, 1e-2, 1e-1, 1, 10, 100} | alpha = c * n_fit |
| Lasso | r in {1, .5, .2, .1, .05, .02, .01} | alpha = r * lambda_max |
| Elastic Net | the same r × l1_ratio in {0.2, 0.5, 0.8} | alpha = r * lambda_max / l1_ratio |

- lambda_max = max_j \|x_j'(y - mean y)\| / n_fit on the fit's own standardized rows: the smallest penalty that zeroes every coefficient.
- Criterion: the mean over inner test dates of the per-date cross-sectional MSE (each date weighs the same, as in the Goyal-Welch view).
- Tie rule: the larger penalty, then the larger l1_ratio.

**OLS (role `diagnostic`, never eligible).** The same three standardized features and no penalty. It is a comparator for the candidates and nests the zero forecast.

## Clark-West (registered nested pairs only)

    f_it = (y - b)^2 - [(y - f)^2 - (b - f)^2] = 2 (y - b)(f - b)

- The per-date mean of f goes through T2's inference (fold-block t decides; Newey-West lag 40 covers the 21-session label overlap; bootstrap).
- The test is one-sided: a positive mean favours the larger model. Stored p-values are two-sided, so the info check `clark_west_one_sided` records t against t_0.95.
- **Pairs.**
  - Under the excess label: only (ols, zero).
  - Under a raw label, also (ols, pooled_mean): pooled_mean is OLS with every slope fixed at zero, fitted on the same purged expanding rows.
- Penalized and selected models are never nested in this sense. Their comparisons use the established loss-difference framework: `mse_improvement_vs_*` with fold-block t and bootstrap, plus Goyal-Welch.
- **Caveat.** T2's size simulation covers rank-IC series only, so the size of Clark-West and of the MSE-difference tests is not simulated.

## Checks added

| Check | Severity | Meaning |
|---|---|---|
| `evaluation_asof_matches_folds` | leakage | Verified before training. |
| `nested_selection_inside_training` | leakage | Per candidate: every outer fold's inner folds were verified. |
| `selection_fits_converged` | instrument | No outer or inner fit stopped at `max_iter`. |
| `excess_label_centred` | instrument | Max \|pooled_mean\| ≤ 1e-12. |
| `clark_west_one_sided` | info | One-sided t against t_0.95; size not simulated. |
| `candidate_below_selection_control` | research_warning | Active when the candidate's decision-method `mse_improvement_vs_per_symbol_mean` ≤ 0. Reported, never a gate: on four hindsight names the per-symbol mean captures selection. |

## Output contract 1.2 (additive)

- **`coefficients`** (training scope): model, fold, feature (with `__intercept__`), standardized coefficient, fitted median, mean and scale, alpha, converged.
- **`selection`** (training scope): model, fold, grid point (penalty, l1_ratio), inner score, inner dates and folds, inner window bounds, latest inner label end, chosen.
- **New role value** `diagnostic`.
- Tables are versioned: 1.0 and 1.1 runs stay readable, and the dashboard shows their missing tables as labelled gaps. It re-verifies from stored rows that:
  - inner labels end before each outer test;
  - every fold's test labels end by the evaluation as-of;
  - the new date columns are inside the lockbox scan.

## Dashboard

- **New Evidence page "Linear models"**, from the T3.1 components:
  - stored estimate cards (rank IC, d_t vs zero, d_t vs the selection control);
  - chosen penalty by fold;
  - inner scores (heatmap);
  - coefficient paths;
  - a descriptive stability table (selection frequency, sign consistency);
  - fitted transform parameters;
  - Clark-West labelled "nested: ols vs zero";
  - the selection-control warning.
- **Candidate identity in the design system** (`theme.CANDIDATE_COLOURS`), each ≥ 4.5:1 on white and disjoint from the T2 model palette:

  | Model | Colour | Marker |
  |---|---|---|
  | ridge | `#6A3D9A` | diamond (candidate) |
  | lasso | `#1B7F79` | diamond (candidate) |
  | elastic_net | `#B15928` | diamond (candidate) |
  | ols | `#5E6B2B` | triangle (diagnostic) |
- **Defaults.** Candidates are selected and preferred by default where models are compared.

## Running it (phase C)

```bash
.venv/bin/python scripts/run_null_diagnostics.py --with-linear-models
```

- Development mode, excess label by default. Add `--label raw` for the raw label.
- Before choosing the permutation draw count for candidates, time one pass. 49 draws is the default; 19 still resolves p = 0.05.

## Verification (phase A)

- Tests: the excess label (`test_excess_label.py`), nested plumbing, Clark-West and checks (`test_linear_models.py`), contract 1.2 and compatibility with 1.0 and 1.1 runs, and the dashboard on a T4 run (`test_dashboard_linear.py`). Integration uses statistics-free doubles in place of the owner core.
- The owner-core contract (`test_linear_core.py`, 12 strict xfails) was validated both ways before hand-off: a correct implementation passes all 12, and three plausible mistakes each fail at least one. The three mistakes were observation-weighted selection, sample standard deviation and a reversed tie rule.
- Mutation probes: 18 deliberate violations, each in an isolated copy, all caught:
  - excess availability, incomplete dates, the universe check;
  - inner verification and `train_start`, the chosen point, the intercept row;
  - the Clark-West adjustment and its pairs;
  - the excess and pooled-mean refusal, the pre-training as-of check, the selection-control check direction;
  - the loader's three invariants, candidate colours, table versioning, partial runs.

## Owner exercise

Implement `src/stock_agent/model_diagnostics/linear.py`. Everything around it is done and tested with statistics-free doubles.

1. **`LinearForecast.fit` / `predict`**:
   - median-impute and standardize on the fit rows (population std; a scale of ~0 → 1.0);
   - compute the penalty scale (formulas above);
   - fit sklearn `Ridge`, `Lasso` or `ElasticNet` (`selection="cyclic"`, the given `tol` and `max_iter`) or `LinearRegression`;
   - set `medians_`, `means_`, `scales_`, `coef_`, `intercept_`, `alpha_` and `converged_` (Lasso and Elastic Net: `n_iter_ < max_iter`);
   - refuse a second fit (RuntimeError);
   - `predict` = `intercept_ + X_std @ coef_`, row-wise.
2. **`select_hyperparameters`**:
   - date-normalized inner MSE per grid point;
   - the lowest wins, with the tie rule;
   - return a `Selection` with every score and date count.
3. **Tests.** In `tests/test_linear_core.py`, remove the `@OWNER` marker from each test you make pass. A strict xfail that starts passing fails the suite, so the markers cannot be forgotten.
4. **Gate.** `python3 scripts/verify.py`. Then ask for the core review (data-science reviewer) and phase C.

Hints:
- Standardize before computing lambda_max.
- `LinearRegression` already centres.
- sklearn's Lasso and ElasticNet objectives divide the squared error by 2n; Ridge does not.
- `np.nanmedian` handles imputation statistics.

## Dependency review: scikit-learn 1.9.1

- **Value.** Mature, well-tested solvers for Ridge (closed form), Lasso and Elastic Net (coordinate descent), with convergence reporting. Writing coordinate descent by hand would add risk without teaching value beyond the exercise.
- **Alternatives.** A numpy closed form handles Ridge and OLS only. statsmodels has weaker Elastic Net support and a heavier API. glmnet bindings are not maintained for Python 3.12.
- **Cost.**
  - New packages: scikit-learn, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0 and cloudpickle 3.1.2. narwhals was already present.
  - About 60 MB installed.
  - All have cp312 wheels.
  - Fits on 3 features are milliseconds; BLAS threads are limited to 1 for reproducible sums.
- **Security.** BSD-3 (scikit-learn, joblib, threadpoolctl, cloudpickle) and BSD (scipy) licences, installed from PyPI. No network access at run time. scikit-learn 1.9.1 was released 2026-09-10.
- **Deployment.** Pinned in `requirements.txt`, so CI installs the same version. Transitive packages are unpinned, like numpy and pandas.

## Learning Notes

1. **Problem.** Fit regularized linear forecasts of the cross-sectional excess return, and choose their penalty without ever looking at the period being evaluated.
2. **Concept.**
   - Ridge shrinks every coefficient toward zero (an L2 penalty).
   - Lasso can set coefficients exactly to zero (an L1 penalty: selection).
   - Elastic Net mixes the two, which stabilizes Lasso when features are correlated.
   - Nested validation: the outer folds estimate performance; inner folds inside each outer training set choose the penalty. Using the outer test to choose would make the reported performance optimistic.
3. **Assumptions.**
   - A linear relation between standardized features and excess returns.
   - Stationarity within each training window.
   - Squared-error loss as the selection criterion.
   - The penalty grids span the useful range (checked visually on the selection heatmap).
4. **Formulation.**
   - Ridge: min ||y - Xb||^2 + alpha ||b||^2.
   - Lasso: min (1/2n)||y - Xb||^2 + alpha ||b||_1.
   - Elastic Net: min (1/2n)||y - Xb||^2 + alpha (rho ||b||_1 + (1 - rho)/2 ||b||^2).
   - Clark-West: E[2(y - b)(f - b)] > 0 for a nested pair.
5. **Small example.** With standardized x and y = 0.5x + noise, OLS gives b ≈ 0.5. Lasso with alpha = 0.1 · lambda_max gives b ≈ 0.45 (soft-thresholded by alpha). At alpha = lambda_max, b = 0.
6. **Leakage risks.**
   - Choosing alpha on the outer test period. Prevented by inner folds inside the outer training set.
   - Inner validation labels overlapping the outer test. Prevented by the inner purge and holdout at the outer test start.
   - Scaling features with test rows. Prevented by per-fit standardization.
   - An excess benchmark that uses a later-maturing label. Prevented by the max-end availability rule.
   - A penalty scale computed from rows outside the fit. lambda_max comes from the fit's own rows.
7. **Misleading-result risks.**
   - Many grid points and three model families invite selection on noise: the variants are recorded, and the nulls rerun selection.
   - Correlated features make Lasso's selected set unstable: read the stability table, not single folds.
   - With four hindsight names the per-symbol mean "wins" on selection, not skill.
   - The Clark-West and MSE-difference sizes are not simulated.
   - Overlapping 20-day labels shrink the effective sample.
8. **Industry validation.**
   - Purged walk-forward evaluation with nested tuning.
   - Comparison against nulls and simple benchmarks after costs (T5).
   - Coefficient stability across time.
   - Clark-West or Diebold-Mariano tests for forecast comparisons.
   - A holdout used once (T6).
9. **What to learn next.**
   - The regularization path and degrees of freedom of Ridge.
   - The bias Lasso adds, and the relaxed Lasso.
   - Stability selection.
   - Why Clark-West's adjustment is needed (estimated zero parameters inflate the larger model's MSPE).
