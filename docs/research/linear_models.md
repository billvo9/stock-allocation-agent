# Linear models with nested purged selection (roadmap ticket T4)

Status (2026-10-11): **phases A, B and C done** on `feature/linear-model-baselines`; merge pending.
- Phase A: the plumbing, the excess label, contract 1.2 and the dashboard.
- Phase B: the estimator core.
- Phase C: the real development run, the independent reviews and the results below.

Code:
- `features/training.py` (excess label)
- `model_diagnostics/nested.py` (selection plumbing)
- `model_diagnostics/linear.py` (estimator core)
- `model_diagnostics/scoring.py` (Clark-West)
- `model_diagnostics/runner.py` (integration and checks)
- `dashboard/` (Linear models page)

## Owner decisions

**2026-10-10**
- **Primary target.** The point-in-time cross-sectional equal-weight excess forward return. The raw forward return is stored beside it, in `inputs.target_return_raw`; every metric is computed on the excess label.
- **Dependency.** scikit-learn, pinned.
- **Models.** Ridge, Lasso and Elastic Net, with hyperparameters chosen only by nested purged temporal validation inside each outer training fold. OLS is an optional diagnostic comparator.
- **Evaluation.** Reuse T2's metrics, inference, nulls and controls, and the T3 dashboard. No lockbox. Four names are not product evidence.
- **The per-symbol mean** is reported as a research warning and control, never a hard gate.
- **Clark-West** only for truly nested comparisons.

**2026-10-11: implementation delegated.** The owner explicitly authorized implementing the remaining estimator core (preprocessing, the Ridge, Lasso and Elastic Net wrappers, the prediction contract and nested hyperparameter selection), superseding the earlier decision that the owner implements it. The learning objective is kept through the explanations and Learning Notes below and one optional exercise.

## Label: equal-weight excess return

For date t, C_t is the set of universe names present on t:

    target_return     = raw_i,t - (1/|C_t|) * sum over j in C_t of raw_j,t
    target_return_raw = raw_i,t

- **Availability (point-in-time).** The excess depends on every member's label, so every row on t takes:
  - `target_end_date` = the latest member end;
  - `target_start_date` = the earliest member start.

  Purge (train iff `target_end_date < test_start`), lockbox evidence, `score_available_at` and the dashboard's maturity rule then hold unchanged.
- **Fail closed.**
  - If any present name has no finite raw label (for example a missing exit price), or fewer than `min_cross_section` (2) names are present, every row on t is unlabeled. Truncation never changes a matured label (tested).
  - C_t is "universe names with a row on t". A missing exit price never shrinks it, but a name whose feature row is missing on t is not in C_t (see "Mandatory before large-universe validation").
  - A symbol outside the universe is refused, so index rows can never enter the benchmark.
- **Consequences.**
  - Subtracting the same date mean from every name leaves each date's ranking unchanged. So a *fixed* forecast scores the same rank IC under either label: per_symbol_mean is +0.1263 (fold-block t 2.29) under both on the real frame. A model *fitted* on the excess label learns different coefficients from a raw-label fit, so its rank IC can differ.
  - Squared errors, hit-rate base rates and rank-position levels refer to excess returns.
  - `pooled_mean` is identically zero under this label, because complete cross-sections sum to zero. On the real frame its largest absolute prediction is 6.7e-19. So zero is the null comparator, and the runner refuses pooled_mean as a comparator.
- **Pre-training cross-check.** Every fold's latest test-label end must be on or before the evaluation as-of, `lockbox_evidence.max_date`. Otherwise `run_diagnostics` raises before any estimator is built. Real frame: as-of 2024-12-31 equals the maximum `score_available_at`, and all 24 folds equal their latest test-label end.

## Nested purged selection

**Inner folds.** Built once for each outer fold and reused for every grid point, permutation draw and stale-feature run:
- `make_expanding_folds(frame, windows, holdout_start = outer test_start, train_start = outer train_start)`.
- Windows: the last 4 blocks of 63 sessions before the outer test start, with at least 252 training sessions. On the real frame each grid point is scored on 231 inner dates.
- An inner training row's label ends before its inner test start. An inner test row whose label ends at or after the outer test start is held out.
- Every inner row is checked to be an outer training row whose label ends strictly before the outer test start.
- Forward-only folds need no embargo (`folds.py`).

**Selection.**
- Each grid point runs once through the validation harness on the inner folds.
- `select_hyperparameters` scores every point, and the lowest score wins.
- Then ONE harness call refits all outer folds. Its factory hands each fold its chosen scalars, so the harness's cross-fold guards stay in force. Estimators receive scalars only, never a grid.
- Selection is rerun inside every permutation and stale-feature draw, so those nulls include the tuning.

**Pre-registered grids.** These live in each model's params, so any change is a new ledger variant:

| Model | Grid | Penalty used |
|---|---|---|
| Ridge | c in {1e-3, 1e-2, 1e-1, 1, 10, 100} | alpha = c * n_fit |
| Lasso | r in {1, .5, .2, .1, .05, .02, .01} | alpha = r * lambda_max |
| Elastic Net | the same r × l1_ratio in {0.2, 0.5, 0.8} | alpha = r * lambda_max / l1_ratio |

- lambda_max = max_j \|z_j'(y - mean y)\| / n_fit on the fit's own standardized rows. It is the smallest Lasso penalty that zeroes every coefficient.
- **What is scale-free and what is not.**
  - Ridge's c and Lasso's r are scale-free: rescaling the label rescales the coefficients exactly, and c does not depend on n.
  - Elastic Net's L1 strength, alpha × l1_ratio = r × lambda_max, equals Lasso's at the same r. Its L2 strength, alpha × (1 - l1_ratio), carries the label's units. So the L2 part is **not** scale-free.
  - Measured in review: fitting on 100·y moves Elastic Net's coefficients 4–14% beyond the expected ×100.
  - With labels in return units the L2 term is tiny. Elastic Net chose the same r as Lasso in 24 of 24 folds, and its rank IC, top-minus-bottom and hit rate equal Lasso's to four decimals.
  - Count it as a near-duplicate of Lasso when counting variants. Changing the parameterization would be a new ledger variant, and that is an owner decision.
- **Criterion.** The mean over inner test dates of the per-date cross-sectional MSE (each date weighs the same, as in the Goyal-Welch view).
- **Tie rule.** Scores within 1e-12 × max(1, best) of the best are tied: an absolute tolerance at these MSE scales (~1e-3). Among tied points, the larger penalty wins, then the larger l1_ratio. In the real run, exact ties occur only between Elastic Net points at r = 1, which all give the same all-zero model.

**OLS (role `diagnostic`, never eligible).** The same three standardized features with no penalty. It is a comparator for the candidates and nests the zero forecast: its intercept is the training label mean, ~1e-18 under the excess label.

## Estimator core (phase B)

`LinearForecast(method, penalty, l1_ratio, tol=1e-10, max_iter=100_000)`, fitted once per fold:
1. Median-impute each feature with the fit rows' medians.
2. Standardize with the fit rows' mean and population standard deviation (a scale of ~0 becomes 1.0).
3. Compute the penalty from the formulas above, on the fit's own standardized rows.
4. Fit sklearn `Ridge(solver="cholesky")`, `Lasso`, `ElasticNet` (both `selection="cyclic"`) or `LinearRegression`.
5. Record `converged_ = False` if the solver raised a ConvergenceWarning or used all `max_iter` iterations. Other warnings still surface.

`predict` imputes with the stored medians, standardizes with the stored means and scales, and returns `intercept_ + sum_j z_j * coef_j` row by row. The sum is a deliberate row-wise sum, not a matrix product: BLAS can change the summation order with the batch shape. So one date at a time equals the whole window exactly, which is tested with zero tolerance.

**Formula verification.**
- Tests derive every fold-specific value from that fold's eligible training rows only.
  - An adversarial test gives every non-training row extreme features and labels. That fold's coefficients, transform parameters and selection rows stay exactly equal, while a later fold does change.
  - Inner fits use their own lambda_max, not the outer fold's.
- Data-science review:
  - Ridge with alpha = c·n equals the closed form (Z'Z/n + cI)⁻¹ Z'y/n to 1e-12.
  - `lambda_max` equals the maximum of scikit-learn's own `lasso_path` / `enet_path` exactly, for l1_ratio 1, 0.5 and 0.2.
  - r = 1 gives exactly zero slopes.
- The 12 contract tests written in phase A (`tests/test_linear_core.py`) pass unchanged. Only their xfail markers were removed.

## Clark-West (registered nested pairs only)

    f_it = (y - b)^2 - [(y - f)^2 - (b - f)^2] = 2 (y - b)(f - b)

- The per-date mean of f goes through T2's inference: fold-block t decides; Newey-West lag 40 covers the 21-session label overlap; bootstrap. Each date weighs the same whatever its row count (tested).
- The test is one-sided: a positive mean favours the larger model. Stored p-values are two-sided, so the info check `clark_west_one_sided` records t against t_0.95. It is informational, never a success gate.
- **Pairs.**
  - Under the excess label: only (ols, zero).
  - Under a raw label, also (ols, pooled_mean).
- Penalized and selected models are never nested in this sense. They use `mse_improvement_vs_*` (fold-block t, bootstrap) and Goyal-Welch.
- **Caveat.** T2's size simulation covers rank-IC series only, so the size of the Clark-West and MSE-difference tests is not simulated.

## Checks added

| Check | Severity | Meaning |
|---|---|---|
| `evaluation_asof_matches_folds` | leakage | Verified before training (the run stops on a violation). |
| `nested_selection_inside_training` | leakage | Per candidate: folds whose stored selection rows show every inner label ending before the outer test start. |
| `selection_fits_converged` | instrument | No outer or inner fit stopped early. Undetermined if an estimator has no flag. |
| `excess_label_centred` | instrument | Max \|pooled_mean\| ≤ 1e-12. |
| `clark_west_one_sided` | info | One-sided t against t_0.95; size not simulated. |
| `candidate_below_selection_control` | research_warning | Active when the candidate's decision-method `mse_improvement_vs_per_symbol_mean` ≤ 0. Reported, never a gate: on four hindsight names the per-symbol mean captures selection. |

## Output contract 1.2 (additive)

- **`coefficients`** (training scope): model, fold, feature (with `__intercept__`), standardized coefficient, fitted median, mean and scale, alpha, converged.
- **`selection`** (training scope): model, fold, grid point (penalty, l1_ratio), inner score, inner dates and folds, inner window bounds, latest inner label end, chosen. Tests tie each row to what was actually scored and fitted.
- **New role value** `diagnostic`.
- Tables are versioned per table. 1.0 and 1.1 runs stay readable, and the dashboard shows their missing tables as labelled gaps. The dashboard re-verifies from stored rows that:
  - inner labels end before each outer test;
  - every fold's test labels end by the evaluation as-of;
  - the new date columns are inside the lockbox scan.

## Dashboard

- **New Evidence page "Linear models"**, built from the T3.1 components:
  - stored estimate cards (rank IC, d_t vs zero, d_t vs the selection control);
  - chosen penalty by fold;
  - inner scores (heatmap);
  - coefficient paths;
  - a descriptive stability table;
  - fitted transform parameters;
  - Clark-West labelled "nested: ols vs zero";
  - the selection-control warning.
- **Coverage is shown with every partial rank IC** (validation cards, Linear models card, rank-IC forest), e.g. "defined on 796 of 1489 scored dates, 13 of 24 folds".
- **Squared-error cards** (d_t, Clark-West) use scientific notation, and out-of-sample R² uses percent, so 1e-6 values never read "0.000".
- **The stability table's caption** states that expanding windows share most training rows, so cross-fold agreement is not independent evidence.
- **Candidate identity** (`theme.CANDIDATE_COLOURS`): each colour is ≥ 4.5:1 on white and disjoint from the T2 palette.

  | Model | Colour | Marker |
  |---|---|---|
  | ridge | `#6A3D9A` | diamond (candidate) |
  | lasso | `#1B7F79` | diamond (candidate) |
  | elastic_net | `#B15928` | diamond (candidate) |
  | ols | `#5E6B2B` | triangle (diagnostic) |

## Running it

```bash
.venv/bin/python scripts/run_null_diagnostics.py --with-linear-models
```

Development mode, excess label by default (`--label raw` for the raw label). The 2025+ lockbox is never loaded.

**Runtime benchmark (measured 2026-10-11, before choosing the draw count).**

| Step | Time |
|---|---|
| Inner folds for all 24 outer folds | 0.4 s |
| Main pass: ridge / lasso / elastic_net | 3.2 s / 3.5 s / 10.0 s (16.7 s total) |
| One ridge permutation pass (selection rerun) | 3.0 s |

Projected candidate null cost: 9 draws ≈ 3.3 min, 19 ≈ 6.1 min, 49 ≈ 14.4 min. 49 draws (T2's default) was affordable, so it was kept; the full runs took 15m15s and 17m48s.

**Reading the permutation p-values.** With 49 draws, p = (1 + exceedances) / 50, so the resolution is 0.02. The smallest possible p is 0.02, and p = 0.08 means 3 exceedances, with a 95% Clopper-Pearson interval of [0.013, 0.169] for the true p. These are development diagnostics, not confirmatory significance evidence. Use at least 199 draws before reading any p near 0.05.

## Development-run results

**Provenance.**
- Run `20261011T191112Z-50c1ca320e9c`, from commit `1338674` with a clean tree. Schema 1.2, run status WARNING, no record issues.
- Confirming run `20261011T222805Z-284566803c62`, from commit `99a8405` (after the review fixes) with a clean tree, took 17m48s. Its predictions, selection, coefficients, metrics, checks and null draws are identical to the first run's, value for value.
- Universe: INTC, MRVL, MU, NVDA (SNDK excluded: no pre-lockbox rows), **chosen with hindsight**.
- Evaluation: excess label, 24 outer folds with test windows starting 2019-01 to 2024-10, 1489 scored dates (5956 rows), as-of 2024-12-31.
- Everything in this section is **measured** from the stored tables. Rows marked post-hoc were computed afterwards from stored predictions, curves and metrics; they are not stored metric rows. Nothing here is illustrative.

**Out-of-sample results (fold-block t unless stated; d_t in units of 1e-4).**

| Model | Role | Mean rank IC [95%] | t | d_t vs zero | t | OOS R² vs zero | d_t vs per_symbol_mean | Permutation p (49) | Hit rate |
|---|---|---|---|---|---|---|---|---|---|
| ridge | candidate | +0.020 [−0.095, +0.135] | 0.36 | +0.013 | 0.04 | +0.02% | −3.44 (t −2.02) | 0.32 | 0.500 |
| lasso | candidate | +0.070 [−0.105, +0.246] **on 13/24 folds** | 0.88 | −0.011 | −0.03 | −0.02% | −3.47 (t −2.04) | 0.08 | 0.511 |
| elastic_net | candidate | +0.070 [−0.105, +0.246] **on 13/24 folds** | 0.88 | −0.011 | −0.03 | −0.01% | −3.47 (t −2.04) | 0.08 | 0.511 |
| ols | diagnostic | +0.018 [−0.097, +0.133] | 0.33 | −0.258 | −0.60 | −0.38% | −3.72 (t −2.30) | not run | 0.504 |
| per_symbol_mean | control | +0.126 [+0.012, +0.240] | 2.29 | +3.458 | 2.01 | +5.16% | — | 0.04 | 0.567 |
| momentum_20d | control (score) | +0.024 [−0.090, +0.137] | 0.43 | score only | | | | 0.30 | 0.503 |
| random_noise | null (score) | +0.015 [−0.020, +0.049] | 0.88 | score only | | | | not run | 0.509 |
| memorizer | canary | +0.014 [−0.129, +0.157] | 0.20 | −53.9 | −3.80 | −77% | −57.4 | not run | 0.477 |

- **Minimum detectable rank IC** (fold-block): ridge 0.163, OLS 0.163; lasso and elastic_net 0.245 (13 folds).
- **Ridge vs OLS** d_t: +0.27e-4 (t 1.39). Lasso vs OLS: +0.25e-4 (t 1.27).
- **Clark-West, (ols, zero):** +0.153e-4, fold-block t 0.37, below the one-sided critical value 1.714. Information only.
- **Stale-feature controls (ridge)** at lags 260, 290 and 330: rank-IC differences −0.059, −0.010 and −0.067 (|t| ≤ 1.07). For lasso they use only 292, 481 and 607 paired dates (df 4, 7, 9); do not read them.
- **Checks.**
  - All leakage and instrument checks pass:
    - nested selection inside training on 24 of 24 folds for each candidate;
    - the as-of matches the folds;
    - the excess label is centred;
    - every fit converged.
  - Research warnings: `control_apparent_skill` (per_symbol_mean), and `candidate_below_selection_control` for ridge, lasso and elastic_net.
  - The simulated-size info rows for Newey-West and the bootstrap fail as in T2.

**Unavailable or undefined statistics (recorded, not dropped).**
- **zero and pooled_mean.** Rank IC unavailable (`constant_prediction`). pooled_mean's d_t vs zero is unavailable (`zero_variance`): it is identically zero under this label.
- **lasso and elastic_net.**
  - Rank IC undefined on 693 of 1489 scored dates (`constant_prediction`), the 11 folds where selection chose the all-zero model: folds 0, 4–6 and 12–18.
  - The per-fold rows say `undefined`, while the per-date rows say `constant_prediction`; that per-fold reason gap is an existing T3 item.
  - Their pooled rank IC uses 796 dates with fold-block df 12.
- **Not run:** no permutation draws for OLS, random_noise and memorizer, and no Clark-West for any non-nested pair (by design).

**Post-hoc: why Lasso's 0.070 is not "better than Ridge's 0.020".**

| Model | IC, all 24 folds | IC, Lasso's 13 active folds | IC, Lasso's 11 all-zero folds |
|---|---|---|---|
| ridge | 0.020 | **0.068** | −0.037 |
| ols | 0.018 | **0.066** | −0.038 |
| momentum_20d (unfitted score) | 0.024 | **0.069** | −0.031 |
| lasso / elastic_net | n/a | **0.0705** | undefined |

- On the same folds, every linear model and the raw momentum score agree, so the gap is which dates are counted, not the model.
- Counting Lasso's constant dates as 0, its rank IC is 0.038 by fold blocks (t 0.88, 24 blocks) or 0.040 per date.
- The on/off switch did not time anything either. Momentum's IC differs by 0.100 between the active and inactive folds, with standard error 0.109 (t 0.92).

**Post-hoc: the permutation nulls are not centred at zero for Lasso.**
- The 49 null draws average −0.029 (SD 0.070) for Lasso and −0.030 for Elastic Net, against +0.002 (SD 0.043) for Ridge. All 49 draws are finite.
- So the stored "observed minus null" for Lasso is +0.105 (p 0.08), larger than the raw IC. Against a zero-centred null its p would be about 0.18–0.20.
- The cause of the negative centre is not established. One hypothesis, not tested: volatility levels carry symbol identity, and permuting them across blocks produces anti-learning.
- The block permutation also breaks the symbol–feature link, so it rejects pure selection too (per_symbol_mean p = 0.04). On this universe it cannot separate feature skill from hindsight selection.

**Post-hoc: the date-common part of each forecast only adds loss.**
- Each date's excess labels sum to zero. So the date's MSE splits exactly into a within-date error plus f̄_t², the square of that date's average forecast.
- The features are standardized over pooled rows, not demeaned within each date: 57–65% of each feature's variance is common to all names on a date (between-date share in the stored `inputs`: daily_return 0.65, momentum_20d 0.57, volatility_20d 0.58).

| Model | d_t vs zero (per-date mean) | Cost of the date-common part | Within-date advantage |
|---|---|---|---|
| ridge | +1.5e-6 | 7.3e-6 | +8.8e-6 |
| lasso | −1.0e-6 | 8.6e-6 | +7.6e-6 |
| ols | −2.6e-5 | 2.4e-5 | −1.8e-6 |

- The stored `cross_sectional_r2` is the cleaner measure here: ridge +0.13%, lasso +0.11%, OLS −0.03%.
- The inner MSE criterion pays the same cost, which pushes selection toward the zero model.

**Selected hyperparameters (chosen per outer fold).**

| Folds (test start) | ridge c | lasso r | elastic_net (r, l1_ratio) |
|---|---|---|---|
| 0 (2019-01) | 100 | 1 | (1, 0.8) |
| 1 (2019-04) | 10 | 0.5 | (0.5, 0.2) |
| 2 (2019-07) | 1 | 0.2 | (0.2, 0.2) |
| 3 (2019-10) | 100 | 0.5 | (0.5, 0.2) |
| 4–5 (2020-01, 2020-04) | 100 | 1 | (1, 0.8) |
| 6 (2020-07) | 10 | 1 | (1, 0.8) |
| 7 (2020-10) | 0.1 | 0.05 | (0.05, 0.2) |
| 8–11 (2020-12 to 2021-10) | 0.001 | 0.01 | (0.01, 0.8) |
| 12 (2021-12) | 100 | 1 | (1, 0.8) |
| 13 (2022-04) | 10 | 1 | (1, 0.8) |
| 14–17 (2022-07 to 2023-04) | 100 | 1 | (1, 0.8) |
| 18 (2023-07) | 10 | 1 | (1, 0.8) |
| 19–22 (2023-10 to 2024-07) | 0.001 | 0.01 | (0.01, 0.8) |
| 23 (2024-10) | 0.001 | 0.01 | (0.01, 0.2) |

- **Distributions.**
  - ridge: {0.001: 9, 0.1: 1, 1: 1, 10: 4, 100: 9}.
  - lasso: {0.01: 9, 0.05: 1, 0.2: 1, 0.5: 2, 1 (all-zero): 11}.
  - elastic_net: the same r as lasso in 24/24 folds; l1_ratio 0.8 in 19 folds and 0.2 in 5.
- **The scores are nearly flat.**
  - Across each fold's grid, inner scores span a median 1.2% of the inner MSE (max 3.7%).
  - The best and second-best points differ by a median 0.013% (ridge) and 0.05% (lasso).
  - Ridge picked a grid end in 18 of 24 folds, and lasso in 20 of 24.
  - Consecutive folds share 3 of 4 inner windows, so choices come in runs. With 231 inner dates (about 11 independent 20-day periods), selection behaves like "keep the signal if it worked in the last year".
- **Ridge cannot choose a zero model.** At c = 100 it shrinks a one-feature slope 101-fold, but keeps the ranking (roughly momentum's). Lasso's r = 1 is the all-zero model. That asymmetry explains why Ridge always ranks and Lasso sometimes does not.

**Coefficient stability and feature selection** (standardized coefficients over 24 folds; descriptive only).

| Model | momentum_20d | daily_return | volatility_20d |
|---|---|---|---|
| ols | median +0.0056, positive 24/24 | median −0.0012, negative 24/24 | median +0.0003, positive 16/24 |
| ridge | median +0.0006, positive 24/24 | negative 16/24 | positive 13/24 |
| lasso | selected 13/24, always positive | selected 10/24, always negative | selected 9/24, always positive |
| elastic_net | same as lasso | same as lasso | same as lasso |

- Expanding windows share most of their training rows, so "momentum positive in every fold" is close to one observation, not 24.
- Ridge's daily_return sign follows the penalty. Heavy shrinkage moves toward each feature's marginal effect (+), light shrinkage toward the partial effect (−). That is not instability.
- The positive volatility coefficient may be beta times a rising market, or NVDA's identity (highest volatility and highest mean excess return). The low-volatility literature expects the opposite sign.

**What can and cannot be concluded.**
- **Can:** the pipeline is leakage-safe and reproducible, and none of the candidates shows detectable skill at this power: d_t vs zero |t| ≤ 0.04, and OOS R² 95% intervals of about ±1%.
- **Cannot:** that there is no skill. An OOS R² of 1% or a rank IC of 0.05 would matter economically and would go undetected here (minimum detectable IC 0.16–0.25).
- **Not evidence:**
  - Lasso's apparent lead over Ridge;
  - p = 0.08;
  - the coefficient signs;
  - per_symbol_mean's "win", which measures hindsight selection of four names.
- **None of this is product evidence.**

## Warnings and limitations

- **Universe.** Four hindsight-selected semiconductor names (survivorship-biased): an engineering sandbox, not product-grade evidence.
- **Coverage.** Lasso and Elastic Net rank IC are conditional on 13 of 24 folds. Compare models only on the same dates (the dashboard now shows coverage).
- **Hit rate counts noise signs.**
  - On all-zero dates Lasso predicts about ±4e-19, and those 2772 rows are scored with a meaningless sign: hit rate 0.501, against 0.522 on the other rows.
  - pooled_mean's hit rate (0.486) is entirely such signs.
  - A zero tolerance would change T2's metric definition, so it needs owner approval.
- **Permutation p-values** have resolution 0.02 with 49 draws; Lasso's null is not centred at zero.
- **Multiple testing.**
  - Nominally there are 3 candidates plus OLS and 34 grid points. Effectively there are about 2 forecasts: Ridge ranks like OLS, and Elastic Net duplicates Lasso.
  - The development window has now been scored several times; scratch runs are not in the ledger.
- **Untested sizes.** The sizes of the Clark-West and MSE-difference tests are not simulated.
- **Benchmark.** The excess label is not beta-neutral.
- **Costs** are not measured (T5); daily_return enters with a flipping sign, a high-turnover component.

## Mandatory before large-universe validation

1. **Variable universe: complete cross-sections and delistings.**
   - The fail-closed rule is the right default: defining the benchmark over names whose labels complete would let future survival leak in.
   - Its cost grows with the universe. Each name without an exit price blanks every date whose 20-day label it would cover, and those dates cluster in stress periods (pre-takeover, pre-failure).
   - Illustration only, not measured: 2–5 delistings a year in ~100 names would lose roughly 17–42% of dates.
   - C_t is also "names with a row on t", not a point-in-time investable membership.
   - Required:
     - a point-in-time membership table independent of feature rows;
     - delisting returns from a security master (with a documented fallback, e.g. −30% for performance delistings when missing);
     - a reinvestment rule;
     - per-date reason codes.
   - Each changes label or availability semantics: owner approval.
2. **Row-weighted fitting versus date-weighted tuning.**
   - Fits minimize squared error over rows, so dates with more names and more dispersion dominate.
   - Selection and the Goyal-Welch view weigh dates equally, and the decision metric is rank IC.
   - Here they coincide: every scored date has 4 names. With a variable universe they diverge.
   - Pre-register one rule: sample weights 1/N_t (with weighted standardization and lambda_max), within-date standardized or ranked labels, or an IC-based inner criterion.
   - Formula change: owner approval.

Also recorded for the next modelling round (each is a new variant or a definition change, so owner decisions):
- within-date demeaned or ranked features;
- a pre-registered one-standard-error rule, with inner score standard errors stored;
- 4 vs 8 vs all inner windows;
- ≥ 199 permutation draws;
- Romano-Wolf or Hansen SPA across candidates;
- a cap-weighted benchmark in T5.

## Verification

**Phase A.**
- Tests: the excess label, nested plumbing, Clark-West and checks, contract 1.2 and 1.0/1.1 compatibility, and the dashboard on a T4 run. Integration uses statistics-free doubles.
- The 12 core contract tests were validated before hand-off against a reference implementation and three plausible mistakes.
- 18 mutation probes, all caught.

**Phases B and C.**
- **Core and pipeline tests** (`test_linear_core.py`, `test_linear_estimator.py`, `test_linear_models.py`):
  - the 12 contract tests pass unchanged;
  - the real core under adversarial perturbation (all three penalized methods);
  - per-fit inner lambda_max;
  - exact one-date-at-a-time prediction with NaN imputation;
  - planted signal and pure noise (Lasso keeps under 60% of OLS's total slope size on noise);
  - stored selection and coefficient rows equal an independent re-scoring and the fitted estimators;
  - the tie tolerance;
  - each convergence signal on its own, and the failing and undetermined convergence check;
  - other warnings still surface;
  - selection rerun on each null draw's own inputs;
  - a single outer refit call;
  - the strict inner-label boundary;
  - a stored inner label reaching the test start failing the leakage check;
  - date-equal Clark-West weighting;
  - the reader's column-version check;
  - dashboard coverage, number formats and the stability table.
- **Mutation probes.**
  - The test review ran 49 probes on `1338674`; 22 survived.
  - After the fixes, 22 targeted probes (the survivors plus the new code) were run, each in its own scratch copy of the working tree: all killed.
- **Reviews.** Independent data-science, quantitative and test-quality reviews found no blockers and no leakage. Their should-fix items are addressed above, or recorded as owner decisions.

## Dependency review: scikit-learn 1.9.1

- **Value.** Mature, well-tested solvers for Ridge (closed form), Lasso and Elastic Net (coordinate descent), with convergence reporting.
- **Alternatives.** A numpy closed form handles Ridge and OLS only. statsmodels has weaker Elastic Net support. glmnet bindings are not maintained for Python 3.12.
- **Cost.**
  - New packages: scikit-learn, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0 and cloudpickle 3.1.2. narwhals was already present.
  - About 60 MB installed.
  - Fits on 3 features take milliseconds. All linear fits (including OLS) run with BLAS limited to 1 thread, so sums are reproducible.
- **Security.** BSD-3 (scikit-learn, joblib, threadpoolctl, cloudpickle) and BSD (scipy) licences, installed from PyPI. No network access at run time. scikit-learn 1.9.1 was released 2026-09-10.
- **Deployment.** Pinned in `requirements.txt`. Transitive packages are unpinned, like numpy and pandas.

## Optional exercise (bounded, about 30 lines of pandas)

Using run `20261011T191112Z-50c1ca320e9c`:
1. Load the `metrics` table (`stock_agent.dashboard.loader.load_run` or `pd.read_parquet`) and take the per-fold `mean_rank_ic` rows (`fold_id` not null) for `lasso`, `ridge` and `momentum_20d`.
2. Compute:
   - Lasso's zero-filled fold-block mean, counting its undefined folds as 0;
   - Ridge's and momentum's mean over only the folds where Lasso is defined.
3. Expected: 0.038, 0.068 and 0.069.
4. In two sentences, explain why Lasso's stored 0.0705 against Ridge's 0.0199 is not evidence that Lasso ranks better, and which number you would put in a report.

## Learning Notes

1. **Problem.** Fit regularized linear forecasts of the cross-sectional excess return; choose their penalty without looking at the period being evaluated; then judge them fairly against nulls and controls.
2. **Concept.**
   - Ridge (L2) shrinks every coefficient.
   - Lasso (L1) can set coefficients exactly to zero.
   - Elastic Net mixes the two.
   - Nested validation: inner folds inside each outer training set choose the penalty, and the outer folds estimate performance.
   - A model that is sometimes constant has a rank IC that is *conditional on being active*: coverage is part of the result.
3. **Assumptions.**
   - A linear relation between standardized features and excess returns.
   - Stationarity within each training window.
   - Squared error as the selection loss.
   - Folds exchangeable for the fold-block t.
   - No symbol-identity effect: violated on four hindsight names.
4. **Formulation.**
   - Ridge (sklearn): min ||y − Zb||² + α||b||², with α = c·n. Dividing by n gives (1/n)||y − Zb||² + c||b||², so c is per observation.
   - Lasso: min (1/2n)||y − Zb||² + α||b||₁, with α = r·λ_max and λ_max = max_j |z_j'(y − ȳ)|/n.
   - Elastic Net: min (1/2n)||y − Zb||² + α(ρ||b||₁ + (1 − ρ)/2 ||b||²), with α = r·λ_max/ρ.
   - Selection criterion: (1/T) Σ_t (1/N_t) Σ_i (y_it − f_it)², each date weighs the same.
   - Excess-label MSE split: (1/N) Σ_i (y_it − f_it)² = (1/N) Σ_i (y_it − (f_it − f̄_t))² + f̄_t², because Σ_i y_it = 0.
   - Clark-West: E[2(y − b)(f − b)] > 0 for a nested pair.
5. **Worked examples (illustrative numbers, not from the run).**
   - **Ridge, one standardized feature.** b = b_OLS / (1 + c). With b_OLS = 0.004: c = 0.001 gives 0.003996, c = 1 gives 0.002, and c = 100 gives 0.0000396. The 101-fold shrink keeps the sign, so the ranking survives.
   - **Lasso, one standardized feature.** λ_max = |z'y/n| = |b_OLS|, and b = (1 − r)·b_OLS. With b_OLS = 0.5: r = 0.1 gives 0.45, and r = 1 gives 0, the all-zero model.
   - **Elastic Net's units.** For one feature, b = (1 − r)·λ₁ / (1 + r·λ₁(1 − ρ)/ρ).
     - With λ₁ = 0.002 (returns as decimals), r = 0.5 and ρ = 0.2, the denominator is 1.004: 0.4% extra shrinkage, so the fit is practically Lasso.
     - With the same labels in percent, λ₁ = 0.2 and the denominator is 1.4: 29% extra shrinkage. Same data, different model: not scale-free.
   - **Date weighting.** Date A has 4 names with squared errors 4e-4 each; date B has 2 names with 1e-4 each.
     - Date-weighted: (4e-4 + 1e-4)/2 = 2.5e-4.
     - Row-weighted: (4·4e-4 + 2·1e-4)/6 = 3.0e-4.
     - With a variable universe, the fit (rows) and the selection (dates) optimize different targets.
   - **MSE split.** Labels (+0.03, −0.01, −0.01, −0.01) and forecasts (0.012, 0.002, 0.002, 0.002) give MSE 1.89e-4: 1.6875e-4 within the date, plus 0.0045² = 2.025e-5 from the shared part.
   - **Coverage.** 13 active folds at 0.0705 and 11 constant folds counted as 0 give 13/24 × 0.0705 = 0.038.
6. **Leakage risks** (each is prevented and tested):
   - choosing α on the outer test period;
   - inner labels overlapping the outer test;
   - scaling or λ_max computed from rows outside the fit;
   - an excess benchmark using a later-maturing label;
   - tuning done once and reused in the null draws.

   Future risk: dropping a date because a name later delists is look-ahead sample selection.
7. **Ways results could mislead.**
   - Statistics computed on different date subsets.
   - Null distributions not centred at zero.
   - Near-duplicate models inflating the variant count.
   - Signs of 1e-19 noise counted as hits.
   - Overlapping training windows inflating "stability".
   - Selection among near-ties.
   - Repeated scoring of one development window.
8. **Practitioner validation.**
   - Report coverage-weighted IC and compare models on the same dates.
   - Cross-sectionally demeaned or rank-normalized features (Gu, Kelly and Xiu 2020).
   - The one-standard-error rule.
   - Delisting-adjusted returns (Shumway 1997).
   - Romano-Wolf or Hansen SPA across models.
   - Costs and turnover (T5).
   - A holdout used once (T6).
9. **What to learn next.**
   - Fama-MacBeth versus pooled regression.
   - How glmnet and scikit-learn parameterize Elastic Net.
   - The one-standard-error rule and stability selection.
   - Ridge's effective degrees of freedom.
   - Delisting bias.
   - White's Reality Check.
