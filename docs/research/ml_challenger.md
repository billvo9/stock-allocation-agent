# A simple nonlinear challenger: HistGradientBoosting

Status (2026-10-14): implemented on `feature/ml-gbm-challenger` (stacked on the experiment-runner PR); merge pending.

Code:
- `forecasting/nonlinear.py`;
- runner wiring in `forecasting/experiment.py`;
- `model_diagnostics/nested.py` and `runner.py`, which now store selection rows for any nested-selected model.

Configs: `configs/ml/equity_xs_challenger_{nonlinear,linear,null}.yaml`. Tests: `tests/test_challenger.py`.

**Question.** Does a nonlinear model add anything over the T4 linear baselines, measured honestly?
- same folds, same nested purged tuning, same nulls and checks;
- the increment over every linear baseline on identical rows, with uncertainty;
- the variants it adds counted for multiple testing.

## The challenger

- **Library.** scikit-learn's `HistGradientBoostingRegressor`. It ships with the pinned scikit-learn 1.9.1, so no dependency is added.
- **Preprocessing.** Training-row median imputation, the features' contract policy `impute_training_median`. No scaling: trees do not need it.
- **Settings.** `loss="squared_error"`, `early_stopping=False` and a fixed `random_state`.
  - Early stopping is off because it carves a *random* validation split out of the training rows. That would be a non-temporal split inside a temporal protocol (tested).
- **Fitting.** One fit per fold, single-threaded inside the nested harness, so results are reproducible.

### Pre-registered grid: a one-dimensional complexity path

| penalty | max_leaf_nodes | max_iter | learning_rate | min_samples_leaf | l2 |
|---|---|---|---|---|---|
| 4 (simplest) | 4 | 50 | 0.05 | 200 | 1.0 |
| 3 | 8 | 100 | 0.05 | 100 | 1.0 |
| 2 | 16 | 150 | 0.05 | 50 | 1.0 |
| 1 (most complex) | 31 | 200 | 0.05 | 20 | 0.0 |

- **Tie rule.** `penalty` is the path index, and T4's tie rule picks the larger penalty, so ties go to the simpler model.
- **Contract.** The `selection` table keeps its contract-1.2 columns: `penalty` holds the index, and `l1_ratio` is NaN.
- **Selection.** The same inner folds and the same criterion as the linear models (date-normalized inner MSE).
- **Identity.** The grid is part of the variant id, so changing it is a new variant (tested).

## The comparison

The challenger is one more `candidate` in the same `run_diagnostics` call. It therefore:
- **is scored on exactly the linear models' rows** (tested);
- **is compared with T2's registered nulls:** zero, the per-symbol mean, random noise, the canary;
- **gets T4's leakage checks**, including `nested_selection_inside_training`;
- **counts for multiple testing:** `candidates_in_run` and `grid_points_in_run` grow by one and by four, and it joins the Holm family.

**Increment over every linear baseline** (ridge, lasso, elastic net, OLS), on identical rows, one value per date. Fold-block t decides; Newey-West and the bootstrap are reported beside it.
- `rank_ic_difference` (zero-filled): does it order names better?
- `mse_improvement_vs_reference`: does it reduce squared error?
- `oos_r2_vs_reference` = 1 − SSE_challenger / SSE_linear, with a bootstrap interval.

## Results (measured, synthetic)

Shipped configs: 30 names, 1500 sessions, h = 20, 14 outer folds. The signal strength s = 4e-4 per day is the same as the linear planted config. Fold-block t, df 13.

**Runs.**
- nonlinear `20261014T190944Z-610c010d6e25`, linear `20261014T191352Z-a43c61b23363`, null `20261014T191744Z-7b652254ad31`;
- all from commit `fd63d77`, with every leakage check passing (18 of 18 per run).

**Provenance.**
- `git_dirty` is true for the reason given in `ml_pipeline.md`, the untracked `.venv` symlink.
- The null run's diff hash (`60b96d50…`) also lists this untracked document. It equals SHA-256("" + "\0" + ".venv\ndocs/research/ml_challenger.md\n"), so no source file differed.

**Multiple testing.** Each run had 4 candidates and 38 grid points (6 + 7 + 21 linear, plus 4 for the challenger), and the challenger joins the Holm family.

| Panel | Model | Coverage-weighted rank IC (t) | OOS R² vs zero | Increment over ridge: rank IC (t) | OOS R² vs ridge | MSE improvement vs ridge, t | Holm p |
|---|---|---|---|---|---|---|---|
| nonlinear | ridge | 0.017 (0.79) | −0.04% | — | — | — | 0.67 |
| nonlinear | hist_gradient_boosting | 0.033 (1.08) | +0.19% | +0.016 (0.56) | +0.23% | 0.23 | 0.60 |
| linear | ridge | 0.082 (2.97) | +0.28% | — | — | — | 0.016 |
| linear | hist_gradient_boosting | 0.065 (2.28) | +0.59% | −0.016 (−0.74) | +0.31% | 0.31 | 0.038 |
| null | ridge | 0.021 (1.16) | −0.06% | — | — | — | 0.33 |
| null | hist_gradient_boosting | −0.024 (−0.83) | −1.33% | −0.045 (−1.60) | −1.27% | −2.27 | 0.79 |

**Against the other linear baselines** (rank-IC difference t; MSE improvement t):
- **nonlinear panel:**
  - lasso 0.93 and 0.25; elastic net 0.93 and 0.25;
  - OLS 0.51 and **2.54**. The only "win" is over unpenalized OLS, which pays an estimation-noise cost (OOS R² vs zero −0.94%).
- **linear panel:** every |t| ≤ 0.74.
- **null panel:** the challenger is significantly worse than ridge, lasso and elastic net on squared error (t −2.27 to −2.40).

**Reading.**
- **Nonlinear panel.** At the shipped strength the planted nonlinear effect is too weak for this design. The challenger's point estimates are positive but nowhere near credible.
- **Linear panel.** It ranks a little worse than ridge (−0.016, t −0.74) and gains nothing credible on squared error.
- **Null panel.** It loses squared error to the regularized linear models: flexible models fit noise. This is the cost a challenger must overcome.

## Is the increment credible?

Credible here means a positive paired increment over the **best** linear baseline, with a fold-block t above 2, in a family whose size is known (Holm). It also needs a mechanism that explains it.

**Fast tests** (30 names, h = 10, s = 1.5e-3, 8 folds):
- **planted nonlinear signal:** the challenger beats ridge by rank IC 0.095 (t 3.21) and MSE (t 6.47). The fit selected the simplest grid point (4 leaves) in 8 of 8 folds.
- **linear signal:** no credible increment (rank IC t 0.99, MSE t 0.53).

Both outcomes are asserted in `tests/test_challenger.py`.

**At the shipped strength.**
- Nothing is credible: the planted nonlinear effect sits below the design's detection power.
- A "win" against OLS alone is not an increment over linear models. OLS is the weakest baseline because it pays an estimation-noise cost, so beating it only shows the challenger is regularized.

**What would make an increment credible on real data.**
- It holds against ridge, the best-tuned linear baseline, not just OLS.
- It holds on the zero-filled rank IC *and* on squared error.
- It survives the Holm family, and the permutation null with at least 199 draws, with the challenger's tuning rerun in every draw. That costs about 238 boosted fits per draw at this size, so it was not run here.
- It persists across disjoint periods.
- It has an economic story (which feature interactions?). Nonlinear models find spurious interactions easily when there are few names.

## Limitations

- **Synthetic data only.** The planted nonlinear drift is exactly orthogonal to the characteristic within each date (the residual of |z| on (1, z)), so no linear function of that characteristic can use it.
  - Linear models can still capture part of it through price features. Past returns carry the realized planted drift, so momentum is a legitimate channel (seen with 16 names).
- **No permutation null for the challenger.** It is too expensive at this size; the null config is the null evidence instead.
- **Only the simplest point chosen.** On these panels the challenger almost always chose the simplest grid point. The grid's upper end is untested by the data.

## Learning Notes

1. **Problem.** Decide whether a flexible model adds forecasting value beyond regularized linear models, without fooling ourselves.
2. **Concept.** A paired comparison on identical rows removes the common noise in both models' errors. The per-date loss difference d_t = L(linear) − L(challenger) is the object of inference (the Diebold-Mariano idea). Fold blocks handle the overlap of 20-day labels.
3. **Assumptions.**
   - Fold blocks are roughly exchangeable.
   - The inner-MSE criterion is a fair tuning loss for both families.
   - The grid is pre-registered, so selection noise is included in the outer evaluation.
4. **Formulas.**
   - Increment = mean_t d_t; t = mean(block means) / (sd(block means)/√B).
   - OOS R² vs the linear model = 1 − Σ e²_challenger / Σ e²_linear.
5. **Example.**
   - Two models, 14 folds. The challenger's block-mean MSE advantage averages 2e-6 with sd 1e-5.
   - t = 2e-6 / (1e-5/√14) = 0.75: not credible, even if the point estimate is positive.
6. **Leakage risks guarded.**
   - Early stopping on a random split (disabled).
   - Tuning on the outer test window (nested).
   - Imputing with test-period medians (training medians only; tested).
   - Per-date prediction calls (harness), so no batch can see later rows.
7. **Ways results could mislead.**
   - Beating OLS but not ridge.
   - Conditional rank IC on lasso's active dates.
   - Many grid points with a near-flat inner score.
   - A significant increment at an unrealistically strong planted effect.
8. **Practitioner validation.**
   - Gu, Kelly and Xiu (2020) compare trees and nets with linear models on the same splits, and report Diebold-Mariano tests between models.
   - Hansen's SPA test when many challengers compete.
9. **Next to learn.** Diebold-Mariano and Giacomini-White tests (conditional predictive ability), and why model-selection noise inflates apparent gains.
